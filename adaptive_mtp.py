# SPDX-License-Identifier: Apache-2.0
"""
Adaptive MTP with Rollback for Qwen 3.5-27B on vLLM 0.19 / GH200.

Problem
-------
Qwen 3.5's MTP head was trained for 1-step prediction but vLLM chains it 7
times (num_speculative_tokens=7).  Acceptance rates degrade monotonically:

    position:  0    1    2    3    4    5    6
    rate:     ~80% ~70% ~60% ~50% ~40% ~30% ~20%

Running all 7 steps unconditionally wastes forward passes on positions 5-6
(draft compute >> expected token gain).  This module implements two
complementary mechanisms:

Mechanism 1 — EMA-gated chain length (per-step "should I stop here?")
    Track acceptance rate per position with exponential moving average.
    Before each draft step K, check: has EMA[K] dropped below THRESHOLD?
    If so, stop drafting at K-1.  Use a "suppression window" to hold the
    shorter chain for M steps before re-trying.

Mechanism 2 — Immediate rollback (within a single draft sequence)
    After the draft sequence is produced, scan from the tail: if the last
    ROLLBACK_WINDOW consecutive proposed positions were ALL rejected (from
    the *previous* step's outcome), shorten max_chain by 1 immediately for
    the *next* step, without waiting for EMA convergence.

Both mechanisms share a single ``AdaptiveMTPController`` that lives for the
lifetime of the server.  It is called before and after each propose() round.

Integration
-----------
This file ships as a drop-in wrapper around EagleProposer.  Zero changes to
vLLM internals are required — the wrapper intercepts propose() and adapts
num_steps based on the controller state.

Quick-start
-----------
    from adaptive_mtp import AdaptiveMTPProposer

    # In gpu_model_runner, replace:
    #   self.drafter = EagleProposer(vllm_config, device, self)
    # with:
    #   self.drafter = AdaptiveMTPProposer(vllm_config, device, self)

    # Acceptance feedback is passed in after each rejection-sampling step:
    #   self.drafter.record_acceptance(accepted_per_pos)
    #   # accepted_per_pos: list[int], len == num_speculative_tokens,
    #   # entry k = number of sequences that reached position k accepted.

For standalone testing / benchmarking, use AdaptiveMTPController directly.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters (tunable via env vars to match llama.cpp convention)
# ---------------------------------------------------------------------------
import os

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))

# EMA decay — higher = slower to forget (0.10 ≈ 10-step horizon)
EMA_ALPHA            = _env_float("AMTP_EMA_ALPHA",           0.10)

# Stop extending chain at position K if EMA[K] < THRESHOLD.
#
# The EMA tracks the *unconditional* rate: P(pos k accepted AND all prior
# positions accepted) / num_drafts.  For Qwen 3.5 the geometric chain
# gives P(reach k AND accept k) = prod(rates[0..k]).  At position 4 that
# is ~0.80*0.70*0.60*0.50*0.40 ≈ 0.067, meaning each draft step at pos 4
# produces only 0.067 accepted tokens per request on average.
#
# Break-even: a draft step at position k is worthwhile only if the expected
# token gain (EMA[k] * 1 token) exceeds the draft cost.  On GH200 with
# a 27B model and 2-layer MTP head, the MTP head costs ~3% of a verify
# forward.  Break-even rate ≈ 0.03.  We use 0.05 with margin.
STOP_THRESHOLD       = _env_float("AMTP_STOP_THRESHOLD",      0.05)

# After cutting chain to K-1, hold that length for SUPPRESS_WINDOW steps
# before re-allowing K.  Prevents flapping.
SUPPRESS_WINDOW      = _env_int(  "AMTP_SUPPRESS_WINDOW",     32)

# Immediate rollback: if the last ROLLBACK_WINDOW steps all had 0 accepted
# at some position, shorten chain immediately (without waiting for EMA).
ROLLBACK_WINDOW      = _env_int(  "AMTP_ROLLBACK_WINDOW",     3)

# Re-probe: every REPROBE_INTERVAL suppression steps, allow a single longer
# chain to re-sample whether tail acceptance has recovered.
REPROBE_INTERVAL     = _env_int(  "AMTP_REPROBE_INTERVAL",    64)

# Minimum chain length.  Never go below 1.
MIN_CHAIN            = _env_int(  "AMTP_MIN_CHAIN",           1)

# Log interval (in propose() calls)
LOG_INTERVAL         = _env_int(  "AMTP_LOG_INTERVAL",        200)


# ---------------------------------------------------------------------------
# Per-position EMA tracker
# ---------------------------------------------------------------------------

class PositionEMA:
    """Exponential moving average of acceptance rate per draft position.

    We track the *unconditional* rate per position:

        EMA[k] ≈ E[accepted_at_pos_k] / num_drafts

    where accepted_at_pos_k counts only sequences that reached position k
    AND had the draft token accepted.  Because speculative decoding requires
    accepting all prefix positions, this is a joint probability:

        P(pos k accepted) = prod(true_rate[j] for j in 0..k)

    This means EMA[k] falls exponentially with k, which is exactly what we
    want to threshold against draft cost.

    For positions NOT drafted this step (chain was shortened), we do NOT
    update the EMA — the last measured value persists.

    Thread-safety: single-threaded (GPU model runner hot path).
    """

    def __init__(self, max_positions: int, alpha: float = EMA_ALPHA):
        self.max_positions = max_positions
        self.alpha = alpha
        # Initialise optimistically using the Qwen 3.5 profile heuristic.
        # These priors are overwritten quickly once real data arrives.
        # E.g. pos 0: P≈0.80, pos 1: 0.80*0.70≈0.56, pos 2: ≈0.34, etc.
        self._ema: list[float] = [
            0.80 * (0.70 ** k) for k in range(max_positions)
        ]
        # How many updates have been applied per position (for cold-start)
        self._n_updates: list[int] = [0] * max_positions

    def update(
        self,
        accepted_per_pos: list[int],
        num_drafts: int,
        num_drafted_positions: int,
    ) -> None:
        """Update EMA for positions 0..num_drafted_positions-1.

        Args:
            accepted_per_pos:      len >= max_positions.  Entry k = number of
                                   sequences that had position k accepted
                                   (unconditional: also requires prefix ok).
            num_drafts:            batch size (denominator).
            num_drafted_positions: how many positions were actually drafted.
                                   Positions >= this are NOT updated.
        """
        if num_drafts <= 0:
            return
        for k in range(min(num_drafted_positions, self.max_positions)):
            rate = accepted_per_pos[k] / num_drafts
            if self._n_updates[k] == 0:
                self._ema[k] = rate          # cold start: exact first sample
            else:
                self._ema[k] = (1 - self.alpha) * self._ema[k] + self.alpha * rate
            self._n_updates[k] += 1

    def get(self, pos: int) -> float:
        return self._ema[pos]

    def all(self) -> list[float]:
        return list(self._ema)


# ---------------------------------------------------------------------------
# Rollback tracker
# ---------------------------------------------------------------------------

class RollbackTracker:
    """Detects when the tail positions are consistently useless.

    Keeps a ring buffer of per-position accepted counts over the last
    ROLLBACK_WINDOW steps.  If ALL steps in the window had 0 accepted at
    position K, we trigger an immediate rollback (shorten chain to K).

    The caller decides what to do with the rollback signal.
    """

    def __init__(self, max_positions: int, window: int = ROLLBACK_WINDOW):
        self.max_positions = max_positions
        self.window = window
        # ring buffer: deque of list[int] (accepted_per_pos snapshots)
        self._history: deque[list[int]] = deque(maxlen=window)

    def push(self, accepted_per_pos: list[int], num_drafted: int) -> Optional[int]:
        """Push a new observation.  Returns the rollback position (new max
        chain length) if a rollback is triggered, else None.

        A rollback fires when ALL window observations had zero accepted at
        position K and K was actually drafted.

        We check from the tail (highest position) downward so we catch the
        most-aggressive rollback first.
        """
        self._history.append(list(accepted_per_pos))

        if len(self._history) < self.window:
            # Not enough history yet
            return None

        # For each position from the tail down, check if all window entries = 0
        for k in range(min(num_drafted, self.max_positions) - 1, -1, -1):
            all_zero = all(snap[k] == 0 for snap in self._history)
            if all_zero:
                # Roll back to k (stop drafting AT k, so new chain = k)
                return k

        return None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveMTPStats:
    """Accumulated stats for logging."""
    n_propose_calls: int = 0
    n_tokens_drafted: int = 0
    n_tokens_accepted: int = 0
    n_rollbacks: int = 0
    n_reprobe_steps: int = 0
    chain_lengths: list[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)

    def tok_efficiency(self) -> float:
        if self.n_tokens_drafted == 0:
            return float("nan")
        return self.n_tokens_accepted / self.n_tokens_drafted


class AdaptiveMTPController:
    """Central controller — call before/after each propose() round.

    Lifecycle
    ---------
        controller = AdaptiveMTPController(max_positions=7)

        # Before propose():
        chain_len = controller.get_chain_length()

        # After rejection sampling returns per-position acceptance info:
        controller.record_acceptance(
            accepted_per_pos=...,   # list[int] len=max_positions
            num_drafts=batch_size,
            actual_chain=chain_len,
        )
    """

    def __init__(
        self,
        max_positions: int,
        ema_alpha: float = EMA_ALPHA,
        stop_threshold: float = STOP_THRESHOLD,
        suppress_window: int = SUPPRESS_WINDOW,
        rollback_window: int = ROLLBACK_WINDOW,
        reprobe_interval: int = REPROBE_INTERVAL,
        min_chain: int = MIN_CHAIN,
        log_interval: int = LOG_INTERVAL,
    ):
        self.max_positions = max_positions
        self.stop_threshold = stop_threshold
        self.suppress_window = suppress_window
        self.reprobe_interval = reprobe_interval
        self.min_chain = min_chain
        self.log_interval = log_interval

        self._ema = PositionEMA(max_positions, alpha=ema_alpha)
        self._rollback = RollbackTracker(max_positions, window=rollback_window)

        # Current effective chain length (can be shortened by EMA or rollback)
        self._current_chain: int = max_positions

        # Suppression: per-position suppression counters.
        # _suppress[k] > 0 means position k is suppressed for that many more steps.
        self._suppress: list[int] = [0] * max_positions

        # Reprobe counter: used to occasionally allow a longer chain
        self._reprobe_counter: int = 0
        self._reprobe_active: bool = False

        self._stats = AdaptiveMTPStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_chain_length(self) -> int:
        """Return the number of draft steps to run this round.

        Call this BEFORE propose().  The returned value may be less than
        max_positions based on accumulated acceptance statistics.
        """
        self._stats.n_propose_calls += 1

        # Re-probe logic: every reprobe_interval suppressed steps, allow
        # one full-length chain to re-sample tail acceptance.
        if self._is_any_suppressed():
            self._reprobe_counter += 1
            if self._reprobe_counter % self.reprobe_interval == 0:
                self._reprobe_active = True
                self._stats.n_reprobe_steps += 1
                logger.debug(
                    "AdaptiveMTP: reprobe step (counter=%d)", self._reprobe_counter
                )
                return self.max_positions

        self._reprobe_active = False

        # Walk positions from 0 upward.  The chain runs positions 0..K-1
        # where K is the first position that is suppressed or below threshold.
        chain = self.min_chain
        for k in range(self.max_positions):
            if self._suppress[k] > 0:
                # Suppressed — do not extend past k
                break
            # Check EMA threshold
            if self._ema.get(k) < self.stop_threshold:
                # EMA says this position is not worth drafting
                # Suppress it for suppress_window steps
                self._suppress[k] = self.suppress_window
                logger.debug(
                    "AdaptiveMTP: suppressing pos %d (EMA=%.3f < threshold=%.3f) "
                    "for %d steps",
                    k, self._ema.get(k), self.stop_threshold, self.suppress_window,
                )
                break
            chain = k + 1  # Position k is allowed — extend chain to at least k+1

        chain = max(self.min_chain, min(chain, self.max_positions))
        self._current_chain = chain
        self._stats.chain_lengths.append(chain)
        return chain

    def record_acceptance(
        self,
        accepted_per_pos: list[int],
        num_drafts: int,
        actual_chain: int,
    ) -> None:
        """Feed back acceptance results after rejection sampling.

        Args:
            accepted_per_pos: list of length max_positions.  Entry k = how
                              many requests had position k accepted.
                              Positions beyond actual_chain should be 0.
            num_drafts:       number of requests that were speculated this step.
            actual_chain:     how many positions were actually drafted.
        """
        if num_drafts <= 0:
            return

        # Ensure correct length
        if len(accepted_per_pos) < self.max_positions:
            accepted_per_pos = list(accepted_per_pos) + [0] * (
                self.max_positions - len(accepted_per_pos)
            )

        # Update EMA
        self._ema.update(accepted_per_pos, num_drafts, actual_chain)

        # Tick down suppression counters (only for positions that were drafted)
        for k in range(actual_chain):
            if self._suppress[k] > 0:
                self._suppress[k] -= 1

        # Rollback check — immediate chain shortening
        rb = self._rollback.push(accepted_per_pos, actual_chain)
        if rb is not None and not self._reprobe_active:
            new_chain = max(self.min_chain, rb)
            if new_chain < self._current_chain:
                logger.debug(
                    "AdaptiveMTP: rollback triggered — shortening chain from %d to %d",
                    self._current_chain, new_chain,
                )
                self._stats.n_rollbacks += 1
                # Force suppression of positions new_chain..actual_chain-1
                for k in range(new_chain, actual_chain):
                    self._suppress[k] = max(self._suppress[k], self.suppress_window)
                self._current_chain = new_chain

        # Accumulate stats
        self._stats.n_tokens_drafted += actual_chain * num_drafts
        self._stats.n_tokens_accepted += sum(accepted_per_pos[:actual_chain])

        # Logging
        if self._stats.n_propose_calls % self.log_interval == 0:
            self._log_stats()

    def get_ema_rates(self) -> list[float]:
        """Return current EMA acceptance rates for all positions."""
        return self._ema.all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_any_suppressed(self) -> bool:
        return any(s > 0 for s in self._suppress)

    def _log_stats(self) -> None:
        ema = self._ema.all()
        ema_str = " ".join(f"{v:.3f}" for v in ema)
        suppress_str = " ".join(str(s) for s in self._suppress)

        recent_chains = self._stats.chain_lengths[-self.log_interval:]
        mean_chain = sum(recent_chains) / max(1, len(recent_chains))

        elapsed = time.monotonic() - self._stats.start_time
        tok_eff = self._stats.tok_efficiency()

        logger.info(
            "AdaptiveMTP [step=%d]: "
            "mean_chain=%.2f/%d, "
            "tok_efficiency=%.3f, "
            "rollbacks=%d, "
            "reprobe=%d, "
            "EMA=[%s], "
            "suppress=[%s], "
            "elapsed=%.1fs",
            self._stats.n_propose_calls,
            mean_chain,
            self.max_positions,
            tok_eff,
            self._stats.n_rollbacks,
            self._stats.n_reprobe_steps,
            ema_str,
            suppress_str,
            elapsed,
        )


# ---------------------------------------------------------------------------
# vLLM proposer wrapper
# ---------------------------------------------------------------------------

try:
    from vllm.v1.spec_decode.eagle import EagleProposer, SpecDecodeBaseProposer
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.sample.metadata import SamplingMetadata
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    # Provide stubs so the module still imports cleanly for offline use
    class EagleProposer:  # type: ignore[no-redef]
        pass
    class SpecDecodeBaseProposer:  # type: ignore[no-redef]
        pass


if _VLLM_AVAILABLE:

    class AdaptiveMTPProposer(EagleProposer):
        """EagleProposer subclass that adapts draft chain length per step.

        Drop-in replacement for EagleProposer.  All standard EAGLE /
        MTP logic is inherited unchanged.  Only propose() is overridden to:

          1. Ask the AdaptiveMTPController how many steps to run.
          2. Temporarily patch self.num_speculative_tokens to that value.
          3. Restore the original value after propose() returns.
          4. Expect record_acceptance() to be called after rejection sampling.

        Integration in gpu_model_runner.py
        -----------------------------------
        Replace:
            self.drafter = EagleProposer(vllm_config, device, self)
        with:
            from adaptive_mtp import AdaptiveMTPProposer
            self.drafter = AdaptiveMTPProposer(vllm_config, device, self)

        Then after the rejection sampler runs, call:
            if isinstance(self.drafter, AdaptiveMTPProposer):
                self.drafter.record_acceptance(
                    accepted_per_pos=spec_decoding_stats.num_accepted_tokens_per_pos,
                    num_drafts=spec_decoding_stats.num_drafts,
                )

        The accepted_per_pos list comes directly from SpecDecodingStats which
        vLLM already computes — no additional instrumentation needed.
        """

        def __init__(self, vllm_config, device: torch.device, runner=None):
            super().__init__(vllm_config, device, runner)

            self._controller = AdaptiveMTPController(
                max_positions=self.num_speculative_tokens,
            )
            self._original_num_spec = self.num_speculative_tokens
            # Track what chain we used in the last propose() so
            # record_acceptance() knows the actual_chain without an extra arg.
            self._last_chain: int = self.num_speculative_tokens

            logger.info(
                "AdaptiveMTPProposer: max_positions=%d, "
                "EMA_ALPHA=%.3f, STOP_THRESHOLD=%.2f, "
                "SUPPRESS_WINDOW=%d, ROLLBACK_WINDOW=%d",
                self.num_speculative_tokens,
                EMA_ALPHA, STOP_THRESHOLD, SUPPRESS_WINDOW, ROLLBACK_WINDOW,
            )

        # ------------------------------------------------------------------
        # Core override
        # ------------------------------------------------------------------

        def propose(
            self,
            target_token_ids: torch.Tensor,
            target_positions: torch.Tensor,
            target_hidden_states: torch.Tensor,
            next_token_ids: torch.Tensor,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            mm_embed_inputs=None,
            num_rejected_tokens_gpu=None,
            slot_mappings=None,
        ) -> torch.Tensor:
            """Propose draft tokens with adaptive chain length.

            The output tensor always has shape [batch_size, original_num_spec]
            to keep the downstream rejection sampler happy.  When fewer steps
            are run, the trailing columns are filled with a sentinel (0 = PAD).
            vLLM's rejection sampler only looks at columns 0..num_draft_tokens[i]-1
            per request, so the PAD values are never consumed.

            Important: num_draft_tokens in SpecDecodeMetadata is set BEFORE
            propose() is called (by the scheduler).  Because we can only shorten
            the chain, not lengthen it, we need to truncate the draft output to
            match.  The correct fix is to also update SpecDecodeMetadata, which
            requires hooking into the model runner.  See note below.
            """
            # 1. Ask controller how many steps to run this round
            chain = self._controller.get_chain_length()
            self._last_chain = chain

            # 2. Temporarily limit num_speculative_tokens
            #    EagleProposer uses self.num_speculative_tokens in its propose()
            #    loop (line: for token_index in range(self.num_speculative_tokens - 1))
            self.num_speculative_tokens = chain

            try:
                draft_ids = super().propose(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    next_token_ids=next_token_ids,
                    token_indices_to_sample=token_indices_to_sample,
                    common_attn_metadata=common_attn_metadata,
                    sampling_metadata=sampling_metadata,
                    mm_embed_inputs=mm_embed_inputs,
                    num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                    slot_mappings=slot_mappings,
                )
            finally:
                # 3. Always restore
                self.num_speculative_tokens = self._original_num_spec

            # draft_ids shape: [batch_size, chain]
            # Pad to [batch_size, original_num_spec] with 0 if chain < max
            if chain < self._original_num_spec:
                batch_size = draft_ids.shape[0]
                pad_cols = self._original_num_spec - chain
                pad = torch.zeros(
                    batch_size, pad_cols,
                    dtype=draft_ids.dtype, device=draft_ids.device,
                )
                draft_ids = torch.cat([draft_ids, pad], dim=1)

            return draft_ids

        # ------------------------------------------------------------------
        # Feedback hook — call after rejection sampling
        # ------------------------------------------------------------------

        def record_acceptance(
            self,
            accepted_per_pos: list[int],
            num_drafts: int,
        ) -> None:
            """Feed per-position acceptance counts back to the controller.

            Call this immediately after vLLM's rejection sampler has run.
            The ``accepted_per_pos`` is exactly ``SpecDecodingStats.num_accepted_tokens_per_pos``.

            Args:
                accepted_per_pos: list[int] of length num_speculative_tokens.
                                  accepted_per_pos[k] = number of requests
                                  that had draft token k accepted.
                num_drafts:       number of requests speculated this step.
            """
            self._controller.record_acceptance(
                accepted_per_pos=accepted_per_pos,
                num_drafts=num_drafts,
                actual_chain=self._last_chain,
            )

        def get_ema_rates(self) -> list[float]:
            """Expose current EMA acceptance rates (useful for debugging)."""
            return self._controller.get_ema_rates()


# ---------------------------------------------------------------------------
# Standalone simulation / benchmark harness
# ---------------------------------------------------------------------------

def simulate(
    n_steps: int = 2000,
    max_positions: int = 7,
    true_rates: Optional[list[float]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Simulate the adaptive controller against a synthetic acceptance model.

    ``true_rates[k]`` is the probability that a draft token at position k is
    accepted (Bernoulli).  The default matches the Qwen 3.5 profile from the
    problem statement.

    Returns a summary dict with throughput gain estimates.
    """
    rng = np.random.default_rng(seed)

    if true_rates is None:
        # Qwen 3.5 empirical profile from problem statement
        true_rates = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    assert len(true_rates) == max_positions

    ctrl = AdaptiveMTPController(
        max_positions=max_positions,
        log_interval=n_steps + 1,  # suppress mid-sim logs
    )

    # Metrics
    total_drafted = 0
    total_accepted = 0
    total_saved_steps = 0    # steps NOT run due to adaptive shortening
    chain_hist = []

    batch_size = 8  # simulate a small decode batch

    for step in range(n_steps):
        chain = ctrl.get_chain_length()
        chain_hist.append(chain)

        # Simulate batch acceptance at each position.
        # accepted_per_pos[k] = number of requests that accepted pos k.
        # A request accepts pos k iff it accepted ALL of 0..k-1 AND Bernoulli(rate[k]).
        # This is the unconditional joint probability encoding used by vLLM's
        # SpecDecodingStats.num_accepted_tokens_per_pos.
        accepted_per_pos = [0] * max_positions
        for _ in range(batch_size):
            for k in range(chain):
                if rng.random() < true_rates[k]:
                    accepted_per_pos[k] += 1
                else:
                    # Rejected at k — positions k+1..chain-1 not reached
                    break

        ctrl.record_acceptance(
            accepted_per_pos=accepted_per_pos,
            num_drafts=batch_size,
            actual_chain=chain,
        )

        total_drafted  += chain * batch_size
        # Each accepted position yields one output token
        total_accepted += sum(accepted_per_pos[:chain])
        total_saved_steps += (max_positions - chain) * batch_size

    # Baseline: always run max_positions
    baseline_drafted = max_positions * batch_size * n_steps
    # Expected accepted under baseline (geometric chain acceptance)
    baseline_accepted = 0.0
    for k in range(max_positions):
        p_reach_k = 1.0
        for j in range(k + 1):
            p_reach_k *= true_rates[j]
        baseline_accepted += p_reach_k * batch_size * n_steps

    summary = {
        "n_steps": n_steps,
        "max_positions": max_positions,
        "true_rates": true_rates,
        "adaptive_drafted": total_drafted,
        "adaptive_accepted": total_accepted,
        "adaptive_efficiency": total_accepted / max(1, total_drafted),
        "baseline_drafted": baseline_drafted,
        "baseline_accepted": baseline_accepted,
        "baseline_efficiency": baseline_accepted / max(1, baseline_drafted),
        "draft_steps_saved_pct": 100.0 * total_saved_steps / max(1, n_steps * max_positions * batch_size),
        "mean_chain": np.mean(chain_hist),
        "chain_hist": chain_hist,
        "final_ema": ctrl.get_ema_rates(),
        "rollbacks": ctrl._stats.n_rollbacks,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Adaptive MTP Simulation — {n_steps} steps, batch={batch_size}")
        print(f"{'='*60}")
        print(f"True acceptance rates: {[f'{r:.2f}' for r in true_rates]}")
        print(f"\nBaseline (chain={max_positions} always):")
        print(f"  Drafted tokens:     {baseline_drafted:,}")
        print(f"  Accepted tokens:    {baseline_accepted:,.1f}")
        print(f"  Efficiency:         {summary['baseline_efficiency']:.3f}")
        print(f"\nAdaptive MTP:")
        print(f"  Mean chain length:  {summary['mean_chain']:.2f}/{max_positions}")
        print(f"  Drafted tokens:     {total_drafted:,}")
        print(f"  Accepted tokens:    {total_accepted:,}")
        print(f"  Efficiency:         {summary['adaptive_efficiency']:.3f}")
        print(f"  Draft steps saved:  {summary['draft_steps_saved_pct']:.1f}%")
        print(f"  Rollbacks fired:    {summary['rollbacks']}")
        print(f"\nFinal EMA rates:     {[f'{v:.3f}' for v in summary['final_ema']]}")
        print(f"{'='*60}\n")

        # Estimate throughput impact on GH200
        # Single request throughput: 145 tok/s, target model forward = T_verify
        # Each draft forward ≈ 1/7 of a verify forward (MTP head is lightweight).
        # Simplified model:
        #   Total time per output token = T_verify + chain * T_draft
        # where T_draft << T_verify (MTP head ≈ 2 transformer layers vs 28).
        # On GH200 (900GB/s HBM3e), memory-bandwidth-limited for decode.
        # 27B model @ bf16 ≈ 54 GB.  Time to stream = 54e9 / 900e9 ≈ 60ms.
        # MTP head ≈ 2% of full model (2/28 layers heuristic) ≈ 1.2ms per step.
        T_verify_ms = 60.0   # rough GH200 streaming time for 27B bf16
        T_draft_ms  =  2.0   # MTP head forward (2-layer equiv)
        baseline_tok_per_s = 1000.0 * (1 + baseline_accepted / (baseline_drafted / max_positions)) / (T_verify_ms + max_positions * T_draft_ms)
        adaptive_mean_chain = summary["mean_chain"]
        adaptive_tok_per_s  = 1000.0 * (1 + total_accepted / (total_drafted / adaptive_mean_chain)) / (T_verify_ms + adaptive_mean_chain * T_draft_ms)
        print(f"Throughput model (single request, GH200 heuristic):")
        print(f"  Baseline:  {baseline_tok_per_s:.1f} tok/s")
        print(f"  Adaptive:  {adaptive_tok_per_s:.1f} tok/s")
        print(f"  Delta:     {adaptive_tok_per_s - baseline_tok_per_s:+.1f} tok/s")

    return summary


# ---------------------------------------------------------------------------
# Model-runner patch helper (alternative to subclassing)
# ---------------------------------------------------------------------------

def patch_eagle_proposer(drafter) -> None:
    """Monkey-patch an existing EagleProposer instance in-place.

    Use this if you cannot change the class instantiation site.

        from adaptive_mtp import patch_eagle_proposer
        patch_eagle_proposer(self.drafter)
        # Now self.drafter.record_acceptance(...) is available
        # and propose() automatically adapts chain length.

    CAUTION: patches instance methods, not the class.  Safe for a single
    drafter instance.
    """
    import types

    original_propose = drafter.propose
    original_num_spec = drafter.num_speculative_tokens

    ctrl = AdaptiveMTPController(max_positions=original_num_spec)
    _last_chain = [original_num_spec]  # mutable cell

    def patched_propose(self_inner, *args, **kwargs):
        chain = ctrl.get_chain_length()
        _last_chain[0] = chain
        self_inner.num_speculative_tokens = chain
        try:
            result = original_propose(*args, **kwargs)
        finally:
            self_inner.num_speculative_tokens = original_num_spec
        # Pad if needed
        if chain < original_num_spec:
            pad = torch.zeros(
                result.shape[0], original_num_spec - chain,
                dtype=result.dtype, device=result.device,
            )
            result = torch.cat([result, pad], dim=1)
        return result

    def record_acceptance(accepted_per_pos: list[int], num_drafts: int) -> None:
        ctrl.record_acceptance(
            accepted_per_pos=accepted_per_pos,
            num_drafts=num_drafts,
            actual_chain=_last_chain[0],
        )

    def get_ema_rates() -> list[float]:
        return ctrl.get_ema_rates()

    drafter.propose = types.MethodType(patched_propose, drafter)
    drafter.record_acceptance = record_acceptance
    drafter.get_ema_rates = get_ema_rates
    drafter._adaptive_controller = ctrl

    logger.info("AdaptiveMTP: patched EagleProposer in-place (max_positions=%d)", original_num_spec)


# ---------------------------------------------------------------------------
# GPU model runner hook (call from propose_draft_token_ids aftermath)
# ---------------------------------------------------------------------------

def make_acceptance_hook(drafter, spec_decoding_stats_attr: str = "_spec_stats"):
    """Return a callable that feeds acceptance data back after each step.

    Usage in gpu_model_runner.py, at the end of the step where
    SpecDecodingStats is finalized:

        _acceptance_hook = make_acceptance_hook(self.drafter)
        # ... inside execute_model(), after rejection sampling:
        _acceptance_hook(spec_decoding_stats)

    Where spec_decoding_stats is a SpecDecodingStats instance with:
        .num_drafts
        .num_accepted_tokens_per_pos   (list[int], len=num_spec_tokens)
    """
    def hook(stats) -> None:
        if not hasattr(drafter, "record_acceptance"):
            return
        drafter.record_acceptance(
            accepted_per_pos=stats.num_accepted_tokens_per_pos,
            num_drafts=stats.num_drafts,
        )
    return hook


# ---------------------------------------------------------------------------
# Entry point — run the simulation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default: Qwen 3.5 MTP profile
    simulate(n_steps=2000, max_positions=7, verbose=True)

    print("\n--- Degraded tail scenario (positions 4-6 near-zero) ---")
    simulate(
        n_steps=2000,
        max_positions=7,
        true_rates=[0.82, 0.73, 0.61, 0.48, 0.12, 0.05, 0.03],
        verbose=True,
    )

    print("\n--- High-entropy text (all positions low) ---")
    simulate(
        n_steps=2000,
        max_positions=7,
        true_rates=[0.55, 0.40, 0.28, 0.18, 0.11, 0.07, 0.04],
        verbose=True,
    )
