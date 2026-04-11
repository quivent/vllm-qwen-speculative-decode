"""Enhanced MTP Proposer with DeltaNet hidden-state adjustment.

Wraps the standard EagleProposer (or AdaptiveMTPProposer) to inject
DeltaNet adjustment steps between MTP chain iterations.  The adjustment
refines the hidden state so that deeper draft positions (3-6) produce
tokens closer to what the full model would generate.

Architecture
------------
Standard MTP chain (no adjustment)::

    MTP(token_0, hidden_0) -> hidden_1 -> MTP(token_1, hidden_1) -> hidden_2 -> ...

Each step accumulates representation error because the MTP head is a
single decoder layer approximating the full 64-layer model.

Enhanced chain (with DeltaNet adjustment)::

    MTP(token_0, hidden_0) -> hidden_1
      -> DeltaNet_adjust(hidden_1) -> hidden_1'
      -> MTP(token_1, hidden_1') -> hidden_2
      -> DeltaNet_adjust(hidden_2) -> hidden_2'
      -> ...

The DeltaNet layers correct the hidden state between chain steps,
reducing the drift that causes acceptance degradation at deeper positions.

Integration
-----------
In ``gpu_model_runner.py`` (or wherever the proposer is created)::

    from enhanced_mtp_proposer import EnhancedMTPProposer

    # Instead of:
    #   self.drafter = EagleProposer(vllm_config, device, self)
    # Use:
    self.drafter = EnhancedMTPProposer(
        vllm_config, device, self,
        num_adjust_layers=4,
        adjust_after_position=1,  # skip adjustment for pos 0-1 (already accurate)
    )

The enhanced proposer is a drop-in replacement.  It overrides only the
chain portion of ``propose()`` — the first MTP forward pass and all
scheduling/metadata handling remain unchanged.

Environment Variables
---------------------
``DELTANET_ADJUST_LAYERS``
    Number of DeltaNet layers to use (default: 4).
``DELTANET_ADJUST_AFTER``
    Start adjusting after this chain position (default: 1).
    Positions 0 and 1 typically have high acceptance already.
``DELTANET_ADJUST_STRATEGY``
    Layer selection: ``spread``, ``last``, ``first`` (default: ``spread``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch import nn

from deltanet_adjuster import DeltaNetAdjuster, estimate_cost

logger = logging.getLogger(__name__)


class EnhancedMTPProposer:
    """EagleProposer wrapper that adds DeltaNet adjustment between chain steps.

    This class wraps an existing proposer instance (EagleProposer or any
    subclass) rather than inheriting from it.  This avoids import-time
    coupling to vLLM internals and works with any proposer that has the
    standard ``propose()`` interface.

    The enhancement is injected by monkey-patching the wrapped proposer's
    ``model.chain_forward`` method (if it exists) or by overriding the
    ``propose()`` method entirely.

    Parameters
    ----------
    proposer : object
        An EagleProposer (or AdaptiveMTPProposer) instance.
    model_backbone : nn.Module
        The Qwen3_5Model backbone (``runner.model.model.model`` typically)
        from which DeltaNet layers are extracted.
    num_adjust_layers : int
        Number of DeltaNet layers for the adjuster (default from env).
    adjust_after_position : int
        Chain position after which to start adjusting (default from env).
    strategy : str
        Layer selection strategy for DeltaNetAdjuster.
    """

    def __init__(
        self,
        proposer: object,
        model_backbone: nn.Module,
        num_adjust_layers: Optional[int] = None,
        adjust_after_position: Optional[int] = None,
        strategy: Optional[str] = None,
    ):
        self._proposer = proposer
        self._num_adjust_layers = num_adjust_layers or int(
            os.environ.get("DELTANET_ADJUST_LAYERS", "4")
        )
        self._adjust_after = adjust_after_position if adjust_after_position is not None else int(
            os.environ.get("DELTANET_ADJUST_AFTER", "1")
        )
        self._strategy = strategy or os.environ.get(
            "DELTANET_ADJUST_STRATEGY", "spread"
        )

        # Build the adjuster from the backbone's DeltaNet layers
        self._adjuster = DeltaNetAdjuster.from_model(
            model_backbone,
            num_layers=self._num_adjust_layers,
            strategy=self._strategy,
        )

        # Stats
        self._step_count = 0
        self._adjustments_applied = 0

        # Log cost estimate
        est = estimate_cost(num_deltanet_layers=self._num_adjust_layers)
        logger.info(
            "EnhancedMTPProposer: %d DeltaNet adjustment layers, "
            "adjust after position %d, strategy=%s, "
            "adjustment cost = %.1f%% of 1 MTP step",
            self._num_adjust_layers,
            self._adjust_after,
            self._strategy,
            est.adjustment_vs_mtp * 100,
        )
        logger.info("Adjuster: %s", self._adjuster)

    def __getattr__(self, name: str):
        """Delegate everything to the wrapped proposer."""
        return getattr(self._proposer, name)

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
        """Propose draft tokens with DeltaNet adjustment between chain steps.

        The first MTP forward pass runs exactly as in the base proposer.
        For subsequent chain steps (positions > adjust_after), we intercept
        the hidden state after each MTP step and run it through the
        DeltaNet adjuster before feeding it back.

        Implementation approach:
        We temporarily wrap the draft model's ``chain_forward`` (if it
        uses chained MTP) or intercept the hidden states in the
        propose loop by wrapping ``self.model.__call__``.
        """
        self._step_count += 1
        p = self._proposer

        # Check if the model supports chain_forward (Qwen3.5 MTP models do)
        has_chain_forward = hasattr(p, 'model') and hasattr(p.model, 'chain_forward')

        if not has_chain_forward:
            # No chain_forward — fall back to base propose with hidden
            # state interception via the model forward wrapper.
            return self._propose_with_forward_interception(
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

        # Strategy: wrap model.__call__ to inject adjustment after each
        # chain step.  We count forward calls — the first is the initial
        # MTP pass, subsequent ones are chain steps.
        return self._propose_with_adjustment(
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

    def _propose_with_adjustment(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs,
        num_rejected_tokens_gpu,
        slot_mappings,
    ) -> torch.Tensor:
        """Intercept hidden states between chain steps via model wrapper.

        We temporarily replace the draft model's forward method with one
        that applies DeltaNet adjustment to the ``hidden_states`` input
        (which carries the previous step's output) for chain positions
        beyond ``self._adjust_after``.
        """
        p = self._proposer
        model = p.model
        original_forward = model.forward
        chain_position = [0]  # mutable counter in closure
        adjuster = self._adjuster
        adjust_after = self._adjust_after
        adjustments = [0]

        def adjusted_forward(
            input_ids=None,
            positions=None,
            hidden_states=None,
            intermediate_tensors=None,
            inputs_embeds=None,
            **kwargs,
        ):
            """Wrapper that adjusts hidden_states before chain steps."""
            pos = chain_position[0]
            chain_position[0] += 1

            # Apply adjustment for chain positions beyond the threshold.
            # The hidden_states parameter carries the previous MTP output.
            if (
                pos > adjust_after
                and hidden_states is not None
            ):
                hidden_states = adjuster.adjust(hidden_states)
                adjustments[0] += 1

            return original_forward(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        # Temporarily replace forward
        model.forward = adjusted_forward
        try:
            result = p.propose(
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
            model.forward = original_forward

        self._adjustments_applied += adjustments[0]

        if self._step_count % 500 == 0:
            logger.info(
                "EnhancedMTP [step=%d]: total adjustments=%d, "
                "avg adjustments/propose=%.1f",
                self._step_count,
                self._adjustments_applied,
                self._adjustments_applied / max(1, self._step_count),
            )

        return result

    def _propose_with_forward_interception(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs,
        num_rejected_tokens_gpu,
        slot_mappings,
    ) -> torch.Tensor:
        """Fallback: adjust target_hidden_states before the first pass.

        If the model doesn't use chain_forward, we can still improve
        the initial hidden state fed into the MTP head.  This gives a
        smaller benefit but is always applicable.
        """
        # Adjust the target hidden states before they enter the MTP head
        target_hidden_states = self._adjuster.adjust(target_hidden_states)

        return self._proposer.propose(
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

    # ------------------------------------------------------------------
    # Passthrough for common proposer attributes
    # ------------------------------------------------------------------

    @property
    def num_speculative_tokens(self):
        return self._proposer.num_speculative_tokens

    @num_speculative_tokens.setter
    def num_speculative_tokens(self, value):
        self._proposer.num_speculative_tokens = value

    def record_acceptance(self, *args, **kwargs):
        """Delegate to wrapped proposer if it supports adaptive feedback."""
        if hasattr(self._proposer, 'record_acceptance'):
            return self._proposer.record_acceptance(*args, **kwargs)


# ---------------------------------------------------------------------------
# Throughput model: is the adjustment worth it?
# ---------------------------------------------------------------------------

def analyze_tradeoff(
    current_rates: Optional[list[float]] = None,
    improved_rates: Optional[list[float]] = None,
    num_adjust_layers: int = 4,
    num_chain_steps: int = 7,
) -> dict:
    """Analyze the throughput tradeoff of DeltaNet adjustment.

    Parameters
    ----------
    current_rates : list[float]
        Per-position acceptance rates without adjustment.
    improved_rates : list[float]
        Expected per-position acceptance rates with adjustment.
    num_adjust_layers : int
        DeltaNet layers used per adjustment step.
    num_chain_steps : int
        Total MTP chain length.

    Returns
    -------
    dict with analysis results.
    """
    if current_rates is None:
        # Empirical Qwen3.5 MTP rates from benchmark data
        current_rates = [0.87, 0.68, 0.54, 0.39, 0.28, 0.21, 0.16]
    if improved_rates is None:
        # Conservative estimate: adjustment corrects ~50% of the drift
        # at each position.  Positions 0-1 unchanged (already accurate).
        improved_rates = [0.87, 0.70, 0.62, 0.55, 0.45, 0.36, 0.28]

    est = estimate_cost(num_deltanet_layers=num_adjust_layers)

    current_expected = sum(current_rates[:num_chain_steps])
    improved_expected = sum(improved_rates[:num_chain_steps])

    # Draft cost: MTP steps + adjustment steps
    # Adjustment happens between steps, so (num_chain_steps - 1) adjustments
    # (but we skip the first adjust_after positions)
    num_adjustments = num_chain_steps - 1  # worst case: adjust after every step
    mtp_only_cost = num_chain_steps * est.mtp_step_flops
    adjustment_cost = num_adjustments * est.adjustment_total_flops
    total_draft_cost = mtp_only_cost + adjustment_cost

    # Full model verify cost
    verify_cost = est.full_model_forward_flops

    # Throughput = expected_accepted_tokens / (verify_cost + draft_cost)
    current_throughput = current_expected / (verify_cost + mtp_only_cost)
    improved_throughput = improved_expected / (verify_cost + total_draft_cost)

    result = {
        "current_expected_tokens": current_expected,
        "improved_expected_tokens": improved_expected,
        "token_gain_pct": (improved_expected / current_expected - 1) * 100,
        "draft_cost_overhead_pct": (total_draft_cost / mtp_only_cost - 1) * 100,
        "net_throughput_change_pct": (improved_throughput / current_throughput - 1) * 100,
        "adjustment_flops_per_mtp_step_pct": est.adjustment_vs_mtp * 100,
        "break_even_improvement_factor": (
            # What improvement factor at positions 2-6 makes this break even?
            # break_even: improved_expected / (verify + total_draft) = current_expected / (verify + mtp_only)
            # => improved_expected = current_expected * (verify + total_draft) / (verify + mtp_only)
            current_expected * (verify_cost + total_draft_cost)
            / (verify_cost + mtp_only_cost)
            / current_expected
        ),
    }

    return result


def print_tradeoff_analysis():
    """Print full tradeoff analysis to stdout."""
    result = analyze_tradeoff()

    print("DeltaNet Adjustment Tradeoff Analysis")
    print("=" * 50)
    print(f"Expected tokens (current):  {result['current_expected_tokens']:.3f}")
    print(f"Expected tokens (improved): {result['improved_expected_tokens']:.3f}")
    print(f"Token gain:                 {result['token_gain_pct']:+.1f}%")
    print(f"Draft cost overhead:        {result['draft_cost_overhead_pct']:+.1f}%")
    print(f"Net throughput change:       {result['net_throughput_change_pct']:+.1f}%")
    print(f"Break-even factor:          {result['break_even_improvement_factor']:.3f}x")
    print()
    print(f"Adjustment cost per step:   {result['adjustment_flops_per_mtp_step_pct']:.1f}% of MTP step")
    print()

    # Sensitivity analysis: what if the improvement is larger or smaller?
    print("Sensitivity analysis (varying improvement magnitude):")
    print(f"{'Scenario':<30} {'Expected':<10} {'Overhead':<10} {'Net':>8}")
    print("-" * 58)

    scenarios = [
        ("No improvement", [0.87, 0.68, 0.54, 0.39, 0.28, 0.21, 0.16]),
        ("25% drift correction", [0.87, 0.69, 0.57, 0.45, 0.33, 0.25, 0.19]),
        ("50% drift correction", [0.87, 0.70, 0.62, 0.55, 0.45, 0.36, 0.28]),
        ("75% drift correction", [0.87, 0.72, 0.68, 0.62, 0.53, 0.43, 0.35]),
        ("Full correction", [0.87, 0.75, 0.72, 0.68, 0.60, 0.50, 0.40]),
    ]

    for name, rates in scenarios:
        r = analyze_tradeoff(improved_rates=rates)
        print(
            f"  {name:<28} {r['improved_expected_tokens']:.3f}     "
            f"{r['draft_cost_overhead_pct']:+.1f}%   "
            f"{r['net_throughput_change_pct']:+.1f}%"
        )

    # Layer count sensitivity
    print()
    print("Layer count sensitivity (at 50% drift correction):")
    print(f"  {'Layers':<8} {'Overhead':<12} {'Net throughput':>14}")
    print("  " + "-" * 34)
    for n_layers in [1, 2, 3, 4, 6]:
        r = analyze_tradeoff(num_adjust_layers=n_layers)
        print(
            f"  {n_layers:<8} {r['draft_cost_overhead_pct']:+.1f}%{'':<6}"
            f"{r['net_throughput_change_pct']:+.1f}%"
        )


if __name__ == "__main__":
    print()
    print_tradeoff_analysis()
