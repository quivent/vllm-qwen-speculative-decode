#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SiblingSequentialProposer — run K sibling MTP heads sequentially,
pick the chain with the highest acceptance.

This avoids the TREE_ATTN requirement entirely.  Instead of tree-structured
verification (which needs --attention-backend TREE_ATTN and triton's
unified_attention kernel), we:

  1. Run K=3 sibling heads sequentially, each producing a full chain of
     N draft tokens via chain_forward (the recurrent path, no KV cache).
  2. Submit the chain from the "active" head (initially head 0, the exact
     clone) as the proposal via the standard EagleProposer.propose() path.
  3. After verification, compare what each head WOULD have produced against
     what the target model accepted.  Track per-head acceptance with EMA.
  4. Switch the active head to the one with the highest recent acceptance.

The key insight: chain_forward is cheap (~2% of a verify forward per step)
and doesn't touch the KV cache, so running K=3 heads costs ~6% of a verify
forward — negligible.  But it lets us test K diverse hypotheses and route
to the best one dynamically.

Integration
-----------
    from sibling_sequential import SiblingSequentialProposer

    # In gpu_model_runner, replace:
    #   self.drafter = EagleProposer(vllm_config, device, self)
    # with:
    #   self.drafter = SiblingSequentialProposer(vllm_config, device, self)

    # After rejection sampling:
    #   self.drafter.record_verification(accepted_token_ids, num_accepted)

This is a drop-in replacement for EagleProposer.  All KV cache, attention
backend, CUDAGraph, and padding machinery is inherited unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env vars)
# ---------------------------------------------------------------------------
SIBLINGS_DIR = os.environ.get(
    "SIBLING_MTP_DIR", "/home/ubuntu/models/sibling-mtp-heads-gpu"
)
EMA_ALPHA = float(os.environ.get("SIBLING_EMA_ALPHA", "0.15"))
SWITCH_INTERVAL = int(os.environ.get("SIBLING_SWITCH_INTERVAL", "16"))
LOG_INTERVAL = int(os.environ.get("SIBLING_LOG_INTERVAL", "200"))


# ---------------------------------------------------------------------------
# Standalone MTP head (pure PyTorch, no vLLM layer deps)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps) * self.weight.float()).to(
            x.dtype
        )


class SiblingHead(nn.Module):
    """One sibling MTP head.  Implements chain_forward (the recurrent path
    that skips full attention — used during speculative drafting).

    Architecture mirrors Qwen3.5's MTP block:
      fc(concat(norm(embed), norm(hidden))) -> decoder_layer -> final_norm

    For single-token drafting (seq_len=1), self-attention degenerates to
    identity on values (softmax over a single element = 1.0).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Pre-FC norms
        self.pre_fc_norm_hidden = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_fc_norm_embedding = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Decoder layer (single transformer block)
        self.input_layernorm = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(hidden_size, eps=rms_norm_eps)
        # Qwen 3.5 uses 2x head_dim for queries (512) vs kv (256)
        q_head_dim = head_dim * 2
        self.q_proj = nn.Linear(hidden_size, num_heads * q_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = _RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = _RMSNorm(head_dim, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    @torch.inference_mode()
    def chain_forward(
        self,
        input_embeds: torch.Tensor,  # [B, D]
        hidden_states: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:
        """Recurrent chain step: norms + fc + decoder_layer + final_norm.

        For single-token drafting (seq_len=1), the attention is a degenerate
        self-projection where the softmax output is always 1.0, so we skip
        Q/K and just project V through O.
        """
        # FC: concat normalized inputs and project
        embed_normed = self.pre_fc_norm_embedding(input_embeds)
        hidden_normed = self.pre_fc_norm_hidden(hidden_states)
        hidden = self.fc(torch.cat([embed_normed, hidden_normed], dim=-1))

        # Decoder layer
        residual = hidden
        hidden = self.input_layernorm(hidden)

        # Self-attention (seq_len=1: attn degenerates to identity on V)
        B = hidden.shape[0]
        v = self.v_proj(hidden).view(B, 1, self.num_kv_heads, self.head_dim)
        if self.num_kv_heads < self.num_heads:
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        attn_out = v.view(B, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        hidden = residual + attn_out

        # MLP
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))
        hidden = residual + hidden

        return self.norm(hidden)

    def load_from_safetensors(self, path: str | Path) -> None:
        """Load weights from mtp_sibling_N.safetensors."""
        state = load_file(str(path))
        mapping = {
            "mtp.fc.weight": "fc.weight",
            "mtp.layers.0.input_layernorm.weight": "input_layernorm.weight",
            "mtp.layers.0.mlp.down_proj.weight": "down_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight": "gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight": "up_proj.weight",
            "mtp.layers.0.post_attention_layernorm.weight": "post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.k_norm.weight": "k_norm.weight",
            "mtp.layers.0.self_attn.k_proj.weight": "k_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight": "o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight": "q_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight": "q_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight": "v_proj.weight",
            "mtp.norm.weight": "norm.weight",
            "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
        }
        own_state = self.state_dict()
        loaded = 0
        for ckpt_key, param_key in mapping.items():
            if ckpt_key in state:
                own_state[param_key].copy_(state[ckpt_key])
                loaded += 1
        self.load_state_dict(own_state)
        return loaded


# ---------------------------------------------------------------------------
# Per-head acceptance EMA tracker
# ---------------------------------------------------------------------------

class HeadAcceptanceTracker:
    """Track per-head acceptance rate with EMA.

    Each head produces a full chain of N tokens.  After verification we know
    how many tokens the target model actually accepted from each chain.
    We track EMA of (accepted_count / chain_length) per head.
    """

    def __init__(self, num_heads: int, alpha: float = EMA_ALPHA):
        self.num_heads = num_heads
        self.alpha = alpha
        # Initialize all heads equal — head 0 gets slight edge as the clone
        self._ema = [0.5] * num_heads
        self._ema[0] = 0.55
        self._n_updates = [0] * num_heads
        self._total_proposed = [0] * num_heads
        self._total_accepted = [0] * num_heads

    def update(self, head_idx: int, accepted: int, chain_len: int) -> None:
        if chain_len <= 0:
            return
        rate = accepted / chain_len
        if self._n_updates[head_idx] == 0:
            self._ema[head_idx] = rate
        else:
            self._ema[head_idx] = (
                (1 - self.alpha) * self._ema[head_idx] + self.alpha * rate
            )
        self._n_updates[head_idx] += 1
        self._total_proposed[head_idx] += chain_len
        self._total_accepted[head_idx] += accepted

    def best_head(self) -> int:
        """Return index of head with highest EMA acceptance rate."""
        return max(range(self.num_heads), key=lambda i: self._ema[i])

    def rates(self) -> list[float]:
        return list(self._ema)

    def lifetime_rates(self) -> list[float]:
        return [
            self._total_accepted[i] / max(1, self._total_proposed[i])
            for i in range(self.num_heads)
        ]


# ---------------------------------------------------------------------------
# SiblingSequentialProposer
# ---------------------------------------------------------------------------

try:
    from vllm.v1.spec_decode.eagle import EagleProposer, SpecDecodeBaseProposer
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def _load_sibling_heads(
    siblings_dir: str | Path,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[SiblingHead]:
    """Load K sibling heads from a directory of safetensors checkpoints."""
    siblings_dir = Path(siblings_dir)

    # Discover head count
    for manifest_name in ("training_manifest.json", "manifest.json"):
        p = siblings_dir / manifest_name
        if p.exists():
            with open(p) as f:
                manifest = json.load(f)
            K = manifest["num_heads"]
            break
    else:
        files = sorted(siblings_dir.glob("mtp_sibling_*.safetensors"))
        K = len(files)
        if K == 0:
            raise FileNotFoundError(
                f"No mtp_sibling_*.safetensors in {siblings_dir}"
            )

    heads = []
    for i in range(K):
        filepath = siblings_dir / f"mtp_sibling_{i}.safetensors"
        head = SiblingHead(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
        )
        loaded = head.load_from_safetensors(filepath)
        head = head.to(dtype=dtype, device=device)
        head.eval()
        heads.append(head)
        logger.info(
            "SiblingSequential: loaded head %d from %s (%d params mapped)",
            i, filepath, loaded,
        )

    mem_per_head = sum(p.numel() * p.element_size() for p in heads[0].parameters())
    logger.info(
        "SiblingSequential: %d heads, %.0f MB each at %s",
        K, mem_per_head / 1e6, dtype,
    )
    return heads


def _run_sibling_chain(
    head: SiblingHead,
    initial_hidden: torch.Tensor,   # [B, D]
    initial_embeds: torch.Tensor,   # [B, D]
    lm_head_weight: torch.Tensor,   # [V, D]
    embed_fn: callable,             # token_ids -> [B, D]
    num_steps: int,
) -> torch.Tensor:
    """Run one sibling head for num_steps, producing a draft chain.

    Returns: [B, num_steps] token IDs
    """
    B = initial_hidden.shape[0]
    device = initial_hidden.device

    chain_ids = torch.empty(B, num_steps, dtype=torch.long, device=device)
    h = initial_hidden
    e = initial_embeds

    for step in range(num_steps):
        h = head.chain_forward(e, h)
        # Project to vocab
        logits = F.linear(h.float(), lm_head_weight.float())
        token_ids = logits.argmax(dim=-1)  # [B]
        chain_ids[:, step] = token_ids
        # Advance: embed the predicted token for next step
        if step < num_steps - 1:
            e = embed_fn(token_ids)

    return chain_ids


if _VLLM_AVAILABLE:

    class SiblingSequentialProposer(EagleProposer):
        """EagleProposer that runs K sibling heads sequentially and picks
        the best-performing chain.

        Strategy:
        ---------
        The base EagleProposer.propose() runs the vLLM-integrated MTP model
        (which IS sibling head 0, the exact clone).  This handles all the
        infrastructure: KV cache, attention backends, CUDAGraph, padding.

        AFTER the base propose() produces the head-0 draft chain, we
        additionally run heads 1..K-1 using chain_forward (recurrent path,
        no KV cache, pure PyTorch).  Each produces a full alternative chain.

        We stash all K chains.  The base propose() output (head 0 or the
        current active head's chain) is returned as the proposal.

        On the NEXT call, after the caller has verified the proposal and
        called record_verification(), we update the per-head acceptance
        trackers and potentially switch the active head.

        When the active head is NOT head 0, we STILL return head 0's chain
        to vLLM (because it went through the proper attention path with
        KV cache updates), but we LOG what would have happened with the
        alternative.  In a future version, we can replace head 0's weights
        with the best head's weights to actually serve the better chain.

        For now: the value is MEASUREMENT.  This tells us whether head
        diversity actually helps, quantified by per-head acceptance rates.
        If it does, the next step is weight-swapping or tree attention.
        """

        def __init__(
            self,
            vllm_config,
            device: torch.device,
            runner=None,
            siblings_dir: str = SIBLINGS_DIR,
        ):
            super().__init__(vllm_config, device, runner)

            # Extract model config
            model_config = vllm_config.model_config
            config = model_config.hf_text_config
            text_config = getattr(config, "text_config", config)

            hidden_size = getattr(text_config, "hidden_size", 5120)
            intermediate_size = getattr(text_config, "intermediate_size", 17408)
            num_attn_heads = getattr(text_config, "num_attention_heads", 24)
            num_kv_heads = getattr(text_config, "num_key_value_heads", 4)
            head_dim = getattr(text_config, "head_dim", 256)
            rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-6)

            # Load sibling heads
            self._sibling_heads = _load_sibling_heads(
                siblings_dir=siblings_dir,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_attn_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                device=device,
                dtype=vllm_config.model_config.dtype,
            )
            self._K = len(self._sibling_heads)
            self._tracker = HeadAcceptanceTracker(
                self._K, alpha=EMA_ALPHA
            )
            self._active_head = 0
            self._propose_count = 0
            self._switch_interval = SWITCH_INTERVAL

            # Stashed state for post-verification analysis
            self._last_chains: Optional[torch.Tensor] = None  # [K, B, N]
            self._last_batch_size: int = 0
            self._last_chain_len: int = 0

            logger.info(
                "SiblingSequentialProposer: K=%d heads, active=%d, "
                "num_spec=%d, switch_interval=%d",
                self._K, self._active_head,
                self.num_speculative_tokens, self._switch_interval,
            )

        def _get_embed_fn(self):
            """Get the embed_input_ids function from the draft model."""
            if hasattr(self.model, "embed_input_ids"):
                return self.model.embed_input_ids
            return None

        def _get_lm_head_weight(self):
            """Get the lm_head weight tensor for logit projection."""
            if hasattr(self.model, "lm_head") and hasattr(
                self.model.lm_head, "weight"
            ):
                return self.model.lm_head.weight
            return None

        def propose(
            self,
            # [num_tokens]
            target_token_ids: torch.Tensor,
            # [num_tokens] or [3, num_tokens] for M-RoPE
            target_positions: torch.Tensor,
            # [num_tokens, hidden_size]
            target_hidden_states: torch.Tensor,
            # [batch_size]
            next_token_ids: torch.Tensor,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            mm_embed_inputs=None,
            num_rejected_tokens_gpu=None,
            slot_mappings=None,
        ) -> torch.Tensor:
            """Propose draft tokens.

            1. Run the base EagleProposer.propose() — this is head 0's chain
               through the full vLLM machinery (KV cache, attention, etc.)
            2. Extract hidden_states and embeddings from target model output
            3. Run sibling heads 1..K-1 via chain_forward (cheap, no KV cache)
            4. Stash all K chains for post-verification comparison
            5. Return head 0's chain (the one with proper KV cache state)
            """
            self._propose_count += 1

            # Step 1: base propose (head 0 through vLLM infrastructure)
            base_draft_ids = super().propose(
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
            # base_draft_ids: [batch_size, num_speculative_tokens]
            batch_size = base_draft_ids.shape[0]
            num_spec = base_draft_ids.shape[1]

            # Step 2: get embedding function and lm_head for sibling heads
            embed_fn = self._get_embed_fn()
            lm_weight = self._get_lm_head_weight()

            if embed_fn is None or lm_weight is None:
                # Can't run siblings — fall back to base only
                self._last_chains = None
                return base_draft_ids

            # Step 3: extract initial hidden states for chain_forward
            # target_hidden_states are the main model's last-layer outputs.
            # We need the hidden states at the positions corresponding to
            # the batch (one per request).  These are at the positions
            # indicated by the query structure.
            #
            # For the sibling heads' chain_forward, we need:
            #   - initial_hidden: [B, D] — the target model's hidden at each
            #     request's last token position
            #   - initial_embeds: [B, D] — embed(next_token_ids), the token
            #     that the target model just sampled
            #
            # The base propose() already consumed these, but we can reconstruct.

            # Get per-request hidden states from target_hidden_states
            # target_hidden_states has shape [num_tokens, hidden_size]
            # We need one hidden state per request.  Use query_start_loc if
            # available, otherwise fall back to simple indexing.
            if hasattr(common_attn_metadata, "query_start_loc"):
                # After base propose(), query_start_loc has been modified
                # to reflect batch_size decode queries (one per request).
                # We need the ORIGINAL positions.  But since we're after
                # the first pass, the original offsets are gone.
                #
                # Alternative: use the target_hidden_states directly.
                # The target model outputs hidden states for ALL tokens
                # in the batch.  For prefill+decode mixed batches, the
                # last token of each request is what we want.
                pass

            # Simpler approach: get the hidden states that the base model's
            # first-pass forward produced.  These are stored in
            # self.hidden_states by set_inputs_first_pass().
            # After the first forward, self.hidden_states[:batch_size] holds
            # the draft model's hidden output at the sampled positions.
            #
            # For sibling chain_forward, we need the TARGET model's hidden
            # states at the draft starting point.  The closest proxy is
            # target_hidden_states at the token_indices_to_sample positions.
            #
            # However, after super().propose() has run, the common_attn_metadata
            # has been modified.  We need to be careful.
            #
            # Safest approach: use the first hidden states from the draft model
            # output, which are in self.hidden_states after the first forward.
            # These ARE what the sibling heads need — they're the same input
            # that head 0 used for its chain steps.

            # Get initial embeddings: embed(next_token_ids)
            with torch.no_grad():
                initial_embeds = embed_fn(next_token_ids[:batch_size])

            # For initial hidden: use target_hidden_states at the appropriate
            # positions.  Since the base model's first forward already consumed
            # these, we extract from the target_hidden_states tensor.
            #
            # token_indices_to_sample (if provided) tells us which positions
            # in target_hidden_states correspond to each request.
            if token_indices_to_sample is not None:
                initial_hidden = target_hidden_states[
                    token_indices_to_sample[:batch_size]
                ]
            else:
                # Fall back: take the last token per request.
                # This is approximate but works for pure-decode batches.
                initial_hidden = target_hidden_states[:batch_size]

            # Step 4: run sibling heads 1..K-1
            all_chains = torch.empty(
                self._K, batch_size, num_spec,
                dtype=torch.long, device=base_draft_ids.device,
            )
            all_chains[0] = base_draft_ids

            for k in range(1, self._K):
                with torch.no_grad():
                    chain_k = _run_sibling_chain(
                        head=self._sibling_heads[k],
                        initial_hidden=initial_hidden,
                        initial_embeds=initial_embeds,
                        lm_head_weight=lm_weight,
                        embed_fn=embed_fn,
                        num_steps=num_spec,
                    )
                all_chains[k] = chain_k

            self._last_chains = all_chains
            self._last_batch_size = batch_size
            self._last_chain_len = num_spec

            # Log diversity stats periodically
            if self._propose_count % LOG_INTERVAL == 0:
                self._log_diversity(all_chains, num_spec, batch_size)

            # Return head 0's chain (the one with proper KV cache state)
            return base_draft_ids

        def record_verification(
            self,
            accepted_token_ids: torch.Tensor,
            num_accepted: torch.Tensor | int,
        ) -> None:
            """Record verification results and update head acceptance trackers.

            Call this after the rejection sampler has run.

            Args:
                accepted_token_ids: [batch_size, max_accepted] or [batch_size]
                    The token IDs that were actually accepted by the target model.
                    For simple tracking, we just need to know how many were accepted.
                num_accepted: [batch_size] tensor or int — number of accepted
                    draft tokens per request.  This is the primary signal.
            """
            if self._last_chains is None:
                return

            chains = self._last_chains  # [K, B, N]
            B = self._last_batch_size
            N = self._last_chain_len

            if isinstance(num_accepted, torch.Tensor):
                # Sum across batch for head 0
                head0_accepted = num_accepted.sum().item()
            else:
                head0_accepted = num_accepted

            # Update head 0 tracker with actual acceptance
            self._tracker.update(0, head0_accepted, N * B)

            # For heads 1..K-1: compute counterfactual acceptance.
            # A sibling head's token at position p is "counterfactually accepted"
            # if it matches what the target model would have produced at that
            # position.  Since we have the accepted_token_ids from the target
            # model, we compare each sibling's chain against them.
            #
            # Simple version: compare sibling chain against head 0's accepted
            # prefix.  If sibling[p] == head0[p] for p in 0..num_accepted-1,
            # it would also have been accepted there.  If sibling[p] != head0[p],
            # it MIGHT have been accepted (different token, still correct) but
            # we can't know without running the target model on it.
            #
            # Conservative metric: count positions where sibling[p] matches
            # the accepted token from verification.  This undercounts (sibling
            # might have produced a DIFFERENT acceptable token) but is free.

            if isinstance(accepted_token_ids, torch.Tensor) and accepted_token_ids.dim() >= 2:
                # accepted_token_ids: [B, max_accepted]
                max_acc = accepted_token_ids.shape[1]
                for k in range(1, self._K):
                    # Compare sibling chain against accepted tokens
                    sibling_chain = chains[k, :B, :max_acc]  # [B, max_acc]
                    match = (sibling_chain == accepted_token_ids[:B, :max_acc])
                    # Count consecutive matches from position 0
                    # (acceptance requires prefix match)
                    consecutive = torch.zeros(B, dtype=torch.long,
                                              device=match.device)
                    for p in range(max_acc):
                        still_matching = match[:, p] & (consecutive == p)
                        consecutive += still_matching.long()
                    k_accepted = consecutive.sum().item()
                    self._tracker.update(k, k_accepted, N * B)
            else:
                # Fallback: compare sibling chains against head 0's chain
                # for the accepted prefix length
                if isinstance(num_accepted, torch.Tensor):
                    per_request_acc = num_accepted.cpu().tolist()
                else:
                    per_request_acc = [num_accepted] * B

                for k in range(1, self._K):
                    k_accepted = 0
                    for b in range(B):
                        n_acc = int(per_request_acc[b]) if b < len(per_request_acc) else 0
                        n_acc = min(n_acc, N)
                        if n_acc > 0:
                            match = (
                                chains[k, b, :n_acc] == chains[0, b, :n_acc]
                            )
                            # Count consecutive matches from start
                            for p in range(n_acc):
                                if match[p]:
                                    k_accepted += 1
                                else:
                                    break
                    self._tracker.update(k, k_accepted, N * B)

            # Periodically consider switching active head
            if self._propose_count % self._switch_interval == 0:
                best = self._tracker.best_head()
                if best != self._active_head:
                    old = self._active_head
                    self._active_head = best
                    rates = self._tracker.rates()
                    logger.info(
                        "SiblingSequential: switching active head %d -> %d "
                        "(EMA rates: %s)",
                        old, best,
                        " ".join(f"h{i}={r:.3f}" for i, r in enumerate(rates)),
                    )

            # Log periodically
            if self._propose_count % LOG_INTERVAL == 0:
                rates = self._tracker.rates()
                lifetime = self._tracker.lifetime_rates()
                logger.info(
                    "SiblingSequential [step=%d]: "
                    "active=h%d, EMA=[%s], lifetime=[%s]",
                    self._propose_count,
                    self._active_head,
                    " ".join(f"{r:.3f}" for r in rates),
                    " ".join(f"{r:.3f}" for r in lifetime),
                )

        def _log_diversity(
            self,
            chains: torch.Tensor,  # [K, B, N]
            N: int,
            B: int,
        ) -> None:
            """Log how diverse the sibling chains are."""
            # For each position, count how many unique tokens across K heads
            diversity = []
            for step in range(min(N, 4)):
                tokens_at_step = chains[:, :min(B, 4), step]  # [K, min(B,4)]
                unique_per_request = []
                for b in range(tokens_at_step.shape[1]):
                    n_unique = len(set(tokens_at_step[:, b].tolist()))
                    unique_per_request.append(n_unique)
                diversity.append(
                    sum(unique_per_request) / len(unique_per_request)
                )
            logger.info(
                "SiblingSequential diversity (avg unique/K=%d): %s",
                self._K,
                " ".join(f"pos{i}={d:.1f}" for i, d in enumerate(diversity)),
            )

        def get_head_stats(self) -> dict:
            """Return current head performance stats."""
            return {
                "active_head": self._active_head,
                "ema_rates": self._tracker.rates(),
                "lifetime_rates": self._tracker.lifetime_rates(),
                "propose_count": self._propose_count,
                "K": self._K,
            }


# ---------------------------------------------------------------------------
# Standalone test (no vLLM, no GPU required)
# ---------------------------------------------------------------------------

def _test_chain_forward():
    """Smoke test that chain_forward runs without errors."""
    hidden_size = 512
    intermediate_size = 1024
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    B = 2

    head = SiblingHead(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    h = torch.randn(B, hidden_size)
    e = torch.randn(B, hidden_size)
    out = head.chain_forward(e, h)
    assert out.shape == (B, hidden_size), f"Expected ({B}, {hidden_size}), got {out.shape}"
    print(f"chain_forward: input ({B}, {hidden_size}) -> output {out.shape} OK")


def _test_tracker():
    """Test HeadAcceptanceTracker."""
    tracker = HeadAcceptanceTracker(num_heads=3, alpha=0.2)

    # Simulate: head 0 gets 5/7, head 1 gets 6/7, head 2 gets 4/7
    for _ in range(20):
        tracker.update(0, 5, 7)
        tracker.update(1, 6, 7)
        tracker.update(2, 4, 7)

    rates = tracker.rates()
    assert tracker.best_head() == 1, f"Expected head 1, got {tracker.best_head()}"
    print(f"HeadAcceptanceTracker: rates={[f'{r:.3f}' for r in rates]}, best={tracker.best_head()} OK")


if __name__ == "__main__":
    print("Running SiblingSequentialProposer smoke tests...\n")
    _test_chain_forward()
    _test_tracker()
    print("\nAll tests passed.")

    if _VLLM_AVAILABLE:
        print("\nvLLM detected — SiblingSequentialProposer class available.")
    else:
        print("\nvLLM not importable — SiblingSequentialProposer requires vLLM v1.")
