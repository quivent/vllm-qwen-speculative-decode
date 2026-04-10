#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Sibling MTP Proposer for vLLM.

Loads K sibling MTP heads (fine-tuned for diversity) and runs them in
parallel at draft time.  Each head produces one candidate token per
draft position, yielding K candidates per position -> tree-structured
verification via vLLM's tree attention.

Architecture
------------
Standard single-head MTP (what vLLM ships):
    At each draft position t, run the single MTP head once -> 1 candidate token.
    Total draft sequence: linear chain of N tokens.

Sibling MTP (this module):
    At each draft position t, run K heads in parallel -> K candidate tokens.
    Total draft structure: tree with branching factor K at each of N positions.
    Verification uses tree attention (already in vLLM) to check all branches
    in a single forward pass.

The key insight: each sibling head is cheap (~200 MB at int4) and shares
embed_tokens + lm_head with the main model.  Running K=3 heads costs ~3x
the MTP compute (~6% of a verify forward) but produces 3x the candidates,
dramatically increasing the chance that at least one candidate is accepted.

Integration
-----------
    from sibling_mtp_proposer import SiblingMTPProposer

    # In gpu_model_runner, replace:
    #   self.drafter = EagleProposer(vllm_config, device, self)
    # with:
    #   self.drafter = SiblingMTPProposer(vllm_config, device, self,
    #       siblings_dir="/home/ubuntu/models/mtp-siblings-trained")

This wraps EagleProposer and overrides propose() to use sibling heads.
The base class handles all the vLLM machinery (KV cache, attention backends,
padding, CUDAGraph, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight MTP head (standalone, no vLLM layer dependencies)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps) * self.weight.float()).to(x.dtype)


class SiblingHead(nn.Module):
    """One sibling MTP head.  Implements chain_forward (the recurrent path
    that skips full attention — used during speculative drafting).

    This is a pure-PyTorch implementation that loads from the safetensors
    checkpoint produced by mtp_clone.py / mtp_diversity_train.py.
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.pre_fc_norm_hidden = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_fc_norm_embedding = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Decoder layer
        # NOTE: Qwen 3.5 uses 2x head_dim for queries (512) vs kv (256)
        q_head_dim = head_dim * 2
        self.input_layernorm = _RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(hidden_size, eps=rms_norm_eps)
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
        self-projection where the softmax output is always 1.0.
        """
        # FC: concat normalized inputs and project
        embed_normed = self.pre_fc_norm_embedding(input_embeds)
        hidden_normed = self.pre_fc_norm_hidden(hidden_states)
        hidden = self.fc(torch.cat([embed_normed, hidden_normed], dim=-1))

        # Decoder layer: attention + MLP
        residual = hidden
        hidden = self.input_layernorm(hidden)

        # Self-attention (seq_len=1: attn = identity on values)
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
        for ckpt_key, param_key in mapping.items():
            if ckpt_key in state:
                own_state[param_key].copy_(state[ckpt_key])
        self.load_state_dict(own_state)


# ---------------------------------------------------------------------------
# Sibling ensemble container
# ---------------------------------------------------------------------------

class SiblingEnsemble(nn.Module):
    """K sibling MTP heads sharing embed_tokens and lm_head.

    At each draft position:
    1. All K heads receive the same (hidden_state, input_embed) pair
    2. Each head produces its own hidden output via chain_forward
    3. Each head's output is projected to logits via the shared lm_head
    4. Greedy argmax (or sampling) yields K candidate tokens per position
    """

    def __init__(
        self,
        siblings_dir: str | Path,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vocab_size: int,
        rms_norm_eps: float = 1e-6,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        siblings_dir = Path(siblings_dir)
        manifest_path = siblings_dir / "manifest.json"

        # Try training_manifest.json first (trained heads), fall back to manifest.json
        if (siblings_dir / "training_manifest.json").exists():
            with open(siblings_dir / "training_manifest.json") as f:
                manifest = json.load(f)
            K = manifest["num_heads"]
        elif manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            K = manifest["num_heads"]
        else:
            # Auto-discover from files
            files = sorted(siblings_dir.glob("mtp_sibling_*.safetensors"))
            K = len(files)
            if K == 0:
                raise FileNotFoundError(f"No mtp_sibling_*.safetensors in {siblings_dir}")

        self.K = K
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Load K sibling heads
        self.heads = nn.ModuleList()
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
            head.load_from_safetensors(filepath)
            head = head.to(dtype=dtype, device=device)
            head.eval()
            self.heads.append(head)

        logger.info(
            "SiblingEnsemble: loaded %d heads from %s (%.0f MB each at %s)",
            K, siblings_dir,
            sum(p.numel() * p.element_size() for p in self.heads[0].parameters()) / 1e6,
            dtype,
        )

    @torch.inference_mode()
    def draft_candidates(
        self,
        hidden_states: torch.Tensor,    # [B, D] from main model
        input_embeds: torch.Tensor,      # [B, D] embed_tokens(prev_token)
        lm_head_weight: torch.Tensor,    # [V, D] shared lm_head
        num_steps: int = 1,
        embed_fn: Optional[callable] = None,  # embed_tokens function
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run K sibling heads for num_steps chained draft positions.

        Returns:
            draft_token_ids: [B, num_steps, K] - K candidate token IDs per position
            draft_logits: [B, num_steps, K, V] - logits (optional, for scoring)
            draft_hidden_states: [B, num_steps, K, D] - hidden states per head per step

        The tree structure for verification:
            Position 0: K candidates from K heads
            Position 1: for each of the K accepted branches at pos 0,
                        K new candidates -> K^2 leaves (but we can prune)
            ...etc.

        For simplicity this implementation drafts a single step at a time
        and returns all K candidates.  Multi-step chaining creates a flat
        tree (K candidates per step, each step re-rooted on the main model's
        hidden state) rather than an exponential K^N tree.
        """
        B, D = hidden_states.shape
        K = self.K
        device = hidden_states.device

        all_token_ids = []   # list of [B, K] per step
        all_logits = []      # list of [B, K, V] per step
        all_hidden = []      # list of [B, K, D] per step

        current_hidden = hidden_states
        current_embeds = input_embeds

        for step in range(num_steps):
            step_ids = torch.empty(B, K, dtype=torch.long, device=device)
            step_logits = torch.empty(B, K, self.vocab_size, device=device,
                                       dtype=hidden_states.dtype)
            step_hidden = torch.empty(B, K, D, device=device, dtype=hidden_states.dtype)

            for k, head in enumerate(self.heads):
                h_out = head.chain_forward(current_embeds, current_hidden)
                logits = F.linear(h_out.float(), lm_head_weight.float())

                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    token_ids = logits.argmax(dim=-1)

                step_ids[:, k] = token_ids
                step_logits[:, k] = logits.to(hidden_states.dtype)
                step_hidden[:, k] = h_out

            all_token_ids.append(step_ids)
            all_logits.append(step_logits)
            all_hidden.append(step_hidden)

            # For multi-step: advance using the FIRST head's prediction as the
            # continuation token (head 0 is the exact clone, highest fidelity).
            # The other heads' predictions are alternative branches.
            if step < num_steps - 1 and embed_fn is not None:
                next_token = step_ids[:, 0]  # [B]
                current_embeds = embed_fn(next_token)  # [B, D]
                current_hidden = step_hidden[:, 0]     # [B, D] from head 0

        return (
            torch.stack(all_token_ids, dim=1),   # [B, num_steps, K]
            torch.stack(all_logits, dim=1),       # [B, num_steps, K, V]
            torch.stack(all_hidden, dim=1),       # [B, num_steps, K, D]
        )


# ---------------------------------------------------------------------------
# Tree structure utilities for vLLM tree attention verification
# ---------------------------------------------------------------------------

def build_sibling_tree(
    K: int,
    num_steps: int,
    max_tree_tokens: int = 64,
) -> dict:
    """Build a tree attention structure for K siblings over num_steps.

    The tree has a flat structure per step:
        Level 0 (root): 1 node (the verified token from main model)
        Level 1: K children (one per sibling head)
        Level 2: K children of the best Level-1 node (re-rooted)
        ...
        Level N: K children

    Total nodes = 1 + K * num_steps

    Returns a dict compatible with vLLM's TreeAttentionMetadata:
        - tree_token_ids: candidate token IDs in tree order
        - tree_parent_ids: parent index for each node (-1 for root)
        - tree_position_offsets: position offset from the root
    """
    total_nodes = 1 + K * num_steps
    if total_nodes > max_tree_tokens:
        # Truncate steps to fit budget
        num_steps = (max_tree_tokens - 1) // K
        total_nodes = 1 + K * num_steps

    parent_ids = [-1]  # root has no parent
    position_offsets = [0]  # root is at position 0

    prev_level_start = 0  # index of the "chosen" parent at each level

    for step in range(num_steps):
        parent_idx = prev_level_start  # all K siblings share the same parent
        for k in range(K):
            parent_ids.append(parent_idx)
            position_offsets.append(step + 1)
        # Next level's parent is the first sibling (head 0 = exact clone)
        prev_level_start = len(parent_ids) - K  # first sibling of this level

    return {
        "num_nodes": total_nodes,
        "num_steps": num_steps,
        "K": K,
        "parent_ids": parent_ids,
        "position_offsets": position_offsets,
    }


def format_tree_draft(
    draft_token_ids: torch.Tensor,  # [B, num_steps, K]
    tree_structure: dict,
) -> torch.Tensor:
    """Flatten the per-step sibling candidates into tree-order token IDs.

    Args:
        draft_token_ids: [B, num_steps, K] from SiblingEnsemble.draft_candidates
        tree_structure: from build_sibling_tree

    Returns:
        tree_token_ids: [B, num_tree_nodes] in the order expected by tree attention
                        (root token first, then K siblings per level)
    """
    B = draft_token_ids.shape[0]
    num_steps = tree_structure["num_steps"]
    K = tree_structure["K"]
    device = draft_token_ids.device

    # The root token is NOT included here — it's the verified token from the
    # main model, which vLLM handles separately.  We only return the draft tokens.
    # Total draft tokens = K * num_steps
    num_draft = K * num_steps
    tree_ids = torch.empty(B, num_draft, dtype=torch.long, device=device)

    for step in range(num_steps):
        start = step * K
        tree_ids[:, start:start + K] = draft_token_ids[:, step, :K]

    return tree_ids


# ---------------------------------------------------------------------------
# vLLM Proposer Integration
# ---------------------------------------------------------------------------

try:
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.sample.metadata import SamplingMetadata
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    class EagleProposer:  # type: ignore[no-redef]
        pass


if _VLLM_AVAILABLE:

    class SiblingMTPProposer(EagleProposer):
        """EagleProposer subclass that uses K sibling MTP heads for tree drafting.

        Drop-in replacement for EagleProposer.  The base class handles:
        - KV cache management
        - Attention backend setup
        - CUDAGraph integration
        - Padding and batching

        This subclass overrides propose() to:
        1. Run the standard first-pass through the base MTP model (inherits)
        2. Additionally run K-1 sibling heads on the same hidden states
        3. Combine into a tree structure for verification

        When tree attention is available, the K candidates per position are
        verified in a single forward pass.  When it's not, we fall back to
        linear chain drafting using only head 0 (the exact clone).
        """

        def __init__(
            self,
            vllm_config,
            device: torch.device,
            runner=None,
            siblings_dir: str = "/home/ubuntu/models/mtp-siblings-trained",
        ):
            super().__init__(vllm_config, device, runner)

            # Load model config for architecture params
            model_config = vllm_config.model_config
            config = model_config.hf_text_config
            text_config = getattr(config, "text_config", config)

            hidden_size = getattr(text_config, "hidden_size", 5120)
            intermediate_size = getattr(text_config, "intermediate_size", 17408)
            num_heads = getattr(text_config, "num_attention_heads", 24)
            num_kv_heads = getattr(text_config, "num_key_value_heads", 4)
            head_dim = getattr(text_config, "head_dim", 256)
            vocab_size = getattr(text_config, "vocab_size", 248320)
            rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-6)

            # Load sibling ensemble
            self._ensemble = SiblingEnsemble(
                siblings_dir=siblings_dir,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                vocab_size=vocab_size,
                rms_norm_eps=rms_norm_eps,
                device=str(device),
                dtype=torch.bfloat16,
            )

            self._K = self._ensemble.K
            self._tree = build_sibling_tree(
                K=self._K,
                num_steps=self.num_speculative_tokens,
            )

            logger.info(
                "SiblingMTPProposer: K=%d siblings, %d spec tokens, "
                "tree has %d nodes",
                self._K, self.num_speculative_tokens, self._tree["num_nodes"],
            )

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
            """Propose draft tokens using K sibling MTP heads.

            Strategy:
            1. Run the base EagleProposer.propose() to get head-0 draft
               (this handles all the vLLM infrastructure — KV cache, positions, etc.)
            2. Extract the hidden states from the base model's forward pass
            3. Run sibling heads 1..K-1 on the same hidden states
            4. Return combined candidates

            If tree attention is not available, falls back to linear drafting
            (head-0 only, same as standard MTP).
            """
            # Run the base proposer (this IS head 0's draft chain)
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

            # For sibling drafting, we need the hidden states that the base
            # model produced.  These are stored in self.hidden_states by the
            # base propose() call.  We run siblings on the FIRST position's
            # hidden state (the one corresponding to the next_token_ids).
            #
            # The siblings operate in chain_forward mode (no KV cache needed),
            # so we can run them independently of the base model's attention.

            # Get embed function from the base model
            if hasattr(self.model, 'embed_input_ids'):
                embed_fn = self.model.embed_input_ids
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_input_ids'):
                embed_fn = self.model.model.embed_input_ids
            else:
                # Can't access embeddings -> fall back to base draft only
                return base_draft_ids

            # Get lm_head weight
            if hasattr(self.model, 'lm_head'):
                lm_head = self.model.lm_head
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_head'):
                lm_head = self.model.model.lm_head
            else:
                return base_draft_ids

            if hasattr(lm_head, 'weight'):
                lm_weight = lm_head.weight
            else:
                return base_draft_ids

            # Extract hidden states for the first draft position
            # target_hidden_states are from the MAIN model's last layer
            # We use the hidden states at the token_indices_to_sample positions
            first_hidden = target_hidden_states[
                common_attn_metadata.query_start_loc[:-1]
                if hasattr(common_attn_metadata, 'query_start_loc')
                else torch.arange(batch_size, device=target_hidden_states.device)
            ]
            if first_hidden.shape[0] > batch_size:
                first_hidden = first_hidden[:batch_size]

            # Run sibling heads 1..K-1 for each draft position
            # Build the sibling candidates alongside the base draft
            sibling_ids = torch.zeros(
                batch_size, num_spec, self._K,
                dtype=torch.long, device=base_draft_ids.device,
            )
            sibling_ids[:, :, 0] = base_draft_ids  # Head 0 = base model's draft

            # Chain through positions using head 0's tokens as the backbone
            h = first_hidden[:batch_size]  # [B, D]
            for step in range(min(num_spec, self._tree["num_steps"])):
                if step == 0:
                    prev_token = next_token_ids[:batch_size]
                else:
                    prev_token = base_draft_ids[:, step - 1]

                with torch.no_grad():
                    embeds = embed_fn(prev_token)  # [B, D]

                # Run each sibling head (skip head 0, already done by base)
                for k in range(1, self._K):
                    head = self._ensemble.heads[k]
                    h_out = head.chain_forward(embeds, h)
                    logits = F.linear(h_out.float(), lm_weight.float())
                    sibling_ids[:, step, k] = logits.argmax(dim=-1)

                # Advance hidden state using head 0's output for next step
                h_out_0 = self._ensemble.heads[0].chain_forward(embeds, h)
                h = h_out_0

            # If tree attention is available, return the tree-formatted candidates
            # Otherwise return just the base draft (linear chain)
            #
            # TODO: When vLLM's tree attention supports custom tree structures,
            # return format_tree_draft(sibling_ids, self._tree) here.
            # For now, we return the base draft IDs and log the sibling alternatives
            # for analysis.
            self._last_sibling_ids = sibling_ids  # stash for analysis/debugging

            # Count diversity: how many unique tokens per position
            for step in range(min(3, num_spec)):
                unique_per_batch = []
                for b in range(min(4, batch_size)):
                    n_unique = len(set(sibling_ids[b, step].tolist()))
                    unique_per_batch.append(n_unique)
                if step == 0:
                    logger.debug(
                        "SiblingMTP pos %d: %s unique tokens (K=%d) for first 4 requests",
                        step, unique_per_batch, self._K,
                    )

            return base_draft_ids

        def get_sibling_candidates(self) -> Optional[torch.Tensor]:
            """Return the last set of sibling candidates for debugging/analysis.

            Returns [B, num_steps, K] tensor or None if no drafting has occurred.
            """
            return getattr(self, '_last_sibling_ids', None)


# ---------------------------------------------------------------------------
# Standalone test / benchmark
# ---------------------------------------------------------------------------

def benchmark_sibling_heads(
    siblings_dir: str = "/home/ubuntu/models/mtp-siblings",
    device: str = "cuda",
    batch_size: int = 8,
    num_steps: int = 7,
    warmup: int = 10,
    iters: int = 100,
):
    """Benchmark sibling head inference latency."""
    import time

    # Model config for Qwen 3.5-27B
    hidden_size = 5120
    intermediate_size = 17408
    num_heads = 24
    num_kv_heads = 4
    head_dim = 256
    vocab_size = 248320

    print(f"Loading sibling ensemble from {siblings_dir}...")
    ensemble = SiblingEnsemble(
        siblings_dir=siblings_dir,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        device=device,
        dtype=torch.bfloat16,
    )

    # Create dummy inputs
    hidden = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    embeds = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    lm_weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)

    # Dummy embed function
    embed_table = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)
    def embed_fn(ids):
        return embed_table[ids]

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    for _ in range(warmup):
        ensemble.draft_candidates(hidden, embeds, lm_weight, num_steps=num_steps, embed_fn=embed_fn)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iters} iters, batch={batch_size}, steps={num_steps})...")
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(iters):
        ensemble.draft_candidates(hidden, embeds, lm_weight, num_steps=num_steps, embed_fn=embed_fn)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    ms_per_call = elapsed / iters * 1000
    K = ensemble.K
    tokens_per_call = batch_size * num_steps * K

    print(f"\nResults:")
    print(f"  K={K} heads, {num_steps} steps, batch={batch_size}")
    print(f"  {ms_per_call:.2f} ms / draft call")
    print(f"  {tokens_per_call / (ms_per_call / 1000):.0f} candidate tokens/sec")
    print(f"  {ms_per_call / num_steps:.2f} ms / step (K heads parallel)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        siblings_dir = sys.argv[1]
    else:
        siblings_dir = "/home/ubuntu/models/mtp-siblings"

    if torch.cuda.is_available():
        benchmark_sibling_heads(siblings_dir=siblings_dir)
    else:
        print("No CUDA available. SiblingEnsemble requires GPU for benchmarking.")
        print("Building tree structure test...")
        tree = build_sibling_tree(K=3, num_steps=7)
        print(f"Tree: {tree['num_nodes']} nodes, {tree['num_steps']} steps")
        print(f"Parent IDs: {tree['parent_ids']}")
        print(f"Position offsets: {tree['position_offsets']}")
