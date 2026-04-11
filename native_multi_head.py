"""NativeMultiHeadProposer — K sibling MTP heads via weight-swapping.

Runs K different MTP head weight sets through the SAME EagleProposer
path (compiled, CUDA-graphed). Each propose() call swaps weights
in-place (memcpy to same tensor addresses), runs the standard draft
loop, swaps back. Returns the chain from the best-performing head
based on tracked acceptance rates.

Weight swap cost: ~1ms per head on GH200 (849MB at 900GB/s).
This is negligible vs the ~5ms verify forward pass.

Key detail: vLLM fuses q/k/v into qkv_proj and gate/up into
gate_up_proj at load time. Safetensors have unfused weights.
We pre-fuse at bank load time so swap is just copy_().
"""

import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)

# Env config
_SIBLING_HEADS_DIR = os.environ.get(
    "SIBLING_MTP_HEADS_DIR",
    "/home/ubuntu/models/sibling-mtp-heads-gpu",
)
_NUM_SIBLING_HEADS = int(os.environ.get("NUM_SIBLING_HEADS", "0"))
_HEAD_SELECTION = os.environ.get("SIBLING_HEAD_SELECTION", "round_robin")
# EMA decay for acceptance rate tracking (lower = more responsive)
_EMA_ALPHA = float(os.environ.get("SIBLING_EMA_ALPHA", "0.1"))


def _remap_safetensor_key(key: str) -> str:
    """Map safetensors key (mtp.X) to model parameter key (model.X).

    The Qwen3_5MTP.load_weights remaps mtp.* -> model.* so that's
    how the parameters appear in the live nn.Module.
    """
    if key.startswith("mtp."):
        return "model." + key[4:]
    return key


# Stacked/fused parameter mappings used by vLLM's Qwen3.5 models.
# (fused_name, component_name, shard_id)
_STACKED_PARAMS = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]


def _fuse_weights_to_match_model(
    raw_weights: dict[str, torch.Tensor],
    model_params: dict[str, nn.Parameter],
) -> dict[str, torch.Tensor]:
    """Fuse raw safetensor weights to match vLLM's runtime parameter layout.

    vLLM fuses q_proj + k_proj + v_proj -> qkv_proj.weight and
    gate_proj + up_proj -> gate_up_proj.weight at model load time.
    The safetensors have the unfused names. This function produces
    tensors whose shapes exactly match the live model parameters.

    For qkv_proj: the fused weight is [q; k; v] concatenated along dim 0.
    For gate_up_proj: the fused weight is [gate; up] concatenated along dim 0.

    Non-fused weights (fc, norms, down_proj, o_proj) pass through directly.
    """
    fused: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    # Group raw weights by their layer prefix to find fuseable sets.
    # E.g., model.layers.0.self_attn.q_proj.weight ->
    #   prefix=model.layers.0.self_attn, component=q_proj, suffix=weight
    fuse_groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name, tensor in raw_weights.items():
        for fused_name, component_name, _shard_id in _STACKED_PARAMS:
            if f".{component_name}." in name:
                # Extract the prefix (everything before the component name)
                prefix = name.split(f".{component_name}.")[0]
                suffix = name.split(f".{component_name}.")[1]
                group_key = f"{prefix}.{fused_name}.{suffix}"
                fuse_groups[group_key][component_name] = tensor
                consumed.add(name)
                break

    # Build fused tensors
    for fused_param_name, components in fuse_groups.items():
        if "qkv_proj" in fused_param_name:
            # Must have all three: q, k, v
            if all(k in components for k in ("q_proj", "k_proj", "v_proj")):
                fused_tensor = torch.cat(
                    [components["q_proj"], components["k_proj"], components["v_proj"]],
                    dim=0,
                )
                # Verify shape matches model parameter
                if fused_param_name in model_params:
                    expected_shape = model_params[fused_param_name].data.shape
                    if fused_tensor.shape != expected_shape:
                        logger.warning(
                            "Fused qkv_proj shape mismatch for %s: "
                            "got %s, expected %s. Using raw cat.",
                            fused_param_name, fused_tensor.shape, expected_shape,
                        )
                fused[fused_param_name] = fused_tensor
            else:
                logger.warning(
                    "Incomplete qkv_proj group for %s: have %s",
                    fused_param_name, list(components.keys()),
                )
        elif "gate_up_proj" in fused_param_name:
            if all(k in components for k in ("gate_proj", "up_proj")):
                fused_tensor = torch.cat(
                    [components["gate_proj"], components["up_proj"]],
                    dim=0,
                )
                if fused_param_name in model_params:
                    expected_shape = model_params[fused_param_name].data.shape
                    if fused_tensor.shape != expected_shape:
                        logger.warning(
                            "Fused gate_up_proj shape mismatch for %s: "
                            "got %s, expected %s. Using raw cat.",
                            fused_param_name, fused_tensor.shape, expected_shape,
                        )
                fused[fused_param_name] = fused_tensor
            else:
                logger.warning(
                    "Incomplete gate_up_proj group for %s: have %s",
                    fused_param_name, list(components.keys()),
                )

    # Pass through non-fused weights unchanged
    for name, tensor in raw_weights.items():
        if name not in consumed:
            fused[name] = tensor

    return fused


class SiblingWeightBank:
    """Pre-loaded weight tensors for K sibling MTP heads.

    All tensors are stored on the same GPU as the model, pre-fused to
    match vLLM's runtime parameter layout (qkv_proj, gate_up_proj).
    """

    def __init__(
        self,
        heads_dir: str,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
        model_params: dict[str, nn.Parameter] | None = None,
    ):
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.head_weights: list[dict[str, torch.Tensor]] = []

        t0 = time.monotonic()
        for i in range(num_heads):
            path = Path(heads_dir) / f"mtp_sibling_{i}.safetensors"
            if not path.exists():
                raise FileNotFoundError(
                    f"Sibling head weights not found: {path}"
                )
            raw_weights = {}
            with safe_open(str(path), framework="pt", device=str(device)) as f:
                for key in f.keys():
                    param_name = _remap_safetensor_key(key)
                    tensor = f.get_tensor(key)
                    if tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    raw_weights[param_name] = tensor

            # Fuse to match runtime parameter layout
            if model_params is not None:
                weights = _fuse_weights_to_match_model(raw_weights, model_params)
            else:
                weights = raw_weights

            self.head_weights.append(weights)

        elapsed = time.monotonic() - t0
        total_mb = sum(
            sum(t.numel() * t.element_size() for t in hw.values())
            for hw in self.head_weights
        ) / (1024 ** 2)
        logger.info(
            "Loaded %d sibling head weight sets (%.0f MB total) in %.2fs",
            num_heads, total_mb, elapsed,
        )

    def get_param_names(self) -> set[str]:
        """Return the set of parameter names covered by sibling heads."""
        if not self.head_weights:
            return set()
        return set(self.head_weights[0].keys())


class NativeMultiHeadProposer(EagleProposer):
    """EagleProposer that runs K sibling MTP heads via weight swapping.

    On each propose() call:
    1. Save the original (head-0) weights
    2. For each sibling head K:
       a. Swap weights in-place (copy_ to same tensor addresses)
       b. Call super().propose() — runs through CUDA graphs
       c. Collect draft tokens
    3. Restore original weights
    4. Return the best chain based on acceptance rate history

    CUDA graph compatibility: copy_() writes to the same memory
    addresses. The CUDA graph replays see the updated data without
    needing recapture.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)

        self.num_sibling_heads = _NUM_SIBLING_HEADS
        self.head_selection = _HEAD_SELECTION

        # Per-head acceptance rate tracking (EMA)
        # Index 0 = original head, 1..K = siblings
        self._acceptance_ema: dict[int, float] = defaultdict(lambda: 0.5)
        self._ema_alpha = _EMA_ALPHA
        self._total_proposals = 0
        self._round_robin_idx = 0

        # Log interval for stats
        self._log_interval = int(os.environ.get("SIBLING_LOG_INTERVAL", "100"))

        logger.info(
            "NativeMultiHeadProposer: %d sibling heads, selection=%s",
            self.num_sibling_heads, self.head_selection,
        )

    def load_model(self, target_model: nn.Module) -> None:
        """Load the base model, then load sibling weight banks."""
        super().load_model(target_model)

        if self.num_sibling_heads <= 0:
            logger.warning(
                "NativeMultiHeadProposer: NUM_SIBLING_HEADS=0, "
                "falling back to single-head mode"
            )
            self._weight_bank = None
            self._original_weights: dict[str, torch.Tensor] = {}
            self._swappable_params: dict[str, nn.Parameter] = {}
            return

        # Get model params for shape-aware fusion
        model_params = dict(self.model.named_parameters())

        # Load sibling weights with fusion
        self._weight_bank = SiblingWeightBank(
            heads_dir=_SIBLING_HEADS_DIR,
            num_heads=self.num_sibling_heads,
            device=self.device,
            dtype=self.dtype,
            model_params=model_params,
        )

        # Identify which model parameters correspond to sibling weights.
        # We need the actual nn.Parameter references for in-place copy.
        sibling_param_names = self._weight_bank.get_param_names()

        self._swappable_params: dict[str, nn.Parameter] = {}
        matched = 0
        unmatched = []
        for name in sibling_param_names:
            if name in model_params:
                # Verify shapes match
                sibling_shape = self._weight_bank.head_weights[0][name].shape
                model_shape = model_params[name].data.shape
                if sibling_shape != model_shape:
                    logger.error(
                        "Shape mismatch for '%s': sibling=%s, model=%s. "
                        "Skipping this parameter.",
                        name, sibling_shape, model_shape,
                    )
                    continue
                self._swappable_params[name] = model_params[name]
                matched += 1
            else:
                unmatched.append(name)

        if unmatched:
            logger.warning(
                "Sibling weight keys not found in model params (%d/%d): %s",
                len(unmatched), len(sibling_param_names), unmatched[:5],
            )

        logger.info(
            "Matched %d/%d sibling weight keys to model parameters",
            matched, len(sibling_param_names),
        )

        # Snapshot the original (head-0) weights so we can restore them.
        # These are clones stored on GPU — same cost as one sibling set.
        self._original_weights: dict[str, torch.Tensor] = {}
        for name, param in self._swappable_params.items():
            self._original_weights[name] = param.data.clone()

        total_swappable_mb = sum(
            p.data.numel() * p.data.element_size()
            for p in self._swappable_params.values()
        ) / (1024 ** 2)
        logger.info(
            "Swappable parameter footprint: %.0f MB (%d tensors)",
            total_swappable_mb, len(self._swappable_params),
        )

    def _swap_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """In-place copy sibling weights into model parameters.

        Uses copy_() which writes to the SAME memory addresses,
        preserving CUDA graph compatibility.
        """
        for name, param in self._swappable_params.items():
            if name in weights:
                param.data.copy_(weights[name], non_blocking=True)

    def _restore_original_weights(self) -> None:
        """Restore the original (head-0) weights."""
        self._swap_weights(self._original_weights)

    def _select_head_order(self) -> list[int]:
        """Determine which heads to run and in what order.

        Returns list of head indices (0 = original, 1+ = siblings).
        """
        total_heads = 1 + self.num_sibling_heads  # original + siblings

        if self.head_selection == "round_robin":
            # Run only one head per call, rotating
            idx = self._round_robin_idx % total_heads
            self._round_robin_idx += 1
            return [idx]

        elif self.head_selection == "best":
            # Run only the head with highest acceptance EMA
            best_idx = max(range(total_heads), key=lambda i: self._acceptance_ema[i])
            return [best_idx]

        elif self.head_selection == "all":
            # Run all heads, return best result
            return sorted(
                range(total_heads),
                key=lambda i: self._acceptance_ema[i],
                reverse=True,
            )

        elif self.head_selection == "top2":
            # Run top-2 heads by acceptance rate
            ranked = sorted(
                range(total_heads),
                key=lambda i: self._acceptance_ema[i],
                reverse=True,
            )
            return ranked[:2]

        else:
            logger.warning(
                "Unknown head selection '%s', using round_robin",
                self.head_selection,
            )
            return [0]

    def _get_sibling_weights(self, head_idx: int) -> dict[str, torch.Tensor] | None:
        """Get weight dict for a head index. 0 = original, 1+ = sibling."""
        if head_idx == 0:
            return self._original_weights
        sibling_idx = head_idx - 1
        if self._weight_bank is None or sibling_idx >= self.num_sibling_heads:
            return None
        return self._weight_bank.head_weights[sibling_idx]

    def update_acceptance_rate(self, head_idx: int, accepted: int, total: int) -> None:
        """Update EMA acceptance rate for a head."""
        if total == 0:
            return
        rate = accepted / total
        old = self._acceptance_ema[head_idx]
        self._acceptance_ema[head_idx] = (
            self._ema_alpha * rate + (1 - self._ema_alpha) * old
        )

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        # Fast path: no siblings loaded
        if not self._weight_bank or self.num_sibling_heads <= 0:
            return super().propose(
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

        head_order = self._select_head_order()
        self._total_proposals += 1

        if len(head_order) == 1:
            # Single head — swap, run, restore
            head_idx = head_order[0]
            needs_swap = head_idx != 0
            if needs_swap:
                weights = self._get_sibling_weights(head_idx)
                if weights is not None:
                    self._swap_weights(weights)
                    # Sync to ensure weights are visible before CUDA graph replay
                    torch.cuda.current_stream().synchronize()

            try:
                result = super().propose(
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
                if needs_swap:
                    self._restore_original_weights()
                    torch.cuda.current_stream().synchronize()

            self._maybe_log_stats()
            return result

        # Multi-head: run each, pick best by acceptance EMA.
        # IMPORTANT: propose() mutates common_attn_metadata in place.
        # For multi-head mode we must snapshot and restore between runs.
        # This is the expensive path — prefer round_robin or best.
        saved_seq_lens = common_attn_metadata.seq_lens.clone()
        saved_max_seq_len = common_attn_metadata.max_seq_len
        saved_max_query_len = common_attn_metadata.max_query_len
        saved_num_actual_tokens = common_attn_metadata.num_actual_tokens
        saved_slot_mapping = common_attn_metadata.slot_mapping.clone()
        saved_query_start_loc = common_attn_metadata.query_start_loc.clone()

        # Snapshot CPU-side shadows if they exist
        saved_seq_lens_cpu = (
            common_attn_metadata._seq_lens_cpu.clone()
            if common_attn_metadata._seq_lens_cpu is not None
            else None
        )
        saved_num_computed_cpu = (
            common_attn_metadata._num_computed_tokens_cpu.clone()
            if common_attn_metadata._num_computed_tokens_cpu is not None
            else None
        )

        best_result = None
        best_head_idx = head_order[0]
        best_score = -1.0

        for head_idx in head_order:
            # Restore metadata state before each head run (except first)
            if best_result is not None:
                common_attn_metadata.seq_lens.copy_(saved_seq_lens)
                common_attn_metadata.max_seq_len = saved_max_seq_len
                common_attn_metadata.max_query_len = saved_max_query_len
                common_attn_metadata.num_actual_tokens = saved_num_actual_tokens
                common_attn_metadata.slot_mapping.copy_(saved_slot_mapping)
                common_attn_metadata.query_start_loc.copy_(saved_query_start_loc)
                common_attn_metadata._seq_lens_cpu = saved_seq_lens_cpu
                common_attn_metadata._num_computed_tokens_cpu = saved_num_computed_cpu

            needs_swap = head_idx != 0
            if needs_swap:
                weights = self._get_sibling_weights(head_idx)
                if weights is not None:
                    self._swap_weights(weights)
                    torch.cuda.current_stream().synchronize()

            try:
                result = super().propose(
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
                if needs_swap:
                    self._restore_original_weights()
                    torch.cuda.current_stream().synchronize()

            score = self._acceptance_ema[head_idx]
            if best_result is None or score > best_score:
                best_result = result.clone()
                best_head_idx = head_idx
                best_score = score

        self._maybe_log_stats()
        return best_result

    def _maybe_log_stats(self) -> None:
        """Periodically log acceptance rate stats."""
        if self._total_proposals % self._log_interval != 0:
            return
        total_heads = 1 + self.num_sibling_heads
        rates = {
            f"head_{i}": f"{self._acceptance_ema[i]:.3f}"
            for i in range(total_heads)
        }
        logger.info(
            "Sibling head stats (proposal %d): acceptance_ema=%s, selection=%s",
            self._total_proposals, rates, self.head_selection,
        )

    def get_head_stats(self) -> dict:
        """Return current head performance stats for external monitoring."""
        total_heads = 1 + self.num_sibling_heads
        return {
            "total_proposals": self._total_proposals,
            "selection_mode": self.head_selection,
            "acceptance_ema": {
                i: self._acceptance_ema[i] for i in range(total_heads)
            },
        }
