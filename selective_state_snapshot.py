"""Selective DeltaNet state snapshot for modal_mtp draft mode.

Problem
-------
modal_mtp runs draft forwards through the target model with attention
layers skipped (identity).  The DeltaNet (GDN) recurrent layers still
execute and **write** to the shared recurrent state cache
(``layer.kv_cache``).  After N draft steps the state for every active
slot is corrupted relative to the verify-pass baseline.  The verify
pass must start from the pre-draft state for correctness.

Full-cache snapshot is prohibitively expensive: 256 slots x 48 layers
x (conv + temporal) ~ 18.7 GB (bfloat16).  But only the *active batch*
slots are touched by draft forwards.  A selective snapshot copies only
those slots, bringing cost down to O(batch_size) instead of O(max_slots).

State geometry (Qwen3.5-27B, TP=1)
-----------------------------------
Each GatedDeltaNetAttention layer stores two tensors in ``layer.kv_cache``:

  kv_cache[0]  conv_state    shape: (num_slots, 3, 10240)   dtype: bfloat16
               Transposed to (num_slots, 10240, 3) during forward; stored
               as (num_slots, conv_kernel-1, conv_dim) in the cache.
               conv_dim = key_dim*2 + value_dim
                        = 128*16*2 + 128*48 = 10240.

  kv_cache[1]  ssm_state     shape: (num_slots, 48, 128, 128) dtype: bfloat16
               Temporal (recurrent) state.
               (num_slots, num_v_heads/tp, head_v_dim, head_k_dim)

Per-slot per-layer: conv = 60 KB, temporal = 1.5 MB  => ~1.56 MB/slot/layer.
48 GDN layers => ~74.8 MB per slot.

Selective snapshot for 8 active slots: ~598 MB.
Selective snapshot for 32 active slots: ~2.3 GB.
Full snapshot for 256 slots: ~18.7 GB.

Usage
-----
Integrate into ModalMTPProposer.propose():

    from selective_state_snapshot import (
        selective_snapshot,
        selective_restore,
    )

    # Before draft loop
    gdn_layers = get_gdn_layers(inner_model)
    snapshot = selective_snapshot(gdn_layers, active_slot_ids)

    # ... draft forwards ...

    # After draft loop (in finally block)
    selective_restore(gdn_layers, snapshot, active_slot_ids)
"""

from __future__ import annotations

import torch
from torch import nn


# -------------------------------------------------------------------
# Layer discovery
# -------------------------------------------------------------------

def get_gdn_layers(model: nn.Module) -> list:
    """Return all GatedDeltaNetAttention layers from the model.

    Works with both Qwen3_5Model and Qwen3NextModel.  Layers are
    returned in module-traversal order (which matches layer index
    order).
    """
    # Import here to avoid hard dependency when used standalone.
    try:
        from vllm.model_executor.layers.mamba.gdn_linear_attn import (
            GatedDeltaNetAttention,
        )
    except ImportError:
        GatedDeltaNetAttention = None

    layers = []
    for mod in model.modules():
        if GatedDeltaNetAttention is not None and isinstance(
            mod, GatedDeltaNetAttention
        ):
            layers.append(mod)
    return layers


# -------------------------------------------------------------------
# Snapshot / Restore
# -------------------------------------------------------------------

SnapshotData = list[tuple[torch.Tensor, torch.Tensor]]
"""Per-layer list of (conv_state_slice, ssm_state_slice) clones."""


def selective_snapshot(
    gdn_layers: list,
    active_slot_ids: torch.Tensor,
) -> SnapshotData:
    """Snapshot only the active slots' DeltaNet state across all layers.

    Parameters
    ----------
    gdn_layers:
        List of GatedDeltaNetAttention modules (from get_gdn_layers).
    active_slot_ids:
        1-D integer tensor of slot indices that will be mutated by
        draft forwards.  Shape: (batch_size,).  These are the cache
        slot indices from the block table (``block_table[:, 0]``).

    Returns
    -------
    List of (conv_clone, ssm_clone) tensor pairs, one per layer.
    Clones are contiguous and detached.
    """
    snapshot: SnapshotData = []
    idx = active_slot_ids  # [B]

    for layer in gdn_layers:
        conv_state = layer.kv_cache[0]   # (num_slots, conv_kernel-1, conv_dim)
        ssm_state = layer.kv_cache[1]    # (num_slots, num_v_heads, head_v_dim, head_k_dim)

        # index_select on dim 0 copies only the active slots.
        conv_clone = conv_state.index_select(0, idx).clone()
        ssm_clone = ssm_state.index_select(0, idx).clone()

        snapshot.append((conv_clone, ssm_clone))

    return snapshot


def selective_restore(
    gdn_layers: list,
    snapshot: SnapshotData,
    active_slot_ids: torch.Tensor,
) -> None:
    """Restore snapshotted state back into the active slots.

    Parameters
    ----------
    gdn_layers:
        Same list passed to selective_snapshot.
    snapshot:
        Return value from selective_snapshot.
    active_slot_ids:
        Same slot IDs tensor used for the snapshot.
    """
    idx = active_slot_ids  # [B]

    for (conv_clone, ssm_clone), layer in zip(snapshot, gdn_layers):
        conv_state = layer.kv_cache[0]
        ssm_state = layer.kv_cache[1]

        # Scatter the saved values back into the cache.
        # index_copy_ writes snapshot rows back to their original slots.
        conv_state.index_copy_(0, idx, conv_clone)
        ssm_state.index_copy_(0, idx, ssm_clone)


# -------------------------------------------------------------------
# Memory cost calculation
# -------------------------------------------------------------------

def estimate_snapshot_bytes(
    num_active_slots: int,
    num_gdn_layers: int = 48,
    num_v_heads: int = 48,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    num_k_heads: int = 16,
    conv_kernel_size: int = 4,
    tp_size: int = 1,
    dtype_bytes: int = 2,  # bfloat16
) -> dict[str, int | float]:
    """Calculate memory cost for a selective snapshot.

    Returns a dict with per-slot, per-layer, and total breakdowns.
    """
    conv_dim = (head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads) // tp_size
    conv_width = conv_kernel_size - 1  # stored columns

    conv_elements_per_slot = conv_width * conv_dim
    ssm_elements_per_slot = (
        (num_v_heads // tp_size) * head_v_dim * head_k_dim
    )

    conv_bytes_per_slot_per_layer = conv_elements_per_slot * dtype_bytes
    ssm_bytes_per_slot_per_layer = ssm_elements_per_slot * dtype_bytes
    total_per_slot_per_layer = conv_bytes_per_slot_per_layer + ssm_bytes_per_slot_per_layer

    total_per_slot = total_per_slot_per_layer * num_gdn_layers
    total = total_per_slot * num_active_slots

    return {
        "conv_bytes_per_slot_per_layer": conv_bytes_per_slot_per_layer,
        "ssm_bytes_per_slot_per_layer": ssm_bytes_per_slot_per_layer,
        "total_bytes_per_slot_per_layer": total_per_slot_per_layer,
        "total_bytes_per_slot": total_per_slot,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
        "num_active_slots": num_active_slots,
        "num_gdn_layers": num_gdn_layers,
    }


# -------------------------------------------------------------------
# Integration patch for ModalMTPProposer.propose()
# -------------------------------------------------------------------

def get_active_slot_ids(
    common_attn_metadata,
    batch_size: int,
    runner=None,
    gdn_kv_cache_spec=None,
) -> torch.Tensor:
    """Extract the DeltaNet cache slot indices for the active batch.

    These are the mamba-transformed block_table[:, 0] values — the same
    indices used by GDNAttentionMetadata.non_spec_state_indices_tensor
    in the draft forward path.

    Parameters
    ----------
    common_attn_metadata:
        The CommonAttentionMetadata from the verify pass.
    batch_size:
        Number of active requests in the batch.
    runner:
        GPUModelRunner (optional, for mamba_cache_mode).
    gdn_kv_cache_spec:
        KV cache spec for GDN layers (optional, for block table transform).

    Returns
    -------
    1-D int tensor of shape (batch_size,) with slot indices.
    """
    block_table = common_attn_metadata.block_table_tensor

    if gdn_kv_cache_spec is not None:
        from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
        cache_mode = (
            runner.cache_config.mamba_cache_mode
            if runner and hasattr(runner, "cache_config")
            else "align"
        )
        block_table = mamba_get_block_table_tensor(
            block_table,
            common_attn_metadata.seq_lens,
            gdn_kv_cache_spec,
            cache_mode,
        )

    return block_table[:batch_size, 0]


# -------------------------------------------------------------------
# Main: print memory estimates
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("DeltaNet Selective State Snapshot — Memory Estimates")
    print("=" * 60)
    print()
    print("Model: Qwen3.5-27B (48 GDN layers out of 64 total)")
    print("Config: linear_key_head_dim=128, linear_num_key_heads=16,")
    print("        linear_num_value_heads=48, linear_value_head_dim=128")
    print("        conv_kernel_size=4, dtype=bfloat16")
    print()

    # State shapes
    conv_dim = 128 * 16 * 2 + 128 * 48
    print(f"conv_state per slot per layer:  (3, {conv_dim})")
    print(f"  = {3 * conv_dim:,} elements = {3 * conv_dim * 2:,} bytes = {3 * conv_dim * 2 / 1024:.1f} KB")
    print()
    ssm_shape = (48, 128, 128)
    ssm_elements = 48 * 128 * 128
    print(f"ssm_state per slot per layer:   {ssm_shape}")
    print(f"  = {ssm_elements:,} elements = {ssm_elements * 2:,} bytes = {ssm_elements * 2 / (1024**2):.2f} MB")
    print()

    for num_slots in [1, 8, 16, 32, 64, 128, 256]:
        info = estimate_snapshot_bytes(num_slots)
        label = f"{num_slots:>3} slots"
        if info["total_gb"] >= 1.0:
            print(f"  {label}: {info['total_gb']:.2f} GB")
        else:
            print(f"  {label}: {info['total_mb']:.1f} MB")

    print()
    print("Conclusion: selective snapshot for typical batch sizes (8-32)")
    print("costs 0.6-2.3 GB vs 18.7 GB for full 256-slot snapshot.")
    print("This is the difference between viable and prohibitive.")
    print()
    print("Integration point: ModalMTPProposer.propose() in")
    print("vllm/v1/spec_decode/modal_mtp.py, lines 288-292.")
    print("Replace `deltanet_snapshot = None` with:")
    print()
    print("    gdn_layers = get_gdn_layers(self._inner_model)")
    print("    active_slots = get_active_slot_ids(")
    print("        common_attn_metadata, batch_size,")
    print("        self._runner, self._gdn_kv_cache_spec)")
    print("    deltanet_snapshot = selective_snapshot(gdn_layers, active_slots)")
    print()
    print("And in the finally block (line 401-402):")
    print()
    print("    if deltanet_snapshot is not None:")
    print("        selective_restore(gdn_layers, deltanet_snapshot, active_slots)")
