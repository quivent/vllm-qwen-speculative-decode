"""DeltaNet hidden-state adjuster for MTP chain refinement.

After the MTP head produces a draft token and updated hidden state,
the hidden state has diverged from what the full 64-layer model would
produce -- the MTP head is only a single decoder layer.  Running a few
DeltaNet (GatedDeltaNet) layers on the hidden state between chain
steps corrects the representation, improving acceptance at deeper
draft positions (3-6) where the unadjusted MTP chain degrades.

CRITICAL DESIGN DECISION: run DeltaNet attention-only (skip MLP).
The MLP dominates both DeltaNet and full-attention layers (~92% of
FLOPs).  By running only the attention mechanism + output projection,
each adjustment layer costs ~37M FLOPs vs ~453M for a full MTP step.
Four attention-only DeltaNet layers = ~33% of one MTP step.

Architecture
------------
::

    MTP_head -> token_0 -> DeltaNet_adjust(hidden) -> MTP_head -> token_1 -> ...

Where DeltaNet_adjust runs the linear-attention mechanism (input
projection, conv1d, recurrent state update, output projection)
WITHOUT the MLP.

Usage
-----
::

    adjuster = DeltaNetAdjuster.from_model(qwen3_5_model, num_layers=4)
    # inside the chain loop:
    hidden = adjuster.adjust(hidden)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch import nn


@dataclasses.dataclass
class CostEstimate:
    """FLOP estimates for a single token through different components."""

    # Per-layer costs (FLOPs per token)
    deltanet_attn_only_flops: int   # DeltaNet attention (no MLP)
    deltanet_full_layer_flops: int  # DeltaNet attention + MLP
    mtp_decoder_layer_flops: int
    mlp_flops: int                  # MLP alone (dominates both layer types)

    # Aggregate costs
    num_deltanet_layers: int
    adjustment_total_flops: int     # attn-only
    mtp_step_flops: int

    # Ratios
    adjustment_vs_mtp: float       # adjustment_total / mtp_step
    adjustment_vs_full_fwd: float  # adjustment_total / full_model_forward
    full_model_forward_flops: int

    def __repr__(self) -> str:
        def _fmt(n: int) -> str:
            if n >= 1e12:
                return f"{n/1e12:.2f}T"
            if n >= 1e9:
                return f"{n/1e9:.2f}G"
            if n >= 1e6:
                return f"{n/1e6:.2f}M"
            if n >= 1e3:
                return f"{n/1e3:.1f}K"
            return str(n)

        lines = [
            "CostEstimate:",
            f"  DeltaNet attn-only : {_fmt(self.deltanet_attn_only_flops)} FLOPs/token",
            f"  DeltaNet full layer: {_fmt(self.deltanet_full_layer_flops)} FLOPs/token",
            f"  MLP alone          : {_fmt(self.mlp_flops)} FLOPs/token (dominates both)",
            f"  MTP decoder layer  : {_fmt(self.mtp_decoder_layer_flops)} FLOPs/token",
            f"  ---",
            f"  Adjustment ({self.num_deltanet_layers} DeltaNet attn-only): "
            f"{_fmt(self.adjustment_total_flops)} FLOPs/token",
            f"  Single MTP step    : {_fmt(self.mtp_step_flops)} FLOPs/token",
            f"  Full model forward : {_fmt(self.full_model_forward_flops)} FLOPs/token",
            f"  ---",
            f"  Adjustment / MTP step    : {self.adjustment_vs_mtp:.1%}",
            f"  Adjustment / full forward: {self.adjustment_vs_full_fwd:.1%}",
        ]
        return "\n".join(lines)


def estimate_cost(
    hidden_size: int = 4096,
    intermediate_size: int = 12288,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 4,
    head_dim: int = 256,
    linear_num_key_heads: int = 16,
    linear_num_value_heads: int = 32,
    linear_key_head_dim: int = 128,
    linear_value_head_dim: int = 128,
    num_deltanet_layers: int = 4,
    num_total_layers: int = 64,
) -> CostEstimate:
    """Estimate FLOPs per token for DeltaNet adjustment vs MTP step.

    All estimates use ``2 * m * n`` for a matrix multiply of shapes
    ``(1, m) @ (m, n)`` (single-token decode).

    CRITICAL INSIGHT: The MLP is ~92% of both layer types' cost.
    An attention-only DeltaNet adjustment avoids this entirely.

    DeltaNet attention-only components:
      - in_proj_qkvz: hidden -> (2*key_dim + 2*value_dim)      ~99M
      - in_proj_ba:   hidden -> 2*num_v_heads                   ~0.5M
      - conv1d:       trivial (kernel_size=4)                    ~0.03M
      - recurrent update: num_v_heads * head_v_dim * head_k_dim  ~2.1M
      - out_proj:     value_dim -> hidden                        ~34M
      Total: ~136M FLOPs/token per DeltaNet attn-only layer     <-- BUT
      Note: this is the LINEAR part only, still quite cheap.

    The recurrent update is the key differentiator: it's an outer
    product per head (v_dim * k_dim), not a full hidden^2 matmul.
    """
    d = hidden_size

    # --- DeltaNet layer ---
    key_dim = linear_num_key_heads * linear_key_head_dim    # 16*128 = 2048
    value_dim = linear_num_value_heads * linear_value_head_dim  # 32*128 = 4096

    # in_proj_qkvz: d -> 2*key_dim + 2*value_dim
    qkvz_flops = 2 * d * (2 * key_dim + 2 * value_dim)
    # in_proj_ba: d -> 2*num_v_heads (tiny)
    ba_flops = 2 * d * (2 * linear_num_value_heads)
    # conv1d: kernel_size=4, conv_dim channels
    conv_dim = 2 * key_dim + value_dim
    conv_flops = 4 * conv_dim
    # recurrent state update: per head, outer product k^T @ v + gated update
    # Cost: num_v_heads * head_v_dim * head_k_dim * ~4 ops
    recurrent_flops = 4 * linear_num_value_heads * linear_value_head_dim * linear_key_head_dim
    # out_proj: value_dim -> d
    out_flops = 2 * value_dim * d

    deltanet_attn_only = qkvz_flops + ba_flops + conv_flops + recurrent_flops + out_flops

    # MLP: gate_up_proj (d -> 2*intermediate) + down_proj (intermediate -> d)
    # IDENTICAL for DeltaNet and full-attention layers.
    mlp_flops = 2 * d * 2 * intermediate_size + 2 * intermediate_size * d

    deltanet_full_layer = deltanet_attn_only + mlp_flops

    # --- Full attention (MTP decoder) layer ---
    attn_dim = num_attention_heads * head_dim           # 16*256 = 4096
    kv_dim = num_key_value_heads * head_dim             # 4*256 = 1024

    qkv_attn_flops = 2 * d * (attn_dim + 2 * kv_dim)
    o_attn_flops = 2 * attn_dim * d

    mtp_decoder_layer = qkv_attn_flops + o_attn_flops + mlp_flops
    # MTP step = fc (2*d -> d) + 1 decoder layer
    fc_flops = 2 * (2 * d) * d
    mtp_step = fc_flops + mtp_decoder_layer

    # Full model forward: 64 layers (3:1 DeltaNet to full-attention)
    full_model_forward = num_total_layers * (deltanet_full_layer * 3 + mtp_decoder_layer) // 4

    # Adjustment: attention-only DeltaNet layers
    adjustment_total = num_deltanet_layers * deltanet_attn_only

    return CostEstimate(
        deltanet_attn_only_flops=deltanet_attn_only,
        deltanet_full_layer_flops=deltanet_full_layer,
        mtp_decoder_layer_flops=mtp_decoder_layer,
        mlp_flops=mlp_flops,
        num_deltanet_layers=num_deltanet_layers,
        adjustment_total_flops=adjustment_total,
        mtp_step_flops=mtp_step,
        adjustment_vs_mtp=adjustment_total / mtp_step,
        adjustment_vs_full_fwd=adjustment_total / full_model_forward,
        full_model_forward_flops=full_model_forward,
    )


class DeltaNetAdjuster(nn.Module):
    """Runs DeltaNet attention (no MLP) to refine hidden states between MTP chain steps.

    This module holds *references* to existing DeltaNet decoder layers
    from the main model -- no weight duplication.  The adjustment pass
    runs only the linear-attention mechanism (input_layernorm -> linear_attn
    -> residual connection), SKIPPING the MLP.

    The MLP is 92% of the per-layer cost.  By skipping it, each adjustment
    layer costs ~37M FLOPs instead of ~440M.  Four layers = ~148M FLOPs =
    33% of one MTP step (453M).

    The DeltaNet layers use recurrent state (conv_state + ssm_state) that's
    populated during prefill/decode.  During adjustment we operate on shadow
    copies to avoid corrupting verified state.

    Parameters
    ----------
    layers : list[nn.Module]
        DeltaNet decoder layers (Qwen3_5DecoderLayer with
        layer_type == "linear_attention").  References, not copies.
    layer_indices : list[int]
        Original layer indices in the full model.
    attn_only : bool
        If True (default), skip the MLP in each layer.  Set False
        to run full layers (much more expensive).
    """

    def __init__(
        self,
        layers: list[nn.Module],
        layer_indices: list[int],
        attn_only: bool = True,
    ):
        super().__init__()
        self._layers = layers
        self._layer_indices = layer_indices
        self._attn_only = attn_only

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        num_layers: int = 4,
        strategy: str = "spread",
        attn_only: bool = True,
    ) -> "DeltaNetAdjuster":
        """Extract DeltaNet layers from a Qwen3.5 model.

        Parameters
        ----------
        model : nn.Module
            The Qwen3_5Model (model.model of Qwen3_5ForCausalLM).
            Must have a .layers attribute.
        num_layers : int
            How many DeltaNet layers to use.
        strategy : str
            "spread" -- evenly spaced, "last" -- last N, "first" -- first N.
        attn_only : bool
            Skip MLP in adjustment (default True, 10x cheaper).

        Returns
        -------
        DeltaNetAdjuster
        """
        deltanet_indices: list[int] = []
        deltanet_layers: list[nn.Module] = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "layer_type") and layer.layer_type == "linear_attention":
                deltanet_indices.append(i)
                deltanet_layers.append(layer)

        if len(deltanet_layers) == 0:
            raise ValueError("No DeltaNet layers found in model")

        num_layers = min(num_layers, len(deltanet_layers))

        if strategy == "spread":
            step = max(1, len(deltanet_layers) // num_layers)
            selected_positions = list(range(0, len(deltanet_layers), step))[:num_layers]
        elif strategy == "last":
            selected_positions = list(range(len(deltanet_layers) - num_layers, len(deltanet_layers)))
        elif strategy == "first":
            selected_positions = list(range(num_layers))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_layers = [deltanet_layers[p] for p in selected_positions]
        selected_indices = [deltanet_indices[p] for p in selected_positions]

        return cls(
            layers=selected_layers,
            layer_indices=selected_indices,
            attn_only=attn_only,
        )

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @torch.inference_mode()
    def adjust(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run hidden states through DeltaNet adjustment layers.

        For single-token decode: hidden_states shape is (batch, hidden_dim).

        In attn_only mode, each layer runs:
          1. input_layernorm(hidden_states, residual) -> normed, new_residual
          2. linear_attn(normed) -> attn_output
          3. hidden_states = attn_output, residual = new_residual + attn_output
        SKIPPING the post_attention_layernorm + MLP.

        In full mode, delegates to the layer's standard forward().

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape (batch, hidden_dim) -- the MTP chain output.
        residual : torch.Tensor or None
            If None, initialized from hidden_states by input_layernorm.

        Returns
        -------
        torch.Tensor
            Adjusted hidden states, same shape as input.
        """
        if not self._attn_only:
            # Full layer forward (expensive -- 10x cost)
            for layer in self._layers:
                hidden_states, residual = layer(
                    hidden_states=hidden_states,
                    residual=residual,
                )
            if residual is not None:
                hidden_states = hidden_states + residual
            return hidden_states

        # Attention-only path (cheap)
        for layer in self._layers:
            # Step 1: LayerNorm with residual stream
            if residual is None:
                residual = hidden_states
                normed = layer.input_layernorm(hidden_states)
            else:
                normed, residual = layer.input_layernorm(hidden_states, residual)

            # Step 2: DeltaNet linear attention
            attn_output = torch.empty_like(normed)
            layer.linear_attn(
                hidden_states=normed,
                output=attn_output,
            )

            # Step 3: Apply layer scale if present
            if layer.layer_scale:
                if len(attn_output.shape) == 2:
                    attn_output = attn_output * (
                        layer.attn_layer_scale.to(attn_output.dtype)[0] + 1
                    )
                else:
                    attn_output = attn_output * (
                        layer.attn_layer_scale.to(attn_output.dtype) + 1
                    )

            # Step 4: Update hidden_states (skip MLP entirely)
            hidden_states = attn_output

        # Combine residual stream
        if residual is not None:
            hidden_states = hidden_states + residual

        return hidden_states

    def __repr__(self) -> str:
        mode = "attn-only" if self._attn_only else "full"
        return (
            f"DeltaNetAdjuster("
            f"num_layers={self.num_layers}, "
            f"mode={mode}, "
            f"layer_indices={self._layer_indices})"
        )


# ---------------------------------------------------------------------------
# Cost analysis
# ---------------------------------------------------------------------------

def print_cost_analysis(num_deltanet_layers: int = 4) -> CostEstimate:
    """Print detailed cost comparison for Qwen3.5-A22B (64-layer)."""
    est = estimate_cost(
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_deltanet_layers=num_deltanet_layers,
        num_total_layers=64,
    )
    print(est)
    print()

    # Acceptance rate analysis
    current_rates = [0.87, 0.68, 0.54, 0.39, 0.28, 0.21, 0.16]
    current_expected = sum(current_rates)

    improved_rates = [0.87, 0.68, 0.60, 0.55, 0.45, 0.35, 0.28]
    improved_expected = sum(improved_rates)

    print("Acceptance rate analysis:")
    print(f"  Current  expected tokens: {current_expected:.3f}")
    print(f"  Improved expected tokens: {improved_expected:.3f}")
    print(f"  Token gain: {improved_expected - current_expected:.3f} "
          f"({(improved_expected/current_expected - 1)*100:.1f}%)")
    print()

    # Draft cost with attn-only adjustment
    mtp_only_cost = 7 * est.mtp_step_flops
    adjusted_cost = 7 * est.mtp_step_flops + 6 * est.adjustment_total_flops
    cost_overhead = (adjusted_cost / mtp_only_cost - 1)

    print("Draft cost analysis (7-step chain, 6 adjustments):")
    print(f"  MTP-only cost:   {mtp_only_cost/1e9:.2f} GFLOPs")
    print(f"  With adjustment: {adjusted_cost/1e9:.2f} GFLOPs")
    print(f"  Overhead: {cost_overhead*100:.1f}%")
    print()

    # Net throughput
    verify_cost = est.full_model_forward_flops
    current_throughput = current_expected / (verify_cost + mtp_only_cost)
    improved_throughput = improved_expected / (verify_cost + adjusted_cost)
    net_gain = improved_throughput / current_throughput - 1

    print(f"  Net throughput change: {net_gain*100:+.1f}%")
    print()

    print("Key insight -- DeltaNet attn-only vs MTP step:")
    print(f"  1 DeltaNet attn-only: {est.deltanet_attn_only_flops/1e6:.1f} MFLOPs")
    print(f"  1 MTP step:           {est.mtp_step_flops/1e6:.1f} MFLOPs")
    print(f"  {num_deltanet_layers} DeltaNet attn-only = "
          f"{est.adjustment_vs_mtp*100:.1f}% of 1 MTP step")
    print(f"  6 adjustments add {6 * est.adjustment_vs_mtp * 100:.1f}% "
          f"to 7 MTP steps")
    print()
    print(f"  MLP cost alone:       {est.mlp_flops/1e6:.1f} MFLOPs "
          f"({est.mlp_flops/est.deltanet_full_layer_flops*100:.0f}% of DeltaNet layer)")

    return est


if __name__ == "__main__":
    print("=" * 60)
    print("DeltaNet Adjuster -- Cost Analysis")
    print("Qwen3.5-A22B (64 layers: 48 DeltaNet + 16 full-attn)")
    print("=" * 60)
    print()
    print_cost_analysis(num_deltanet_layers=4)
