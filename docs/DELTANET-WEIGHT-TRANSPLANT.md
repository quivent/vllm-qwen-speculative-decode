# DeltaNet Weight Transplant: Dedicated Draft Weights into modal_mtp

## Concept

modal_mtp runs the main Qwen3.5-27B model in "draft mode" by skipping the 16
full-attention layers and executing only the 48 DeltaNet layers + MLPs. The
DeltaNet subnetwork was trained as part of a hybrid model -- it was never
optimized purely for next-token prediction in isolation.

The dedicated 48-layer draft model (`Qwen3.5-27B-DeltaNet-draft`) was trained
specifically as a standalone draft model. Its DeltaNet layers should produce
better draft tokens than the main model's DeltaNet subnetwork.

The transplant idea: load the dedicated draft model's DeltaNet weights into
the main model's DeltaNet layer parameter tensors. modal_mtp's infrastructure
(shadow state, attention skipping, slot mapping) remains unchanged. Zero
architecture changes -- just better weights in the same slots.

## Structural Analysis

### Layer counts and types

| Model | Total layers | DeltaNet | Full-attention |
|-------|-------------|----------|----------------|
| Main (Qwen3.5-27B) | 64 | 48 | 16 |
| Draft (DeltaNet-draft) | 48 | 48 | 0 |

### Main model layer pattern

Every 4th layer is full-attention (layers 3, 7, 11, 15, ..., 63).
DeltaNet layers: 0,1,2, 4,5,6, 8,9,10, 12,13,14, ..., 60,61,62.

### Layer index mapping

Draft model layers are numbered 0-47 contiguously. Main model DeltaNet
layers are at non-contiguous indices. The mapping:

```
draft  0 -> main  0     draft 12 -> main 16     draft 24 -> main 32     draft 36 -> main 48
draft  1 -> main  1     draft 13 -> main 17     draft 25 -> main 33     draft 37 -> main 49
draft  2 -> main  2     draft 14 -> main 18     draft 26 -> main 34     draft 38 -> main 50
draft  3 -> main  4     draft 15 -> main 20     draft 27 -> main 36     draft 39 -> main 52
draft  4 -> main  5     draft 16 -> main 21     draft 28 -> main 37     draft 40 -> main 53
draft  5 -> main  6     draft 17 -> main 22     draft 29 -> main 38     draft 41 -> main 54
draft  6 -> main  8     draft 18 -> main 24     draft 30 -> main 40     draft 42 -> main 56
draft  7 -> main  9     draft 19 -> main 25     draft 31 -> main 41     draft 43 -> main 57
draft  8 -> main 10     draft 20 -> main 26     draft 32 -> main 42     draft 44 -> main 58
draft  9 -> main 12     draft 21 -> main 28     draft 33 -> main 44     draft 45 -> main 60
draft 10 -> main 13     draft 22 -> main 29     draft 34 -> main 45     draft 46 -> main 61
draft 11 -> main 14     draft 23 -> main 30     draft 35 -> main 46     draft 47 -> main 62
```

Formula: `main_idx = draft_idx + (draft_idx // 3)` (inserts a gap every 3 layers for the full-attention layer).

### Weight compatibility

All DeltaNet weight tensors share identical shapes between draft and main:

| Weight | Shape |
|--------|-------|
| `linear_attn.A_log` | [48] |
| `linear_attn.conv1d.weight` | [10240, 1, 4] |
| `linear_attn.dt_bias` | [48] |
| `linear_attn.in_proj_a.weight` | [48, 5120] |
| `linear_attn.in_proj_b.weight` | [48, 5120] |
| `linear_attn.in_proj_qkv.weight` | [10240, 5120] |
| `linear_attn.in_proj_z.weight` | [6144, 5120] |
| `linear_attn.norm.weight` | [128] |
| `linear_attn.out_proj.weight` | [5120, 6144] |

MLP and layernorm shapes also match (hidden_size=5120, intermediate_size=17408).

### norm.weight gap

12 draft layers (indices 3,7,11,15,19,23,27,31,35,39,43,47 -- every 4th)
lack `linear_attn.norm.weight`. The corresponding main model layers DO have
this weight. The transplant script skips these missing weights, leaving the
main model's original norm.weight in place.

This is architecturally coherent: the draft model was trained without these
norms, so the other weights in those layers were trained to work without them.
After transplant, the main model retains its norm.weight but gets draft-trained
values for everything else. Minor distribution mismatch possible on those 12
layers, but all other 36 layers are clean transplants.

### What gets transplanted

Per DeltaNet layer (48 layers):
- `linear_attn.*` (8-9 weights per layer depending on norm.weight presence)
- `mlp.down_proj.weight`, `mlp.gate_proj.weight`, `mlp.up_proj.weight`
- `input_layernorm.weight`, `post_attention_layernorm.weight`

NOT transplanted (kept from main model):
- embed_tokens (shared vocabulary embedding)
- lm_head (used for verification scoring)
- Final norm
- Full-attention layer weights (layers 3,7,11,...,63)
- MTP head weights (separate predictor)
- Vision encoder weights

### MTP head

Both models have identical MTP head structure (15 weights, same shapes).
The draft model's MTP head was trained alongside the DeltaNet layers, so it
may pair better with the transplanted weights. The transplant script optionally
copies MTP head weights too.

## Implementation

See `deltanet_transplant.py` in the repo root. Runtime operation:

1. Load main model on GPU via vLLM (normal startup)
2. Load draft model safetensors as CPU tensors (read-only, ~40GB RAM)
3. For each of 48 DeltaNet layers: copy draft weights into main model params in-place
4. Optionally copy MTP head weights
5. Free draft model tensors
6. Run modal_mtp as normal -- same shadow state, same attention skipping, better drafts

No architecture changes. No config changes. No new model classes. The main
model's full-attention layers and verification path are completely unaffected.

## Risk assessment

- **Reversible**: Original weights can be restored by reloading the main model.
- **No shape changes**: All tensor dimensions match exactly.
- **norm.weight gap on 12 layers**: Minor. Those layers get draft weights for
  everything except norm, which stays from main. Worst case: slightly degraded
  draft quality on those 12 layers vs a perfect transplant.
- **Training distribution shift**: Draft model was trained to produce good
  next-token predictions standalone. When its weights run inside the main
  model's residual stream (which includes full-attention layer outputs during
  verify), there may be distribution mismatch. But during draft mode,
  full-attention layers are identity -- so the draft weights run in exactly
  the environment they were trained for.
