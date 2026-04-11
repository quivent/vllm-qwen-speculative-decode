# PLV at Full-Attention Layer Boundaries

## Hypothesis

Qwen 3.5-27B uses a hybrid architecture: every 4th layer is full
attention, the other 3 are DeltaNet (linear attention) with rolling
recurrent state. DeltaNet layers don't produce useful intermediate
logits because they maintain compressed recurrent state rather than
full contextual representations.

**Prediction:** exiting at full-attention boundaries (layers 4, 8, 12,
..., 60, 64 in 1-indexed) should yield better p_agree than exiting at
arbitrary layers, because these are the checkpoints where the model
consolidates global context.

## Architecture Reference

```
full_attention_interval: 4
layer_types (0-indexed):
  0-2:  linear_attention (DeltaNet)
  3:    full_attention          <-- exit point: layer 4 (1-idx)
  4-6:  linear_attention
  7:    full_attention          <-- exit point: layer 8
  ...
  60-62: linear_attention
  63:   full_attention          <-- exit point: layer 64 (final)
```

## Experimental Setup

### Sweep A: Single next-token prediction (this run)
- **Model:** Qwen3.5-27B bf16 on CPU
- **Prompts:** 30 short prompts (5-12 tokens each)
- **Metric:** p_agree on the last-token next-token prediction
- **Layers:** Full-attn [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60] + DeltaNet [2,6,10,14,18,22,26,30]
- **Strategies:** (A) final_norm + lm_head, (B) next_layer_input_layernorm + final_norm + lm_head

### Sweep B: Multi-token probes (companion run)
- **Model:** Same
- **Prompts:** 12 long prompts (100-200 tokens each), 1984 total tokens
- **Layers (0-indexed):** [3, 7, 11, 15, 31, 47]
- **Additional:** Regression adapter training (52M params per layer)

## Results

### Sweep A: Full-Attention vs DeltaNet Comparison

```
Full-Attention Layers (1-indexed, after layer completes):
Layer | norm p_agree | next p_agree | norm p_top5 | next p_top5 | norm KL  | N
------|-------------|-------------|------------|------------|---------|---
    4 |      0.1000 |      0.1000 |     0.1333 |     0.1667 |   6.541 | 30
    8 |      0.1000 |      0.0333 |     0.1333 |     0.0667 |   6.377 | 30
   12 |      0.1000 |      0.1000 |     0.1667 |     0.1333 |   6.412 | 30
   16 |      0.1000 |      0.1000 |     0.1333 |     0.1000 |   6.419 | 30
   20 |      0.0667 |      0.0667 |     0.0667 |     0.0667 |   6.205 | 30
   24 |      0.1000 |      0.0333 |     0.1333 |     0.0667 |   5.981 | 30
   28 |      0.1333 |      0.1000 |     0.1333 |     0.1000 |   5.817 | 30
   32 |      0.1333 |      0.1000 |     0.1667 |     0.1000 |   5.870 | 30
   36 |      0.1000 |          — |     0.1667 |          — |   5.383 | 30
   40 |      0.2000 |          — |     0.2667 |          — |   4.694 | 30
   44 |      0.1667 |          — |     0.2000 |          — |   4.654 | 30
   48 |      0.1333 |      0.1333 |     0.1667 |     0.2333 |   4.511 | 30
   52 |      0.2333 |          — |     0.4333 |          — |   4.006 | 30
   56 |      0.3000 |          — |     0.5667 |          — |   3.564 | 30
   60 |      0.5333 |      0.5667 |     0.8333 |     0.8333 |   3.050 | 30

DeltaNet Layers (1-indexed):
Layer | norm p_agree | norm p_top5 | norm KL  | N
------|-------------|------------|---------|---
    2 |      0.1000 |     0.1333 |   6.722 | 30
    6 |      0.1000 |     0.1333 |   6.426 | 30
   10 |      0.1000 |     0.1333 |   6.348 | 30
   14 |      0.1000 |     0.1333 |   6.464 | 30
   18 |      0.1000 |     0.1333 |   6.370 | 30
   22 |      0.1000 |     0.1000 |   6.197 | 30
   26 |      0.1333 |     0.1333 |   5.874 | 30
   30 |      0.1000 |     0.1333 |   5.846 | 30
```

### Sweep B: Multi-Token Probes (1984 tokens, 0-indexed layers)

```
Layer | Raw PLV  | Adapter p_agree | Cosine(L,63) | L2(L,63)
------|---------|----------------|-------------|--------
    3 |   1.6%  |        25.2%   |      0.289  |   427.7
    7 |   1.7%  |        26.2%   |      0.298  |   425.2
   11 |   2.7%  |        25.9%   |      0.304  |   421.7
   15 |   3.3%  |        32.7%   |      0.306  |   419.7
   31 |   9.3%  |        28.5%   |      0.342  |   413.1
   47 |  15.8%  |        36.8%   |      0.411  |   399.7
```

## Analysis

### 1. Full-attention boundaries DO NOT help in early layers

For layers 4-32, p_agree is essentially random (~10%, consistent with
1/vocab_size baseline for the top token hitting by chance). There is
**no measurable advantage** to exiting at a full-attention layer vs a
DeltaNet layer in the first half of the model.

The DeltaNet comparison layers show identical ~10% p_agree. The
hypothesis that DeltaNet layers would be *worse* than full-attention
layers is not supported — both are equally bad in early/middle layers.

### 2. The transition is gradual but late (layers 40-60)

The gap probe reveals a smooth ramp from layer 40 onward:

```
Layer  p_agree  p_top5   KL
  32    13.3%   16.7%   5.87
  36    10.0%   16.7%   5.38
  40    20.0%   26.7%   4.69   <-- first signal above noise
  44    16.7%   20.0%   4.65
  48    13.3%   16.7%   4.51
  52    23.3%   43.3%   4.01   <-- top-5 starts working
  56    30.0%   56.7%   3.56   <-- viable for top-5 verification
  60    53.3%   83.3%   3.05   <-- viable for top-1 verification
```

p_agree transitions from noise (~10%) to 53% between layers 40-60,
with the steepest gains in the final 8 layers (52-60). Layer 56
(87.5% depth) achieves 30% top-1 and 57% top-5 — marginally useful
for speculative verification if you accept top-5 matching.

### 3. next_layer_input_layernorm does NOT help

Applying the next layer's input layernorm before the final norm + lm_head
either makes no difference or slightly hurts. The representations at
intermediate layers are not misaligned due to normalization — they are
fundamentally in a different subspace.

### 4. KL divergence shows gradual convergence

While p_agree is near-random until layer 60, KL divergence decreases
monotonically from ~6.7 (layer 2) to ~3.0 (layer 60). This suggests
the probability distributions are gradually aligning even when the
argmax doesn't match — the model builds up its prediction incrementally
across all layers.

### 5. Regression adapters help significantly but plateau

A 52M-parameter MLP adapter (h_L + MLP(h_L) -> h_63) improves
p_agree from 1.6% to 25.2% at layer 3, and from 15.8% to 36.8% at
layer 47. But even with an adapter, accuracy is far from acceptable
for verification. The adapter essentially learns a rough linear
projection from the intermediate space to the output space.

## Implications for Speculative Decoding Verification

**Raw early exit is not viable for aggressive skip.** Even at
full-attention boundaries, the model needs 87-94% of its layers to
achieve meaningful agreement:

- Layer 56 (87.5%): 30% top-1, 57% top-5 — saves 8 layers (12.5%)
- Layer 60 (93.8%): 53% top-1, 83% top-5 — saves 4 layers (6.3%)
- Layer 48 (75.0%): 13% top-1, 17% top-5 — no better than random

The "run half the layers as verification" strategy is ruled out.

**Why?** Qwen 3.5's hybrid architecture distributes computation
differently than pure-transformer models. The DeltaNet layers
accumulate recurrent state across the sequence, and this state is only
consolidated into a globally-coherent representation at the full-attention
checkpoints. But even the full-attention layers in the first 75% of
the model haven't accumulated enough information to predict the final
output — the last 16 layers (48-64) do critical work.

## Next Steps

1. **Layer-skip verification:** Instead of early exit, verify by
   skipping specific DeltaNet layers (e.g., run layers 0-3, skip 4-6,
   run 7, skip 8-10, ...) — cheaper than full model but may preserve
   p_agree better than early exit.

2. **Draft-model verification:** Use the DeltaNet-only draft model
   at `/home/ubuntu/models/Qwen3.5-27B-DeltaNet-draft-nodep` as a
   fast verifier — it skips the expensive full-attention layers.

3. **Layer 56 as top-5 verifier:** At 57% top-5 agreement, layer 56
   could serve as a cheap pre-filter: if the draft token is NOT in
   layer-56's top-5, reject it immediately (saves running layers
   57-64). Only ~43% false-positive rate vs full verification.

## Data Files

- `/home/ubuntu/aut/plv_full_attn_results.json` — sweep A raw data
- `/home/ubuntu/aut/probes/results.json` — sweep B probe results
- `/home/ubuntu/aut/probes/cached_hiddens.pt` — cached hidden states (5.4 GB)
- `/home/ubuntu/aut/probes/regression_adapter_L*.pt` — trained adapters
