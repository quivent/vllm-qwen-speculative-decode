# Cascade MTP Training Strategy

## Problem

Qwen 3.5's MTP head was trained for depth-1 prediction. When chained 7 times, acceptance degrades: 87% → 68% → 54% → 39% → 28% → 21% → 16%. The head sees drifted hidden states at deeper positions it was never trained on.

## Solution: Depth-Specific Cascade Heads

Train separate heads for each chain depth. Each head is fine-tuned from the stock MTP weights on the hidden states produced by the EXACT chain of predecessors it will follow at inference.

```
Qwen hidden → Head₀ → Head₁(Head₀ output) → Head₂(Head₁ output) → ...
```

## Architecture

Each head is identical to the stock MTP head:
- `pre_fc_norm_embedding` (RMSNorm 5120)
- `pre_fc_norm_hidden` (RMSNorm 5120)  
- `fc` (Linear 10240 → 5120, concatenates [embed, hidden])
- 1 decoder layer (full attention + MLP, 849MB)
- `norm` (RMSNorm 5120)

Shared across all heads: `embed_tokens` (2.5GB) and `lm_head` (2.5GB).

## Training Loss

```
L = CE(token_pred, target) + λ_h × MSE(output_hidden, ideal_hidden)
```

- **CE**: predict the correct next token at this depth
- **MSE**: produce a hidden state close to what the FULL model would produce at this position

The MSE term is **critical**. Without it:
- Each head optimizes its own token prediction but lets hidden state drift
- The next head in the cascade gets degraded input
- Error compounds across depths

With it:
- Each head actively reduces accumulated drift
- The next head gets a cleaner input
- The cascade self-corrects

## Training Data

**Source**: The SAME model checkpoint that will verify drafts at inference. This is non-negotiable — if training hidden states don't match inference hidden states, the head produces mismatched outputs (verified: 0% acceptance when using cached hidden states from a different run).

**Collection**:
1. Run the W4A16 model (the exact serving checkpoint) on N prompts
2. At each token position, capture:
   - `hidden_states[-1][pos]` — the ideal hidden state
   - `input_ids[pos+1]` — the token at this position
3. These pairs are the training targets

**For depth D training**:
1. Run heads 0..D-1 on the collected hidden states to get the actual chain output
2. This chain output is the INPUT to head D
3. Target: predict `input_ids[pos+D+1]` and match `hidden_states[-1][pos+D+1]`

## GPU Memory Plan

| Component | VRAM |
|---|---|
| Serving model (W4A16, reduced KV) | ~35GB |
| Training (MTP head + optimizer + embeddings) | ~15GB |
| Headroom | ~48GB |
| **Total** | **~50GB of 98GB** |

The serving model MUST run during training to generate correct hidden states. Use `--gpu-memory-utilization 0.35 --max-model-len 4096` to limit KV cache.

## Training Config

```bash
# Collect hidden states from the SERVING model via API
python3 collect_hidden_states.py \
    --endpoint http://localhost:8001 \
    --num-prompts 500 \
    --output /tmp/ideal_hidden_states.pt

# Train cascade heads
python3 cascade_mtp_corrective.py \
    --hidden-states /tmp/ideal_hidden_states.pt \
    --stock-head /path/to/stock_mtp_head.safetensors \
    --output-dir /path/to/cascade-heads/ \
    --num-depths 7 \
    --epochs 3 \
    --lr 5e-5 \
    --lambda-h 1.0 \
    --batch-size 8
```

## Deployment

At inference, swap the MTP head weights at each chain step:

```python
for step in range(num_speculative_tokens):
    # Swap to this depth's head
    cascade_heads[step].copy_weights_to(mtp_model)
    # Run one draft step
    draft_token, hidden = mtp_model.forward(...)
```

Weight swap via `copy_()` is CUDA-graph compatible (verified: graphs capture pointers, not values). Cost: ~0.2ms per swap on GH200.

## What Failed and Why

1. **Sibling heads (diversity training)**: Trained 3 copies with noise + diversity loss. Pushed apart but not toward better predictions. 0% acceptance — diverged from target distribution.

2. **Agent's cascade (CE-only)**: Trained on cached hidden states from a DIFFERENT forward pass than inference. 0% acceptance — hidden state mismatch between training and serving.

3. **My corrective version (fc+norm only)**: Skipped the decoder layer. 0% accuracy — the decoder layer (attention + MLP) is essential, not optional.

## What Must Be Different

1. **Hidden states from the SERVING model** — not from a separate bf16 model load
2. **Full decoder layer** in each head — not just fc+norm
3. **MSE loss on hidden states** — not just CE on tokens
4. **Same dtype** as serving (bfloat16) — not float32 from training
5. **Test after EACH depth** — not train all 7 then discover it's broken

## Expected Outcome

If trained correctly, each depth head should maintain ~70-80% acceptance at its position (vs the stock head's 87→16% degradation). This would increase expected accepted tokens from 3.1 to ~5.2, yielding ~40% throughput improvement.
