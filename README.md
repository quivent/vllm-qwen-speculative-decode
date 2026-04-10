# vllm-qwen-speculative-decode

Speculative decoding optimization toolkit for vLLM + Qwen 3.5-27B MTP heads on NVIDIA GH200.

Four independent strategies for increasing tokens/second, all composable:

## Strategies

### 1. Adaptive MTP (`adaptive_mtp.py`)

Drop-in `EagleProposer` wrapper. Tracks per-position acceptance rate with exponential moving average and shortens the draft chain when tail positions stop paying for themselves.

- Zero vLLM internals changes
- Saves ~55% of draft forward passes
- Biggest win on low-acceptance workloads (prose, open-ended generation)

```python
from adaptive_mtp import AdaptiveMTPProposer
# swap EagleProposer -> AdaptiveMTPProposer in gpu_model_runner
```

### 2. Sibling MTP Heads (`microgreens/`)

Clone the MTP head K times with noise perturbation. Each head specializes on different plausible continuations. Verified in one batched pass via tree attention.

| Component | Size (bf16) |
|---|---|
| Per-head unique params (fc + decoder + norms) | 849 MB |
| K=3 total | 2.5 GB |
| embed_tokens + lm_head (shared, not duplicated) | 0 |

```bash
# Clone heads from bf16 checkpoint
python microgreens/mtp_clone.py \
    --model-path /path/to/Qwen3.5-27B \
    --num-heads 3 \
    --output-dir ./sibling-heads

# Fine-tune with diversity loss
python microgreens/mtp_diversity_train.py \
    --model-path /path/to/Qwen3.5-27B \
    --head-dir ./sibling-heads \
    --dataset wikitext \
    --epochs 2
```

### 3. Partial-Layer Verification (`partial_layer_verify.py`)

Run only the first N of 64 transformer layers for draft token verification. If early-exit logits agree with draft tokens, skip remaining layers.

- Default N=32 (50% depth), ~85-90% agreement with full model
- Requires `--enforce-eager` (no CUDA graphs) for conditional branching
- Production path: two-graph CUDA dispatch (see `docs/TWO-GRAPH-CUDA-DISPATCH.md`)

```bash
# Measure agreement rate across exit layers
python partial_layer_verify.py \
    --model-path /path/to/Qwen3.5-27B \
    --sweep
```

### 4. Branching Tree Speculation

Use top-K candidates from MTP head instead of argmax, verified in one batched forward pass with tree attention mask.

**Status:** Blocked by vLLM 0.19 bug — `propose_tree()` assumes Eagle-style model, crashes on MTP with `AttributeError: 'EagleProposer' has no attribute 'positions'`. Patch in `eagle-tree-drafting.patch` fixes the tuple-unpack bug but the positions issue remains.

## Baseline

Measured on GH200 (96GB HBM3e, 900 GB/s), Qwen 3.5-27B W4A16, vLLM 0.19, MTP spec=7:

| Workload | tok/s |
|---|---|
| Code generation | 236 |
| Mixed (5-prompt suite) | 188 |
| Prose / explanation | 130 |

Content type is the dominant variable in MTP acceptance rate.

## Files

```
adaptive_mtp.py              # Strategy 1: adaptive chain length
partial_layer_verify.py      # Strategy 3: partial-layer verification
microgreens/
  mtp_clone.py               # Strategy 2: clone MTP head weights
  mtp_diversity_train.py     # Strategy 2: fine-tune with diversity loss
  sibling_mtp_proposer.py    # Strategy 2: vLLM EagleProposer integration
scripts/
  bench-tok-s.py             # 5-prompt throughput benchmark
  vllm-tree-spec.sh          # Tree attention launch config
docs/
  TWO-GRAPH-CUDA-DISPATCH.md # Design doc for CUDA graph conditional branching
eagle-tree-drafting.patch    # Fix for propose_tree() MTP tuple-unpack bug
```

## Requirements

- vLLM >= 0.19
- Qwen 3.5-27B (bf16 checkpoint for MTP head cloning; W4A16 for serving)
- NVIDIA GPU with >= 24GB VRAM (GH200 recommended)
- PyTorch >= 2.1

## Known Issues

- **GPTQ quantized models strip MTP weights.** Clone sibling heads from the bf16 checkpoint.
- **vLLM 0.19 tree attention + MTP incompatible.** `propose_tree()` crashes on MTP models. Branching tree strategy blocked until upstream fix.
- **Qwen 3.5 MTP asymmetric head_dim.** Queries use 512, keys/values use 256. All code in this repo accounts for this.

## License

Apache-2.0
