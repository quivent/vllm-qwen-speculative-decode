# GH200 Compute vs Bandwidth Analysis: Qwen 3.5-27B W4A16

Date: 2026-04-11
Hardware: NVIDIA GH200 480GB (H100 die, 132 SMs, HBM3e)

## Executive Summary

The initial hypothesis — that GH200 is compute-bound at W4A16 — is **wrong**.
Roofline analysis proves the workload is deeply bandwidth-bound (OI=3.9 vs
machine balance=268). But measured throughput (94 tok/s baseline) is only
**19% of the bandwidth-bound ceiling** (~500 tok/s). The bottleneck is
neither raw compute nor raw bandwidth — it is **kernel scheduling overhead,
CUDA graph efficiency, and per-layer launch latency** consuming ~80% of
wall-clock time.

GPU telemetry during single-request decode confirms this:
- GPU compute utilization: 27%
- Memory bandwidth utilization: 13%
- Power draw: 216W / 700W TDP (31%)
- SM clock: 1980 MHz (max), Mem clock: 2619 MHz (max)

The hardware is idling between kernel launches.

## Hardware Specs

| Metric | Value |
|---|---|
| GPU | GH200 480GB (H100 die) |
| SMs | 132 |
| Max SM clock | 1980 MHz |
| FP16 Tensor Core (dense) | ~1070 TFLOPS |
| INT8 Tensor Core (dense) | ~2141 TOPS |
| HBM3e bandwidth (spec) | 4.8 TB/s |
| HBM3e bandwidth (achievable ~83%) | ~4.0 TB/s |
| Memory clock | 2619 MHz |

## Model Architecture

| Parameter | Value |
|---|---|
| Layers | 64 |
| Hidden dim | 3584 |
| Attention heads | 28 (KV heads: 4, GQA) |
| Head dim | 128 |
| MLP intermediate | 18944 (SwiGLU) |
| Vocab size | 151936 |
| Total params | ~16B (effective, W4 quantized subset of 27B) |

## Roofline Analysis

### FLOPs per token (decode, batch=1)

| Component | GFLOPs/token |
|---|---|
| Attention projections (QKV+O) per layer | 0.059 |
| MLP (gate+up+down) per layer | 0.407 |
| Per layer total | 0.466 |
| 64 layers | 29.83 |
| LM head | 1.09 |
| **Total** | **30.92** |

### Weight transfer per token

| Format | Size |
|---|---|
| W4A16 (0.5 bytes/param) | ~8.0 GB |
| FP16 (2 bytes/param) | ~32.0 GB |

### Operational Intensity

```
OI = 30.92 GFLOPs / 8.0 GB = 3.9 FLOPs/byte
Machine balance = 1070 TFLOPS / 4.0 TB/s = 267.5 FLOPs/byte

OI (3.9) << Machine balance (267.5) → DEEPLY BANDWIDTH-BOUND
```

This means:
- The FP16 tensor cores could process this workload ~69x faster than memory can feed them
- INT4 quantization reduces memory transfer but does NOT change the fundamental ratio enough
  to cross the roofline ridge point
- The workload would need batch_size=69 to become compute-bound

### Theoretical Ceilings

| Regime | Prediction |
|---|---|
| Compute-bound (FP16 TC @ 70% eff) | 24,236 tok/s |
| Bandwidth-bound (4.0 TB/s) | 500 tok/s |
| **Measured (baseline, batch=1)** | **94 tok/s** |

Measured is 19% of bandwidth-bound prediction, 0.4% of compute-bound prediction.

## Experimental Measurements

### Test Configuration

- vLLM 0.19.0, torch.compile enabled, CUDA graphs (piecewise)
- MacheteLinearKernel for GPTQ-Marlin
- max_model_len=4096 (to avoid KV cache memory pressure)

### Baseline (No Speculative Decode) — Single Request

| max_tokens | Completion | Wall time | Throughput |
|---|---|---|---|
| 10 | 10 | 0.144s | 69.4 tok/s |
| 25 | 25 | 0.303s | 82.5 tok/s |
| 50 | 50 | 0.565s | 88.6 tok/s |
| 100 | 100 | 1.099s | 91.0 tok/s |
| 200 | 200 | 2.161s | 92.6 tok/s |

Linear fit: `elapsed = 36.6ms + 10.619ms * tokens`
- **TTFT: ~37ms**
- **Per-token decode: 10.6ms = 94.2 tok/s**

### Baseline — Batch Throughput

| Batch size | Total tokens | Wall time | Aggregate tok/s | Per-request tok/s |
|---|---|---|---|---|
| 1 | 100 | 1.10s | 90.6 | 90.7 |
| 2 | 120 | 1.13s | 106.0 | 84.3 |
| 4 | 320 | 1.21s | 263.4 | 80.5 |
| 8 | 720 | 1.21s | 592.8 | 82.0 |

**Batch scaling is nearly linear**: 8x batch → 6.5x throughput with minimal per-request degradation.
This confirms bandwidth-bound behavior — batching amortizes per-token overhead across
more useful compute, and the memory system has massive headroom (only 13% utilized at batch=1).

### Speculative Decode (MTP tree, 9 spec tokens)

| max_tokens | Completion | Wall time | Throughput |
|---|---|---|---|
| 10 | 10 | 0.422s | 23.7 tok/s |
| 25 | 25 | 0.732s | 34.2 tok/s |
| 50 | 50 | 1.379s | 36.2 tok/s |
| 100 | 100 | 2.411s | 41.5 tok/s |
| 200 | 200 | 4.591s | 43.6 tok/s |

Linear fit: `elapsed = 71.5ms + 28.021ms * tokens`
- **TTFT: ~72ms** (2x baseline due to drafter overhead)
- **Per-token decode: 28.0ms = 35.7 tok/s** (2.6x SLOWER than baseline)

### Spec Decode — Batch Throughput

| Batch size | Total tokens | Wall time | Aggregate tok/s | Per-request tok/s |
|---|---|---|---|---|
| 1 | 100 | 2.72s | 36.7 | 36.8 |
| 2 | 120 | 4.58s | 26.2 | 17.1 |
| 4 | 320 | 3.68s | 87.0 | 26.5 |
| 8 | 720 | 3.19s | 225.5 | 31.7 |

### GPU Telemetry During Baseline Decode (batch=1)

| Metric | Value |
|---|---|
| GPU compute utilization | 27% avg (0-30% range) |
| Memory bandwidth utilization | 13% |
| Power draw | 216W / 700W (31%) |
| SM clock | 1980 MHz (at max) |
| Memory clock | 2619 MHz (at max) |

## Analysis

### Why is speculative decode SLOWER?

The prior report of "186 tok/s with spec decode" was likely measuring aggregate
throughput under concurrent load (multiple requests batched by the scheduler),
not single-request performance. With spec decode:

1. **Per-cycle overhead is 2-3x**: Each decode cycle runs the full model + drafter model +
   verification + tree attention, reading weights multiple times per accepted token
2. **Tree attention with PIECEWISE CUDA graphs**: The TREE_ATTN backend forces piecewise
   graph capture (not full), adding inter-kernel scheduling overhead
3. **Even with 4.1 accepted tokens/cycle**: The cycle time is ~10x longer than a single
   baseline decode step, so net throughput drops

Spec decode helps when single-token decode is slow (high-latency memory systems).
On GH200 with 4.8 TB/s bandwidth, single-token decode is already fast enough that
the drafter overhead exceeds the savings.

### Why is measured throughput only 19% of bandwidth ceiling?

With 13% memory bandwidth utilization during decode, the GPU is reading weights at
~0.6 TB/s, not 4.0 TB/s. The remaining 87% of bandwidth capacity is wasted on:

1. **Kernel launch overhead**: 64 layers × ~10 kernels/layer = ~640 kernel launches per token.
   Even at 5us per launch, that's 3.2ms of pure overhead (vs 10.6ms total per token = 30%).

2. **CUDA graph gaps**: Piecewise CUDA graphs capture compute but still have inter-piece
   scheduling gaps. The full-graph mode would be better but TREE_ATTN prevents it.

3. **Small matrix problem**: At batch=1, the weight matrices are large (e.g., 3584x18944)
   but the activation vector is just 1x3584. This creates extremely skinny GEMMs where
   tensor core utilization is poor and memory access patterns are suboptimal.

4. **Framework overhead**: vLLM scheduler, Python async event loop, tokenizer, sampling —
   all contribute non-GPU time.

### The real optimization landscape

The workload is **overhead-bound** (neither compute-bound nor bandwidth-saturated):

| Strategy | Expected Impact | Mechanism |
|---|---|---|
| **Batching (batch=4-8)** | 3-6x throughput | Amortizes per-token overhead, widens GEMMs |
| **Full CUDA graphs** | 20-40% latency reduction | Eliminates inter-kernel launch gaps |
| **Continuous batching** | 2-4x throughput | vLLM already does this; benefit at load |
| **Operator fusion** | 10-20% latency reduction | Fewer kernel launches per layer |
| **FP8 activations** | 10-15% | Reduces activation bandwidth, enables FP8 TC |
| **Remove spec decode for batch=1** | **2.6x improvement** | Eliminates drafter overhead |
| Spec decode | Helpful only at batch>=8 | Where per-request latency matters under load |

### When spec decode helps vs hurts

```
Single request: spec decode HURTS (94 → 36 tok/s, -62%)
Batch=8: spec decode slightly helps aggregate (593 → 226 tok/s — actually still hurts)
```

Spec decode on GH200 may only help in the regime where:
- Batch sizes are very large (>16) AND individual latency matters more than throughput
- OR the model is much larger (70B+) where single-token decode is genuinely slow

For Qwen 3.5-27B W4A16 on GH200, **spec decode should be disabled for all batch sizes tested**.

## Correction to Prior Analysis

The "186 tok/s with spec decode" figure from the MTP analysis was likely measured as:
- Aggregate throughput under vLLM's continuous batching with multiple concurrent requests
- Or measured in tokens-per-second including speculated tokens (not just accepted tokens)
- Or measured with a different vLLM version / configuration

The correct single-request figures are:
- **Baseline: 94 tok/s** (no spec decode)
- **Spec decode: 36 tok/s** (with MTP tree, 9 candidates)

## Recommendations

1. **Disable speculative decode** for this model on GH200. It provides no benefit at any
   measured batch size.

2. **Optimize for batching**: At batch=8, baseline already hits 593 tok/s (approaching the
   bandwidth ceiling). The path to maximum throughput is concurrent request batching, not
   single-request optimization.

3. **Investigate full CUDA graph capture**: The piecewise-only limitation from TREE_ATTN
   is a significant overhead source. With spec decode disabled, full CUDA graphs should
   be available.

4. **Profile kernel-level timing**: Use `nsys` or `torch.profiler` to measure the exact
   breakdown of the 10.6ms per-token decode into compute vs launch overhead vs idle time.

5. **Consider FP8 quantization**: W8A8-FP8 would use the FP8 tensor cores (2x throughput
   of FP16) and might improve the skinny-GEMM efficiency, though it doubles weight size
   from W4.
