# Two-Graph CUDA Dispatch for Partial-Layer Verification

**Status:** Design proposal
**Date:** 2026-04-10
**Target:** vLLM v1 speculative decoding verification path
**Platform:** GH200 (Grace Hopper, unified memory, NVLink-C2C)

---

## 1. Problem

vLLM v1 uses CUDA graph capture for inference. A CUDA graph records a
fixed kernel sequence at capture time and replays it at runtime with
zero CPU-side launch overhead. The critical constraint: **CUDA graphs
cannot contain conditional branches.** Once captured, the kernel DAG
is immutable.

Partial-layer verification (PLV) requires a conditional:

> "If early-exit logits from layer N agree with all draft tokens,
> skip layers N+1 through 63. Otherwise, run the full model."

This conditional cannot exist inside a single CUDA graph. The graph
must either run N layers or 64 layers -- it cannot decide at runtime.

Running eager mode (no CUDA graphs) to get the branch is possible but
surrenders the latency advantage of graph replay, which eliminates
kernel launch overhead entirely. On a 64-layer model with ~200+
kernels per forward pass, eager mode adds 50-150us of CPU dispatch
overhead per verification step.

---

## 2. Solution: Two-Graph Dispatch

Capture two CUDA graphs at startup:

```
graph_partial:  layers[0..N-1] -> RMSNorm -> lm_head projection
graph_full:     layers[0..63]  -> RMSNorm -> lm_head projection
```

At runtime, the conditional lives on the CPU between two graph
replays:

```
                 GPU                          CPU
                  |                            |
  graph_partial.replay()                      |
        |                                     |
        +--- D2H: argmax results ------------>|
                                              |  compare early argmax
                                              |  vs draft token IDs
                                              |
                              all agree? -----+
                             /          \
                           yes           no
                            |             |
                         accept      graph_full.replay()
                         tokens            |
                            |         use full logits for
                            |         rejection sampling
                            |              |
                            +----- next step ------+
```

The branch happens on CPU. Each arm dispatches (or skips) a
pre-captured CUDA graph. No eager-mode penalty.

---

## 3. Architecture Diagram

```
+------------------------------------------------------------------+
|  gpu_model_runner.execute_model()                                |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |  _determine_batch_execution_and_padding()                  |  |
|  |  -> cudagraph_dispatcher.dispatch()                        |  |
|  |  -> returns CUDAGraphMode + BatchDescriptor                |  |
|  +------------------------------------------------------------+  |
|                           |                                      |
|                           v                                      |
|  +------------------------------------------------------------+  |
|  |  set_forward_context(cudagraph_runtime_mode=...)           |  |
|  |  _model_forward(input_ids, positions, ...)                 |  |
|  |                                                            |  |
|  |    CUDAGraphWrapper.__call__()                             |  |
|  |      if mode matches -> replay cached graph                |  |
|  |      if no entry     -> capture new graph                  |  |
|  |      if mode=NONE    -> eager runnable()                   |  |
|  +------------------------------------------------------------+  |
|                           |                                      |
|                           v                                      |
|  +------------------------------------------------------------+  |
|  |  PLV TWO-GRAPH INSERTION POINT                  [NEW]      |  |
|  |                                                            |  |
|  |  Phase 1: dispatch graph_partial                           |  |
|  |    - Forward through layers 0..N-1, norm, lm_head          |  |
|  |    - D2H copy: early_argmax -> pinned CPU tensor           |  |
|  |    - torch.cuda.current_stream().synchronize()             |  |
|  |                                                            |  |
|  |  Phase 2: CPU comparison                                   |  |
|  |    - Compare early_argmax vs draft_token_ids               |  |
|  |    - If all match: accept, skip graph_full                 |  |
|  |    - If any mismatch: dispatch graph_full                  |  |
|  |                                                            |  |
|  |  Phase 3: (conditional) dispatch graph_full                |  |
|  |    - Forward through all 64 layers, norm, lm_head          |  |
|  |    - Use full logits for standard rejection sampling       |  |
|  +------------------------------------------------------------+  |
|                           |                                      |
|                           v                                      |
|  hidden_states[logits_indices] -> compute_logits -> sampling     |
+------------------------------------------------------------------+
```

---

## 4. Execution Flow (Step by Step)

### 4.1 Startup: Graph Capture

During `capture_model()` (gpu_model_runner.py:5972), vLLM iterates
over `cudagraph_dispatcher.get_capture_descs()` and calls
`_warmup_and_capture()` for each `(CUDAGraphMode, BatchDescriptor)`.

For PLV, we add a second capture pass:

1. **Capture `graph_full`** -- the standard graph. This is what vLLM
   already captures. No change needed.

2. **Capture `graph_partial`** -- a new graph that runs only layers
   0 through N-1, then norm + lm_head. This requires:
   - A model wrapper or submodel that truncates after layer N-1
   - Capturing it under a distinct `BatchDescriptor` (e.g., with a
     `partial=True` flag) or under a separate `CUDAGraphWrapper`
     instance

The capture loop in `_capture_cudagraphs()` (gpu_model_runner.py:6122)
already iterates batch descriptors and delegates to
`_warmup_and_capture()` (gpu_model_runner.py:6088). The pattern for
adding a second graph set is established by the `modal_mtp`
draft-mode capture at line 6031-6051, which captures a separate set
of graphs with `draft_mode=True` descriptors.

### 4.2 Runtime: Verification Step

When a verification batch arrives in `execute_model()`
(gpu_model_runner.py:3778):

1. **Preprocessing** -- unchanged. `_update_states()`,
   `_prepare_inputs()`, attention metadata built as usual.

2. **Dispatch `graph_partial`:**
   - Set forward context with the partial-graph batch descriptor
   - Call `_model_forward()` (gpu_model_runner.py:3499) which
     invokes `self.model(...)` -- the `CUDAGraphWrapper` at the
     outermost level
   - `CUDAGraphWrapper.__call__()` (cuda_graph.py:233) checks
     `forward_context.batch_descriptor`, finds the partial entry,
     calls `entry.cudagraph.replay()` (cuda_graph.py:355)
   - Output: `early_hidden_states` from layer N-1

3. **Norm + lm_head on early hidden states:**
   - `model.compute_logits(early_hidden_states[logits_indices])`
   - Produces `early_logits` tensor on GPU

4. **D2H sync:**
   - `early_argmax = early_logits.argmax(dim=-1)` (GPU kernel)
   - Copy to pinned CPU memory:
     `early_argmax_cpu = early_argmax.to('cpu', non_blocking=False)`
   - This is the synchronization point. On GH200 with unified
     memory / NVLink-C2C, this is ~5-10us for a small tensor
     (K draft tokens, typically K=5).

5. **CPU comparison:**
   - `match = torch.equal(early_argmax_cpu, draft_token_ids_cpu)`
   - This is a handful of int comparisons. Negligible latency.

6. **Branch:**
   - **If match:** Accept all draft tokens. No `graph_full` needed.
     Return early. Net savings: full-model latency minus
     partial-model latency minus D2H sync.
   - **If no match:** Dispatch `graph_full`. Set forward context
     with full-graph batch descriptor. Call `_model_forward()`
     again. Use full logits for rejection sampling as usual. Net
     cost: partial-model latency + D2H sync (overhead vs baseline).

### 4.3 Fallback

If `graph_partial` is not available for a given batch size (e.g.,
batch size exceeds `max_cudagraph_capture_size`), fall back to
`graph_full` only -- standard verification. The dispatcher already
returns `CUDAGraphMode.NONE` for uncaptured sizes
(cudagraph_dispatcher.py:280).

---

## 5. Memory Overhead Analysis

### 5.1 CUDA Graph Memory

Each captured CUDA graph allocates memory proportional to:
- All intermediate activation tensors alive during the forward pass
- All kernel parameters and workspace buffers

For a single batch size, `graph_partial` (N layers) uses roughly
`N/64` of the memory of `graph_full` (64 layers). However, CUDA
graph memory pools are shared via `get_global_graph_pool()`
(cuda_graph.py:200), and vLLM captures graphs largest-first so
smaller graphs can reuse the pool (cudagraph_dispatcher.py:340-347).

**Estimated overhead per batch size:**

| Component | graph_full (64L) | graph_partial (8L) | Delta |
|-----------|------------------|--------------------|-------|
| Activations | ~2.0 GB | ~0.25 GB | +0.25 GB |
| Graph metadata | ~50 MB | ~10 MB | +10 MB |
| **Total per BS** | ~2.05 GB | ~0.26 GB | **~0.26 GB** |

With the shared graph pool, `graph_partial` mostly reuses memory
already allocated for `graph_full`. Realistic overhead: **~200-500 MB
total** across all captured batch sizes, not 2x.

### 5.2 Pinned Memory

One small pinned-memory buffer for the D2H argmax transfer:
`K * sizeof(int64)` where K = number of draft tokens (typically 5).
This is 40 bytes. Negligible.

---

## 6. Latency Analysis

### 6.1 D2H Synchronization Cost

The D2H copy is the critical overhead when the early exit succeeds.

**On GH200 (NVLink-C2C, 900 GB/s bidirectional):**
- Payload: K * 8 bytes = 40 bytes (K=5 draft tokens)
- Theoretical transfer: <1us
- Synchronization overhead (stream sync): 3-8us
- **Total D2H latency: ~5-10us with pinned memory**

For comparison, a single transformer layer on a 70B model takes
~500-800us on GH200. The sync cost is <2% of one layer.

**On discrete GPU (PCIe 5.0, ~64 GB/s):**
- Same payload, but PCIe latency floor: ~5us
- Synchronization overhead: ~10-15us
- **Total D2H latency: ~15-25us**

Still small relative to layer compute.

### 6.2 End-to-End Latency Scenarios

Assume: 64-layer model, N=8 early-exit layers, K=5 draft tokens.

| Scenario | graph_partial | D2H sync | CPU cmp | graph_full | Total | vs Baseline |
|----------|--------------|----------|---------|------------|-------|-------------|
| Early exit hits (all match) | 4ms | 0.01ms | ~0 | skip | **4.01ms** | **-87.5%** |
| Early exit misses | 4ms | 0.01ms | ~0 | 32ms | **36.01ms** | **+12.5%** |
| Baseline (graph_full only) | -- | -- | -- | 32ms | **32ms** | -- |

**Break-even hit rate:** If P is the probability of early exit:
```
E[latency] = P * 4.01 + (1-P) * 36.01
Baseline   = 32.0

Break-even: 32 = P * 4.01 + (1-P) * 36.01
             32 = 36.01 - 32.0 * P
              P = 4.01 / 32.0 = 12.5%
```

At >12.5% early-exit hit rate, two-graph dispatch is a net win.
Empirical PLV hit rates on coding/reasoning workloads are typically
40-70%, making this strongly favorable.

---

## 7. Integration Points in vLLM Source

All paths reference the installed vLLM at:
`/home/ubuntu/.local/lib/python3.10/site-packages/vllm/`

### 7.1 CudagraphDispatcher (cudagraph_dispatcher.py)

| Line | What | Change |
|------|------|--------|
| 15-69 | `CudagraphDispatcher.__init__()` | Add `partial_layer_count` config; initialize keys for partial graphs |
| 44-47 | `cudagraph_keys` dict | Add entries for partial-mode descriptors (new `BatchDescriptor` flag or separate key set) |
| 165-232 | `initialize_cudagraph_keys()` | Generate partial-graph keys alongside full-graph keys |
| 234-323 | `dispatch()` | Add dispatch path: when verification batch, return partial-mode descriptor first |
| 325-349 | `get_capture_descs()` | Include partial-mode descriptors in capture list |

### 7.2 CUDAGraphWrapper (compilation/cuda_graph.py)

| Line | What | Change |
|------|------|--------|
| 145-168 | `CUDAGraphWrapper` class | Either use a second instance for partial graphs (like modal_mtp pattern) or extend entry keying |
| 233-256 | `__call__()` | No change needed if using separate wrapper instance. If sharing, dispatch by batch descriptor partial flag |
| 283-339 | Capture path | Capture partial-model graph. The `runnable` must be the truncated model |
| 354-356 | Replay path | `entry.cudagraph.replay()` -- no change, works for any captured graph |

### 7.3 GPUModelRunner (v1/worker/gpu_model_runner.py)

| Line | What | Change |
|------|------|--------|
| 3499-3529 | `_model_forward()` | Override in PLV subclass to implement two-phase dispatch |
| 3552-3618 | `_determine_batch_execution_and_padding()` | For verification batches, return partial-mode descriptor |
| 3778-4048 | `execute_model()` | After `_model_forward()` returns early hidden states (line 4042-4048), insert D2H sync + CPU comparison + conditional graph_full dispatch |
| 4077-4078 | `compute_logits()` call | Split into two: early logits from partial, full logits from full (conditional) |
| 5972-6027 | `capture_model()` | Add capture pass for partial graphs (follow modal_mtp pattern at lines 6031-6051) |
| 6088-6120 | `_warmup_and_capture()` | Warmup partial model before capture |
| 6122-6167 | `_capture_cudagraphs()` | Iterate partial-mode batch descriptors |

### 7.4 Model Truncation

The partial graph needs a truncated forward pass. Options:

1. **Model wrapper** that intercepts `forward()` and returns after
   layer N-1 + norm + lm_head. Similar to how `modal_mtp` uses
   `set_draft_mode(True)` (gpu_model_runner.py:6037).

2. **Hook-based truncation** using `register_forward_hook()` on
   layer N-1 to capture intermediate hidden states and short-circuit.

Option 1 is cleaner and follows existing vLLM patterns.

---

## 8. Risk Analysis

### 8.1 Correctness Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Early-exit logits disagree with full-model logits even when argmax matches | Low | Argmax comparison is sufficient for accept/reject. Full rejection sampling only runs on the full-model path. |
| KV cache inconsistency between partial and full runs | High | graph_partial writes KV entries for layers 0..N-1. If graph_full runs, it must NOT re-compute layers 0..N-1 (cache already written). Either: (a) graph_full starts from layer N using cached KV, or (b) graph_full re-runs all layers and overwrites. Option (b) is simpler and wastes ~12.5% compute. |
| Attention metadata mismatch between two graph dispatches in same step | Medium | Both graphs use the same batch descriptor for attention. Slot mappings and KV cache pointers are identical. Only the layer range differs. |
| Draft token comparison uses stale CPU data | Low | D2H sync is explicit and blocking. No race. |

### 8.2 Performance Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Low early-exit hit rate makes two-graph slower than baseline | Medium | Monitor hit rate. Disable PLV dynamically if hit rate < 15% over a window. |
| Graph capture OOM with two graph sets | Medium | Partial graphs are ~1/8 the size of full graphs. Use lazy capture (only capture on first use). |
| Stream synchronization overhead on discrete GPUs | Low | Even on PCIe, D2H for 40 bytes is <25us. Negligible vs layer compute. |
| Interference with existing spec decode (Eagle, ngram) | Medium | PLV verification is orthogonal to draft generation. Integration must ensure draft model's own CUDA graphs are unaffected. |

### 8.3 Engineering Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| vLLM upstream changes to CUDAGraphWrapper break PLV | High | PLV should subclass or wrap, not monkeypatch. Follow the modal_mtp precedent (separate graph set, new descriptor flag). |
| Complexity of two forward passes per verification step | Medium | Encapsulate in a `PLVModelRunner` subclass of `GPUModelRunner`. Override `_model_forward()` only. |
| LoRA interaction with partial graphs | Low | Partial graphs use the same LoRA dispatch as full. No special handling. |

---

## 9. Comparison with Alternatives

### 9.1 Eager Mode (No CUDA Graphs)

```
Approach: Run verification in eager mode. Branch natively in Python.
```

| Dimension | Eager Mode | Two-Graph Dispatch |
|-----------|-----------|-------------------|
| Implementation complexity | Low | Medium |
| Kernel launch overhead | +50-150us per step (CPU dispatch for ~200 kernels) | ~0 (graph replay) |
| Branching flexibility | Full (any Python conditional) | Binary (partial vs full) |
| Memory overhead | None (no graph capture) | +200-500 MB |
| Latency (early exit) | ~4.1ms (partial layers + dispatch overhead) | ~4.01ms |
| Latency (full model) | ~32.1ms | ~36.01ms (partial + full) |
| Break-even hit rate | 0% (always same or better than baseline eager) | ~12.5% |

**Verdict:** Eager mode is simpler but surrenders graph-replay
speedup for the full-model path. On GH200 where kernel launch
overhead is a larger fraction of total time (fast compute, same CPU),
this matters. Two-graph dispatch preserves graph replay for both
paths.

### 9.2 Soft Early-Exit Probe (No Branch)

```
Approach: Always run all 64 layers. Compute early-exit logits as a
side channel from layer N. Use them for cheap confidence estimation
but never skip layers.
```

| Dimension | Soft Probe | Two-Graph Dispatch |
|-----------|-----------|-------------------|
| Implementation complexity | Low (add a probe head, no graph changes) | Medium |
| Latency savings | None (always runs full model) | Up to 87.5% on hits |
| Use case | Confidence scoring, adaptive draft length | Actual compute savings |
| CUDA graph compatibility | Full (single graph, no branch) | Requires two graphs |
| Memory overhead | Probe head weights (~50 MB) | +200-500 MB |

**Verdict:** Soft probe is useful for confidence estimation but
provides zero latency savings. It complements rather than replaces
two-graph dispatch. A hybrid approach: use the soft probe's
confidence to decide whether to even attempt the partial-graph path.

### 9.3 CUDA Graph with Conditional Nodes (CUDA 12.4+)

```
Approach: Use CUDA graph conditional nodes (cudaGraphConditionalHandle)
to embed the branch inside a single graph.
```

| Dimension | Conditional Nodes | Two-Graph Dispatch |
|-----------|------------------|-------------------|
| Implementation complexity | High (new CUDA API, no PyTorch support) | Medium |
| CUDA version requirement | 12.4+ | Any (CUDA 11.x+) |
| PyTorch support | None as of PyTorch 2.6 | Full |
| vLLM support | None | Follows existing patterns |
| Latency overhead | Zero (branch inside graph) | D2H sync ~5-10us |
| Portability | NVIDIA only, recent drivers | Any CUDA GPU |

**Verdict:** Conditional graph nodes are the theoretically optimal
solution but have no PyTorch or vLLM support path today. Two-graph
dispatch is implementable now with existing APIs.

---

## 10. Implementation Sequence

1. **Model truncation wrapper** -- create `PartialModelWrapper` that
   runs layers 0..N-1 + norm + lm_head. Test in eager mode.

2. **Extend `BatchDescriptor`** -- add `partial: bool` field (or use
   a parallel descriptor namespace). Update `CudagraphDispatcher` to
   generate and dispatch partial keys.

3. **Capture partial graphs** -- add capture pass in
   `capture_model()` following the `modal_mtp` pattern.

4. **Two-phase `_model_forward()`** -- override in
   `PLVModelRunner(GPUModelRunner)`. Implement dispatch-sync-compare-
   dispatch flow.

5. **Benchmarking** -- measure D2H sync cost, early-exit hit rates,
   and end-to-end throughput on GH200 with DeepSeek-V3/R1.

6. **Adaptive control** -- add hit-rate monitor. Disable PLV when
   hit rate falls below threshold.
