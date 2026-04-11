# CUDA Graph Weight Swap: Feasibility for NativeMultiHeadProposer

**Date:** 2026-04-11
**Status:** Feasible -- empirically verified

## Question

The NativeMultiHeadProposer swaps MTP head weights between K sibling
heads and calls `propose()` through vLLM's CUDA-graphed path each time.
Does in-place weight copy (`w.copy_(new_values)`) produce correct results
when replaying a captured CUDA graph?

## Answer

**Yes.** CUDA graphs capture kernel launches and memory **pointers**, not
memory **values**. In-place weight swaps at the same pointer are visible
on replay.

## Evidence

### 1. PyTorch CUDA graph semantics

From the PyTorch CUDA graph contract:

- `torch.cuda.CUDAGraph.capture()` records the sequence of CUDA kernel
  launches and the **device pointers** they reference.
- On `replay()`, the same kernels execute against the same pointers.
  Whatever data lives at those pointers at replay time is what the
  kernels read.
- Static tensors (weights, buffers) are not snapshotted -- only their
  addresses are baked into the graph.

### 2. Empirical test -- raw CUDA graph

```
w = torch.randn(1000, 1000, device='cuda')
x = torch.randn(1, 1000, device='cuda')
out = torch.empty(1, 1000, device='cuda')

# warmup cuBLAS, then capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out.copy_(x @ w.T)

g.replay()
result1 = out.clone()

w.copy_(torch.randn(1000, 1000, device='cuda'))   # in-place swap
g.replay()
result2 = out.clone()
```

Results:
- `torch.allclose(result1, result2)` = **False** (swap is visible)
- `torch.allclose(result2, x @ w.T)` = **True** (result is correct)

### 3. Empirical test -- torch.compile (reduce-overhead / inductor)

```python
@torch.compile(mode="reduce-overhead")
def forward(x, w):
    return x @ w.T
```

After warmup, in-place `w.copy_(...)` between calls:
- Compiled graph sees the new weights: **True**
- Result matches direct computation: **True**

Inductor does not cache weight values in the compiled artifact. It
records the tensor metadata (shape, stride, dtype, device, pointer) and
re-reads from the live pointer on each invocation.

### 4. vLLM CUDAGraphWrapper analysis

File: `vllm/compilation/cuda_graph.py`

Key observations:

- **Capture** (lines 308-314): `torch.cuda.graph(cudagraph)` wraps
  `self.runnable(*args, **kwargs)`. The runnable is the model forward.
  Model weights are not passed as explicit `args` -- they live on
  `self.runnable` (the model's `nn.Module` parameters).

- **No weight snapshotting**: The wrapper records `input_addresses`
  (line 279-281) for **args** only (activations / hidden states), not
  for model parameters. Weights are accessed via the module's parameter
  tensors, which are static device pointers.

- **Replay** (line 355): `entry.cudagraph.replay()` replays the
  captured kernels. Weight tensors are read from their original device
  pointers. If you wrote new values to those pointers via `.copy_()`,
  the replay reads the new values.

- **Debug check** (lines 341-349): Only validates that *input*
  (activation) addresses match. No check on weight addresses -- they
  are assumed immutable. This is fine for the swap pattern because
  the pointers stay the same; only the data changes.

## Constraint: pointer stability

The one hard requirement is that the weight tensor's **storage pointer
must not change**. Operations that preserve the pointer:

- `w.copy_(new_data)` -- YES, safe
- `w.data.copy_(new_data)` -- YES, safe
- `w[:] = new_data` -- YES, safe (calls `copy_` under the hood)

Operations that may **reallocate** and break the graph:

- `w = new_tensor` -- NO, rebinds the Python name to a new pointer
- `w.data = new_tensor` -- NO, replaces the storage
- `del w; w = ...` -- NO

## Conclusion for NativeMultiHeadProposer

The K-head weight-swap approach is feasible:

1. At init, allocate one set of weight tensors for the MTP head.
2. Before each `propose()` call for head k, do
   `mtp_head.lm_head.weight.copy_(head_weights[k])`.
3. Replay the CUDA graph. The graph reads the new weight values.

The cost is one `copy_` per head per propose step (a device-to-device
memcpy). For typical MTP head sizes (vocab_size x hidden_dim, e.g.
152K x 4096 = ~600MB in fp16), this takes ~0.1-0.3ms on modern GPUs --
small relative to the forward pass itself.

No CUDA graph re-capture is needed. No torch.compile invalidation
occurs. The approach is sound.
