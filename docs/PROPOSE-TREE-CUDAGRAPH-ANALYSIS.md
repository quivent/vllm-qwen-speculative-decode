# propose_tree() CUDA Graph Compatibility Analysis

**Date:** 2026-04-11
**File:** `vllm/v1/spec_decode/eagle.py` (lines 1033-1221)

## Verdict

**propose_tree() IS CUDA-graph compatible.** The method already contains
the correct dispatch plumbing. The 22 vs 186 tok/s gap is caused by the
config path rejecting the combination, not by graph-incompatible operations
in propose_tree itself.

## Evidence

### 1. No dynamic control flow that breaks CUDA graphs

The `for level in range(tree_depth - 1)` loop (line 1085) iterates a
**fixed** number of times determined by the tree structure at init. Within
each iteration:

- `torch.where`, `repeat_interleave`, `torch.cat`, `torch.topk`, `argmax`
  are all graph-safe ops (no host-dependent branching on tensor values).
- The `if level_num_drafts > 1` and `if num_children > 1` guards (lines
  1097, 1103) are **Python-level** conditionals on values fixed at init
  (`cu_drafts_per_level`, `child_drafts_per_level`). They do not change
  between calls. CUDA graphs capture the taken branch and replay it.
- `exceeds_max_model_len` (line 1088) produces a boolean mask used in
  `torch.where` and `masked_fill_` -- pure tensor ops, graph-safe.

### 2. Serial loop (line 555) vs propose_tree (line 1085): structurally identical dispatch

Serial loop:
```python
cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
    self._determine_batch_execution_and_padding(batch_size)
)
# ... then uses cudagraph_runtime_mode in set_forward_context
```

propose_tree:
```python
cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
    num_tokens
)
num_input_tokens = batch_desc.num_tokens
# ... then uses cudagraph_runtime_mode in set_forward_context
```

Both call `cudagraph_dispatcher.dispatch()` -- propose_tree calls it
directly, the serial loop wraps it in `_determine_batch_execution_and_padding`
(which adds DP coordination). The model forward call structure is identical:
copy into buffers, dispatch, `set_forward_context`, `self.model(...)`.

### 3. build_for_drafting() is graph-safe

`TreeAttentionMetadataBuilder.build_for_drafting()` (tree_attn.py:218):
- Slices `self.tree_attn_bias` (a fixed tensor computed at init)
- Calls `self.build(0, common_attn_metadata, fast_build=True)`
- `build()` constructs a `TreeAttentionMetadata` dataclass from fields
  already on the `common_attn_metadata` -- no CUDA mallocs, no host syncs

The `split_decodes_and_prefills` call inside `build()` reads
`query_start_loc_cpu`, which is a pre-existing CPU tensor (not a D2H
transfer). In the drafting path, `common_attn_metadata` is constructed
via `replace()` at line 1119, and the `query_start_loc_cpu` field carries
over from the parent metadata. This is the same behavior as `--enforce-eager`.

Note: `build_for_drafting` runs **outside** the CUDA graph capture/replay
boundary. It constructs metadata that is then passed to `set_forward_context`.
The CUDA graph only captures the model forward pass, not the metadata
construction.

### 4. cudagraph_dispatcher.dispatch() handles variable num_tokens

At each tree level, `num_tokens = batch_size * total_num_drafts` changes.
The dispatcher pads to the nearest capture size via
`_bs_to_padded_graph_size` and looks up a matching `BatchDescriptor` in
`cudagraph_keys[PIECEWISE]`. This is exactly how the serial loop handles
different batch sizes.

The dispatcher returns `CUDAGraphMode.NONE` if `num_tokens` exceeds
`max_cudagraph_capture_size`, which is a clean fallback, not a crash.

### 5. TreeAttentionImpl.forward() is graph-safe

The forward pass (tree_attn.py:363) calls `unified_attention` (a Triton
kernel) with the `qq_bias` parameter for the tree attention mask. This is a
static tensor. The kernel is graph-capturable.

## What blocks it: the config path

`initialize_cudagraph_keys()` (eagle.py:373) gates on
`self.speculative_config.enforce_eager`. When tree attention is selected
(via `speculative_token_tree`), the config validation elsewhere forces
`enforce_eager=True`, which causes `eagle_cudagraph_mode = CUDAGraphMode.NONE`.

The fix is in the config validation layer, not in propose_tree().

## Potential minor issue (not a blocker)

propose_tree calls `dispatch()` directly instead of
`_determine_batch_execution_and_padding()`. This means it skips DP
coordination (`coordinate_batch_across_dp`). For single-GPU this is fine.
For DP setups with tree attention, this would need the wrapper call instead.
This is a DP-specific concern, not a CUDA graph concern.

## Summary

| Question | Answer |
|---|---|
| Dynamic control flow? | No -- loop bounds and branches are init-time constants |
| build_for_drafting graph-safe? | Yes -- runs outside graph boundary, pure metadata construction |
| dispatch() handles tree batches? | Yes -- pads to capture sizes, same mechanism as serial |
| TreeAttentionImpl.forward() graph-safe? | Yes -- Triton kernel with static bias tensor |
| **What needs to change?** | **Config validation path only** -- remove the enforce_eager gate for tree attention |
