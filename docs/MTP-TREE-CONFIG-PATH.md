# MTP + TREE_ATTN Config Path Analysis

Investigation: does `speculative_token_tree` survive the MTP config flow in
vLLM 0.19.0's `SpeculativeConfig.__post_init__`?

## Verdict

**The hypothesis is wrong.** The `speculative_token_tree` parameter IS correctly
preserved through the MTP self-speculative path. The `"Unsupported speculative
method: 'mtp'"` error has a different cause (see below).

## MTP self-speculative flow (method="mtp", model=None)

File: `vllm/config/speculative.py` (installed at
`~/.local/lib/python3.10/site-packages/vllm/config/speculative.py`)

1. **Line 371-375**: If method is None and model is not "ngram", method defaults
   to `"draft_model"`. If user passes `method="mtp"`, it stays `"mtp"`.

2. **Line 377-381**: Deprecated MTP model types (e.g. `"deepseek_mtp"`) collapse
   to `"mtp"`.

3. **Line 386-404** (`if self.model is None and num_speculative_tokens is not None`):
   For `method == "mtp"`, sets `self.model = self.target_model_config.model`
   (self-speculative: draft = target). Sets quantization to match target.

4. **Lines 424-500** (elif chain): method `"mtp"` does NOT match `"ngram"`,
   `"modal_mtp"`, `"suffix"`, or `"extract_hidden_states"`.

5. **Line 501** (`else` block): MTP lands here. Since `self.model` was set at
   step 3, it enters the `if self.model is not None` branch at line 507.

6. **Lines 511-539**: `is_standalone_draft` is False (model == target), so
   `_skip_mtp_conversion` stays False. `ModelConfig(...)` is created with
   `hf_overrides=SpeculativeConfig.hf_config_override`, which converts e.g.
   `qwen3_5` -> `qwen3_5_mtp`.

7. **Line 526**: Auto-detection: `draft_model_config.hf_config.model_type` (now
   `qwen3_5_mtp`) IS in `get_args(MTPModelTypes)`. Method confirmed as `"mtp"`.

8. **Lines 639-656**: Tree processing.
   - If `speculative_token_tree is None`: generates linear chain
     `[(0,), (0,0), ..., (0,...,0)]`.
   - If user provided it: parses with `ast.literal_eval`, sorts breadth-first.
   - **The tree IS preserved. It is NOT overwritten.**

## Runtime dispatch

- `use_eagle()` (line 881) returns `True` for `method == "mtp"`.
- Model runner (line 554): `use_eagle()` -> creates `EagleProposer`.
- `EagleProposer.__init__` (eagle.py:270-272) reads `speculative_token_tree`
  from config and parses it into `self.tree_choices`.
- `TreeAttentionBackend.__init__` (tree_attn.py:164-178) reads
  `speculative_token_tree` from `vllm_config.speculative_config` and builds
  `tree_attn_bias` matrix.

Linear chain tree `[(0,), (0,0), ..., (0,0,0,0,0,0,0)]` produces a causal
attention mask equivalent to standard sequential draft verification.

## When "Unsupported speculative method: 'mtp'" DOES fire

The error at line 553 triggers when ALL of these are true:

1. Execution reaches the else block (line 501) -- method is not ngram,
   modal_mtp, suffix, or extract_hidden_states.
2. `self.model is not None` (line 507).
3. The auto-detection chain (lines 544-551) fails to match.
4. `self.method != "draft_model"` (line 550).

**Concrete scenario**: user passes `method="mtp"` with a DIFFERENT model
(standalone draft, not self-speculative). Then:

- `self.model` is already set (not None), so line 386 block is skipped entirely.
- In the else block, `is_standalone_draft = True` (model != target).
- `_skip_mtp_conversion = True`, so `hf_config_override` does NOT convert
  `qwen3_5` -> `qwen3_5_mtp`.
- At line 526, `"qwen3_5"` is NOT in `MTPModelTypes` (only `"qwen3_5_mtp"` is).
- Falls through to line 553: `"Unsupported speculative method: 'mtp'"`.

This is the standalone-draft-model + explicit-mtp-method path. It is NOT the
self-speculative path.

## modal_mtp vs mtp tree handling

| Aspect | method="mtp" | method="modal_mtp" |
|---|---|---|
| Tree set at | Line 648 (else block) | Line 467 (modal_mtp block) |
| User tree sorted? | Yes (breadth-first, line 654) | No (only auto-generates) |
| Proposer | EagleProposer | ModalMTPProposer |
| Attn backend | TREE_ATTN reads tree | TREE_ATTN reads tree |

`modal_mtp` always generates a linear chain and ignores user-provided trees.
`mtp` respects user-provided trees and sorts them.

## Summary

- TREE_ATTN with linear-chain MTP works. The config path is sound.
- `speculative_token_tree` flows through: JSON -> SpeculativeConfig field ->
  `__post_init__` preserves/sorts it -> EagleProposer reads it ->
  TreeAttentionBackend builds mask from it.
- The "Unsupported speculative method" error is unrelated to TREE_ATTN. It
  fires when passing a standalone draft model path with `method="mtp"` for a
  model whose `hf_config.model_type` doesn't auto-detect as an MTP type after
  `_skip_mtp_conversion` blocks the conversion.
- No fix needed for tree processing. If the error is being hit, the fix is in
  the auto-detection logic or in ensuring the correct model/method combination
  is passed.
