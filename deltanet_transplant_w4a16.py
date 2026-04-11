#!/usr/bin/env python3
"""W4A16-to-W4A16 DeltaNet weight transplant.

Copies quantized (qweight/qzeros/scales) and non-quantized weights
from the DeltaNet draft W4A16 model into the main W4A16 model checkpoint,
respecting the draft->main layer index mapping.

Skips in_proj_a/in_proj_b which have format mismatches (fp16 in main,
quantized in draft). These are small LoRA-style projections.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Draft layer i -> main layer MAIN_DN[i]
MAIN_DN = [i for i in range(64) if i % 4 != 3]  # 48 DeltaNet layers
assert len(MAIN_DN) == 48

# Suffixes that should NOT skip (skip in_proj_a, in_proj_b due to format mismatch)
SKIP_PATTERNS = {"in_proj_a.", "in_proj_b."}


def should_skip(suffix: str) -> bool:
    return any(p in suffix for p in SKIP_PATTERNS)


def build_copy_map(
    main_keys: set[str],
    draft_keys: set[str],
    include_mtp: bool = True,
) -> dict[str, str]:
    """Build main_key -> draft_key map for all copyable weights."""
    copy_map = {}  # main_key -> draft_key

    for draft_i, main_i in enumerate(MAIN_DN):
        draft_prefix = f"model.language_model.layers.{draft_i}."
        main_prefix = f"model.language_model.layers.{main_i}."

        for dk in draft_keys:
            if not dk.startswith(draft_prefix):
                continue
            suffix = dk[len(draft_prefix):]
            if should_skip(suffix):
                continue
            mk = main_prefix + suffix
            if mk in main_keys:
                copy_map[mk] = dk

    if include_mtp:
        for dk in draft_keys:
            if dk.startswith("mtp.") and dk in main_keys:
                copy_map[dk] = dk

    return copy_map


def merge(
    main_model_path: str,
    draft_model_path: str,
    output_path: str,
    include_mtp: bool = True,
) -> None:
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(main_model_path, "model.safetensors.index.json")) as f:
        main_index = json.load(f)
    with open(os.path.join(draft_model_path, "model.safetensors.index.json")) as f:
        draft_index = json.load(f)

    main_wm = main_index["weight_map"]
    draft_wm = draft_index["weight_map"]

    copy_map = build_copy_map(set(main_wm.keys()), set(draft_wm.keys()), include_mtp)
    logger.info("Will transplant %d weights", len(copy_map))

    # Pre-load all draft tensors we need, grouped by shard
    draft_shard_keys: dict[str, list[str]] = {}
    for mk, dk in copy_map.items():
        shard = draft_wm[dk]
        draft_shard_keys.setdefault(shard, []).append(dk)

    draft_tensors: dict[str, torch.Tensor] = {}
    for shard, keys in draft_shard_keys.items():
        path = os.path.join(draft_model_path, shard)
        logger.info("Loading %d draft weights from %s", len(keys), shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in keys:
                draft_tensors[k] = f.get_tensor(k)

    # Process main model shards
    main_shards = sorted(set(main_wm.values()))
    new_weight_map = {}
    replaced = 0

    for shard_name in main_shards:
        shard_keys = [k for k, v in main_wm.items() if v == shard_name]
        shard_path = os.path.join(main_model_path, shard_name)

        logger.info("Processing %s (%d keys)", shard_name, len(shard_keys))
        tensors_out = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in shard_keys:
                if key in copy_map:
                    dk = copy_map[key]
                    dt = draft_tensors[dk]
                    # Validate shape
                    orig_shape = tuple(f.get_slice(key).get_shape())
                    if dt.shape != torch.Size(list(orig_shape)):
                        logger.error(
                            "Shape mismatch: %s %s vs draft %s %s — skipping",
                            key, orig_shape, dk, tuple(dt.shape),
                        )
                        tensors_out[key] = f.get_tensor(key)
                    else:
                        tensors_out[key] = dt
                        replaced += 1
                else:
                    tensors_out[key] = f.get_tensor(key)
                new_weight_map[key] = shard_name

        out_path = os.path.join(output_path, shard_name)
        save_file(tensors_out, out_path)
        logger.info("Wrote %s", out_path)

    # Write index
    new_index = {
        "metadata": main_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy config/tokenizer files
    for fname in os.listdir(main_model_path):
        if fname.endswith((".json", ".jinja", ".txt", ".model")) and fname != "model.safetensors.index.json":
            src = os.path.join(main_model_path, fname)
            dst = os.path.join(output_path, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    logger.info(
        "Done: %d/%d weights transplanted. Output: %s",
        replaced, len(main_wm), output_path,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="W4A16 DeltaNet weight transplant")
    p.add_argument("--main-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-mtp", action="store_true")
    args = p.parse_args()

    merge(args.main_model, args.draft_model, args.output, include_mtp=not args.no_mtp)
