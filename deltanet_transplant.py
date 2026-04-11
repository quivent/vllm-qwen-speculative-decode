#!/usr/bin/env python3
"""DeltaNet Weight Transplant: copy dedicated draft model weights into the
main Qwen3.5-27B model's DeltaNet layer slots.

This script operates on a live vLLM model (in-place parameter swap) or can
produce a merged safetensors checkpoint on disk.

Usage (in-place, called from Python after vLLM model load):

    from deltanet_transplant import transplant_deltanet_weights
    transplant_deltanet_weights(
        target_model=runner.model,
        draft_model_path="/home/ubuntu/models/Qwen3.5-27B-DeltaNet-draft",
        include_mlp=True,
        include_layernorm=True,
        include_mtp=False,
    )

Usage (checkpoint merge, CLI):

    python deltanet_transplant.py \
        --main-model /home/ubuntu/models/Qwen3.5-27B \
        --draft-model /home/ubuntu/models/Qwen3.5-27B-DeltaNet-draft \
        --output /home/ubuntu/models/Qwen3.5-27B-transplanted \
        --include-mlp --include-layernorm --include-mtp
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------

# Main model: 64 layers, DeltaNet at every index except 3,7,11,...,63
# Draft model: 48 layers, all DeltaNet
# draft_idx -> main_idx: insert a gap every 3 layers for the full-attention layer
MAIN_DELTANET_INDICES = [
    i for i in range(64) if i % 4 != 3
]  # [0,1,2, 4,5,6, 8,9,10, ..., 60,61,62]

assert len(MAIN_DELTANET_INDICES) == 48

# Per-layer weight suffixes to transplant
LINEAR_ATTN_SUFFIXES = [
    "linear_attn.A_log",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
]

MLP_SUFFIXES = [
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
]

LAYERNORM_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
]

MTP_KEYS = [
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
]

# Draft layers that lack linear_attn.norm.weight (every 4th: 3,7,11,...,47)
DRAFT_LAYERS_MISSING_NORM = set(range(3, 48, 4))


def _build_transplant_map(
    include_mlp: bool = True,
    include_layernorm: bool = True,
) -> list[tuple[str, str]]:
    """Build (draft_key, main_key) pairs for all DeltaNet layer weights.

    Returns list of (draft_weight_name, main_weight_name) tuples.
    """
    pairs = []
    for draft_idx, main_idx in enumerate(MAIN_DELTANET_INDICES):
        suffixes = list(LINEAR_ATTN_SUFFIXES)
        if include_mlp:
            suffixes.extend(MLP_SUFFIXES)
        if include_layernorm:
            suffixes.extend(LAYERNORM_SUFFIXES)

        for suffix in suffixes:
            # Skip norm.weight for draft layers that don't have it
            if suffix == "linear_attn.norm.weight" and draft_idx in DRAFT_LAYERS_MISSING_NORM:
                continue
            draft_key = f"model.language_model.layers.{draft_idx}.{suffix}"
            main_key = f"model.language_model.layers.{main_idx}.{suffix}"
            pairs.append((draft_key, main_key))

    return pairs


def _load_draft_tensors(
    draft_model_path: str,
    needed_keys: set[str],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load specific tensors from draft model safetensors shards."""
    from safetensors import safe_open

    index_path = os.path.join(draft_model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Group by shard to minimize file opens
    shard_to_keys: dict[str, list[str]] = {}
    for key in needed_keys:
        if key not in weight_map:
            logger.warning("Draft model missing weight: %s (skipping)", key)
            continue
        shard = weight_map[key]
        shard_to_keys.setdefault(shard, []).append(key)

    tensors = {}
    for shard, keys in shard_to_keys.items():
        shard_path = os.path.join(draft_model_path, shard)
        logger.info("Loading %d weights from %s", len(keys), shard)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)

    return tensors


# ---------------------------------------------------------------------------
# In-place transplant (live vLLM model)
# ---------------------------------------------------------------------------

def transplant_deltanet_weights(
    target_model: torch.nn.Module,
    draft_model_path: str,
    include_mlp: bool = True,
    include_layernorm: bool = True,
    include_mtp: bool = False,
) -> dict[str, int]:
    """Transplant draft model DeltaNet weights into a live target model.

    Args:
        target_model: The loaded Qwen3.5 model (e.g., runner.model).
        draft_model_path: Path to the draft model directory.
        include_mlp: Also transplant MLP weights for DeltaNet layers.
        include_layernorm: Also transplant layernorm weights.
        include_mtp: Also transplant MTP head weights.

    Returns:
        Dict with counts: {"transplanted": N, "skipped": M, "shape_mismatch": K}
    """
    # Build the mapping
    pairs = _build_transplant_map(include_mlp, include_layernorm)

    # Add MTP keys if requested
    mtp_pairs = []
    if include_mtp:
        for key in MTP_KEYS:
            mtp_pairs.append((key, key))  # same name in both models

    all_pairs = pairs + mtp_pairs
    draft_keys_needed = {dk for dk, _ in all_pairs}

    # Load draft tensors to CPU
    logger.info("Loading %d draft model weights...", len(draft_keys_needed))
    draft_tensors = _load_draft_tensors(draft_model_path, draft_keys_needed)

    # Build target model state dict index
    target_state = dict(target_model.named_parameters())

    # Navigate through possible wrapper layers
    # vLLM may wrap as: model.language_model.layers.X...
    # or the parameter names might have a prefix
    # Try to find the right prefix
    sample_main_key = f"model.language_model.layers.0.linear_attn.A_log"
    prefix = ""
    if sample_main_key not in target_state:
        # Try common prefixes
        for p in ["language_model.", "model.", ""]:
            test_key = f"{p}layers.0.linear_attn.A_log"
            if test_key in target_state:
                prefix = p
                break
        else:
            # Search for it
            for name in target_state:
                if "layers.0.linear_attn.A_log" in name:
                    prefix = name.split("layers.0.linear_attn.A_log")[0]
                    logger.info("Auto-detected parameter prefix: '%s'", prefix)
                    break

    def _to_param_name(weight_name: str) -> str:
        """Convert safetensors weight name to model parameter name."""
        # Strip the 'model.language_model.' or similar prefix from the
        # safetensors name and prepend the detected runtime prefix
        if weight_name.startswith("model.language_model."):
            suffix = weight_name[len("model.language_model."):]
        elif weight_name.startswith("model."):
            suffix = weight_name[len("model."):]
        else:
            suffix = weight_name
        return f"{prefix}{suffix}"

    stats = {"transplanted": 0, "skipped": 0, "shape_mismatch": 0}

    for draft_key, main_key in all_pairs:
        if draft_key not in draft_tensors:
            stats["skipped"] += 1
            continue

        param_name = _to_param_name(main_key)
        if param_name not in target_state:
            # Try the raw key
            if main_key in target_state:
                param_name = main_key
            else:
                logger.warning("Target model missing param: %s (tried %s)", main_key, param_name)
                stats["skipped"] += 1
                continue

        param = target_state[param_name]
        draft_tensor = draft_tensors[draft_key]

        if param.shape != draft_tensor.shape:
            logger.error(
                "Shape mismatch for %s -> %s: %s vs %s",
                draft_key, param_name, draft_tensor.shape, param.shape,
            )
            stats["shape_mismatch"] += 1
            continue

        # In-place copy to the target parameter
        with torch.no_grad():
            param.data.copy_(draft_tensor.to(param.device, param.dtype))
        stats["transplanted"] += 1

    logger.info(
        "Transplant complete: %d weights copied, %d skipped, %d shape mismatches",
        stats["transplanted"], stats["skipped"], stats["shape_mismatch"],
    )
    return stats


# ---------------------------------------------------------------------------
# Checkpoint merge (offline)
# ---------------------------------------------------------------------------

def merge_checkpoint(
    main_model_path: str,
    draft_model_path: str,
    output_path: str,
    include_mlp: bool = True,
    include_layernorm: bool = True,
    include_mtp: bool = False,
) -> None:
    """Create a merged checkpoint: main model with draft DeltaNet weights.

    Reads the main model shards, replaces DeltaNet layer weights with draft
    model weights, and writes new safetensors shards to output_path.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(output_path, exist_ok=True)

    # Load indices
    with open(os.path.join(main_model_path, "model.safetensors.index.json")) as f:
        main_index = json.load(f)

    pairs = _build_transplant_map(include_mlp, include_layernorm)
    mtp_pairs = []
    if include_mtp:
        for key in MTP_KEYS:
            mtp_pairs.append((key, key))
    all_pairs = pairs + mtp_pairs

    # Build replacement map: main_key -> draft_key
    replacement_map = {main_key: draft_key for draft_key, main_key in all_pairs}
    draft_keys_needed = {dk for dk, _ in all_pairs}

    logger.info("Loading draft model weights...")
    draft_tensors = _load_draft_tensors(draft_model_path, draft_keys_needed)

    # Group main weights by shard
    main_weight_map = main_index["weight_map"]
    shards = sorted(set(main_weight_map.values()))

    new_weight_map = {}
    replaced_count = 0

    for shard_name in shards:
        shard_path = os.path.join(main_model_path, shard_name)
        keys_in_shard = [k for k, v in main_weight_map.items() if v == shard_name]

        logger.info("Processing %s (%d weights)", shard_name, len(keys_in_shard))

        tensors_out = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys_in_shard:
                if key in replacement_map:
                    draft_key = replacement_map[key]
                    if draft_key in draft_tensors:
                        tensors_out[key] = draft_tensors[draft_key]
                        replaced_count += 1
                    else:
                        tensors_out[key] = f.get_tensor(key)
                else:
                    tensors_out[key] = f.get_tensor(key)
                new_weight_map[key] = shard_name

        out_shard_path = os.path.join(output_path, shard_name)
        save_file(tensors_out, out_shard_path)
        logger.info("Wrote %s", out_shard_path)

    # Write new index
    new_index = {
        "metadata": main_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    index_out = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_out, "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy config and tokenizer files
    for fname in os.listdir(main_model_path):
        if fname.endswith(".json") and fname != "model.safetensors.index.json":
            src = os.path.join(main_model_path, fname)
            dst = os.path.join(output_path, fname)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
        if fname.endswith((".jinja", ".txt")):
            src = os.path.join(main_model_path, fname)
            dst = os.path.join(output_path, fname)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

    logger.info(
        "Merge complete: %d weights replaced out of %d total. Output: %s",
        replaced_count, len(main_weight_map), output_path,
    )


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------

def validate_shapes(
    main_model_path: str,
    draft_model_path: str,
    include_mlp: bool = True,
    include_layernorm: bool = True,
    include_mtp: bool = False,
) -> bool:
    """Validate that all transplant weight shapes match without loading full tensors."""
    from safetensors import safe_open

    with open(os.path.join(main_model_path, "model.safetensors.index.json")) as f:
        main_index = json.load(f)
    with open(os.path.join(draft_model_path, "model.safetensors.index.json")) as f:
        draft_index = json.load(f)

    pairs = _build_transplant_map(include_mlp, include_layernorm)
    if include_mtp:
        for key in MTP_KEYS:
            pairs.append((key, key))

    # Cache opened files to avoid re-opening
    main_shapes: dict[str, tuple] = {}
    draft_shapes: dict[str, tuple] = {}

    def _get_shape(model_path: str, index: dict, key: str, cache: dict) -> Optional[tuple]:
        if key in cache:
            return cache[key]
        if key not in index["weight_map"]:
            return None
        shard = os.path.join(model_path, index["weight_map"][key])
        with safe_open(shard, framework="pt") as f:
            shape = tuple(f.get_slice(key).get_shape())
            cache[key] = shape
            return shape

    ok = True
    for draft_key, main_key in pairs:
        ds = _get_shape(draft_model_path, draft_index, draft_key, draft_shapes)
        ms = _get_shape(main_model_path, main_index, main_key, main_shapes)
        if ds is None:
            print(f"  SKIP (missing in draft): {draft_key}")
            continue
        if ms is None:
            print(f"  SKIP (missing in main):  {main_key}")
            continue
        if ds != ms:
            print(f"  MISMATCH: {draft_key} {ds} != {main_key} {ms}")
            ok = False
        else:
            print(f"  OK: {draft_key} {ds}")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="DeltaNet weight transplant: merge dedicated draft model "
                    "weights into main model checkpoint",
    )
    parser.add_argument("--main-model", required=True, help="Path to main Qwen3.5-27B model")
    parser.add_argument("--draft-model", required=True, help="Path to DeltaNet draft model")
    parser.add_argument("--output", help="Output path for merged checkpoint (omit for dry-run)")
    parser.add_argument("--include-mlp", action="store_true", default=True,
                        help="Transplant MLP weights (default: yes)")
    parser.add_argument("--no-mlp", action="store_true", help="Skip MLP weights")
    parser.add_argument("--include-layernorm", action="store_true", default=True,
                        help="Transplant layernorm weights (default: yes)")
    parser.add_argument("--no-layernorm", action="store_true", help="Skip layernorm weights")
    parser.add_argument("--include-mtp", action="store_true",
                        help="Also transplant MTP head weights")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate shapes, don't merge")

    args = parser.parse_args()

    include_mlp = not args.no_mlp
    include_layernorm = not args.no_layernorm

    if args.validate_only:
        print("Validating shapes...")
        ok = validate_shapes(
            args.main_model, args.draft_model,
            include_mlp, include_layernorm, args.include_mtp,
        )
        print(f"\nValidation: {'PASS' if ok else 'FAIL'}")
        return

    if not args.output:
        print("No --output specified. Running shape validation only.")
        ok = validate_shapes(
            args.main_model, args.draft_model,
            include_mlp, include_layernorm, args.include_mtp,
        )
        print(f"\nValidation: {'PASS' if ok else 'FAIL'}")
        print("Specify --output to create merged checkpoint.")
        return

    merge_checkpoint(
        args.main_model, args.draft_model, args.output,
        include_mlp, include_layernorm, args.include_mtp,
    )


if __name__ == "__main__":
    main()
