#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MTP Head Cloner for Qwen 3.5-27B Sibling Speculative Decoding.

Loads the single trained MTP head from the original Qwen 3.5-27B checkpoint,
clones it K times with configurable Gaussian noise perturbation, and saves
K separate checkpoint files.  Only the unique per-head parameters are cloned
(fc, decoder layer, norms).  embed_tokens and lm_head are shared at runtime
and NOT duplicated.

MTP weight keys in the original checkpoint (15 total):
    mtp.fc.weight
    mtp.layers.0.input_layernorm.weight
    mtp.layers.0.mlp.{down,gate,up}_proj.weight
    mtp.layers.0.post_attention_layernorm.weight
    mtp.layers.0.self_attn.{k,q}_norm.weight
    mtp.layers.0.self_attn.{k,o,q,v}_proj.weight
    mtp.norm.weight
    mtp.pre_fc_norm_embedding.weight
    mtp.pre_fc_norm_hidden.weight

Usage:
    python mtp_clone.py \
        --model-dir /home/ubuntu/models/Qwen3.5-27B \
        --output-dir /home/ubuntu/models/mtp-siblings \
        --num-heads 3 \
        --sigmas 0.0,0.01,0.02 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# The 15 unique MTP weight keys (everything NOT embed_tokens / lm_head)
# ---------------------------------------------------------------------------
MTP_UNIQUE_KEYS = [
    "mtp.fc.weight",
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
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]


def load_mtp_weights(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load only MTP-unique weights from a safetensors checkpoint.

    Reads the index file to find which shards contain MTP weights, then
    loads only those shards and extracts the relevant tensors.  This avoids
    loading the full 54 GB model into RAM.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"No safetensors index at {index_path}. "
            "Point --model-dir at the original Qwen3.5-27B (bf16) checkpoint."
        )

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Figure out which shard files we need
    needed_shards: dict[str, list[str]] = {}  # shard_file -> [key, ...]
    for key in MTP_UNIQUE_KEYS:
        if key not in weight_map:
            raise KeyError(
                f"MTP key '{key}' not found in {index_path}. "
                "This checkpoint may not contain MTP weights (e.g. GPTQ-quantized models "
                "often strip MTP).  Use the original bf16 checkpoint."
            )
        shard = weight_map[key]
        needed_shards.setdefault(shard, []).append(key)

    print(f"Loading MTP weights from {len(needed_shards)} shard(s)...")
    weights: dict[str, torch.Tensor] = {}

    for shard_file, keys in needed_shards.items():
        shard_path = model_dir / shard_file
        print(f"  {shard_file}: {len(keys)} tensor(s)")
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                weights[key] = f.get_tensor(key).clone()

    assert len(weights) == len(MTP_UNIQUE_KEYS), (
        f"Expected {len(MTP_UNIQUE_KEYS)} weights, got {len(weights)}"
    )
    return weights


def clone_with_noise(
    weights: dict[str, torch.Tensor],
    sigma: float,
    rng: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Clone a weight dict, adding N(0, sigma * std(w)) noise to each tensor.

    For sigma=0 this is an exact copy.  The noise is scaled relative to each
    tensor's own standard deviation so that large and small weight matrices
    receive proportional perturbation.

    Norm weights (1-D, typically near 1.0) get sigma/10 to avoid destabilizing
    the normalization layers.
    """
    cloned: dict[str, torch.Tensor] = {}
    for key, tensor in weights.items():
        t = tensor.clone().float()  # work in fp32 for noise addition
        if sigma > 0:
            is_norm = "norm" in key and t.dim() == 1
            effective_sigma = sigma * 0.1 if is_norm else sigma
            noise_scale = t.std().item() * effective_sigma
            if noise_scale > 0:
                noise = torch.randn(t.shape, generator=rng, dtype=torch.float32)
                t = t + noise * noise_scale
        cloned[key] = t.to(tensor.dtype)  # back to original dtype (bf16)
    return cloned


def main():
    parser = argparse.ArgumentParser(description="Clone Qwen 3.5 MTP head into K siblings")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/ubuntu/models/Qwen3.5-27B",
        help="Path to original Qwen3.5-27B checkpoint with MTP weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/models/mtp-siblings",
        help="Output directory for cloned head checkpoints",
    )
    parser.add_argument(
        "--num-heads", "-K",
        type=int,
        default=3,
        help="Number of sibling heads to create",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default="0.0,0.01,0.02",
        help="Comma-separated noise sigma per head (len must == num-heads)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible noise",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Output dtype for cloned weights",
    )
    args = parser.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    if len(sigmas) != args.num_heads:
        raise ValueError(
            f"--sigmas has {len(sigmas)} values but --num-heads is {args.num_heads}"
        )

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    target_dtype = dtype_map[args.dtype]

    # Step 1: Load the original MTP weights (only ~787 MB at bf16)
    t0 = time.monotonic()
    mtp_weights = load_mtp_weights(args.model_dir)
    print(f"Loaded {len(mtp_weights)} MTP tensors in {time.monotonic() - t0:.1f}s")

    # Print size summary
    total_bytes = sum(t.numel() * t.element_size() for t in mtp_weights.values())
    print(f"Total MTP unique params: {total_bytes / 1e6:.1f} MB")
    for key, tensor in sorted(mtp_weights.items()):
        mb = tensor.numel() * tensor.element_size() / 1e6
        print(f"  {key}: {list(tensor.shape)} {tensor.dtype} ({mb:.1f} MB)")

    # Step 2: Clone K times with noise
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    manifest = {
        "source_model": str(args.model_dir),
        "num_heads": args.num_heads,
        "sigmas": sigmas,
        "seed": args.seed,
        "dtype": args.dtype,
        "mtp_keys": MTP_UNIQUE_KEYS,
        "heads": [],
    }

    for head_idx in range(args.num_heads):
        sigma = sigmas[head_idx]
        print(f"\nCloning head {head_idx} (sigma={sigma})...")
        t1 = time.monotonic()

        cloned = clone_with_noise(mtp_weights, sigma, rng)

        # Convert to target dtype
        cloned = {k: v.to(target_dtype) for k, v in cloned.items()}

        # Save as safetensors
        filename = f"mtp_sibling_{head_idx}.safetensors"
        filepath = output_dir / filename
        save_file(cloned, str(filepath))

        file_mb = filepath.stat().st_size / 1e6
        print(f"  Saved {filepath} ({file_mb:.1f} MB) in {time.monotonic() - t1:.1f}s")

        manifest["heads"].append({
            "index": head_idx,
            "sigma": sigma,
            "file": filename,
            "size_mb": round(file_mb, 1),
        })

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Total time: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
