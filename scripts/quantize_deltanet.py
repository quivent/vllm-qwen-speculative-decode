#!/usr/bin/env python3
"""
Quantize Qwen3.5-27B-DeltaNet-draft from bf16 to W4A16 (GPTQ format, RTN).

Runs entirely on CPU. Produces GPTQ-compatible safetensors that vLLM can load.
"""

import json
import os
import shutil
import math
from pathlib import Path
from collections import OrderedDict

import torch
import safetensors.torch as st

SRC = Path("/home/ubuntu/models/Qwen3.5-27B-DeltaNet-draft")
DST = Path("/home/ubuntu/models/Qwen3.5-27B-DeltaNet-draft-W4A16")
BITS = 4
GROUP_SIZE = 128
SYMMETRIC = True

# Tensors matching these patterns get quantized (must be 2D and large enough)
QUANTIZE_SUFFIXES = [
    ".in_proj_qkv.weight",
    ".in_proj_z.weight",
    ".in_proj_a.weight",
    ".in_proj_b.weight",
    ".out_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
]

# Never quantize these
SKIP_NAMES = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
}


def should_quantize(name: str, shape: tuple) -> bool:
    if name in SKIP_NAMES:
        return False
    if len(shape) != 2:
        return False
    # Only quantize if input dim (shape[1]) is divisible by group_size
    if shape[1] % GROUP_SIZE != 0:
        return False
    for suffix in QUANTIZE_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def quantize_tensor_rtn(weight: torch.Tensor) -> tuple:
    """
    Quantize a 2D weight tensor to 4-bit using Round-To-Nearest (RTN).
    Returns (qweight, qzeros, scales) in GPTQ format.

    weight: [out_features, in_features] in float
    qweight: [in_features // 8, out_features] packed int32 (8 x 4-bit per int32)
    scales: [in_features // group_size, out_features] in float16
    qzeros: [in_features // group_size, out_features // 8] packed int32
    """
    out_features, in_features = weight.shape
    assert in_features % GROUP_SIZE == 0

    num_groups = in_features // GROUP_SIZE

    # Work in float32 for precision
    w = weight.float()

    # Reshape to [out_features, num_groups, group_size]
    w = w.reshape(out_features, num_groups, GROUP_SIZE)

    if SYMMETRIC:
        # Symmetric: range is [-max_abs, max_abs], zero_point = 2^(bits-1)
        maxq = 2 ** BITS - 1  # 15
        zero_point = 2 ** (BITS - 1)  # 8

        max_abs = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-5)
        scale = max_abs / (2 ** (BITS - 1))  # each side gets 8 levels

        # Quantize
        q = torch.round(w / scale).clamp(-zero_point, maxq - zero_point) + zero_point
        q = q.to(torch.int32)

        # scales: [num_groups, out_features]
        scales_out = scale.squeeze(2).transpose(0, 1).to(torch.float16)

        # qzeros: all zero_point (8 for symmetric 4-bit)
        # Shape: [num_groups, out_features]
        zeros = torch.full((num_groups, out_features), zero_point, dtype=torch.int32)
    else:
        maxq = 2 ** BITS - 1
        wmin = w.amin(dim=2, keepdim=True)
        wmax = w.amax(dim=2, keepdim=True)

        scale = (wmax - wmin).clamp(min=1e-5) / maxq
        zero_point = torch.round(-wmin / scale).clamp(0, maxq).to(torch.int32)

        q = torch.round(w / scale + zero_point.float()).clamp(0, maxq).to(torch.int32)

        scales_out = scale.squeeze(2).transpose(0, 1).to(torch.float16)
        zeros = zero_point.squeeze(2).transpose(0, 1)

    # q shape: [out_features, num_groups, group_size]
    # Reshape to [out_features, in_features]
    q = q.reshape(out_features, in_features)

    # Pack qweight: GPTQ format packs along out_features dimension
    # qweight shape: [in_features // 8, out_features] where 8 = 32 // bits
    pack_factor = 32 // BITS  # 8

    # Transpose to [in_features, out_features] for packing
    q_t = q.transpose(0, 1).contiguous()  # [in_features, out_features]

    assert out_features % pack_factor == 0 or True  # we pack along in_features
    # Actually in GPTQ, qweight is [in_features // pack_factor, out_features]
    # where pack_factor values from the in_features dimension are packed

    # Repack: pack along rows (in_features dimension)
    assert in_features % pack_factor == 0
    q_t = q_t.reshape(in_features // pack_factor, pack_factor, out_features)

    qweight = torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        qweight |= (q_t[:, j, :] << (BITS * j))

    # Pack qzeros: [num_groups, out_features // pack_factor]
    # Wait — let me check the GPTQ convention more carefully.
    # In GPTQ marlin/exllama format:
    # qweight: [in_features // pack_factor, out_features]
    # qzeros:  [num_groups, out_features // pack_factor]
    # scales:  [num_groups, out_features]

    assert out_features % pack_factor == 0, f"out_features {out_features} not divisible by {pack_factor}"

    qzeros_packed = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
    zeros_reshaped = zeros.reshape(num_groups, out_features // pack_factor, pack_factor)
    for j in range(pack_factor):
        qzeros_packed |= (zeros_reshaped[:, :, j] << (BITS * j))

    return qweight.contiguous(), qzeros_packed.contiguous(), scales_out.contiguous()


def main():
    print(f"Quantizing {SRC} -> {DST}")
    print(f"Config: {BITS}-bit, group_size={GROUP_SIZE}, symmetric={SYMMETRIC}")

    DST.mkdir(parents=True, exist_ok=True)

    # Load source index
    with open(SRC / "model.safetensors.index.json") as f:
        src_index = json.load(f)

    weight_map = src_index["weight_map"]

    # Group tensors by source shard
    shard_to_tensors = {}
    for tensor_name, shard_file in weight_map.items():
        shard_to_tensors.setdefault(shard_file, []).append(tensor_name)

    # Plan: which tensors get quantized
    quantize_count = 0
    skip_count = 0

    new_weight_map = {}
    all_output_tensors = OrderedDict()
    output_shard_idx = 0
    output_shard_size = 0
    MAX_SHARD_SIZE = 4 * 1024**3  # 4GB per shard
    output_shards = []
    current_shard_tensors = OrderedDict()

    # Process each source shard
    sorted_shards = sorted(shard_to_tensors.keys())

    for shard_file in sorted_shards:
        tensor_names = shard_to_tensors[shard_file]
        print(f"\nProcessing {shard_file} ({len(tensor_names)} tensors)...")

        shard_data = st.load_file(str(SRC / shard_file))

        for tname in sorted(tensor_names):
            tensor = shard_data[tname]

            if should_quantize(tname, tensor.shape):
                quantize_count += 1
                base_name = tname.rsplit(".weight", 1)[0]

                qweight, qzeros, scales = quantize_tensor_rtn(tensor)

                # Add quantized tensors
                for suffix, data in [(".qweight", qweight), (".qzeros", qzeros), (".scales", scales)]:
                    qname = base_name + suffix
                    tensor_bytes = data.numel() * data.element_size()

                    if output_shard_size + tensor_bytes > MAX_SHARD_SIZE and current_shard_tensors:
                        # Save current shard
                        output_shard_idx += 1
                        output_shards.append((output_shard_idx, current_shard_tensors))
                        current_shard_tensors = OrderedDict()
                        output_shard_size = 0

                    current_shard_tensors[qname] = data
                    output_shard_size += tensor_bytes

                orig_mb = tensor.numel() * tensor.element_size() / 1024**2
                new_mb = (qweight.numel() * 4 + qzeros.numel() * 4 + scales.numel() * 2) / 1024**2
                print(f"  Quantized {tname}: {tensor.shape} ({orig_mb:.1f}MB -> {new_mb:.1f}MB)")
            else:
                skip_count += 1
                tensor_bytes = tensor.numel() * tensor.element_size()

                if output_shard_size + tensor_bytes > MAX_SHARD_SIZE and current_shard_tensors:
                    output_shard_idx += 1
                    output_shards.append((output_shard_idx, current_shard_tensors))
                    current_shard_tensors = OrderedDict()
                    output_shard_size = 0

                current_shard_tensors[tname] = tensor
                output_shard_size += tensor_bytes

        # Free memory
        del shard_data

    # Save final shard
    if current_shard_tensors:
        output_shard_idx += 1
        output_shards.append((output_shard_idx, current_shard_tensors))

    total_shards = len(output_shards)
    print(f"\n{'='*60}")
    print(f"Quantized {quantize_count} tensors, kept {skip_count} as-is")
    print(f"Writing {total_shards} output shards...")

    # Write shards and build weight map
    new_weight_map = {}
    total_size = 0

    for idx, (shard_num, tensors) in enumerate(output_shards):
        if total_shards == 1:
            shard_name = "model.safetensors"
        else:
            shard_name = f"model-{shard_num:05d}-of-{total_shards:05d}.safetensors"

        shard_path = DST / shard_name
        print(f"  Writing {shard_name} ({len(tensors)} tensors)...")
        st.save_file(tensors, str(shard_path))

        shard_size = shard_path.stat().st_size
        total_size += shard_size
        print(f"    Size: {shard_size / 1024**3:.2f} GB")

        for tname in tensors:
            new_weight_map[tname] = shard_name

        # Free memory
        output_shards[idx] = None

    # Write index
    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map
    }
    with open(DST / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy config files and add quantization config
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "merges.txt", "preprocessor_config.json",
                   "chat_template.jinja"]:
        src_path = SRC / fname
        if src_path.exists():
            shutil.copy2(str(src_path), str(DST / fname))

    # Update config.json with quantization config
    with open(DST / "config.json") as f:
        config = json.load(f)

    config["quantization_config"] = {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": SYMMETRIC,
    }

    with open(DST / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Also write standalone quantization_config.json
    with open(DST / "quantization_config.json", "w") as f:
        json.dump(config["quantization_config"], f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total output size: {total_size / 1024**3:.2f} GB")
    print(f"Compression ratio: {42.7 / (total_size / 1024**3):.1f}x")
    print(f"Output: {DST}")
    print("Done!")


if __name__ == "__main__":
    main()
