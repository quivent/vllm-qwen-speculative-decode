#!/usr/bin/env python3
"""PLV benchmark: partial-layer verification at full-attention boundaries.

Qwen 3.5: every 4th layer is full attention, others are DeltaNet linear.
Exits ONLY at full-attention boundaries where global context is consolidated.

Hooks all exit layers simultaneously (negligible overhead).
Measures p_agree on next-token prediction for each prompt.

Two strategies:
  A) final_norm + lm_head (standard early exit)
  B) next_layer_input_layernorm + final_norm + lm_head
"""

import argparse
import gc
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 20+ prompts for statistical significance — each yields 1 data point per layer
PROMPTS = [
    "The capital of France is",
    "Write a Python binary search function.",
    "TCP differs from UDP because",
    "The three laws of thermodynamics are",
    "In quantum mechanics, uncertainty means",
    "def fibonacci(n):",
    "A transformer processes tokens by",
    "Stacks and queues differ because",
    "SELECT * FROM users WHERE",
    "General relativity predicts that",
    "The mitochondria is the",
    "Machine learning models are trained by",
    "The speed of light in a vacuum is",
    "HTTP status code 404 means",
    "In Python, a list comprehension is",
    "The Pythagorean theorem states that",
    "Photosynthesis converts sunlight into",
    "A binary tree is a data structure where",
    "The boiling point of water at sea level is",
    "Recursion in programming means",
    "The chemical formula for water is",
    "An API endpoint typically returns",
    "The largest planet in our solar system is",
    "In statistics, the mean is calculated by",
    "A hash table provides O(1) lookup because",
    "The human genome contains approximately",
    "Docker containers differ from virtual machines because",
    "The derivative of x squared is",
    "In economics, inflation refers to",
    "A neural network's backpropagation algorithm",
]

FULL_ATTN_LAYERS_1IDX = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
SWEEP_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 48, 60]


def get_model_parts(model):
    if hasattr(model, 'language_model'):
        backbone = model.language_model.model
        lm_head = model.language_model.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        backbone = model.model
        lm_head = model.lm_head
    else:
        raise ValueError("Cannot find model backbone")
    return backbone, lm_head, backbone.norm


def run_sweep(model, tokenizer, exit_layers, device="cpu"):
    """Hook all exit layers at once, run each prompt once, measure p_agree."""
    backbone, lm_head, final_norm = get_model_parts(model)
    num_layers = len(backbone.layers)

    layer_types = None
    if hasattr(model, 'config') and hasattr(model.config, 'text_config'):
        tc = model.config.text_config
        if hasattr(tc, 'layer_types'):
            layer_types = tc.layer_types

    valid_layers = []
    for el in exit_layers:
        if el > num_layers:
            continue
        if layer_types and layer_types[el - 1] != "full_attention":
            print(f"  SKIP {el}: {layer_types[el-1]}", flush=True)
            continue
        valid_layers.append(el)

    print(f"Layers: {valid_layers}, Prompts: {len(PROMPTS)}", flush=True)

    # Hook ALL layers at once — each captures only [1, 1, 5120] = 10KB
    captured = {}
    def make_hook(idx):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[idx] = t[:, -1:, :].detach().clone()
        return fn

    hooks = [backbone.layers[el-1].register_forward_hook(make_hook(el)) for el in valid_layers]

    # Also hook some non-full-attention layers for comparison
    comparison_layers = [2, 6, 10, 14, 18, 22, 26, 30]  # DeltaNet layers
    comp_captured = {}
    def make_comp_hook(idx):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            comp_captured[idx] = t[:, -1:, :].detach().clone()
        return fn
    comp_hooks = [backbone.layers[el-1].register_forward_hook(make_comp_hook(el)) for el in comparison_layers if el <= num_layers]

    # Accumulators
    stats = {el: {"norm_only": {"agree": 0, "top5": 0, "kl_sum": 0.0, "n": 0},
                   "next_input_norm": {"agree": 0, "top5": 0, "kl_sum": 0.0, "n": 0}}
             for el in valid_layers}
    comp_stats = {el: {"agree": 0, "top5": 0, "kl_sum": 0.0, "n": 0}
                  for el in comparison_layers if el <= num_layers}

    t_start = time.time()

    for pi, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]

        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False)

        elapsed = time.time() - t_start
        full_logits = out.logits[0, -1]
        full_argmax = full_logits.argmax().item()
        full_top5 = set(full_logits.topk(5).indices.tolist())
        full_probs = torch.softmax(full_logits.float(), dim=-1)

        for el in valid_layers:
            if el not in captured:
                continue
            h = captured[el]

            for strat, use_next in [("norm_only", False), ("next_input_norm", True)]:
                h2 = h
                if use_next and el < num_layers:
                    nl = backbone.layers[el]
                    if hasattr(nl, 'input_layernorm'):
                        h2 = nl.input_layernorm(h2)
                early_logits = lm_head(final_norm(h2))[0, 0]
                ea = early_logits.argmax().item()

                s = stats[el][strat]
                s["n"] += 1
                s["agree"] += int(ea == full_argmax)
                if ea in full_top5:
                    s["top5"] += 1
                ep = torch.softmax(early_logits.float(), dim=-1)
                s["kl_sum"] += torch.sum(full_probs * (torch.log(full_probs + 1e-10) - torch.log(ep + 1e-10))).item()

        # Comparison: DeltaNet layers with norm_only
        for el in comparison_layers:
            if el not in comp_captured or el > num_layers:
                continue
            h = comp_captured[el]
            early_logits = lm_head(final_norm(h))[0, 0]
            ea = early_logits.argmax().item()
            s = comp_stats[el]
            s["n"] += 1
            s["agree"] += int(ea == full_argmax)
            if ea in full_top5:
                s["top5"] += 1
            ep = torch.softmax(early_logits.float(), dim=-1)
            s["kl_sum"] += torch.sum(full_probs * (torch.log(full_probs + 1e-10) - torch.log(ep + 1e-10))).item()

        captured.clear()
        comp_captured.clear()
        del out
        gc.collect()

        print(f"  [{pi+1}/{len(PROMPTS)}] {elapsed:.0f}s  prompt={prompt[:40]}...", flush=True)

    for h in hooks:
        h.remove()
    for h in comp_hooks:
        h.remove()

    # Compile results
    results = {}
    for el in valid_layers:
        results[el] = {}
        for strat in ["norm_only", "next_input_norm"]:
            s = stats[el][strat]
            n = s["n"]
            results[el][strat] = {
                "p_agree": s["agree"] / n if n > 0 else 0,
                "p_top5": s["top5"] / n if n > 0 else 0,
                "avg_kl": s["kl_sum"] / n if n > 0 else 0,
                "agreed": s["agree"],
                "total": n,
            }

    comp_results = {}
    for el in comparison_layers:
        if el not in comp_stats or el > num_layers:
            continue
        s = comp_stats[el]
        n = s["n"]
        comp_results[el] = {
            "p_agree": s["agree"] / n if n > 0 else 0,
            "p_top5": s["top5"] / n if n > 0 else 0,
            "avg_kl": s["kl_sum"] / n if n > 0 else 0,
            "agreed": s["agree"],
            "total": n,
        }

    return results, comp_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-27B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sweep-layers", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading {args.model_path} on {args.device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=args.device)
    model.eval()

    exit_layers = [int(x) for x in args.sweep_layers.split(",")] if args.sweep_layers else SWEEP_LAYERS
    for el in exit_layers:
        assert el in FULL_ATTN_LAYERS_1IDX, f"Layer {el} not full-attention"

    print(f"Expected time: ~{len(PROMPTS) * 15 / 60:.0f} min ({len(PROMPTS)} fwd passes)", flush=True)

    results, comp_results = run_sweep(model, tokenizer, exit_layers, args.device)

    # Full-attention boundary results
    print(f"\n{'='*100}", flush=True)
    print(f"FULL-ATTENTION BOUNDARY LAYERS (exit at layer N after full-attn completes)", flush=True)
    print(f"{'='*100}", flush=True)
    hdr = f"{'Layer':>6} | {'Type':>12} | {'norm p_agree':>12} | {'next p_agree':>12} | {'norm p_top5':>12} | {'next p_top5':>12} | {'norm KL':>10} | {'next KL':>10} | {'N':>4}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for el in sorted(results.keys()):
        ra = results[el]["norm_only"]
        rb = results[el]["next_input_norm"]
        print(f"{el:6d} | {'full_attn':>12} | {ra['p_agree']:12.4f} | {rb['p_agree']:12.4f} | {ra['p_top5']:12.4f} | {rb['p_top5']:12.4f} | {ra['avg_kl']:10.4f} | {rb['avg_kl']:10.4f} | {ra['total']:4d}", flush=True)

    # DeltaNet comparison
    print(f"\n{'='*100}", flush=True)
    print(f"DELTANET (LINEAR ATTENTION) LAYERS — comparison baseline", flush=True)
    print(f"{'='*100}", flush=True)
    hdr2 = f"{'Layer':>6} | {'Type':>12} | {'norm p_agree':>12} | {'norm p_top5':>12} | {'norm KL':>10} | {'N':>4}"
    print(hdr2, flush=True)
    print("-" * len(hdr2), flush=True)
    for el in sorted(comp_results.keys()):
        r = comp_results[el]
        print(f"{el:6d} | {'deltanet':>12} | {r['p_agree']:12.4f} | {r['p_top5']:12.4f} | {r['avg_kl']:10.4f} | {r['total']:4d}", flush=True)

    # JSON
    json_path = "/home/ubuntu/aut/plv_full_attn_results.json"
    out = {
        "full_attention": {str(k): v for k, v in results.items()},
        "deltanet_comparison": {str(k): v for k, v in comp_results.items()},
        "config": {
            "prompts": len(PROMPTS),
            "model": args.model_path,
            "exit_layers": [int(x) for x in results.keys()],
            "comparison_layers": [int(x) for x in comp_results.keys()],
        }
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
