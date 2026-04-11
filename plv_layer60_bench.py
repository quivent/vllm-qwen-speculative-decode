#!/usr/bin/env python3
"""Partial-Layer Verification benchmark: layer 60 early exit with fallback.

Measures ACTUAL timing difference between:
  - Full 64-layer forward pass
  - 60-layer forward pass (early exit)
  - Two-pass: 60 layers, then conditionally +4 more

Also measures per-layer timing to check if layers 60-63 are
disproportionately expensive (layer 63 is full_attention).

Architecture of layers 56-63:
  56: linear_attention (DeltaNet)
  57: linear_attention
  58: linear_attention
  59: full_attention  <-- expensive
  60: linear_attention
  61: linear_attention
  62: linear_attention
  63: full_attention  <-- expensive

Uses bf16 model on CPU. Timing RATIOS are the deliverable, not
absolute times (GPU has different bottlenecks).
"""

import argparse
import gc
import json
import time
import statistics
import torch
import torch.nn.functional as F
from pathlib import Path


MODEL_PATH = "/home/ubuntu/models/Qwen3.5-27B"

PROMPTS = [
    "The capital of France is",
    "Write a Python binary search function.",
    "TCP differs from UDP because",
    "The three laws of thermodynamics are",
    "In quantum mechanics, uncertainty means",
    "def fibonacci(n):",
    "A transformer processes tokens by",
    "SELECT * FROM users WHERE",
    "General relativity predicts that",
    "The mitochondria is the",
    "Machine learning models are trained by",
    "HTTP status code 404 means",
    "The Pythagorean theorem states that",
    "A binary tree is a data structure where",
    "Recursion in programming means",
]


def load_model(device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Loading model on {device} in bf16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=device if device == "cpu" else {"": 0},
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def get_parts(model):
    if hasattr(model, 'language_model'):
        backbone = model.language_model.model
        lm_head = model.language_model.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        backbone = model.model
        lm_head = model.lm_head
    else:
        raise ValueError("Cannot find backbone")
    return backbone, lm_head, backbone.norm


def timed_forward_range(backbone, lm_head, norm, input_ids, start_layer, end_layer, device="cpu"):
    """Run layers [start_layer, end_layer) and return (hidden_states, time_ns).

    If start_layer == 0, starts from embedding.
    If end_layer < total, returns pre-norm hidden states.
    If end_layer == total, applies norm + returns final hidden states.
    """
    num_layers = len(backbone.layers)

    with torch.inference_mode():
        if start_layer == 0:
            # Full forward from embedding
            t0 = time.perf_counter_ns()
            out = backbone.embed_tokens(input_ids)

            # Detect if using transformers with rotary_emb
            if hasattr(backbone, 'rotary_emb'):
                position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                position_embeddings = backbone.rotary_emb(out, position_ids)
                for i in range(start_layer, end_layer):
                    layer_out = backbone.layers[i](out, position_embeddings=position_embeddings)
                    out = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            else:
                residual = None
                positions = torch.arange(input_ids.shape[1], device=device)
                for i in range(start_layer, end_layer):
                    out, residual = backbone.layers[i](
                        positions=positions, hidden_states=out, residual=residual)

            if end_layer >= num_layers:
                out = backbone.norm(out)
            t1 = time.perf_counter_ns()
            return out, t1 - t0
        else:
            raise ValueError("Continuation from mid-layer needs cached hidden states")


def timed_partial_then_remaining(backbone, lm_head, norm, input_ids, exit_layer, device="cpu"):
    """Run 0..exit_layer, get logits; then run exit_layer..64, get logits.

    Returns (partial_logits, full_logits, time_partial_ns, time_remaining_ns, time_full_ns).
    """
    num_layers = len(backbone.layers)

    with torch.inference_mode():
        # --- Partial pass: layers 0..exit_layer ---
        t0 = time.perf_counter_ns()
        h = backbone.embed_tokens(input_ids)

        if hasattr(backbone, 'rotary_emb'):
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            position_embeddings = backbone.rotary_emb(h, position_ids)
            for i in range(exit_layer):
                layer_out = backbone.layers[i](h, position_embeddings=position_embeddings)
                h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            h_at_exit = h.clone()  # save for continuation
            h_normed = backbone.norm(h)
            partial_logits = lm_head(h_normed)
        else:
            residual = None
            positions = torch.arange(input_ids.shape[1], device=device)
            for i in range(exit_layer):
                h, residual = backbone.layers[i](
                    positions=positions, hidden_states=h, residual=residual)
            h_at_exit = h.clone()
            residual_at_exit = residual.clone() if residual is not None else None
            h_normed, _ = backbone.norm(h.clone(), residual.clone() if residual is not None else None)
            partial_logits = lm_head(h_normed)
        t1 = time.perf_counter_ns()
        time_partial = t1 - t0

        # --- Remaining pass: layers exit_layer..64 ---
        t2 = time.perf_counter_ns()
        h = h_at_exit
        if hasattr(backbone, 'rotary_emb'):
            for i in range(exit_layer, num_layers):
                layer_out = backbone.layers[i](h, position_embeddings=position_embeddings)
                h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            h_normed = backbone.norm(h)
            full_logits = lm_head(h_normed)
        else:
            residual = residual_at_exit
            for i in range(exit_layer, num_layers):
                h, residual = backbone.layers[i](
                    positions=positions, hidden_states=h, residual=residual)
            h_normed, _ = backbone.norm(h, residual)
            full_logits = lm_head(h_normed)
        t3 = time.perf_counter_ns()
        time_remaining = t3 - t2

    # Full = partial + remaining (approximate, since we cloned)
    time_full = time_partial + time_remaining

    return partial_logits, full_logits, time_partial, time_remaining, time_full


def measure_per_layer_timing(backbone, input_ids, device="cpu", n_repeats=1):
    """Measure time per layer to identify expensive ones."""
    num_layers = len(backbone.layers)
    layer_times = {}

    with torch.inference_mode():
        h = backbone.embed_tokens(input_ids)

        if hasattr(backbone, 'rotary_emb'):
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            position_embeddings = backbone.rotary_emb(h, position_ids)

            for i in range(num_layers):
                times = []
                for _ in range(n_repeats):
                    t0 = time.perf_counter_ns()
                    layer_out = backbone.layers[i](h, position_embeddings=position_embeddings)
                    t1 = time.perf_counter_ns()
                    times.append(t1 - t0)
                h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                layer_times[i] = statistics.median(times) if n_repeats > 1 else times[0]
        else:
            residual = None
            positions = torch.arange(input_ids.shape[1], device=device)
            for i in range(num_layers):
                times = []
                for _ in range(n_repeats):
                    t0 = time.perf_counter_ns()
                    h_new, res_new = backbone.layers[i](
                        positions=positions, hidden_states=h, residual=residual)
                    t1 = time.perf_counter_ns()
                    times.append(t1 - t0)
                h, residual = h_new, res_new
                layer_times[i] = statistics.median(times) if n_repeats > 1 else times[0]

    return layer_times


def run_benchmark(model, tokenizer, exit_layer=60, device="cpu", max_prompts=None):
    backbone, lm_head, norm = get_parts(model)
    num_layers = len(backbone.layers)

    # Get layer types — check both config.text_config and config directly
    layer_types = None
    cfg = getattr(model, 'config', None)
    if cfg:
        tc = getattr(cfg, 'text_config', cfg)
        layer_types = getattr(tc, 'layer_types', None)
    if layer_types is None:
        # Fallback: load from config.json directly
        import json as _json
        cfg_path = Path(MODEL_PATH) / "config.json"
        if cfg_path.exists():
            raw = _json.load(open(cfg_path))
            tc_raw = raw.get("text_config", raw)
            layer_types = tc_raw.get("layer_types")

    prompts = PROMPTS[:max_prompts] if max_prompts else PROMPTS

    print(f"\n{'='*80}")
    print(f"PLV Layer-60 Benchmark")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Exit layer: {exit_layer}/{num_layers}")
    print(f"  Layers {exit_layer}-{num_layers-1}: ", end="")
    if layer_types:
        for i in range(exit_layer, num_layers):
            print(f"L{i}={layer_types[i][:4]} ", end="")
    print(f"\n  Device: {device}")
    print(f"  Prompts: {len(prompts)}")
    print(f"{'='*80}\n")

    # ========== Phase 1: Per-layer timing ==========
    print("Phase 1: Per-layer timing (first prompt only)...")
    first_ids = tokenizer(prompts[0], return_tensors="pt").to(device)["input_ids"]
    layer_times = measure_per_layer_timing(backbone, first_ids, device)

    total_layer_time = sum(layer_times.values())
    skipped_time = sum(layer_times[i] for i in range(exit_layer, num_layers))

    print(f"\n  Total layer time: {total_layer_time/1e6:.1f} ms")
    print(f"  Layers 0-{exit_layer-1}: {(total_layer_time - skipped_time)/1e6:.1f} ms ({(1 - skipped_time/total_layer_time)*100:.1f}%)")
    print(f"  Layers {exit_layer}-{num_layers-1}: {skipped_time/1e6:.1f} ms ({skipped_time/total_layer_time*100:.1f}%)")

    # Per-layer breakdown for last 8 layers
    print(f"\n  Per-layer timing (layers {max(0, exit_layer-4)}-{num_layers-1}):")
    for i in range(max(0, exit_layer - 4), num_layers):
        lt = layer_types[i][:4] if layer_types else "?"
        pct = layer_times[i] / total_layer_time * 100
        print(f"    L{i:2d} ({lt}): {layer_times[i]/1e6:8.2f} ms ({pct:.1f}%)")

    # Average by type
    if layer_types:
        full_attn_times = [layer_times[i] for i in range(num_layers) if layer_types[i] == "full_attention"]
        linear_times = [layer_times[i] for i in range(num_layers) if layer_types[i] == "linear_attention"]
        avg_full = statistics.mean(full_attn_times) if full_attn_times else 0
        avg_linear = statistics.mean(linear_times) if linear_times else 0
        print(f"\n  Avg full_attention layer: {avg_full/1e6:.2f} ms")
        print(f"  Avg linear_attention layer: {avg_linear/1e6:.2f} ms")
        print(f"  Ratio full/linear: {avg_full/avg_linear:.2f}x" if avg_linear > 0 else "")

    # ========== Phase 2: Two-pass measurement ==========
    print(f"\nPhase 2: Two-pass verification measurement...")

    results = {
        "exit_layer": exit_layer,
        "total_layers": num_layers,
        "device": device,
        "per_layer_timing_ns": {str(k): v for k, v in layer_times.items()},
        "prompts": [],
    }

    all_partial_ms = []
    all_remaining_ms = []
    all_full_ms = []
    agree_count = 0
    top5_agree = 0
    total_tokens = 0

    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        seq_len = ids.shape[1]

        t_start = time.time()
        partial_logits, full_logits, t_partial, t_remaining, t_full = \
            timed_partial_then_remaining(backbone, lm_head, norm, ids, exit_layer, device)

        # Compare last-token predictions
        partial_pred = partial_logits[0, -1].argmax().item()
        full_pred = full_logits[0, -1].argmax().item()
        full_top5 = set(full_logits[0, -1].topk(5).indices.tolist())

        agreed = partial_pred == full_pred
        in_top5 = partial_pred in full_top5

        agree_count += int(agreed)
        top5_agree += int(in_top5)
        total_tokens += 1

        all_partial_ms.append(t_partial / 1e6)
        all_remaining_ms.append(t_remaining / 1e6)
        all_full_ms.append(t_full / 1e6)

        # KL divergence
        p_full = F.softmax(full_logits[0, -1].float(), dim=-1)
        p_partial = F.softmax(partial_logits[0, -1].float(), dim=-1)
        kl = (p_full * (torch.log(p_full + 1e-10) - torch.log(p_partial + 1e-10))).sum().item()

        results["prompts"].append({
            "prompt": prompt[:60],
            "seq_len": seq_len,
            "agreed": agreed,
            "in_top5": in_top5,
            "kl_div": kl,
            "partial_ms": t_partial / 1e6,
            "remaining_ms": t_remaining / 1e6,
            "full_ms": t_full / 1e6,
        })

        elapsed = time.time() - t_start
        print(f"  [{pi+1}/{len(prompts)}] {elapsed:.1f}s  agree={agreed}  top5={in_top5}  "
              f"partial={t_partial/1e6:.0f}ms  +remain={t_remaining/1e6:.0f}ms  "
              f"KL={kl:.2f}  {prompt[:40]}...")

    # ========== Phase 3: Analysis ==========
    p_agree = agree_count / total_tokens
    p_top5 = top5_agree / total_tokens
    avg_partial = statistics.mean(all_partial_ms)
    avg_remaining = statistics.mean(all_remaining_ms)
    avg_full = statistics.mean(all_full_ms)

    # The two-pass expected time:
    # If agree (p_agree fraction): just partial time
    # If disagree (1 - p_agree): partial + remaining time (= full)
    expected_twopass = p_agree * avg_partial + (1 - p_agree) * avg_full
    savings_pct = (1 - expected_twopass / avg_full) * 100

    # Overhead of the norm+lm_head computation at exit point (included in partial time)
    # This is unavoidable overhead even when we proceed to full
    overhead_ratio = avg_partial / avg_full

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"  p_agree (top-1):     {p_agree:.4f} ({agree_count}/{total_tokens})")
    print(f"  p_agree (top-5):     {p_top5:.4f} ({top5_agree}/{total_tokens})")
    print(f"  Avg partial time:    {avg_partial:.1f} ms ({avg_partial/avg_full*100:.1f}% of full)")
    print(f"  Avg remaining time:  {avg_remaining:.1f} ms ({avg_remaining/avg_full*100:.1f}% of full)")
    print(f"  Avg full time:       {avg_full:.1f} ms")
    print(f"")
    print(f"  Expected two-pass time:    {expected_twopass:.1f} ms")
    print(f"  vs full verification:      {avg_full:.1f} ms")
    print(f"  Net savings:               {savings_pct:.2f}%")
    print(f"")

    # Layer cost breakdown
    layers_60_63_time = sum(layer_times[i] for i in range(exit_layer, num_layers))
    theoretical_savings_pct = layers_60_63_time / total_layer_time * 100
    theoretical_net = p_agree * theoretical_savings_pct

    print(f"  Theoretical layer {exit_layer}-{num_layers-1} cost: {theoretical_savings_pct:.1f}% of total layers")
    print(f"  Theoretical net savings (p_agree * layer_pct): {theoretical_net:.2f}%")
    print(f"")

    # GPU projection
    # On GPU, a full verify step ~5ms at 186 tok/s
    gpu_verify_ms = 5.0  # approximate
    gpu_savings_ms = gpu_verify_ms * (savings_pct / 100)
    gpu_token_time_ms = 1000 / 186  # ~5.4ms per token
    throughput_improvement = gpu_savings_ms / gpu_token_time_ms * 100

    print(f"  GPU PROJECTION (assuming 5ms verify, 186 tok/s):")
    print(f"    Savings per verify:     {gpu_savings_ms:.3f} ms")
    print(f"    Token time at 186t/s:   {gpu_token_time_ms:.1f} ms")
    print(f"    Throughput improvement:  {throughput_improvement:.2f}%")
    print(f"")

    # Verdict
    print(f"  VERDICT: ", end="")
    if savings_pct < 1.0:
        print("NOT WORTH IT. Savings < 1%. Complexity cost exceeds benefit.")
    elif savings_pct < 3.0:
        print("MARGINAL. 1-3% savings may not survive pipeline overhead.")
    elif savings_pct < 5.0:
        print("POSSIBLE. 3-5% savings worth trying if integration is clean.")
    else:
        print("PROMISING. >5% savings, worth integrating.")

    # Save results
    results["summary"] = {
        "p_agree_top1": p_agree,
        "p_agree_top5": p_top5,
        "avg_partial_ms": avg_partial,
        "avg_remaining_ms": avg_remaining,
        "avg_full_ms": avg_full,
        "expected_twopass_ms": expected_twopass,
        "savings_pct": savings_pct,
        "theoretical_layer_pct": theoretical_savings_pct,
        "theoretical_net_pct": theoretical_net,
    }

    if layer_types:
        skipped_types = [layer_types[i] for i in range(exit_layer, num_layers)]
        n_full_attn = sum(1 for t in skipped_types if t == "full_attention")
        n_linear = sum(1 for t in skipped_types if t == "linear_attention")
        results["skipped_layers"] = {
            "count": num_layers - exit_layer,
            "full_attention": n_full_attn,
            "linear_attention": n_linear,
            "types": skipped_types,
        }

    out_path = Path("/home/ubuntu/aut/plv_layer60_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PLV layer-60 two-pass benchmark")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--exit-layer", type=int, default=60, help="Early exit layer")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit prompts")
    args = parser.parse_args()

    model, tokenizer = load_model(args.device)
    run_benchmark(model, tokenizer, exit_layer=args.exit_layer,
                  device=args.device, max_prompts=args.max_prompts)


if __name__ == "__main__":
    main()
