#!/usr/bin/env python3
"""Simple p_agree benchmark for partial-layer verification.

Measures how often the argmax at layer N matches the argmax at layer 64
by hooking into the model's forward pass.
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Write a Python function that implements binary search.",
    "Explain the difference between TCP and UDP.",
    "What are the three laws of thermodynamics?",
    "Write a bash script that monitors disk usage.",
    "The capital of France is",
]


def measure_p_agree(model, tokenizer, exit_layers, max_new_tokens=30, device="cuda"):
    """Hook into model layers to capture intermediate hidden states."""

    # Find the text model backbone and its layers
    if hasattr(model, 'language_model'):
        backbone = model.language_model.model
        lm_head = model.language_model.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        backbone = model.model
        lm_head = model.lm_head
    else:
        raise ValueError("Cannot find model backbone")

    norm = backbone.norm
    num_layers = len(backbone.layers)
    print(f"Model has {num_layers} layers, norm={type(norm).__name__}")

    results = {}
    for el in exit_layers:
        if el >= num_layers:
            continue

        # Hook to capture hidden states after layer `el`
        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is typically (hidden_states,) or (hidden_states, ...)
                if isinstance(output, tuple):
                    captured[layer_idx] = output[0].detach()
                else:
                    captured[layer_idx] = output.detach()
            return hook_fn

        # Register hooks on the exit layer and the last layer
        hook_exit = backbone.layers[el - 1].register_forward_hook(make_hook(el))
        hook_full = backbone.layers[num_layers - 1].register_forward_hook(make_hook(num_layers))

        total_agree = 0
        total_tokens = 0
        partial_ms = 0
        full_ms = 0

        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                t0 = time.perf_counter()
                outputs = model(**inputs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

            full_logits = outputs.logits  # [1, seq_len, vocab]
            full_argmax = full_logits[0, -1].argmax().item()

            # Get intermediate hidden states from hook
            if el in captured and num_layers in captured:
                h_partial = captured[el]
                # Apply norm + lm_head to get early logits
                h_partial_normed = norm(h_partial)
                early_logits = lm_head(h_partial_normed)
                early_argmax = early_logits[0, -1].argmax().item()

                agreed = early_argmax == full_argmax
                total_agree += int(agreed)
                total_tokens += 1

            captured.clear()

        hook_exit.remove()
        hook_full.remove()

        p_agree = total_agree / total_tokens if total_tokens > 0 else 0
        results[el] = {
            "p_agree": p_agree,
            "agreed": total_agree,
            "total": total_tokens,
        }
        print(f"  exit_layer={el:3d}/{num_layers}: p_agree={p_agree:.4f} ({total_agree}/{total_tokens})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-27B")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    exit_layers = [4, 8, 16, 24, 32, 40, 48, 56, 60]
    print(f"\nSweeping exit layers: {exit_layers}")
    results = measure_p_agree(model, tokenizer, exit_layers, args.max_new_tokens)

    print("\n=== Summary ===")
    for el, r in sorted(results.items()):
        print(f"  Layer {el:3d}: p_agree={r['p_agree']:.4f}")


if __name__ == "__main__":
    main()
