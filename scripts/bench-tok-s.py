#!/usr/bin/env python3
"""Quick tok/s benchmark for vLLM speculative decoding configs.

Usage:
    python3 scripts/bench-tok-s.py [--endpoint URL] [--trials N] [--max-tokens N]

Measures generation throughput on a fixed set of prompts.
"""

import argparse
import json
import subprocess
import sys
import time

ENDPOINT = "http://localhost:8001/v1/chat/completions"
MODEL = "qwen3.5-27b"

PROMPTS = [
    "Write a Python function that implements binary search on a sorted array. Include docstring and type hints.",
    "Explain the difference between TCP and UDP in exactly 200 words.",
    "Write a Rust function that parses a TOML file into a HashMap<String, String>.",
    "What are the three laws of thermodynamics? Be concise.",
    "Write a bash script that monitors disk usage and sends an alert if any partition exceeds 90%.",
]


def bench_one(prompt, max_tokens, endpoint):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    })
    t0 = time.time()
    r = subprocess.run(
        ["curl", "-s", "-X", "POST",
         "-H", "Content-Type: application/json",
         "-d", body, endpoint],
        capture_output=True, text=True, timeout=120,
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        return None
    try:
        resp = json.loads(r.stdout)
        usage = resp.get("usage", {})
        completion = usage.get("completion_tokens", 0)
        return {"tokens": completion, "elapsed": elapsed, "tok_s": completion / elapsed if elapsed > 0 else 0}
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    all_results = []
    for trial in range(args.trials):
        for i, prompt in enumerate(PROMPTS):
            r = bench_one(prompt, args.max_tokens, args.endpoint)
            if r is None:
                print(f"  trial {trial+1} prompt {i+1}: FAILED")
                continue
            all_results.append(r)
            print(f"  trial {trial+1} prompt {i+1}: {r['tokens']} tok in {r['elapsed']:.2f}s = {r['tok_s']:.1f} tok/s")

    if all_results:
        avg = sum(r["tok_s"] for r in all_results) / len(all_results)
        total_tok = sum(r["tokens"] for r in all_results)
        total_time = sum(r["elapsed"] for r in all_results)
        aggregate = total_tok / total_time if total_time > 0 else 0
        print(f"\nResults ({len(all_results)} measurements):")
        print(f"  Per-request average: {avg:.1f} tok/s")
        print(f"  Aggregate:           {aggregate:.1f} tok/s")
        print(f"  Total tokens:        {total_tok}")
    else:
        print("No successful measurements.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
