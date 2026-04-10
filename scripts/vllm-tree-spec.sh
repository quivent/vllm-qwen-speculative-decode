#!/usr/bin/env bash
# Launch vLLM with branching tree speculation for Qwen 3.5-27B.
#
# Requirements:
#   1. eagle.py model_returns_tuple fix applied (propose_tree unpacking bug)
#   2. --attention-backend TREE_ATTN forces tree attention globally
#      (required because propose_tree builds TreeAttentionMetadata which
#       must match the attention impl; flash_attn impl rejects it)
#
# Tree shape: 3-3 (3 root candidates, 3 children per root = 9 nodes, depth 2)
#   Level 0: top-3 tokens from MTP head → (0,), (1,), (2,)
#   Level 1: top-3 from each root → (0,0),(0,1),(0,2),(1,0),...,(2,2)
#   Total: 12 draft tokens verified in one batched forward pass
#
# The tree attention bias mask ensures each node only attends to its
# ancestor path. Verification accepts the longest matching path.

set -euo pipefail

MODEL="${QWEN_MODEL:-/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-4bit}"

# 3-2 tree: 3 roots, 2 children per root = 9 total nodes
# Sorted breadth-first as vLLM expects.
TREE='[(0,), (1,), (2,), (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]'

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype auto \
    --quantization gptq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --speculative-model "[mtp]" \
    --speculative-token-tree "$TREE" \
    --attention-backend TREE_ATTN \
    --enforce-eager \
    "$@"
