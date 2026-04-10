#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Diversity Fine-Tuning for Sibling MTP Heads.

Loads K cloned MTP heads and trains them jointly on a text corpus with:
  1. Standard cross-entropy next-token prediction (each head at its draft position)
  2. Diversity penalty: minimize pairwise cosine similarity between sibling
     heads' logit distributions at the same position

The MTP head architecture (from qwen3_5_mtp.py):
  - pre_fc_norm_hidden (RMSNorm 5120)
  - pre_fc_norm_embedding (RMSNorm 5120)
  - fc (Linear 10240 -> 5120, no bias)
  - layers[0] (Qwen3_5DecoderLayer — self_attn + MLP)
  - norm (RMSNorm 5120)

embed_tokens and lm_head are shared across all heads (frozen, not trained).

Loss = (1/K) * sum_k CE(head_k, target_{pos+k+1}) + lambda * diversity_penalty
diversity_penalty = mean over all (i,j) pairs of cosine_sim(logits_i, logits_j)

Usage:
    python mtp_diversity_train.py \
        --siblings-dir /home/ubuntu/models/mtp-siblings \
        --model-dir /home/ubuntu/models/Qwen3.5-27B \
        --output-dir /home/ubuntu/models/mtp-siblings-trained \
        --dataset wikitext \
        --max-examples 10000 \
        --epochs 1 \
        --lr 1e-4 \
        --lambda-div 0.1 \
        --batch-size 4 \
        --seq-len 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Lightweight MTP head (pure PyTorch, no vLLM dependencies)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm matching Qwen3_5RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(x.dtype)


class MTPHead(nn.Module):
    """Standalone MTP head for training.  Mirrors Qwen3_5MultiTokenPredictor
    but uses plain PyTorch layers (no TP, no vLLM compilation).

    Architecture:
        input: (hidden_states [B, D], input_ids [B])
        1. embed = embed_tokens(input_ids)  [shared, frozen]
        2. embed = pre_fc_norm_embedding(embed)
        3. hidden = pre_fc_norm_hidden(hidden_states)
        4. concat = cat([embed, hidden], dim=-1)  -> [B, 2D]
        5. hidden = fc(concat)  -> [B, D]
        6. hidden = decoder_layer(hidden)  [self_attn + MLP]
        7. hidden = norm(hidden)
        8. logits = lm_head(hidden)  [shared, frozen]

    For training we use chain_forward (skip attention, just fc + norm)
    because we don't have KV cache infrastructure.  This is the same path
    used by adaptive chained MTP at inference time.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 num_kv_heads: int, head_dim: int, vocab_size: int,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size

        # Norms (unique per head, trainable)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # FC projection: 2*hidden -> hidden (unique per head, trainable)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Decoder layer components (unique per head, trainable)
        # Self-attention
        # NOTE: Qwen 3.5's q_proj has 2x head_dim for queries (512) vs kv (256).
        # Actual shapes from checkpoint: q_proj [12288, 5120], o_proj [5120, 6144]
        # where 12288 = 24 * 512 (q_head_dim) and 6144 = 24 * 256 (head_dim)
        q_head_dim = head_dim * 2  # Qwen 3.5 uses double-width queries
        self.q_proj = nn.Linear(hidden_size, num_heads * q_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # MLP
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Store dims for attention
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_head_dim = q_head_dim

    def chain_forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Lightweight chain step: embed norms + fc + decoder layer + final norm.

        Args:
            input_embeds: [B, D] - already embedded token IDs (from shared embed_tokens)
            hidden_states: [B, D] - hidden states from previous layer / main model

        Returns:
            [B, D] - output hidden states ready for lm_head projection
        """
        # Step 1-4: Norm + concat + project
        embed_normed = self.pre_fc_norm_embedding(input_embeds)
        hidden_normed = self.pre_fc_norm_hidden(hidden_states)
        concat = torch.cat([embed_normed, hidden_normed], dim=-1)
        hidden = self.fc(concat)

        # Step 5: Simplified decoder layer (no causal mask, single-token "sequence")
        # For training the MTP head we process one position at a time, so
        # attention is effectively a self-projection (seq_len=1).
        residual = hidden
        hidden = self.input_layernorm(hidden)

        # Self-attention (degenerate: seq_len=1, so attention = identity transform)
        # With seq_len=1, the softmax output is always [1.0], so attention
        # simply passes through the values.  We skip Q/K computation and just
        # project through V and O.
        B = hidden.shape[0]
        v = self.v_proj(hidden).view(B, 1, self.num_kv_heads, self.head_dim)
        # GQA expand
        if self.num_kv_heads < self.num_heads:
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        attn_out = v.view(B, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        hidden = residual + attn_out

        # MLP
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        gate = F.silu(self.gate_proj(hidden))
        up = self.up_proj(hidden)
        hidden = self.down_proj(gate * up)
        hidden = residual + hidden

        # Final norm
        hidden = self.norm(hidden)
        return hidden

    def load_from_safetensors(self, path: str | Path) -> None:
        """Load weights from a sibling checkpoint (mtp_sibling_N.safetensors).

        Maps the checkpoint key names (mtp.*) to this module's parameter names.
        """
        state = load_file(str(path))
        mapping = {
            "mtp.fc.weight": "fc.weight",
            "mtp.layers.0.input_layernorm.weight": "input_layernorm.weight",
            "mtp.layers.0.mlp.down_proj.weight": "down_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight": "gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight": "up_proj.weight",
            "mtp.layers.0.post_attention_layernorm.weight": "post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.k_norm.weight": "k_norm.weight",
            "mtp.layers.0.self_attn.k_proj.weight": "k_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight": "o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight": "q_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight": "q_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight": "v_proj.weight",
            "mtp.norm.weight": "norm.weight",
            "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
        }
        own_state = self.state_dict()
        for ckpt_key, param_key in mapping.items():
            if ckpt_key in state:
                own_state[param_key].copy_(state[ckpt_key])
            else:
                print(f"  WARNING: {ckpt_key} not found in checkpoint")
        self.load_state_dict(own_state)

    def save_to_safetensors(self, path: str | Path) -> None:
        """Save weights back in the mtp.* key format for vLLM compatibility."""
        reverse_mapping = {
            "fc.weight": "mtp.fc.weight",
            "input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
            "down_proj.weight": "mtp.layers.0.mlp.down_proj.weight",
            "gate_proj.weight": "mtp.layers.0.mlp.gate_proj.weight",
            "up_proj.weight": "mtp.layers.0.mlp.up_proj.weight",
            "post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
            "k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
            "k_proj.weight": "mtp.layers.0.self_attn.k_proj.weight",
            "o_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
            "q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "q_proj.weight": "mtp.layers.0.self_attn.q_proj.weight",
            "v_proj.weight": "mtp.layers.0.self_attn.v_proj.weight",
            "norm.weight": "mtp.norm.weight",
            "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
        }
        state = self.state_dict()
        out = {}
        for param_key, ckpt_key in reverse_mapping.items():
            out[ckpt_key] = state[param_key].contiguous()
        save_file(out, str(path))


# ---------------------------------------------------------------------------
# Dataset: token-level sequences from HF datasets or raw text
# ---------------------------------------------------------------------------

class TokenSequenceDataset(Dataset):
    """Pre-tokenized sequences for MTP training.

    Each item is a dict with:
        - input_ids: [seq_len] token IDs
    The training loop slices these into (context, target) pairs where each
    MTP head k predicts token at position t+k+1 given hidden state at t.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        """
        Args:
            token_ids: [N] flat tensor of token IDs (the whole corpus concatenated)
            seq_len: length of each training sequence
        """
        self.token_ids = token_ids
        self.seq_len = seq_len
        # Number of complete sequences we can form
        self.n_sequences = (len(token_ids) - 1) // seq_len  # -1 for target offset

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # +K extra tokens for multi-position targets
        # We grab seq_len + max_offset tokens, training loop handles the rest
        end = start + self.seq_len + 1  # +1 minimum for next-token target
        return {"input_ids": self.token_ids[start:end]}


def load_dataset_tokens(
    dataset_name: str,
    tokenizer_path: str,
    max_examples: int,
    seq_len: int,
) -> TokenSequenceDataset:
    """Load and tokenize a dataset.  Returns a TokenSequenceDataset."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if dataset_name == "wikitext":
        from datasets import load_dataset
        print("Loading wikitext-103-raw-v1...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [row["text"] for row in ds if len(row["text"].strip()) > 50]
        texts = texts[:max_examples]
    elif os.path.isfile(dataset_name):
        print(f"Loading text from {dataset_name}...")
        with open(dataset_name) as f:
            raw = f.read()
        # Split into chunks ~500 chars each
        texts = [raw[i:i+500] for i in range(0, len(raw), 500)]
        texts = texts[:max_examples]
    else:
        # Try as HF dataset name
        from datasets import load_dataset
        print(f"Loading {dataset_name} from HF...")
        ds = load_dataset(dataset_name, split="train")
        # Try common text column names
        text_col = None
        for col in ["text", "content", "sentence", "document"]:
            if col in ds.column_names:
                text_col = col
                break
        if text_col is None:
            text_col = ds.column_names[0]
        texts = [row[text_col] for row in ds if len(str(row[text_col]).strip()) > 50]
        texts = texts[:max_examples]

    print(f"Tokenizing {len(texts)} examples...")
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

    token_ids = torch.tensor(all_ids, dtype=torch.long)
    print(f"Total tokens: {len(token_ids):,} ({len(token_ids) / seq_len:.0f} sequences)")

    return TokenSequenceDataset(token_ids, seq_len)


# ---------------------------------------------------------------------------
# Shared components loader (embed_tokens, lm_head)
# ---------------------------------------------------------------------------

def load_shared_weights(model_dir: str | Path) -> tuple[nn.Embedding, nn.Linear]:
    """Load embed_tokens and lm_head from the original checkpoint.

    These are shared across all sibling heads and kept frozen during training.
    """
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Load embed_tokens
    embed_key = "model.language_model.embed_tokens.weight"
    embed_shard = model_dir / weight_map[embed_key]
    print(f"Loading embed_tokens from {weight_map[embed_key]}...")
    with torch.no_grad():
        from safetensors import safe_open
        with safe_open(str(embed_shard), framework="pt", device="cpu") as f:
            embed_weight = f.get_tensor(embed_key)

    vocab_size, hidden_size = embed_weight.shape
    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    embed_tokens.weight.data.copy_(embed_weight)
    embed_tokens.weight.requires_grad_(False)

    # Load lm_head
    lm_head_key = "lm_head.weight"
    lm_head_shard = model_dir / weight_map[lm_head_key]
    print(f"Loading lm_head from {weight_map[lm_head_key]}...")
    with torch.no_grad():
        with safe_open(str(lm_head_shard), framework="pt", device="cpu") as f:
            lm_head_weight = f.get_tensor(lm_head_key)

    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    lm_head.weight.data.copy_(lm_head_weight)
    lm_head.weight.requires_grad_(False)

    print(f"  embed_tokens: {list(embed_weight.shape)} ({embed_weight.numel() * 2 / 1e6:.0f} MB)")
    print(f"  lm_head: {list(lm_head_weight.shape)} ({lm_head_weight.numel() * 2 / 1e6:.0f} MB)")

    return embed_tokens, lm_head


# ---------------------------------------------------------------------------
# Diversity loss
# ---------------------------------------------------------------------------

def diversity_penalty(logits_list: list[torch.Tensor]) -> torch.Tensor:
    """Compute mean pairwise cosine similarity between sibling heads' logits.

    Args:
        logits_list: K tensors of shape [B, V] (one per head)

    Returns:
        Scalar: mean cosine similarity across all K*(K-1)/2 pairs.
        We MAXIMIZE diversity by MINIMIZING this value.
    """
    K = len(logits_list)
    if K < 2:
        return torch.tensor(0.0, device=logits_list[0].device)

    # Normalize logits for cosine similarity
    normed = [F.normalize(l.float(), dim=-1) for l in logits_list]

    total_sim = torch.tensor(0.0, device=logits_list[0].device)
    n_pairs = 0
    for i, j in combinations(range(K), 2):
        # Cosine sim per sample, then mean over batch
        sim = (normed[i] * normed[j]).sum(dim=-1).mean()
        total_sim = total_sim + sim
        n_pairs += 1

    return total_sim / n_pairs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    heads: list[MTPHead],
    embed_tokens: nn.Embedding,
    lm_head: nn.Linear,
    dataset: TokenSequenceDataset,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_div: float = 0.1,
    device: str = "cuda",
    grad_accum_steps: int = 1,
    log_interval: int = 50,
) -> dict:
    """Train K sibling MTP heads jointly.

    The training procedure simulates multi-step drafting:
    - We take a sequence of tokens [t0, t1, ..., tN]
    - At each position t, the main model produces hidden_state h_t
    - Head k predicts token t+k+1 given (h_t, token t+k)
    - Since we don't have the real main model hidden states, we use
      embed_tokens(t) as a proxy for h_t (a common MTP training shortcut)

    This is a simplified training setup.  Production training would use
    actual hidden states from the main model's forward pass (distillation).
    """
    K = len(heads)

    # Move everything to device
    embed_tokens = embed_tokens.to(device)
    lm_head = lm_head.to(device)
    for head in heads:
        head.to(device)
        head.train()

    # Optimizer: train only the head parameters (embed/lm_head frozen)
    all_params = []
    for head in heads:
        all_params.extend(head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    # LR scheduler: cosine decay
    total_steps = epochs * len(dataset) // (batch_size * grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps), eta_min=lr * 0.1
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    stats = {
        "steps": [],
        "ce_loss": [],
        "div_loss": [],
        "total_loss": [],
    }

    step = 0
    t_start = time.monotonic()

    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # [B, seq_len+1]
            B, L = input_ids.shape

            # We need at least K+2 tokens per sequence (context + K targets)
            if L < K + 2:
                continue

            # Pick a random position in the sequence to simulate the draft point
            # Position t: hidden_state from t, each head k predicts token at t+k+1
            max_t = L - K - 1  # ensure all K targets exist
            t = torch.randint(0, max(1, max_t), (B,), device=device)

            # Gather context token and target tokens for each head
            # context_ids[b] = input_ids[b, t[b]]
            # target_ids[b, k] = input_ids[b, t[b] + k + 1]
            context_ids = input_ids[torch.arange(B, device=device), t]
            # For simplicity, use a fixed position in the middle of the sequence
            mid = L // 2
            context_ids = input_ids[:, mid]

            # Proxy hidden states: use embeddings as stand-in for main model output
            with torch.no_grad():
                hidden_states = embed_tokens(context_ids)  # [B, D]

            # Run each head and compute losses
            all_logits = []
            total_ce = torch.tensor(0.0, device=device)

            for k, head in enumerate(heads):
                target_pos = mid + k + 1
                if target_pos >= L:
                    break

                # The head sees the token at position mid+k (the previous draft token)
                draft_input_ids = input_ids[:, mid + k]
                with torch.no_grad():
                    draft_embeds = embed_tokens(draft_input_ids)  # [B, D]

                # Forward through head
                out_hidden = head.chain_forward(draft_embeds, hidden_states)
                logits = F.linear(out_hidden.float(), lm_head.weight.float())  # [B, V]
                all_logits.append(logits)

                # CE loss: predict the next token at position mid+k+1
                targets = input_ids[:, target_pos]  # [B]
                ce = F.cross_entropy(logits, targets)
                total_ce = total_ce + ce

                # Update hidden_states for the next head in the chain
                hidden_states = out_hidden.detach()

            if len(all_logits) == 0:
                continue

            # Average CE over heads
            ce_loss = total_ce / len(all_logits)

            # Diversity penalty (only on logits at the SAME position for all heads)
            # Re-run all heads from the same starting point to get comparable logits
            with torch.no_grad():
                shared_hidden = embed_tokens(context_ids)
                shared_draft_embeds = embed_tokens(input_ids[:, mid])

            same_pos_logits = []
            for head in heads:
                out = head.chain_forward(shared_draft_embeds, shared_hidden)
                logits = F.linear(out.float(), lm_head.weight.float())
                same_pos_logits.append(logits)

            div_loss = diversity_penalty(same_pos_logits)

            # Total loss
            loss = ce_loss + lambda_div * div_loss

            # Backward
            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

            if step % log_interval == 0:
                elapsed = time.monotonic() - t_start
                print(
                    f"[epoch {epoch+1}/{epochs}, step {step}] "
                    f"CE={ce_loss.item():.4f} div={div_loss.item():.4f} "
                    f"total={loss.item() * grad_accum_steps:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"({elapsed:.0f}s)"
                )
                stats["steps"].append(step)
                stats["ce_loss"].append(ce_loss.item())
                stats["div_loss"].append(div_loss.item())
                stats["total_loss"].append(loss.item() * grad_accum_steps)

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diversity fine-tune sibling MTP heads")
    parser.add_argument("--siblings-dir", type=str, default="/home/ubuntu/models/mtp-siblings",
                        help="Directory with mtp_sibling_*.safetensors from mtp_clone.py")
    parser.add_argument("--model-dir", type=str, default="/home/ubuntu/models/Qwen3.5-27B",
                        help="Original Qwen3.5-27B for embed_tokens/lm_head and config")
    parser.add_argument("--output-dir", type=str, default="/home/ubuntu/models/mtp-siblings-trained",
                        help="Output directory for fine-tuned heads")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name (wikitext, HF name, or path to .txt file)")
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-div", type=float, default=0.1,
                        help="Weight for diversity penalty")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    siblings_dir = Path(args.siblings_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    text_config = config["text_config"]
    hidden_size = text_config["hidden_size"]          # 5120
    intermediate_size = text_config["intermediate_size"]  # 17408
    num_heads = text_config["num_attention_heads"]    # 24
    num_kv_heads = text_config["num_key_value_heads"] # 4
    head_dim = text_config["head_dim"]                # 256
    vocab_size = text_config["vocab_size"]            # 248320
    rms_norm_eps = text_config["rms_norm_eps"]        # 1e-6

    print(f"Model config: hidden={hidden_size}, inter={intermediate_size}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"vocab={vocab_size}")

    # Load manifest to discover sibling heads
    manifest_path = siblings_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    K = manifest["num_heads"]
    print(f"\nLoading {K} sibling MTP heads...")

    # Create and load heads
    heads: list[MTPHead] = []
    for head_info in manifest["heads"]:
        idx = head_info["index"]
        filepath = siblings_dir / head_info["file"]
        print(f"  Head {idx} (sigma={head_info['sigma']}) from {filepath}")
        head = MTPHead(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
        )
        head.load_from_safetensors(filepath)
        heads.append(head)

    # Load shared weights (frozen)
    print("\nLoading shared weights (embed_tokens, lm_head)...")
    embed_tokens, lm_head = load_shared_weights(model_dir)

    # Load dataset
    print(f"\nPreparing dataset ({args.dataset}, max {args.max_examples} examples)...")
    dataset = load_dataset_tokens(
        args.dataset,
        str(model_dir),
        max_examples=args.max_examples,
        seq_len=args.seq_len,
    )

    # Train
    print(f"\nStarting training: {args.epochs} epoch(s), batch={args.batch_size}, "
          f"lr={args.lr}, lambda_div={args.lambda_div}")
    stats = train(
        heads=heads,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_div=args.lambda_div,
        device=args.device,
        grad_accum_steps=args.grad_accum,
        log_interval=args.log_interval,
    )

    # Save trained heads
    print(f"\nSaving trained heads to {output_dir}...")
    for i, head in enumerate(heads):
        head.cpu()
        filepath = output_dir / f"mtp_sibling_{i}.safetensors"
        head.save_to_safetensors(filepath)
        mb = filepath.stat().st_size / 1e6
        print(f"  Head {i}: {filepath} ({mb:.1f} MB)")

    # Save training manifest
    train_manifest = {
        "source_siblings": str(siblings_dir),
        "source_model": str(model_dir),
        "num_heads": K,
        "training": {
            "dataset": args.dataset,
            "max_examples": args.max_examples,
            "epochs": args.epochs,
            "lr": args.lr,
            "lambda_div": args.lambda_div,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
        },
        "stats": {
            "final_ce_loss": stats["ce_loss"][-1] if stats["ce_loss"] else None,
            "final_div_loss": stats["div_loss"][-1] if stats["div_loss"] else None,
            "total_steps": stats["steps"][-1] if stats["steps"] else 0,
        },
    }
    with open(output_dir / "training_manifest.json", "w") as f:
        json.dump(train_manifest, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
