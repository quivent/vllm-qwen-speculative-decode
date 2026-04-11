#!/usr/bin/env python3
"""Cascade MTP training with corrective hidden state loss.

Each depth head is trained with TWO objectives:
  1. CE: predict the correct next token
  2. MSE: produce a hidden state close to the IDEAL (full model) hidden state

The MSE term is critical — it forces each head to reduce accumulated drift
so the next head in the cascade gets a clean input. Without it, heads
optimize their own prediction but let the hidden state degrade further.

Loss = CE(token_pred, target) + lambda_h * MSE(output_hidden, ideal_hidden)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


class CascadeMTPHead(nn.Module):
    """Single MTP head matching Qwen3.5's chain_forward path.

    chain_forward does:
      1. embed = embed_tokens(token_id)
      2. embed_normed = pre_fc_norm_embedding(embed)
      3. hidden_normed = pre_fc_norm_hidden(hidden_state)
      4. h = fc(cat(hidden_normed, embed_normed))  # [hidden*2] -> [hidden]
      5. h = norm(h)

    No decoder layer in chain_forward — it's skipped for speed.
    """

    def __init__(self, hidden_size=5120, intermediate_size=10240):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=1e-6)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=1e-6)
        self.fc = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_state, token_embedding):
        """Run chain_forward path. Returns output hidden state."""
        h_normed = self.pre_fc_norm_hidden(hidden_state)
        e_normed = self.pre_fc_norm_embedding(token_embedding)
        combined = torch.cat([h_normed, e_normed], dim=-1)
        h = self.fc(combined)
        h = self.norm(h)
        return h

    @classmethod
    def from_safetensors(cls, path, hidden_size=5120):
        head = cls(hidden_size=hidden_size, intermediate_size=hidden_size * 2)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                # Map mtp.X -> module attribute
                short = key.replace("mtp.", "")
                if short == "fc.weight":
                    head.fc.weight.data.copy_(f.get_tensor(key))
                elif short == "norm.weight":
                    head.norm.weight.data.copy_(f.get_tensor(key))
                elif short == "pre_fc_norm_hidden.weight":
                    head.pre_fc_norm_hidden.weight.data.copy_(f.get_tensor(key))
                elif short == "pre_fc_norm_embedding.weight":
                    head.pre_fc_norm_embedding.weight.data.copy_(f.get_tensor(key))
        return head


def collect_ideal_hidden_states(model, tokenizer, prompts, max_tokens=50, device="cpu"):
    """Run full model, capture hidden states at each token position.

    Returns list of dicts, each with:
      - input_ids: [seq_len]
      - hidden_states: [seq_len, hidden_size] from the model backbone output
      - token_ids: [seq_len] the token at each position
    """
    backbone, lm_head, norm = get_model_parts(model)

    all_data = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        ids = inputs["input_ids"][0]  # [seq_len]

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)

        # Last hidden state before lm_head (after final norm in model)
        # This is the IDEAL hidden state at each position
        last_hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden]

        all_data.append({
            "input_ids": ids,
            "hidden_states": last_hidden.detach(),
        })

    return all_data


def get_model_parts(model):
    if hasattr(model, 'language_model'):
        backbone = model.language_model.model
        lm_head = model.language_model.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        backbone = model.model
        lm_head = model.lm_head
    else:
        raise ValueError("Cannot find model backbone")
    norm = backbone.norm
    return backbone, lm_head, norm


def train_cascade(
    model_path: str,
    stock_head_path: str,
    output_dir: str,
    num_depths: int = 7,
    num_prompts: int = 500,
    max_tokens: int = 50,
    epochs: int = 3,
    lr: float = 5e-5,
    lambda_h: float = 1.0,
    batch_size: int = 8,
    device: str = "cpu",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device if device == "cpu" else "auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    # Load shared weights
    print("Loading embed_tokens and lm_head...")
    backbone, lm_head, norm = get_model_parts(model)
    embed_tokens = backbone.embed_tokens

    # Get prompts from wikitext
    print("Loading dataset...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    prompts = [t for t in ds["text"] if len(t) > 100][:num_prompts]
    print(f"Using {len(prompts)} prompts")

    # Collect ideal hidden states from full model
    print("Collecting ideal hidden states from full model...")
    ideal_data = collect_ideal_hidden_states(model, tokenizer, prompts, max_tokens, device)

    # Build training pairs: for each position, we need
    # (prev_hidden, token_embedding, ideal_next_hidden, target_token)
    print("Building training pairs...")
    all_pairs = []
    for d in ideal_data:
        ids = d["input_ids"]        # [seq_len]
        h = d["hidden_states"]      # [seq_len, hidden]
        seq_len = ids.shape[0]

        for pos in range(seq_len - 1):
            all_pairs.append({
                "hidden": h[pos].detach(),
                "token_id": ids[pos + 1].item(),  # next token
                "ideal_hidden": h[pos + 1].detach() if pos + 1 < seq_len else h[pos].detach(),
                "target_token": ids[pos + 1].item() if pos + 1 < seq_len else ids[pos].item(),
            })

    print(f"Total training pairs: {len(all_pairs)}")

    # Free the full model
    del model
    torch.cuda.empty_cache() if device != "cpu" else None

    # Load stock MTP head as starting point
    print(f"Loading stock MTP head from {stock_head_path}...")

    # Train each depth head
    prev_heads = []

    for depth in range(num_depths):
        print(f"\n{'='*60}")
        print(f"Training depth {depth} head")
        print(f"{'='*60}")

        # Initialize from stock weights
        head = CascadeMTPHead.from_safetensors(stock_head_path)
        head = head.to(device).to(torch.bfloat16)
        head.train()

        optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

        for epoch in range(epochs):
            total_ce = 0
            total_mse = 0
            total_correct = 0
            total_tokens = 0

            # Shuffle pairs
            import random
            random.shuffle(all_pairs)

            for i in range(0, len(all_pairs) - batch_size, batch_size):
                batch = all_pairs[i:i + batch_size]

                # Get hidden states — either from ideal (depth 0) or from chain
                hidden_batch = []
                ideal_next_batch = []
                token_id_batch = []
                target_batch = []

                for pair in batch:
                    h = pair["hidden"].to(device)

                    # For depth > 0: run through previous heads to get drifted state
                    if depth > 0:
                        with torch.no_grad():
                            current_h = h.unsqueeze(0)
                            for d, prev_head in enumerate(prev_heads[:depth]):
                                # Get the token embedding for this depth
                                tok_id = pair["token_id"]
                                tok_emb = embed_tokens(torch.tensor([tok_id], device=device))
                                current_h = prev_head(current_h, tok_emb)
                            h = current_h.squeeze(0)

                    hidden_batch.append(h)
                    ideal_next_batch.append(pair["ideal_hidden"].to(device))
                    token_id_batch.append(pair["token_id"])
                    target_batch.append(pair["target_token"])

                hidden_t = torch.stack(hidden_batch)  # [B, hidden]
                ideal_t = torch.stack(ideal_next_batch)  # [B, hidden]
                tok_ids = torch.tensor(token_id_batch, device=device)
                targets = torch.tensor(target_batch, device=device)

                # Get token embeddings
                tok_emb = embed_tokens(tok_ids)  # [B, hidden]

                # Forward through this depth's head
                output_h = head(hidden_t, tok_emb)  # [B, hidden]

                # Token prediction loss
                logits = F.linear(output_h.float(), lm_head.weight.float())  # [B, vocab]
                ce_loss = F.cross_entropy(logits, targets)

                # Hidden state correction loss
                mse_loss = F.mse_loss(output_h.float(), ideal_t.float())

                # Combined loss
                loss = ce_loss + lambda_h * mse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_ce += ce_loss.item()
                total_mse += mse_loss.item()
                preds = logits.argmax(dim=-1)
                total_correct += (preds == targets).sum().item()
                total_tokens += len(batch)

            n_steps = max(1, total_tokens // batch_size)
            acc = total_correct / max(1, total_tokens)
            print(f"  Epoch {epoch+1}/{epochs}: CE={total_ce/n_steps:.4f} "
                  f"MSE={total_mse/n_steps:.4f} acc={acc:.4f}")

        # Save this depth's head
        head.eval()
        state = {}
        for name, param in head.named_parameters():
            state[f"mtp.{name}"] = param.data.cpu()

        save_path = os.path.join(output_dir, f"depth_{depth}.safetensors")
        save_file(state, save_path)
        print(f"  Saved to {save_path}")

        # Add to chain for next depth's training
        prev_heads.append(head)

    # Save manifest
    manifest = {
        "num_depths": num_depths,
        "lambda_h": lambda_h,
        "epochs": epochs,
        "lr": lr,
        "num_prompts": num_prompts,
        "training_pairs": len(all_pairs),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCascade complete. {num_depths} heads saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.5-27B")
    parser.add_argument("--stock-head", default="/home/ubuntu/models/sibling-mtp-heads/mtp_sibling_0.safetensors")
    parser.add_argument("--output-dir", default="/home/ubuntu/models/cascade-mtp-heads")
    parser.add_argument("--num-depths", type=int, default=7)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda-h", type=float, default=1.0, help="Weight for hidden state MSE loss")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_cascade(
        model_path=args.model_path,
        stock_head_path=args.stock_head,
        output_dir=args.output_dir,
        num_depths=args.num_depths,
        num_prompts=args.num_prompts,
        epochs=args.epochs,
        lr=args.lr,
        lambda_h=args.lambda_h,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
