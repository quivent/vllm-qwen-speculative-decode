#!/usr/bin/env python3
"""
Early-exit verification probe for Qwen 3.5-27B.

Strategies:
  1. Raw PLV: apply final_norm + lm_head to intermediate hidden states
  2. Regression adapter: train MLP(h_L) -> h_63, then eval via norm+lm_head
  3. Cosine similarity: alignment between intermediate and final representations

Two-phase design:
  Phase 1: Load model, collect hidden states + raw PLV, cache to disk
  Phase 2: Train adapters from cached data (no model needed)
"""

import os, sys, time, json, copy, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

MODEL_PATH = "/home/ubuntu/models/Qwen3.5-27B"
HIDDEN_DIM = 5120
VOCAB_SIZE = 248320

PROBE_LAYERS = [3, 7, 11, 15, 31, 47]
ALL_CAPTURE_LAYERS = PROBE_LAYERS + [63]

PROMPTS = [
    "The history of artificial intelligence began in the 1950s when Alan Turing proposed the question of whether machines can think. This led to the development of early programs that could play chess and prove mathematical theorems. In the following decades, researchers explored symbolic AI, expert systems, and eventually neural networks. The field experienced several winters where funding dried up and progress stalled. However, the advent of deep learning in the 2010s, powered by large datasets and GPU computing, led to breakthroughs in image recognition, natural language processing, and game playing. Today, large language models represent the latest wave of AI advancement, capable of generating human-like text, writing code, and reasoning about complex problems.",
    "Quantum computing represents a fundamentally different approach to computation that leverages quantum mechanical phenomena such as superposition and entanglement. Unlike classical bits that exist as either 0 or 1, quantum bits or qubits can exist in superposition of both states simultaneously. This property, combined with entanglement where qubits become correlated in ways that have no classical analog, enables quantum computers to process certain types of problems exponentially faster than classical computers. Key algorithms include Shor's algorithm for factoring large numbers and Grover's algorithm for searching unsorted databases. Current quantum computers face significant challenges including decoherence, error rates, and the need for extremely low temperatures.",
    "The evolution of programming languages reflects the changing needs and capabilities of computing over the past seven decades. Assembly language gave programmers direct control over hardware but was tedious and error-prone. Fortran and COBOL introduced high-level abstractions for scientific and business computing respectively. C provided a powerful balance of low-level access and high-level constructs, becoming the foundation for operating systems. Object-oriented languages like C++ and Java organized code around data structures and their associated operations. Functional languages like Haskell emphasized immutability and mathematical purity. Python's simplicity and extensive libraries made it dominant in data science and machine learning. Rust introduced ownership semantics to prevent memory safety bugs at compile time without garbage collection overhead.",
    "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections with other neurons. This vast network enables consciousness, memory, language, creativity, and all forms of cognition. Neuroscientists have mapped many brain regions to specific functions: the prefrontal cortex handles executive functions and planning, the hippocampus is critical for forming new memories, Broca's and Wernicke's areas process language production and comprehension, and the cerebellum coordinates fine motor control. Neural plasticity allows the brain to reorganize itself throughout life, forming new connections in response to learning and experience. Understanding how computation emerges from biological neural networks remains one of science's greatest challenges.",
    "Climate change is driven primarily by the greenhouse effect, where gases like carbon dioxide, methane, and nitrous oxide trap heat in Earth's atmosphere. Since the Industrial Revolution, human activities including burning fossil fuels, deforestation, and industrial agriculture have increased atmospheric CO2 from about 280 parts per million to over 420 ppm. This has already caused approximately 1.1 degrees Celsius of global warming, leading to rising sea levels, more frequent extreme weather events, ocean acidification, and shifts in ecosystems. The Paris Agreement aims to limit warming to 1.5 degrees above pre-industrial levels, requiring rapid decarbonization of energy systems, transportation, and industry.",
    "import numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader, Dataset\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, num_heads, dropout=0.1):\n        super().__init__()\n        self.d_model = d_model\n        self.num_heads = num_heads\n        self.d_k = d_model // num_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n        self.dropout = nn.Dropout(dropout)\n\n    def forward(self, query, key, value, mask=None):\n        batch_size = query.size(0)\n        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)\n        if mask is not None:\n            scores = scores.masked_fill(mask == 0, -1e9)\n        attention = F.softmax(scores, dim=-1)\n        attention = self.dropout(attention)\n        context = torch.matmul(attention, V)\n        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)\n        return self.W_o(context)",
    "The theory of general relativity, published by Albert Einstein in 1915, fundamentally changed our understanding of gravity. Rather than being a force acting at a distance as Newton described, gravity is the curvature of spacetime caused by mass and energy. This framework predicted several phenomena that have since been confirmed: the bending of light around massive objects (gravitational lensing), the existence of black holes where spacetime curvature becomes infinite, gravitational time dilation where clocks run slower in stronger gravitational fields, and gravitational waves which are ripples in spacetime caused by accelerating massive objects. The detection of gravitational waves by LIGO in 2015, a century after Einstein's prediction, confirmed one of the theory's most dramatic consequences.",
    "Database design principles have evolved significantly since Edgar Codd introduced the relational model in 1970. Normalization rules help eliminate data redundancy and maintain consistency. The ACID properties of transactions ensure atomicity, consistency, isolation, and durability. However, as web-scale applications emerged, the limitations of relational databases led to the NoSQL movement. Document stores like MongoDB offer flexible schemas, key-value stores like Redis provide extreme performance for simple lookups, column-family stores like Cassandra handle massive write loads, and graph databases like Neo4j excel at relationship-heavy queries. The CAP theorem demonstrates that distributed systems must choose between consistency, availability, and partition tolerance. Modern architectures often use polyglot persistence, choosing the right database for each use case.",
    "The immune system is a complex network of cells, tissues, and organs that defends the body against pathogens. Innate immunity provides immediate, non-specific defense through barriers like skin and mucous membranes, as well as cells like neutrophils and macrophages that engulf invaders. Adaptive immunity develops over time and provides targeted responses through T cells and B cells. B cells produce antibodies that bind to specific antigens on pathogens, marking them for destruction. T helper cells coordinate immune responses, while cytotoxic T cells directly kill infected cells. Memory cells persist after infection, enabling faster and stronger responses upon re-exposure to the same pathogen. Vaccines exploit this memory mechanism by exposing the immune system to harmless versions of pathogens, training it to respond quickly to future infections.",
    "The architecture of modern microprocessors has evolved dramatically since the first Intel 4004 in 1971. Moore's Law predicted that transistor density would double approximately every two years, driving exponential improvements in computing power. Modern CPUs employ sophisticated techniques including pipelining, out-of-order execution, branch prediction, and speculative execution to maximize instruction throughput. Multi-core designs allow parallel processing of independent workloads. Cache hierarchies with L1, L2, and L3 caches reduce memory access latency. SIMD instructions enable data-parallel operations crucial for multimedia and scientific computing. GPU architectures take parallelism further with thousands of simple cores optimized for throughput rather than latency. The end of Dennard scaling and the slowing of Moore's Law have driven interest in specialized accelerators like TPUs for machine learning and FPGAs for configurable logic.",
    "Distributed systems present unique challenges that arise from the fundamental constraints of networked computing. The fallacies of distributed computing remind us that the network is not reliable, latency is not zero, bandwidth is not infinite, the network is not secure, topology does change, there is not one administrator, transport cost is not zero, and the network is not homogeneous. Consensus protocols like Paxos and Raft enable multiple nodes to agree on values despite failures. Byzantine fault tolerance handles malicious actors. Eventual consistency models trade immediate consistency for availability and performance. Vector clocks and Lamport timestamps establish causal ordering of events. Service mesh architectures provide observability, traffic management, and security between microservices.",
    "Organic chemistry studies carbon-based compounds and their reactions. Carbon's ability to form four covalent bonds and create stable chains and rings makes it uniquely suited as the backbone of biological molecules. Functional groups determine the chemical properties of organic molecules: hydroxyl groups make alcohols, carboxyl groups create acids, amino groups form the basis of proteins, and phosphate groups are essential in nucleic acids and energy transfer. Reaction mechanisms describe the step-by-step process of bond breaking and formation. Nucleophilic substitution, electrophilic addition, elimination, and rearrangement reactions represent fundamental organic transformations. Understanding these mechanisms enables chemists to design synthetic routes for pharmaceuticals, polymers, and countless other materials that underpin modern civilization.",
]


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, h):
        variance = h.float().pow(2).mean(-1, keepdim=True)
        h = h.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * h).to(h.dtype)


class RegressionAdapter(nn.Module):
    """2-layer MLP: h_L + MLP(h_L) -> predicted h_63"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight, gain=0.1)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, h):
        return h + self.net(h)


def check_gpu():
    if not torch.cuda.is_available():
        return False
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU free memory: {free:.1f} GB")
    return free > 60


def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading tokenizer...")
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


def find_layers(m):
    for a in [lambda m: m.model.layers, lambda m: m.language_model.model.layers]:
        try: return a(m)
        except AttributeError: pass
    raise RuntimeError("no layers")

def find_lm_head(m):
    for a in [lambda m: m.lm_head, lambda m: m.language_model.lm_head]:
        try: return a(m)
        except AttributeError: pass
    raise RuntimeError("no lm_head")

def find_norm(m):
    for a in [lambda m: m.model.norm, lambda m: m.language_model.model.norm]:
        try: return a(m)
        except AttributeError: pass
    return None


def phase1_collect(device):
    """Collect hidden states + raw PLV + cache everything."""
    save_dir = Path("/home/ubuntu/aut/probes")
    save_dir.mkdir(exist_ok=True)
    cache = save_dir / "cached_hiddens.pt"

    if cache.exists():
        print(f"Cache found at {cache}, skipping phase 1")
        return torch.load(cache, map_location="cpu", weights_only=False)

    model, tokenizer = load_model(device)
    layers = find_layers(model)
    lm_head = find_lm_head(model)
    norm = find_norm(model)

    hook_data = {}
    def make_hook(idx):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hook_data[idx] = h.detach().cpu().float()
        return fn

    hooks = [layers[l].register_forward_hook(make_hook(l)) for l in ALL_CAPTURE_LAYERS]

    all_argmax = []
    all_h = {l: [] for l in ALL_CAPTURE_LAYERS}
    raw_plv_correct = {l: 0 for l in PROBE_LAYERS}
    raw_plv_total = {l: 0 for l in PROBE_LAYERS}
    total = 0

    print(f"\nForward pass on {len(PROMPTS)} prompts...")
    for i, p in enumerate(PROMPTS):
        inp = tokenizer(p, return_tensors="pt").to(device)
        seq = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**inp)
        am = out.logits[0].argmax(-1).cpu()
        all_argmax.append(am)
        for l in ALL_CAPTURE_LAYERS:
            all_h[l].append(hook_data[l][0])
        total += seq
        print(f"  [{i+1}/{len(PROMPTS)}] {seq} tok (cumul {total})")

    for h in hooks:
        h.remove()

    # Compute raw PLV inline (one pass per prompt already done, now just matmul)
    print("\nComputing raw PLV...")
    raw_plv = {}
    for l in PROBE_LAYERS:
        correct = tot = 0
        for h, am in zip(all_h[l], all_argmax):
            with torch.no_grad():
                normed = norm(h.to(torch.bfloat16).to(device))
                # Process in chunks to avoid huge intermediate tensors
                chunk_size = 64
                for cs in range(0, normed.shape[0], chunk_size):
                    chunk = normed[cs:cs+chunk_size]
                    pred = lm_head(chunk).argmax(-1).cpu()
                    am_chunk = am[cs:cs+chunk_size]
                    correct += (pred == am_chunk).sum().item()
                    tot += am_chunk.shape[0]
        raw_plv[l] = correct / tot
        print(f"  L{l:2d}: {raw_plv[l]*100:.1f}%")

    # Extract norm + lm_head weights
    norm_sd = {k: v.cpu().float().clone() for k, v in norm.state_dict().items()}
    lm_head_sd = {k: v.cpu().float().clone() for k, v in lm_head.state_dict().items()}

    cat_am = torch.cat(all_argmax)
    cat_h = {l: torch.cat(all_h[l]) for l in ALL_CAPTURE_LAYERS}

    # Free model
    del model, layers, lm_head, norm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    data = {
        "hiddens": cat_h,
        "argmax": cat_am,
        "raw_plv": raw_plv,
        "norm_sd": norm_sd,
        "lm_head_sd": lm_head_sd,
        "total_tokens": total,
    }

    print(f"Saving cache ({total} tokens)...")
    torch.save(data, cache)
    print(f"Cache saved to {cache}")
    return data


def phase2_train(data):
    """Train adapters + evaluate."""
    hiddens = data["hiddens"]
    argmax = data["argmax"]
    raw_plv = data["raw_plv"]
    n = argmax.shape[0]

    # Reconstruct frozen norm + lm_head
    frozen_norm = SimpleRMSNorm(HIDDEN_DIM)
    frozen_norm.load_state_dict(data["norm_sd"])
    frozen_norm.eval()

    frozen_lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
    frozen_lm_head.load_state_dict(data["lm_head_sd"])
    frozen_lm_head.eval()

    h_final = hiddens[63].float()

    # Cosine similarity
    print("\n--- Cosine similarity with final layer ---")
    cos_sims = {}
    for l in PROBE_LAYERS:
        cos = F.cosine_similarity(hiddens[l].float(), h_final, dim=-1).mean().item()
        cos_sims[l] = cos
        print(f"  L{l:2d}: {cos:.4f}")

    # L2 distance
    print("\n--- L2 distance to final layer ---")
    l2_dists = {}
    for l in PROBE_LAYERS:
        d = (hiddens[l].float() - h_final).norm(dim=-1).mean().item()
        l2_dists[l] = d
        print(f"  L{l:2d}: {d:.2f}")

    # Train regression adapters
    print("\n" + "=" * 60)
    print("Training regression adapters (h_L -> h_63)")
    print("=" * 60)

    adapter_accs = {}
    probes = {}

    for l in PROBE_LAYERS:
        print(f"\nLayer {l} (full-attn #{(l+1)//4}):")
        h_l = hiddens[l].float()

        # Train/val split
        perm = torch.randperm(n)
        split = int(0.8 * n)
        tr_s, tr_t = h_l[perm[:split]], h_final[perm[:split]]
        va_s, va_t = h_l[perm[split:]], h_final[perm[split:]]
        va_am = argmax[perm[split:]]

        adapter = RegressionAdapter(HIDDEN_DIM)
        n_params = sum(p.numel() for p in adapter.parameters())
        print(f"  Params: {n_params:,} ({n_params/1e6:.1f}M)")

        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

        for ep in range(50):
            adapter.train()
            idx = torch.randperm(tr_s.shape[0])
            tloss = nb = 0
            for st in range(0, tr_s.shape[0], 256):
                bi = idx[st:st+256]
                pred = adapter(tr_s[bi])
                loss = F.mse_loss(pred, tr_t[bi])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tloss += loss.item()
                nb += 1
            sched.step()

            if (ep+1) % 10 == 0 or ep == 0:
                adapter.eval()
                with torch.no_grad():
                    val_pred = adapter(va_s)
                    val_mse = F.mse_loss(val_pred, va_t).item()
                    cos = F.cosine_similarity(val_pred, va_t, dim=-1).mean().item()
                print(f"  Ep {ep+1:3d}: train_mse={tloss/nb:.6f} val_mse={val_mse:.6f} val_cos={cos:.4f}")

        # Evaluate adapter accuracy (adapter -> norm -> lm_head -> argmax)
        adapter.eval()
        with torch.no_grad():
            adapted = adapter(va_s)
            correct = 0
            total_eval = 0
            for st in range(0, adapted.shape[0], 64):
                chunk = adapted[st:st+64]
                normed = frozen_norm(chunk)
                logits = frozen_lm_head(normed)
                pred = logits.argmax(-1)
                correct += (pred == va_am[st:st+64]).sum().item()
                total_eval += chunk.shape[0]
        acc = correct / total_eval
        adapter_accs[l] = acc
        probes[l] = adapter
        print(f"  >>> Adapter p_agree: {acc*100:.1f}%")

        # Also evaluate: what's raw PLV on just the val set for comparison
        with torch.no_grad():
            correct_raw = 0
            for st in range(0, va_s.shape[0], 64):
                chunk = va_s[st:st+64]
                normed = frozen_norm(chunk)
                logits = frozen_lm_head(normed)
                pred = logits.argmax(-1)
                correct_raw += (pred == va_am[st:st+64]).sum().item()
        raw_val = correct_raw / total_eval
        print(f"  >>> Raw PLV (val set): {raw_val*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Layer':>6} {'FA#':>4} {'Raw PLV':>9} {'Adapter':>9} {'Cos(L,63)':>10} {'L2(L,63)':>9}")
    print("-" * 50)
    for l in PROBE_LAYERS:
        fa = (l+1) // 4
        r = raw_plv.get(l, 0)
        a = adapter_accs.get(l, 0)
        c = cos_sims.get(l, 0)
        d = l2_dists.get(l, 0)
        print(f"{l:>6} {fa:>4} {r*100:>8.1f}% {a*100:>8.1f}% {c:>10.4f} {d:>9.2f}")

    # Save
    save_dir = Path("/home/ubuntu/aut/probes")
    for l, p in probes.items():
        torch.save(p.state_dict(), save_dir / f"regression_adapter_L{l}.pt")

    results = {
        "probe_layers": PROBE_LAYERS,
        "raw_plv": {str(l): raw_plv.get(l) for l in PROBE_LAYERS},
        "adapter_p_agree": {str(l): adapter_accs[l] for l in PROBE_LAYERS},
        "cosine_sim_with_L63": {str(l): cos_sims[l] for l in PROBE_LAYERS},
        "l2_dist_to_L63": {str(l): l2_dists[l] for l in PROBE_LAYERS},
        "total_tokens": n,
        "num_prompts": len(PROMPTS),
        "model": MODEL_PATH,
        "adapter_arch": "Linear(5120,10240)->GELU->Linear(10240,5120)+residual, 52M params",
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_dir}/")


def main():
    print("=" * 60)
    print("Early-Exit Verification Probe for Qwen 3.5-27B")
    print("=" * 60)

    device = "cuda" if check_gpu() else "cpu"
    print(f"Device: {device}\n")

    data = phase1_collect(device)
    phase2_train(data)


if __name__ == "__main__":
    main()
