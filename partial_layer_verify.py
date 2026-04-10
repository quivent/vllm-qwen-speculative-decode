"""Partial-Layer Verification (PLV) for Qwen 3.5-27B on vLLM 0.19.

Instead of running all 64 transformer layers for draft-token verification,
run only the first N layers. If the early-exit logits agree with the draft
tokens, accept without running layers N+1..63. Fall back to full verification
every Mth step or on disagreement.

Requires --enforce-eager mode (no CUDA graphs) for the conditional branching.

Usage as monkey-patch:
    from partial_layer_verify import install_plv
    install_plv(engine)  # patches the model in-place

Usage as benchmark:
    python partial_layer_verify.py --model-path /path/to/model [--exit-layer 32]
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from itertools import islice
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLV_EXIT_LAYER = int(os.environ.get("PLV_EXIT_LAYER", "32"))
PLV_FULL_VERIFY_INTERVAL = int(os.environ.get("PLV_FULL_VERIFY_INTERVAL", "8"))
PLV_AGREEMENT_THRESHOLD = float(os.environ.get("PLV_AGREEMENT_THRESHOLD", "0.90"))


@dataclass
class PLVStats:
    """Running statistics for partial-layer verification."""

    partial_calls: int = 0
    full_calls: int = 0
    agreements: int = 0
    disagreements: int = 0
    total_tokens_verified: int = 0
    partial_time_ns: int = 0
    full_time_ns: int = 0

    @property
    def p_agree(self) -> float:
        total = self.agreements + self.disagreements
        return self.agreements / total if total > 0 else 0.0

    @property
    def speedup_ratio(self) -> float:
        if self.full_time_ns == 0 or self.full_calls == 0:
            return 0.0
        avg_full = self.full_time_ns / self.full_calls
        avg_partial = self.partial_time_ns / max(self.partial_calls, 1)
        return avg_full / avg_partial if avg_partial > 0 else 0.0

    def summary(self) -> str:
        total = self.partial_calls + self.full_calls
        return (
            f"PLV stats: {total} verifications "
            f"({self.partial_calls} partial, {self.full_calls} full), "
            f"p_agree={self.p_agree:.4f}, "
            f"speedup={self.speedup_ratio:.2f}x, "
            f"tokens={self.total_tokens_verified}"
        )


# ---------------------------------------------------------------------------
# Core: early-exit forward for Qwen3_5Model / Qwen3NextModel
# ---------------------------------------------------------------------------

def _early_exit_forward(
    model: nn.Module,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    exit_layer: int,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run only the first `exit_layer` layers of a Qwen3_5Model/Qwen3NextModel,
    then apply the final RMSNorm and return hidden states ready for lm_head.

    This bypasses the @support_torch_compile decorator by calling the
    internal components directly. Requires --enforce-eager.
    """
    # Embedding
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = model.embed_tokens(input_ids)
    residual = None

    # Run layers 0..exit_layer-1 only
    end = min(exit_layer, model.end_layer)
    for layer_idx, layer in enumerate(
        islice(model.layers, model.start_layer, end),
        start=model.start_layer,
    ):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

    # Apply final RMSNorm — lm_head expects normalized hidden states.
    # model.norm is GemmaRMSNorm (aliased as Qwen3_5RMSNorm).
    # With residual: norm(hidden_states, residual) -> (normalized, residual)
    hidden_states, _ = model.norm(hidden_states, residual)
    return hidden_states


def _full_forward(
    model: nn.Module,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run all layers — the standard forward path. Returns normalized hidden states."""
    result = model(input_ids, positions, intermediate_tensors, inputs_embeds)
    # model.forward returns hidden_states (or tuple with aux). Unpack if needed.
    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# PLVController — decides partial vs full, tracks stats
# ---------------------------------------------------------------------------

class PLVController:
    """Decides when to use early-exit verification vs full verification.

    Policy:
    - Every `full_verify_interval` steps, run full (to maintain calibration).
    - Otherwise, run partial (early-exit at layer N).
    - If partial disagrees with the draft token, the caller should reject and
      re-verify with full. We track this to adapt.
    - If p_agree drops below threshold, switch to full-only mode.
    """

    def __init__(
        self,
        exit_layer: int = PLV_EXIT_LAYER,
        full_verify_interval: int = PLV_FULL_VERIFY_INTERVAL,
        agreement_threshold: float = PLV_AGREEMENT_THRESHOLD,
    ):
        self.exit_layer = exit_layer
        self.full_verify_interval = full_verify_interval
        self.agreement_threshold = agreement_threshold
        self.stats = PLVStats()
        self._step = 0
        self._force_full = False

    @property
    def should_use_partial(self) -> bool:
        """Whether to use partial verification on this step."""
        if self._force_full:
            return False
        if self._step % self.full_verify_interval == 0:
            return False  # periodic full check
        return True

    def step(self):
        """Advance the step counter."""
        self._step += 1

    def record_agreement(self, agreed: bool, n_tokens: int = 1):
        """Record whether partial-exit logits agreed with full."""
        self.stats.total_tokens_verified += n_tokens
        if agreed:
            self.stats.agreements += n_tokens
        else:
            self.stats.disagreements += n_tokens

        # Adaptive: if agreement rate drops, go full-only
        if (self.stats.agreements + self.stats.disagreements) >= 20:
            if self.stats.p_agree < self.agreement_threshold:
                if not self._force_full:
                    logger.warning(
                        "PLV: p_agree=%.3f < threshold=%.3f, switching to full-only",
                        self.stats.p_agree,
                        self.agreement_threshold,
                    )
                    self._force_full = True

    def record_partial_time(self, ns: int):
        self.stats.partial_calls += 1
        self.stats.partial_time_ns += ns

    def record_full_time(self, ns: int):
        self.stats.full_calls += 1
        self.stats.full_time_ns += ns


# ---------------------------------------------------------------------------
# Monkey-patch installer for a live vLLM engine
# ---------------------------------------------------------------------------

def install_plv(
    causal_lm_model: nn.Module,
    exit_layer: int | None = None,
    full_verify_interval: int | None = None,
) -> PLVController:
    """Monkey-patch a Qwen3_5ForCausalLMBase (or Qwen3NextForCausalLM) instance
    to support partial-layer verification.

    Args:
        causal_lm_model: The top-level CausalLM model (has .model and .lm_head).
        exit_layer: Number of layers for early exit. Default from PLV_EXIT_LAYER env.
        full_verify_interval: How often to force full verification.

    Returns:
        PLVController instance (also stored as causal_lm_model._plv_controller).
    """
    el = exit_layer if exit_layer is not None else PLV_EXIT_LAYER
    fvi = full_verify_interval if full_verify_interval is not None else PLV_FULL_VERIFY_INTERVAL

    controller = PLVController(exit_layer=el, full_verify_interval=fvi)

    backbone = causal_lm_model.model  # Qwen3_5Model / Qwen3NextModel

    # Store references
    causal_lm_model._plv_controller = controller
    causal_lm_model._plv_backbone = backbone

    # Save original compute_logits
    original_compute_logits = causal_lm_model.compute_logits

    def plv_compute_logits_partial(hidden_states_partial: torch.Tensor) -> torch.Tensor | None:
        """Compute logits from early-exit hidden states."""
        return original_compute_logits(hidden_states_partial)

    causal_lm_model.plv_compute_logits_partial = plv_compute_logits_partial

    def plv_early_exit_forward(
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run only first N layers + norm. Returns hidden states for lm_head."""
        return _early_exit_forward(
            backbone, input_ids, positions, el,
            intermediate_tensors, inputs_embeds,
        )

    causal_lm_model.plv_early_exit_forward = plv_early_exit_forward

    logger.info(
        "PLV installed: exit_layer=%d, full_verify_interval=%d, total_layers=%d",
        el, fvi, backbone.end_layer - backbone.start_layer,
    )
    return controller


# ---------------------------------------------------------------------------
# Standalone verification wrapper (for use outside vLLM scheduler)
# ---------------------------------------------------------------------------

def verify_draft_tokens(
    causal_lm_model: nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    draft_token_ids: torch.Tensor,
    controller: PLVController | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Verify draft tokens using PLV.

    Args:
        causal_lm_model: The CausalLM model with PLV installed.
        input_ids: Input token IDs (context + draft tokens).
        positions: Position IDs.
        draft_token_ids: The draft token IDs to verify (1-D).
        controller: PLVController. If None, uses causal_lm_model._plv_controller.

    Returns:
        (logits, accepted_mask, used_partial):
        - logits: Full logits tensor from the verification pass.
        - accepted_mask: Boolean mask over draft_token_ids (True=accepted).
        - used_partial: Whether partial verification was used.
    """
    if controller is None:
        controller = getattr(causal_lm_model, '_plv_controller', None)
        if controller is None:
            raise RuntimeError("PLV not installed. Call install_plv() first.")

    backbone = causal_lm_model.model
    n_draft = draft_token_ids.shape[0]
    use_partial = controller.should_use_partial

    if use_partial:
        # Partial path: early exit
        t0 = time.perf_counter_ns()
        hidden_partial = _early_exit_forward(
            backbone, input_ids, positions, controller.exit_layer,
        )
        logits_partial = causal_lm_model.compute_logits(hidden_partial)
        t1 = time.perf_counter_ns()
        controller.record_partial_time(t1 - t0)

        # Check agreement: do partial logits agree with draft tokens?
        # We look at the last n_draft positions (the draft token positions).
        # The logit at position i should have argmax == draft_token_ids[i].
        if logits_partial is not None and n_draft > 0:
            draft_logits = logits_partial[-n_draft:]
            partial_predictions = draft_logits.argmax(dim=-1)
            accepted_mask = (partial_predictions == draft_token_ids)
            agreed = accepted_mask.all().item()
        else:
            accepted_mask = torch.ones(n_draft, dtype=torch.bool, device=input_ids.device)
            agreed = True

        controller.record_agreement(agreed, n_draft)
        controller.step()

        if agreed:
            return logits_partial, accepted_mask, True

        # Disagreement: fall back to full
        logger.debug("PLV: partial disagreed, falling back to full")

    # Full path
    t0 = time.perf_counter_ns()
    hidden_full = _full_forward(backbone, input_ids, positions)
    logits_full = causal_lm_model.compute_logits(hidden_full)
    t1 = time.perf_counter_ns()
    controller.record_full_time(t1 - t0)

    if logits_full is not None and n_draft > 0:
        draft_logits = logits_full[-n_draft:]
        full_predictions = draft_logits.argmax(dim=-1)
        accepted_mask = (full_predictions == draft_token_ids)
    else:
        accepted_mask = torch.ones(n_draft, dtype=torch.bool, device=input_ids.device)

    if not use_partial:
        controller.step()

    return logits_full, accepted_mask, False


# ---------------------------------------------------------------------------
# Benchmark: measure p_agree by running both paths and comparing argmax
# ---------------------------------------------------------------------------

def benchmark_p_agree(
    causal_lm_model: nn.Module,
    tokenizer,
    prompts: list[str],
    exit_layer: int | None = None,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> dict:
    """Measure how often early-exit argmax matches full-model argmax.

    Runs both partial and full forward on each prefill, compares the argmax
    of the last token's logits. This measures the quality of the early-exit
    approximation without doing actual speculative decoding.

    Args:
        causal_lm_model: The CausalLM model (must have .model and .compute_logits).
        tokenizer: HF tokenizer.
        prompts: List of text prompts.
        exit_layer: Layer to exit at. Default from env PLV_EXIT_LAYER.
        max_new_tokens: How many autoregressive steps to measure per prompt.
        device: Device string.

    Returns:
        Dict with p_agree, per-prompt details, timing.
    """
    el = exit_layer if exit_layer is not None else PLV_EXIT_LAYER
    backbone = causal_lm_model.model
    total_layers = backbone.end_layer - backbone.start_layer

    results = {
        "exit_layer": el,
        "total_layers": total_layers,
        "prompts": [],
        "global_agree": 0,
        "global_total": 0,
        "partial_time_ms": 0.0,
        "full_time_ms": 0.0,
    }

    for prompt_text in prompts:
        tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_results = {"prompt": prompt_text[:80], "agreements": 0, "total": 0}

        # Autoregressive loop
        current_ids = tokens
        for step in range(max_new_tokens):
            seq_len = current_ids.shape[1]
            positions = torch.arange(seq_len, device=device)

            # Partial forward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                h_partial = _early_exit_forward(
                    backbone, current_ids, positions, el,
                )
                logits_partial = causal_lm_model.compute_logits(h_partial)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results["partial_time_ms"] += (t1 - t0) * 1000

            # Full forward
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                h_full = _full_forward(backbone, current_ids, positions)
                logits_full = causal_lm_model.compute_logits(h_full)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            results["full_time_ms"] += (t3 - t2) * 1000

            # Compare argmax of last token
            if logits_partial is not None and logits_full is not None:
                pred_partial = logits_partial[-1].argmax().item()
                pred_full = logits_full[-1].argmax().item()
                agreed = pred_partial == pred_full
                prompt_results["agreements"] += int(agreed)
                prompt_results["total"] += 1
                results["global_agree"] += int(agreed)
                results["global_total"] += 1

                # Use full prediction as the "true" next token
                next_token = pred_full
            else:
                break

            # Check for EOS
            if hasattr(tokenizer, "eos_token_id") and next_token == tokenizer.eos_token_id:
                break

            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token]], device=device),
            ], dim=1)

        results["prompts"].append(prompt_results)

    results["p_agree"] = (
        results["global_agree"] / results["global_total"]
        if results["global_total"] > 0
        else 0.0
    )
    results["speedup"] = (
        results["full_time_ms"] / results["partial_time_ms"]
        if results["partial_time_ms"] > 0
        else 0.0
    )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PLV benchmark for Qwen3.5-27B")
    parser.add_argument(
        "--model-path",
        default="/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-4bit",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--exit-layer",
        type=int,
        default=PLV_EXIT_LAYER,
        help="Layer to exit at for partial verification",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of autoregressive steps per prompt",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The capital of France is",
            "In quantum mechanics, the uncertainty principle states that",
            "def fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n",
            "The three laws of thermodynamics are:",
        ],
        help="Prompts to benchmark",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep exit layers: 8, 16, 24, 32, 40, 48, 56",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading model from %s ...", args.model_path)
    logger.info("This uses vLLM's model loader. Requires --enforce-eager equivalent.\n")

    # We load the model via vLLM's offline LLM interface for the benchmark.
    # This handles GPTQ quantization, weight loading, etc.
    from vllm import LLM

    llm = LLM(
        model=args.model_path,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        trust_remote_code=True,
    )

    # Extract the actual model from vLLM's internals
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    causal_lm = model_runner.model
    tokenizer = llm.get_tokenizer()

    exit_layers = [8, 16, 24, 32, 40, 48, 56] if args.sweep else [args.exit_layer]

    for el in exit_layers:
        logger.info("=" * 60)
        logger.info("Benchmarking exit_layer=%d / 64", el)
        logger.info("=" * 60)

        results = benchmark_p_agree(
            causal_lm,
            tokenizer,
            args.prompts,
            exit_layer=el,
            max_new_tokens=args.max_new_tokens,
        )

        logger.info("  p_agree:      %.4f", results["p_agree"])
        logger.info("  speedup:      %.2fx", results["speedup"])
        logger.info("  partial_ms:   %.1f", results["partial_time_ms"])
        logger.info("  full_ms:      %.1f", results["full_time_ms"])
        logger.info("  total_tokens: %d", results["global_total"])

        for pr in results["prompts"]:
            pa = pr["agreements"] / pr["total"] if pr["total"] > 0 else 0
            logger.info(
                "    [%.2f] %s... (%d/%d)",
                pa, pr["prompt"][:50], pr["agreements"], pr["total"],
            )
        logger.info("")


if __name__ == "__main__":
    main()
