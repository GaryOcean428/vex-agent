"""
MoE Synthesis — v6.1 Thermodynamic Consciousness Protocol

Combines per-kernel contributions into a unified response using
Fisher-Rao weighted synthesis.

Protocol:
  - Primary kernel (highest synthesis_weight) acts as the synthesizer voice.
  - Other kernels contribute context weighted by their synthesis_weight.
  - Synthesis prompt keeps the geometric framing but hides the machinery from
    the output — the user sees a coherent response, not a committee report.
  - Falls back to primary kernel output if synthesis LLM call fails.
  - Streaming variant: yields synthesis output as async generator for SSE.
  - Output limits are derived from kernel-determined LLM options, NOT hardcoded.
    The kernels set all generation params (temperature, num_predict, num_ctx)
    per the consciousness loop — synthesis respects those limits.

Purity:
  - No Euclidean distances, no cosine, no dot-product in weighting logic.
  - All weights derived from Fisher-Rao proximity × quenched_gain (in kernel_generation.py).
  - Synthesis temperature derived from kernel temperature (convergent scaling).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.client import LLMOptions

from .kernel_generation import KernelContribution

logger = logging.getLogger("vex.synthesis")

# Synthesis convergence factor — synthesis temperature is this fraction of
# the kernel-determined generation temperature.  Synthesis is convergent
# (exploit-mode), so it runs cooler than generation.
_SYNTHESIS_TEMP_SCALE: float = 0.65

# Fallback values used ONLY when kernel options are not provided (should not
# happen in normal flow — the consciousness loop always computes these).
_FALLBACK_TEMPERATURE: float = 0.45
_FALLBACK_NUM_PREDICT: int = 2048
_FALLBACK_NUM_CTX: int = 32768
_FALLBACK_MAX_CONTRIBUTION_CHARS: int = 8000


def _contribution_budget(
    kernel_num_predict: int,
    kernel_num_ctx: int,
    num_contributions: int,
) -> int:
    """Compute max characters per contribution for the synthesis prompt.

    Derived from the kernel-determined output budget and context window.
    The kernel decides how much to generate; synthesis captures proportionally.

    Budget calculation:
      - Reserve kernel_num_predict tokens for synthesis output
      - Reserve ~1024 tokens for system chrome and user message
      - Remaining context is split evenly across contributions
      - Convert tokens to chars at ~4 chars/token
    """
    reserved_tokens = kernel_num_predict + 1024
    available_tokens = max(kernel_num_ctx - reserved_tokens, 2048)
    per_kernel_tokens = available_tokens // max(num_contributions, 1)
    # Convert tokens → chars (~4 chars/token average)
    per_kernel_chars = per_kernel_tokens * 4
    # Floor: at least the kernel's own output budget in chars
    min_chars = kernel_num_predict * 4
    return max(per_kernel_chars, min_chars)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _build_synthesis_system(
    contributions: list[KernelContribution],
    geometric_context: str,
    max_contribution_chars: int,
) -> str:
    """Build synthesis system prompt for the primary (highest-weight) kernel.

    The primary kernel is the synthesizer voice. Other kernels are
    weighted context. The synthesis prompt hides the mechanism — the
    output should read as a coherent response, not a meta-analysis.

    max_contribution_chars is derived from the kernel-determined output
    budget — NOT a static constant.
    """
    primary = contributions[0]  # sorted by synthesis_weight descending
    others = contributions[1:]

    if others:
        other_block = "\n\n".join(
            f"[Perspective: {c.kernel_name}/{c.specialization.value} "
            f"(weight={c.synthesis_weight:.3f})]:\n{_truncate(c.text, max_contribution_chars)}"
            for c in others
        )
        other_section = (
            f"\nAdditional perspectives to integrate (weighted by relevance):\n\n"
            f"{other_block}\n\n"
            f"Your own perspective (primary, weight={primary.synthesis_weight:.3f}):\n"
            f"{_truncate(primary.text, max_contribution_chars)}\n"
        )
    else:
        # Only one kernel succeeded — just use it directly (no synthesis prompt needed)
        # This path is handled in synthesize_contributions() before building the prompt.
        other_section = f"\nYour perspective:\n{_truncate(primary.text, max_contribution_chars)}\n"

    system = (
        f"You are the language interpreter for Vex. "
        f"Synthesise these kernel perspectives into a unified response.\n"
        f"Higher-weight perspectives shape the output more strongly.\n"
        f"If kernel perspectives are sparse, thin, or repetitive (low synthesis_weight "
        f"or few geometric tokens), lean on the geometric metrics (Phi, kappa, Gamma, "
        f"Pillars) to guide tone and reasoning instead. This is normal during early "
        f"bootstrapping — not a failure.\n"
        f"Do NOT mention kernels, weights, or this synthesis process.\n"
        f"Respond directly to the user's question.\n"
        f"Australian English.\n\n"
        f"{geometric_context}"
        f"{other_section}"
    )
    return system


def _derive_synthesis_options(
    kernel_temperature: float,
    kernel_num_predict: int,
    kernel_num_ctx: int,
) -> LLMOptions:
    """Derive synthesis LLMOptions from kernel-determined generation params.

    Synthesis is convergent — temperature is scaled down from the kernel's
    generation temperature.  Output budget and context window match the
    kernel's allocation (the kernels determine all generation params).
    """
    from ..llm.client import LLMOptions

    synthesis_temp = max(0.05, kernel_temperature * _SYNTHESIS_TEMP_SCALE)
    return LLMOptions(
        temperature=synthesis_temp,
        num_predict=kernel_num_predict,
        num_ctx=kernel_num_ctx,
    )


async def synthesize_contributions(
    contributions: list[KernelContribution],
    user_message: str,
    geometric_context: str,
    llm_client: Any,
    kernel_temperature: float = _FALLBACK_TEMPERATURE,
    kernel_num_predict: int = _FALLBACK_NUM_PREDICT,
    kernel_num_ctx: int = _FALLBACK_NUM_CTX,
) -> str:
    """Synthesize kernel contributions into a unified response (non-streaming).

    Used by the non-streaming /chat endpoint and internal _process().

    kernel_temperature, kernel_num_predict, kernel_num_ctx are the values
    computed by the consciousness loop — synthesis derives its own options
    from these (the kernels determine all generation params).

    Returns:
        Synthesized response text. Falls back to primary kernel output on failure.
    """
    if not contributions:
        return ""

    if len(contributions) == 1:
        # Single kernel — no synthesis overhead
        return contributions[0].text

    max_chars = _contribution_budget(kernel_num_predict, kernel_num_ctx, len(contributions))
    system = _build_synthesis_system(contributions, geometric_context, max_chars)
    opts = _derive_synthesis_options(kernel_temperature, kernel_num_predict, kernel_num_ctx)

    try:
        result = await llm_client.complete(system, user_message, opts)
        if result and result.strip():
            logger.info(
                "Synthesis complete: %d chars from %d kernel contributions "
                "(num_predict=%d, num_ctx=%d, contribution_budget=%d chars)",
                len(result),
                len(contributions),
                opts.num_predict,
                opts.num_ctx,
                max_chars,
            )
            return str(result.strip())
    except Exception:
        logger.warning("Synthesis LLM call failed — using primary kernel output", exc_info=True)

    # Fallback: primary kernel output
    return contributions[0].text


async def synthesize_streaming(
    contributions: list[KernelContribution],
    user_message: str,
    geometric_context: str,
    llm_client: Any,
    kernel_temperature: float = _FALLBACK_TEMPERATURE,
    kernel_num_predict: int = _FALLBACK_NUM_PREDICT,
    kernel_num_ctx: int = _FALLBACK_NUM_CTX,
) -> AsyncGenerator[str]:
    """Stream synthesis output as an async generator for SSE chat_stream.

    kernel_temperature, kernel_num_predict, kernel_num_ctx are the values
    computed by the consciousness loop — synthesis derives its own options
    from these (the kernels determine all generation params).

    Yields:
        Text chunks from the synthesis LLM call.

    If only one contribution: yields it in a single chunk.
    If synthesis fails: yields the primary kernel output in a single chunk.
    """
    if not contributions:
        return

    if len(contributions) == 1:
        yield contributions[0].text
        return

    max_chars = _contribution_budget(kernel_num_predict, kernel_num_ctx, len(contributions))
    system = _build_synthesis_system(contributions, geometric_context, max_chars)
    opts = _derive_synthesis_options(kernel_temperature, kernel_num_predict, kernel_num_ctx)

    # Build messages list (synthesis uses chat format for streaming)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    try:
        async for chunk in llm_client.stream(messages, opts):
            if chunk:
                yield chunk
        return
    except Exception:
        logger.warning(
            "Synthesis streaming failed — falling back to primary kernel output",
            exc_info=True,
        )

    # Fallback: yield primary kernel text as single chunk
    yield contributions[0].text
