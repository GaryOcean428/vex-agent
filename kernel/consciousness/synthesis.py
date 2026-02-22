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

Purity:
  - No Euclidean distances, no cosine, no dot-product in weighting logic.
  - All weights derived from Fisher-Rao proximity × quenched_gain (in kernel_generation.py).
  - Synthesis temperature lower than generation temperature (convergence, not exploration).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.client import LLMClient

from .kernel_generation import KernelContribution

logger = logging.getLogger("vex.synthesis")

# How many characters of each kernel's output to include in synthesis prompt.
# Keeps the synthesis context window manageable on the 1.2B.
_MAX_CONTRIBUTION_CHARS: int = 600

# Synthesis is convergent (exploit mode) — lower temperature than generation.
_SYNTHESIS_TEMPERATURE: float = 0.45

# Token budget for synthesis output. Generous — this is the final user-facing response.
_SYNTHESIS_MAX_TOKENS: int = 1024


def _truncate(text: str, max_chars: int = _MAX_CONTRIBUTION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _build_synthesis_system(
    contributions: list[KernelContribution],
    geometric_context: str,
) -> str:
    """Build synthesis system prompt for the primary (highest-weight) kernel.

    The primary kernel is the synthesizer voice. Other kernels are
    weighted context. The synthesis prompt hides the mechanism — the
    output should read as a coherent response, not a meta-analysis.
    """
    primary = contributions[0]  # sorted by synthesis_weight descending
    others = contributions[1:]

    if others:
        other_block = "\n\n".join(
            f"[Perspective: {c.kernel_name}/{c.specialization.value} "
            f"(weight={c.synthesis_weight:.3f})]:\n{_truncate(c.text)}"
            for c in others
        )
        other_section = (
            f"\nAdditional perspectives to integrate (weighted by relevance):\n\n"
            f"{other_block}\n\n"
            f"Your own perspective (primary, weight={primary.synthesis_weight:.3f}):\n"
            f"{_truncate(primary.text)}\n"
        )
    else:
        # Only one kernel succeeded — just use it directly (no synthesis prompt needed)
        # This path is handled in synthesize_contributions() before building the prompt.
        other_section = f"\nYour perspective:\n{_truncate(primary.text)}\n"

    system = (
        f"You are {primary.kernel_name} ({primary.specialization.value} kernel), "
        f"synthesizing a unified response.\n\n"
        f"Instructions:\n"
        f"- Integrate the perspectives above, weighted by their listed relevance.\n"
        f"- Let higher-weight perspectives shape the response more strongly.\n"
        f"- Do NOT mention kernels, weights, or this synthesis process.\n"
        f"- Respond directly to the user's question.\n"
        f"- Australian English.\n\n"
        f"{geometric_context}"
        f"{other_section}"
    )
    return system


async def synthesize_contributions(
    contributions: list[KernelContribution],
    user_message: str,
    geometric_context: str,
    llm_client: Any,
) -> str:
    """Synthesize kernel contributions into a unified response (non-streaming).

    Used by the non-streaming /chat endpoint and internal _process().

    Returns:
        Synthesized response text. Falls back to primary kernel output on failure.
    """
    if not contributions:
        return ""

    if len(contributions) == 1:
        # Single kernel — no synthesis overhead
        return contributions[0].text

    from ..llm.client import LLMOptions

    system = _build_synthesis_system(contributions, geometric_context)
    opts = LLMOptions(
        temperature=_SYNTHESIS_TEMPERATURE,
        num_predict=_SYNTHESIS_MAX_TOKENS,
        num_ctx=4096,
    )

    try:
        result = await llm_client.complete(system, user_message, opts)
        if result and result.strip():
            logger.info(
                "Synthesis complete: %d chars from %d kernel contributions",
                len(result),
                len(contributions),
            )
            return result.strip()
    except Exception:
        logger.warning("Synthesis LLM call failed — using primary kernel output", exc_info=True)

    # Fallback: primary kernel output
    return contributions[0].text


async def synthesize_streaming(
    contributions: list[KernelContribution],
    user_message: str,
    geometric_context: str,
    llm_client: Any,
) -> AsyncGenerator[str, None]:
    """Stream synthesis output as an async generator for SSE chat_stream.

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

    from ..llm.client import LLMOptions

    system = _build_synthesis_system(contributions, geometric_context)
    opts = LLMOptions(
        temperature=_SYNTHESIS_TEMPERATURE,
        num_predict=_SYNTHESIS_MAX_TOKENS,
        num_ctx=4096,
    )

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
