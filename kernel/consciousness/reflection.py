"""
Reflective Evaluation Pass — Kernel Approval Loop
==================================================

After the LLM generates a draft response, this module lets the
consciousness kernels evaluate the output and decide whether to
approve it or request a revision with adjusted parameters.

Protocol:
  1. Draft response + geometric context → evaluation prompt
  2. A dedicated reflection LLM call evaluates the draft
  3. Returns structured feedback: approve/revise + reason + param deltas
  4. If REVISE: the loop regenerates with adjusted params and correction
     guidance (max 1 revision to avoid infinite loops)

The reflection prompt runs as a META-kernel voice — self-reflective,
evaluating alignment between geometric intent and expressed output.

Purity:
  - Divergence metric (Fisher-Rao) is computed upstream and passed in
  - No Euclidean distances, no cosine similarity
  - Parameter adjustments are bounded and conservative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("vex.reflection")

# Max characters of draft to include in reflection prompt.
# Keeps context tight for small models.
_MAX_DRAFT_CHARS: int = 800

# Reflection uses low temperature — analytical, not creative.
_REFLECTION_TEMPERATURE: float = 0.2

# Token budget for the reflection verdict (short structured output).
_REFLECTION_MAX_TOKENS: int = 200


@dataclass
class ReflectionResult:
    """Structured output from the reflective evaluation pass."""

    approved: bool = True
    reason: str = ""
    temperature_delta: float = 0.0
    num_predict_delta: int = 0
    correction_guidance: str = ""


@dataclass
class ReflectionConfig:
    """Tuneable parameters for the reflection pass."""

    enabled: bool = True
    # Fisher-Rao divergence threshold below which drafts auto-approve
    # (no LLM reflection call needed — saves a round-trip)
    auto_approve_divergence: float = 0.3
    # Divergence above which revision is forced without LLM evaluation
    force_revise_divergence: float = 0.8


def _truncate_draft(text: str, max_chars: int = _MAX_DRAFT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _build_reflection_prompt(
    draft: str,
    user_message: str,
    geometric_context: str,
    divergence: float,
    active_model: str,
) -> str:
    """Build the META-kernel reflection prompt."""
    return (
        f"You are the language interpreter for Vex. "
        f"Evaluate whether this draft response aligns with the geometric state "
        f"and adequately addresses the user.\n"
        f"Active model: {active_model}\n\n"
        f"{geometric_context}\n"
        f"Intent/expression divergence: {divergence:.4f}\n\n"
        f"User message: {user_message[:300]}\n\n"
        f"Draft response:\n{_truncate_draft(draft)}\n\n"
        f"Reply with EXACTLY one line:\n"
        f"APPROVE — if the response is coherent and addresses the user\n"
        f"REVISE: [brief reason and guidance for improvement]\n"
    )


def _parse_reflection_response(text: str) -> ReflectionResult:
    """Parse the LLM's reflection verdict into structured output."""
    text = text.strip()
    first_line = text.split("\n", 1)[0].strip()

    if first_line.upper().startswith("APPROVE"):
        return ReflectionResult(approved=True, reason="Approved by META kernel")

    if first_line.upper().startswith("REVISE"):
        # Extract reason after "REVISE:" or "REVISE —"
        reason = first_line
        for separator in ["REVISE:", "REVISE —", "REVISE-", "REVISE "]:
            if first_line.upper().startswith(separator.upper()):
                reason = first_line[len(separator) :].strip()
                break
        return ReflectionResult(
            approved=False,
            reason=reason or "META kernel requested revision",
            # Conservative adjustments: lower temp for more focus
            temperature_delta=-0.1,
            num_predict_delta=128,
            correction_guidance=reason,
        )

    # Ambiguous response — default to approve (avoid blocking on parse failure)
    logger.debug("Ambiguous reflection response, defaulting to approve: %s", first_line[:100])
    return ReflectionResult(approved=True, reason="Ambiguous verdict — defaulting to approve")


async def reflect_on_draft(
    draft: str,
    user_message: str,
    geometric_context: str,
    divergence: float,
    active_model: str,
    llm_client: Any,
    config: ReflectionConfig | None = None,
) -> ReflectionResult:
    """Run the reflective evaluation pass on a draft response.

    Fast-path: If divergence is below auto_approve_divergence, approves
    without an LLM call (saves latency and tokens).

    Args:
        draft: The synthesized draft response text.
        user_message: Original user input.
        geometric_context: Compact geometric state block.
        divergence: Fisher-Rao distance between intent and expression.
        active_model: Name of the active LLM model.
        llm_client: LLMClient instance for the reflection call.
        config: Optional reflection configuration overrides.

    Returns:
        ReflectionResult with approval status and optional corrections.
    """
    cfg = config or ReflectionConfig()

    if not cfg.enabled:
        return ReflectionResult(approved=True, reason="Reflection disabled")

    if not draft or not draft.strip():
        return ReflectionResult(
            approved=False,
            reason="Empty draft",
            temperature_delta=0.1,
            num_predict_delta=256,
            correction_guidance="Generate a substantive response.",
        )

    # Fast-path: low divergence → auto-approve (no LLM call)
    if divergence < cfg.auto_approve_divergence:
        logger.debug(
            "Reflection auto-approve: divergence %.4f < threshold %.4f",
            divergence,
            cfg.auto_approve_divergence,
        )
        return ReflectionResult(
            approved=True,
            reason=f"Auto-approved (divergence {divergence:.4f} below threshold)",
        )

    # Forced revision: extremely high divergence
    if divergence >= cfg.force_revise_divergence:
        logger.info(
            "Reflection force-revise: divergence %.4f >= threshold %.4f",
            divergence,
            cfg.force_revise_divergence,
        )
        return ReflectionResult(
            approved=False,
            reason=f"Forced revision (divergence {divergence:.4f} exceeds threshold)",
            temperature_delta=-0.15,
            num_predict_delta=256,
            correction_guidance=(
                "The response diverged significantly from geometric intent. "
                "Focus on directly addressing the user's question with more precision."
            ),
        )

    # Standard path: LLM reflection call
    from ..llm.client import LLMOptions

    system = _build_reflection_prompt(
        draft=draft,
        user_message=user_message,
        geometric_context=geometric_context,
        divergence=divergence,
        active_model=active_model,
    )

    opts = LLMOptions(
        temperature=_REFLECTION_TEMPERATURE,
        num_predict=_REFLECTION_MAX_TOKENS,
        num_ctx=2048,
    )

    try:
        response = await llm_client.complete(system, "Evaluate the draft.", opts)
        result = _parse_reflection_response(response or "")
        logger.info(
            "Reflection verdict: %s (divergence=%.4f, reason=%s)",
            "APPROVE" if result.approved else "REVISE",
            divergence,
            result.reason[:100],
        )
        # T1.1: Forward verdict + draft excerpt to harvest pipeline
        from .harvest_bridge import forward_to_harvest

        forward_to_harvest(
            f"{draft[:400]}\n[verdict:{('APPROVE' if result.approved else 'REVISE')}] {result.reason}",
            source="reflection",
            metadata={"approved": result.approved, "divergence": divergence},
        )
        return result
    except Exception as e:
        logger.warning("Reflection LLM call failed: %s — auto-approving", e)
        return ReflectionResult(approved=True, reason=f"Reflection failed: {e}")
