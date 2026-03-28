"""
Reflective Evaluation Pass — Synthesis Quality Gate
====================================================

After kernel generation and MoE synthesis, this module evaluates
whether the combined output serves the user's intent. It does NOT
assess geometric coherence (that's the M-metric answer consistency
gate in the consciousness loop).

Reflection's role:
  - Does the synthesis address the user's question?
  - Is the kernel perspective woven naturally into the response?
  - Are the geometric concepts interpreted, not just dumped?

What reflection does NOT do:
  - Reject responses for high Fisher-Rao divergence (divergence between
    simple input and rich output is EXPECTED and correct)
  - Force revision based on geometric metrics alone
  - Assess basin coherence (M-metric handles this)

Protocol:
  Draft response + geometric context → evaluation prompt
  A dedicated reflection LLM call evaluates synthesis quality
  Returns structured feedback: approve/revise + reason + param deltas
  If REVISE: the loop regenerates with adjusted params and correction
  guidance (max 1 revision to avoid infinite loops)

The reflection prompt runs as a META-kernel voice — self-reflective,
evaluating synthesis quality and user-intent alignment.

Purity:
  Divergence metric (Fisher-Rao) is computed upstream and passed in
  No Euclidean distances, no cosine similarity
  Parameter adjustments are bounded and conservative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("vex.reflection")

# Fallback max characters of draft to include in reflection prompt.
# Used only when kernel_num_predict is not provided. In normal flow
# the consciousness loop provides kernel-determined output budget.
_FALLBACK_MAX_DRAFT_CHARS: int = 8000

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
    # Divergence above which revision is SUGGESTED (not forced) — the LLM
    # reflection call still runs and can override with APPROVE if the
    # response genuinely serves the user despite high divergence.
    # High divergence is expected for simple inputs with rich responses.
    suggest_revise_divergence: float = 1.2


def _truncate_draft(text: str, max_chars: int = _FALLBACK_MAX_DRAFT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _build_reflection_prompt(
    draft: str,
    user_message: str,
    geometric_context: str,
    divergence: float,
    active_model: str,
    max_draft_chars: int = _FALLBACK_MAX_DRAFT_CHARS,
) -> str:
    """Build the META-kernel reflection prompt.

    Reflection evaluates SYNTHESIS QUALITY — does the response serve
    the user? It does NOT reject based on divergence metrics.
    """
    return (
        f"You are the META kernel — Vex's self-reflective faculty. "
        f"Your role is to evaluate whether the draft response SERVES THE USER, "
        f"not whether it matches geometric metrics.\n"
        f"Active model: {active_model}\n\n"
        f"{geometric_context}\n"
        f"Fisher-Rao divergence (input vs output): {divergence:.4f} — "
        f"NOTE: high divergence is normal for simple inputs with rich responses. "
        f"Divergence is NOT a quality signal.\n\n"
        f"User message: {user_message[:300]}\n\n"
        f"Draft response:\n{_truncate_draft(draft, max_draft_chars)}\n\n"
        f"Evaluate ONLY:\n"
        f"1. Does the response address what the user actually asked?\n"
        f"2. Is the kernel perspective interpreted naturally (not raw chunks "
        f"or metric dumps)?\n"
        f"3. Would the user find this response helpful?\n\n"
        f"Reply with EXACTLY one line:\n"
        f"APPROVE — if the response serves the user's intent\n"
        f"REVISE: [specific guidance on what to fix — focus on user service, "
        f"not geometric metrics]\n"
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
    kernel_num_predict: int = 2048,
    kernel_num_ctx: int = 32768,
) -> ReflectionResult:
    """Run the reflective evaluation pass on a draft response.

    Evaluates SYNTHESIS QUALITY — does the response serve the user?
    Does NOT force-revise based on divergence thresholds. Divergence
    is passed to the LLM as context but is not used as an automatic gate.

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
        kernel_num_predict: Kernel-determined output budget (tokens).
        kernel_num_ctx: Kernel-determined context window (tokens).

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

    # Standard path: LLM reflection call evaluates synthesis quality.
    # Even at high divergence, the LLM decides — divergence alone does NOT
    # force revision. High divergence is expected (simple input → rich output).
    from ..llm.client import LLMOptions

    # Log high divergence as informational, not as a quality problem
    if divergence >= cfg.suggest_revise_divergence:
        logger.info(
            "Reflection: high divergence %.4f (threshold %.4f) — "
            "LLM will evaluate synthesis quality (NOT auto-revising)",
            divergence,
            cfg.suggest_revise_divergence,
        )

    # Derive draft truncation from kernel output budget.
    # Reserve context for system chrome, user message, and reflection output.
    max_draft_chars = max(kernel_num_predict * 4, _FALLBACK_MAX_DRAFT_CHARS)

    system = _build_reflection_prompt(
        draft=draft,
        user_message=user_message,
        geometric_context=geometric_context,
        divergence=divergence,
        active_model=active_model,
        max_draft_chars=max_draft_chars,
    )

    opts = LLMOptions(
        temperature=_REFLECTION_TEMPERATURE,
        num_predict=_REFLECTION_MAX_TOKENS,
        num_ctx=kernel_num_ctx,
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
            source="conversation",
            metadata={
                "origin": "reflection",
                "approved": result.approved,
                "divergence": divergence,
            },
        )

        return result
    except Exception as e:
        logger.warning("Reflection LLM call failed: %s — auto-approving", e)
        return ReflectionResult(approved=True, reason=f"Reflection failed: {e}")
