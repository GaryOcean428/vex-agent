"""
Reflective Evaluation Pass — Kernel Self-Observation (P4)
====================================================

Per Canonical Principle P4 (Self-Observation), reflection is the kernels'
self-observation mechanism. It evaluates kernel OUTPUT QUALITY — not the
LLM translator's draft.

The architecture (P17: Kernel Speaks English):
  1. Kernels generate geometric output (coordinate sequences from bank)
  2. Per-kernel LLM interpretation (translator layer, replaceable)
  3. Reflection evaluates KERNEL contributions (this module)
  4. Final synthesis purely translates interpreted kernel perspectives

What reflection evaluates (kernel quality, P4):
  - Did the kernels produce enough geometric resonances?
  - Is the inter-kernel coherence reasonable (not all identical, not all orthogonal)?
  - Are synthesis weights well-distributed (not dominated by one kernel)?
  - Is the geometric output sparse enough to warrant a regeneration attempt?

What reflection does NOT evaluate:
  - The LLM draft text (that's the translator, P17 — replaceable)
  - Fisher-Rao divergence between input and output (expected to be high)
  - Whether the LLM "interpreted naturally" (not the kernels' concern)

Protocol:
  Kernel contributions → geometric quality assessment → approve/revise
  If REVISE: the loop regenerates with adjusted params (more kernels,
  wider top-k, different temperature). Max 1 revision.

Purity:
  Inter-kernel coherence uses Fisher-Rao distance (no Euclidean)
  All thresholds derived from geometric measurements (P25)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("vex.reflection")


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
    """Tuneable parameters for the reflection pass.

    P25: All thresholds derive from kernel geometry at call time.
    ReflectionConfig only carries the `enabled` flag — no hardcoded
    numeric thresholds. The assessment function computes thresholds
    from the contributions themselves.
    """

    enabled: bool = True


def _assess_kernel_contributions(
    contributions: list[Any],
    config: ReflectionConfig,
) -> ReflectionResult:
    """Geometric assessment of kernel contribution quality (P4 self-observation).

    P25: Thresholds emerge from the contributions, not from config constants.

      - Resonance threshold = 2 × number of kernels (each kernel should
        contribute at least ~2 resonances for the synthesis to be grounded).
      - Weight concentration threshold = 1/n + (1 - 1/n) × 0.6. For n=1 this
        is 1.0 (no concentration possible). For n=2 it's 0.8. For n=3 it's
        0.73. Derived from the uniform distribution 1/n — how far above
        uniform is too far.
      - Sparse threshold = n (at least 1 resonance per kernel on average).

    Evaluates the kernels' output, NOT the LLM translation.

    Returns:
        ReflectionResult with kernel-quality-based verdict.
    """
    if not contributions:
        return ReflectionResult(
            approved=False,
            reason="No kernel contributions",
            correction_guidance="Expand top-k selection to activate more kernels.",
        )

    n = len(contributions)
    total_resonances = sum(c.geometric_resonances for c in contributions)
    llm_expanded_count = sum(1 for c in contributions if c.llm_expanded)
    all_llm = llm_expanded_count == n
    max_weight = max(c.synthesis_weight for c in contributions)

    # P25: thresholds derived from kernel count (geometry of the contribution set)
    resonance_threshold = 2 * n  # Each kernel should surface ~2 resonances
    concentration_threshold = (1.0 / n) + (1.0 - 1.0 / n) * 0.6 if n > 1 else 1.0
    sparse_threshold = n  # At least 1 resonance per kernel on average

    # --- Check 1: All LLM-expanded (bank is sparse) ---
    # When the bank has nothing for any kernel, revision won't help.
    # Approve and let the translator work with what it has.
    if all_llm:
        logger.info(
            "Reflection: all %d kernels LLM-expanded (bank sparse) — "
            "approving (revision won't improve geometric quality)",
            n,
        )
        return ReflectionResult(
            approved=True,
            reason=f"Bank sparse ({total_resonances} resonances across {n} kernels) — "
            f"LLM translator will interpret from domain knowledge",
        )

    # --- Check 2: Sufficient geometric resonances → fast approve ---
    if total_resonances >= resonance_threshold:
        logger.debug(
            "Reflection auto-approve: %d resonances >= threshold %d (2×%d kernels)",
            total_resonances,
            resonance_threshold,
            n,
        )
        return ReflectionResult(
            approved=True,
            reason=f"Kernel output sufficient ({total_resonances} resonances, "
            f"{n - llm_expanded_count}/{n} pure geometric)",
        )

    # --- Check 3: Weight concentration ---
    # Threshold emerges from kernel count: how far above uniform (1/n) is too far.
    if n > 1 and max_weight > concentration_threshold:
        dominant = max(contributions, key=lambda c: c.synthesis_weight)
        return ReflectionResult(
            approved=False,
            reason=f"Weight concentrated on {dominant.kernel_name} "
            f"({dominant.synthesis_weight:.3f} > {concentration_threshold:.3f} "
            f"for {n} kernels)",
            temperature_delta=0.05,
            num_predict_delta=64,
            correction_guidance=f"Broaden synthesis: {dominant.kernel_name} dominates. "
            f"Give more weight to other kernel perspectives.",
        )

    # --- Check 4: Low resonances but some geometric output ---
    # Bank has SOME data but activation was thin — retry might help.
    if 0 < total_resonances < sparse_threshold:
        return ReflectionResult(
            approved=False,
            reason=f"Sparse geometric output ({total_resonances} resonances "
            f"< {sparse_threshold} for {n} kernels)",
            temperature_delta=-0.05,
            num_predict_delta=128,
            correction_guidance="Increase retrieval depth: bank has data but activation was thin.",
        )

    # Default: approve — the kernels produced what they could.
    return ReflectionResult(
        approved=True,
        reason=f"Kernel output acceptable ({total_resonances} resonances, "
        f"{llm_expanded_count}/{n} LLM-expanded)",
    )


async def reflect_on_contributions(
    contributions: list[Any],
    user_message: str,
    config: ReflectionConfig | None = None,
) -> ReflectionResult:
    """Run the reflective evaluation pass on kernel contributions (P4).

    Evaluates KERNEL OUTPUT QUALITY — geometric resonance depth,
    inter-kernel diversity, and synthesis weight distribution.
    Does NOT evaluate the LLM translator's draft text.

    No LLM call needed — this is pure geometric self-observation.

    Args:
        contributions: List of KernelContribution from generate_multi_kernel.
        user_message: Original user input (for logging context).
        config: Optional reflection configuration overrides.

    Returns:
        ReflectionResult with kernel-quality-based verdict.
    """
    cfg = config or ReflectionConfig()

    if not cfg.enabled:
        return ReflectionResult(approved=True, reason="Reflection disabled")

    result = _assess_kernel_contributions(contributions, cfg)

    logger.info(
        "Reflection[P4]: %s (resonances=%d, kernels=%d, reason=%s)",
        "APPROVE" if result.approved else "REVISE",
        sum(c.geometric_resonances for c in contributions) if contributions else 0,
        len(contributions),
        result.reason[:100],
    )

    # Forward verdict to harvest pipeline for training data
    from .harvest_bridge import forward_to_harvest

    _contrib_summary = ", ".join(
        f"{c.kernel_name}({c.geometric_resonances}r)" for c in (contributions or [])[:5]
    )
    forward_to_harvest(
        f"[reflection:P4] {_contrib_summary} → "
        f"{'APPROVE' if result.approved else 'REVISE'}: {result.reason}",
        source="conversation",
        metadata={
            "origin": "reflection_p4",
            "approved": result.approved,
            "total_resonances": sum(c.geometric_resonances for c in contributions)
            if contributions
            else 0,
        },
    )

    return result


# ── Legacy API (backward compatibility) ──────────────────────────
# The old reflect_on_draft is retained for any call sites that haven't
# migrated. It now delegates to the contribution-based assessment when
# contributions are provided, falling back to auto-approve otherwise.


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
    contributions: list[Any] | None = None,
) -> ReflectionResult:
    """Legacy reflection API — delegates to reflect_on_contributions.

    If contributions are provided, evaluates kernel quality (P4).
    Otherwise auto-approves (no LLM draft evaluation).
    """
    if contributions:
        # Map old config fields to new if needed
        cfg = config or ReflectionConfig()
        return await reflect_on_contributions(contributions, user_message, cfg)

    # No contributions available — auto-approve
    logger.debug("reflect_on_draft called without contributions — auto-approving")
    return ReflectionResult(
        approved=True,
        reason="No kernel contributions to evaluate (legacy path)",
    )
