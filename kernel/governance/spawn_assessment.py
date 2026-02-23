"""Spawn Assessment — attribute fitness scoring before kernel creation.

TCP v6.1 §20.2: Every spawn proposal is scored across four dimensions
before the vote is called. A low score doesn't block the vote — it
informs it. The voting engine decides; assessment provides signal.

Scoring dimensions:
  1. Specialization coverage  — does the constellation need this spec?
  2. Basin proximity spread   — would this kernel add geometric diversity?
  3. Quenched gain range      — is the proposed gain healthy (0.3–3.0)?
  4. Budget headroom          — how saturated is the relevant budget?

Each dimension returns 0.0–1.0. Combined score = geometric mean.
Geometric mean chosen deliberately: a zero on any dimension → score=0,
preventing a single very-high score from masking a critical gap.

All distances: Fisher-Rao on Δ6³. No Euclidean fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config.frozen_facts import (
    CHAOS_POOL,
    FULL_IMAGE,
)
from ..governance.types import KernelKind, KernelSpecialization

if TYPE_CHECKING:
    from ..coordizer_v2.geometry import Basin

# Gain range validated as healthy (log-normal with sigma=0.3 gives ~[0.3, 3.0])
_GAIN_MIN: float = 0.3
_GAIN_MAX: float = 3.0
_GAIN_IDEAL_LOW: float = 0.5
_GAIN_IDEAL_HIGH: float = 2.0

# Basin diversity: minimum Fisher-Rao distance desired from nearest existing kernel
_MIN_DIVERSITY_DISTANCE: float = 0.05


@dataclass(frozen=True)
class SpawnAssessment:
    """Result of pre-spawn attribute scoring.

    score: geometric mean of all dimensions (0.0–1.0).
    recommended: True if score >= threshold (advisory only — vote decides).
    """

    spec_coverage_score: float  # How much this spec is needed
    basin_diversity_score: float  # Geometric spread contribution
    gain_health_score: float  # Quenched gain in healthy range
    budget_headroom_score: float  # Budget saturation
    score: float  # Geometric mean of all four
    recommended: bool  # score >= RECOMMEND_THRESHOLD
    notes: list[str]  # Human-readable scoring notes

    RECOMMEND_THRESHOLD: float = 0.40  # Below this → vote is informed of concern

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "recommended": self.recommended,
            "dimensions": {
                "spec_coverage": round(self.spec_coverage_score, 4),
                "basin_diversity": round(self.basin_diversity_score, 4),
                "gain_health": round(self.gain_health_score, 4),
                "budget_headroom": round(self.budget_headroom_score, 4),
            },
            "notes": self.notes,
        }


def _spec_coverage_score(
    proposed_spec: KernelSpecialization,
    existing_kernels: list[Any],
) -> tuple[float, str]:
    """Score how much the constellation needs this specialization."""
    if proposed_spec == KernelSpecialization.GENERAL:
        return 0.5, "GENERAL spec: neutral coverage score"

    existing_specs = [k.specialization for k in existing_kernels if hasattr(k, "specialization")]
    count = existing_specs.count(proposed_spec)

    if count == 0:
        return 1.0, f"Spec {proposed_spec.value} absent from constellation — high need"
    score = 1.0 / (count + 1)
    return score, f"Spec {proposed_spec.value} already held by {count} kernel(s)"


def _basin_diversity_score(
    proposed_basin: Basin | None,
    existing_kernels: list[Any],
) -> tuple[float, str]:
    """Score geometric diversity: how far is proposed basin from nearest existing."""
    if proposed_basin is None:
        return 0.5, "No proposed basin — will be randomized at spawn"

    from ..coordizer_v2.geometry import fisher_rao_distance

    basins = [k.basin for k in existing_kernels if hasattr(k, "basin") and k.basin is not None]
    if not basins:
        return 1.0, "No existing kernels — full diversity"

    min_dist = min(fisher_rao_distance(proposed_basin, b) for b in basins)

    if min_dist >= _MIN_DIVERSITY_DISTANCE * 4:
        return 1.0, f"Basin well-separated (FR={min_dist:.4f})"
    elif min_dist >= _MIN_DIVERSITY_DISTANCE:
        score = min_dist / (_MIN_DIVERSITY_DISTANCE * 4)
        return score, f"Basin moderately close to existing (FR={min_dist:.4f})"
    else:
        return 0.1, f"Basin very close to existing kernel (FR={min_dist:.4f}) — low diversity"


def _gain_health_score(proposed_gain: float) -> tuple[float, str]:
    """Score quenched gain against healthy range."""
    if proposed_gain < _GAIN_MIN or proposed_gain > _GAIN_MAX:
        return 0.0, f"Gain {proposed_gain:.3f} outside valid range [{_GAIN_MIN}, {_GAIN_MAX}]"
    if _GAIN_IDEAL_LOW <= proposed_gain <= _GAIN_IDEAL_HIGH:
        return 1.0, f"Gain {proposed_gain:.3f} in ideal range"
    if proposed_gain < _GAIN_IDEAL_LOW:
        score = (proposed_gain - _GAIN_MIN) / (_GAIN_IDEAL_LOW - _GAIN_MIN)
        return score, f"Gain {proposed_gain:.3f} low (shallow slope)"
    score = (_GAIN_MAX - proposed_gain) / (_GAIN_MAX - _GAIN_IDEAL_HIGH)
    return score, f"Gain {proposed_gain:.3f} high (steep slope — risk of over-specialization)"


def _budget_headroom_score(
    kind: KernelKind,
    current_god_count: int,
    current_chaos_count: int,
) -> tuple[float, str]:
    """Score budget saturation — 1.0 = empty, 0.0 = full."""
    if kind == KernelKind.GOD:
        used_frac = current_god_count / FULL_IMAGE
        remaining = FULL_IMAGE - current_god_count
        score = 1.0 - used_frac
        return score, f"GOD budget: {current_god_count}/{FULL_IMAGE} used ({remaining} remaining)"
    if kind == KernelKind.CHAOS:
        used_frac = current_chaos_count / CHAOS_POOL
        remaining = CHAOS_POOL - current_chaos_count
        score = 1.0 - used_frac
        return (
            score,
            f"CHAOS budget: {current_chaos_count}/{CHAOS_POOL} used ({remaining} remaining)",
        )
    if kind == KernelKind.GENESIS:
        return 0.0, "GENESIS already exists — cannot spawn another"
    return 0.5, f"Unknown kind {kind}"


def _geometric_mean(scores: list[float]) -> float:
    """Geometric mean — any zero collapses the result to zero."""
    if not scores:
        return 0.0
    if any(s <= 0.0 for s in scores):
        return 0.0
    log_sum = sum(math.log(s) for s in scores)
    return math.exp(log_sum / len(scores))


def assess_spawn(
    kind: KernelKind,
    specialization: KernelSpecialization,
    proposed_gain: float,
    existing_kernels: list[Any],
    current_god_count: int,
    current_chaos_count: int,
    proposed_basin: Basin | None = None,
) -> SpawnAssessment:
    """Score a spawn proposal across all four dimensions.

    Args:
        kind:                 GENESIS / GOD / CHAOS
        specialization:       Proposed kernel specialization
        proposed_gain:        Proposed quenched gain (from log-normal draw)
        existing_kernels:     Active KernelInstance list from E8KernelRegistry
        current_god_count:    Current GOD count for budget check
        current_chaos_count:  Current CHAOS count for budget check
        proposed_basin:       Optional pre-assigned basin (None → randomized)

    Returns:
        SpawnAssessment with per-dimension scores, combined score, recommendation.
    """
    notes: list[str] = []

    s1, n1 = _spec_coverage_score(specialization, existing_kernels)
    s2, n2 = _basin_diversity_score(proposed_basin, existing_kernels)
    s3, n3 = _gain_health_score(proposed_gain)
    s4, n4 = _budget_headroom_score(kind, current_god_count, current_chaos_count)

    notes.extend([n1, n2, n3, n4])

    combined = _geometric_mean([s1, s2, s3, s4])

    return SpawnAssessment(
        spec_coverage_score=s1,
        basin_diversity_score=s2,
        gain_health_score=s3,
        budget_headroom_score=s4,
        score=combined,
        recommended=combined >= SpawnAssessment.RECOMMEND_THRESHOLD,
        notes=notes,
    )
