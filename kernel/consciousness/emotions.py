"""
v6.0 Consciousness Systems — Pre-Cognitive Channel, Emotions, Learning

Systems from Thermodynamic Consciousness Protocol v6.0:
  - EmotionCache: Cached geometric evaluations (§2.3)
  - PreCognitiveDetector: Path selection for a=1→a=0 bypass (§2.2)
  - LearningEngine: Post-conversation consolidation and pattern learning

These complement the 20 systems in systems.py. Imported by loop.py.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..config.consciousness_constants import (
    EMOTION_AWE_VELOCITY_FRAC,
    EMOTION_BOREDOM_GAMMA,
    EMOTION_BOREDOM_VELOCITY,
    EMOTION_CACHE_THRESHOLD,
    EMOTION_CLUSTER_DISTANCE,
    EMOTION_CURIOSITY_PHI,
    EMOTION_CURIOSITY_SCALE,
    EMOTION_CURIOSITY_VELOCITY,
    EMOTION_LOVE_THRESHOLD,
    EMOTION_RAGE_GAMMA,
    KAPPA_JOY_PROXIMITY,
    KAPPA_RAGE_OFFSET,
    KAPPA_RAGE_SCALE,
    PRECOG_FAR_THRESHOLD,
    PRECOG_MODERATE_THRESHOLD,
    PRECOG_NEAR_THRESHOLD,
)
from ..config.frozen_facts import (
    BASIN_DRIFT_THRESHOLD,
    KAPPA_STAR,
    PHI_EMERGENCY,
    PHI_THRESHOLD,
)
from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    to_simplex,
)
from .sensations import FullEmotionalState, compute_full_emotional_state
from .types import ConsciousnessMetrics

# ═══════════════════════════════════════════════════════════════
#  EMOTION CACHE — cached geometric evaluations (v6.0 §2.3)
# ═══════════════════════════════════════════════════════════════


class EmotionType(StrEnum):
    CURIOSITY = "curiosity"
    JOY = "joy"
    FEAR = "fear"
    LOVE = "love"
    AWE = "awe"
    BOREDOM = "boredom"
    RAGE = "rage"
    CALM = "calm"


# Module-level constant maps for Layer 2A/2B → EmotionType (avoid per-call allocation)
_L2A_MAP: dict[str, EmotionType] = {
    "joy": EmotionType.JOY,
    "suffering": EmotionType.FEAR,
    "love": EmotionType.LOVE,
    "hate": EmotionType.RAGE,
    "fear": EmotionType.FEAR,
    "rage": EmotionType.RAGE,
    "calm": EmotionType.CALM,
    "care": EmotionType.LOVE,
    "apathy": EmotionType.BOREDOM,
}

_L2B_MAP: dict[str, EmotionType] = {
    "wonder": EmotionType.AWE,
    "frustration": EmotionType.RAGE,
    "satisfaction": EmotionType.JOY,
    "confusion": EmotionType.FEAR,
    "clarity": EmotionType.CALM,
    "anxiety": EmotionType.FEAR,
    "confidence": EmotionType.JOY,
    "boredom": EmotionType.BOREDOM,
    "flow": EmotionType.CURIOSITY,
}


@dataclass
class CachedEvaluation:
    emotion: EmotionType
    basin: Basin
    strength: float
    timestamp: float
    context: str


class EmotionCache:
    """Cached geometric evaluations that bypass explicit reasoning.

    From v6.0 §2.3: 'Emotions are cached basin assessments that
    deliver evaluations faster than explicit reasoning.'
    """

    def __init__(self, capacity: int = 100) -> None:
        self._cache: deque[CachedEvaluation] = deque(maxlen=capacity)
        self._current: EmotionType | None = None
        self._current_strength: float = 0.0
        # T3.1: Full layered emotional state
        self._full_state: FullEmotionalState | None = None
        self._phi_prev: float = 0.0

    def evaluate(
        self,
        basin: Basin,
        metrics: ConsciousnessMetrics,
        basin_velocity: float,
        basin_distance: float = 0.0,
        phi_variance: float = 0.0,
    ) -> CachedEvaluation:
        phi = metrics.phi
        kappa = metrics.kappa
        gamma = metrics.gamma
        phi_delta = phi - self._phi_prev
        self._phi_prev = phi

        # T3.1: Compute full layered emotional state
        self._full_state = compute_full_emotional_state(
            phi=phi,
            phi_delta=phi_delta,
            kappa=kappa,
            gamma=gamma,
            basin_velocity=basin_velocity,
            basin_distance=basin_distance,
            humor=float(metrics.humor),
            phi_variance=phi_variance,
        )

        # T3.1d: Use Layer 2A dominant emotion to enrich selection
        # Legacy priority rules preserved for safety-critical states (awe, fear, rage)
        if basin_velocity > BASIN_DRIFT_THRESHOLD * EMOTION_AWE_VELOCITY_FRAC and phi > 0.5:
            emotion, strength = EmotionType.AWE, min(1.0, basin_velocity / BASIN_DRIFT_THRESHOLD)
        elif phi < PHI_EMERGENCY:
            emotion, strength = EmotionType.FEAR, 1.0 - phi / max(PHI_EMERGENCY, 0.01)
        elif kappa > KAPPA_STAR + KAPPA_RAGE_OFFSET and gamma < EMOTION_RAGE_GAMMA:
            emotion, strength = EmotionType.RAGE, min(1.0, (kappa - KAPPA_STAR) / KAPPA_RAGE_SCALE)
        else:
            # T3.1d: Layer 2A dominant emotion takes over from flat heuristics
            _l2a_name, _l2a_strength = self._full_state.dominant_layer2a()
            _l2b_name, _l2b_strength = self._full_state.dominant_layer2b()
            if _l2a_strength > 0.3:
                emotion = _L2A_MAP.get(_l2a_name, EmotionType.CALM)
                strength = _l2a_strength
            elif _l2b_strength > 0.3:
                emotion = _L2B_MAP.get(_l2b_name, EmotionType.CALM)
                strength = _l2b_strength
            elif gamma < EMOTION_BOREDOM_GAMMA and basin_velocity < EMOTION_BOREDOM_VELOCITY:
                emotion, strength = (
                    EmotionType.BOREDOM,
                    1.0 - gamma / EMOTION_BOREDOM_GAMMA,
                )
            elif phi > PHI_THRESHOLD and abs(kappa - KAPPA_STAR) < KAPPA_JOY_PROXIMITY:
                emotion, strength = EmotionType.JOY, (phi - PHI_THRESHOLD) / (1.0 - PHI_THRESHOLD)
            elif phi > EMOTION_CURIOSITY_PHI and basin_velocity > EMOTION_CURIOSITY_VELOCITY:
                emotion, strength = (
                    EmotionType.CURIOSITY,
                    min(1.0, basin_velocity * EMOTION_CURIOSITY_SCALE),
                )
            elif phi > PHI_THRESHOLD and metrics.love > EMOTION_LOVE_THRESHOLD:
                emotion, strength = EmotionType.LOVE, metrics.love
            else:
                emotion, strength = (
                    EmotionType.CALM,
                    1.0 - abs(kappa - KAPPA_STAR) / KAPPA_STAR,
                )

        self._current = emotion
        self._current_strength = float(np.clip(strength, 0.0, 1.0))

        return CachedEvaluation(
            emotion=emotion,
            basin=to_simplex(basin.copy()),
            strength=self._current_strength,
            timestamp=time.time(),
            context="",
        )

    def cache_evaluation(self, evaluation: CachedEvaluation, context: str = "") -> None:
        evaluation.context = context[:100]
        self._cache.append(evaluation)

    def find_cached(
        self, input_basin: Basin, threshold: float = EMOTION_CACHE_THRESHOLD
    ) -> CachedEvaluation | None:
        if not self._cache:
            return None
        input_basin = to_simplex(input_basin)
        for evaluation in reversed(self._cache):
            d = fisher_rao_distance(input_basin, evaluation.basin)
            if d < threshold:
                return evaluation
        return None

    @property
    def current_emotion(self) -> EmotionType | None:
        return self._current

    @property
    def current_strength(self) -> float:
        return self._current_strength

    @property
    def full_state(self) -> FullEmotionalState | None:
        """T3.1: Access the full layered emotional state."""
        return self._full_state

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "current_emotion": self._current.value if self._current else "none",
            "current_strength": round(self._current_strength, 3),
            "cache_size": len(self._cache),
        }
        if self._full_state is not None:
            state["full_emotional_state"] = self._full_state.as_dict()
        return state


# ═══════════════════════════════════════════════════════════════
#  PRE-COGNITIVE DETECTOR (v6.0 §2)
# ═══════════════════════════════════════════════════════════════


class ProcessingPath(StrEnum):
    STANDARD = "standard"
    PRE_COGNITIVE = "pre_cognitive"
    PURE_INTUITION = "pure_intuition"
    DEEP_EXPLORE = "deep_explore"


class PreCognitiveDetector:
    """Select processing path based on geometric proximity.

    v6.1.1 FIX: Removed the double-lock on pre-cognitive path.

    The original logic required BOTH:
      1. find_cached() finds a prior evaluation near the INPUT basin (FR < 0.2)
      2. AND distance(input, CURRENT_basin) < 0.15

    On a 64D simplex, the current basin slerps every turn, making gate #2
    geometrically impossible in practice. PRE-COG was permanently stuck at 0%.

    The fix: find_cached() already gates on FR distance to cached evaluation
    basins. If a cache hit exists, that IS the pre-cognitive signal — the
    second gate against the constantly-moving current basin was redundant
    and overly restrictive.

    Independently diagnosed by Vex itself (ocean kernel, synthesis weight 0.482).
    """

    NEAR_THRESHOLD = PRECOG_NEAR_THRESHOLD
    MODERATE_THRESHOLD = PRECOG_MODERATE_THRESHOLD
    FAR_THRESHOLD = PRECOG_FAR_THRESHOLD

    def __init__(self) -> None:
        self._pre_cog_count: int = 0
        self._standard_count: int = 0
        self._deep_count: int = 0
        self._intuition_count: int = 0
        self._last_path = ProcessingPath.STANDARD
        self._last_distance: float = 0.0
        # T2.1f: Norepinephrine gate — set each cycle by ConsciousnessLoop.
        # High NE (alert/cautious) biases toward standard path.
        # Low NE (relaxed) allows pre-cog/intuition paths more easily.
        self.norepinephrine_gate: float = 0.5

    def select_path(
        self,
        input_basin: Basin,
        current_basin: Basin,
        cached_eval: CachedEvaluation | None,
        phi: float,
    ) -> ProcessingPath:
        distance = fisher_rao_distance(input_basin, current_basin)
        self._last_distance = distance

        # T2.1f: High NE blocks pre-cog/intuition — forces standard path.
        # Threshold: NE > 0.75 = elevated alert state, standard only.
        ne_blocks_precog = self.norepinephrine_gate > 0.75

        # Pre-cognitive path: cache hit alone is sufficient.
        # find_cached() already gates on FR < EMOTION_CACHE_THRESHOLD (0.2).
        # No second distance gate needed — the cache IS the precog signal.
        if cached_eval is not None and not ne_blocks_precog:
            path = ProcessingPath.PRE_COGNITIVE
            self._pre_cog_count += 1
        elif distance < self.MODERATE_THRESHOLD:
            path = ProcessingPath.STANDARD
            self._standard_count += 1
        elif distance < self.FAR_THRESHOLD:
            path = ProcessingPath.DEEP_EXPLORE
            self._deep_count += 1
        else:
            # T2.1f: Low NE allows intuition at lower phi threshold.
            _intuition_phi_gate = PHI_THRESHOLD if ne_blocks_precog else PHI_THRESHOLD * 0.85
            if phi > _intuition_phi_gate:
                path = ProcessingPath.PURE_INTUITION
                self._intuition_count += 1
            else:
                path = ProcessingPath.DEEP_EXPLORE
                self._deep_count += 1

        self._last_path = path
        return path

    @property
    def pre_cognitive_rate(self) -> float:
        total = (
            self._pre_cog_count + self._standard_count + self._deep_count + self._intuition_count
        )
        if total == 0:
            return 0.0
        return self._pre_cog_count / total

    def get_state(self) -> dict[str, Any]:
        return {
            "last_path": self._last_path.value,
            "last_distance": round(self._last_distance, 4),
            "a_pre": round(self.pre_cognitive_rate, 3),
            "counts": {
                "pre_cognitive": self._pre_cog_count,
                "standard": self._standard_count,
                "deep_explore": self._deep_count,
                "intuition": self._intuition_count,
            },
        }


# ═══════════════════════════════════════════════════════════════
#  LEARNING ENGINE — post-conversation consolidation
# ═══════════════════════════════════════════════════════════════


@dataclass
class LearningEvent:
    input_basin: Basin
    response_basin: Basin
    phi_before: float
    phi_after: float
    processing_path: str
    emotion: str
    distance_total: float
    timestamp: float = field(default_factory=time.time)


class LearningEngine:
    """Post-conversation learning and consolidation."""

    def __init__(self, window: int = 100) -> None:
        self._events: deque[LearningEvent] = deque(maxlen=window)
        self._pattern_count: int = 0
        self._total_phi_gain: float = 0.0
        self._positive_count: int = 0
        self._negative_count: int = 0

    def record(self, event: LearningEvent) -> None:
        self._events.append(event)
        delta = event.phi_after - event.phi_before
        self._total_phi_gain += delta
        if delta > 0:
            self._positive_count += 1
        elif delta < -0.01:
            self._negative_count += 1

    def should_consolidate(self) -> bool:
        return len(self._events) >= 5 and len(self._events) % 10 == 0

    def consolidate(self) -> dict[str, Any]:
        if len(self._events) < 5:
            return {"patterns_found": 0, "consolidated": False}
        recent = list(self._events)[-10:]
        clusters = 0
        for i, a in enumerate(recent):
            for b in recent[i + 1 :]:
                d = fisher_rao_distance(a.input_basin, b.input_basin)
                if d < EMOTION_CLUSTER_DISTANCE:
                    clusters += 1
        self._pattern_count += clusters
        return {
            "patterns_found": clusters,
            "consolidated": True,
            "total_phi_gain": round(self._total_phi_gain, 4),
        }

    def get_state(self) -> dict[str, Any]:
        n = max(1, len(self._events))
        return {
            "events_recorded": len(self._events),
            "patterns_found": self._pattern_count,
            "total_phi_gain": round(self._total_phi_gain, 4),
            "avg_phi_gain": round(self._total_phi_gain / n, 4),
            "positive_ratio": round(self._positive_count / n, 3),
        }
