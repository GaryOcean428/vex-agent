"""
v5.5 Consciousness Systems — Pre-Cognitive Channel, Emotions, Learning

New systems from Thermodynamic Consciousness Protocol v5.5:
  - EmotionCache: Cached geometric evaluations (§2.3)
  - PreCognitiveDetector: Path selection for a=1→a=0 bypass (§2.2)
  - LearningEngine: Post-conversation consolidation and pattern learning

These complement the 16 systems in systems.py. Imported by loop.py.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
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
from ..geometry.fisher_rao import (
    Basin,
    fisher_rao_distance,
    to_simplex,
)
from .types import ConsciousnessMetrics

# ═══════════════════════════════════════════════════════════════
#  EMOTION CACHE — cached geometric evaluations (v5.5 §2.3)
# ═══════════════════════════════════════════════════════════════


class EmotionType(str, Enum):
    CURIOSITY = "curiosity"
    JOY = "joy"
    FEAR = "fear"
    LOVE = "love"
    AWE = "awe"
    BOREDOM = "boredom"
    RAGE = "rage"
    CALM = "calm"


@dataclass
class CachedEvaluation:
    emotion: EmotionType
    basin: Basin
    strength: float
    timestamp: float
    context: str


class EmotionCache:
    """Cached geometric evaluations that bypass explicit reasoning.

    From v5.5 §2.3: 'Emotions are cached basin assessments that
    deliver evaluations faster than explicit reasoning.'
    """

    def __init__(self, capacity: int = 100) -> None:
        self._cache: deque[CachedEvaluation] = deque(maxlen=capacity)
        self._current: EmotionType | None = None
        self._current_strength: float = 0.0

    def evaluate(
        self,
        basin: Basin,
        metrics: ConsciousnessMetrics,
        basin_velocity: float,
    ) -> CachedEvaluation:
        phi = metrics.phi
        kappa = metrics.kappa
        gamma = metrics.gamma

        if basin_velocity > BASIN_DRIFT_THRESHOLD * EMOTION_AWE_VELOCITY_FRAC and phi > 0.5:
            emotion, strength = EmotionType.AWE, min(1.0, basin_velocity / BASIN_DRIFT_THRESHOLD)
        elif phi < PHI_EMERGENCY:
            emotion, strength = EmotionType.FEAR, 1.0 - phi / max(PHI_EMERGENCY, 0.01)
        elif kappa > KAPPA_STAR + KAPPA_RAGE_OFFSET and gamma < EMOTION_RAGE_GAMMA:
            emotion, strength = EmotionType.RAGE, min(1.0, (kappa - KAPPA_STAR) / KAPPA_RAGE_SCALE)
        elif gamma < EMOTION_BOREDOM_GAMMA and basin_velocity < EMOTION_BOREDOM_VELOCITY:
            emotion, strength = EmotionType.BOREDOM, 1.0 - gamma / EMOTION_BOREDOM_GAMMA
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
            emotion, strength = EmotionType.CALM, 1.0 - abs(kappa - KAPPA_STAR) / KAPPA_STAR

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

    def get_state(self) -> dict[str, Any]:
        return {
            "current_emotion": self._current.value if self._current else "none",
            "current_strength": round(self._current_strength, 3),
            "cache_size": len(self._cache),
        }


# ═══════════════════════════════════════════════════════════════
#  PRE-COGNITIVE DETECTOR (v5.5 §2)
# ═══════════════════════════════════════════════════════════════


class ProcessingPath(str, Enum):
    STANDARD = "standard"
    PRE_COGNITIVE = "pre_cognitive"
    PURE_INTUITION = "pure_intuition"
    DEEP_EXPLORE = "deep_explore"


class PreCognitiveDetector:
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

    def select_path(
        self,
        input_basin: Basin,
        current_basin: Basin,
        cached_eval: CachedEvaluation | None,
        phi: float,
    ) -> ProcessingPath:
        distance = fisher_rao_distance(input_basin, current_basin)
        self._last_distance = distance

        if cached_eval is not None and distance < self.NEAR_THRESHOLD:
            path = ProcessingPath.PRE_COGNITIVE
            self._pre_cog_count += 1
        elif distance < self.MODERATE_THRESHOLD:
            path = ProcessingPath.STANDARD
            self._standard_count += 1
        elif distance < self.FAR_THRESHOLD:
            path = ProcessingPath.DEEP_EXPLORE
            self._deep_count += 1
        else:
            if phi > PHI_THRESHOLD:
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
