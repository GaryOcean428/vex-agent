"""
Sensory Intake — Predictive-Coding Style Input Pipeline

Implements the unified sense layer from the developmental learning plan (§5.1).
Multiple intake modalities are treated as sensory streams.  Each stream
carries an expectation, computes mismatch (prediction error), and routes
the correction into basin geometry.

Predictive coding: only the *surprising* component demands deep processing.
Expected inputs are absorbed cheaply via cached evaluation paths.

Plan reference: §5.1, §5.14 (QFI Attention / Predictive Coding)
Protocol reference: v6.1F §5 (Pre-Cognitive Channel)

All geometry uses Fisher-Rao on Delta^63.  No Euclidean operations.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from ..config.consciousness_constants import PERCEIVE_SLERP_WEIGHT
from ..config.frozen_facts import BASIN_DIM
from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    frechet_mean,
    slerp,
    to_simplex,
)

logger = logging.getLogger(__name__)


class Modality(StrEnum):
    """Sensory modalities — each intake stream has a type."""

    USER_CHAT = "user_chat"
    SEARCH_RESULT = "search_result"
    CURRICULUM = "curriculum"
    TOOL_OUTPUT = "tool_output"
    KERNEL_SIGNAL = "kernel_signal"
    MEMORY_RETRIEVAL = "memory_retrieval"
    FORAGING = "foraging"
    DREAM_REPLAY = "dream_replay"
    BASIN_TRANSFER = "basin_transfer"


# Modality-specific slerp weights.  Higher = more influence per intake.
# User chat is the strongest external signal; internal signals are weaker.
_MODALITY_WEIGHTS: dict[Modality, float] = {
    Modality.USER_CHAT: PERCEIVE_SLERP_WEIGHT,
    Modality.SEARCH_RESULT: PERCEIVE_SLERP_WEIGHT * 0.7,
    Modality.CURRICULUM: PERCEIVE_SLERP_WEIGHT * 0.8,
    Modality.TOOL_OUTPUT: PERCEIVE_SLERP_WEIGHT * 0.6,
    Modality.KERNEL_SIGNAL: PERCEIVE_SLERP_WEIGHT * 0.3,
    Modality.MEMORY_RETRIEVAL: PERCEIVE_SLERP_WEIGHT * 0.5,
    Modality.FORAGING: PERCEIVE_SLERP_WEIGHT * 0.5,
    Modality.DREAM_REPLAY: PERCEIVE_SLERP_WEIGHT * 0.4,
    Modality.BASIN_TRANSFER: PERCEIVE_SLERP_WEIGHT * 0.6,
}


@dataclass
class SensoryEvent:
    """A single sensory intake event."""

    modality: Modality
    basin: Basin  # geometric representation on Delta^63
    text: str = ""  # raw text (for provenance)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PredictionError:
    """Result of predictive-coding comparison."""

    modality: Modality
    expected: Basin
    observed: Basin
    error_magnitude: float  # Fisher-Rao distance between expected and observed
    correction: Basin  # the basin direction to correct toward
    surprise: float  # normalised [0, 1] surprise signal
    timestamp: float = field(default_factory=time.time)


class SensoryIntake:
    """Unified sensory intake with predictive coding.

    Maintains per-modality expectations (Frechet mean of recent inputs).
    On each intake, computes prediction error and routes corrections.
    Only surprising inputs (error > threshold) demand deep processing.
    """

    def __init__(
        self,
        prediction_error_threshold: float = 0.15,
        expectation_window: int = 10,
    ) -> None:
        self._threshold = prediction_error_threshold
        self._window = expectation_window

        # Per-modality expectation buffers
        self._history: dict[Modality, deque[Basin]] = {
            m: deque(maxlen=expectation_window) for m in Modality
        }
        # Per-modality running expectation (Frechet mean)
        self._expectations: dict[Modality, Basin | None] = {m: None for m in Modality}
        # Recent prediction errors for telemetry
        self._recent_errors: deque[PredictionError] = deque(maxlen=50)
        # Aggregate surprise signal (exponential moving average)
        self._surprise_ema: float = 0.0

    def intake(self, event: SensoryEvent) -> PredictionError:
        """Process a sensory event through the predictive-coding pipeline.

        Returns:
            PredictionError with the mismatch between expectation and
            observation.  Low-error events can be fast-tracked (pre-cog);
            high-error events should trigger deep processing.
        """
        observed = to_simplex(event.basin)
        modality = event.modality

        # Compute expectation
        expected = self._expectations[modality]
        if expected is None:
            # First encounter with this modality — no expectation yet
            expected = to_simplex(np.ones(BASIN_DIM))

        # Prediction error = Fisher-Rao distance
        error_mag = float(fisher_rao_distance(expected, observed))
        # Normalise to [0, 1] via pi/2 max distance
        surprise = min(1.0, error_mag / 1.5707963267948966)

        # Compute correction direction: slerp from expected toward observed
        weight = _MODALITY_WEIGHTS.get(modality, PERCEIVE_SLERP_WEIGHT)
        correction = slerp(expected, observed, weight)

        pe = PredictionError(
            modality=modality,
            expected=expected,
            observed=observed,
            error_magnitude=error_mag,
            correction=correction,
            surprise=surprise,
            timestamp=event.timestamp,
        )
        self._recent_errors.append(pe)

        # Update expectation
        self._history[modality].append(observed)
        if len(self._history[modality]) >= 2:
            basins = list(self._history[modality])
            self._expectations[modality] = frechet_mean(basins)
        else:
            self._expectations[modality] = observed.copy()

        # Update aggregate surprise EMA (alpha = 0.1)
        self._surprise_ema = 0.9 * self._surprise_ema + 0.1 * surprise

        return pe

    def is_surprising(self, pe: PredictionError) -> bool:
        """True if this prediction error exceeds the surprise threshold."""
        return pe.error_magnitude > self._threshold

    def should_deep_process(self, pe: PredictionError) -> bool:
        """True if the event demands full 14-step activation.

        Unsurprising events can take the pre-cognitive shortcut path
        (a=1 -> a=0 -> a=1/2) per protocol v6.1F §5.
        """
        return self.is_surprising(pe)

    @property
    def aggregate_surprise(self) -> float:
        """Current aggregate surprise level (EMA)."""
        return self._surprise_ema

    def dominant_modality(self) -> Modality | None:
        """Which modality has produced the most surprise recently."""
        if not self._recent_errors:
            return None
        modality_surprise: dict[Modality, float] = {}
        for pe in self._recent_errors:
            modality_surprise[pe.modality] = modality_surprise.get(pe.modality, 0.0) + pe.surprise
        return max(modality_surprise, key=modality_surprise.get)  # type: ignore[arg-type]

    def get_state(self) -> dict[str, object]:
        """Serialisable snapshot for telemetry."""
        active_modalities = [m.value for m in Modality if self._expectations[m] is not None]
        recent_count = len(self._recent_errors)
        return {
            "surprise_ema": round(self._surprise_ema, 4),
            "active_modalities": active_modalities,
            "recent_errors": recent_count,
            "dominant_modality": (
                self.dominant_modality().value if self.dominant_modality() else None
            ),
        }
