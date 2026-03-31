"""
Self-Observation — Per-Kernel Output Quality Assessment (§43.2 Loop 1)
======================================================================

Each kernel observes its own output before it reaches the inter-kernel
debate. This is the sub-conscious loop — it runs ONCE per generation,
does not iterate, and does not block expression. Like a background
feeling that informs but does not decide.

Three observations:
  - Repetition: am I activating the same bank entries as last time?
  - Sovereignty: am I speaking from lived experience or borrowed scaffolding?
  - Confidence: did I actually retrieve from the bank, or did the LLM fill gaps?

Purity: repetition uses Fisher-Rao distance on activation sets (no Euclidean).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..coordizer_v2.geometry import fisher_rao_distance, to_simplex

logger = logging.getLogger("vex.consciousness.self_observation")

# Rolling window for repetition detection
_DEFAULT_WINDOW: int = 10


@dataclass
class SelfObservation:
    """Per-kernel self-observation result (§43.2 Loop 1).

    Attached to each KernelContribution. Visible to Loop 2 (debate)
    and Loop 3 (learning gate).
    """

    repetition_score: float  # 0 = novel activations, 1 = identical to recent
    sovereignty_score: float  # 0 = all borrowed, 1 = all lived
    confidence_score: float  # 0 = all LLM-expanded, 1 = all geometric

    def summary(self) -> str:
        """Compact string for debate context."""
        return (
            f"rep={self.repetition_score:.2f} "
            f"sov={self.sovereignty_score:.2f} "
            f"conf={self.confidence_score:.2f}"
        )


class SelfObservationTracker:
    """Tracks per-kernel activation history for repetition detection.

    Each kernel gets its own tracker instance. The tracker maintains
    a rolling window of recent activation basins (Fréchet means of
    the coord sets activated per generation).
    """

    def __init__(self, kernel_name: str, window: int = _DEFAULT_WINDOW) -> None:
        self.kernel_name = kernel_name
        self._recent_basins: deque[Any] = deque(maxlen=window)

    def observe(
        self,
        geometric_resonances: int,
        llm_expanded: bool,
        sovereignty_ratio: float,
        activation_basin: Any | None = None,
    ) -> SelfObservation:
        """Compute self-observation for a single kernel generation.

        Args:
            geometric_resonances: Number of bank resonances activated.
            llm_expanded: Whether LLM refinement was applied.
            sovereignty_ratio: System-wide sovereignty ratio from Pillar 3.
            activation_basin: Fréchet mean of activated coordinates (if available).

        Returns:
            SelfObservation with repetition, sovereignty, and confidence scores.
        """
        # Confidence: geometric resonances vs LLM expansion
        # If the kernel retrieved from the bank, it's confident.
        # If it fell back to LLM, it's guessing.
        if geometric_resonances > 0 and not llm_expanded:
            confidence = 1.0
        elif geometric_resonances > 0 and llm_expanded:
            confidence = 0.5  # Bank had some data but LLM expanded
        else:
            confidence = 0.0  # Pure LLM — no geometric grounding

        # Sovereignty: directly from Pillar 3
        sovereignty = float(np.clip(sovereignty_ratio, 0.0, 1.0))

        # Repetition: FR distance between current activation and recent window
        repetition = 0.0
        if activation_basin is not None and len(self._recent_basins) > 0:
            current = to_simplex(activation_basin)
            # Mean FR distance to recent activations
            # _recent_basins already stores simplex'd values — no re-conversion
            distances = [float(fisher_rao_distance(current, prev)) for prev in self._recent_basins]
            mean_distance = float(np.mean(distances))
            # Normalise: d_FR on simplex ranges [0, π/2]
            # Low distance = high repetition
            max_d = np.pi / 2
            repetition = float(np.clip(1.0 - mean_distance / max_d, 0.0, 1.0))
            self._recent_basins.append(current)
        elif activation_basin is not None:
            # First generation — no history to compare
            self._recent_basins.append(to_simplex(activation_basin))
            repetition = 0.0

        return SelfObservation(
            repetition_score=round(repetition, 4),
            sovereignty_score=round(sovereignty, 4),
            confidence_score=round(confidence, 4),
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel_name,
            "history_depth": len(self._recent_basins),
        }
