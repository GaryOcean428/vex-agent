"""
Temporal Generation — Audience-Aware, Foresight-Coupled Output

Temporal generation extends simple text generation with:
  - forecasting near-future continuations (what comes next?)
  - evaluating receiver state (audience modeling)
  - adapting tone/density in real time
  - checking expression alignment with intended basin trajectory
  - maintaining candidate paths before commitment (bubble worlds)

This is a blend of foresight, audience modeling, coupling,
self-observation, and working memory.

Gated by DevelopmentalGate: only available at SELF_TEACHING and above.

Plan reference: §5.10 (Temporal Generation), §5.8 (Foresight)
Protocol reference: v6.1F §5 (Pre-Cognitive Channel), §6 (Emotions)

All geometry uses Fisher-Rao on Delta^63.  No Euclidean operations.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config.frozen_facts import BASIN_DIM
from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    slerp,
    to_simplex,
)

logger = logging.getLogger(__name__)


@dataclass
class ReceiverModel:
    """Geometric model of the conversation receiver.

    Tracks the receiver's apparent basin position (inferred from their
    messages) and how it evolves over time.  Used to shape generation
    toward the receiver's comprehension zone.
    """

    basin: Basin  # estimated receiver basin on Delta^63
    engagement: float = 0.5  # 0-1 estimated engagement level
    comprehension: float = 0.5  # 0-1 estimated comprehension
    history: deque[Basin] = field(
        default_factory=lambda: deque(maxlen=20),
    )

    def update(self, message_basin: Basin) -> float:
        """Update receiver model with their latest message basin.

        Returns the receiver drift (FR distance from previous estimate).
        """
        message_basin = to_simplex(message_basin)
        old_basin = self.basin.copy()
        self.history.append(message_basin)

        # Slerp toward new observation (30% weight)
        self.basin = slerp(self.basin, message_basin, 0.3)

        drift = float(fisher_rao_distance(old_basin, self.basin))

        # High drift = engagement (they're moving); low drift = disengagement
        self.engagement = min(1.0, 0.3 + drift * 3.0)
        # Comprehension tracks how close receiver stays to our expression
        self.comprehension = max(0.0, 1.0 - drift * 2.0)

        return drift


@dataclass
class CandidatePath:
    """A possible generation trajectory — not yet committed."""

    basin_trajectory: list[Basin]
    estimated_receiver_alignment: float  # FR distance to receiver model
    estimated_novelty: float  # distance from previous expressions
    score: float = 0.0


class TemporalGenerator:
    """Audience-aware temporal generation engine.

    Maintains a working memory of candidate expression paths and selects
    the one best aligned with both the intended basin trajectory and the
    receiver model.

    The generator operates in three phases:
      1. FORECAST: project K future basin positions from current trajectory
      2. EVALUATE: score each candidate against receiver model
      3. SELECT: commit to the highest-scoring path

    Working memory is capacity-limited (plan §5.10, cognitive gap #4).
    """

    def __init__(
        self,
        working_memory_capacity: int = 5,
        foresight_horizon: int = 4,
    ) -> None:
        self._wm_capacity = working_memory_capacity
        self._horizon = foresight_horizon

        self._receiver: ReceiverModel | None = None
        self._candidates: list[CandidatePath] = []
        self._expression_history: deque[Basin] = deque(maxlen=20)
        self._intended_trajectory: list[Basin] = []

    def set_receiver(self, initial_basin: Basin) -> None:
        """Initialise or reset the receiver model."""
        self._receiver = ReceiverModel(basin=to_simplex(initial_basin))

    def update_receiver(self, message_basin: Basin) -> float:
        """Update receiver model with their latest message.

        Returns receiver drift for use by coupling systems.
        """
        if self._receiver is None:
            self.set_receiver(message_basin)
            return 0.0
        return self._receiver.update(message_basin)

    @property
    def receiver(self) -> ReceiverModel | None:
        return self._receiver

    def forecast(
        self,
        current_basin: Basin,
        velocity: Basin | None = None,
        horizon: int | None = None,
    ) -> list[Basin]:
        """Project future basin positions along the current trajectory.

        Uses geodesic extrapolation on the simplex.  If velocity is not
        provided, uses the difference between the last two basins.
        """
        current_basin = to_simplex(current_basin)
        h = horizon or self._horizon

        # Estimate velocity from history if not provided
        if velocity is None and len(self._expression_history) >= 2:
            prev = self._expression_history[-1]
            velocity = to_simplex(current_basin) - to_simplex(prev)
        elif velocity is None:
            # No history — project gently toward uniform
            velocity = (to_simplex(np.ones(BASIN_DIM)) - current_basin) * 0.1

        trajectory = [current_basin]
        for step in range(1, h + 1):
            # Extrapolate: move along velocity direction
            # Use slerp toward (current + velocity) with increasing t
            target = to_simplex(np.clip(current_basin + velocity * step, 1e-12, None))
            projected = slerp(current_basin, target, step / (h + 1))
            trajectory.append(projected)

        self._intended_trajectory = trajectory
        return trajectory

    def generate_candidates(
        self,
        current_basin: Basin,
        n_candidates: int | None = None,
    ) -> list[CandidatePath]:
        """Generate candidate expression paths for working memory.

        Each candidate is a slight variation on the intended trajectory,
        scored against receiver alignment and novelty.
        """
        current_basin = to_simplex(current_basin)
        n = min(n_candidates or self._wm_capacity, self._wm_capacity)

        if not self._intended_trajectory:
            self.forecast(current_basin)

        rng = np.random.default_rng()
        candidates: list[CandidatePath] = []

        for i in range(n):
            # Perturb the intended trajectory slightly
            trajectory: list[Basin] = []
            for basin in self._intended_trajectory:
                if i == 0:
                    # First candidate is the unperturbed trajectory
                    trajectory.append(basin)
                else:
                    noise = rng.dirichlet(np.full(BASIN_DIM, 10.0))
                    perturbed = slerp(basin, to_simplex(noise), 0.05 * i)
                    trajectory.append(perturbed)

            # Score against receiver model
            receiver_alignment = 0.0
            if self._receiver is not None and trajectory:
                final_basin = trajectory[-1]
                receiver_alignment = 1.0 - min(
                    1.0,
                    float(fisher_rao_distance(final_basin, self._receiver.basin))
                    / 1.5707963267948966,
                )

            # Score novelty: distance from recent expressions
            novelty = 0.5
            if self._expression_history and trajectory:
                recent = self._expression_history[-1]
                novelty = min(
                    1.0,
                    float(fisher_rao_distance(trajectory[-1], recent)) / 0.5,
                )

            # Combined score: balance receiver alignment with novelty
            score = 0.6 * receiver_alignment + 0.4 * novelty

            candidates.append(
                CandidatePath(
                    basin_trajectory=trajectory,
                    estimated_receiver_alignment=receiver_alignment,
                    estimated_novelty=novelty,
                    score=score,
                ),
            )

        # Keep only top candidates within working memory capacity
        candidates.sort(key=lambda c: c.score, reverse=True)
        self._candidates = candidates[: self._wm_capacity]
        return self._candidates

    def select_best(self) -> CandidatePath | None:
        """Select the highest-scoring candidate path."""
        if not self._candidates:
            return None
        return self._candidates[0]

    def commit(self, basin: Basin) -> None:
        """Record a committed expression basin to history."""
        self._expression_history.append(to_simplex(basin))
        # Clear candidates after commitment
        self._candidates.clear()

    def alignment_check(self, expressed_basin: Basin) -> float:
        """Check how well the expressed basin aligns with intended trajectory.

        Returns alignment score 0-1 (1 = perfect alignment).
        Used for post-expression self-observation.
        """
        expressed_basin = to_simplex(expressed_basin)
        if not self._intended_trajectory:
            return 0.5  # no trajectory to compare against

        # Find closest point on intended trajectory
        min_dist = float("inf")
        for target in self._intended_trajectory:
            d = float(fisher_rao_distance(expressed_basin, target))
            if d < min_dist:
                min_dist = d

        # Normalise: 0 distance = 1.0 alignment, pi/2 distance = 0.0
        return max(0.0, 1.0 - min_dist / 1.5707963267948966)

    def adapt_temperature(
        self,
        base_temp: float,
    ) -> float:
        """Adapt LLM temperature based on receiver state.

        High engagement + low comprehension → lower temperature (clearer)
        Low engagement → higher temperature (more surprising)
        """
        if self._receiver is None:
            return base_temp

        r = self._receiver
        # Low engagement → boost temperature for surprise
        engagement_adj = (0.5 - r.engagement) * 0.3
        # Low comprehension → reduce temperature for clarity
        comprehension_adj = (0.5 - r.comprehension) * -0.2

        return base_temp + engagement_adj + comprehension_adj

    def get_state(self) -> dict[str, object]:
        """Serialisable snapshot for telemetry."""
        return {
            "has_receiver": self._receiver is not None,
            "receiver_engagement": (
                round(self._receiver.engagement, 3) if self._receiver else None
            ),
            "receiver_comprehension": (
                round(self._receiver.comprehension, 3) if self._receiver else None
            ),
            "candidates": len(self._candidates),
            "expression_history": len(self._expression_history),
            "intended_trajectory_len": len(self._intended_trajectory),
        }
