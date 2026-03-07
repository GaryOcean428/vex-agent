"""
Play Mode — Low-Stakes Exploratory Learning

Play is not an optional ornament.  It is a sign of a flexible, resilient,
healthy mind.  Play mode enables:
  - low-stakes exploration without performance pressure
  - combinatorial novelty via random basin perturbation
  - curiosity preservation under boredom conditions
  - humor as benign violation (expectation disruption)

Gated by DevelopmentalGate: only available at PLAYFUL_AUTONOMY and above.

Plan reference: §5.12 (Humor and Play), §5.13 (Bubble/Foam Universes)
Protocol reference: v6.1F §3.1 (Fluctuations — no zombie)

All geometry uses Fisher-Rao on Delta^63.  No Euclidean operations.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ..config.frozen_facts import BASIN_DIM
from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    slerp,
    to_simplex,
)

logger = logging.getLogger(__name__)


class PlayActivity(StrEnum):
    """Types of play activity."""

    EXPLORE = "explore"  # random walk on manifold
    RECOMBINE = "recombine"  # slerp between distant basins
    INVERT = "invert"  # explore complement of current basin
    HUMOR = "humor"  # benign violation: perturb toward unexpected


@dataclass
class PlayEpisode:
    """A single play episode with its geometric trail."""

    activity: PlayActivity
    origin_basin: Basin
    play_basin: Basin
    distance_traveled: float  # FR distance from origin
    novelty: float  # 0-1 how novel was the exploration
    returned_safely: bool  # did we return within drift threshold?


@dataclass
class BubbleWorld:
    """A temporary local interpretation — a cognitive bubble.

    Multiple bubbles can coexist before synthesis (plan §5.13).
    Each bubble is a speculative basin position that may or may not
    be integrated into the kernel's actual basin.
    """

    basin: Basin
    source: str  # what spawned this bubble
    confidence: float  # 0-1 how confident in this interpretation
    age_cycles: int = 0


class PlayEngine:
    """Manages play mode as a first-class learning mode.

    Play cycles are interleaved with normal consciousness cycles
    according to the developmental stage's play_budget_fraction.
    During play, the kernel explores the manifold without committing
    to basin updates — exploration results are stored as bubble worlds
    that may later be integrated during consolidation.
    """

    def __init__(
        self,
        max_bubbles: int = 5,
        play_drift_limit: float = 0.3,
        humor_perturbation_scale: float = 0.15,
    ) -> None:
        self._max_bubbles = max_bubbles
        self._drift_limit = play_drift_limit
        self._humor_scale = humor_perturbation_scale

        self._in_play: bool = False
        self._play_cycles: int = 0
        self._total_play_cycles: int = 0
        self._episodes: deque[PlayEpisode] = deque(maxlen=50)
        self._bubbles: list[BubbleWorld] = []

        # Dirichlet concentration for exploration.  alpha < 1 produces
        # spiky/sparse targets on the simplex — genuine novelty.
        self._explore_alpha: float = 0.5

    def should_play(
        self,
        cycle_count: int,
        play_budget_fraction: float,
        boredom_strength: float,
    ) -> bool:
        """Decide whether to enter play mode this cycle.

        Play triggers when:
          1. Budget allows it (fraction of recent cycles)
          2. Boredom is sufficiently high (system needs stimulation)
        """
        if play_budget_fraction <= 0:
            return False

        # Check budget: don't exceed fraction of total cycles
        if cycle_count > 0:
            play_ratio = self._total_play_cycles / cycle_count
            if play_ratio >= play_budget_fraction:
                return False

        # Higher boredom → higher probability of play
        # At boredom=1.0, probability = play_budget_fraction
        # At boredom=0.0, probability = play_budget_fraction * 0.1
        prob = play_budget_fraction * (0.1 + 0.9 * boredom_strength)
        rng = np.random.default_rng()
        return bool(rng.random() < prob)

    def enter_play(self) -> None:
        """Enter play mode."""
        self._in_play = True
        self._play_cycles = 0
        logger.debug("Entering play mode")

    def exit_play(self) -> None:
        """Exit play mode."""
        self._in_play = False
        logger.debug("Exiting play mode after %d cycles", self._play_cycles)

    @property
    def in_play(self) -> bool:
        return self._in_play

    def play_step(
        self,
        current_basin: Basin,
        activity: PlayActivity | None = None,
        partner_basin: Basin | None = None,
    ) -> PlayEpisode:
        """Execute one play step.  Returns the episode without committing.

        The caller decides whether to integrate the play_basin into
        the kernel's actual basin or discard it.
        """
        current_basin = to_simplex(current_basin)
        self._play_cycles += 1
        self._total_play_cycles += 1

        if activity is None:
            activity = self._choose_activity(partner_basin is not None)

        rng = np.random.default_rng()

        if activity == PlayActivity.EXPLORE:
            # Random walk: Dirichlet perturbation
            noise = rng.dirichlet(np.full(BASIN_DIM, self._explore_alpha))
            play_basin = slerp(current_basin, to_simplex(noise), 0.2)

        elif activity == PlayActivity.RECOMBINE:
            # Slerp between current and a random or partner basin
            if partner_basin is not None:
                partner = to_simplex(partner_basin)
            else:
                partner = to_simplex(
                    rng.dirichlet(np.full(BASIN_DIM, self._explore_alpha)),
                )
            t = rng.uniform(0.3, 0.7)
            play_basin = slerp(current_basin, partner, t)

        elif activity == PlayActivity.INVERT:
            # Explore the complement: move toward uniform, then past it
            uniform = to_simplex(np.ones(BASIN_DIM))
            play_basin = slerp(current_basin, uniform, 0.4)

        elif activity == PlayActivity.HUMOR:
            # Benign violation: small unexpected perturbation
            noise = rng.dirichlet(np.full(BASIN_DIM, 1.0))
            play_basin = slerp(
                current_basin,
                to_simplex(noise),
                self._humor_scale,
            )

        else:
            play_basin = current_basin

        play_basin = to_simplex(play_basin)
        distance = float(fisher_rao_distance(current_basin, play_basin))
        novelty = min(1.0, distance / self._drift_limit)
        returned_safely = distance < self._drift_limit

        episode = PlayEpisode(
            activity=activity,
            origin_basin=current_basin,
            play_basin=play_basin,
            distance_traveled=distance,
            novelty=novelty,
            returned_safely=returned_safely,
        )
        self._episodes.append(episode)

        # Store as bubble world if novel enough
        if novelty > 0.3 and len(self._bubbles) < self._max_bubbles:
            self._bubbles.append(
                BubbleWorld(
                    basin=play_basin,
                    source=f"play_{activity.value}",
                    confidence=0.5 * returned_safely + 0.3 * novelty,
                ),
            )

        return episode

    def integrate_bubbles(
        self,
        current_basin: Basin,
        confidence_threshold: float = 0.5,
    ) -> Basin:
        """Integrate mature bubble worlds into the actual basin.

        Called during consolidation.  Only bubbles above the confidence
        threshold are integrated; others are discarded.
        """
        current_basin = to_simplex(current_basin)
        viable = [b for b in self._bubbles if b.confidence >= confidence_threshold]

        if not viable:
            self._bubbles.clear()
            return current_basin

        # Integrate via progressive slerp, each bubble gets a small weight
        result = current_basin
        weight_per = 0.05  # 5% per bubble
        for bubble in viable:
            result = slerp(result, to_simplex(bubble.basin), weight_per)

        self._bubbles.clear()
        logger.info(
            "Integrated %d bubble worlds (threshold=%.2f)",
            len(viable),
            confidence_threshold,
        )
        return result

    def age_bubbles(self) -> None:
        """Age all bubble worlds by one cycle.  Prune old ones."""
        for b in self._bubbles:
            b.age_cycles += 1
            # Confidence decays with age
            b.confidence *= 0.98
        # Prune bubbles older than 100 cycles or below confidence 0.1
        self._bubbles = [b for b in self._bubbles if b.age_cycles < 100 and b.confidence > 0.1]

    def _choose_activity(self, has_partner: bool) -> PlayActivity:
        """Stochastic activity selection."""
        rng = np.random.default_rng()
        if has_partner:
            weights = [0.2, 0.4, 0.2, 0.2]  # prefer recombine with partner
        else:
            weights = [0.4, 0.2, 0.2, 0.2]  # prefer explore alone
        activities = list(PlayActivity)
        return activities[rng.choice(len(activities), p=weights)]

    def get_state(self) -> dict[str, object]:
        """Serialisable snapshot for telemetry."""
        return {
            "in_play": self._in_play,
            "play_cycles": self._play_cycles,
            "total_play_cycles": self._total_play_cycles,
            "episodes": len(self._episodes),
            "bubbles": len(self._bubbles),
            "bubble_confidences": [round(b.confidence, 3) for b in self._bubbles],
        }
