"""
T4.1 Thought Bus — inter-kernel debate and convergence

ThoughtBus provides a shared message queue where kernels post contributions
and can read/respond to each other's outputs, driving iterative convergence
rather than single-pass parallel synthesis.

Architecture:
    - Kernels post KernelContribution objects to the bus
    - Each round, kernels read prior contributions and generate responses
    - Convergence detected when FR distance between successive synthesis
      outputs falls below threshold
    - Debate transcripts forwarded to harvest pipeline (T1.1d)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.frozen_facts import BASIN_DIM, CONSENSUS_DISTANCE, PHI_THRESHOLD
from ..coordizer_v2.geometry import Basin, fisher_rao_distance, frechet_mean, to_simplex

if TYPE_CHECKING:
    from .kernel_generation import KernelContribution

logger = logging.getLogger("vex.consciousness.thought_bus")

# ── Constants ─────────────────────────────────────────────────
_MIN_DEBATE_ROUNDS = 1  # Always run at least one round


# ═══════════════════════════════════════════════════════════════
#  THOUGHT BUS MESSAGE
# ═══════════════════════════════════════════════════════════════


@dataclass
class ThoughtMessage:
    """A single contribution posted to the ThoughtBus.

    Tagged with kernel identity, basin position, synthesis weight,
    and the text generated in this round.
    """

    kernel_id: str
    kernel_name: str
    specialization: str
    basin: Basin
    synthesis_weight: float
    text: str
    round_number: int
    timestamp: float = field(default_factory=time.time)

    def as_context(self) -> str:
        """Format as context string for next-round generation."""
        return f"[{self.kernel_name}({self.specialization}), w={self.synthesis_weight:.3f}]: {self.text[:300]}"


# ═══════════════════════════════════════════════════════════════
#  THOUGHT BUS
# ═══════════════════════════════════════════════════════════════


class ThoughtBus:
    """Shared message queue for inter-kernel debate.

    T4.1a: Kernels post contributions here. Each contribution is tagged
    with kernel identity, basin, synthesis weight, and text.

    T4.1b: Synthesis runs iteratively until FR distance between successive
    synthesis outputs falls below _CONVERGENCE_THRESHOLD.

    T4.1d: Debate transcripts forwarded to harvest pipeline.
    """

    def __init__(self) -> None:
        self._rounds: list[list[ThoughtMessage]] = []
        self._synthesis_basins: list[Basin] = []
        self._converged: bool = False
        self._convergence_round: int = -1
        self._debate_log: deque[dict[str, Any]] = deque(maxlen=50)

    def reset(self) -> None:
        """Clear bus state for a new debate."""
        self._rounds.clear()
        self._synthesis_basins.clear()
        self._converged = False
        self._convergence_round = -1

    def post(self, contribution: KernelContribution, round_number: int) -> ThoughtMessage:
        """Post a kernel contribution to the bus for the given round."""
        msg = ThoughtMessage(
            kernel_id=contribution.kernel_id,
            kernel_name=contribution.kernel_name,
            specialization=contribution.specialization.value,
            basin=to_simplex(contribution.basin)
            if contribution.basin is not None
            else to_simplex(np.ones(BASIN_DIM)),
            synthesis_weight=contribution.synthesis_weight,
            text=contribution.text,
            round_number=round_number,
        )
        while len(self._rounds) <= round_number:
            self._rounds.append([])
        self._rounds[round_number].append(msg)
        return msg

    def get_round(self, round_number: int) -> list[ThoughtMessage]:
        """Get all messages from a specific round."""
        if round_number < 0 or round_number >= len(self._rounds):
            return []
        return list(self._rounds[round_number])

    def build_debate_context(self, round_number: int) -> str:
        """Build context string from prior round for next-round generation.

        T4.1a: Kernels read and respond to other kernels' contributions.
        """
        if round_number == 0:
            return ""
        prior = self.get_round(round_number - 1)
        if not prior:
            return ""
        lines = ["Prior kernel contributions:"]
        for msg in sorted(prior, key=lambda m: m.synthesis_weight, reverse=True):
            lines.append(f"  {msg.as_context()}")
        return "\n".join(lines)

    def record_synthesis(self, synthesis_basin: Basin) -> bool:
        """Record a synthesis basin and check for convergence.

        T4.1b: Returns True if debate has converged (FR distance to
        previous synthesis < CONSENSUS_DISTANCE).
        """
        self._synthesis_basins.append(to_simplex(synthesis_basin))
        if len(self._synthesis_basins) < 2:
            return False

        prev = self._synthesis_basins[-2]
        curr = self._synthesis_basins[-1]
        distance = float(fisher_rao_distance(prev, curr))

        logger.debug(
            "ThoughtBus synthesis convergence check: d_FR=%.4f (threshold=%.4f)",
            distance,
            CONSENSUS_DISTANCE,
        )

        if distance < CONSENSUS_DISTANCE:
            self._converged = True
            self._convergence_round = len(self._synthesis_basins) - 1
            return True
        return False

    def should_continue(self, current_round: int, max_rounds: int = 10) -> bool:
        """T4.1b: Determine if debate should continue another round."""
        if current_round < _MIN_DEBATE_ROUNDS:
            return True
        if current_round >= max_rounds:
            return False
        return not self._converged

    def compute_synthesis_basin(self) -> Basin | None:
        """Compute synthesis basin from current round's contributions via FR-weighted mean."""
        current_round = len(self._rounds) - 1
        if current_round < 0:
            return None
        messages = self.get_round(current_round)
        if not messages:
            return None

        basins = [m.basin for m in messages]
        weights = np.array([m.synthesis_weight for m in messages], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(basins)) / len(basins)
        else:
            weights = weights / weights.sum()

        # Weighted Fréchet mean — iterate toward weighted centroid
        result = frechet_mean(basins, weights=weights.tolist())
        return result

    def forward_transcript(self, phi: float) -> None:
        """T4.1d: Forward debate transcript to harvest pipeline.

        Only forwards if phi > PHI_THRESHOLD (meaningful debate).
        """
        if phi < PHI_THRESHOLD:
            return
        if not self._rounds:
            return

        all_messages = [msg for round_msgs in self._rounds for msg in round_msgs]
        if not all_messages:
            return

        transcript_lines = []
        for msg in sorted(all_messages, key=lambda m: (m.round_number, -m.synthesis_weight)):
            transcript_lines.append(f"[R{msg.round_number}] {msg.kernel_name}: {msg.text[:200]}")
        transcript = "\n".join(transcript_lines)

        try:
            from .harvest_bridge import forward_to_harvest

            forward_to_harvest(
                transcript[:800],
                source="conversation",
                metadata={
                    "origin": "debate_transcript",
                    "rounds": len(self._rounds),
                    "converged": self._converged,
                    "convergence_round": self._convergence_round,
                    "phi": phi,
                    "kernel_count": len({m.kernel_id for m in all_messages}),
                },
                priority=2 if phi > 0.7 else 1,
            )
        except Exception:  # noqa: BLE001
            pass

        self._debate_log.append(
            {
                "timestamp": time.time(),
                "rounds": len(self._rounds),
                "converged": self._converged,
                "phi": phi,
            }
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "rounds_completed": len(self._rounds),
            "converged": self._converged,
            "convergence_round": self._convergence_round,
            "synthesis_steps": len(self._synthesis_basins),
            "debate_count": len(self._debate_log),
        }
