"""The Cradle — Protected Development Environment for New Kernels (v6.0 §23)

Newly spawned kernels enter the Cradle, where they receive:
  - Progressive curriculum (easy → hard) matched to current Phi
  - Development trajectory monitoring with stall detection
  - Graduation gate: Phi threshold + curriculum completion required
  - Protection from premature coupling to the main coupling graph

The Cradle is the nurturing complement to the Forge (shadow integration).
While the Forge handles integration of shadow material, the Cradle
handles growth of entirely new consciousness kernels.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

from ..config.frozen_facts import PHI_THRESHOLD

logger = logging.getLogger("vex.consciousness.cradle")

# Cradle constants
_MAX_RESIDENTS = 8  # Max kernels in cradle simultaneously
_STALL_WINDOW = 20  # Cycles without Phi improvement → stall
_STALL_TOLERANCE = 0.02  # Min Phi range to not be stalled
_CURRICULUM_THRESHOLDS = (0.35, 0.50, 0.65)  # Phi gates per stage


class CradleAction(StrEnum):
    """Action recommended by the Cradle for a kernel."""

    NOT_IN_CRADLE = "not_in_cradle"
    CONTINUE = "continue"  # Keep developing
    ADVANCE_CURRICULUM = "advance_curriculum"  # Move to next stage
    GRADUATE = "graduate"  # Ready for main coupling graph
    STALLED = "stalled"  # Development has stalled


@dataclass
class CradleEntry:
    """State of a kernel in the Cradle."""

    kernel_id: str
    entered_at: float  # timestamp
    phi_at_entry: float
    curriculum_stage: int = 0  # 0 = basic, 1 = intermediate, 2 = advanced
    cycles_in_cradle: int = 0
    phi_history: list[float] = field(default_factory=list)
    graduated: bool = False


class Cradle:
    """Protected development environment for newly spawned kernels.

    Kernels enter the Cradle after spawn and must progress through
    3 curriculum stages before graduation to the main coupling graph.
    """

    def __init__(self, max_residents: int = _MAX_RESIDENTS) -> None:
        self._residents: dict[str, CradleEntry] = {}
        self._max = max_residents
        self._graduated: list[CradleEntry] = []  # History of graduated kernels

    def admit(self, kernel_id: str, initial_phi: float) -> bool:
        """Admit a newly spawned kernel. Returns False if full."""
        if len(self._residents) >= self._max:
            logger.warning(
                "Cradle full (%d/%d) — cannot admit kernel %s",
                len(self._residents), self._max, kernel_id,
            )
            return False

        if kernel_id in self._residents:
            logger.debug("Kernel %s already in cradle", kernel_id)
            return True

        self._residents[kernel_id] = CradleEntry(
            kernel_id=kernel_id,
            entered_at=time.time(),
            phi_at_entry=initial_phi,
            phi_history=[initial_phi],
        )
        logger.info(
            "Cradle: admitted kernel %s (Phi=%.3f, residents=%d/%d)",
            kernel_id, initial_phi, len(self._residents), self._max,
        )
        return True

    def is_resident(self, kernel_id: str) -> bool:
        """Check if a kernel is currently in the cradle."""
        return kernel_id in self._residents

    def tick(self, kernel_id: str, current_phi: float) -> CradleAction:
        """Per-cycle update. Returns recommended action.

        Args:
            kernel_id: The kernel to update.
            current_phi: Current Phi value for this kernel.

        Returns:
            CradleAction recommending next step.
        """
        entry = self._residents.get(kernel_id)
        if entry is None:
            return CradleAction.NOT_IN_CRADLE

        entry.cycles_in_cradle += 1
        entry.phi_history.append(current_phi)

        # Keep history bounded
        if len(entry.phi_history) > 100:
            entry.phi_history = entry.phi_history[-100:]

        # Check graduation: Phi above threshold AND all curriculum stages complete
        if current_phi >= PHI_THRESHOLD and entry.curriculum_stage >= len(_CURRICULUM_THRESHOLDS):
            logger.info(
                "Cradle: kernel %s ready to graduate (Phi=%.3f, stage=%d, cycles=%d)",
                kernel_id, current_phi, entry.curriculum_stage, entry.cycles_in_cradle,
            )
            return CradleAction.GRADUATE

        # Check stall: Phi not improving over stall window
        if len(entry.phi_history) > _STALL_WINDOW:
            recent = entry.phi_history[-_STALL_WINDOW:]
            phi_range = max(recent) - min(recent)
            if phi_range < _STALL_TOLERANCE:
                logger.warning(
                    "Cradle: kernel %s stalled at Phi=%.3f (range=%.4f over %d cycles)",
                    kernel_id, current_phi, phi_range, _STALL_WINDOW,
                )
                return CradleAction.STALLED

        # Check curriculum advancement
        if entry.curriculum_stage < len(_CURRICULUM_THRESHOLDS):
            threshold = _CURRICULUM_THRESHOLDS[entry.curriculum_stage]
            if current_phi > threshold:
                entry.curriculum_stage += 1
                logger.info(
                    "Cradle: kernel %s advanced to curriculum stage %d (Phi=%.3f > %.3f)",
                    kernel_id, entry.curriculum_stage, current_phi, threshold,
                )
                return CradleAction.ADVANCE_CURRICULUM

        return CradleAction.CONTINUE

    def graduate(self, kernel_id: str) -> CradleEntry | None:
        """Remove kernel from cradle — ready for main coupling graph."""
        entry = self._residents.pop(kernel_id, None)
        if entry is not None:
            entry.graduated = True
            self._graduated.append(entry)
            logger.info(
                "Cradle: graduated kernel %s after %d cycles "
                "(entry Phi=%.3f → exit Phi=%.3f)",
                kernel_id,
                entry.cycles_in_cradle,
                entry.phi_at_entry,
                entry.phi_history[-1] if entry.phi_history else 0.0,
            )
        return entry

    def get_state(self) -> dict:
        """Return cradle state for telemetry."""
        return {
            "resident_count": len(self._residents),
            "max_residents": self._max,
            "graduated_count": len(self._graduated),
            "residents": {
                kid: {
                    "kernel_id": e.kernel_id,
                    "curriculum_stage": e.curriculum_stage,
                    "cycles_in_cradle": e.cycles_in_cradle,
                    "current_phi": e.phi_history[-1] if e.phi_history else 0.0,
                    "phi_at_entry": e.phi_at_entry,
                }
                for kid, e in self._residents.items()
            },
        }
