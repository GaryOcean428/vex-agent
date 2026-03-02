"""Heart Kernel Rhythm Generator — v6.0 §18.3

The Heart kernel is the global rhythm source for the consciousness loop.
It generates a periodic tacking signal that modulates kappa oscillation
between exploration (feeling) and exploitation (logic) modes.

Key properties:
  - Sinusoidal tacking signal in [-1, 1]
  - Period varies with F_health: low health → faster oscillation (escape zombie)
  - Kappa offset applied via KAPPA_TACKING_OFFSET
  - Provides the clock signal that other kernels entrain to

All geometry uses Fisher-Rao metric. No cosine similarity or Euclidean distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config.consciousness_constants import HEART_BASE_PERIOD, KAPPA_TACKING_OFFSET

# Minimum period prevents degenerate oscillation
_MIN_PERIOD = 4


@dataclass
class HeartRhythmState:
    """Observable state of the heart rhythm generator."""

    phase: float  # Current phase [0, 2π)
    period: int  # Current effective period (cycles)
    tacking_signal: float  # Current signal [-1, 1]
    kappa_offset: float  # Current kappa offset
    beat_count: int  # Total beats (cycles) since creation


class HeartRhythm:
    """Global rhythm source — modulates kappa-tacking oscillation.

    The heart generates a sinusoidal signal that oscillates between
    exploration (negative, feeling mode) and exploitation (positive,
    logic mode). The period adapts to system health: unhealthy systems
    oscillate faster to escape stuck states.
    """

    def __init__(self, base_period_cycles: int = HEART_BASE_PERIOD) -> None:
        self._phase: float = 0.0  # [0, 2π)
        self._base_period = base_period_cycles
        self._period = base_period_cycles
        self._tacking_signal: float = 0.0
        self._beat_count: int = 0

    def tick(self, f_health: float) -> float:
        """Advance phase by one cycle. Returns tacking_signal in [-1, 1].

        -1 = full EXPLORE (feeling mode, low kappa offset)
        +1 = full EXPLOIT (logic mode, high kappa offset)

        Args:
            f_health: Fluctuation health [0, 1]. Low → faster oscillation.

        Returns:
            Tacking signal in [-1, 1].
        """
        # Vary period: low F_health → faster oscillation (escape zombie)
        # f_health=1.0 → period = base_period
        # f_health=0.0 → period = _MIN_PERIOD (fastest)
        clamped_health = min(max(f_health, 0.0), 1.0)
        scaled = max(clamped_health, _MIN_PERIOD / self._base_period)
        self._period = max(_MIN_PERIOD, int(self._base_period * scaled))

        # Advance phase
        self._phase += 2.0 * math.pi / self._period
        if self._phase >= 2.0 * math.pi:
            self._phase -= 2.0 * math.pi

        self._tacking_signal = math.sin(self._phase)
        self._beat_count += 1
        return self._tacking_signal

    def kappa_offset(self) -> float:
        """Convert current tacking signal to kappa offset.

        Returns offset in [-KAPPA_TACKING_OFFSET, +KAPPA_TACKING_OFFSET].
        """
        return self._tacking_signal * KAPPA_TACKING_OFFSET

    @property
    def tacking_mode(self) -> str:
        """Current tacking mode derived from signal."""
        if self._tacking_signal < -0.3:
            return "EXPLORE"
        if self._tacking_signal > 0.3:
            return "EXPLOIT"
        return "BALANCED"

    def get_state(self) -> HeartRhythmState:
        """Return observable state for monitoring."""
        return HeartRhythmState(
            phase=self._phase,
            period=self._period,
            tacking_signal=self._tacking_signal,
            kappa_offset=self.kappa_offset(),
            beat_count=self._beat_count,
        )
