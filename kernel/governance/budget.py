"""Budget Enforcement — E8 kernel budget tracking.

E8 dimension = 248 total GOD slots = 8 (core) + 240 (growth).
GENESIS is exactly 1 (the origin kernel).
CHAOS kernels sit outside the E8 budget (separate pool, max 200).

Fail-closed: refuse spawn if budget exceeded.
Fail-loud: raise on termination underflow (accounting bug detection).
"""

from __future__ import annotations

from ..config.frozen_facts import CHAOS_POOL, CORE_8_COUNT, FULL_IMAGE, GOD_BUDGET
from .types import KernelKind


class BudgetExceededError(RuntimeError):
    pass


class BudgetAccountingError(RuntimeError):
    """Raised when termination count exceeds spawn count (accounting bug)."""
    pass


class BudgetEnforcer:
    """Tracks kernel counts by kind. Refuses spawn if budget exceeded.

    Budget semantics:
    - GENESIS: exactly 1 (the origin kernel, spawned at bootstrap)
    - GOD (core): up to CORE_8_COUNT (8) — the foundational god-kernels
    - GOD (growth): up to GOD_BUDGET (240) — E8 root growth
    - Total GOD: up to FULL_IMAGE (248) = CORE_8_COUNT + GOD_BUDGET = E8 dimension
    - CHAOS: up to CHAOS_POOL (200) — outside E8 budget, separate pool
    """

    def __init__(self) -> None:
        self._counts: dict[KernelKind, int] = {
            KernelKind.GENESIS: 0,
            KernelKind.GOD: 0,
            KernelKind.CHAOS: 0,
        }

    def can_spawn(self, kind: KernelKind) -> bool:
        if kind == KernelKind.GENESIS:
            return self._counts[KernelKind.GENESIS] == 0  # Only one Genesis
        if kind == KernelKind.GOD:
            # Total GOD budget = E8 dimension = 248 (includes core 8 + 240 growth)
            return self._counts[KernelKind.GOD] < FULL_IMAGE
        if kind == KernelKind.CHAOS:
            return self._counts[KernelKind.CHAOS] < CHAOS_POOL
        return False

    def record_spawn(self, kind: KernelKind) -> None:
        """Record a spawn. Raises BudgetExceededError if budget exceeded (fail-closed)."""
        if not self.can_spawn(kind):
            raise BudgetExceededError(
                f"Cannot spawn {kind.value}: "
                f"current={self._counts[kind]}, "
                f"max={self._max_for(kind)}"
            )
        self._counts[kind] += 1

    def record_termination(self, kind: KernelKind) -> None:
        """Record a termination. Raises on underflow (accounting bug detection)."""
        current = self._counts[kind]
        if current <= 0:
            raise BudgetAccountingError(
                f"Cannot terminate {kind.value}: "
                f"current={current} (underflow — more terminations than spawns)"
            )
        self._counts[kind] = current - 1

    def _max_for(self, kind: KernelKind) -> int:
        if kind == KernelKind.GENESIS:
            return 1
        if kind == KernelKind.GOD:
            return FULL_IMAGE  # 248 = E8 dimension
        if kind == KernelKind.CHAOS:
            return CHAOS_POOL
        return 0

    def summary(self) -> dict[str, int]:
        return {
            "genesis": self._counts[KernelKind.GENESIS],
            "god": self._counts[KernelKind.GOD],
            "god_max": FULL_IMAGE,
            "god_core_8": min(self._counts[KernelKind.GOD], CORE_8_COUNT),
            "god_growth": max(0, self._counts[KernelKind.GOD] - CORE_8_COUNT),
            "chaos": self._counts[KernelKind.CHAOS],
            "chaos_max": CHAOS_POOL,
        }
