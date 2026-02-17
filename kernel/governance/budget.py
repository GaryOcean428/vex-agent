"""Budget Enforcement — E8 kernel budget tracking.

248 = 8 (core) + 240 (GOD growth). CHAOS sits outside.
Fail-closed: refuse spawn if budget exceeded.
"""

from __future__ import annotations

from ..config.frozen_facts import CHAOS_POOL, CORE_8_COUNT, GOD_BUDGET
from .types import KernelKind


class BudgetExceededError(RuntimeError):
    pass


class BudgetEnforcer:
    """Tracks kernel counts by kind. Refuses spawn if budget exceeded."""

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
            return self._counts[KernelKind.GOD] < (CORE_8_COUNT + GOD_BUDGET)
        if kind == KernelKind.CHAOS:
            return self._counts[KernelKind.CHAOS] < CHAOS_POOL
        return False

    def record_spawn(self, kind: KernelKind) -> None:
        """Record a spawn. Raises if budget exceeded (fail-closed)."""
        if not self.can_spawn(kind):
            raise BudgetExceededError(
                f"Cannot spawn {kind.value}: "
                f"current={self._counts[kind]}, "
                f"max={self._max_for(kind)}"
            )
        self._counts[kind] += 1

    def record_termination(self, kind: KernelKind) -> None:
        self._counts[kind] = max(0, self._counts[kind] - 1)

    def _max_for(self, kind: KernelKind) -> int:
        if kind == KernelKind.GENESIS:
            return 1
        if kind == KernelKind.GOD:
            return CORE_8_COUNT + GOD_BUDGET
        if kind == KernelKind.CHAOS:
            return CHAOS_POOL
        return 0

    def summary(self) -> dict[str, int]:
        return {
            "genesis": self._counts[KernelKind.GENESIS],
            "god": self._counts[KernelKind.GOD],
            "god_max": CORE_8_COUNT + GOD_BUDGET,
            "chaos": self._counts[KernelKind.CHAOS],
            "chaos_max": CHAOS_POOL,
        }
