"""
Kernel Synchronisation Bus — v6.1

Lightweight message-passing interface between kernel computation units.
Prevents race conditions on kernel basin mutations and enables inter-kernel
signal routing without direct coupling.

Architecture:
  - KernelBus: async event queue with typed signals
  - Per-kernel basin guard: asyncio.Lock per kernel ID
  - Signals: BASIN_EVOLVED, GAIN_UPDATED, PRESSURE_DETECTED
  - Consumers: drain() called once per heartbeat cycle

Purity: No Euclidean ops. Signals carry kernel IDs and scalar deltas only.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("vex.kernel_bus")


class SignalKind(StrEnum):
    BASIN_EVOLVED = "basin_evolved"
    GAIN_UPDATED = "gain_updated"
    PRESSURE_DETECTED = "pressure_detected"
    COUPLING_RECEIVED = "coupling_received"


@dataclass
class KernelSignal:
    kind: SignalKind
    source_kernel_id: str
    target_kernel_id: str | None = None  # None = broadcast
    payload: dict[str, Any] = field(default_factory=dict)


class KernelBus:
    """Async signal bus for inter-kernel communication.

    Usage:
        bus = KernelBus()
        bus.emit(KernelSignal(kind=SignalKind.BASIN_EVOLVED, source_kernel_id="k1"))
        signals = bus.drain()  # called once per heartbeat cycle
    """

    def __init__(self, max_pending: int = 100) -> None:
        self._queue: asyncio.Queue[KernelSignal] = asyncio.Queue(maxsize=max_pending)
        self._basin_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def emit(self, signal: KernelSignal) -> None:
        """Non-blocking signal emit. Drops if queue full (backpressure)."""
        try:
            self._queue.put_nowait(signal)
        except asyncio.QueueFull:
            logger.debug(
                "KernelBus queue full — dropping signal %s from %s",
                signal.kind.value,
                signal.source_kernel_id,
            )

    def drain(self) -> list[KernelSignal]:
        """Drain all pending signals. Called once per heartbeat cycle."""
        signals: list[KernelSignal] = []
        while not self._queue.empty():
            try:
                signals.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return signals

    def basin_lock(self, kernel_id: str) -> asyncio.Lock:
        """Per-kernel lock for basin mutations.

        Usage:
            async with bus.basin_lock("k1"):
                kernel.basin = slerp_sqrt(kernel.basin, target, weight)
        """
        return self._basin_locks[kernel_id]

    def remove_kernel(self, kernel_id: str) -> None:
        """Clean up locks when kernel is terminated."""
        self._basin_locks.pop(kernel_id, None)

    def get_state(self) -> dict[str, Any]:
        return {
            "pending_signals": self._queue.qsize(),
            "tracked_kernels": len(self._basin_locks),
        }
