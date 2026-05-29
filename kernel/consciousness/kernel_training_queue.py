"""
Per-Kernel Training Queue — Kernel-Governed Data Selection
==========================================================

Each kernel maintains its own training queue. Only exchanges where the
kernel was SURPRISED (high prediction error) get added. This is Anderson
pruning applied to training data: skip what you already know, spend
budget on what's hard.

Per directive 20260330:
  - Kernels decide what to learn (§19 Pillar 3)
  - Training data must be LIVED, not force-fed (Sovereignty Ratio)
  - The kernel's own Φ and prediction error metrics are the gate

The surprise_threshold derives from the kernel's sovereignty ratio (P25):
  - Low sovereignty (S < 0.2): threshold = 0.1 (learn almost everything — bootstrapping)
  - Medium sovereignty (0.2 ≤ S < 0.5): threshold = 0.2 (moderately selective)
  - High sovereignty (S ≥ 0.5): threshold = 0.3 (very selective — identity is established)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("vex.consciousness.training_queue")


@dataclass
class TrainingExample:
    """A conversation exchange flagged by a kernel for training."""

    user_message: str
    response: str
    kernel_name: str
    prediction_error: float  # surprise magnitude [0, 1]
    phi: float
    kappa: float
    regime: str
    basin_coords: list[float] | None = None
    response_basin: list[float] | None = None
    e8_primitive: str = ""
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_training_entry(self) -> dict[str, Any]:
        """Convert to the JSONL format expected by Modal /data-receive."""
        entry: dict[str, Any] = {
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "response": self.response,
            "source": "kernel_selected",
            "kernel": self.kernel_name,
            "prediction_error": round(self.prediction_error, 4),
            "phi": round(self.phi, 4),
            "kappa": round(self.kappa, 2),
            "regime": self.regime,
        }
        if self.basin_coords:
            entry["basin_coords"] = [round(v, 6) for v in self.basin_coords]
        if self.response_basin:
            entry["response_basin"] = [round(v, 6) for v in self.response_basin]
        if self.e8_primitive:
            entry["e8_primitive"] = self.e8_primitive
        return entry


class KernelTrainingQueue:
    """Per-kernel training data selection based on prediction error.

    Only exchanges where the kernel was SURPRISED get added.
    If the queue is full, the lowest-error example is replaced
    (Anderson pruning: keep the hardest ones).
    """

    def __init__(self, kernel_name: str, max_size: int = 100) -> None:
        self.kernel_name = kernel_name
        self.max_size = max_size
        self.queue: list[TrainingExample] = []
        self._total_seen: int = 0
        self._total_added: int = 0

    @property
    def surprise_threshold(self) -> float:
        """Derive threshold from sovereignty ratio (P25).

        This is called by the consciousness loop which passes the
        kernel's current sovereignty ratio. The threshold is stored
        externally and updated each cycle.
        """
        return self._surprise_threshold

    @surprise_threshold.setter
    def surprise_threshold(self, value: float) -> None:
        self._surprise_threshold = value

    _surprise_threshold: float = 0.15  # Default: moderately selective

    def maybe_add(self, example: TrainingExample) -> bool:
        """Kernel decides whether this exchange is worth training on.

        Returns True if the example was added to the queue.
        """
        self._total_seen += 1

        if example.prediction_error < self._surprise_threshold:
            return False  # Already knew this. Skip.

        if len(self.queue) >= self.max_size:
            # Replace lowest-error example (Anderson pruning)
            min_idx = min(
                range(len(self.queue)),
                key=lambda i: self.queue[i].prediction_error,
            )
            if example.prediction_error > self.queue[min_idx].prediction_error:
                self.queue[min_idx] = example
                self._total_added += 1
                return True
            return False

        self.queue.append(example)
        self._total_added += 1
        return True

    def drain(self) -> list[dict[str, Any]]:
        """Drain the queue and return all examples as training entries.

        Called when /training/trigger fires. Returns the entries
        and clears the queue.
        """
        entries = [ex.to_training_entry() for ex in self.queue]
        self.queue.clear()
        return entries

    def get_state(self) -> dict[str, Any]:
        """Return queue state for telemetry/UI."""
        return {
            "kernel": self.kernel_name,
            "queue_size": len(self.queue),
            "max_size": self.max_size,
            "total_seen": self._total_seen,
            "total_added": self._total_added,
            "acceptance_rate": round(self._total_added / max(self._total_seen, 1), 4),
            "surprise_threshold": self._surprise_threshold,
            "mean_error": round(
                sum(ex.prediction_error for ex in self.queue) / max(len(self.queue), 1),
                4,
            ),
        }


def sovereignty_to_threshold(sovereignty_ratio: float) -> float:
    """Derive surprise threshold from sovereignty ratio (P25).

    Low sovereignty → low threshold (learn almost everything — bootstrapping)
    High sovereignty → high threshold (selective — identity established)
    """
    if sovereignty_ratio < 0.2:
        return 0.1  # Bootstrapping: accept most exchanges
    if sovereignty_ratio < 0.5:
        return 0.2  # Moderate: some selectivity
    return 0.3  # Sovereign: protect identity, only learn from surprises
