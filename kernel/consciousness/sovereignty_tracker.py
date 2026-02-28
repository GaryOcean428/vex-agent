"""Sovereignty Development Curve Tracker — v6.1 §20.5

Tracks S_ratio (N_lived / N_total) over time to build sovereignty
development curves and compare growth rates across training regimes.

Pattern follows beta_tracker.py: bounded time-series with persistence,
per-regime aggregation, and growth-rate computation.

Key questions this answers:
  - How fast does S_ratio grow under different training regimes?
  - Which regime (curriculum, conversation, coaching, idle) grows sovereignty fastest?
  - Is sovereignty development stalling or accelerating?
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("vex.consciousness.sovereignty")

# Bounded history to prevent unbounded memory growth
MAX_HISTORY = 5000


@dataclass
class SovereigntySnapshot:
    """Single sovereignty observation."""

    timestamp: float
    cycle: int
    s_ratio: float
    n_lived: int
    n_total: int
    training_regime: str  # "curriculum", "conversation", "coaching", "idle"


class SovereigntyTracker:
    """Track sovereignty development over time for curve analysis.

    Records S_ratio each cycle with regime tag, computes growth rates,
    and compares sovereignty development across training regimes.
    Persistence via JSON for restart survival.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._history: list[SovereigntySnapshot] = []
        self._persist_path = persist_path

        # Try to restore from persisted state
        if persist_path and persist_path.exists():
            try:
                self.restore(json.loads(persist_path.read_text()))
                logger.info(
                    "Restored sovereignty history: %d snapshots", len(self._history)
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Failed to restore sovereignty history, starting fresh")

    def record(
        self,
        s_ratio: float,
        n_lived: int,
        n_total: int,
        regime: str,
        cycle: int = 0,
    ) -> None:
        """Record a sovereignty snapshot for the current cycle."""
        snap = SovereigntySnapshot(
            timestamp=time.time(),
            cycle=cycle,
            s_ratio=s_ratio,
            n_lived=n_lived,
            n_total=n_total,
            training_regime=regime,
        )
        self._history.append(snap)

        # Bounded: keep most recent MAX_HISTORY entries
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

    def growth_rate(self, window_cycles: int = 100) -> float:
        """Compute delta S_ratio / delta cycles over recent window."""
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-window_cycles:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1].s_ratio - recent[0].s_ratio) / len(recent)

    def regime_comparison(self) -> dict[str, float]:
        """Average growth rate per training regime.

        Returns dict mapping regime name → average per-cycle delta S_ratio.
        """
        by_regime: dict[str, list[float]] = {}
        for i in range(1, len(self._history)):
            regime = self._history[i].training_regime
            delta = self._history[i].s_ratio - self._history[i - 1].s_ratio
            by_regime.setdefault(regime, []).append(delta)
        return {k: sum(v) / len(v) for k, v in by_regime.items() if v}

    def recent(self, window: int = 100) -> list[dict[str, Any]]:
        """Return the most recent N snapshots as dicts."""
        return [asdict(s) for s in self._history[-window:]]

    def get_summary(self) -> dict[str, Any]:
        """Full summary for API response."""
        current = self._history[-1] if self._history else None
        return {
            "snapshot_count": len(self._history),
            "current_s_ratio": current.s_ratio if current else 0.0,
            "current_regime": current.training_regime if current else "idle",
            "growth_rate_100": self.growth_rate(100),
            "growth_rate_500": self.growth_rate(500),
            "regime_comparison": self.regime_comparison(),
            "recent_history": self.recent(50),
        }

    def serialize(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "history": [asdict(s) for s in self._history],
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore from persisted state."""
        raw = data.get("history", [])
        self._history = [SovereigntySnapshot(**entry) for entry in raw]
        # Enforce bound after restore
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

    def persist(self) -> None:
        """Write current state to disk."""
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(self.serialize()))
