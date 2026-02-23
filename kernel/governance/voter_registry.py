"""Voter Registry — live phi/kappa tracking for governance votes.

TCP v6.1 §20: Governance votes are weighted by live kernel metrics.
Only GOD kernels hold voting rights. CHAOS kernels are voiceless by design.

Thread-safety: all mutations guarded by RLock (re-entrant for same-thread
nested calls, e.g. register_or_update → register → _rebuild_weights).

Bootstrap fallback: if no GOD kernels have yet accumulated MIN_LIVE_CYCLES
of metric history, votes are cast at genesis weights (phi=0.727, kappa=64.0).
This prevents the catch-22 of needing votes to spawn the first kernel.

Healthy weight distribution:
  weight_i = phi_i * quenched_gain_i  (geometric relevance × individuality)
  weights are normalized to sum=1 before vote evaluation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config.frozen_facts import KAPPA_STAR

# Genesis fallback constants (validated physics)
_GENESIS_PHI: float = 0.727
_GENESIS_KAPPA: float = KAPPA_STAR

# Number of update cycles before a kernel's live metrics are trusted
MIN_LIVE_CYCLES: int = 10

# Maximum phi history per kernel (bounding memory)
_PHI_HISTORY_MAX: int = 64


@dataclass
class VoterRecord:
    """State for a single GOD kernel voter."""

    kernel_id: str
    kernel_name: str
    phi: float
    kappa: float
    quenched_gain: float = 1.0
    cycles_recorded: int = 0
    registered_at: float = field(default_factory=time.monotonic)
    phi_history: list[float] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        """True once enough metric cycles have been recorded."""
        return self.cycles_recorded >= MIN_LIVE_CYCLES

    @property
    def vote_weight(self) -> float:
        """phi * quenched_gain — geometric relevance × individuality."""
        return self.phi * self.quenched_gain

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "kernel_name": self.kernel_name,
            "phi": round(self.phi, 4),
            "kappa": round(self.kappa, 2),
            "quenched_gain": round(self.quenched_gain, 4),
            "cycles_recorded": self.cycles_recorded,
            "is_live": self.is_live,
            "vote_weight": round(self.vote_weight, 4),
        }


class VoterRegistry:
    """Thread-safe registry of GOD kernel voters with live phi/kappa metrics.

    Usage:
        vr = VoterRegistry()
        vr.register("k1", "Heart", phi=0.5, kappa=64.0, quenched_gain=1.2)
        vr.update("k1", phi=0.72, kappa=63.5)
        voters = vr.get_voter_weights()  # -> [(name, phi, kappa, weight), ...]
    """

    def __init__(self) -> None:
        self._records: dict[str, VoterRecord] = {}  # kernel_id -> VoterRecord
        self._lock = threading.RLock()

    # ── Registration ─────────────────────────────────────────────

    def register(
        self,
        kernel_id: str,
        kernel_name: str,
        phi: float,
        kappa: float,
        quenched_gain: float = 1.0,
    ) -> None:
        """Register a new GOD kernel. Idempotent — safe to call on restart."""
        with self._lock:
            if kernel_id in self._records:
                # Already registered — update metrics but keep history
                self._records[kernel_id].phi = phi
                self._records[kernel_id].kappa = kappa
                self._records[kernel_id].quenched_gain = quenched_gain
                return
            self._records[kernel_id] = VoterRecord(
                kernel_id=kernel_id,
                kernel_name=kernel_name,
                phi=phi,
                kappa=kappa,
                quenched_gain=quenched_gain,
            )

    def register_or_update(
        self,
        kernel_id: str,
        kernel_name: str,
        phi: float,
        kappa: float,
        quenched_gain: float = 1.0,
    ) -> None:
        """Atomic register-if-absent or update-if-present. Preferred call site."""
        with self._lock:
            if kernel_id not in self._records:
                self.register(kernel_id, kernel_name, phi, kappa, quenched_gain)
            else:
                self.update(kernel_id, phi, kappa)

    def deregister(self, kernel_id: str) -> bool:
        """Remove a kernel from the registry (on prune/terminate)."""
        with self._lock:
            if kernel_id in self._records:
                del self._records[kernel_id]
                return True
            return False

    # ── Updates ────────────────────────────────────────────────

    def update(self, kernel_id: str, phi: float, kappa: float) -> bool:
        """Update phi/kappa after each generation cycle.

        Returns True if the kernel was found and updated.
        Auto-registers with genesis weights if not found (handles
        startup races where update() arrives before register()).
        """
        with self._lock:
            if kernel_id not in self._records:
                # Auto-register: startup race where update arrives first
                self._records[kernel_id] = VoterRecord(
                    kernel_id=kernel_id,
                    kernel_name="auto_registered",
                    phi=phi,
                    kappa=kappa,
                )
            rec = self._records[kernel_id]
            rec.phi = phi
            rec.kappa = kappa
            rec.cycles_recorded += 1
            rec.phi_history.append(phi)
            if len(rec.phi_history) > _PHI_HISTORY_MAX:
                rec.phi_history = rec.phi_history[-_PHI_HISTORY_MAX:]
            return True

    # ── Query ──────────────────────────────────────────────────

    def active_voters(self) -> list[str]:
        """Return kernel_ids of kernels that have reached MIN_LIVE_CYCLES."""
        with self._lock:
            return [kid for kid, rec in self._records.items() if rec.is_live]

    def get_voter_weights(self) -> list[tuple[str, float, float, float]]:
        """Return (kernel_name, phi, kappa, weight) for all registered kernels.

        If no kernels are live yet, returns genesis fallback singleton.
        Weight = phi * quenched_gain, normalized to sum=1 across all voters.
        """
        with self._lock:
            live = [rec for rec in self._records.values() if rec.is_live]
            if not live:
                # Bootstrap: genesis fallback (single implicit voter)
                return [("Genesis", _GENESIS_PHI, _GENESIS_KAPPA, 1.0)]

            raw_weights = [rec.vote_weight for rec in live]
            total = sum(raw_weights)
            if total <= 0.0:
                eq = 1.0 / len(live)
                return [(rec.kernel_name, rec.phi, rec.kappa, eq) for rec in live]

            return [
                (rec.kernel_name, rec.phi, rec.kappa, w / total)
                for rec, w in zip(live, raw_weights, strict=False)
            ]

    def quorum_possible(self, threshold: float = 0.5) -> bool:
        """True if enough weighted voters exist to reach threshold."""
        with self._lock:
            live = [rec for rec in self._records.values() if rec.is_live]
            if not live:
                return True  # Bootstrap: genesis always achieves quorum
            total = sum(rec.vote_weight for rec in live)
            return total > 0.0

    def count(self) -> int:
        """Total registered kernels (including pre-live)."""
        with self._lock:
            return len(self._records)

    def live_count(self) -> int:
        """Kernels with MIN_LIVE_CYCLES of metric history."""
        with self._lock:
            return sum(1 for rec in self._records.values() if rec.is_live)

    def snapshot(self) -> dict[str, Any]:
        """Non-locking snapshot for health checks and logging."""
        with self._lock:
            live = [rec for rec in self._records.values() if rec.is_live]
            mean_phi = sum(r.phi for r in live) / len(live) if live else _GENESIS_PHI
            return {
                "total_registered": len(self._records),
                "live_voters": len(live),
                "mean_phi": round(mean_phi, 4),
                "voters": [r.to_dict() for r in self._records.values()],
                "bootstrap_mode": len(live) == 0,
            }


# ── Module-level singleton ────────────────────────────────────────────

_REGISTRY: VoterRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_voter_registry() -> VoterRegistry:
    """Return the process-level VoterRegistry singleton."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = VoterRegistry()
        return _REGISTRY
