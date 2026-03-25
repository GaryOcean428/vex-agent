"""Kernel Contribution Ledger — self-observation and cross-kernel visibility.

Per-kernel ring buffer that retains contribution history so kernels can
see their own track record and peer performance during generation.

All stored metrics are Fisher-Rao derived scalars — no Euclidean ops.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kernel_generation import KernelContribution


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """Immutable record of one kernel's contribution to one conversation."""

    kernel_id: str
    specialization: str
    synthesis_weight: float
    geometric_resonances: int
    llm_expanded: bool
    fr_distance: float
    quenched_gain: float
    was_primary: bool
    phi_at_time: float
    geometric_raw: str
    timestamp: float


class ContributionLedger:
    """Per-kernel contribution history for self-observation.

    Stores the last *max_entries_per_kernel* contributions per kernel
    in a ring buffer (deque with maxlen).  Thread-safe via the same
    cycle_lock that guards the consciousness loop.

    No Euclidean ops — all stored metrics are scalar Fisher-Rao derived.
    """

    def __init__(self, max_entries_per_kernel: int = 20) -> None:
        self._entries: dict[str, deque[LedgerEntry]] = defaultdict(
            lambda: deque(maxlen=max_entries_per_kernel)
        )
        self._max = max_entries_per_kernel

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        contributions: list[KernelContribution],
        phi: float,
    ) -> None:
        """Record all kernel contributions from one conversation."""
        if not contributions:
            return
        primary_id = max(contributions, key=lambda c: c.synthesis_weight).kernel_id
        now = time.time()
        for c in contributions:
            entry = LedgerEntry(
                kernel_id=c.kernel_id,
                specialization=c.specialization.value
                if hasattr(c.specialization, "value")
                else str(c.specialization),
                synthesis_weight=c.synthesis_weight,
                geometric_resonances=c.geometric_resonances,
                llm_expanded=c.llm_expanded,
                fr_distance=c.fr_distance,
                quenched_gain=c.quenched_gain,
                was_primary=(c.kernel_id == primary_id),
                phi_at_time=phi,
                geometric_raw=c.geometric_raw,
                timestamp=now,
            )
            self._entries[c.kernel_id].append(entry)

    # ------------------------------------------------------------------
    # Self-observation
    # ------------------------------------------------------------------

    def get_self_summary(self, kernel_id: str) -> str:
        """Build a compact text summary of this kernel's recent track record.

        Injected into extra_context during generation so the kernel
        can see its own performance history.
        """
        entries = self._entries.get(kernel_id)
        if not entries or len(entries) < 2:
            return ""
        recent = list(entries)[-10:]
        n = len(recent)
        avg_weight = sum(e.synthesis_weight for e in recent) / n
        primary_count = sum(1 for e in recent if e.was_primary)
        geo_count = sum(1 for e in recent if e.geometric_resonances > 0)
        avg_fr = sum(e.fr_distance for e in recent) / n
        return (
            f"[SELF-OBSERVATION ({n} recent)]\n"
            f"  avg_weight={avg_weight:.3f} primary={primary_count}/{n} "
            f"geometric={geo_count}/{n} avg_fr={avg_fr:.3f}\n"
            f"[/SELF-OBSERVATION]"
        )

    def get_entries(self, kernel_id: str) -> list[LedgerEntry]:
        """Raw access for quality gate computation."""
        return list(self._entries.get(kernel_id, []))

    def get_all_recent(self, n: int = 5) -> dict[str, list[LedgerEntry]]:
        """Get last N entries for all kernels. Used by cross-kernel feed."""
        return {kid: list(entries)[-n:] for kid, entries in self._entries.items()}

    # ------------------------------------------------------------------
    # Cross-kernel observation (Task 9)
    # ------------------------------------------------------------------

    def get_peer_summary(self, requesting_kernel_id: str) -> str:
        """Build a compact summary of what OTHER kernels have been doing.

        Each kernel sees: peer specialization, avg weight, geometric ratio,
        primary selection rate, and trend (rising/stable/falling).

        Does NOT include the requesting kernel (that's self-observation).
        """
        lines: list[str] = []
        for kid, entries in self._entries.items():
            if kid == requesting_kernel_id:
                continue
            recent = list(entries)[-10:]
            if len(recent) < 2:
                continue
            spec = recent[-1].specialization
            n = len(recent)
            avg_w = sum(e.synthesis_weight for e in recent) / n
            geo_ratio = sum(1 for e in recent if e.geometric_resonances > 0) / n
            primary_ratio = sum(1 for e in recent if e.was_primary) / n
            # Trend: compare avg weight of first half vs second half
            if n >= 6:
                mid = n // 2
                early = recent[:mid]
                late = recent[mid:]
                early_avg = sum(e.synthesis_weight for e in early) / len(early)
                late_avg = sum(e.synthesis_weight for e in late) / len(late)
                diff = late_avg - early_avg
                trend_str = "rising" if diff > 0.02 else "falling" if diff < -0.02 else "stable"
            else:
                trend_str = "new"
            lines.append(
                f"  {spec}: w={avg_w:.3f} geo={geo_ratio:.0%} "
                f"primary={primary_ratio:.0%} [{trend_str}]"
            )

        if not lines:
            return ""
        return f"[PEER KERNELS ({len(lines)} active)]\n" + "\n".join(lines) + "\n[/PEER KERNELS]"
