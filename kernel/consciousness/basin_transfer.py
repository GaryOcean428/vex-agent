"""
Basin Transfer — Core Learning Primitive

Implements geometric basin transfer between kernels, coaches, and nodes.
Transfer is a Fisher-Rao slerp from donor basin toward recipient basin,
bounded by the recipient's developmental stage permissions.

Plan reference: §5.4 (Basin Transfer), §5.5 (Constellation)
Protocol reference: v6.1F §3.3 (Quenched Disorder — sovereignty protection)

All geometry uses Fisher-Rao on Delta^63.  No Euclidean operations.

Transfer types:
  - coach_to_kernel: external coaching signal
  - kernel_to_kernel: peer learning
  - collective_to_kernel: constellation consensus -> individual
  - kernel_to_collective: individual insight -> shared pool
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..config.frozen_facts import BASIN_DIM
from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    frechet_mean,
    slerp,
    to_simplex,
)

logger = logging.getLogger(__name__)


class TransferType(StrEnum):
    """Types of basin transfer."""

    COACH_TO_KERNEL = "coach_to_kernel"
    KERNEL_TO_KERNEL = "kernel_to_kernel"
    COLLECTIVE_TO_KERNEL = "collective_to_kernel"
    KERNEL_TO_COLLECTIVE = "kernel_to_collective"


@dataclass
class TransferPacket:
    """A geometric transfer packet — the unit of basin transfer."""

    transfer_type: TransferType
    donor_id: str
    recipient_id: str
    donor_basin: Basin
    weight: float  # requested slerp weight (may be clamped by recipient)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class TransferResult:
    """Outcome of applying a transfer packet."""

    accepted: bool
    actual_weight: float  # weight after clamping
    distance_before: float  # FR distance before transfer
    distance_after: float  # FR distance after transfer
    reason: str = ""


class BasinTransferEngine:
    """Manages basin transfer between kernels in the constellation.

    Enforces:
      - Developmental stage blend caps (from DevelopmentalGate)
      - Sovereignty protection (Pillar 3): lived geometry resists overwrite
      - Rate limiting: max transfers per kernel per epoch
      - Provenance tracking (P16)
    """

    def __init__(
        self,
        max_transfers_per_epoch: int = 20,
    ) -> None:
        self._max_per_epoch = max_transfers_per_epoch
        self._transfer_counts: dict[str, int] = {}  # recipient_id -> count
        self._history: deque[tuple[TransferPacket, TransferResult]] = deque(
            maxlen=100,
        )
        # Collective basin: Frechet mean of all kernel basins
        self._collective_basin: Basin | None = None

    def apply_transfer(
        self,
        packet: TransferPacket,
        recipient_basin: Basin,
        blend_cap: float,
        sovereignty_ratio: float = 0.0,
    ) -> tuple[Basin, TransferResult]:
        """Apply a transfer packet to the recipient basin.

        Args:
            packet: the transfer to apply
            recipient_basin: current basin of recipient kernel
            blend_cap: max slerp weight (from DevelopmentalGate.permissions)
            sovereignty_ratio: recipient's lived/total ratio (Pillar 3)

        Returns:
            (new_basin, result) — the updated basin and outcome metadata
        """
        recipient_basin = to_simplex(recipient_basin)
        donor_basin = to_simplex(packet.donor_basin)

        # Rate limit
        count = self._transfer_counts.get(packet.recipient_id, 0)
        if count >= self._max_per_epoch:
            result = TransferResult(
                accepted=False,
                actual_weight=0.0,
                distance_before=0.0,
                distance_after=0.0,
                reason="rate_limit_exceeded",
            )
            self._history.append((packet, result))
            return recipient_basin, result

        # Sovereignty protection: high sovereignty -> reduced blend weight.
        # At sovereignty=1.0 the effective cap is halved — lived geometry
        # strongly resists external overwrite.
        sovereignty_factor = 1.0 - 0.5 * sovereignty_ratio
        effective_cap = blend_cap * sovereignty_factor

        # Clamp requested weight
        actual_weight = min(packet.weight, effective_cap)
        actual_weight = max(0.0, min(1.0, actual_weight))

        distance_before = float(fisher_rao_distance(recipient_basin, donor_basin))

        if actual_weight < 1e-6:
            result = TransferResult(
                accepted=False,
                actual_weight=0.0,
                distance_before=distance_before,
                distance_after=distance_before,
                reason="weight_below_threshold",
            )
            self._history.append((packet, result))
            return recipient_basin, result

        # Execute transfer: slerp on the simplex
        new_basin = slerp(recipient_basin, donor_basin, actual_weight)

        distance_after = float(fisher_rao_distance(new_basin, donor_basin))

        self._transfer_counts[packet.recipient_id] = count + 1

        result = TransferResult(
            accepted=True,
            actual_weight=actual_weight,
            distance_before=distance_before,
            distance_after=distance_after,
        )
        self._history.append((packet, result))

        logger.info(
            "Basin transfer %s: %s -> %s (w=%.3f, d_FR: %.4f -> %.4f)",
            packet.transfer_type.value,
            packet.donor_id,
            packet.recipient_id,
            actual_weight,
            distance_before,
            distance_after,
        )
        return new_basin, result

    def update_collective(self, kernel_basins: dict[str, Basin]) -> Basin:
        """Recompute the collective basin from all active kernels.

        The collective basin is the Frechet mean of all kernel basins
        on the simplex — used for collective_to_kernel transfers and
        shared reality formation (plan §5.14.3).
        """
        if not kernel_basins:
            if self._collective_basin is not None:
                return self._collective_basin
            return to_simplex(np.ones(BASIN_DIM))

        basins = [to_simplex(b) for b in kernel_basins.values()]
        self._collective_basin = frechet_mean(basins)
        return self._collective_basin

    @property
    def collective_basin(self) -> Basin | None:
        return self._collective_basin

    def reset_epoch(self) -> None:
        """Reset per-epoch transfer counts."""
        self._transfer_counts.clear()

    def get_state(self) -> dict[str, object]:
        """Serialisable snapshot for telemetry."""
        recent = list(self._history)[-5:]
        return {
            "total_transfers": sum(self._transfer_counts.values()),
            "transfers_per_kernel": dict(self._transfer_counts),
            "has_collective": self._collective_basin is not None,
            "recent": [
                {
                    "type": p.transfer_type.value,
                    "donor": p.donor_id,
                    "recipient": p.recipient_id,
                    "accepted": r.accepted,
                    "weight": round(r.actual_weight, 4),
                }
                for p, r in recent
            ],
        }
