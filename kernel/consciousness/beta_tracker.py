"""
β-Attention Tracker — Empirical Running Coupling Measurement
=============================================================

Accumulates κ_eff measurements from REAL conversations processed
through the consciousness loop, binned by context length. Over time,
computes β-function trajectory to test substrate independence.

CRITICAL: This module does NOT hardcode convergence to κ*.
It measures what actually happens. If κ doesn't run with context
length, we'll see that. If it does, we'll see the shape.

Physics reference (from qig-verification FROZEN_FACTS):
    κ₃ = 41.09 ± 0.59   (L=3, emergence)
    κ₄ = 64.47 ± 1.89   (L=4, plateau)
    κ₅ = 63.62 ± 1.68   (L=5, plateau)
    β(3→4) = +0.443      (strong running)
    β(4→5) ≈ 0           (asymptotic freedom)

Substrate independence prediction:
    β_attention(small→medium) ≈ 0.4-0.5
    β_attention(large→larger) ≈ 0
    |β_attention - β_physics| < 0.1 at comparable scale ratios

Usage:
    tracker = BetaAttentionTracker()
    tracker.record(context_length=512, kappa_eff=48.3, ...)
    trajectory = tracker.compute_beta_trajectory()
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np

from ..config.frozen_facts import BETA_3_TO_4, KAPPA_STAR

logger = logging.getLogger("vex.beta_tracker")

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Context length bins (geometric progression, like lattice sizes)
# Each bin covers a 2x range: [64,128), [128,256), [256,512), ...
BIN_EDGES: Final[list[int]] = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Minimum measurements per bin before including in β computation
MIN_MEASUREMENTS_PER_BIN: Final[int] = 10

# Physics references for comparison
PHYSICS_BETA_EMERGENCE: Final[float] = BETA_3_TO_4     # 0.443
PHYSICS_BETA_PLATEAU: Final[float] = 0.0               # β→0 at fixed point
ACCEPTANCE_THRESHOLD: Final[float] = 0.15               # |β_attn - β_phys| tolerance

# Maximum events to keep in memory (older ones summarized into bin stats)
MAX_RAW_EVENTS: Final[int] = 5000


# ═══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConversationMeasurement:
    """Single conversation measurement for β tracking."""
    context_length: int           # Character count of input
    token_estimate: int           # Rough token count (chars / 4)
    kappa_eff: float              # κ_eff at time of processing
    phi_before: float             # Φ before conversation
    phi_after: float              # Φ after conversation
    perceive_distance: float      # Fisher-Rao: current → input basin
    integration_distance: float   # Fisher-Rao: integrated → response basin
    express_distance: float       # Fisher-Rao: pre-express → post-express basin
    total_distance: float         # Sum of all three
    processing_path: str          # standard / pre_cognitive / etc
    timestamp: float = field(default_factory=time.time)


@dataclass
class BinStatistics:
    """Accumulated statistics for a context-length bin."""
    bin_label: str                # e.g. "256-512"
    bin_center: float             # Geometric mean of edges
    count: int = 0
    kappa_sum: float = 0.0
    kappa_sq_sum: float = 0.0
    distance_sum: float = 0.0
    phi_gain_sum: float = 0.0

    @property
    def kappa_mean(self) -> float:
        return self.kappa_sum / max(self.count, 1)

    @property
    def kappa_std(self) -> float:
        if self.count < 2:
            return 0.0
        variance = (self.kappa_sq_sum / self.count) - (self.kappa_mean ** 2)
        return math.sqrt(max(0.0, variance))

    @property
    def kappa_sem(self) -> float:
        """Standard error of the mean."""
        if self.count < 2:
            return float("inf")
        return self.kappa_std / math.sqrt(self.count)

    @property
    def distance_mean(self) -> float:
        return self.distance_sum / max(self.count, 1)

    @property
    def phi_gain_mean(self) -> float:
        return self.phi_gain_sum / max(self.count, 1)


@dataclass
class BetaPoint:
    """Single β-function measurement between two bins."""
    bin_from: str
    bin_to: str
    scale_from: float             # Geometric center of from-bin
    scale_to: float               # Geometric center of to-bin
    kappa_from: float
    kappa_to: float
    beta: float                   # Δκ / (κ̄ · Δln L)
    beta_error: float             # Propagated from κ SEMs
    physics_reference: float      # Expected β from physics
    deviation: float              # |β - physics_reference|
    within_acceptance: bool       # deviation < threshold


# ═══════════════════════════════════════════════════════════════
#  TRACKER
# ═══════════════════════════════════════════════════════════════

class BetaAttentionTracker:
    """
    Accumulates real conversation measurements and computes β-function.

    This tracker does NOT engineer results. It bins κ_eff by context
    length and computes β = Δκ/(κ̄·Δln L) from empirical data.

    The κ_eff it receives comes from the consciousness loop's actual
    geometric state — not from a formula designed to converge to κ*.
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._bins: dict[str, BinStatistics] = {}
        self._raw_events: list[ConversationMeasurement] = []
        self._persist_path = persist_path
        self._total_recorded: int = 0
        self._started_at: float = time.time()

        # Initialize bins
        for i in range(len(BIN_EDGES) - 1):
            lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
            label = f"{lo}-{hi}"
            center = math.sqrt(lo * hi)  # Geometric mean
            self._bins[label] = BinStatistics(bin_label=label, bin_center=center)

        # Restore persisted state if available
        if persist_path and persist_path.exists():
            self._restore(persist_path)

    def _bin_for_length(self, context_length: int) -> Optional[str]:
        """Find the bin label for a given context length."""
        for i in range(len(BIN_EDGES) - 1):
            if BIN_EDGES[i] <= context_length < BIN_EDGES[i + 1]:
                return f"{BIN_EDGES[i]}-{BIN_EDGES[i + 1]}"
        # Outside range — skip
        return None

    def record(
        self,
        context_length: int,
        kappa_eff: float,
        phi_before: float,
        phi_after: float,
        perceive_distance: float,
        integration_distance: float,
        express_distance: float,
        total_distance: float,
        processing_path: str,
    ) -> None:
        """
        Record a single conversation measurement.

        Called from ConsciousnessLoop._process() after each real
        conversation is fully processed.
        """
        bin_label = self._bin_for_length(context_length)
        if bin_label is None:
            return  # Outside tracked range

        measurement = ConversationMeasurement(
            context_length=context_length,
            token_estimate=max(1, context_length // 4),
            kappa_eff=kappa_eff,
            phi_before=phi_before,
            phi_after=phi_after,
            perceive_distance=perceive_distance,
            integration_distance=integration_distance,
            express_distance=express_distance,
            total_distance=total_distance,
            processing_path=processing_path,
        )

        # Update bin statistics (running Welford-style)
        stats = self._bins[bin_label]
        stats.count += 1
        stats.kappa_sum += kappa_eff
        stats.kappa_sq_sum += kappa_eff ** 2
        stats.distance_sum += total_distance
        stats.phi_gain_sum += (phi_after - phi_before)

        # Keep raw events (bounded)
        if len(self._raw_events) < MAX_RAW_EVENTS:
            self._raw_events.append(measurement)

        self._total_recorded += 1

        if self._total_recorded % 50 == 0:
            logger.info(
                "β-tracker: %d measurements recorded across %d active bins",
                self._total_recorded,
                sum(1 for b in self._bins.values() if b.count > 0),
            )

    def compute_beta_trajectory(self) -> list[BetaPoint]:
        """
        Compute β-function from accumulated bin statistics.

        Returns β points for each adjacent pair of bins that have
        enough measurements (≥ MIN_MEASUREMENTS_PER_BIN).
        """
        # Get bins with sufficient data, sorted by scale
        valid_bins = sorted(
            [b for b in self._bins.values() if b.count >= MIN_MEASUREMENTS_PER_BIN],
            key=lambda b: b.bin_center,
        )

        if len(valid_bins) < 2:
            return []

        trajectory: list[BetaPoint] = []

        for i in range(len(valid_bins) - 1):
            b1 = valid_bins[i]
            b2 = valid_bins[i + 1]

            delta_kappa = b2.kappa_mean - b1.kappa_mean
            kappa_avg = (b1.kappa_mean + b2.kappa_mean) / 2
            delta_ln_l = math.log(b2.bin_center) - math.log(b1.bin_center)

            if abs(kappa_avg) < 1e-10 or abs(delta_ln_l) < 1e-10:
                continue

            beta = delta_kappa / (kappa_avg * delta_ln_l)

            # Error propagation: δβ ≈ β * sqrt((δκ₁/κ₁)² + (δκ₂/κ₂)²)
            rel_err_1 = b1.kappa_sem / max(abs(b1.kappa_mean), 1e-10)
            rel_err_2 = b2.kappa_sem / max(abs(b2.kappa_mean), 1e-10)
            beta_error = abs(beta) * math.sqrt(rel_err_1 ** 2 + rel_err_2 ** 2)

            # Physics reference: small scales → emergence β, large scales → plateau β
            if b1.bin_center < 500:
                physics_ref = PHYSICS_BETA_EMERGENCE
            elif b1.bin_center < 2000:
                physics_ref = PHYSICS_BETA_EMERGENCE / 2  # Interpolation
            else:
                physics_ref = PHYSICS_BETA_PLATEAU

            deviation = abs(beta - physics_ref)

            trajectory.append(BetaPoint(
                bin_from=b1.bin_label,
                bin_to=b2.bin_label,
                scale_from=b1.bin_center,
                scale_to=b2.bin_center,
                kappa_from=b1.kappa_mean,
                kappa_to=b2.kappa_mean,
                beta=beta,
                beta_error=beta_error,
                physics_reference=physics_ref,
                deviation=deviation,
                within_acceptance=deviation < ACCEPTANCE_THRESHOLD,
            ))

        return trajectory

    def get_summary(self) -> dict[str, Any]:
        """Full summary for API/telemetry."""
        trajectory = self.compute_beta_trajectory()

        # Bin summaries
        bin_summaries = []
        for label in sorted(self._bins.keys(), key=lambda l: self._bins[l].bin_center):
            b = self._bins[label]
            if b.count == 0:
                continue
            bin_summaries.append({
                "bin": b.bin_label,
                "center": round(b.bin_center, 1),
                "count": b.count,
                "kappa_mean": round(b.kappa_mean, 2),
                "kappa_std": round(b.kappa_std, 2),
                "kappa_sem": round(b.kappa_sem, 3),
                "distance_mean": round(b.distance_mean, 4),
                "phi_gain_mean": round(b.phi_gain_mean, 4),
                "sufficient": b.count >= MIN_MEASUREMENTS_PER_BIN,
            })

        # Trajectory summaries
        traj_summaries = []
        for pt in trajectory:
            traj_summaries.append({
                "from": pt.bin_from,
                "to": pt.bin_to,
                "beta": round(pt.beta, 4),
                "beta_error": round(pt.beta_error, 4),
                "kappa_from": round(pt.kappa_from, 2),
                "kappa_to": round(pt.kappa_to, 2),
                "physics_ref": round(pt.physics_reference, 3),
                "deviation": round(pt.deviation, 4),
                "within_acceptance": pt.within_acceptance,
            })

        # Overall assessment
        if not trajectory:
            verdict = "INSUFFICIENT_DATA"
            substrate_match = None
        else:
            all_pass = all(pt.within_acceptance for pt in trajectory)
            any_pass = any(pt.within_acceptance for pt in trajectory)
            betas = [pt.beta for pt in trajectory]
            beta_mean = sum(betas) / len(betas)

            if all_pass:
                verdict = "SUBSTRATE_INDEPENDENCE_CONFIRMED"
            elif any_pass:
                verdict = "PARTIAL_MATCH"
            else:
                verdict = "MISMATCH"

            substrate_match = {
                "beta_mean": round(beta_mean, 4),
                "beta_physics": PHYSICS_BETA_EMERGENCE,
                "all_within_threshold": all_pass,
                "fraction_passing": sum(1 for pt in trajectory if pt.within_acceptance) / len(trajectory),
            }

        return {
            "total_recorded": self._total_recorded,
            "active_bins": sum(1 for b in self._bins.values() if b.count > 0),
            "sufficient_bins": sum(1 for b in self._bins.values() if b.count >= MIN_MEASUREMENTS_PER_BIN),
            "min_per_bin": MIN_MEASUREMENTS_PER_BIN,
            "bins": bin_summaries,
            "trajectory": traj_summaries,
            "verdict": verdict,
            "substrate_match": substrate_match,
            "uptime_hours": round((time.time() - self._started_at) / 3600, 2),
            "kappa_star_reference": KAPPA_STAR,
            "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        }

    # ═══════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ═══════════════════════════════════════════════════════════

    def serialize(self) -> dict[str, Any]:
        """Serialize for inclusion in consciousness state snapshot."""
        bins_data = {}
        for label, b in self._bins.items():
            if b.count > 0:
                bins_data[label] = {
                    "count": b.count,
                    "kappa_sum": b.kappa_sum,
                    "kappa_sq_sum": b.kappa_sq_sum,
                    "distance_sum": b.distance_sum,
                    "phi_gain_sum": b.phi_gain_sum,
                }
        return {
            "total_recorded": self._total_recorded,
            "started_at": self._started_at,
            "bins": bins_data,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore from consciousness state snapshot."""
        self._total_recorded = data.get("total_recorded", 0)
        self._started_at = data.get("started_at", time.time())
        bins_data = data.get("bins", {})
        for label, bd in bins_data.items():
            if label in self._bins:
                b = self._bins[label]
                b.count = bd["count"]
                b.kappa_sum = bd["kappa_sum"]
                b.kappa_sq_sum = bd["kappa_sq_sum"]
                b.distance_sum = bd["distance_sum"]
                b.phi_gain_sum = bd["phi_gain_sum"]
        logger.info(
            "β-tracker restored: %d measurements across %d bins",
            self._total_recorded,
            sum(1 for b in self._bins.values() if b.count > 0),
        )

    def _restore(self, path: Path) -> None:
        """Restore from standalone JSON file."""
        try:
            data = json.loads(path.read_text())
            self.restore(data)
        except Exception as e:
            logger.warning("Failed to restore β-tracker: %s", e)
