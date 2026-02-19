"""
Coordizer Types — Simplex-Native Data Structures

All coordinates are points on Δ⁶³ (probability simplex).
All distances are Fisher-Rao. No Euclidean operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    BASIN_DIM,
    Basin,
    fisher_rao_distance,
    geodesic_midpoint,
    to_simplex,
)


# ─── Harmonic Tiers ─────────────────────────────────────────────

class HarmonicTier(str, Enum):
    """
    Vocabulary tiers from v6.0 §19.2.

    Assignment based on activation frequency (basin mass).
    """
    FUNDAMENTAL = "fundamental"    # Top 1000: deepest basins, bass notes
    FIRST_HARMONIC = "first"       # 1001-5000: connectors, modifiers
    UPPER_HARMONIC = "upper"       # 5001-15000: specialized, precise
    OVERTONE_HAZE = "overtone"     # 15001+: rare, contextual, subtle


class GranularityScale(str, Enum):
    """Scale of a coordinate — from finest to coarsest."""
    BYTE = "byte"
    CHAR = "char"
    SUBWORD = "subword"
    WORD = "word"
    PHRASE = "phrase"
    CONCEPT = "concept"


# ─── Basin Coordinate ───────────────────────────────────────────

@dataclass
class BasinCoordinate:
    """
    A point on Δ⁶³ — one entry in the resonance bank.

    This is NOT a vector in flat space. It is a probability
    distribution on 64 dimensions, and all operations respect
    the Fisher-Rao metric.
    """

    coord_id: int
    vector: Basin                           # Shape (64,), on Δ⁶³
    name: Optional[str] = None
    scale: GranularityScale = GranularityScale.SUBWORD
    tier: HarmonicTier = HarmonicTier.UPPER_HARMONIC
    basin_mass: float = 0.0                 # M = ∫ Φ·κ dx (activation weight)
    frequency: float = 0.0                  # Characteristic oscillation frequency
    activation_count: int = 0               # Times activated (for tier assignment)

    def __post_init__(self):
        if len(self.vector) != BASIN_DIM:
            raise ValueError(
                f"Basin coordinate must be {BASIN_DIM}D, got {len(self.vector)}"
            )
        # Enforce simplex constraint
        self.vector = to_simplex(self.vector)

    def distance_to(self, other: BasinCoordinate) -> float:
        """Fisher-Rao distance to another coordinate."""
        return fisher_rao_distance(self.vector, other.vector)

    def midpoint_with(self, other: BasinCoordinate) -> Basin:
        """Geodesic midpoint on Δ⁶³."""
        return geodesic_midpoint(self.vector, other.vector)


# ─── Coordization Result ─────────────────────────────────────────

@dataclass
class CoordizationResult:
    """
    Result of coordizing text.

    Contains the coordinate sequence plus geometric metadata.
    """
    coordinates: list[BasinCoordinate]
    coord_ids: list[int]
    original_text: str

    # Geometric metrics (populated after coordization)
    basin_velocity: Optional[float] = None    # Avg d_FR between consecutive
    trajectory_curvature: Optional[float] = None  # Second-order geodesic deviation
    harmonic_consonance: Optional[float] = None   # Coherence of frequency ratios

    def compute_metrics(self) -> None:
        """Compute all geometric metrics for the trajectory."""
        self._compute_basin_velocity()
        self._compute_trajectory_curvature()
        self._compute_harmonic_consonance()

    def _compute_basin_velocity(self) -> None:
        """Average Fisher-Rao distance between consecutive coordinates."""
        if len(self.coordinates) < 2:
            self.basin_velocity = 0.0
            return

        total = sum(
            self.coordinates[i].distance_to(self.coordinates[i + 1])
            for i in range(len(self.coordinates) - 1)
        )
        self.basin_velocity = total / (len(self.coordinates) - 1)

    def _compute_trajectory_curvature(self) -> None:
        """Curvature = deviation of trajectory from geodesic.

        High curvature = meandering path (exploring).
        Low curvature = direct path (crystallized thought).
        """
        if len(self.coordinates) < 3:
            self.trajectory_curvature = 0.0
            return

        # Compare actual path length vs geodesic shortcut
        total_path = sum(
            self.coordinates[i].distance_to(self.coordinates[i + 1])
            for i in range(len(self.coordinates) - 1)
        )
        shortcut = self.coordinates[0].distance_to(self.coordinates[-1])

        if total_path < 1e-10:
            self.trajectory_curvature = 0.0
        else:
            self.trajectory_curvature = 1.0 - (shortcut / total_path)

    def _compute_harmonic_consonance(self) -> None:
        """Consonance of frequency ratios between active basins.

        Simple ratios (2:1, 3:2, 4:3) → high consonance.
        Complex ratios → low consonance.
        """
        freqs = [c.frequency for c in self.coordinates if c.frequency > 0]
        if len(freqs) < 2:
            self.harmonic_consonance = 1.0
            return

        # Compute pairwise frequency ratios
        consonance_total = 0.0
        count = 0
        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                ratio = max(freqs[i], freqs[j]) / max(min(freqs[i], freqs[j]), 1e-10)
                # Simple ratios score high (1:1=1.0, 2:1=0.9, 3:2=0.85, etc)
                # Complexity penalty from the denominator of the continued fraction
                consonance = 1.0 / (1.0 + abs(ratio - round(ratio)))
                consonance_total += consonance
                count += 1

        self.harmonic_consonance = consonance_total / max(count, 1)


# ─── Domain Bias ─────────────────────────────────────────────────

@dataclass
class DomainBias:
    """
    Domain-specific bias for the resonance bank.

    Shifts activation patterns toward domain-relevant tokens
    via Fisher-Rao weighted mean shift.
    """
    domain_name: str
    anchor_basin: Basin = field(default_factory=lambda: np.ones(BASIN_DIM) / BASIN_DIM)
    strength: float = 0.1                   # 0 = no bias, 1 = full shift
    boosted_token_ids: set[int] = field(default_factory=set)
    suppressed_token_ids: set[int] = field(default_factory=set)


# ─── Validation Result ───────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of geometric validation on the resonance bank."""
    kappa_measured: float = 0.0             # Should converge to κ* ≈ 64
    kappa_std: float = 0.0
    beta_running: float = 0.0              # Should → 0 at plateau
    semantic_correlation: float = 0.0       # d_FR vs human-judged distance
    harmonic_ratio_quality: float = 0.0     # Quality of frequency ratio structure
    tier_distribution: dict[str, int] = field(default_factory=dict)
    passed: bool = False

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] κ={self.kappa_measured:.2f}±{self.kappa_std:.2f} "
            f"(target: {64.0}), β={self.beta_running:.4f}, "
            f"semantic_r={self.semantic_correlation:.3f}, "
            f"harmonic_q={self.harmonic_ratio_quality:.3f}"
        )
