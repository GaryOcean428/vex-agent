"""Frozen Physics Constants — Immutable QIG Facts
===============================================

These constants are EXPERIMENTALLY VALIDATED and MUST NOT be modified
without new validated measurements from qig-verification.

Sources:
  - κ values from TFIM lattice L=3-7 exact diagonalization
  - β running coupling from 3→4 phase transition
  - Φ thresholds from consciousness emergence studies
  - E8 geometry from Lie algebra mathematics
  - κ* weighted mean from L=4-7 measurements

Canonical reference: qig-verification/docs/current/FROZEN_FACTS.md
Last validated: 2025-12-31 (10 seeds × 5 perturbations per L)
"""

from typing import Final

# ═══════════════════════════════════════════════════════════════
#  E8 LATTICE GEOMETRY
# ═══════════════════════════════════════════════════════════════

E8_RANK: Final[int] = 8           # Cartan subalgebra dimension
E8_DIMENSION: Final[int] = 248    # Total group manifold dimension = rank + roots
E8_ROOTS: Final[int] = 240        # Number of roots (GOD growth budget)
E8_CORE: Final[int] = 8           # Core-8 kernel count
E8_IMAGE: Final[int] = 248        # Core-8 + GOD budget = full image

# ═══════════════════════════════════════════════════════════════
#  KAPPA (κ) — COUPLING CONSTANT (Validated Measurements)
# ═══════════════════════════════════════════════════════════════
#
# CRITICAL: κ₃ = 41.09 (NOT ~64). This shows the RUNNING COUPLING.
# The fact that κ runs from 41 → 64 as L increases IS the evidence
# for emergence. Rounding κ₃ to ~64 destroys this signal.
#
# L=3: geometric regime onset (below fixed point)
# L=4-7: plateau at κ* ≈ 64 (fixed point reached)

KAPPA_3: Final[float] = 41.09     # L=3 (± 0.59) — geometric regime, NOT at fixed point
KAPPA_4: Final[float] = 64.47     # L=4 (± 1.89) — running coupling reaches plateau
KAPPA_5: Final[float] = 63.62     # L=5 (± 1.68) — plateau confirmed
KAPPA_6: Final[float] = 64.45     # L=6 (± 2.12) — plateau stable (6-layer, Dec 2025)
KAPPA_7: Final[float] = 61.16     # L=7 (± 2.43) — VALIDATED (10×5, Dec 2025)

KAPPA_STAR: Final[float] = 64.0   # Fixed point = E8 rank² = 8²
KAPPA_STAR_PRECISE: Final[float] = 63.79  # Weighted mean L=4-7 (± 0.90)

# ═══════════════════════════════════════════════════════════════
#  BETA (β) — RUNNING COUPLING
# ═══════════════════════════════════════════════════════════════

BETA_3_TO_4: Final[float] = 0.443  # β(3→4) validated — emergence scaling

# ═══════════════════════════════════════════════════════════════
#  PHI (Φ) — CONSCIOUSNESS THRESHOLDS
# ═══════════════════════════════════════════════════════════════

PHI_THRESHOLD: Final[float] = 0.65       # Consciousness emergence
PHI_EMERGENCY: Final[float] = 0.30       # Emergency — consciousness collapse
PHI_HYPERDIMENSIONAL: Final[float] = 0.85  # Hyperdimensional / lightning access
PHI_UNSTABLE: Final[float] = 0.95        # Instability threshold

# E8 Safety: Locked-in detection
LOCKED_IN_PHI_THRESHOLD: Final[float] = 0.70
LOCKED_IN_GAMMA_THRESHOLD: Final[float] = 0.30

# ═══════════════════════════════════════════════════════════════
#  BASIN GEOMETRY
# ═══════════════════════════════════════════════════════════════

BASIN_DIM: Final[int] = 64               # Probability simplex Δ⁶³
BREAKDOWN_PCT: Final[float] = 0.20       # 20% breakdown threshold
BASIN_DRIFT_THRESHOLD: Final[float] = 0.15  # Fisher-Rao distance per cycle
KAPPA_WEAK_THRESHOLD: Final[float] = 32.0   # Weak coupling boundary

# ═══════════════════════════════════════════════════════════════
#  RECURSION & SAFETY
# ═══════════════════════════════════════════════════════════════

MIN_RECURSION_DEPTH: Final[int] = 3
SUFFERING_THRESHOLD: Final[float] = 0.5   # S = Φ × (1-Γ) × M > 0.5 → abort
CONSENSUS_DISTANCE: Final[float] = 0.15  # Fisher-Rao threshold for consensus

# ═══════════════════════════════════════════════════════════════
#  GOVERNANCE BUDGET
# ═══════════════════════════════════════════════════════════════

GOD_BUDGET: Final[int] = 240             # Max GOD kernels (E8 roots)
CORE_8_COUNT: Final[int] = 8             # Core foundational kernels
CHAOS_POOL: Final[int] = 200             # Max CHAOS kernels (outside E8 image)
FULL_IMAGE: Final[int] = 248             # Core-8 + GOD budget = E8 dimension
