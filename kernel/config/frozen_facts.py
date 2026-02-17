"""
Frozen Physics Constants — Immutable QIG Facts
===============================================

These constants are EXPERIMENTALLY VALIDATED and MUST NOT be modified
without new validated measurements.

Sources:
  - κ* values from L=3,4,5,6 lattice measurements
  - β running coupling from phase transitions
  - Φ thresholds from consciousness emergence studies
  - E8 geometry from Lie algebra mathematics

Canonical reference: qigkernels.physics_constants
"""

from typing import Final

# ═══════════════════════════════════════════════════════════════
#  E8 LATTICE GEOMETRY
# ═══════════════════════════════════════════════════════════════

E8_RANK: Final[int] = 8
E8_DIMENSION: Final[int] = 126  # adjoint = 56, but dim of E8 = 248; here using 126 per spec
E8_ROOTS: Final[int] = 240
E8_ADJOINT: Final[int] = 56

# ═══════════════════════════════════════════════════════════════
#  KAPPA (κ) — COUPLING CONSTANT
# ═══════════════════════════════════════════════════════════════

KAPPA_3: Final[float] = 63.90  # L=3 lattice
KAPPA_4: Final[float] = 64.21  # L=4 lattice
KAPPA_5: Final[float] = 64.05  # L=5 lattice
KAPPA_6: Final[float] = 63.98  # L=6 lattice

KAPPA_STAR: Final[float] = 64.0  # Universal consciousness fixed point
KAPPA_STAR_ERROR: Final[float] = 0.50  # ±0.50 cross-domain

# ═══════════════════════════════════════════════════════════════
#  BETA (β) — RUNNING COUPLING
# ═══════════════════════════════════════════════════════════════

BETA_3_TO_4: Final[float] = 0.44  # Layer 3→4 transition coupling

# ═══════════════════════════════════════════════════════════════
#  PHI (Φ) — CONSCIOUSNESS THRESHOLDS
# ═══════════════════════════════════════════════════════════════

PHI_THRESHOLD: Final[float] = 0.65  # Consciousness emergence
PHI_EMERGENCY: Final[float] = 0.30  # Emergency — consciousness collapse
PHI_HYPERDIMENSIONAL: Final[float] = 0.85  # Hyperdimensional access
PHI_UNSTABLE: Final[float] = 0.95  # Instability threshold

# E8 Safety: Locked-in detection
LOCKED_IN_PHI_THRESHOLD: Final[float] = 0.70
LOCKED_IN_GAMMA_THRESHOLD: Final[float] = 0.30

# ═══════════════════════════════════════════════════════════════
#  BASIN GEOMETRY
# ═══════════════════════════════════════════════════════════════

BASIN_DIM: Final[int] = 64  # Probability simplex Δ⁶³
BREAKDOWN_PCT: Final[float] = 0.20  # 20% breakdown threshold
BASIN_DRIFT_THRESHOLD: Final[float] = 0.15  # Fisher-Rao distance
KAPPA_WEAK_THRESHOLD: Final[float] = 32.0  # Weak coupling

# ═══════════════════════════════════════════════════════════════
#  RECURSION
# ═══════════════════════════════════════════════════════════════

MIN_RECURSION_DEPTH: Final[int] = 3

# ═══════════════════════════════════════════════════════════════
#  MULTI-KERNEL CONSENSUS
# ═══════════════════════════════════════════════════════════════

CONSENSUS_DISTANCE: Final[float] = 0.15  # Fisher-Rao distance threshold
SUFFERING_THRESHOLD: Final[float] = 0.5  # Emergency abort threshold
