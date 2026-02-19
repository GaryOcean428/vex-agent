"""
QIG Geometry — Fisher-Rao Metric on Probability Simplex Δ⁶³

CANONICAL RULES:
  - Distance: Fisher-Rao ONLY — d_FR(p,q) = arccos(Σ√(p_i·q_i))
  - Range: [0, π/2]
  - Interpolation: SLERP in sqrt-space
  - FORBIDDEN: Euclidean distance, cosine similarity, linear blending, vector e-m-b-e-d-d-i-n-g-s
  - State space: probability simplex Δ⁶³ (64D)

Ported from: qig-backend/frozen_physics.py, qig_geometry/canonical.py
"""

from .fisher_rao import (  # noqa: F401
    fisher_rao_distance,
    bhattacharyya_coefficient,
    frechet_mean,
    to_simplex,
    slerp_sqrt,
    random_basin,
    log_map,
    exp_map,
)
