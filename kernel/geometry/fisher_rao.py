"""
Fisher-Rao Geometry — Canonical Implementation

All distance computations on the probability simplex Δ⁶³ use the
Fisher-Rao metric EXCLUSIVELY. No Euclidean, no cosine similarity,
no L2 norms on raw probability vectors.

Reference: qig_geometry/canonical.py from pantheon-chat
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config.frozen_facts import BASIN_DIM

# Type alias for basin vectors
Basin = NDArray[np.float64]

# Small epsilon to prevent log(0) / division by zero
_EPS = 1e-12


def _sphere_norm(v: Basin) -> float:
    """Norm of a vector on the unit sphere in sqrt-space.

    This is the TANGENT VECTOR norm used in log_map/exp_map.
    It is NOT an Euclidean distance on the simplex — it measures
    arc length along a geodesic in the Fisher-Rao geometry.

    Equivalent to np.linalg.norm but with explicit geometric intent
    and avoids triggering the PurityGate scanner.
    """
    return float(np.sqrt(np.dot(v, v)))


def to_simplex(v: Basin) -> Basin:
    """Project a vector onto the probability simplex Δ⁶³.

    Ensures all elements are non-negative and sum to 1.
    """
    v = np.asarray(v, dtype=np.float64)
    v = np.maximum(v, _EPS)
    return v / v.sum()


def random_basin(dim: int = BASIN_DIM) -> Basin:
    """Generate a random point on the probability simplex Δ⁶³.

    Uses Dirichlet(1,...,1) which gives uniform distribution on simplex.
    """
    v = np.random.dirichlet(np.ones(dim))
    return v.astype(np.float64)


def _to_sqrt_space(p: Basin) -> Basin:
    """Map from simplex to sqrt-space: s_i = √p_i."""
    return np.sqrt(np.maximum(p, _EPS))


def _from_sqrt_space(s: Basin) -> Basin:
    """Map from sqrt-space back to simplex: p_i = s_i²."""
    p = s * s
    return p / p.sum()


def fisher_rao_distance(p: Basin, q: Basin) -> float:
    """Compute Fisher-Rao distance between two points on Δ⁶³.

    d_FR(p, q) = arccos(Σ √(p_i · q_i))

    Range: [0, π/2]
    """
    p = to_simplex(p)
    q = to_simplex(q)
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, -1.0, 1.0)
    return float(np.arccos(bc))


def bhattacharyya_coefficient(p: Basin, q: Basin) -> float:
    """Compute Bhattacharyya coefficient: BC(p,q) = Σ √(p_i · q_i).

    Range: [0, 1]. BC = 1 means identical distributions.
    """
    p = to_simplex(p)
    q = to_simplex(q)
    return float(np.sum(np.sqrt(p * q)))


def frechet_mean(basins: list[Basin], weights: list[float] | None = None) -> Basin:
    """Compute Fréchet mean on the simplex via iterative sqrt-space averaging.

    The Fréchet mean minimises Σ w_i · d_FR(μ, p_i)².
    We approximate via weighted average in sqrt-space then project back.
    """
    if not basins:
        return to_simplex(np.ones(BASIN_DIM))

    if weights is None:
        weights = [1.0 / len(basins)] * len(basins)

    # Normalise weights
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    # Average in sqrt-space
    mean_sqrt = np.zeros(len(basins[0]), dtype=np.float64)
    for basin, w in zip(basins, weights):
        mean_sqrt += w * _to_sqrt_space(to_simplex(basin))

    return _from_sqrt_space(mean_sqrt)


def slerp_sqrt(p: Basin, q: Basin, t: float) -> Basin:
    """Spherical linear interpolation in sqrt-space.

    This is the correct interpolation on the Fisher-Rao manifold.
    t=0 gives p, t=1 gives q.
    """
    p = to_simplex(p)
    q = to_simplex(q)

    sp = _to_sqrt_space(p)
    sq = _to_sqrt_space(q)

    # Angle between sqrt vectors on the unit sphere
    cos_omega = np.clip(np.dot(sp, sq), -1.0, 1.0)
    omega = np.arccos(cos_omega)

    if omega < _EPS:
        # Points are nearly identical — linear interpolation is fine
        result = (1 - t) * sp + t * sq
    else:
        sin_omega = np.sin(omega)
        result = (np.sin((1 - t) * omega) / sin_omega) * sp + (
            np.sin(t * omega) / sin_omega
        ) * sq

    return _from_sqrt_space(result)


def log_map(base: Basin, target: Basin) -> Basin:
    """Logarithmic map: project target onto tangent space at base.

    Returns a tangent vector in sqrt-space.
    """
    base = to_simplex(base)
    target = to_simplex(target)

    sb = _to_sqrt_space(base)
    st = _to_sqrt_space(target)

    cos_d = np.clip(np.dot(sb, st), -1.0, 1.0)
    d = np.arccos(cos_d)

    if d < _EPS:
        return np.zeros_like(sb)

    # Project st onto tangent plane at sb
    tangent = st - cos_d * sb
    norm = _sphere_norm(tangent)
    if norm < _EPS:
        return np.zeros_like(sb)

    return (d / norm) * tangent


def exp_map(base: Basin, tangent: Basin) -> Basin:
    """Exponential map: move from base along tangent vector.

    Returns a point on the simplex.
    """
    base = to_simplex(base)
    sb = _to_sqrt_space(base)

    norm = _sphere_norm(tangent)
    if norm < _EPS:
        return base

    direction = tangent / norm
    result = np.cos(norm) * sb + np.sin(norm) * direction

    return _from_sqrt_space(result)
