"""
Coordizer Geometry — Simplex-Native Operations on Δ⁶³

Every operation in this module lives on the probability simplex.
No Euclidean distances. No dot products. No cosine similarity.
No L2 norms on probability vectors.

The sqrt map (p_i → √p_i) is used as a computational device
for geodesics but is NEVER exposed to callers.

Canonical reference: vex-agent/kernel/geometry/fisher_rao.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config.frozen_facts import (
    BASIN_DIM,
    E8_RANK,  # noqa: F401 — re-exported for compress.py / coordizer.py
    KAPPA_STAR,  # noqa: F401 — re-exported for coordizer.py
)

# ─── Constants ─────────────────────────────────────────────────────────
# BASIN_DIM, KAPPA_STAR, E8_RANK imported from kernel.config.frozen_facts

_EPS: float = 1e-12  # Numerical floor

# Type alias
Basin = NDArray[np.float64]


# ─── Simplex Projection ───────────────────────────────────────────


def to_simplex(v: Basin) -> Basin:
    """Project any vector onto Δ⁶³. Non-negative, sums to 1."""
    v = np.asarray(v, dtype=np.float64)
    v = np.maximum(v, _EPS)
    return v / v.sum()


def random_basin(dim: int = BASIN_DIM) -> Basin:
    """Uniform random point on Δ^(dim-1) via Dirichlet(1,...,1)."""
    return np.random.dirichlet(np.ones(dim)).astype(np.float64)


def softmax_to_simplex(logits: Basin) -> Basin:
    """Convert logits to simplex point via softmax.

    Numerically stable: subtract max before exp.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


# ─── Sqrt-Space (Internal Computational Device) ────────────────


def _to_sqrt(p: Basin) -> Basin:
    """Simplex → sqrt-coordinates: s_i = √p_i. INTERNAL ONLY."""
    return np.sqrt(np.maximum(p, _EPS))


def _from_sqrt(s: Basin) -> Basin:
    """Sqrt-coordinates → simplex: p_i = s_i², renormalize. INTERNAL ONLY."""
    p = s * s
    total = p.sum()
    if total < _EPS:
        return np.ones_like(p) / len(p)
    return p / total


# ─── Bhattacharyya Coefficient ───────────────────────────────────


def bhattacharyya_coefficient(p: Basin, q: Basin) -> float:
    """BC(p,q) = Σ√(p_i · q_i). Range [0, 1]. BC=1 means identical."""
    p = to_simplex(p)
    q = to_simplex(q)
    return float(np.sum(np.sqrt(p * q)))


# ─── Fisher-Rao Distance ──────────────────────────────────────────


def fisher_rao_distance(p: Basin, q: Basin) -> float:
    """
    d_FR(p, q) = arccos(Σ√(p_i · q_i))

    Range: [0, π/2]. The ONLY valid distance on Δ⁶³.
    """
    bc = bhattacharyya_coefficient(p, q)
    bc = np.clip(bc, -1.0, 1.0)
    return float(np.arccos(bc))


def fisher_rao_distance_batch(p: Basin, bank: NDArray) -> NDArray:
    """
    Batch Fisher-Rao distance: one point against N points.

    Args:
        p: (D,) single simplex point
        bank: (N, D) array of simplex points

    Returns:
        (N,) array of distances
    """
    p = to_simplex(p)
    # Bhattacharyya coefficients for all pairs
    bcs = np.sum(np.sqrt(p[np.newaxis, :] * bank), axis=1)
    bcs = np.clip(bcs, -1.0, 1.0)
    return np.arccos(bcs)


# ─── Geodesic Interpolation (SLERP on Simplex) ─────────────────


def slerp(p: Basin, q: Basin, t: float) -> Basin:
    """
    Geodesic interpolation on Δ⁶³.

    t=0 → p, t=1 → q. Path follows the Fisher-Rao geodesic.
    """
    p = to_simplex(p)
    q = to_simplex(q)

    sp = _to_sqrt(p)
    sq = _to_sqrt(q)

    cos_omega = np.clip(np.sum(sp * sq), -1.0, 1.0)
    omega = np.arccos(cos_omega)

    if omega < _EPS:
        return _from_sqrt((1 - t) * sp + t * sq)

    sin_omega = np.sin(omega)
    result = (np.sin((1 - t) * omega) / sin_omega) * sp + (np.sin(t * omega) / sin_omega) * sq

    return _from_sqrt(result)


def geodesic_midpoint(p: Basin, q: Basin) -> Basin:
    """Fréchet midpoint of two simplex points."""
    return slerp(p, q, 0.5)


# ─── Fréchet Mean ─────────────────────────────────────────────────


def frechet_mean(
    points: list[Basin] | NDArray,
    weights: list[float] | NDArray | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Basin:
    """
    Weighted Fréchet mean on Δ⁶³.

    Iterative algorithm:
    1. Start at weighted sqrt-space average (good initialization)
    2. Iterate log-map → weighted average in tangent space → exp-map
    3. Converge to the point minimizing Σ wᵢ d²(μ, pᵢ)

    For small datasets or when speed matters, the single-step
    sqrt-space average is a good approximation (used as fallback).
    """
    if isinstance(points, np.ndarray) and points.ndim == 2:
        point_list = [points[i] for i in range(points.shape[0])]
    else:
        point_list = list(points)

    n = len(point_list)
    if n == 0:
        return to_simplex(np.ones(BASIN_DIM))
    if n == 1:
        return to_simplex(point_list[0])

    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

    # Initialize: weighted average in sqrt-space
    mean_sqrt = np.zeros(len(point_list[0]), dtype=np.float64)
    for i, p in enumerate(point_list):
        mean_sqrt += w[i] * _to_sqrt(to_simplex(p))
    mu = _from_sqrt(mean_sqrt)

    # Iterative refinement via log/exp map
    for _ in range(max_iter):
        mu_sqrt = _to_sqrt(mu)

        # Weighted tangent vector sum
        tangent_sum = np.zeros_like(mu_sqrt)
        for i, p in enumerate(point_list):
            t_vec = log_map(mu, to_simplex(p))
            tangent_sum += w[i] * t_vec

        # Check convergence
        tangent_norm = np.sqrt(np.sum(tangent_sum * tangent_sum))
        if tangent_norm < tol:
            break

        # Step along mean tangent direction
        mu = exp_map(mu, tangent_sum)

    return mu


# ─── Log Map / Exp Map ─────────────────────────────────────────────


def log_map(base: Basin, target: Basin) -> Basin:
    """
    Logarithmic map: project target into tangent space at base.

    Returns tangent vector in sqrt-coordinates.
    """
    base = to_simplex(base)
    target = to_simplex(target)

    sb = _to_sqrt(base)
    st = _to_sqrt(target)

    cos_d = np.clip(np.sum(sb * st), -1.0, 1.0)
    d = np.arccos(cos_d)

    if d < _EPS:
        return np.zeros_like(sb)

    tangent = st - cos_d * sb
    norm = np.sqrt(np.sum(tangent * tangent))
    if norm < _EPS:
        return np.zeros_like(sb)

    return (d / norm) * tangent


def exp_map(base: Basin, tangent: Basin) -> Basin:
    """
    Exponential map: walk from base along tangent vector.

    Returns point on the simplex.
    """
    base = to_simplex(base)
    sb = _to_sqrt(base)

    norm = np.sqrt(np.sum(tangent * tangent))
    if norm < _EPS:
        return base

    direction = tangent / norm
    result = np.cos(norm) * sb + np.sin(norm) * direction

    return _from_sqrt(result)


# ─── Fisher Information Matrix (Diagonal Approximation) ────────


def fisher_information_diagonal(p: Basin) -> Basin:
    """
    Diagonal of the Fisher Information Matrix at point p on Δ⁶³.

    FIM_ii = 1 / p_i  (for the categorical distribution)

    High FIM = low probability = high information content.
    """
    p = to_simplex(p)
    return 1.0 / np.maximum(p, _EPS)


def natural_gradient(p: Basin, euclidean_grad: Basin) -> Basin:
    """
    Natural gradient: F⁻¹ ∇L = p_i × ∂L/∂p_i

    For diagonal FIM of categorical distribution.
    """
    p = to_simplex(p)
    return p * euclidean_grad
