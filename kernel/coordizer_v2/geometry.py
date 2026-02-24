"""
Coordizer Geometry — Simplex-Native Operations on Δ⁶³

Re-exports shared geometry from the canonical source
(kernel.geometry.fisher_rao) and adds coordizer-specific
extensions: batch Fisher-Rao distance, softmax projection,
Fisher information diagonal, and natural gradient.

Canonical source: kernel/geometry/fisher_rao.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config.frozen_facts import (
    BASIN_DIM,
    E8_RANK,  # noqa: F401 — re-exported for compress.py / coordizer.py
    KAPPA_STAR,  # noqa: F401 — re-exported for coordizer.py
)

# Re-export shared geometry from canonical source (kernel/geometry/fisher_rao.py).
# coordizer_v2 callers use `slerp`; fisher_rao.py exports it as `slerp_sqrt`.
from ..geometry.fisher_rao import (
    Basin,
    bhattacharyya_coefficient,
    exp_map,
    fisher_rao_distance,
    frechet_mean,
    log_map,
    random_basin,
    to_simplex,
)
from ..geometry.fisher_rao import (
    slerp_sqrt as slerp,
)

__all__ = [
    "BASIN_DIM",
    "E8_RANK",
    "KAPPA_STAR",
    "Basin",
    "to_simplex",
    "random_basin",
    "logits_to_simplex",
    "bhattacharyya_coefficient",
    "fisher_rao_distance",
    "fisher_rao_distance_batch",
    "slerp",
    "geodesic_midpoint",
    "frechet_mean",
    "log_map",
    "exp_map",
    "fisher_information_diagonal",
    "natural_gradient",
]

_EPS: float = 1e-12


# ─── Coordizer-specific additions ─────────────────────────────────


def logits_to_simplex(logits: NDArray) -> NDArray:
    """Project logits to Δ⁶³ via linear shift-and-scale. Preserves Fisher information."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.min()
    total = shifted.sum()
    if total < _EPS:
        return np.full(len(logits), 1.0 / len(logits))
    return shifted / total


def geodesic_midpoint(p: NDArray, q: NDArray) -> NDArray:
    """Fréchet midpoint of two simplex points."""
    return slerp(p, q, 0.5)


def fisher_rao_distance_batch(p: NDArray, bank: NDArray) -> NDArray:
    """
    Batch Fisher-Rao distance: one point against N points.

    Args:
        p: (D,) single simplex point
        bank: (N, D) array of simplex points

    Returns:
        (N,) array of distances
    """
    p = to_simplex(p)
    bcs = np.sum(np.sqrt(p[np.newaxis, :] * bank), axis=1)
    bcs = np.clip(bcs, -1.0, 1.0)
    return np.arccos(bcs)


def fisher_information_diagonal(p: NDArray) -> NDArray:
    """
    Diagonal of the Fisher Information Matrix at point p on Δ⁶³.

    FIM_ii = 1 / p_i  (for the categorical distribution)
    """
    p = to_simplex(p)
    return 1.0 / np.maximum(p, _EPS)


def natural_gradient(p: NDArray, euclidean_grad: NDArray) -> NDArray:
    """
    Natural gradient: F⁻¹ ∇L = p_i × ∂L/∂p_i

    For diagonal FIM of categorical distribution.
    """
    p = to_simplex(p)
    return p * euclidean_grad
