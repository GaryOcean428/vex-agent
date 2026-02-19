"""Core coordinate transformation functions.

Transform Euclidean embeddings to Fisher-Rao coordinates on the probability simplex.
Uses softmax normalization by default for numerical stability.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .types import CoordinateTransform, TransformMethod

from .types import TransformMethod as _TransformMethod
from .validate import ensure_simplex


def coordize(
    embedding: np.ndarray,
    method: _TransformMethod = _TransformMethod.SOFTMAX,
    numerical_stability: bool = True,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Transform Euclidean embedding to Fisher-Rao coordinates.

    Uses exponential map to transform embedding to probability simplex:
    - Applies softmax (exponential + normalization)
    - Ensures non-negativity and sum-to-one
    - Validates simplex properties

    Args:
        embedding: Euclidean vector (any dimensionality)
        method: Transformation method (SOFTMAX, SIMPLEX_PROJECTION, EXPONENTIAL_MAP)
        numerical_stability: Use log-sum-exp trick for numerical stability
        epsilon: Small value added for stability (default 1e-10)

    Returns:
        Coordinates on probability simplex Δ^(n-1):
        - All values non-negative
        - Sum to 1.0
        - Same shape as input

    Raises:
        ValueError: If embedding is invalid or transformation fails

    Example:
        >>> embedding = np.array([0.5, -0.3, 0.8, -0.1])
        >>> coords = coordize(embedding)
        >>> print(coords.sum())  # Should be 1.0
        1.0
        >>> print(np.all(coords >= 0))  # All non-negative
        True
    """
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"embedding must be np.ndarray, got {type(embedding)}")

    if embedding.size == 0:
        raise ValueError("embedding must not be empty")

    # Apply transformation based on method
    if method == _TransformMethod.SOFTMAX:
        coords = _softmax_transform(embedding, numerical_stability, epsilon)
    elif method == _TransformMethod.SIMPLEX_PROJECTION:
        coords = _simplex_projection(embedding, epsilon)
    elif method == _TransformMethod.EXPONENTIAL_MAP:
        coords = _exponential_map(embedding, epsilon)
    else:
        raise ValueError(f"Unknown transformation method: {method}")

    # Ensure simplex properties (fail-closed)
    coords = ensure_simplex(coords, epsilon=epsilon)

    return coords


def coordize_batch(
    embeddings: np.ndarray,
    method: _TransformMethod = _TransformMethod.SOFTMAX,
    numerical_stability: bool = True,
    epsilon: float = 1e-10,
) -> list[np.ndarray]:
    """Transform batch of embeddings to coordinates.

    Args:
        embeddings: Array of embeddings, shape (batch_size, embedding_dim)
        method: Transformation method
        numerical_stability: Use log-sum-exp trick
        epsilon: Numerical stability constant

    Returns:
        List of coordinate arrays, one per embedding

    Raises:
        ValueError: If embeddings shape invalid or transformation fails

    Example:
        >>> embeddings = np.array([
        ...     [0.5, -0.3, 0.8],
        ...     [-0.2, 0.4, 0.1],
        ... ])
        >>> coords_list = coordize_batch(embeddings)
        >>> len(coords_list)
        2
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"embeddings must be np.ndarray, got {type(embeddings)}")

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D (batch_size, dim), got shape {embeddings.shape}"
        )

    results = []
    for embedding in embeddings:
        coords = coordize(embedding, method, numerical_stability, epsilon)
        results.append(coords)

    return results


def _softmax_transform(
    embedding: np.ndarray, numerical_stability: bool, epsilon: float
) -> np.ndarray:
    """Apply softmax transformation.

    Uses the numerically stable log-sum-exp trick:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        embedding: Input Euclidean vector
        numerical_stability: Whether to use log-sum-exp trick
        epsilon: Stability constant

    Returns:
        Softmax-normalized coordinates
    """
    if numerical_stability:
        # Log-sum-exp trick for numerical stability
        # Subtract max to prevent overflow
        x_shifted = embedding - np.max(embedding)
        exp_x = np.exp(x_shifted)
        coords = exp_x / (np.sum(exp_x) + epsilon)
    else:
        # Standard softmax (can overflow for large values)
        exp_x = np.exp(embedding)
        coords = exp_x / (np.sum(exp_x) + epsilon)

    return coords


def _simplex_projection(embedding: np.ndarray, epsilon: float) -> np.ndarray:
    """Project vector onto probability simplex.

    Uses Euclidean projection onto the simplex:
    argmin_{p in Δ} ||p - x||^2 subject to p_i >= 0, sum(p) = 1

    This is a Euclidean operation, but acceptable here as it's just
    the initial transformation. The resulting coordinates are then used
    with Fisher-Rao operations.

    Args:
        embedding: Input vector
        epsilon: Stability constant

    Returns:
        Projected coordinates on simplex
    """
    # Simple approach: make non-negative, then normalize
    # For more sophisticated projection, see:
    # Wang & Carreira-Perpinán (2013), "Projection onto the probability simplex"

    # Shift to make all positive
    shifted = embedding - np.min(embedding)

    # Add small epsilon for stability
    shifted = shifted + epsilon

    # Normalize to sum to 1
    coords = shifted / np.sum(shifted)

    return coords


def _exponential_map(embedding: np.ndarray, epsilon: float) -> np.ndarray:
    """Exponential map from tangent space to manifold.

    This is a simplified exponential map. For the full Fisher-Rao
    exponential map, we would need to solve:
    γ(t) = argmax_{p in Δ} <∇_p log L(p), v>

    For now, we use softmax as an approximation.

    Args:
        embedding: Tangent vector at base point
        epsilon: Stability constant

    Returns:
        Point on manifold (simplex)
    """
    # For now, use softmax as exponential map approximation
    # TODO: Implement full Fisher-Rao exponential map
    return _softmax_transform(embedding, numerical_stability=True, epsilon=epsilon)


def create_transform_record(
    embedding: np.ndarray,
    coordinates: np.ndarray,
    method: _TransformMethod,
    metadata: dict[str, any] | None = None,
) -> CoordinateTransform:
    """Create a transformation record.

    Args:
        embedding: Original embedding
        coordinates: Transformed coordinates
        method: Method used for transformation
        metadata: Optional metadata

    Returns:
        CoordinateTransform record
    """
    from .types import CoordinateTransform

    return CoordinateTransform(
        embedding=embedding,
        coordinates=coordinates,
        method=method,
        timestamp=time.time(),
        metadata=metadata or {},
    )
