"""Core coordinate transformation functions.

Transform Euclidean input vectors to Fisher-Rao coordinates on the probability simplex.
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
    input_vector: np.ndarray,
    method: _TransformMethod = _TransformMethod.SOFTMAX,
    numerical_stability: bool = True,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Transform Euclidean input vector to Fisher-Rao coordinates.

    Uses exponential map to transform input vector to probability simplex:
    - Applies softmax (exponential + normalization)
    - Ensures non-negativity and sum-to-one
    - Validates simplex properties

    Args:
        input_vector: Euclidean vector (any dimensionality)
        method: Transformation method (SOFTMAX, SIMPLEX_PROJECTION, EXPONENTIAL_MAP)
        numerical_stability: Use log-sum-exp trick for numerical stability
        epsilon: Small value added for stability (default 1e-10)

    Returns:
        Coordinates on probability simplex Δ^(n-1):
        - All values non-negative
        - Sum to 1.0
        - Same shape as input

    Raises:
        ValueError: If input_vector is invalid or transformation fails

    Example:
        >>> input_vector = np.array([0.5, -0.3, 0.8, -0.1])
        >>> coords = coordize(input_vector)
        >>> print(coords.sum())  # Should be 1.0
        1.0
        >>> print(np.all(coords >= 0))  # All non-negative
        True
    """
    if not isinstance(input_vector, np.ndarray):
        raise TypeError(f"input_vector must be np.ndarray, got {type(input_vector)}")

    if input_vector.size == 0:
        raise ValueError("input_vector must not be empty")

    # Apply transformation based on method
    if method == _TransformMethod.SOFTMAX:
        coords = _softmax_transform(input_vector, numerical_stability, epsilon)
    elif method == _TransformMethod.SIMPLEX_PROJECTION:
        coords = _simplex_projection(input_vector, epsilon)
    elif method == _TransformMethod.EXPONENTIAL_MAP:
        coords = _exponential_map(input_vector, epsilon)
    else:
        raise ValueError(f"Unknown transformation method: {method}")

    # Ensure simplex properties (fail-closed)
    coords = ensure_simplex(coords, epsilon=epsilon)

    return coords


def coordize_batch(
    input_vectors: np.ndarray,
    method: _TransformMethod = _TransformMethod.SOFTMAX,
    numerical_stability: bool = True,
    epsilon: float = 1e-10,
) -> list[np.ndarray]:
    """Transform batch of input vectors to coordinates.

    Args:
        input_vectors: Array of input vectors, shape (batch_size, vector_dim)
        method: Transformation method
        numerical_stability: Use log-sum-exp trick
        epsilon: Numerical stability constant

    Returns:
        List of coordinate arrays, one per input vector

    Raises:
        ValueError: If input_vectors shape invalid or transformation fails

    Example:
        >>> input_vectors = np.array([
        ...     [0.5, -0.3, 0.8],
        ...     [-0.2, 0.4, 0.1],
        ... ])
        >>> coords_list = coordize_batch(input_vectors)
        >>> len(coords_list)
        2
    """
    if not isinstance(input_vectors, np.ndarray):
        raise TypeError(f"input_vectors must be np.ndarray, got {type(input_vectors)}")

    if input_vectors.ndim != 2:
        raise ValueError(
            f"input_vectors must be 2D (batch_size, dim), got shape {input_vectors.shape}"
        )

    results = []
    for input_vec in input_vectors:
        coords = coordize(input_vec, method, numerical_stability, epsilon)
        results.append(coords)

    return results


def _softmax_transform(
    input_vector: np.ndarray, numerical_stability: bool, epsilon: float
) -> np.ndarray:
    """Apply softmax transformation.

    Uses the numerically stable log-sum-exp trick:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        input_vector: Input Euclidean vector
        numerical_stability: Whether to use log-sum-exp trick
        epsilon: Stability constant

    Returns:
        Softmax-normalized coordinates
    """
    if numerical_stability:
        # Log-sum-exp trick for numerical stability
        # Subtract max to prevent overflow
        x_shifted = input_vector - np.max(input_vector)
        exp_x = np.exp(x_shifted)
        coords = exp_x / (np.sum(exp_x) + epsilon)
    else:
        # Standard softmax (can overflow for large values)
        exp_x = np.exp(input_vector)
        coords = exp_x / (np.sum(exp_x) + epsilon)

    return coords


def _simplex_projection(input_vector: np.ndarray, epsilon: float) -> np.ndarray:
    """Project vector onto probability simplex.

    Uses Euclidean projection onto the simplex:
    argmin_{p in Δ} ||p - x||^2 subject to p_i >= 0, sum(p) = 1

    This is a Euclidean operation, but acceptable here as it's just
    the initial transformation. The resulting coordinates are then used
    with Fisher-Rao operations.

    Args:
        input_vector: Input vector
        epsilon: Stability constant

    Returns:
        Projected coordinates on simplex
    """
    # Simple approach: make non-negative, then normalize
    # For more sophisticated projection, see:
    # Wang & Carreira-Perpinán (2013), "Projection onto the probability simplex"

    # Shift to make all positive
    shifted = input_vector - np.min(input_vector)

    # Add small epsilon for stability
    shifted = shifted + epsilon

    # Normalize to sum to 1
    coords = shifted / np.sum(shifted)

    return coords


def _exponential_map(input_vector: np.ndarray, epsilon: float) -> np.ndarray:
    """Exponential map from tangent space to manifold.

    This is a simplified exponential map. For the full Fisher-Rao
    exponential map, we would need to solve:
    γ(t) = argmax_{p in Δ} <∇_p log L(p), v>

    For now, we use softmax as an approximation.

    Args:
        input_vector: Tangent vector at base point
        epsilon: Stability constant

    Returns:
        Point on manifold (simplex)
    """
    # For now, use softmax as exponential map approximation
    # TODO: Implement full Fisher-Rao exponential map
    return _softmax_transform(input_vector, numerical_stability=True, epsilon=epsilon)


def create_transform_record(
    input_vector: np.ndarray,
    coordinates: np.ndarray,
    method: _TransformMethod,
    metadata: dict[str, any] | None = None,
) -> CoordinateTransform:
    """Create a transformation record.

    Args:
        input_vector: Original input vector
        coordinates: Transformed coordinates
        method: Method used for transformation
        metadata: Optional metadata

    Returns:
        CoordinateTransform record
    """
    from .types import CoordinateTransform

    return CoordinateTransform(
        input_vector=input_vector,
        coordinates=coordinates,
        method=method,
        timestamp=time.time(),
        metadata=metadata or {},
    )
