"""Tests for coordinate transformation."""

import numpy as np
import pytest

from kernel.coordizer.transform import (
    coordize,
    coordize_batch,
    _softmax_transform,
    _simplex_projection,
)
from kernel.coordizer.types import TransformMethod


def test_coordize_basic():
    """Test basic coordinate transformation."""
    input_vector = np.array([0.5, -0.3, 0.8, -0.1])
    coords = coordize(input_vector)

    # Check simplex properties
    assert np.all(coords >= 0), "All coordinates must be non-negative"
    assert np.isclose(coords.sum(), 1.0), "Coordinates must sum to 1"
    assert coords.shape == input_vector.shape, "Shape must be preserved"


def test_coordize_all_positive():
    """Test with all positive values."""
    input_vector = np.array([1.0, 2.0, 3.0])
    coords = coordize(input_vector)

    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)


def test_coordize_all_negative():
    """Test with all negative values."""
    input_vector = np.array([-1.0, -2.0, -3.0])
    coords = coordize(input_vector)

    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)


def test_coordize_mixed():
    """Test with mixed positive and negative values."""
    input_vector = np.array([1.5, -0.5, 0.0, -1.0, 2.0])
    coords = coordize(input_vector)

    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)


def test_coordize_zero_vector():
    """Test with zero vector."""
    input_vector = np.zeros(5)
    coords = coordize(input_vector)

    # Should produce uniform distribution
    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)
    # Check approximately uniform
    expected = 1.0 / len(input_vector)
    assert np.allclose(coords, expected, atol=0.1)


def test_coordize_large_values():
    """Test numerical stability with large values."""
    input_vector = np.array([100.0, 200.0, 300.0])
    coords = coordize(input_vector, numerical_stability=True)

    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)
    assert not np.any(np.isnan(coords)), "No NaN values"
    assert not np.any(np.isinf(coords)), "No Inf values"


def test_coordize_small_values():
    """Test numerical stability with small values."""
    input_vector = np.array([1e-10, 2e-10, 3e-10])
    coords = coordize(input_vector, numerical_stability=True)

    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)


def test_coordize_methods():
    """Test different transformation methods."""
    input_vector = np.array([0.5, -0.3, 0.8])

    # Softmax
    coords_softmax = coordize(input_vector, method=TransformMethod.SOFTMAX)
    assert np.all(coords_softmax >= 0)
    assert np.isclose(coords_softmax.sum(), 1.0)

    # Simplex projection
    coords_projection = coordize(input_vector, method=TransformMethod.SIMPLEX_PROJECTION)
    assert np.all(coords_projection >= 0)
    assert np.isclose(coords_projection.sum(), 1.0)

    # Exponential map (currently uses softmax)
    coords_exp = coordize(input_vector, method=TransformMethod.EXPONENTIAL_MAP)
    assert np.all(coords_exp >= 0)
    assert np.isclose(coords_exp.sum(), 1.0)


def test_coordize_batch():
    """Test batch transformation."""
    input_vectors = np.array([
        [0.5, -0.3, 0.8],
        [-0.2, 0.4, 0.1],
        [1.0, 2.0, 3.0],
    ])

    coords_list = coordize_batch(input_vectors)

    assert len(coords_list) == 3, "Should have 3 results"

    for coords in coords_list:
        assert np.all(coords >= 0), "All coordinates non-negative"
        assert np.isclose(coords.sum(), 1.0), "Sum to 1"


def test_coordize_invalid_input():
    """Test error handling for invalid input."""
    # Not a numpy array
    with pytest.raises(TypeError):
        coordize([1, 2, 3])

    # Empty array
    with pytest.raises(ValueError):
        coordize(np.array([]))

    # Unknown method
    with pytest.raises(ValueError):
        coordize(np.array([1, 2, 3]), method="invalid_method")


def test_softmax_transform():
    """Test softmax transformation directly."""
    input_vector = np.array([1.0, 2.0, 3.0])

    # With numerical stability
    coords_stable = _softmax_transform(input_vector, numerical_stability=True, epsilon=1e-10)
    assert np.all(coords_stable >= 0)
    assert np.isclose(coords_stable.sum(), 1.0)

    # Without numerical stability
    coords_unstable = _softmax_transform(input_vector, numerical_stability=False, epsilon=1e-10)
    assert np.all(coords_unstable >= 0)
    assert np.isclose(coords_unstable.sum(), 1.0)

    # Should be approximately equal for reasonable values
    assert np.allclose(coords_stable, coords_unstable, rtol=1e-5)


def test_simplex_projection():
    """Test simplex projection directly."""
    input_vector = np.array([0.5, -0.3, 0.8])
    coords = _simplex_projection(input_vector, epsilon=1e-10)

    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)


def test_coordize_deterministic():
    """Test that coordize is deterministic."""
    input_vector = np.array([0.5, -0.3, 0.8, -0.1])

    coords1 = coordize(input_vector)
    coords2 = coordize(input_vector)

    assert np.allclose(coords1, coords2), "Should produce identical results"


def test_coordize_monotonic():
    """Test that larger values get larger probabilities."""
    input_vector = np.array([1.0, 2.0, 3.0])
    coords = coordize(input_vector)

    # Largest input vector value should get largest probability
    assert coords[2] > coords[1] > coords[0], "Should preserve ordering"


def test_coordize_preserves_shape():
    """Test that output shape matches input shape."""
    shapes = [(3,), (5,), (10,), (64,)]

    for shape in shapes:
        input_vec = np.random.randn(*shape)
        coords = coordize(input_vec)
        assert coords.shape == shape, f"Shape not preserved for {shape}"
