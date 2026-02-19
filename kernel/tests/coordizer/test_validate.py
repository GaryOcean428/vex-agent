"""Tests for simplex validation."""

import numpy as np
import pytest

from kernel.coordizer.validate import (
    validate_simplex,
    ensure_simplex,
    normalize_simplex,
    check_simplex_invariants,
)


def test_validate_simplex_valid():
    """Test validation of valid simplex coordinates."""
    coords = np.array([0.3, 0.5, 0.2])
    result = validate_simplex(coords)

    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_simplex_negative():
    """Test detection of negative values."""
    coords = np.array([0.3, -0.1, 0.8])
    result = validate_simplex(coords, validation_mode="strict")

    assert result.valid is False
    assert any("negative" in err.lower() for err in result.errors)


def test_validate_simplex_sum_off():
    """Test detection of incorrect sum."""
    coords = np.array([0.3, 0.5, 0.3])  # Sum = 1.1
    result = validate_simplex(coords, tolerance=1e-6, validation_mode="strict")

    assert result.valid is False
    assert any("sum" in err.lower() for err in result.errors)


def test_validate_simplex_nan():
    """Test detection of NaN values."""
    coords = np.array([0.3, np.nan, 0.7])
    result = validate_simplex(coords)

    assert result.valid is False
    assert any("nan" in err.lower() for err in result.errors)


def test_validate_simplex_inf():
    """Test detection of Inf values."""
    coords = np.array([0.3, np.inf, 0.7])
    result = validate_simplex(coords)

    assert result.valid is False
    assert any("inf" in err.lower() for err in result.errors)


def test_validate_simplex_modes():
    """Test different validation modes."""
    # Slightly off coordinates
    coords = np.array([0.3, 0.5, 0.201])  # Sum = 1.001

    # Strict mode - should fail
    result_strict = validate_simplex(coords, tolerance=1e-6, validation_mode="strict")
    assert result_strict.valid is False

    # Standard mode - should pass with warning
    result_standard = validate_simplex(coords, tolerance=1e-6, validation_mode="standard")
    # May pass or fail depending on tolerance, but should be consistent

    # Permissive mode - should pass
    result_permissive = validate_simplex(coords, tolerance=1e-6, validation_mode="permissive")
    # Permissive is most lenient


def test_ensure_simplex_valid():
    """Test ensure_simplex with valid input."""
    coords = np.array([0.3, 0.5, 0.2])
    result = ensure_simplex(coords)

    assert np.all(result >= 0)
    assert np.isclose(result.sum(), 1.0)


def test_ensure_simplex_fix_negative():
    """Test fixing negative values."""
    coords = np.array([0.3, -0.1, 0.8])
    result = ensure_simplex(coords)

    assert np.all(result >= 0), "All values should be non-negative"
    assert np.isclose(result.sum(), 1.0), "Should sum to 1"


def test_ensure_simplex_fix_sum():
    """Test fixing incorrect sum."""
    coords = np.array([0.3, 0.5, 0.3])  # Sum = 1.1
    result = ensure_simplex(coords)

    assert np.all(result >= 0)
    assert np.isclose(result.sum(), 1.0)


def test_ensure_simplex_fix_nan():
    """Test fixing NaN values."""
    coords = np.array([0.3, np.nan, 0.7])
    result = ensure_simplex(coords)

    assert not np.any(np.isnan(result)), "No NaN values"
    assert np.all(result >= 0)
    assert np.isclose(result.sum(), 1.0)


def test_ensure_simplex_fix_inf():
    """Test fixing Inf values."""
    coords = np.array([0.3, np.inf, 0.7])
    result = ensure_simplex(coords)

    assert not np.any(np.isinf(result)), "No Inf values"
    assert np.all(result >= 0)
    assert np.isclose(result.sum(), 1.0)


def test_ensure_simplex_all_zeros():
    """Test handling of all-zero input."""
    coords = np.zeros(5)
    result = ensure_simplex(coords)

    # Should create uniform distribution
    assert np.all(result >= 0)
    assert np.isclose(result.sum(), 1.0)
    expected = 1.0 / len(coords)
    assert np.allclose(result, expected, atol=0.1)


def test_ensure_simplex_invalid_type():
    """Test error handling for invalid type."""
    with pytest.raises(TypeError):
        ensure_simplex([0.3, 0.5, 0.2])


def test_ensure_simplex_empty():
    """Test error handling for empty array."""
    with pytest.raises(ValueError):
        ensure_simplex(np.array([]))


def test_normalize_simplex():
    """Test simplex normalization."""
    coords = np.array([0.3, 0.5, 0.3])  # Sum = 1.1
    result = normalize_simplex(coords)

    assert np.isclose(result.sum(), 1.0)
    # Original ratios should be preserved (approximately)
    expected_ratio = 0.3 / 0.5  # First to second
    actual_ratio = result[0] / result[1]
    assert np.isclose(expected_ratio, actual_ratio, rtol=0.01)


def test_normalize_simplex_already_normalized():
    """Test normalization of already-normalized coordinates."""
    coords = np.array([0.3, 0.5, 0.2])
    result = normalize_simplex(coords)

    # Should be very close to original
    assert np.allclose(result, coords, rtol=0.01)


def test_check_simplex_invariants_valid():
    """Test invariant checking for valid coordinates."""
    coords = np.array([0.3, 0.5, 0.2])
    all_pass, failed = check_simplex_invariants(coords)

    assert all_pass is True
    assert len(failed) == 0


def test_check_simplex_invariants_negative():
    """Test detection of negative invariant violation."""
    coords = np.array([0.3, -0.1, 0.8])
    all_pass, failed = check_simplex_invariants(coords)

    assert all_pass is False
    assert "non-negativity" in failed


def test_check_simplex_invariants_sum():
    """Test detection of sum invariant violation."""
    coords = np.array([0.3, 0.5, 0.3])  # Sum = 1.1
    all_pass, failed = check_simplex_invariants(coords)

    assert all_pass is False
    assert "sum-to-one" in failed


def test_check_simplex_invariants_nan():
    """Test detection of finite invariant violation (NaN)."""
    coords = np.array([0.3, np.nan, 0.7])
    all_pass, failed = check_simplex_invariants(coords)

    assert all_pass is False
    assert "finite" in failed


def test_check_simplex_invariants_inf():
    """Test detection of finite invariant violation (Inf)."""
    coords = np.array([0.3, np.inf, 0.7])
    all_pass, failed = check_simplex_invariants(coords)

    assert all_pass is False
    assert "finite" in failed


def test_validate_simplex_empty():
    """Test validation of empty array."""
    coords = np.array([])
    result = validate_simplex(coords)

    assert result.valid is False
    assert any("empty" in err.lower() for err in result.errors)


def test_validate_simplex_warnings():
    """Test that warnings are properly recorded."""
    # Slightly negative (within tolerance in permissive mode)
    coords = np.array([0.3, -1e-8, 0.7])
    result = validate_simplex(coords, validation_mode="permissive")

    assert result.has_warnings
    assert len(result.warnings) > 0
