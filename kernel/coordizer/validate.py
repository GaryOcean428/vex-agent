"""Simplex validation and normalization utilities.

Validate that coordinates satisfy probability simplex properties:
- All values non-negative (coords[i] >= 0)
- Sum to one (sum(coords) = 1.0)
- Finite values (no NaN, no Inf)
"""

from __future__ import annotations

import numpy as np

from .types import ValidationResult


def validate_simplex(
    coordinates: np.ndarray,
    tolerance: float = 1e-6,
    validation_mode: str = "standard",
) -> ValidationResult:
    """Validate simplex properties of coordinates.

    Checks:
    1. Non-negativity: all coords[i] >= 0
    2. Sum to one: |sum(coords) - 1.0| < tolerance
    3. Finite: no NaN, no Inf
    4. Non-empty: coordinates not empty

    Args:
        coordinates: Coordinate array to validate
        tolerance: Tolerance for sum-to-one check (default 1e-6)
        validation_mode: Validation strictness (strict, standard, permissive)

    Returns:
        ValidationResult with validation status, errors, and warnings

    Example:
        >>> coords = np.array([0.3, 0.5, 0.2])
        >>> result = validate_simplex(coords)
        >>> print(result.valid)
        True
        >>> print(result.errors)
        []
    """
    errors = []
    warnings = []

    # Check if array is valid
    if not isinstance(coordinates, np.ndarray):
        errors.append(f"coordinates must be np.ndarray, got {type(coordinates)}")
        return ValidationResult(
            valid=False, errors=errors, warnings=warnings, coordinates=None
        )

    if coordinates.size == 0:
        errors.append("coordinates must not be empty")
        return ValidationResult(
            valid=False, errors=errors, warnings=warnings, coordinates=None
        )

    # Check for NaN and Inf
    if np.any(np.isnan(coordinates)):
        errors.append("coordinates contain NaN values")

    if np.any(np.isinf(coordinates)):
        errors.append("coordinates contain Inf values")

    # Check non-negativity
    if np.any(coordinates < 0):
        negative_count = np.sum(coordinates < 0)
        min_value = np.min(coordinates)
        if validation_mode == "strict":
            errors.append(
                f"coordinates must be non-negative: "
                f"{negative_count} negative values, min={min_value:.6f}"
            )
        elif validation_mode == "standard":
            if min_value < -tolerance:
                errors.append(
                    f"coordinates significantly negative: "
                    f"min={min_value:.6f} (tolerance={tolerance})"
                )
            else:
                warnings.append(
                    f"coordinates slightly negative: "
                    f"min={min_value:.6f} (within tolerance)"
                )
        # permissive mode: only warn
        elif validation_mode == "permissive":
            warnings.append(f"coordinates negative: min={min_value:.6f}")

    # Check sum to one
    coord_sum = np.sum(coordinates)
    sum_error = abs(coord_sum - 1.0)

    if validation_mode == "strict":
        if sum_error > tolerance:
            errors.append(
                f"coordinates must sum to 1.0: "
                f"sum={coord_sum:.6f}, error={sum_error:.6e}"
            )
    elif validation_mode == "standard":
        if sum_error > tolerance * 10:  # More lenient
            errors.append(
                f"coordinates sum significantly off: "
                f"sum={coord_sum:.6f}, error={sum_error:.6e}"
            )
        elif sum_error > tolerance:
            warnings.append(
                f"coordinates sum slightly off: "
                f"sum={coord_sum:.6f}, error={sum_error:.6e}"
            )
    # permissive mode: only warn for large errors
    elif validation_mode == "permissive":
        if sum_error > 0.1:  # Very lenient
            warnings.append(f"coordinates sum off: sum={coord_sum:.6f}")

    valid = len(errors) == 0

    return ValidationResult(
        valid=valid, errors=errors, warnings=warnings, coordinates=coordinates
    )


def ensure_simplex(
    coordinates: np.ndarray,
    epsilon: float = 1e-10,
    max_attempts: int = 3,
) -> np.ndarray:
    """Ensure coordinates satisfy simplex properties, fixing if needed.

    This function is fail-closed: if coordinates cannot be fixed,
    it raises an exception rather than returning invalid coordinates.

    Fixes applied (in order):
    1. Replace NaN/Inf with epsilon
    2. Clip negative values to 0
    3. Re-normalize to sum to 1

    Args:
        coordinates: Coordinate array to fix
        epsilon: Small positive value for stability
        max_attempts: Maximum fix attempts before failing

    Returns:
        Fixed coordinates on simplex

    Raises:
        ValueError: If coordinates cannot be fixed after max_attempts

    Example:
        >>> coords = np.array([0.3, 0.5, 0.21])  # Sum = 1.01 (slightly off)
        >>> fixed = ensure_simplex(coords)
        >>> print(np.isclose(fixed.sum(), 1.0))
        True
    """
    if not isinstance(coordinates, np.ndarray):
        raise TypeError(f"coordinates must be np.ndarray, got {type(coordinates)}")

    if coordinates.size == 0:
        raise ValueError("coordinates must not be empty")

    coords = coordinates.copy()

    for attempt in range(max_attempts):
        # Fix NaN and Inf
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            coords = np.where(np.isnan(coords) | np.isinf(coords), epsilon, coords)

        # Fix negative values
        if np.any(coords < 0):
            coords = np.maximum(coords, 0)

        # Add epsilon for stability
        coords = coords + epsilon

        # Normalize to sum to 1
        coord_sum = np.sum(coords)
        if coord_sum > 0:
            coords = coords / coord_sum
        else:
            # All zeros, create uniform distribution
            coords = np.ones_like(coords) / coords.size

        # Validate result
        result = validate_simplex(coords, tolerance=1e-6, validation_mode="standard")

        if result.valid:
            return coords

        # If still invalid, try again with more aggressive fixes
        coords = coordinates.copy()

    # Failed to fix after max_attempts
    raise ValueError(
        f"Could not fix coordinates to satisfy simplex properties "
        f"after {max_attempts} attempts. "
        f"Errors: {result.errors}"
    )


def normalize_simplex(coordinates: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Normalize coordinates to sum to exactly 1.0.

    Simple normalization: coords / sum(coords)

    Args:
        coordinates: Coordinate array
        epsilon: Small value for stability

    Returns:
        Normalized coordinates

    Raises:
        ValueError: If coordinates sum to zero (after adding epsilon)
    """
    if not isinstance(coordinates, np.ndarray):
        raise TypeError(f"coordinates must be np.ndarray, got {type(coordinates)}")

    coords = coordinates.copy()

    # Add epsilon for stability
    coords = coords + epsilon

    # Normalize
    coord_sum = np.sum(coords)

    if coord_sum <= 0:
        raise ValueError(f"Cannot normalize: coordinates sum to {coord_sum}")

    return coords / coord_sum


def check_simplex_invariants(coordinates: np.ndarray) -> tuple[bool, list[str]]:
    """Quick check of simplex invariants (non-negative, sum to 1).

    Args:
        coordinates: Coordinate array

    Returns:
        Tuple of (all_pass, failed_invariants)

    Example:
        >>> coords = np.array([0.3, 0.5, 0.2])
        >>> all_pass, failed = check_simplex_invariants(coords)
        >>> print(all_pass)
        True
    """
    failed = []

    # Non-negativity
    if np.any(coordinates < 0):
        failed.append("non-negativity")

    # Sum to one
    if not np.isclose(np.sum(coordinates), 1.0):
        failed.append("sum-to-one")

    # Finite
    if np.any(~np.isfinite(coordinates)):
        failed.append("finite")

    return len(failed) == 0, failed
