"""Coordizer type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class TransformMethod(StrEnum):
    """Coordinate transformation methods."""

    SOFTMAX = "softmax"  # Exponential normalization (default)
    SIMPLEX_PROJECTION = "simplex_projection"  # Direct projection to simplex
    EXPONENTIAL_MAP = "exponential_map"  # Exponential map on Fisher-Rao manifold


@dataclass
class CoordinateTransform:
    """Result of coordinate transformation.

    Attributes:
        input_vector: Input Euclidean vector
        coordinates: Output Fisher-Rao coordinates (on probability simplex)
        method: Transformation method used
        timestamp: When transformation occurred
        metadata: Additional transformation metadata
    """

    input_vector: np.ndarray
    coordinates: np.ndarray
    method: TransformMethod
    timestamp: float
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate transform properties."""
        if self.input_vector.shape != self.coordinates.shape:
            raise ValueError(
                f"Shape mismatch: input_vector {self.input_vector.shape} "
                f"vs coordinates {self.coordinates.shape}"
            )


@dataclass
class ValidationResult:
    """Result of simplex validation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors (empty if valid)
        warnings: List of non-fatal warnings
        coordinates: The validated coordinates (normalized if fixed)
    """

    valid: bool
    errors: list[str]
    warnings: list[str]
    coordinates: np.ndarray | None = None

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


@dataclass
class PipelineConfig:
    """Configuration for coordinate transformation pipeline.

    Attributes:
        method: Default transformation method
        validation_mode: Validation strictness (strict, standard, permissive)
        auto_fix: Whether to auto-fix minor validation errors
        batch_size: Maximum batch size for batch processing
        numerical_stability: Use log-sum-exp trick for stability
        epsilon: Small value for numerical stability (default 1e-10)
    """

    method: TransformMethod = TransformMethod.SOFTMAX
    validation_mode: str = "standard"  # strict | standard | permissive
    auto_fix: bool = True
    batch_size: int = 32
    numerical_stability: bool = True
    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        """Validate config."""
        if self.validation_mode not in ("strict", "standard", "permissive"):
            raise ValueError(
                f"Invalid validation_mode: {self.validation_mode}, "
                "must be 'strict', 'standard', or 'permissive'"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")


@dataclass
class HarvestConfig:
    """Configuration for coordinate harvesting from conversations.

    Attributes:
        enabled: Whether harvesting is enabled
        sampling_rate: Fraction of messages to harvest (0.0-1.0)
        min_quality_score: Minimum quality score to accept (0.0-1.0)
        batch_size: Batch size for processing
        max_tokens: Maximum tokens per message to process
    """

    enabled: bool = True
    sampling_rate: float = 0.1  # Harvest 10% of messages
    min_quality_score: float = 0.5
    batch_size: int = 16
    max_tokens: int = 512

    def __post_init__(self) -> None:
        """Validate config."""
        if not 0.0 <= self.sampling_rate <= 1.0:
            raise ValueError(
                f"sampling_rate must be in [0, 1], got {self.sampling_rate}"
            )
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError(
                f"min_quality_score must be in [0, 1], "
                f"got {self.min_quality_score}"
            )


@dataclass
class TransformStats:
    """Statistics for coordinate transformations.

    Attributes:
        total_transforms: Total number of transformations
        successful_transforms: Number of successful transformations
        failed_transforms: Number of failed transformations
        avg_transform_time: Average transformation time (seconds)
        total_warnings: Total number of warnings
        method_counts: Count of transformations by method
    """

    total_transforms: int = 0
    successful_transforms: int = 0
    failed_transforms: int = 0
    avg_transform_time: float = 0.0
    total_warnings: int = 0
    method_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Initialize method counts."""
        if self.method_counts is None:
            self.method_counts = {}

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_transforms == 0:
            return 0.0
        return self.successful_transforms / self.total_transforms

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_transforms == 0:
            return 0.0
        return self.failed_transforms / self.total_transforms
