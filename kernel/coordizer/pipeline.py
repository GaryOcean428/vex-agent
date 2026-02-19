"""Coordinate transformation pipeline orchestration.

Provides end-to-end pipeline for:
- Batch coordinate transformation
- Statistics tracking
- Error handling and recovery
- Configuration management
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .transform import coordize, coordize_batch
from .types import (
    CoordinateTransform,
    PipelineConfig,
    TransformMethod,
    TransformStats,
)
from .validate import validate_simplex


class CoordinatorPipeline:
    """End-to-end coordinate transformation pipeline.

    Manages:
    - Configuration
    - Statistics tracking
    - Batch processing
    - Error handling

    Example:
        >>> pipeline = CoordinatorPipeline()
        >>> embedding = np.array([0.5, -0.3, 0.8])
        >>> coords = pipeline.transform(embedding)
        >>> stats = pipeline.get_stats()
        >>> print(stats.total_transforms)
        1
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self._stats = TransformStats()
        self._transform_times: list[float] = []

    def transform(
        self,
        embedding: np.ndarray,
        method: TransformMethod | None = None,
        validate: bool = True,
    ) -> np.ndarray:
        """Transform single embedding to coordinates.

        Args:
            embedding: Euclidean embedding
            method: Transformation method (uses config default if None)
            validate: Whether to validate output

        Returns:
            Transformed coordinates

        Raises:
            ValueError: If transformation fails
        """
        start_time = time.time()

        try:
            # Use config method if not specified
            transform_method = method or self.config.method

            # Transform
            coordinates = coordize(
                embedding,
                method=transform_method,
                numerical_stability=self.config.numerical_stability,
                epsilon=self.config.epsilon,
            )

            # Validate if requested
            if validate:
                result = validate_simplex(
                    coordinates,
                    tolerance=self.config.epsilon * 10,
                    validation_mode=self.config.validation_mode,
                )

                if not result.valid:
                    raise ValueError(
                        f"Validation failed: {', '.join(result.errors)}"
                    )

                if result.has_warnings:
                    self._stats.total_warnings += len(result.warnings)

            # Record success
            elapsed = time.time() - start_time
            self._record_success(transform_method.value, elapsed)

            return coordinates

        except Exception as e:
            # Record failure
            elapsed = time.time() - start_time
            self._record_failure(elapsed)
            raise ValueError(f"Transformation failed: {e}") from e

    def transform_batch(
        self,
        embeddings: np.ndarray,
        method: TransformMethod | None = None,
        validate: bool = True,
    ) -> list[np.ndarray]:
        """Transform batch of embeddings to coordinates.

        Args:
            embeddings: Batch of embeddings, shape (batch_size, dim)
            method: Transformation method (uses config default if None)
            validate: Whether to validate outputs

        Returns:
            List of coordinate arrays

        Raises:
            ValueError: If any transformation fails
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D (batch_size, dim), "
                f"got shape {embeddings.shape}"
            )

        # Process in batches if needed
        batch_size = self.config.batch_size
        results = []

        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]

            for embedding in batch:
                coords = self.transform(embedding, method, validate)
                results.append(coords)

        return results

    def create_transform_record(
        self,
        embedding: np.ndarray,
        coordinates: np.ndarray,
        method: TransformMethod | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CoordinateTransform:
        """Create a transformation record.

        Args:
            embedding: Original embedding
            coordinates: Transformed coordinates
            method: Method used (uses config default if None)
            metadata: Optional metadata

        Returns:
            CoordinateTransform record
        """
        transform_method = method or self.config.method

        return CoordinateTransform(
            embedding=embedding,
            coordinates=coordinates,
            method=transform_method,
            timestamp=time.time(),
            metadata=metadata or {},
        )

    def get_stats(self) -> TransformStats:
        """Get transformation statistics.

        Returns:
            Current statistics
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = TransformStats()
        self._transform_times = []

    def _record_success(self, method: str, elapsed: float) -> None:
        """Record successful transformation.

        Args:
            method: Method used
            elapsed: Time taken (seconds)
        """
        self._stats.total_transforms += 1
        self._stats.successful_transforms += 1

        # Update method counts
        if method not in self._stats.method_counts:
            self._stats.method_counts[method] = 0
        self._stats.method_counts[method] += 1

        # Update average time
        self._transform_times.append(elapsed)
        self._stats.avg_transform_time = sum(self._transform_times) / len(
            self._transform_times
        )

    def _record_failure(self, elapsed: float) -> None:
        """Record failed transformation.

        Args:
            elapsed: Time taken (seconds)
        """
        self._stats.total_transforms += 1
        self._stats.failed_transforms += 1

        # Still update average time (includes failures)
        self._transform_times.append(elapsed)
        self._stats.avg_transform_time = sum(self._transform_times) / len(
            self._transform_times
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CoordinatorPipeline("
            f"method={self.config.method}, "
            f"transforms={self._stats.total_transforms}, "
            f"success_rate={self._stats.success_rate:.2%})"
        )
