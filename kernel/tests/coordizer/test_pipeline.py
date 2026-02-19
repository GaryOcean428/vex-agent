"""Tests for coordinate transformation pipeline."""

import numpy as np
import pytest

from kernel.coordizer.pipeline import CoordinatorPipeline
from kernel.coordizer.types import PipelineConfig, TransformMethod


def test_pipeline_single_transform():
    """Test pipeline single input vector transformation."""
    pipeline = CoordinatorPipeline()
    input_vector = np.array([0.5, -0.3, 0.8, -0.1])
    coords = pipeline.transform(input_vector)

    assert np.all(coords >= 0), "All coordinates non-negative"
    assert np.isclose(coords.sum(), 1.0), "Sum to 1"
    assert coords.shape == input_vector.shape, "Shape preserved"

    stats = pipeline.get_stats()
    assert stats.total_transforms == 1
    assert stats.successful_transforms == 1
    assert stats.success_rate == 1.0


def test_pipeline_batch_transform():
    """Test pipeline batch transformation."""
    pipeline = CoordinatorPipeline()
    input_vectors = np.array([
        [0.5, -0.3, 0.8],
        [-0.2, 0.4, 0.1],
        [1.0, 2.0, 3.0],
    ])
    coords_list = pipeline.transform_batch(input_vectors)

    assert len(coords_list) == 3
    for coords in coords_list:
        assert np.all(coords >= 0)
        assert np.isclose(coords.sum(), 1.0)

    stats = pipeline.get_stats()
    assert stats.total_transforms == 3
    assert stats.successful_transforms == 3


def test_pipeline_custom_config():
    """Test pipeline with custom configuration."""
    config = PipelineConfig(
        method=TransformMethod.SIMPLEX_PROJECTION,
        validation_mode="strict",
        auto_fix=True,
        batch_size=16,
        numerical_stability=True,
        epsilon=1e-12,
    )
    pipeline = CoordinatorPipeline(config)
    input_vector = np.array([0.5, -0.3, 0.8])
    coords = pipeline.transform(input_vector)

    assert np.all(coords >= 0)
    assert np.isclose(coords.sum(), 1.0)


def test_pipeline_stats_tracking():
    """Test pipeline statistics tracking and reset."""
    pipeline = CoordinatorPipeline()

    for _ in range(5):
        input_vec = np.random.randn(10)
        pipeline.transform(input_vec)

    stats = pipeline.get_stats()
    assert stats.total_transforms == 5
    assert stats.successful_transforms == 5
    assert stats.avg_transform_time > 0

    pipeline.reset_stats()
    stats = pipeline.get_stats()
    assert stats.total_transforms == 0
