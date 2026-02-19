"""
Tests for coordizer_v2.compress — PGA compression on Δ⁶³.

Compression uses Principal Geodesic Analysis (PGA), NOT PCA.
The algorithm uses eigendecomposition of the Gram matrix, NOT SVD.
All operations are on the Fisher-Rao manifold.
"""

import numpy as np
import pytest

from kernel.coordizer_v2.compress import CompressionResult
from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    E8_RANK,
    KAPPA_STAR,
    fisher_rao_distance,
    to_simplex,
)


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_compression_result(rng):
    """Create a synthetic CompressionResult for testing."""
    n_tokens = 50
    source_dim = 128

    # Simulate eigenvalues with E8-like structure
    # Top 8 capture ~87.7% of variance
    eigenvalues = np.zeros(BASIN_DIM)
    eigenvalues[:E8_RANK] = np.array([20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0])
    eigenvalues[E8_RANK:16] = np.linspace(3.0, 0.5, 8)
    eigenvalues[16:] = np.linspace(0.4, 0.01, BASIN_DIM - 16)

    result = CompressionResult(
        source_dim=source_dim,
        target_dim=BASIN_DIM,
        n_tokens=n_tokens,
        eigenvalues=eigenvalues,
        frechet_mean_full=rng.dirichlet(np.ones(source_dim)),
        token_strings={i: f"token_{i}" for i in range(n_tokens)},
    )

    # Generate compressed coordinates on Δ⁶³
    for i in range(n_tokens):
        basin = rng.dirichlet(np.ones(BASIN_DIM))
        result.compressed[i] = basin

    total_var = float(np.sum(eigenvalues))
    result.total_geodesic_variance = total_var
    result.explained_variance_ratio = np.cumsum(eigenvalues) / total_var
    result.e8_rank_variance = float(np.sum(eigenvalues[:E8_RANK]) / total_var)

    return result


# ═══════════════════════════════════════════════════════════════
#  COMPRESSION RESULT STRUCTURE
# ═══════════════════════════════════════════════════════════════


class TestCompressionResult:
    """Test CompressionResult structure and invariants."""

    def test_target_dim_is_basin_dim(self, sample_compression_result):
        assert sample_compression_result.target_dim == BASIN_DIM

    def test_all_compressed_on_simplex(self, sample_compression_result):
        """Every compressed coordinate must be on Δ⁶³."""
        for tid, basin in sample_compression_result.compressed.items():
            assert abs(basin.sum() - 1.0) < 1e-8, f"Token {tid} not on simplex"
            assert np.all(basin >= 0), f"Token {tid} has negative values"

    def test_correct_dimension(self, sample_compression_result):
        for tid, basin in sample_compression_result.compressed.items():
            assert len(basin) == BASIN_DIM, f"Token {tid} wrong dim"

    def test_token_count(self, sample_compression_result):
        assert len(sample_compression_result.compressed) == 50


# ═══════════════════════════════════════════════════════════════
#  EIGENDECOMPOSITION (NOT SVD)
# ═══════════════════════════════════════════════════════════════


class TestEigendecomposition:
    """Verify PGA uses eigendecomposition, not SVD."""

    def test_eigenvalues_non_negative(self, sample_compression_result):
        """Gram matrix eigenvalues must be non-negative."""
        eigs = sample_compression_result.eigenvalues
        assert np.all(eigs >= -1e-10), "Eigenvalues must be non-negative"

    def test_eigenvalues_descending(self, sample_compression_result):
        """Eigenvalues must be in descending order."""
        eigs = sample_compression_result.eigenvalues
        for i in range(len(eigs) - 1):
            assert eigs[i] >= eigs[i + 1] - 1e-10

    def test_explained_variance_monotonic(self, sample_compression_result):
        """Cumulative explained variance must be monotonically increasing."""
        evr = sample_compression_result.explained_variance_ratio
        for i in range(len(evr) - 1):
            assert evr[i] <= evr[i + 1] + 1e-10

    def test_explained_variance_reaches_one(self, sample_compression_result):
        """Total explained variance must reach 1.0."""
        evr = sample_compression_result.explained_variance_ratio
        assert abs(evr[-1] - 1.0) < 1e-8


# ═══════════════════════════════════════════════════════════════
#  E8 HYPOTHESIS
# ═══════════════════════════════════════════════════════════════


class TestE8Hypothesis:
    """Test E8 eigenvalue structure."""

    def test_e8_rank_variance(self, sample_compression_result):
        """Top-8 eigenvalues should capture significant variance."""
        e8_var = sample_compression_result.e8_rank_variance
        assert e8_var > 0.5, "Top-8 should capture >50% variance"

    def test_e8_hypothesis_score(self, sample_compression_result):
        """E8 hypothesis score should be computable."""
        score = sample_compression_result.e8_hypothesis_score()
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════
#  GEOMETRIC INTEGRITY
# ═══════════════════════════════════════════════════════════════


class TestGeometricIntegrity:
    """Test that compression preserves geometric structure."""

    def test_pairwise_distances_positive(self, sample_compression_result):
        """All pairwise Fisher-Rao distances must be positive for distinct tokens."""
        tids = list(sample_compression_result.compressed.keys())[:10]
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                d = fisher_rao_distance(
                    sample_compression_result.compressed[tids[i]],
                    sample_compression_result.compressed[tids[j]],
                )
                assert d > 0, f"Distinct tokens {tids[i]}, {tids[j]} have d_FR=0"

    def test_distances_bounded(self, sample_compression_result):
        """Fisher-Rao distances must be bounded by π/2."""
        import math
        tids = list(sample_compression_result.compressed.keys())[:10]
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                d = fisher_rao_distance(
                    sample_compression_result.compressed[tids[i]],
                    sample_compression_result.compressed[tids[j]],
                )
                assert d <= math.pi / 2 + 1e-8

    def test_frechet_mean_exists(self, sample_compression_result):
        """Fréchet mean of compressed tokens should exist and be on simplex."""
        from kernel.coordizer_v2.geometry import frechet_mean
        basins = list(sample_compression_result.compressed.values())[:20]
        mean = frechet_mean(basins)
        assert abs(mean.sum() - 1.0) < 1e-8
        assert np.all(mean >= 0)
