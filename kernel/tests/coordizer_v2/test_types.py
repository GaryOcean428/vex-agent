"""
Tests for coordizer_v2.types — Basin coordinates and type structures.

All distance checks use Fisher-Rao. No Euclidean contamination.
"""

import numpy as np
import pytest

from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    fisher_rao_distance,
)
from kernel.coordizer_v2.types import (
    BasinCoordinate,
    CoordizationResult,
    DomainBias,
    GranularityScale,
    HarmonicTier,
    ValidationResult,
)

# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_coord(rng):
    return BasinCoordinate(
        coord_id=1,
        vector=rng.dirichlet(np.ones(BASIN_DIM)),
        name="test_token",
        tier=HarmonicTier.FUNDAMENTAL,
        frequency=440.0,
        basin_mass=1.0,
    )


@pytest.fixture
def sample_coord_b(rng):
    rng2 = np.random.RandomState(99)
    return BasinCoordinate(
        coord_id=2,
        vector=rng2.dirichlet(np.ones(BASIN_DIM)),
        name="other_token",
        tier=HarmonicTier.FIRST_HARMONIC,
        frequency=660.0,
        basin_mass=0.8,
    )


# ═══════════════════════════════════════════════════════════════
#  HARMONIC TIER
# ═══════════════════════════════════════════════════════════════


class TestHarmonicTier:
    """Test harmonic tier enum."""

    def test_all_tiers_exist(self):
        expected = {"fundamental", "first", "upper", "overtone"}
        actual = {t.value for t in HarmonicTier}
        assert actual == expected

    def test_string_enum(self):
        assert HarmonicTier.FUNDAMENTAL == "fundamental"
        assert isinstance(HarmonicTier.FUNDAMENTAL, str)


class TestGranularityScale:
    """Test granularity scale enum."""

    def test_all_scales_exist(self):
        expected = {"byte", "char", "subword", "word", "phrase", "concept"}
        actual = {s.value for s in GranularityScale}
        assert actual == expected


# ═══════════════════════════════════════════════════════════════
#  BASIN COORDINATE
# ═══════════════════════════════════════════════════════════════


class TestBasinCoordinate:
    """Test BasinCoordinate creation and operations."""

    def test_creation(self, sample_coord):
        assert sample_coord.coord_id == 1
        assert sample_coord.name == "test_token"
        assert sample_coord.tier == HarmonicTier.FUNDAMENTAL
        assert sample_coord.frequency == 440.0

    def test_vector_on_simplex(self, sample_coord):
        """Vector must be on the probability simplex."""
        v = sample_coord.vector
        assert abs(v.sum() - 1.0) < 1e-10
        assert np.all(v >= 0)

    def test_vector_dimension(self, sample_coord):
        assert len(sample_coord.vector) == BASIN_DIM

    def test_distance_to_uses_fisher_rao(self, sample_coord, sample_coord_b):
        """distance_to must use Fisher-Rao, not Euclidean."""
        d_coord = sample_coord.distance_to(sample_coord_b)
        d_fr = fisher_rao_distance(sample_coord.vector, sample_coord_b.vector)
        assert abs(d_coord - d_fr) < 1e-10

    def test_distance_to_self_is_zero(self, sample_coord):
        d = sample_coord.distance_to(sample_coord)
        assert d < 1e-10

    def test_distance_symmetric(self, sample_coord, sample_coord_b):
        d_ab = sample_coord.distance_to(sample_coord_b)
        d_ba = sample_coord_b.distance_to(sample_coord)
        assert abs(d_ab - d_ba) < 1e-12

    def test_midpoint_with_on_simplex(self, sample_coord, sample_coord_b):
        mid = sample_coord.midpoint_with(sample_coord_b)
        assert abs(mid.sum() - 1.0) < 1e-10
        assert np.all(mid >= 0)

    def test_midpoint_equidistant(self, sample_coord, sample_coord_b):
        """Midpoint must be equidistant from both endpoints (Fisher-Rao)."""
        mid = sample_coord.midpoint_with(sample_coord_b)
        d_a = fisher_rao_distance(sample_coord.vector, mid)
        d_b = fisher_rao_distance(sample_coord_b.vector, mid)
        assert abs(d_a - d_b) < 1e-6


# ═══════════════════════════════════════════════════════════════
#  COORDIZATION RESULT
# ═══════════════════════════════════════════════════════════════


class TestCoordizationResult:
    """Test CoordizationResult metric computation."""

    def _make_result(self, n_coords: int, rng) -> CoordizationResult:
        coords = []
        for i in range(n_coords):
            coords.append(
                BasinCoordinate(
                    coord_id=i,
                    vector=rng.dirichlet(np.ones(BASIN_DIM)),
                    name=f"token_{i}",
                    tier=HarmonicTier.FUNDAMENTAL,
                    frequency=440.0 * (i + 1) / n_coords,
                    basin_mass=1.0,
                )
            )
        return CoordizationResult(
            coordinates=coords,
            coord_ids=[c.coord_id for c in coords],
            original_text="test text",
        )

    def test_compute_metrics_populates_all(self, rng):
        result = self._make_result(5, rng)
        result.compute_metrics()
        assert result.basin_velocity is not None
        assert result.trajectory_curvature is not None
        assert result.harmonic_consonance is not None

    def test_basin_velocity_non_negative(self, rng):
        result = self._make_result(5, rng)
        result.compute_metrics()
        assert result.basin_velocity >= 0.0

    def test_basin_velocity_uses_fisher_rao(self, rng):
        """Basin velocity must be average Fisher-Rao distance."""
        result = self._make_result(3, rng)
        result.compute_metrics()
        # Manually compute
        coords = result.coordinates
        expected = sum(
            fisher_rao_distance(coords[i].vector, coords[i + 1].vector)
            for i in range(len(coords) - 1)
        ) / (len(coords) - 1)
        assert abs(result.basin_velocity - expected) < 1e-10

    def test_trajectory_curvature_range(self, rng):
        result = self._make_result(5, rng)
        result.compute_metrics()
        assert 0.0 <= result.trajectory_curvature <= 1.0

    def test_harmonic_consonance_positive(self, rng):
        result = self._make_result(5, rng)
        result.compute_metrics()
        assert result.harmonic_consonance > 0.0

    def test_single_coord_zero_velocity(self, rng):
        result = self._make_result(1, rng)
        result.compute_metrics()
        assert result.basin_velocity == 0.0
        assert result.trajectory_curvature == 0.0


# ═══════════════════════════════════════════════════════════════
#  DOMAIN BIAS
# ═══════════════════════════════════════════════════════════════


class TestDomainBias:
    """Test DomainBias structure."""

    def test_default_anchor_is_uniform(self):
        bias = DomainBias(domain_name="test")
        assert abs(bias.anchor_basin.sum() - 1.0) < 1e-10
        assert np.allclose(bias.anchor_basin, 1.0 / BASIN_DIM)

    def test_strength_range(self):
        bias = DomainBias(domain_name="test", strength=0.5)
        assert 0.0 <= bias.strength <= 1.0


# ═══════════════════════════════════════════════════════════════
#  VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════


class TestValidationResult:
    """Test ValidationResult structure."""

    def test_default_not_passed(self):
        result = ValidationResult()
        assert result.passed is False

    def test_summary_contains_kappa(self):
        result = ValidationResult(kappa_measured=64.0, kappa_std=1.0)
        s = result.summary()
        assert "64.00" in s

    def test_summary_pass_fail(self):
        result = ValidationResult(passed=True)
        assert "PASS" in result.summary()
        result.passed = False
        assert "FAIL" in result.summary()
