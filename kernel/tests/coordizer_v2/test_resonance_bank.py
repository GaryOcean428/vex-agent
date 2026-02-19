"""
Tests for coordizer_v2.resonance_bank — Standing waves on Δ⁶³.

A resonance bank is a manifold of standing waves, NOT a lookup table.
All activation and generation use Fisher-Rao proximity.
"""

import math
import tempfile

import numpy as np
import pytest

from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    KAPPA_STAR,
    fisher_rao_distance,
    frechet_mean,
    random_basin,
    slerp,
    to_simplex,
)
from kernel.coordizer_v2.resonance_bank import ResonanceBank
from kernel.coordizer_v2.types import DomainBias, HarmonicTier


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_bank(rng) -> ResonanceBank:
    """Create a small resonance bank with 20 resonators."""
    bank = ResonanceBank(target_dim=BASIN_DIM)
    for i in range(20):
        basin = rng.dirichlet(np.ones(BASIN_DIM))
        bank.coordinates[i] = to_simplex(basin)
        bank.token_strings[i] = f"token_{i}"
        bank.tiers[i] = HarmonicTier.FUNDAMENTAL if i < 5 else HarmonicTier.FIRST_HARMONIC
        bank.frequencies[i] = 100.0 + i * 50.0
        bank.basin_mass[i] = 1.0
        bank.activation_counts[i] = 0
    bank._rebuild_matrix()
    return bank


# ═══════════════════════════════════════════════════════════════
#  BANK CREATION
# ═══════════════════════════════════════════════════════════════


class TestBankCreation:
    """Test resonance bank initialisation."""

    def test_bank_size(self, small_bank):
        assert len(small_bank) == 20

    def test_all_coordinates_on_simplex(self, small_bank):
        for tid, basin in small_bank.coordinates.items():
            assert abs(basin.sum() - 1.0) < 1e-10, f"Token {tid} not on simplex"
            assert np.all(basin >= 0), f"Token {tid} has negative values"

    def test_correct_dimension(self, small_bank):
        for tid, basin in small_bank.coordinates.items():
            assert len(basin) == BASIN_DIM

    def test_contains(self, small_bank):
        assert 0 in small_bank
        assert 19 in small_bank
        assert 999 not in small_bank


# ═══════════════════════════════════════════════════════════════
#  STANDING WAVE PROPERTIES
# ═══════════════════════════════════════════════════════════════


class TestStandingWaves:
    """Test standing wave properties of resonators."""

    def test_each_resonator_has_frequency(self, small_bank):
        for tid in small_bank.coordinates:
            assert tid in small_bank.frequencies
            assert small_bank.frequencies[tid] > 0

    def test_each_resonator_has_tier(self, small_bank):
        for tid in small_bank.coordinates:
            assert tid in small_bank.tiers
            assert isinstance(small_bank.tiers[tid], HarmonicTier)

    def test_tier_distribution(self, small_bank):
        dist = small_bank.tier_distribution()
        total = sum(dist.values())
        assert total == len(small_bank)
        assert dist["fundamental"] == 5
        assert dist["first"] == 15  # HarmonicTier.FIRST_HARMONIC.value == "first"


# ═══════════════════════════════════════════════════════════════
#  ACTIVATION (FISHER-RAO PROXIMITY)
# ═══════════════════════════════════════════════════════════════


class TestActivation:
    """Test activation by Fisher-Rao proximity."""

    def test_activate_returns_sorted_by_distance(self, small_bank):
        query = random_basin()
        results = small_bank.activate(query, top_k=10)
        distances = [d for _, d in results]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-10

    def test_activate_self_distance_zero(self, small_bank):
        """Activating with an exact coordinate should return d_FR ≈ 0."""
        exact = small_bank.coordinates[0].copy()
        results = small_bank.activate(exact, top_k=5)
        # First result should be token 0 with distance ≈ 0
        assert results[0][0] == 0
        assert results[0][1] < 1e-6

    def test_activate_uses_fisher_rao(self, small_bank):
        """Verify activation distances match Fisher-Rao computation."""
        query = random_basin()
        results = small_bank.activate(query, top_k=5)
        for tid, dist in results:
            expected_d = fisher_rao_distance(query, small_bank.coordinates[tid])
            assert abs(dist - expected_d) < 1e-8

    def test_activate_ids_matches(self, small_bank):
        query = random_basin()
        results = small_bank.activate(query, top_k=5)
        ids = small_bank.activate_ids(query, top_k=5)
        assert ids == [tid for tid, _ in results]

    def test_activate_top_k_limit(self, small_bank):
        query = random_basin()
        results = small_bank.activate(query, top_k=3)
        assert len(results) == 3


# ═══════════════════════════════════════════════════════════════
#  GEODESIC INTERPOLATION
# ═══════════════════════════════════════════════════════════════


class TestGeodesicInterpolation:
    """Test geodesic operations on bank coordinates."""

    def test_midpoint_between_resonators(self, small_bank):
        """Geodesic midpoint between two resonators is equidistant."""
        a = small_bank.coordinates[0]
        b = small_bank.coordinates[1]
        mid = slerp(a, b, 0.5)
        d_a = fisher_rao_distance(a, mid)
        d_b = fisher_rao_distance(b, mid)
        assert abs(d_a - d_b) < 1e-6

    def test_interpolation_stays_on_simplex(self, small_bank):
        a = small_bank.coordinates[0]
        b = small_bank.coordinates[1]
        for t in np.linspace(0, 1, 11):
            pt = slerp(a, b, t)
            assert abs(pt.sum() - 1.0) < 1e-10
            assert np.all(pt >= 0)

    def test_interpolation_monotonic_distance(self, small_bank):
        """Distance from start increases monotonically along geodesic."""
        a = small_bank.coordinates[0]
        b = small_bank.coordinates[1]
        distances = []
        for t in np.linspace(0, 1, 11):
            pt = slerp(a, b, t)
            distances.append(fisher_rao_distance(a, pt))
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-8


# ═══════════════════════════════════════════════════════════════
#  GENERATION (GEODESIC FORESIGHT)
# ═══════════════════════════════════════════════════════════════


class TestGeneration:
    """Test token generation via geodesic foresight."""

    def test_generate_from_empty_trajectory(self, small_bank):
        tid, basin = small_bank.generate_next([])
        assert tid in small_bank.coordinates
        assert abs(basin.sum() - 1.0) < 1e-10

    def test_generate_from_single_point(self, small_bank):
        trajectory = [small_bank.coordinates[0]]
        tid, basin = small_bank.generate_next(trajectory)
        assert tid in small_bank.coordinates
        assert abs(basin.sum() - 1.0) < 1e-10

    def test_generate_from_trajectory(self, small_bank):
        trajectory = [
            small_bank.coordinates[0],
            small_bank.coordinates[1],
            small_bank.coordinates[2],
        ]
        tid, basin = small_bank.generate_next(trajectory)
        assert tid in small_bank.coordinates

    def test_generate_result_on_simplex(self, small_bank):
        trajectory = [small_bank.coordinates[i] for i in range(3)]
        _, basin = small_bank.generate_next(trajectory)
        assert abs(basin.sum() - 1.0) < 1e-10
        assert np.all(basin >= 0)

    def test_temperature_zero_deterministic(self, small_bank):
        """Temperature=0 should give deterministic output."""
        trajectory = [small_bank.coordinates[0]]
        results = set()
        for _ in range(5):
            tid, _ = small_bank.generate_next(trajectory, temperature=0.0)
            results.add(tid)
        assert len(results) == 1, "Temperature=0 must be deterministic"


# ═══════════════════════════════════════════════════════════════
#  MEAN BASIN
# ═══════════════════════════════════════════════════════════════


class TestMeanBasin:
    """Test mean basin computation."""

    def test_mean_on_simplex(self, small_bank):
        mean = small_bank.mean_basin()
        assert abs(mean.sum() - 1.0) < 1e-10
        assert np.all(mean >= 0)

    def test_mean_uses_frechet(self, small_bank):
        """Mean basin should be the Fréchet mean, not arithmetic mean."""
        mean = small_bank.mean_basin()
        basins = list(small_bank.coordinates.values())
        expected = frechet_mean(basins)
        d = fisher_rao_distance(mean, expected)
        assert d < 0.05, "Mean basin should match Fréchet mean"


# ═══════════════════════════════════════════════════════════════
#  DOMAIN BIAS
# ═══════════════════════════════════════════════════════════════


class TestDomainBias:
    """Test domain bias operations."""

    def test_push_pop_bias(self, small_bank):
        bias = DomainBias(domain_name="test", strength=0.5)
        small_bank.push_domain_bias(bias)
        popped = small_bank.pop_domain_bias()
        assert popped is not None
        assert popped.domain_name == "test"

    def test_clear_biases(self, small_bank):
        small_bank.push_domain_bias(DomainBias(domain_name="a"))
        small_bank.push_domain_bias(DomainBias(domain_name="b"))
        small_bank.clear_domain_biases()
        assert small_bank.pop_domain_bias() is None

    def test_compute_domain_anchor(self, small_bank):
        anchor = small_bank.compute_domain_anchor([0, 1, 2])
        assert abs(anchor.sum() - 1.0) < 1e-10
        assert np.all(anchor >= 0)


# ═══════════════════════════════════════════════════════════════
#  PERSISTENCE
# ═══════════════════════════════════════════════════════════════


class TestPersistence:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, small_bank):
        with tempfile.TemporaryDirectory() as tmpdir:
            small_bank.save(tmpdir)
            loaded = ResonanceBank.from_file(tmpdir)

            assert len(loaded) == len(small_bank)
            for tid in small_bank.coordinates:
                d = fisher_rao_distance(
                    small_bank.coordinates[tid],
                    loaded.coordinates[tid],
                )
                assert d < 1e-10, f"Token {tid} changed after save/load"

    def test_save_load_preserves_tiers(self, small_bank):
        with tempfile.TemporaryDirectory() as tmpdir:
            small_bank.save(tmpdir)
            loaded = ResonanceBank.from_file(tmpdir)
            for tid in small_bank.tiers:
                assert loaded.tiers[tid] == small_bank.tiers[tid]
