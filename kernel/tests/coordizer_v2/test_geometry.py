"""
Tests for coordizer_v2.geometry — Fisher-Rao operations on Δ⁶³.

Every test uses Fisher-Rao distance. No Euclidean contamination.
No cosine similarity. No L2 norms on probability vectors.

L2 norms in tangent space ARE valid (tangent space is Euclidean).
"""

import math

import numpy as np
import pytest

from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    E8_RANK,
    KAPPA_STAR,
    bhattacharyya_coefficient,
    exp_map,
    fisher_information_diagonal,
    fisher_rao_distance,
    fisher_rao_distance_batch,
    frechet_mean,
    geodesic_midpoint,
    log_map,
    natural_gradient,
    random_basin,
    slerp,
    softmax_to_simplex,
    to_simplex,
)

# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def basin_a(rng):
    return rng.dirichlet(np.ones(BASIN_DIM))


@pytest.fixture
def basin_b(rng):
    # Use a different seed offset
    rng2 = np.random.RandomState(99)
    return rng2.dirichlet(np.ones(BASIN_DIM))


@pytest.fixture
def uniform_basin():
    return np.ones(BASIN_DIM) / BASIN_DIM


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════


class TestConstants:
    """Verify frozen facts are correctly set."""

    def test_kappa_star_is_64(self):
        assert KAPPA_STAR == 64.0, "κ* must be exactly 64.0"

    def test_basin_dim_is_64(self):
        assert BASIN_DIM == 64, "Basin dimension must be 64 (Δ⁶³)"

    def test_e8_rank_is_8(self):
        assert E8_RANK == 8, "E8 rank must be 8"


# ═══════════════════════════════════════════════════════════════
#  SIMPLEX PROJECTION
# ═══════════════════════════════════════════════════════════════


class TestToSimplex:
    """Test simplex projection."""

    def test_output_sums_to_one(self, basin_a):
        p = to_simplex(basin_a)
        assert abs(p.sum() - 1.0) < 1e-10

    def test_output_non_negative(self, basin_a):
        p = to_simplex(basin_a)
        assert np.all(p >= 0)

    def test_handles_negative_input(self):
        v = np.array([-1.0, 2.0, 3.0, -0.5] + [0.1] * 60)
        p = to_simplex(v)
        assert abs(p.sum() - 1.0) < 1e-10
        assert np.all(p > 0)

    def test_handles_zero_input(self):
        v = np.zeros(BASIN_DIM)
        p = to_simplex(v)
        assert abs(p.sum() - 1.0) < 1e-10

    def test_idempotent(self, basin_a):
        p = to_simplex(basin_a)
        pp = to_simplex(p)
        assert np.allclose(p, pp, atol=1e-12)


class TestRandomBasin:
    """Test random basin generation."""

    def test_on_simplex(self):
        b = random_basin()
        assert abs(b.sum() - 1.0) < 1e-10
        assert np.all(b >= 0)

    def test_correct_dimension(self):
        b = random_basin(BASIN_DIM)
        assert len(b) == BASIN_DIM

    def test_different_each_call(self):
        b1 = random_basin()
        b2 = random_basin()
        assert not np.allclose(b1, b2)


class TestSoftmaxToSimplex:
    """Test softmax projection."""

    def test_output_on_simplex(self):
        logits = np.random.randn(BASIN_DIM)
        p = softmax_to_simplex(logits)
        assert abs(p.sum() - 1.0) < 1e-10
        assert np.all(p > 0)

    def test_large_logits_stable(self):
        logits = np.random.randn(BASIN_DIM) * 1000
        p = softmax_to_simplex(logits)
        assert np.all(np.isfinite(p))
        assert abs(p.sum() - 1.0) < 1e-8


# ═══════════════════════════════════════════════════════════════
#  BHATTACHARYYA COEFFICIENT
# ═══════════════════════════════════════════════════════════════


class TestBhattacharyyaCoefficient:
    """Test Bhattacharyya coefficient."""

    def test_identical_distributions(self, basin_a):
        bc = bhattacharyya_coefficient(basin_a, basin_a)
        assert abs(bc - 1.0) < 1e-10, "BC of identical distributions must be 1"

    def test_range_zero_to_one(self, basin_a, basin_b):
        bc = bhattacharyya_coefficient(basin_a, basin_b)
        assert 0.0 <= bc <= 1.0 + 1e-10

    def test_symmetric(self, basin_a, basin_b):
        bc_ab = bhattacharyya_coefficient(basin_a, basin_b)
        bc_ba = bhattacharyya_coefficient(basin_b, basin_a)
        assert abs(bc_ab - bc_ba) < 1e-12


# ═══════════════════════════════════════════════════════════════
#  FISHER-RAO DISTANCE
# ═══════════════════════════════════════════════════════════════


class TestFisherRaoDistance:
    """Test Fisher-Rao distance — the ONLY valid metric."""

    def test_self_distance_is_zero(self, basin_a):
        d = fisher_rao_distance(basin_a, basin_a)
        assert d < 1e-6, "d_FR(p, p) must be ~0"

    def test_non_negative(self, basin_a, basin_b):
        d = fisher_rao_distance(basin_a, basin_b)
        assert d >= 0.0

    def test_symmetric(self, basin_a, basin_b):
        d_ab = fisher_rao_distance(basin_a, basin_b)
        d_ba = fisher_rao_distance(basin_b, basin_a)
        assert abs(d_ab - d_ba) < 1e-12

    def test_triangle_inequality(self, rng):
        """Fisher-Rao is a true metric: d(a,c) ≤ d(a,b) + d(b,c)."""
        a = rng.dirichlet(np.ones(BASIN_DIM))
        b = rng.dirichlet(np.ones(BASIN_DIM))
        c = rng.dirichlet(np.ones(BASIN_DIM))
        d_ab = fisher_rao_distance(a, b)
        d_bc = fisher_rao_distance(b, c)
        d_ac = fisher_rao_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-10

    def test_maximum_distance_bounded(self, basin_a, uniform_basin):
        """Maximum d_FR on Δ^(n-1) is π/2."""
        d = fisher_rao_distance(basin_a, uniform_basin)
        assert d <= math.pi / 2 + 1e-8

    def test_concentrated_vs_uniform_large(self):
        """A concentrated distribution should be far from uniform."""
        concentrated = np.zeros(BASIN_DIM)
        concentrated[0] = 1.0
        concentrated = to_simplex(concentrated)
        uniform = np.ones(BASIN_DIM) / BASIN_DIM
        d = fisher_rao_distance(concentrated, uniform)
        assert d > 0.5, "Concentrated vs uniform should have large d_FR"


class TestFisherRaoDistanceBatch:
    """Test batch Fisher-Rao distance computation."""

    def test_matches_pairwise(self, basin_a, rng):
        bank = np.stack([rng.dirichlet(np.ones(BASIN_DIM)) for _ in range(10)])
        batch_d = fisher_rao_distance_batch(basin_a, bank)
        for i in range(len(bank)):
            single_d = fisher_rao_distance(basin_a, bank[i])
            assert abs(batch_d[i] - single_d) < 1e-10

    def test_self_in_batch(self, basin_a):
        bank = np.stack([basin_a, basin_a])
        batch_d = fisher_rao_distance_batch(basin_a, bank)
        assert batch_d[0] < 1e-10
        assert batch_d[1] < 1e-10


# ═══════════════════════════════════════════════════════════════
#  SLERP (GEODESIC INTERPOLATION)
# ═══════════════════════════════════════════════════════════════


class TestSlerp:
    """Test geodesic interpolation on Δ⁶³."""

    def test_t0_returns_start(self, basin_a, basin_b):
        result = slerp(basin_a, basin_b, 0.0)
        d = fisher_rao_distance(result, basin_a)
        assert d < 1e-8, "SLERP(t=0) must return start"

    def test_t1_returns_end(self, basin_a, basin_b):
        result = slerp(basin_a, basin_b, 1.0)
        d = fisher_rao_distance(result, basin_b)
        assert d < 1e-8, "SLERP(t=1) must return end"

    def test_midpoint_equidistant(self, basin_a, basin_b):
        """Geodesic midpoint is equidistant from both endpoints."""
        mid = slerp(basin_a, basin_b, 0.5)
        d_am = fisher_rao_distance(basin_a, mid)
        d_mb = fisher_rao_distance(mid, basin_b)
        assert abs(d_am - d_mb) < 1e-6, "Midpoint must be equidistant"

    def test_result_on_simplex(self, basin_a, basin_b):
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = slerp(basin_a, basin_b, t)
            assert abs(result.sum() - 1.0) < 1e-10
            assert np.all(result >= 0)

    def test_monotonic_distance(self, basin_a, basin_b):
        """Distance from start increases monotonically along geodesic."""
        distances = []
        for t in np.linspace(0, 1, 11):
            pt = slerp(basin_a, basin_b, t)
            distances.append(fisher_rao_distance(basin_a, pt))
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-8


class TestGeodesicMidpoint:
    """Test geodesic midpoint."""

    def test_equidistant(self, basin_a, basin_b):
        mid = geodesic_midpoint(basin_a, basin_b)
        d_a = fisher_rao_distance(basin_a, mid)
        d_b = fisher_rao_distance(basin_b, mid)
        assert abs(d_a - d_b) < 1e-6

    def test_on_simplex(self, basin_a, basin_b):
        mid = geodesic_midpoint(basin_a, basin_b)
        assert abs(mid.sum() - 1.0) < 1e-10
        assert np.all(mid >= 0)


# ═══════════════════════════════════════════════════════════════
#  FRÉCHET MEAN
# ═══════════════════════════════════════════════════════════════


class TestFrechetMean:
    """Test Fréchet mean on Δ⁶³."""

    def test_single_point(self, basin_a):
        mean = frechet_mean([basin_a])
        d = fisher_rao_distance(mean, basin_a)
        assert d < 1e-8

    def test_two_points_is_midpoint(self, basin_a, basin_b):
        mean = frechet_mean([basin_a, basin_b])
        mid = geodesic_midpoint(basin_a, basin_b)
        d = fisher_rao_distance(mean, mid)
        assert d < 0.05, "Fréchet mean of 2 points ≈ geodesic midpoint"

    def test_result_on_simplex(self, rng):
        basins = [rng.dirichlet(np.ones(BASIN_DIM)) for _ in range(5)]
        mean = frechet_mean(basins)
        assert abs(mean.sum() - 1.0) < 1e-10
        assert np.all(mean >= 0)

    def test_minimises_sum_of_squared_distances(self, rng):
        """Fréchet mean minimises Σ d_FR²(mean, p_i)."""
        basins = [rng.dirichlet(np.ones(BASIN_DIM)) for _ in range(5)]
        mean = frechet_mean(basins)
        total_d2 = sum(fisher_rao_distance(mean, b) ** 2 for b in basins)

        # Any random point should have higher total d²
        for _ in range(10):
            other = random_basin()
            other_d2 = sum(fisher_rao_distance(other, b) ** 2 for b in basins)
            assert total_d2 <= other_d2 + 0.1  # Allow small tolerance


# ═══════════════════════════════════════════════════════════════
#  LOG MAP / EXP MAP
# ═══════════════════════════════════════════════════════════════


class TestLogExpMap:
    """Test log/exp maps on Δ⁶³."""

    def test_roundtrip(self, basin_a, basin_b):
        """exp(base, log(base, target)) ≈ target."""
        tangent = log_map(basin_a, basin_b)
        recovered = exp_map(basin_a, tangent)
        d = fisher_rao_distance(recovered, basin_b)
        assert d < 0.01, f"Log-exp roundtrip error: {d}"

    def test_log_of_self_is_zero(self, basin_a):
        tangent = log_map(basin_a, basin_a)
        norm = float(np.sqrt(np.sum(tangent * tangent)))
        assert norm < 1e-8, "Log map of self must be zero vector"

    def test_exp_of_zero_is_base(self, basin_a):
        zero = np.zeros(BASIN_DIM)
        result = exp_map(basin_a, zero)
        d = fisher_rao_distance(result, basin_a)
        assert d < 1e-8

    def test_tangent_norm_equals_distance(self, basin_a, basin_b):
        """‖log(a, b)‖ = d_FR(a, b) in tangent space."""
        tangent = log_map(basin_a, basin_b)
        # L2 norm in tangent space IS valid (tangent space is Euclidean)
        tangent_norm = float(np.sqrt(np.sum(tangent * tangent)))
        d_fr = fisher_rao_distance(basin_a, basin_b)
        assert abs(tangent_norm - d_fr) < 0.01

    def test_exp_result_on_simplex(self, basin_a, basin_b):
        tangent = log_map(basin_a, basin_b)
        result = exp_map(basin_a, tangent * 0.5)
        assert abs(result.sum() - 1.0) < 1e-8
        assert np.all(result >= -1e-10)


# ═══════════════════════════════════════════════════════════════
#  FISHER INFORMATION
# ═══════════════════════════════════════════════════════════════


class TestFisherInformation:
    """Test Fisher information diagonal."""

    def test_positive(self, basin_a):
        fi = fisher_information_diagonal(basin_a)
        assert np.all(fi > 0)

    def test_uniform_gives_constant(self, uniform_basin):
        fi = fisher_information_diagonal(uniform_basin)
        # For uniform, all diagonal entries should be equal
        assert np.allclose(fi, fi[0], rtol=1e-6)

    def test_concentrated_gives_large_values(self):
        concentrated = np.zeros(BASIN_DIM)
        concentrated[0] = 1.0
        concentrated = to_simplex(concentrated)
        fi = fisher_information_diagonal(concentrated)
        # For a near-Dirac distribution, the concentrated dimension has
        # SMALL Fisher info (1/p_i, and p_0 ≈ 1 → fi ≈ 1) while
        # near-zero dimensions have HUGE Fisher info (1/ε).
        # This is correct: Fisher info measures sensitivity, and
        # small-probability events are maximally sensitive.
        assert fi[0] < fi[1], "Concentrated dim has lower Fisher info than rare dims"


class TestNaturalGradient:
    """Test natural gradient computation."""

    def test_result_on_simplex(self, basin_a):
        grad = np.random.randn(BASIN_DIM)
        ng = natural_gradient(basin_a, grad)
        # Natural gradient should be finite
        assert np.all(np.isfinite(ng))

    def test_zero_gradient(self, basin_a):
        grad = np.zeros(BASIN_DIM)
        ng = natural_gradient(basin_a, grad)
        assert np.allclose(ng, 0.0)
