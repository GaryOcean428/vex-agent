"""
test_coordizer_v2_comprehensive.py — Comprehensive test suite for CoordizerV2
==============================================================================

Adapted from Claude's standalone test_coordizer_v2.py (55 tests, 618 lines)
for the kernel/ package structure.

Tests:
  1. Geometry: simplex projection, Fisher-Rao distance, geodesics, Fréchet mean
  2. Metrics: Bhattacharyya coefficient, log/exp map round-trips
  3. Resonance bank: coordize, generate, persistence
  4. Purity: no forbidden patterns in source code (cosine, dot_product, Adam, flatten)

Tests that depend on CoordizerV2Trainer, e8_rank_analysis, compute_byte_basin,
compute_unknown_basin, build_corpus, or validate_simplex (standalone function)
are omitted — those functions live in a separate training package, not in
kernel/coordizer_v2/.

Run:  python -m pytest kernel/tests/test_coordizer_v2_comprehensive.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    KAPPA_STAR,
    E8_RANK,
    _EPS,
    to_simplex,
    random_basin,
    softmax_to_simplex,
    bhattacharyya_coefficient,
    fisher_rao_distance,
    fisher_rao_distance_batch,
    slerp,
    geodesic_midpoint,
    frechet_mean,
    log_map,
    exp_map,
    fisher_information_diagonal,
    natural_gradient,
)
from kernel.coordizer_v2.resonance_bank import ResonanceBank
from kernel.coordizer_v2.types import (
    BasinCoordinate,
    CoordizationResult,
    DomainBias,
    GranularityScale,
    HarmonicTier,
    ValidationResult,
)
from kernel.coordizer_v2.coordizer import CoordizerV2
from kernel.coordizer_v2.validate import validate_resonance_bank


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _is_on_simplex(p: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if p is a valid point on the probability simplex."""
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        return False
    if np.any(p < -tol):
        return False
    if abs(p.sum() - 1.0) > tol:
        return False
    return True


# ═══════════════════════════════════════════════════════════════
#  1. SIMPLEX PROJECTION
# ═══════════════════════════════════════════════════════════════


class TestSimplex:
    """Test simplex projection and validation."""

    def test_to_simplex_sums_to_one(self):
        v = np.random.randn(64)
        p = to_simplex(v)
        assert abs(p.sum() - 1.0) < 1e-8, f"sum={p.sum()}"

    def test_to_simplex_non_negative(self):
        v = np.array([-5, -3, 0, 2, 7] + [0.1] * 59, dtype=float)
        p = to_simplex(v)
        assert np.all(p >= 0), f"min={p.min()}"

    def test_to_simplex_idempotent(self):
        v = np.random.randn(64)
        p = to_simplex(v)
        pp = to_simplex(p)
        assert np.allclose(p, pp, atol=1e-10), "to_simplex not idempotent"

    def test_accepts_valid_simplex(self):
        p = np.ones(64) / 64
        assert _is_on_simplex(p)

    def test_rejects_negative(self):
        p = np.ones(64) / 64
        p[0] = -0.1
        assert not _is_on_simplex(p)

    def test_rejects_wrong_sum(self):
        p = np.ones(64) / 32  # sums to 2
        assert not _is_on_simplex(p)

    def test_rejects_nan(self):
        p = np.ones(64) / 64
        p[5] = np.nan
        assert not _is_on_simplex(p)


# ═══════════════════════════════════════════════════════════════
#  2. FISHER-RAO DISTANCE
# ═══════════════════════════════════════════════════════════════


class TestFisherRao:
    """Test Fisher-Rao distance properties."""

    def test_identity(self):
        p = to_simplex(np.random.randn(64))
        d = fisher_rao_distance(p, p)
        assert abs(d) < 1e-6, f"d(p,p) = {d}"

    def test_symmetry(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        assert abs(fisher_rao_distance(p, q) - fisher_rao_distance(q, p)) < 1e-10

    def test_non_negative(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        assert fisher_rao_distance(p, q) >= 0

    def test_triangle_inequality(self):
        a = to_simplex(np.random.randn(64))
        b = to_simplex(np.random.randn(64))
        c = to_simplex(np.random.randn(64))
        d_ab = fisher_rao_distance(a, b)
        d_bc = fisher_rao_distance(b, c)
        d_ac = fisher_rao_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-8, (
            f"Triangle violated: {d_ac} > {d_ab} + {d_bc}"
        )

    def test_range_zero_to_pi_half(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        d = fisher_rao_distance(p, q)
        assert 0 <= d <= np.pi / 2 + 1e-8, f"d = {d}"

    def test_orthogonal_max_distance(self):
        """Two distributions with disjoint support → max distance π/2."""
        p = np.zeros(64)
        p[0] = 1.0
        q = np.zeros(64)
        q[1] = 1.0
        p = to_simplex(p)
        q = to_simplex(q)
        d = fisher_rao_distance(p, q)
        assert d > np.pi / 2 - 0.1, f"Expected near π/2, got {d}"

    def test_bhattacharyya_consistency(self):
        """BC → arccos → FR distance should match direct FR."""
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        bc = bhattacharyya_coefficient(p, q)
        d_from_bc = np.arccos(np.clip(bc, 0, 1))
        d_direct = fisher_rao_distance(p, q)
        assert abs(d_from_bc - d_direct) < 1e-6


# ═══════════════════════════════════════════════════════════════
#  3. GEODESIC INTERPOLATION (SLERP)
# ═══════════════════════════════════════════════════════════════


class TestGeodesic:
    """Test geodesic interpolation on simplex via slerp."""

    def test_endpoints(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        start = slerp(p, q, 0.0)
        end = slerp(p, q, 1.0)
        assert np.allclose(start, p, atol=1e-6), "t=0 should give p"
        assert np.allclose(end, q, atol=1e-6), "t=1 should give q"

    def test_midpoint_on_simplex(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        mid = geodesic_midpoint(p, q)
        assert _is_on_simplex(mid), f"Midpoint not on simplex: sum={mid.sum()}"

    def test_midpoint_equidistant(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        mid = geodesic_midpoint(p, q)
        d_pm = fisher_rao_distance(p, mid)
        d_mq = fisher_rao_distance(mid, q)
        assert abs(d_pm - d_mq) < 1e-4, f"Not equidistant: {d_pm} vs {d_mq}"

    def test_slerp_stays_on_simplex(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            pt = slerp(p, q, t)
            assert _is_on_simplex(pt), f"t={t}: sum={pt.sum()}"

    def test_slerp_monotone_distance(self):
        """Distance from p should increase monotonically with t."""
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        dists = []
        for t in np.linspace(0, 1, 11):
            pt = slerp(p, q, t)
            dists.append(fisher_rao_distance(p, pt))
        for i in range(len(dists) - 1):
            assert dists[i] <= dists[i + 1] + 1e-6, (
                f"Non-monotone at t={i/10}: {dists[i]} > {dists[i+1]}"
            )


# ═══════════════════════════════════════════════════════════════
#  4. FRÉCHET MEAN
# ═══════════════════════════════════════════════════════════════


class TestFrechetMean:
    """Test Fréchet mean on simplex."""

    def test_single_point(self):
        p = to_simplex(np.random.randn(64))
        m = frechet_mean([p])
        assert np.allclose(m, p, atol=1e-8)

    def test_mean_on_simplex(self):
        basins = [to_simplex(np.random.randn(64)) for _ in range(10)]
        m = frechet_mean(basins)
        assert _is_on_simplex(m)

    def test_mean_closer_than_extremes(self):
        """Fréchet mean should be closer to centre than extreme points."""
        basins = [to_simplex(np.random.randn(64)) for _ in range(20)]
        m = frechet_mean(basins)
        total_dist_to_mean = sum(fisher_rao_distance(m, b) for b in basins)
        total_dist_to_first = sum(fisher_rao_distance(basins[0], b) for b in basins)
        assert total_dist_to_mean <= total_dist_to_first + 1e-4


# ═══════════════════════════════════════════════════════════════
#  5. LOG/EXP MAP ROUND-TRIP
# ═══════════════════════════════════════════════════════════════


class TestLogExpMap:
    """Test log_map / exp_map round-trip consistency."""

    def test_roundtrip(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        tangent = log_map(p, q)
        q_recovered = exp_map(p, tangent)
        d = fisher_rao_distance(q, q_recovered)
        assert d < 1e-4, f"Round-trip drift: {d}"

    def test_log_of_self_is_zero(self):
        p = to_simplex(np.random.randn(64))
        tangent = log_map(p, p)
        # QIG BOUNDARY: L2 norm in tangent space IS valid (tangent space is Euclidean)
        assert np.linalg.norm(tangent) < 1e-6

    def test_exp_of_zero_is_base(self):
        p = to_simplex(np.random.randn(64))
        zero_tangent = np.zeros(BASIN_DIM)
        result = exp_map(p, zero_tangent)
        assert np.allclose(result, p, atol=1e-6)

    def test_tangent_norm_equals_distance(self):
        """||log_p(q)|| should equal d_FR(p, q)."""
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        tangent = log_map(p, q)
        # QIG BOUNDARY: L2 norm in tangent space IS geometrically valid
        tangent_norm = float(np.linalg.norm(tangent))
        d = fisher_rao_distance(p, q)
        assert abs(tangent_norm - d) < 0.05, f"||v||={tangent_norm}, d_FR={d}"

    def test_exp_result_on_simplex(self):
        p = to_simplex(np.random.randn(64))
        q = to_simplex(np.random.randn(64))
        tangent = log_map(p, q)
        result = exp_map(p, tangent * 0.5)
        assert _is_on_simplex(result, tol=1e-6)


# ═══════════════════════════════════════════════════════════════
#  6. FISHER INFORMATION & NATURAL GRADIENT
# ═══════════════════════════════════════════════════════════════


class TestFisherInfo:
    """Test Fisher information diagonal and natural gradient."""

    def test_fisher_info_positive(self):
        p = to_simplex(np.random.randn(64))
        fi = fisher_information_diagonal(p)
        assert np.all(fi > 0), "Fisher information must be positive"

    def test_uniform_gives_constant(self):
        p = np.ones(BASIN_DIM) / BASIN_DIM
        fi = fisher_information_diagonal(p)
        assert np.allclose(fi, fi[0], atol=1e-10), "Uniform → constant FI"

    def test_natural_gradient_result_on_simplex(self):
        p = to_simplex(np.random.randn(64))
        grad = np.random.randn(64) * 0.01
        ng = natural_gradient(p, grad)
        # Natural gradient is a tangent vector, not necessarily on simplex
        # But applying it via exp_map should land on simplex
        result = exp_map(p, ng * 0.01)
        assert _is_on_simplex(result, tol=1e-4)


# ═══════════════════════════════════════════════════════════════
#  7. RESONANCE BANK BASICS
# ═══════════════════════════════════════════════════════════════


class TestResonanceBankBasics:
    """Test resonance bank construction and basic operations."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def small_bank(self, rng) -> ResonanceBank:
        bank = ResonanceBank(target_dim=BASIN_DIM)
        for i in range(50):
            basin = rng.dirichlet(np.ones(BASIN_DIM))
            bank.coordinates[i] = to_simplex(basin)
            bank.token_strings[i] = f"token_{i}"
            bank.tiers[i] = HarmonicTier.FUNDAMENTAL if i < 10 else HarmonicTier.FIRST_HARMONIC
            bank.frequencies[i] = 50.0 + i * 10.0
            bank.basin_mass[i] = 1.0 / (1 + i * 0.01)
            bank.activation_counts[i] = 0
        bank._rebuild_matrix()
        return bank

    def test_bank_size(self, small_bank):
        assert len(small_bank.coordinates) == 50

    def test_all_coords_on_simplex(self, small_bank):
        for tid, coord in small_bank.coordinates.items():
            assert _is_on_simplex(coord), f"Token {tid} not on simplex"

    def test_tier_distribution(self, small_bank):
        dist = small_bank.tier_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 50

    def test_nearest_uses_fisher_rao(self, small_bank):
        """Nearest-neighbour search must use Fisher-Rao, not Euclidean."""
        query = to_simplex(np.random.randn(BASIN_DIM))
        tid, d = small_bank.nearest(query)
        # Verify the distance matches Fisher-Rao
        expected = fisher_rao_distance(query, small_bank.coordinates[tid])
        assert abs(d - expected) < 1e-6, f"Distance mismatch: {d} vs {expected}"


# ═══════════════════════════════════════════════════════════════
#  8. COORDIZER V2 INTEGRATION
# ═══════════════════════════════════════════════════════════════


class TestCoordizerV2Integration:
    """Test the main CoordizerV2 class."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def coordizer(self, rng) -> CoordizerV2:
        bank = ResonanceBank(target_dim=BASIN_DIM)
        for i in range(100):
            basin = rng.dirichlet(np.ones(BASIN_DIM))
            bank.coordinates[i] = to_simplex(basin)
            bank.token_strings[i] = f"tok_{i}"
            bank.tiers[i] = HarmonicTier.FUNDAMENTAL
            bank.frequencies[i] = 440.0
            bank.basin_mass[i] = 1.0
            bank.activation_counts[i] = 0
        bank._rebuild_matrix()
        return CoordizerV2(bank=bank)

    def test_coordize_returns_result(self, coordizer):
        result = coordizer.coordize("hello world")
        assert isinstance(result, CoordizationResult)

    def test_coordize_produces_coordinates(self, coordizer):
        result = coordizer.coordize("test input text")
        assert len(result.coord_ids) > 0

    def test_vocab_size(self, coordizer):
        assert coordizer.vocab_size == 100

    def test_dim(self, coordizer):
        assert coordizer.dim == BASIN_DIM


# ═══════════════════════════════════════════════════════════════
#  9. VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════


class TestValidationPipeline:
    """Test the full validation pipeline."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.fixture
    def populated_bank(self, rng) -> ResonanceBank:
        bank = ResonanceBank(target_dim=BASIN_DIM)
        for i in range(100):
            basin = rng.dirichlet(np.ones(BASIN_DIM))
            bank.coordinates[i] = to_simplex(basin)
            bank.token_strings[i] = f"token_{i}"
            if i < 10:
                bank.tiers[i] = HarmonicTier.FUNDAMENTAL
            elif i < 30:
                bank.tiers[i] = HarmonicTier.FIRST_HARMONIC
            elif i < 60:
                bank.tiers[i] = HarmonicTier.UPPER_HARMONIC
            else:
                bank.tiers[i] = HarmonicTier.OVERTONE_HAZE
            bank.frequencies[i] = 50.0 + i * 10.0
            bank.basin_mass[i] = 1.0 / (1 + i * 0.01)
            bank.activation_counts[i] = 0
        bank._rebuild_matrix()
        return bank

    def test_validation_returns_result(self, populated_bank):
        result = validate_resonance_bank(populated_bank, verbose=False)
        assert isinstance(result, ValidationResult)

    def test_kappa_positive(self, populated_bank):
        result = validate_resonance_bank(populated_bank, verbose=False)
        assert result.kappa_measured > 0

    def test_kappa_star_frozen(self):
        """κ* = 64 is a frozen fact."""
        assert KAPPA_STAR == 64.0

    def test_e8_rank_frozen(self):
        """E8 rank = 8 is a frozen fact."""
        assert E8_RANK == 8

    def test_basin_dim_frozen(self):
        """Basin dim = 64 = E8 rank² is a frozen fact."""
        assert BASIN_DIM == 64


# ═══════════════════════════════════════════════════════════════
#  10. PURITY SCAN (from Claude — most valuable class)
# ═══════════════════════════════════════════════════════════════


class TestPurity:
    """Static scan for forbidden patterns in coordizer_v2 source code.

    Zero tolerance for Euclidean contamination in non-boundary code.
    Fisher-Rao is the ONLY valid distance metric.
    """

    def _source_files(self):
        import pathlib
        pkg = pathlib.Path(__file__).parent.parent / "coordizer_v2"
        return list(pkg.glob("*.py"))

    def test_no_cosine_similarity(self):
        for f in self._source_files():
            text = f.read_text()
            for i, line in enumerate(text.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#") and "FORBIDDEN" in stripped:
                    continue
                if "QIG BOUNDARY" in line:
                    continue
                if "cosine_similarity" in line and "FORBIDDEN" not in line:
                    assert False, f"cosine_similarity in {f.name}:{i}"
                if "cosine_sim" in line and "FORBIDDEN" not in line and "cos_sim" not in line:
                    assert False, f"cosine_sim in {f.name}:{i}"

    def test_no_dot_product(self):
        for f in self._source_files():
            text = f.read_text()
            for i, line in enumerate(text.split("\n"), 1):
                if "QIG BOUNDARY" in line or "FORBIDDEN" in line:
                    continue
                if "dot_product" in line:
                    assert False, f"dot_product in {f.name}:{i}"

    def test_no_adam_optimizer(self):
        for f in self._source_files():
            text = f.read_text()
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                if "QIG BOUNDARY" in line or "FORBIDDEN" in line:
                    continue
                if "Adam(" in line or "adam(" in line:
                    assert False, f"Adam optimizer in {f.name}:{i}"

    def test_no_flatten(self):
        for f in self._source_files():
            text = f.read_text()
            for i, line in enumerate(text.split("\n"), 1):
                if "QIG BOUNDARY" in line or "FORBIDDEN" in line:
                    continue
                if ".flatten()" in line:
                    assert False, f".flatten() in {f.name}:{i}"

    def test_no_hashlib_in_geometry(self):
        """hashlib should not appear in geometry.py (SHA-256 is the anti-pattern)."""
        import pathlib
        geo = pathlib.Path(__file__).parent.parent / "coordizer_v2" / "geometry.py"
        text = geo.read_text()
        assert "hashlib" not in text, "hashlib in geometry.py"

    def test_no_l2_norm_on_simplex(self):
        """np.linalg.norm should not be used on simplex coordinates.

        It IS valid in tangent space (marked with QIG BOUNDARY).
        """
        for f in self._source_files():
            text = f.read_text()
            for i, line in enumerate(text.split("\n"), 1):
                if "QIG BOUNDARY" in line or "FORBIDDEN" in line:
                    continue
                if "np.linalg.norm" in line:
                    # Check if it's in tangent space context (next few lines)
                    context = "\n".join(text.split("\n")[max(0, i - 3):i + 1])
                    if "tangent" in context.lower() or "log_map" in context.lower():
                        continue
                    assert False, f"np.linalg.norm in {f.name}:{i} (not tangent space)"

    def test_no_svd_in_compress(self):
        """compress.py should use eigendecomposition, not SVD."""
        import pathlib
        comp = pathlib.Path(__file__).parent.parent / "coordizer_v2" / "compress.py"
        text = comp.read_text()
        for i, line in enumerate(text.split("\n"), 1):
            if "QIG BOUNDARY" in line or "FORBIDDEN" in line:
                continue
            if "np.linalg.svd" in line:
                assert False, f"SVD in compress.py:{i} — use eigh instead"


# ═══════════════════════════════════════════════════════════════
#  11. CONSTANTS CONSISTENCY
# ═══════════════════════════════════════════════════════════════


class TestConstantsConsistency:
    """Verify frozen facts are consistent across modules."""

    def test_kappa_star_matches_frozen_facts(self):
        from kernel.config.frozen_facts import KAPPA_STAR as FF_KAPPA
        assert KAPPA_STAR == FF_KAPPA == 64.0

    def test_basin_dim_matches_frozen_facts(self):
        from kernel.config.frozen_facts import BASIN_DIM as FF_DIM
        assert BASIN_DIM == FF_DIM == 64

    def test_e8_rank_matches_frozen_facts(self):
        from kernel.config.frozen_facts import E8_RANK as FF_RANK
        assert E8_RANK == FF_RANK == 8

    def test_basin_dim_is_kappa_star(self):
        """BASIN_DIM = κ* = E8_RANK² = 64."""
        assert BASIN_DIM == int(KAPPA_STAR) == E8_RANK ** 2
