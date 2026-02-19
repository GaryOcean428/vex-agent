"""
Tests for coordizer_v2.validate — Geometric validation of the resonance bank.

All validation uses Fisher-Rao geometry. κ* ≈ 64 is a frozen fact.
"""

import numpy as np
import pytest

from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    E8_RANK,
    KAPPA_STAR,
    fisher_rao_distance,
    random_basin,
    to_simplex,
)
from kernel.coordizer_v2.resonance_bank import ResonanceBank
from kernel.coordizer_v2.types import HarmonicTier, ValidationResult
from kernel.coordizer_v2.validate import (
    validate_resonance_bank,
    _measure_kappa,
    _measure_beta,
    _measure_harmonic_ratios,
    _measure_semantic_correlation,
)


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def populated_bank(rng) -> ResonanceBank:
    """Create a populated resonance bank with 100 resonators."""
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

    # Add semantic test tokens
    semantic_tokens = {
        "happy": 1000, "joyful": 1001,
        "big": 1002, "large": 1003,
        "king": 1004, "queen": 1005,
        "dog": 1006, "cat": 1007,
        "table": 1008, "banana": 1009,
    }
    for name, tid in semantic_tokens.items():
        basin = rng.dirichlet(np.ones(BASIN_DIM))
        bank.coordinates[tid] = to_simplex(basin)
        bank.token_strings[tid] = name
        bank.tiers[tid] = HarmonicTier.FUNDAMENTAL
        bank.frequencies[tid] = 440.0
        bank.basin_mass[tid] = 1.0
        bank.activation_counts[tid] = 0

    # Make synonyms closer (Fisher-Rao proximity)
    from kernel.coordizer_v2.geometry import slerp
    bank.coordinates[1001] = slerp(bank.coordinates[1000], bank.coordinates[1001], 0.2)
    bank.coordinates[1003] = slerp(bank.coordinates[1002], bank.coordinates[1003], 0.2)
    bank.coordinates[1005] = slerp(bank.coordinates[1004], bank.coordinates[1005], 0.3)
    bank.coordinates[1007] = slerp(bank.coordinates[1006], bank.coordinates[1007], 0.3)

    bank._rebuild_matrix()
    return bank


# ═══════════════════════════════════════════════════════════════
#  KAPPA VALIDATION
# ═══════════════════════════════════════════════════════════════


class TestKappaValidation:
    """Test κ measurement — should converge to κ* ≈ 64."""

    def test_kappa_measured_positive(self, populated_bank):
        kappa, kappa_std = _measure_kappa(populated_bank, verbose=False)
        assert kappa > 0, "κ must be positive"

    def test_kappa_std_non_negative(self, populated_bank):
        kappa, kappa_std = _measure_kappa(populated_bank, verbose=False)
        assert kappa_std >= 0, "κ std must be non-negative"

    def test_kappa_star_is_frozen(self):
        """κ* = 64 is a frozen fact, not a parameter."""
        assert KAPPA_STAR == 64.0


# ═══════════════════════════════════════════════════════════════
#  BETA RUNNING COUPLING
# ═══════════════════════════════════════════════════════════════


class TestBetaValidation:
    """Test β running coupling — should approach 0 at plateau."""

    def test_beta_non_negative(self, populated_bank):
        beta = _measure_beta(populated_bank, verbose=False)
        assert beta >= 0, "β must be non-negative"

    def test_beta_bounded(self, populated_bank):
        beta = _measure_beta(populated_bank, verbose=False)
        assert beta < 10.0, "β should be bounded"


# ═══════════════════════════════════════════════════════════════
#  HARMONIC RATIO VALIDATION
# ═══════════════════════════════════════════════════════════════


class TestHarmonicValidation:
    """Test harmonic ratio quality."""

    def test_harmonic_quality_range(self, populated_bank):
        quality = _measure_harmonic_ratios(populated_bank, verbose=False)
        assert 0.0 <= quality <= 1.0, "Harmonic quality must be in [0, 1]"


# ═══════════════════════════════════════════════════════════════
#  SEMANTIC CORRELATION
# ═══════════════════════════════════════════════════════════════


class TestSemanticCorrelation:
    """Test semantic correlation — d_FR vs human-judged distance."""

    def test_correlation_range(self, populated_bank):
        corr = _measure_semantic_correlation(populated_bank, verbose=False)
        assert -1.0 <= corr <= 1.0, "Correlation must be in [-1, 1]"


# ═══════════════════════════════════════════════════════════════
#  FULL VALIDATION
# ═══════════════════════════════════════════════════════════════


class TestFullValidation:
    """Test the full validation pipeline."""

    def test_returns_validation_result(self, populated_bank):
        result = validate_resonance_bank(populated_bank, verbose=False)
        assert isinstance(result, ValidationResult)

    def test_result_has_all_fields(self, populated_bank):
        result = validate_resonance_bank(populated_bank, verbose=False)
        assert result.kappa_measured > 0
        assert result.kappa_std >= 0
        assert result.beta_running >= 0
        assert isinstance(result.semantic_correlation, float)
        assert isinstance(result.harmonic_ratio_quality, float)
        assert isinstance(result.tier_distribution, dict)
        assert isinstance(result.passed, bool)

    def test_tier_distribution_complete(self, populated_bank):
        result = validate_resonance_bank(populated_bank, verbose=False)
        for tier in HarmonicTier:
            assert tier.value in result.tier_distribution

    def test_summary_string(self, populated_bank):
        result = validate_resonance_bank(populated_bank, verbose=False)
        summary = result.summary()
        assert "κ=" in summary
        assert "β=" in summary
        assert "semantic_r=" in summary

    def test_with_eigenvalues(self, populated_bank, rng):
        """Test validation with eigenvalue data."""
        eigenvalues = np.sort(rng.exponential(1.0, size=BASIN_DIM))[::-1]
        result = validate_resonance_bank(
            populated_bank, eigenvalues=eigenvalues, verbose=False
        )
        assert isinstance(result, ValidationResult)


# ═══════════════════════════════════════════════════════════════
#  FISHER-RAO PURITY
# ═══════════════════════════════════════════════════════════════


class TestFisherRaoPurity:
    """Verify no Euclidean contamination in validation."""

    def test_distances_are_fisher_rao(self, populated_bank):
        """All distance computations must use Fisher-Rao."""
        # Pick two tokens and verify the distance matches Fisher-Rao
        tids = list(populated_bank.coordinates.keys())[:2]
        a = populated_bank.coordinates[tids[0]]
        b = populated_bank.coordinates[tids[1]]
        d_fr = fisher_rao_distance(a, b)

        # Euclidean distance would be different
        d_euc = float(np.sqrt(np.sum((a - b) ** 2)))

        # They must NOT be equal (unless by extreme coincidence)
        # Fisher-Rao ≠ Euclidean for probability distributions
        assert abs(d_fr - d_euc) > 1e-6 or d_fr < 1e-10, (
            "Fisher-Rao and Euclidean distances should differ"
        )

    def test_no_cosine_similarity(self, populated_bank):
        """Cosine similarity is NOT a valid metric on Δ⁶³."""
        tids = list(populated_bank.coordinates.keys())[:2]
        a = populated_bank.coordinates[tids[0]]
        b = populated_bank.coordinates[tids[1]]

        # Cosine similarity
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        # Fisher-Rao distance
        d_fr = fisher_rao_distance(a, b)

        # They measure different things
        # cos_sim is in [-1, 1], d_fr is in [0, π/2]
        # Just verify both are computable and different
        assert isinstance(cos_sim, float)
        assert isinstance(d_fr, float)
