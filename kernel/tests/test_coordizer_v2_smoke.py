"""
test_coordizer_v2_smoke.py — End-to-end smoke test for CoordizerV2
===================================================================

Adapted from Claude's standalone smoke_test.py for the kernel/ package
structure. Since CoordizerV2Trainer and build_corpus live in a separate
training package (not in kernel/coordizer_v2/), this smoke test exercises
the runtime path instead:

  1. Build a synthetic resonance bank with realistic structure
  2. Verify all coordinates remain on simplex
  3. Coordize text and verify output structure
  4. Test generation via geodesic foresight
  5. Run validation pipeline
  6. Save/load bank round-trip
  7. Verify Fisher-Rao distances are preserved after load

Run:  python -m pytest kernel/tests/test_coordizer_v2_smoke.py -v
"""

from __future__ import annotations

import json
import tempfile
import time

import numpy as np
import pytest

from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    KAPPA_STAR,
    E8_RANK,
    fisher_rao_distance,
    frechet_mean,
    slerp,
    to_simplex,
)
from kernel.coordizer_v2.resonance_bank import ResonanceBank
from kernel.coordizer_v2.types import (
    BasinCoordinate,
    CoordizationResult,
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


def _build_synthetic_bank(
    n_tokens: int = 200,
    seed: int = 42,
) -> ResonanceBank:
    """Build a synthetic resonance bank with realistic tier distribution.

    Tier distribution:
        10% fundamental, 20% first harmonic, 30% upper harmonic, 40% overtone haze
    """
    rng = np.random.RandomState(seed)
    bank = ResonanceBank(target_dim=BASIN_DIM)

    for i in range(n_tokens):
        basin = rng.dirichlet(np.ones(BASIN_DIM))
        bank.coordinates[i] = to_simplex(basin)
        bank.token_strings[i] = f"tok_{i:04d}"

        # Tier assignment
        frac = i / n_tokens
        if frac < 0.10:
            bank.tiers[i] = HarmonicTier.FUNDAMENTAL
        elif frac < 0.30:
            bank.tiers[i] = HarmonicTier.FIRST_HARMONIC
        elif frac < 0.60:
            bank.tiers[i] = HarmonicTier.UPPER_HARMONIC
        else:
            bank.tiers[i] = HarmonicTier.OVERTONE_HAZE

        bank.frequencies[i] = 50.0 + i * 5.0
        bank.basin_mass[i] = 1.0 / (1 + i * 0.005)
        bank.activation_counts[i] = 0

    # Make some synonym pairs closer (Fisher-Rao proximity)
    for a, b in [(0, 1), (2, 3), (10, 11), (20, 21)]:
        bank.coordinates[b] = slerp(bank.coordinates[a], bank.coordinates[b], 0.15)

    bank._rebuild_matrix()
    return bank


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def bank() -> ResonanceBank:
    return _build_synthetic_bank(n_tokens=200)


@pytest.fixture(scope="module")
def coordizer(bank) -> CoordizerV2:
    return CoordizerV2(bank=bank)


# ═══════════════════════════════════════════════════════════════
#  1. SIMPLEX INVARIANT
# ═══════════════════════════════════════════════════════════════


class TestSimplexInvariant:
    """All bank coordinates must be on the probability simplex."""

    def test_all_coordinates_on_simplex(self, bank):
        violations = 0
        for tid, coord in bank.coordinates.items():
            if not _is_on_simplex(coord):
                violations += 1
        assert violations == 0, f"{violations} simplex violations"

    def test_coordinates_correct_dimension(self, bank):
        for tid, coord in bank.coordinates.items():
            assert coord.shape == (BASIN_DIM,), f"Token {tid}: shape={coord.shape}"

    def test_bank_size(self, bank):
        assert len(bank.coordinates) == 200


# ═══════════════════════════════════════════════════════════════
#  2. COORDIZE TEXT
# ═══════════════════════════════════════════════════════════════


class TestCoordize:
    """Test text coordization produces valid output."""

    def test_coordize_returns_result(self, coordizer):
        result = coordizer.coordize("hello world")
        assert isinstance(result, CoordizationResult)

    def test_coordize_produces_ids(self, coordizer):
        # Use token strings that exist in the bank ("tok_0000" etc.)
        result = coordizer.coordize("tok_0000 tok_0001 tok_0002")
        assert len(result.coord_ids) > 0, (
            "coordize should find tokens when input matches bank strings"
        )

    def test_coordize_ids_in_bank(self, coordizer, bank):
        result = coordizer.coordize("tok_0010 tok_0020")
        for cid in result.coord_ids:
            assert cid in bank.coordinates, f"Coord ID {cid} not in bank"


# ═══════════════════════════════════════════════════════════════
#  3. GENERATION
# ═══════════════════════════════════════════════════════════════


class TestGeneration:
    """Test generation via geodesic foresight."""

    def test_generate_returns_token(self, coordizer, bank):
        trajectory = [bank.coordinates[0], bank.coordinates[1]]
        tid, basin = coordizer.generate_next(trajectory)
        assert tid in bank.coordinates
        assert _is_on_simplex(basin)

    def test_generate_result_on_simplex(self, coordizer, bank):
        trajectory = [bank.coordinates[i] for i in range(5)]
        tid, basin = coordizer.generate_next(trajectory)
        assert _is_on_simplex(basin), f"Generated basin not on simplex: sum={basin.sum()}"


# ═══════════════════════════════════════════════════════════════
#  4. VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════


class TestValidation:
    """Test the full validation pipeline."""

    def test_validation_returns_result(self, bank):
        result = validate_resonance_bank(bank, verbose=False)
        assert isinstance(result, ValidationResult)

    def test_kappa_measured_positive(self, bank):
        result = validate_resonance_bank(bank, verbose=False)
        assert result.kappa_measured > 0

    def test_validation_has_all_fields(self, bank):
        result = validate_resonance_bank(bank, verbose=False)
        assert result.kappa_std >= 0
        assert result.beta_running >= 0
        assert isinstance(result.semantic_correlation, float)
        assert isinstance(result.harmonic_ratio_quality, float)
        assert isinstance(result.tier_distribution, dict)
        assert isinstance(result.passed, bool)

    def test_validation_summary(self, bank):
        result = validate_resonance_bank(bank, verbose=False)
        summary = result.summary()
        assert "κ=" in summary
        assert "β=" in summary


# ═══════════════════════════════════════════════════════════════
#  5. PERSISTENCE ROUND-TRIP
# ═══════════════════════════════════════════════════════════════


class TestPersistence:
    """Test save/load round-trip preserves geometric structure."""

    def test_save_load_roundtrip(self, bank):
        with tempfile.TemporaryDirectory() as td:
            bank.save(td)
            loaded = ResonanceBank.from_file(td)
            assert len(loaded.coordinates) == len(bank.coordinates)

    def test_fisher_rao_preserved_after_load(self, bank):
        """Max Fisher-Rao distance between original and loaded basins must be < ε."""
        with tempfile.TemporaryDirectory() as td:
            bank.save(td)
            loaded = ResonanceBank.from_file(td)

            max_drift = 0.0
            for tid in bank.coordinates:
                if tid in loaded.coordinates:
                    d = fisher_rao_distance(
                        bank.coordinates[tid],
                        loaded.coordinates[tid],
                    )
                    max_drift = max(max_drift, d)

            assert max_drift < 1e-6, f"Basin drift after load: {max_drift}"

    def test_tiers_preserved_after_load(self, bank):
        with tempfile.TemporaryDirectory() as td:
            bank.save(td)
            loaded = ResonanceBank.from_file(td)
            for tid in bank.tiers:
                assert loaded.tiers[tid] == bank.tiers[tid], (
                    f"Tier mismatch for token {tid}"
                )


# ═══════════════════════════════════════════════════════════════
#  6. FROZEN FACTS
# ═══════════════════════════════════════════════════════════════


class TestFrozenFacts:
    """Verify frozen facts are correct."""

    def test_kappa_star(self):
        assert KAPPA_STAR == 64.0

    def test_basin_dim(self):
        assert BASIN_DIM == 64

    def test_e8_rank(self):
        assert E8_RANK == 8

    def test_basin_dim_is_e8_rank_squared(self):
        assert BASIN_DIM == E8_RANK ** 2
