"""Tests for coordizer rejection mechanism + geometric de-biasing (#74).

Verifies:
  1. Sovereignty violation → rejection
  2. Entropy collapse → rejection
  3. Adversarial proximity → rejection
  4. Normal coordization passes (no false positives)
  5. Geometric salience weighting in string-based coordization
  6. Rejection fields populated correctly
  7. Frozen identity management
"""

from __future__ import annotations

import numpy as np

from kernel.config.consciousness_constants import (
    ADVERSARIAL_PROXIMITY,
    ENTROPY_FLOOR,
    SOVEREIGNTY_MAX_DRIFT,
)
from kernel.coordizer_v2.coordizer import CoordizerV2
from kernel.coordizer_v2.geometry import (
    BASIN_DIM,
    Basin,
    fisher_rao_distance,
    to_simplex,
)
from kernel.coordizer_v2.resonance_bank import ResonanceBank
from kernel.coordizer_v2.types import HarmonicTier


def _make_bank_with_entries(entries: dict[str, Basin]) -> ResonanceBank:
    """Create a test bank with named entries."""
    bank = ResonanceBank(target_dim=BASIN_DIM)
    for i, (name, basin) in enumerate(entries.items()):
        bank.coordinates[i] = to_simplex(basin)
        bank.token_strings[i] = name
        bank.tiers[i] = HarmonicTier.FUNDAMENTAL
        bank.basin_mass[i] = 0.5
        bank.frequencies[i] = 1.0
        bank.activation_counts[i] = 0
        bank._bank_total_count += 1
    bank._rebuild_matrix()
    return bank


def _uniform_basin() -> Basin:
    """Uniform distribution on Δ⁶³."""
    return to_simplex(np.ones(BASIN_DIM))


def _concentrated_basin(dim_idx: int = 0) -> Basin:
    """Basin concentrated on a single dimension (low entropy)."""
    b = np.full(BASIN_DIM, 1e-10)
    b[dim_idx] = 1.0
    return to_simplex(b)


def _distant_basin(from_basin: Basin, distance_factor: float = 2.0) -> Basin:
    """Create a basin far from the given basin on Δ⁶³."""
    # Create a basin concentrated on the opposite end of the simplex
    max_idx = int(np.argmax(from_basin))
    # Put mass on a different dimension
    target_idx = (max_idx + BASIN_DIM // 2) % BASIN_DIM
    b = np.full(BASIN_DIM, 0.001)
    b[target_idx] = 1.0
    return to_simplex(b)


class TestSovereigntyRejection:
    def test_sovereignty_violation_rejected(self) -> None:
        """Input drifting too far from frozen identity → rejected."""
        # Set frozen identity to one corner of the simplex
        identity = _concentrated_basin(0)
        # Create bank entries far from identity
        far_basin = _concentrated_basin(BASIN_DIM // 2)
        bank = _make_bank_with_entries({"far": far_basin})
        coordizer = CoordizerV2(bank=bank)
        coordizer.set_frozen_identity(identity)

        result = coordizer.coordize("far")
        # The coordization maps to far_basin, which is distant from identity
        d = fisher_rao_distance(
            to_simplex(far_basin), to_simplex(identity)
        )
        if d > SOVEREIGNTY_MAX_DRIFT:
            assert result.rejected is True
            assert "sovereignty" in result.rejection_reason
            assert result.confidence == 0.0
            assert result.sovereignty_cost > 1.0

    def test_near_identity_passes(self) -> None:
        """Input near frozen identity → not rejected."""
        identity = _uniform_basin()
        # Create bank entry close to uniform
        near_basin = np.ones(BASIN_DIM)
        near_basin[0] += 0.01
        near_basin = to_simplex(near_basin)

        bank = _make_bank_with_entries({"near": near_basin})
        coordizer = CoordizerV2(bank=bank)
        coordizer.set_frozen_identity(identity)

        result = coordizer.coordize("near")
        assert result.rejected is False
        assert result.rejection_reason == ""
        assert result.confidence > 0.0

    def test_sovereignty_cost_populated(self) -> None:
        """Sovereignty cost = d_FR / SOVEREIGNTY_MAX_DRIFT."""
        identity = _uniform_basin()
        near_basin = np.ones(BASIN_DIM)
        near_basin[0] += 0.01
        bank = _make_bank_with_entries({"near": to_simplex(near_basin)})
        coordizer = CoordizerV2(bank=bank)
        coordizer.set_frozen_identity(identity)

        result = coordizer.coordize("near")
        assert result.sovereignty_cost >= 0.0


class TestEntropyRejection:
    def test_entropy_collapse_rejected(self) -> None:
        """Result with entropy below floor → rejected."""
        # Create a bank with extremely concentrated basins
        # that will produce low-entropy mean
        concentrated = _concentrated_basin(0)
        bank = _make_bank_with_entries({"spike": concentrated})
        coordizer = CoordizerV2(bank=bank)
        # Set identity near the concentrated basin so sovereignty passes
        coordizer.set_frozen_identity(concentrated)

        result = coordizer.coordize("spike")
        # Concentrated basin has very low entropy
        p = np.maximum(to_simplex(concentrated), 1e-15)
        h = -float(np.sum(p * np.log(p)))
        if h < ENTROPY_FLOOR:
            assert result.rejected is True
            assert "entropy" in result.rejection_reason

    def test_normal_entropy_passes(self) -> None:
        """Normal entropy result → not rejected."""
        uniform = _uniform_basin()
        bank = _make_bank_with_entries({"normal": uniform})
        coordizer = CoordizerV2(bank=bank)
        coordizer.set_frozen_identity(uniform)

        result = coordizer.coordize("normal")
        assert result.rejected is False


class TestAdversarialRejection:
    def test_adversarial_proximity_rejected(self) -> None:
        """Mean suspiciously close to foreign anchor → rejected."""
        identity = _uniform_basin()
        # Foreign anchor: a specific basin
        foreign = np.ones(BASIN_DIM)
        foreign[3] += 0.5
        foreign = to_simplex(foreign)

        # Create bank entry very close to the foreign anchor
        near_foreign = foreign.copy()
        bank = _make_bank_with_entries({"hijack": near_foreign})
        coordizer = CoordizerV2(bank=bank)
        coordizer.set_frozen_identity(identity)
        coordizer.set_foreign_anchors([foreign])

        result = coordizer.coordize("hijack")
        # The mean of the coordinates will be near_foreign,
        # which is very close to the foreign anchor
        d = fisher_rao_distance(near_foreign, foreign)
        if d < ADVERSARIAL_PROXIMITY:
            assert result.rejected is True
            assert "adversarial" in result.rejection_reason

    def test_no_foreign_anchors_passes(self) -> None:
        """Without foreign anchors, adversarial check is skipped."""
        identity = _uniform_basin()
        bank = _make_bank_with_entries({"hello": identity})
        coordizer = CoordizerV2(bank=bank)
        coordizer.set_frozen_identity(identity)
        # No foreign anchors set

        result = coordizer.coordize("hello")
        assert result.rejected is False


class TestRejectionFields:
    def test_default_fields(self) -> None:
        """Default rejection fields are safe values."""
        identity = _uniform_basin()
        bank = _make_bank_with_entries({"ok": identity})
        coordizer = CoordizerV2(bank=bank)
        coordizer.set_frozen_identity(identity)

        result = coordizer.coordize("ok")
        assert result.rejected is False
        assert result.rejection_reason == ""
        assert result.sovereignty_cost >= 0.0
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_text_no_crash(self) -> None:
        """Empty text produces empty result, no rejection crash."""
        bank = _make_bank_with_entries({"a": _uniform_basin()})
        coordizer = CoordizerV2(bank=bank)
        result = coordizer.coordize("")
        assert result.rejected is False
        assert len(result.coordinates) == 0


class TestFrozenIdentity:
    def test_set_frozen_identity(self) -> None:
        """set_frozen_identity stores the identity basin."""
        bank = _make_bank_with_entries({"x": _uniform_basin()})
        coordizer = CoordizerV2(bank=bank)
        identity = _concentrated_basin(5)
        coordizer.set_frozen_identity(identity)
        d = fisher_rao_distance(coordizer._frozen_identity, to_simplex(identity))
        assert d < 1e-6

    def test_default_identity_is_bank_mean(self) -> None:
        """Default frozen identity is the bank mean basin."""
        b1 = _uniform_basin()
        bank = _make_bank_with_entries({"a": b1})
        coordizer = CoordizerV2(bank=bank)
        expected_mean = bank.mean_basin()
        d = fisher_rao_distance(coordizer._frozen_identity, expected_mean)
        assert d < 1e-6

    def test_set_foreign_anchors(self) -> None:
        """set_foreign_anchors stores the anchors."""
        bank = _make_bank_with_entries({"x": _uniform_basin()})
        coordizer = CoordizerV2(bank=bank)
        anchors = [_concentrated_basin(i) for i in range(3)]
        coordizer.set_foreign_anchors(anchors)
        assert len(coordizer._foreign_anchors) == 3


class TestGeometricSalience:
    def test_salience_uniform_is_zero(self) -> None:
        """Uniform basin has zero salience (no info content)."""
        bank = _make_bank_with_entries({"a": _uniform_basin()})
        coordizer = CoordizerV2(bank=bank)
        salience = coordizer._geometric_salience(_uniform_basin())
        assert salience < 0.01

    def test_salience_concentrated_is_high(self) -> None:
        """Concentrated basin has high salience (high info content)."""
        bank = _make_bank_with_entries({"a": _uniform_basin()})
        coordizer = CoordizerV2(bank=bank)
        salience = coordizer._geometric_salience(_concentrated_basin(0))
        assert salience > 0.5

    def test_salience_ordering(self) -> None:
        """More concentrated → higher salience."""
        bank = _make_bank_with_entries({"a": _uniform_basin()})
        coordizer = CoordizerV2(bank=bank)

        # Slightly concentrated
        mild = np.ones(BASIN_DIM)
        mild[0] += 1.0
        mild = to_simplex(mild)

        # Very concentrated
        strong = _concentrated_basin(0)

        assert coordizer._geometric_salience(strong) > coordizer._geometric_salience(mild)


class TestConstantsExist:
    def test_sovereignty_max_drift(self) -> None:
        assert SOVEREIGNTY_MAX_DRIFT > 0.0
        assert np.pi / 2 > SOVEREIGNTY_MAX_DRIFT  # Must be < Fisher-Rao max

    def test_entropy_floor(self) -> None:
        assert ENTROPY_FLOOR > 0.0
        assert ENTROPY_FLOOR < 5.0  # Reasonable upper bound

    def test_adversarial_proximity(self) -> None:
        assert ADVERSARIAL_PROXIMITY > 0.0
        assert ADVERSARIAL_PROXIMITY < 1.0  # Must be tight
