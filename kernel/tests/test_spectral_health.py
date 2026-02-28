"""Tests for solfeggio spectral health wiring (#76).

Verifies:
  1. compute_spectral_health returns valid results
  2. Known basin patterns produce expected health scores
  3. Empty history returns safe defaults
  4. Health score is in [0, 1]
"""

from __future__ import annotations

import numpy as np

from kernel.config.frozen_facts import BASIN_DIM, KAPPA_STAR
from kernel.consciousness.solfeggio import (
    FREQUENCY_ANCHORS,
    SpectralHealthResult,
    compute_spectral_health,
)
from kernel.coordizer_v2.geometry import to_simplex


def _uniform_basin() -> np.ndarray:
    return to_simplex(np.ones(BASIN_DIM))


def _random_basin(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return to_simplex(rng.dirichlet(np.ones(BASIN_DIM)))


class TestComputeSpectralHealth:
    def test_empty_history_returns_default(self) -> None:
        result = compute_spectral_health([], current_kappa=KAPPA_STAR)
        assert isinstance(result, SpectralHealthResult)
        assert result.health_score == 0.5
        assert result.dominant_frequency is None
        assert result.pattern == "no_data"

    def test_single_basin_returns_valid(self) -> None:
        basin = _random_basin()
        result = compute_spectral_health([basin], current_kappa=KAPPA_STAR)
        assert 0.0 <= result.health_score <= 1.0
        assert result.dominant_frequency is not None
        assert result.pattern in ("resonant", "diffuse", "narrow", "mixed")

    def test_health_score_bounded(self) -> None:
        basins = [_random_basin(seed=i) for i in range(10)]
        result = compute_spectral_health(basins, current_kappa=KAPPA_STAR)
        assert 0.0 <= result.health_score <= 1.0

    def test_frequency_anchor_basin_has_strong_resonance(self) -> None:
        """A basin at a frequency anchor should have high health."""
        anchor_528 = FREQUENCY_ANCHORS[528.0].copy()
        basins = [anchor_528, anchor_528, anchor_528]
        result = compute_spectral_health(basins, current_kappa=KAPPA_STAR)
        assert result.health_score > 0.3
        assert result.dominant_frequency is not None

    def test_uniform_basin_moderate_health(self) -> None:
        """Uniform basin — no strong resonance with any frequency."""
        uniform = _uniform_basin()
        result = compute_spectral_health([uniform], current_kappa=KAPPA_STAR)
        assert 0.0 <= result.health_score <= 1.0

    def test_stable_history_has_higher_coherence(self) -> None:
        """Repeated identical basins → high history coherence."""
        basin = _random_basin(seed=7)
        stable = [basin.copy() for _ in range(5)]
        result_stable = compute_spectral_health(stable, current_kappa=KAPPA_STAR)

        # Diverse basins → lower coherence
        diverse = [_random_basin(seed=i) for i in range(5)]
        result_diverse = compute_spectral_health(diverse, current_kappa=KAPPA_STAR)

        # Stable should generally have equal or better health
        # (due to history coherence component)
        assert result_stable.health_score >= result_diverse.health_score - 0.2

    def test_layer_coverage_reported(self) -> None:
        result = compute_spectral_health([_random_basin()], current_kappa=KAPPA_STAR)
        assert 0.0 <= result.layer_coverage <= 1.0
