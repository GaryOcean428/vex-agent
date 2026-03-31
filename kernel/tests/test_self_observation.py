"""Tests for kernel.consciousness.self_observation — §43.2 Loop 1."""

from __future__ import annotations

import numpy as np
import pytest

from kernel.consciousness.self_observation import SelfObservation, SelfObservationTracker


@pytest.fixture
def tracker() -> SelfObservationTracker:
    return SelfObservationTracker("perception", window=5)


def _random_basin(dim: int = 64) -> np.ndarray:
    """Generate a random simplex point."""
    v = np.random.dirichlet(np.ones(dim))
    return v


class TestConfidence:
    def test_pure_geometric(self, tracker: SelfObservationTracker) -> None:
        obs = tracker.observe(geometric_resonances=10, llm_expanded=False, sovereignty_ratio=0.5)
        assert obs.confidence_score == 1.0

    def test_expanded_with_resonances(self, tracker: SelfObservationTracker) -> None:
        obs = tracker.observe(geometric_resonances=5, llm_expanded=True, sovereignty_ratio=0.5)
        assert obs.confidence_score == 0.5

    def test_pure_llm(self, tracker: SelfObservationTracker) -> None:
        obs = tracker.observe(geometric_resonances=0, llm_expanded=True, sovereignty_ratio=0.5)
        assert obs.confidence_score == 0.0


class TestSovereignty:
    def test_clipped_low(self, tracker: SelfObservationTracker) -> None:
        obs = tracker.observe(geometric_resonances=1, llm_expanded=False, sovereignty_ratio=-0.5)
        assert obs.sovereignty_score == 0.0

    def test_clipped_high(self, tracker: SelfObservationTracker) -> None:
        obs = tracker.observe(geometric_resonances=1, llm_expanded=False, sovereignty_ratio=1.5)
        assert obs.sovereignty_score == 1.0

    def test_passthrough(self, tracker: SelfObservationTracker) -> None:
        obs = tracker.observe(geometric_resonances=1, llm_expanded=False, sovereignty_ratio=0.73)
        assert abs(obs.sovereignty_score - 0.73) < 0.01


class TestRepetition:
    def test_no_basin_no_repetition(self, tracker: SelfObservationTracker) -> None:
        obs = tracker.observe(geometric_resonances=1, llm_expanded=False, sovereignty_ratio=0.5)
        assert obs.repetition_score == 0.0

    def test_first_basin_no_repetition(self, tracker: SelfObservationTracker) -> None:
        basin = _random_basin()
        obs = tracker.observe(
            geometric_resonances=1,
            llm_expanded=False,
            sovereignty_ratio=0.5,
            activation_basin=basin,
        )
        assert obs.repetition_score == 0.0
        assert tracker.get_state()["history_depth"] == 1

    def test_identical_basin_high_repetition(self, tracker: SelfObservationTracker) -> None:
        basin = _random_basin()
        # First: establishes history
        tracker.observe(
            geometric_resonances=1,
            llm_expanded=False,
            sovereignty_ratio=0.5,
            activation_basin=basin,
        )
        # Second: identical basin → high repetition
        obs = tracker.observe(
            geometric_resonances=1,
            llm_expanded=False,
            sovereignty_ratio=0.5,
            activation_basin=basin,
        )
        assert obs.repetition_score > 0.9  # Nearly identical → near 1.0

    def test_different_basin_low_repetition(self, tracker: SelfObservationTracker) -> None:
        # Two very different basins
        basin1 = np.zeros(64)
        basin1[0] = 1.0  # Concentrated at dim 0
        basin2 = np.zeros(64)
        basin2[32] = 1.0  # Concentrated at dim 32
        tracker.observe(
            geometric_resonances=1,
            llm_expanded=False,
            sovereignty_ratio=0.5,
            activation_basin=basin1,
        )
        obs = tracker.observe(
            geometric_resonances=1,
            llm_expanded=False,
            sovereignty_ratio=0.5,
            activation_basin=basin2,
        )
        assert obs.repetition_score < 0.5  # Very different → low repetition

    def test_history_window(self, tracker: SelfObservationTracker) -> None:
        """Window of 5 — after 6 basins, oldest is evicted."""
        for _ in range(6):
            tracker.observe(
                geometric_resonances=1,
                llm_expanded=False,
                sovereignty_ratio=0.5,
                activation_basin=_random_basin(),
            )
        assert tracker.get_state()["history_depth"] == 5  # Capped at window


class TestSummary:
    def test_summary_format(self) -> None:
        obs = SelfObservation(repetition_score=0.12, sovereignty_score=0.45, confidence_score=0.78)
        s = obs.summary()
        assert "rep=0.12" in s
        assert "sov=0.45" in s
        assert "conf=0.78" in s
