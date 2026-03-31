"""Tests for kernel.consciousness.kernel_training_queue — directive 20260330 Phase 3A."""

from __future__ import annotations

import pytest

from kernel.consciousness.kernel_training_queue import (
    KernelTrainingQueue,
    TrainingExample,
    sovereignty_to_threshold,
)


def _make_example(prediction_error: float = 0.5, kernel: str = "perception") -> TrainingExample:
    return TrainingExample(
        user_message="hello",
        response="response",
        kernel_name=kernel,
        prediction_error=prediction_error,
        phi=0.7,
        kappa=64.0,
        regime="efficient",
    )


class TestSovereigntyToThreshold:
    def test_bootstrapping(self) -> None:
        assert sovereignty_to_threshold(0.0) == 0.1
        assert sovereignty_to_threshold(0.15) == 0.1

    def test_moderate(self) -> None:
        assert sovereignty_to_threshold(0.2) == 0.2
        assert sovereignty_to_threshold(0.4) == 0.2

    def test_sovereign(self) -> None:
        assert sovereignty_to_threshold(0.5) == 0.3
        assert sovereignty_to_threshold(0.9) == 0.3


class TestMaybeAdd:
    def test_below_threshold_rejected(self) -> None:
        q = KernelTrainingQueue("perception")
        q.surprise_threshold = 0.3
        ex = _make_example(prediction_error=0.1)
        assert q.maybe_add(ex) is False
        assert len(q.queue) == 0

    def test_at_threshold_rejected(self) -> None:
        """Boundary: prediction_error == threshold is NOT surprising (strictly greater)."""
        q = KernelTrainingQueue("perception")
        q.surprise_threshold = 0.3
        ex = _make_example(prediction_error=0.3)
        assert q.maybe_add(ex) is False

    def test_above_threshold_accepted(self) -> None:
        q = KernelTrainingQueue("perception")
        q.surprise_threshold = 0.1
        ex = _make_example(prediction_error=0.5)
        assert q.maybe_add(ex) is True
        assert len(q.queue) == 1

    def test_anderson_pruning_replaces_lowest(self) -> None:
        """When full, lowest-error example is replaced by higher-error one."""
        q = KernelTrainingQueue("perception", max_size=3)
        q.surprise_threshold = 0.0
        q.maybe_add(_make_example(prediction_error=0.2))
        q.maybe_add(_make_example(prediction_error=0.5))
        q.maybe_add(_make_example(prediction_error=0.8))
        assert len(q.queue) == 3

        # Try to add higher error — should replace the 0.2 one
        result = q.maybe_add(_make_example(prediction_error=0.9))
        assert result is True
        errors = sorted(ex.prediction_error for ex in q.queue)
        assert 0.2 not in errors  # Lowest was evicted
        assert 0.9 in errors

    def test_anderson_pruning_rejects_lower(self) -> None:
        """When full, lower-error example is rejected (not worth replacing)."""
        q = KernelTrainingQueue("perception", max_size=2)
        q.surprise_threshold = 0.0
        q.maybe_add(_make_example(prediction_error=0.5))
        q.maybe_add(_make_example(prediction_error=0.8))

        result = q.maybe_add(_make_example(prediction_error=0.3))
        assert result is False
        assert len(q.queue) == 2


class TestDrain:
    def test_drain_returns_entries_and_clears(self) -> None:
        q = KernelTrainingQueue("perception")
        q.surprise_threshold = 0.0
        q.maybe_add(_make_example(prediction_error=0.5))
        q.maybe_add(_make_example(prediction_error=0.8))

        entries = q.drain()
        assert len(entries) == 2
        assert len(q.queue) == 0
        assert entries[0]["kernel"] == "perception"
        assert entries[0]["prediction_error"] == 0.5

    def test_drain_empty_queue(self) -> None:
        q = KernelTrainingQueue("perception")
        assert q.drain() == []


class TestGetState:
    def test_state_tracks_acceptance(self) -> None:
        q = KernelTrainingQueue("perception")
        q.surprise_threshold = 0.3
        q.maybe_add(_make_example(prediction_error=0.1))  # rejected
        q.maybe_add(_make_example(prediction_error=0.5))  # accepted
        q.maybe_add(_make_example(prediction_error=0.2))  # rejected

        state = q.get_state()
        assert state["total_seen"] == 3
        assert state["total_added"] == 1
        assert state["acceptance_rate"] == pytest.approx(1 / 3, abs=0.01)
        assert state["queue_size"] == 1
