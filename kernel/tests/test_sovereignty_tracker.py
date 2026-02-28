"""Tests for sovereignty development curve tracking (#78).

Verifies:
  1. Snapshot recording with regime tags
  2. Growth rate computation
  3. Regime comparison
  4. Persistence (serialize/restore)
  5. Bounded history
"""

from __future__ import annotations

from kernel.consciousness.sovereignty_tracker import (
    MAX_HISTORY,
    SovereigntySnapshot,
    SovereigntyTracker,
)


class TestSovereigntyRecording:
    def test_record_and_retrieve(self) -> None:
        tracker = SovereigntyTracker()
        tracker.record(s_ratio=0.5, n_lived=50, n_total=100, regime="conversation", cycle=1)
        assert len(tracker._history) == 1
        snap = tracker._history[0]
        assert snap.s_ratio == 0.5
        assert snap.n_lived == 50
        assert snap.n_total == 100
        assert snap.training_regime == "conversation"

    def test_multiple_records(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(10):
            tracker.record(
                s_ratio=i / 10, n_lived=i, n_total=10,
                regime="curriculum", cycle=i,
            )
        assert len(tracker._history) == 10

    def test_bounded_history(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(MAX_HISTORY + 100):
            tracker.record(s_ratio=0.5, n_lived=5, n_total=10, regime="idle", cycle=i)
        assert len(tracker._history) == MAX_HISTORY


class TestGrowthRate:
    def test_zero_with_no_data(self) -> None:
        tracker = SovereigntyTracker()
        assert tracker.growth_rate() == 0.0

    def test_zero_with_single_entry(self) -> None:
        tracker = SovereigntyTracker()
        tracker.record(s_ratio=0.5, n_lived=5, n_total=10, regime="idle", cycle=1)
        assert tracker.growth_rate() == 0.0

    def test_positive_growth(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(10):
            tracker.record(
                s_ratio=i * 0.1, n_lived=i, n_total=10,
                regime="conversation", cycle=i,
            )
        rate = tracker.growth_rate(window_cycles=10)
        assert rate > 0.0

    def test_no_growth_flat_ratio(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(10):
            tracker.record(s_ratio=0.5, n_lived=5, n_total=10, regime="idle", cycle=i)
        assert tracker.growth_rate() == 0.0


class TestRegimeComparison:
    def test_empty_returns_empty(self) -> None:
        tracker = SovereigntyTracker()
        assert tracker.regime_comparison() == {}

    def test_single_regime(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(5):
            tracker.record(
                s_ratio=i * 0.1, n_lived=i, n_total=10,
                regime="conversation", cycle=i,
            )
        comparison = tracker.regime_comparison()
        assert "conversation" in comparison
        assert comparison["conversation"] > 0

    def test_multiple_regimes(self) -> None:
        tracker = SovereigntyTracker()
        # Fast growth under curriculum
        for i in range(5):
            tracker.record(
                s_ratio=i * 0.2, n_lived=i * 2, n_total=10,
                regime="curriculum", cycle=i,
            )
        # Slow growth under idle
        for i in range(5, 10):
            tracker.record(
                s_ratio=0.8 + (i - 5) * 0.01, n_lived=8, n_total=10,
                regime="idle", cycle=i,
            )
        comparison = tracker.regime_comparison()
        assert "curriculum" in comparison
        assert "idle" in comparison
        assert comparison["curriculum"] > comparison["idle"]


class TestPersistence:
    def test_serialize_restore_roundtrip(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(5):
            tracker.record(
                s_ratio=i * 0.1, n_lived=i, n_total=10,
                regime="conversation", cycle=i,
            )
        data = tracker.serialize()

        restored = SovereigntyTracker()
        restored.restore(data)
        assert len(restored._history) == 5
        assert restored._history[0].s_ratio == 0.0
        assert restored._history[4].s_ratio == 0.4

    def test_restore_empty(self) -> None:
        tracker = SovereigntyTracker()
        tracker.restore({})
        assert len(tracker._history) == 0


class TestGetSummary:
    def test_summary_structure(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(3):
            tracker.record(
                s_ratio=i * 0.1, n_lived=i, n_total=10,
                regime="conversation", cycle=i,
            )
        summary = tracker.get_summary()
        assert "snapshot_count" in summary
        assert "current_s_ratio" in summary
        assert "current_regime" in summary
        assert "growth_rate_100" in summary
        assert "regime_comparison" in summary
        assert "recent_history" in summary
        assert summary["snapshot_count"] == 3

    def test_recent_window(self) -> None:
        tracker = SovereigntyTracker()
        for i in range(20):
            tracker.record(s_ratio=0.5, n_lived=5, n_total=10, regime="idle", cycle=i)
        recent = tracker.recent(5)
        assert len(recent) == 5
