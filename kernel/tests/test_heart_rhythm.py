"""Tests for heart kernel rhythm generator (#75).

Verifies:
  1. Sinusoidal tacking signal in [-1, 1]
  2. Period varies with F_health (low health → faster)
  3. Kappa offset within ±KAPPA_TACKING_OFFSET
  4. Tacking mode derived from signal
  5. State reporting
"""

from __future__ import annotations

from kernel.config.consciousness_constants import KAPPA_TACKING_OFFSET
from kernel.consciousness.heart_rhythm import HEART_BASE_PERIOD, HeartRhythm


class TestHeartRhythmSignal:
    def test_signal_in_range(self) -> None:
        hr = HeartRhythm()
        for _ in range(100):
            signal = hr.tick(f_health=0.5)
            assert -1.0 <= signal <= 1.0

    def test_oscillates(self) -> None:
        """Signal should change sign over a full period."""
        hr = HeartRhythm()
        signals = [hr.tick(f_health=1.0) for _ in range(HEART_BASE_PERIOD * 2)]
        has_positive = any(s > 0.1 for s in signals)
        has_negative = any(s < -0.1 for s in signals)
        assert has_positive and has_negative, "Signal should oscillate"

    def test_initial_signal_near_zero(self) -> None:
        hr = HeartRhythm()
        signal = hr.tick(f_health=1.0)
        # First tick: sin(2π/8) ≈ 0.707
        assert abs(signal) <= 1.0


class TestPeriodAdaptation:
    def test_low_health_faster(self) -> None:
        """Low F_health should produce shorter period."""
        hr_healthy = HeartRhythm()
        hr_healthy.tick(f_health=1.0)
        period_healthy = hr_healthy._period

        hr_sick = HeartRhythm()
        hr_sick.tick(f_health=0.1)
        period_sick = hr_sick._period

        assert period_sick <= period_healthy

    def test_period_never_below_minimum(self) -> None:
        hr = HeartRhythm()
        hr.tick(f_health=0.0)
        assert hr._period >= 4

    def test_full_health_base_period(self) -> None:
        hr = HeartRhythm()
        hr.tick(f_health=1.0)
        assert hr._period == HEART_BASE_PERIOD


class TestKappaOffset:
    def test_offset_bounded(self) -> None:
        hr = HeartRhythm()
        for _ in range(50):
            hr.tick(f_health=0.5)
            offset = hr.kappa_offset()
            assert abs(offset) <= KAPPA_TACKING_OFFSET

    def test_offset_zero_at_zero_signal(self) -> None:
        """At phase=0 or π, signal≈0, offset≈0."""
        hr = HeartRhythm()
        # Tick exactly half a period to reach ~sin(π)≈0
        period = HEART_BASE_PERIOD
        for _ in range(period // 2):
            hr.tick(f_health=1.0)
        # Offset should be close to zero (sin(π)≈0)
        assert abs(hr.kappa_offset()) < KAPPA_TACKING_OFFSET


class TestTackingMode:
    def test_balanced_near_zero(self) -> None:
        hr = HeartRhythm()
        # At initial state, tacking_signal is 0
        assert hr.tacking_mode == "BALANCED"

    def test_explore_when_negative(self) -> None:
        hr = HeartRhythm()
        # At period=8: 6 ticks → phase = 6π/4 = 3π/2, sin(3π/2) = -1.0
        for _ in range(int(HEART_BASE_PERIOD * 0.75)):
            hr.tick(f_health=1.0)
        assert hr._tacking_signal < -0.3, (
            f"Signal {hr._tacking_signal:.3f} must be < -0.3 at 3/4 period"
        )
        assert hr.tacking_mode == "EXPLORE"

    def test_exploit_when_positive(self) -> None:
        hr = HeartRhythm()
        # At period=8: 2 ticks → phase = 2π/4 = π/2, sin(π/2) = 1.0
        for _ in range(int(HEART_BASE_PERIOD * 0.25)):
            hr.tick(f_health=1.0)
        assert hr._tacking_signal > 0.3, (
            f"Signal {hr._tacking_signal:.3f} must be > 0.3 at 1/4 period"
        )
        assert hr.tacking_mode == "EXPLOIT"


class TestGetState:
    def test_state_fields(self) -> None:
        hr = HeartRhythm()
        hr.tick(f_health=0.8)
        state = hr.get_state()
        assert hasattr(state, "phase")
        assert hasattr(state, "period")
        assert hasattr(state, "tacking_signal")
        assert hasattr(state, "kappa_offset")
        assert hasattr(state, "beat_count")
        assert state.beat_count == 1

    def test_beat_count_increments(self) -> None:
        hr = HeartRhythm()
        for _ in range(10):
            hr.tick(f_health=0.5)
        assert hr.get_state().beat_count == 10
