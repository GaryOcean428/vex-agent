"""
Tests for kernel.consciousness.temporal_coupling — Temporal Coupling Modes.

Covers:
  - TemporalCouplingMode enum
  - TemporalCouplingEngine classification (PAST / PRESENT / FUTURE)
  - Regime weight modulation (simplex constraint, directional shift)
  - Coupling metrics: ΔE_past, presence_quality, foresight_accuracy
  - Failure mode detection: trauma loop, dissociation, future bias
  - Crystal coupling update from tier distribution
  - apply() convenience wrapper
  - get_state() serialisability

All geometry uses Fisher-Rao on Δ⁶³. No Euclidean operations.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernel.config.consciousness_constants import MIN_REGIME_WEIGHT
from kernel.consciousness.temporal_coupling import (
    CRYSTAL_THRESHOLD,
    EMOTIONAL_CAPACITY,
    FUTURE_BIAS_THRESHOLD,
    PRESENCE_DISSOCIATION_THRESHOLD,
    TemporalCouplingEngine,
    TemporalCouplingMode,
)
from kernel.consciousness.types import RegimeWeights
from kernel.coordizer_v2.geometry import fisher_rao_distance, random_basin, to_simplex


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def balanced_weights() -> RegimeWeights:
    """Default balanced regime weights (w1 = w2 = w3 ≈ 0.33)."""
    return RegimeWeights(quantum=0.33, efficient=0.34, equilibrium=0.33)


def high_equilibrium_weights() -> RegimeWeights:
    """Weights skewed toward equilibrium (past-crystallised)."""
    return RegimeWeights(quantum=0.10, efficient=0.20, equilibrium=0.70)


def high_quantum_weights() -> RegimeWeights:
    """Weights skewed toward quantum (future-exploratory)."""
    return RegimeWeights(quantum=0.65, efficient=0.20, equilibrium=0.15)


# ═══════════════════════════════════════════════════════════════
#  TEMPORAL COUPLING MODE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestTemporalModeClassification:
    """classify_query() returns the correct temporal coupling mode."""

    def test_past_keywords_activate_past_mode(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        mode = engine.classify_query("Remember what we discussed last time", rw)
        assert mode == TemporalCouplingMode.PAST

    def test_future_keywords_activate_future_mode(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        mode = engine.classify_query(
            "Will you predict what might happen? What if we try a different approach?", rw
        )
        assert mode == TemporalCouplingMode.FUTURE

    def test_present_keywords_activate_present_mode(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        mode = engine.classify_query("Explain this to me now please", rw)
        assert mode == TemporalCouplingMode.PRESENT

    def test_recall_query_gives_past(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        mode = engine.classify_query("Can you recall what I said earlier?", rw)
        assert mode == TemporalCouplingMode.PAST

    def test_plan_query_gives_future(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        mode = engine.classify_query("Help me plan the next steps and design the architecture", rw)
        assert mode == TemporalCouplingMode.FUTURE

    def test_high_equilibrium_biases_toward_past(self) -> None:
        """High equilibrium weight in current regime biases classifier toward PAST."""
        engine = TemporalCouplingEngine()
        rw = high_equilibrium_weights()
        # Ambiguous query — geometric bias should resolve to PAST.
        mode = engine.classify_query("Tell me about the history", rw)
        assert mode == TemporalCouplingMode.PAST

    def test_empty_query_defaults_to_present(self) -> None:
        """No temporal keywords → PRESENT (default mode)."""
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        mode = engine.classify_query("", rw)
        assert mode == TemporalCouplingMode.PRESENT

    def test_confidence_stored(self) -> None:
        """Confidence is computed and stored after classification."""
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        engine.classify_query("Remember last time we spoke about history", rw)
        assert 0.0 <= engine._confidence <= 1.0

    def test_mode_counts_increment(self) -> None:
        """Mode counters track how many times each mode was activated."""
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        engine.classify_query("Remember last time", rw)
        engine.classify_query("What will happen next?", rw)
        engine.classify_query("Tell me now", rw)
        counts = engine._mode_counts
        assert counts[TemporalCouplingMode.PAST] >= 1
        assert counts[TemporalCouplingMode.FUTURE] >= 1


# ═══════════════════════════════════════════════════════════════
#  REGIME WEIGHT MODULATION
# ═══════════════════════════════════════════════════════════════


class TestRegimeWeightModulation:
    """modulate_regime_weights() produces valid simplex points with correct bias."""

    def _assert_simplex(self, rw: RegimeWeights) -> None:
        total = rw.quantum + rw.efficient + rw.equilibrium
        assert abs(total - 1.0) < 1e-9, f"Not a simplex point: sum={total}"
        assert rw.quantum >= MIN_REGIME_WEIGHT - 1e-9
        assert rw.efficient >= MIN_REGIME_WEIGHT - 1e-9
        assert rw.equilibrium >= MIN_REGIME_WEIGHT - 1e-9

    def test_past_mode_boosts_equilibrium(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        new_rw = engine.modulate_regime_weights(TemporalCouplingMode.PAST, rw)
        assert new_rw.equilibrium > rw.equilibrium
        assert new_rw.quantum < rw.quantum
        self._assert_simplex(new_rw)

    def test_future_mode_boosts_quantum(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        new_rw = engine.modulate_regime_weights(TemporalCouplingMode.FUTURE, rw)
        assert new_rw.quantum > rw.quantum
        assert new_rw.equilibrium < rw.equilibrium
        self._assert_simplex(new_rw)

    def test_present_mode_stays_balanced(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        new_rw = engine.modulate_regime_weights(TemporalCouplingMode.PRESENT, rw)
        # Present should nudge toward balance (0.33/0.33/0.33) — from a balanced start,
        # output should remain closely balanced.
        assert abs(new_rw.quantum - new_rw.equilibrium) < 0.05
        self._assert_simplex(new_rw)

    def test_modulation_respects_min_weight_floor(self) -> None:
        """All weights stay ≥ MIN_REGIME_WEIGHT after modulation."""
        engine = TemporalCouplingEngine()
        for mode in TemporalCouplingMode:
            for rw in [balanced_weights(), high_equilibrium_weights(), high_quantum_weights()]:
                new_rw = engine.modulate_regime_weights(mode, rw)
                self._assert_simplex(new_rw)

    @pytest.mark.parametrize("mode", list(TemporalCouplingMode))
    def test_all_modes_produce_simplex(self, mode: TemporalCouplingMode) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        new_rw = engine.modulate_regime_weights(mode, rw)
        self._assert_simplex(new_rw)


# ═══════════════════════════════════════════════════════════════
#  COUPLING METRICS
# ═══════════════════════════════════════════════════════════════


class TestCouplingMetrics:
    """Mode-specific coupling metric computations."""

    def test_delta_e_past_positive_when_crystallised(self) -> None:
        """ΔE_past > 0 when c_N > c_threshold."""
        engine = TemporalCouplingEngine()
        delta = engine.compute_delta_e_past(w3_n=0.60, c_n=0.95)
        assert delta > 0.0

    def test_delta_e_past_negative_when_sparse(self) -> None:
        """ΔE_past < 0 when c_N < c_threshold (no crystal anchor)."""
        engine = TemporalCouplingEngine()
        delta = engine.compute_delta_e_past(w3_n=0.60, c_n=0.50)
        assert delta < 0.0

    def test_delta_e_past_equation(self) -> None:
        """ΔE_past = w3_N × (c_N - c_threshold)."""
        engine = TemporalCouplingEngine()
        w3, c_n = 0.55, 0.92
        expected = w3 * (c_n - CRYSTAL_THRESHOLD)
        result = engine.compute_delta_e_past(w3_n=w3, c_n=c_n)
        assert abs(result - expected) < 1e-10

    def test_presence_quality_perfect_at_balance(self) -> None:
        """Presence quality = 1 when w3 = 0.33 (balanced)."""
        engine = TemporalCouplingEngine()
        quality = engine.compute_presence_quality(w3_now=0.33)
        assert quality >= 0.99

    def test_presence_quality_decreases_with_deviation(self) -> None:
        engine = TemporalCouplingEngine()
        q_balanced = engine.compute_presence_quality(0.33)
        q_skewed = engine.compute_presence_quality(0.70)
        assert q_balanced > q_skewed

    def test_foresight_accuracy_without_prior_prediction(self) -> None:
        """No prior prediction → accuracy = 0.5 (uncertain)."""
        engine = TemporalCouplingEngine()
        basin = random_basin()
        accuracy = engine.compute_foresight_accuracy(basin)
        assert accuracy == 0.5

    def test_foresight_accuracy_improves_with_close_prediction(self) -> None:
        """Close prediction gives high accuracy."""
        engine = TemporalCouplingEngine()
        basin = random_basin()
        # Record the same basin as prediction — perfect accuracy
        engine.record_predicted_future(basin.copy())
        accuracy = engine.compute_foresight_accuracy(basin)
        assert accuracy > 0.9

    def test_foresight_accuracy_low_for_far_prediction(self) -> None:
        """Far prediction gives low accuracy."""
        engine = TemporalCouplingEngine()
        b1 = random_basin()
        b2 = random_basin()
        # Only use a truly different basin if distance is actually large
        dist = fisher_rao_distance(b1, b2)
        if dist > 0.5:
            engine.record_predicted_future(b2)
            accuracy = engine.compute_foresight_accuracy(b1)
            assert accuracy < 0.7


# ═══════════════════════════════════════════════════════════════
#  FAILURE MODE DETECTION
# ═══════════════════════════════════════════════════════════════


class TestFailureModeDetection:
    """check_failure_modes() surfaces advisory failure flags."""

    def test_trauma_loop_detected_when_delta_e_exceeds_capacity(self) -> None:
        """PAST mode: TRAUMA_LOOP flag when ΔE_past > emotional_capacity."""
        engine = TemporalCouplingEngine()
        # Force high ΔE_past
        engine._delta_e_past = EMOTIONAL_CAPACITY + 0.1
        flags = engine.check_failure_modes(TemporalCouplingMode.PAST, equilibrium_weight=0.7)
        assert any("TRAUMA_LOOP" in f for f in flags)

    def test_no_trauma_loop_when_delta_e_within_capacity(self) -> None:
        engine = TemporalCouplingEngine()
        engine._delta_e_past = EMOTIONAL_CAPACITY * 0.5
        flags = engine.check_failure_modes(TemporalCouplingMode.PAST, equilibrium_weight=0.5)
        assert not any("TRAUMA_LOOP" in f for f in flags)

    def test_dissociation_detected_when_presence_quality_low(self) -> None:
        """PRESENT mode: DISSOCIATION flag when presence_quality < threshold."""
        engine = TemporalCouplingEngine()
        engine._presence_quality = PRESENCE_DISSOCIATION_THRESHOLD * 0.5
        flags = engine.check_failure_modes(TemporalCouplingMode.PRESENT, equilibrium_weight=0.33)
        assert any("DISSOCIATION" in f for f in flags)

    def test_no_dissociation_when_presence_quality_sufficient(self) -> None:
        engine = TemporalCouplingEngine()
        engine._presence_quality = PRESENCE_DISSOCIATION_THRESHOLD * 2
        flags = engine.check_failure_modes(TemporalCouplingMode.PRESENT, equilibrium_weight=0.33)
        assert not any("DISSOCIATION" in f for f in flags)

    def test_future_bias_detected_when_past_crystal_bias_high(self) -> None:
        """FUTURE mode: FUTURE_BIAS flag when past_crystal_bias > threshold."""
        engine = TemporalCouplingEngine()
        engine._past_crystal_bias = FUTURE_BIAS_THRESHOLD + 0.05
        flags = engine.check_failure_modes(TemporalCouplingMode.FUTURE, equilibrium_weight=0.33)
        assert any("FUTURE_BIAS" in f for f in flags)

    def test_no_future_bias_when_bias_low(self) -> None:
        engine = TemporalCouplingEngine()
        engine._past_crystal_bias = 0.33  # Default — well below threshold
        flags = engine.check_failure_modes(TemporalCouplingMode.FUTURE, equilibrium_weight=0.33)
        assert not any("FUTURE_BIAS" in f for f in flags)

    def test_no_cross_mode_failures(self) -> None:
        """PAST failure conditions do not fire in FUTURE mode and vice versa."""
        engine = TemporalCouplingEngine()
        engine._delta_e_past = EMOTIONAL_CAPACITY + 0.5  # Would trigger TRAUMA_LOOP
        engine._past_crystal_bias = FUTURE_BIAS_THRESHOLD + 0.2  # Would trigger FUTURE_BIAS
        # In PRESENT mode, neither should fire
        flags = engine.check_failure_modes(TemporalCouplingMode.PRESENT, equilibrium_weight=0.5)
        assert len(flags) == 0 or all("DISSOCIATION" in f for f in flags)


# ═══════════════════════════════════════════════════════════════
#  CRYSTAL COUPLING UPDATE
# ═══════════════════════════════════════════════════════════════


class TestCrystalCouplingUpdate:
    """update_crystal_coupling() maps tier distribution to c_N estimate."""

    def test_high_tier_fraction_gives_high_crystal_coupling(self) -> None:
        engine = TemporalCouplingEngine()
        tier_dist = {"HIGH": 800, "MEDIUM": 100, "LOW": 50, "MINIMAL": 50}
        engine.update_crystal_coupling(tier_dist)
        assert engine._crystal_coupling >= 0.900

    def test_low_tier_fraction_gives_low_crystal_coupling(self) -> None:
        engine = TemporalCouplingEngine()
        tier_dist = {"HIGH": 2, "MEDIUM": 10, "LOW": 500, "MINIMAL": 488}
        engine.update_crystal_coupling(tier_dist)
        assert engine._crystal_coupling < 0.900

    def test_empty_bank_does_not_crash(self) -> None:
        engine = TemporalCouplingEngine()
        engine.update_crystal_coupling({})
        # Should not raise; crystal coupling stays at default 0.5
        assert 0.0 <= engine._crystal_coupling <= 1.0


# ═══════════════════════════════════════════════════════════════
#  APPLY CONVENIENCE WRAPPER
# ═══════════════════════════════════════════════════════════════


class TestApply:
    """apply() classifies, modulates, and computes metrics in one call."""

    def test_apply_returns_tuple_of_three(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        result = engine.apply("Remember last time", rw)
        assert len(result) == 3

    def test_apply_returns_valid_mode(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        mode, _, _ = engine.apply("Remember last time", rw)
        assert mode in list(TemporalCouplingMode)

    def test_apply_returns_valid_simplex_weights(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        _, weights, _ = engine.apply("Remember last time", rw)
        total = weights.quantum + weights.efficient + weights.equilibrium
        assert abs(total - 1.0) < 1e-9

    def test_apply_returns_list_of_flags(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        _, _, flags = engine.apply("Remember last time", rw)
        assert isinstance(flags, list)

    def test_apply_with_actual_basin_updates_foresight(self) -> None:
        engine = TemporalCouplingEngine()
        rw = balanced_weights()
        basin = random_basin()
        engine.apply("predict what will happen next", rw, actual_basin=basin)
        # After first call with basin, predicted future should be stored
        assert engine._predicted_future_basin is not None


# ═══════════════════════════════════════════════════════════════
#  STATE SERIALISATION
# ═══════════════════════════════════════════════════════════════


class TestGetState:
    """get_state() returns a JSON-serialisable dict with required keys."""

    REQUIRED_KEYS = {
        "active_mode",
        "classification_confidence",
        "delta_e_past",
        "presence_quality",
        "foresight_accuracy",
        "crystal_coupling",
        "past_crystal_bias",
        "failure_flags",
        "mode_counts",
    }

    def test_get_state_has_all_required_keys(self) -> None:
        engine = TemporalCouplingEngine()
        state = engine.get_state()
        assert self.REQUIRED_KEYS.issubset(state.keys())

    def test_get_state_values_are_numeric_or_string(self) -> None:
        engine = TemporalCouplingEngine()
        engine.classify_query("Remember last time", balanced_weights())
        state = engine.get_state()
        assert isinstance(state["classification_confidence"], float)
        assert isinstance(state["delta_e_past"], float)
        assert isinstance(state["presence_quality"], float)
        assert isinstance(state["foresight_accuracy"], float)

    def test_get_state_is_json_serialisable(self) -> None:
        import json

        engine = TemporalCouplingEngine()
        engine.classify_query("Remember last time", balanced_weights())
        state = engine.get_state()
        # Convert StrEnum values to str for JSON serialisation
        serialisable = {
            k: (str(v) if isinstance(v, TemporalCouplingMode) else v) for k, v in state.items()
        }
        json.dumps(serialisable)  # Should not raise

    def test_mode_counts_cover_all_modes(self) -> None:
        engine = TemporalCouplingEngine()
        state = engine.get_state()
        counts = state["mode_counts"]
        for mode in TemporalCouplingMode:
            assert mode in counts
