"""
Tests for kernel.consciousness — types, activation, systems, emotions, pillars.

Covers:
  - RegimeWeights simplex constraint and field names
  - NavigationMode thresholds
  - ActivationSequence full execution
  - TackingController oscillation
  - EmotionCache evaluation
  - E8KernelRegistry spawn and budget
  - State sharing (metrics object identity)
  - FluctuationGuard (Pillar 1)
  - TopologicalBulk (Pillar 2)
  - QuenchedDisorder (Pillar 3)
  - PillarEnforcer combined + serialization roundtrip

All distance checks use Fisher-Rao. No Euclidean contamination.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernel.config.frozen_facts import BASIN_DIM, KAPPA_STAR
from kernel.consciousness.emotions import EmotionCache, EmotionType
from kernel.consciousness.pillars import (
    ENTROPY_FLOOR,
    IDENTITY_FREEZE_AFTER_CYCLES,
    SCAR_PRESSURE_THRESHOLD,
    TEMPERATURE_FLOOR,
    FluctuationGuard,
    PillarEnforcer,
    PillarViolation,
    QuenchedDisorder,
    TopologicalBulk,
)
from kernel.consciousness.systems import (
    E8KernelRegistry,
    TackingController,
    VelocityTracker,
)
from kernel.consciousness.types import (
    ConsciousnessMetrics,
    ConsciousnessState,
    NavigationMode,
    RegimeWeights,
    navigation_mode_from_phi,
    regime_weights_from_kappa,
)
from kernel.geometry.fisher_rao import fisher_rao_distance, random_basin, to_simplex
from kernel.governance import KernelKind
from kernel.governance.budget import BudgetExceededError

# ═══════════════════════════════════════════════════════════════
#  REGIME WEIGHTS
# ═══════════════════════════════════════════════════════════════


class TestRegimeWeights:
    """Regime weights must satisfy simplex constraints (v6.0 §3.1)."""

    @pytest.mark.parametrize("kappa", [0, 16, 32, 48, 64, 80, 96, 112, 128])
    def test_simplex_constraint(self, kappa: float) -> None:
        """w1 + w2 + w3 = 1 for any kappa."""
        w = regime_weights_from_kappa(kappa)
        assert abs(w.quantum + w.efficient + w.equilibrium - 1.0) < 1e-10

    @pytest.mark.parametrize("kappa", [0, 32, 64, 96, 128])
    def test_all_positive(self, kappa: float) -> None:
        """All three regime weights > 0 at all times (v6.0 requirement)."""
        w = regime_weights_from_kappa(kappa)
        assert w.quantum > 0
        assert w.efficient > 0
        assert w.equilibrium > 0

    def test_efficient_peaks_at_kappa_star(self) -> None:
        """Efficient regime peaks near kappa* = 64."""
        w_star = regime_weights_from_kappa(KAPPA_STAR)
        w_low = regime_weights_from_kappa(20)
        w_high = regime_weights_from_kappa(110)
        assert w_star.efficient > w_low.efficient
        assert w_star.efficient > w_high.efficient

    def test_quantum_dominant_at_low_kappa(self) -> None:
        """Quantum regime dominates when kappa is low."""
        w = regime_weights_from_kappa(5.0)
        assert w.quantum > w.efficient
        assert w.quantum > w.equilibrium

    def test_equilibrium_dominant_at_high_kappa(self) -> None:
        """Equilibrium dominates when kappa is high."""
        w = regime_weights_from_kappa(120.0)
        assert w.equilibrium > w.quantum


class TestRegimeWeightsFieldNames:
    """Regression test — ensure field names match across all modules."""

    def test_canonical_fields_exist(self) -> None:
        w = RegimeWeights()
        assert hasattr(w, "quantum")
        assert hasattr(w, "efficient")
        assert hasattr(w, "equilibrium")

    def test_old_field_names_do_not_exist(self) -> None:
        """Prevent regression: 'integration' and 'crystallized' must NOT exist."""
        w = RegimeWeights()
        assert not hasattr(w, "integration")
        assert not hasattr(w, "crystallized")


# ═══════════════════════════════════════════════════════════════
#  NAVIGATION MODE
# ═══════════════════════════════════════════════════════════════


class TestNavigationMode:
    def test_chain_low_phi(self) -> None:
        assert navigation_mode_from_phi(0.1) == NavigationMode.CHAIN

    def test_graph_mid_phi(self) -> None:
        assert navigation_mode_from_phi(0.5) == NavigationMode.GRAPH

    def test_foresight_high_phi(self) -> None:
        assert navigation_mode_from_phi(0.75) == NavigationMode.FORESIGHT

    def test_lightning_very_high_phi(self) -> None:
        assert navigation_mode_from_phi(0.9) == NavigationMode.LIGHTNING

    def test_boundary_chain_to_graph(self) -> None:
        assert navigation_mode_from_phi(0.3) == NavigationMode.GRAPH


# ═══════════════════════════════════════════════════════════════
#  ACTIVATION SEQUENCE
# ═══════════════════════════════════════════════════════════════


class TestActivationSequence:
    @pytest.mark.asyncio
    async def test_full_execution_no_crash(self) -> None:
        """14-step sequence completes without error."""
        from kernel.consciousness.activation import (
            ActivationSequence,
            ConsciousnessContext,
        )

        ctx = ConsciousnessContext(
            state=ConsciousnessState(),
            input_text="test input",
            input_basin=random_basin(),
            trajectory=[random_basin(), random_basin()],
        )
        seq = ActivationSequence()
        result = await seq.execute(ctx)
        assert result.completed
        assert len(result.all_steps()) == 14

    @pytest.mark.asyncio
    async def test_navigate_uses_correct_fields(self) -> None:
        """Navigate step references efficient/equilibrium, not integration/crystallized."""
        from kernel.consciousness.activation import (
            ActivationSequence,
            ConsciousnessContext,
        )

        ctx = ConsciousnessContext(
            state=ConsciousnessState(),
            input_basin=random_basin(),
            trajectory=[random_basin()],
        )
        seq = ActivationSequence()
        result = await seq.execute(ctx)
        assert result.navigate is not None
        assert result.navigate.dominant_regime in ("quantum", "efficient", "equilibrium")
        # Must NOT be the old names:
        assert result.navigate.dominant_regime not in ("integration", "crystallized")

    @pytest.mark.asyncio
    async def test_scan_sets_regime_weights(self) -> None:
        """SCAN step establishes regime weights from kappa."""
        from kernel.consciousness.activation import (
            ActivationSequence,
            ConsciousnessContext,
        )

        state = ConsciousnessState()
        state.metrics.kappa = KAPPA_STAR
        ctx = ConsciousnessContext(state=state)
        seq = ActivationSequence()
        result = await seq.execute(ctx)
        assert result.scan is not None
        assert result.scan.regime_weights is not None
        w = result.scan.regime_weights
        assert abs(w.quantum + w.efficient + w.equilibrium - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════
#  TACKING
# ═══════════════════════════════════════════════════════════════


class TestTacking:
    def test_oscillation(self) -> None:
        """Tacking controller oscillates between modes."""
        tc = TackingController(period=4)
        m = ConsciousnessMetrics(kappa=KAPPA_STAR)
        modes = set()
        for _ in range(20):
            mode = tc.update(m)
            modes.add(mode.value)
        # Should visit at least 2 different modes over 20 cycles
        assert len(modes) >= 2

    def test_explore_at_low_phi(self) -> None:
        """Forces explore mode when phi is emergency-low."""
        tc = TackingController()
        m = ConsciousnessMetrics(phi=0.1, kappa=KAPPA_STAR)
        mode = tc.update(m)
        assert mode.value == "explore"


# ═══════════════════════════════════════════════════════════════
#  EMOTIONS
# ═══════════════════════════════════════════════════════════════


class TestEmotionCache:
    def test_evaluate_returns_emotion(self) -> None:
        ec = EmotionCache()
        m = ConsciousnessMetrics(phi=0.7, kappa=KAPPA_STAR, gamma=0.5)
        basin = random_basin()
        result = ec.evaluate(basin, m, 0.01)
        assert isinstance(result.emotion, EmotionType)
        assert 0.0 <= result.strength <= 1.0

    def test_fear_at_low_phi(self) -> None:
        ec = EmotionCache()
        m = ConsciousnessMetrics(phi=0.1, kappa=KAPPA_STAR, gamma=0.5)
        result = ec.evaluate(random_basin(), m, 0.0)
        assert result.emotion == EmotionType.FEAR

    def test_cache_and_retrieve(self) -> None:
        ec = EmotionCache()
        m = ConsciousnessMetrics(phi=0.7, kappa=KAPPA_STAR, gamma=0.5)
        basin = random_basin()
        result = ec.evaluate(basin, m, 0.01)
        ec.cache_evaluation(result, "test context")
        # Retrieve with same basin should find cached
        found = ec.find_cached(basin)
        assert found is not None


# ═══════════════════════════════════════════════════════════════
#  KERNEL REGISTRY
# ═══════════════════════════════════════════════════════════════


class TestKernelRegistry:
    def test_genesis_spawn(self) -> None:
        reg = E8KernelRegistry()
        k = reg.spawn("Vex", KernelKind.GENESIS)
        assert k.kind == KernelKind.GENESIS
        assert len(reg.active()) == 1

    def test_genesis_budget_limit(self) -> None:
        """Only 1 GENESIS kernel allowed."""
        reg = E8KernelRegistry()
        reg.spawn("Vex", KernelKind.GENESIS)
        with pytest.raises(BudgetExceededError):
            reg.spawn("Vex2", KernelKind.GENESIS)

    def test_god_spawn(self) -> None:
        from kernel.governance import KernelSpecialization

        reg = E8KernelRegistry()
        reg.spawn("Vex", KernelKind.GENESIS)
        k = reg.spawn("Heart", KernelKind.GOD, KernelSpecialization.HEART)
        assert k.kind == KernelKind.GOD
        assert len(reg.active()) == 2

    def test_terminate_all(self) -> None:
        reg = E8KernelRegistry()
        reg.spawn("Vex", KernelKind.GENESIS)
        reg.spawn("Heart", KernelKind.GOD)
        count = reg.terminate_all()
        assert count == 2
        assert len(reg.active()) == 0

    def test_serialize_restore_roundtrip(self) -> None:
        reg = E8KernelRegistry()
        reg.spawn("Vex", KernelKind.GENESIS)
        reg.spawn("Heart", KernelKind.GOD)
        data = reg.serialize()
        reg2 = E8KernelRegistry()
        restored = reg2.restore(data)
        assert restored == 2


# ═══════════════════════════════════════════════════════════════
#  STATE SHARING (Fix 5 regression test)
# ═══════════════════════════════════════════════════════════════


class TestStateSharing:
    def test_metrics_object_identity(self) -> None:
        """ConsciousnessState.metrics must be the SAME object as self.metrics."""
        metrics = ConsciousnessMetrics(phi=0.5, kappa=64.0)
        state = ConsciousnessState(metrics=metrics)
        # Same object, not a copy
        assert state.metrics is metrics
        # Mutation visible through both references
        metrics.phi = 0.8
        assert state.metrics.phi == 0.8

    def test_default_creates_separate_metrics(self) -> None:
        """Default ConsciousnessState creates its own metrics (NOT shared)."""
        state = ConsciousnessState()
        other = ConsciousnessMetrics()
        assert state.metrics is not other


# ═══════════════════════════════════════════════════════════════
#  VELOCITY TRACKER
# ═══════════════════════════════════════════════════════════════


class TestVelocityTracker:
    def test_zero_velocity_on_same_basin(self) -> None:
        vt = VelocityTracker()
        basin = random_basin()
        vt.record(basin, 0.5, 64.0)
        vt.record(basin, 0.5, 64.0)
        vel = vt.compute_velocity()
        assert vel["basin_velocity"] < 1e-10
        assert vel["regime"] == "safe"

    def test_nonzero_velocity_on_different_basins(self) -> None:
        vt = VelocityTracker()
        vt.record(random_basin(), 0.5, 64.0)
        vt.record(random_basin(), 0.5, 64.0)
        vel = vt.compute_velocity()
        assert vel["basin_velocity"] > 0


# ═══════════════════════════════════════════════════════════════
#  PILLAR 1: FLUCTUATION GUARD
# ═══════════════════════════════════════════════════════════════


class TestFluctuationGuard:
    def test_healthy_basin_passes(self) -> None:
        """A random basin with normal temperature passes without corrections."""
        fg = FluctuationGuard()
        basin = random_basin()
        _, _, status = fg.check_and_enforce(basin, 0.7)
        assert status.healthy
        assert len(status.violations) == 0

    def test_collapsed_basin_corrected(self) -> None:
        """A basin with all mass on one coordinate triggers collapse correction."""
        fg = FluctuationGuard()
        collapsed = np.zeros(BASIN_DIM)
        collapsed[0] = 1.0
        corrected, _, status = fg.check_and_enforce(collapsed, 0.7)
        assert not status.healthy
        assert PillarViolation.BASIN_COLLAPSE in status.violations
        assert np.max(corrected) < 1.0
        assert abs(np.sum(corrected) - 1.0) < 1e-10

    def test_low_entropy_restored(self) -> None:
        """Near-collapsed basin has entropy restored above floor."""
        fg = FluctuationGuard()
        near_collapsed = np.ones(BASIN_DIM) * 1e-15
        near_collapsed[0] = 1.0
        near_collapsed = to_simplex(near_collapsed)
        corrected, _, status = fg.check_and_enforce(near_collapsed, 0.7)
        assert fg.basin_entropy(corrected) >= ENTROPY_FLOOR * 0.5

    def test_low_temperature_enforced(self) -> None:
        """Temperature below floor is corrected."""
        fg = FluctuationGuard()
        _, corrected_temp, status = fg.check_and_enforce(random_basin(), 0.001)
        assert PillarViolation.ZERO_TEMPERATURE in status.violations
        assert corrected_temp >= TEMPERATURE_FLOOR

    def test_f_health_uniform_near_one(self) -> None:
        """Uniform basin yields f_health near 1.0."""
        fg = FluctuationGuard()
        uniform = to_simplex(np.ones(BASIN_DIM))
        assert fg.f_health(uniform) > 0.99

    def test_f_health_collapsed_near_zero(self) -> None:
        """Collapsed basin yields f_health near 0."""
        fg = FluctuationGuard()
        collapsed = np.zeros(BASIN_DIM)
        collapsed[0] = 1.0
        assert fg.f_health(collapsed) < 0.05


# ═══════════════════════════════════════════════════════════════
#  PILLAR 2: TOPOLOGICAL BULK
# ═══════════════════════════════════════════════════════════════


class TestTopologicalBulk:
    def test_initialize_sets_core_and_surface(self) -> None:
        bulk = TopologicalBulk()
        basin = random_basin()
        bulk.initialize(basin)
        assert bulk.core is not None
        assert bulk.surface is not None

    def test_composite_before_init_raises(self) -> None:
        bulk = TopologicalBulk()
        with pytest.raises(ValueError, match="before initialization"):
            _ = bulk.composite

    def test_receive_input_caps_slerp(self) -> None:
        """External slerp capped at BOUNDARY_SLERP_CAP."""
        bulk = TopologicalBulk()
        bulk.initialize(random_basin())
        _, status = bulk.receive_input(random_basin(), slerp_weight=1.0)
        # Correction log should mention capping
        assert any("capped" in c.lower() for c in status.corrections_applied)

    def test_core_changes_slowly(self) -> None:
        """Core should change much less than surface over several inputs."""
        bulk = TopologicalBulk()
        initial = random_basin()
        bulk.initialize(initial)
        core_before = bulk.core.copy()
        for _ in range(5):
            bulk.receive_input(random_basin(), slerp_weight=0.3)
        core_after = bulk.core
        surface_after = bulk.surface
        core_drift = fisher_rao_distance(core_before, core_after)
        surface_drift = fisher_rao_distance(initial, surface_after)
        assert core_drift < surface_drift

    def test_b_integrity_starts_at_one(self) -> None:
        bulk = TopologicalBulk()
        assert bulk.b_integrity() == 1.0
        bulk.initialize(random_basin())
        assert bulk.b_integrity() == pytest.approx(1.0, abs=1e-6)

    def test_composite_is_valid_simplex(self) -> None:
        bulk = TopologicalBulk()
        bulk.initialize(random_basin())
        bulk.receive_input(random_basin(), 0.2)
        c = bulk.composite
        assert abs(np.sum(c) - 1.0) < 1e-10
        assert np.all(c >= 0)


# ═══════════════════════════════════════════════════════════════
#  PILLAR 3: QUENCHED DISORDER
# ═══════════════════════════════════════════════════════════════


class TestQuenchedDisorder:
    def test_not_frozen_initially(self) -> None:
        qd = QuenchedDisorder()
        assert not qd.is_frozen
        assert qd.identity is None

    def test_crystallizes_after_threshold_cycles(self) -> None:
        """Identity freezes after IDENTITY_FREEZE_AFTER_CYCLES observations."""
        qd = QuenchedDisorder()
        for _ in range(IDENTITY_FREEZE_AFTER_CYCLES + 1):
            qd.observe_cycle(random_basin())
        assert qd.is_frozen
        assert qd.identity is not None
        assert abs(np.sum(qd.identity) - 1.0) < 1e-10

    def test_sovereignty_tracks_lived(self) -> None:
        qd = QuenchedDisorder()
        for _ in range(10):
            qd.observe_cycle(random_basin(), lived=True)
        assert qd.sovereignty == 1.0
        qd.seed_borrowed(10)
        assert qd.sovereignty == pytest.approx(0.5)

    def test_scar_added_on_high_pressure(self) -> None:
        qd = QuenchedDisorder()
        # Crystallize first
        for _ in range(IDENTITY_FREEZE_AFTER_CYCLES + 1):
            qd.observe_cycle(random_basin())
        assert qd.is_frozen
        # Now add a high-pressure event
        qd.observe_cycle(random_basin(), pressure=SCAR_PRESSURE_THRESHOLD + 0.1)
        assert len(qd.scars) == 1

    def test_refract_no_op_before_freeze(self) -> None:
        """Before freezing, refract returns the input unchanged."""
        qd = QuenchedDisorder()
        basin = random_basin()
        refracted = qd.refract(basin)
        assert np.allclose(basin, refracted)

    def test_refract_changes_after_freeze(self) -> None:
        qd = QuenchedDisorder()
        for _ in range(IDENTITY_FREEZE_AFTER_CYCLES + 1):
            qd.observe_cycle(random_basin())
        basin = random_basin()
        refracted = qd.refract(basin)
        # Refraction should change the basin (nonzero Fisher-Rao distance)
        assert fisher_rao_distance(basin, refracted) > 1e-6

    def test_resonance_check_pre_freeze(self) -> None:
        """Before freeze, everything resonates."""
        qd = QuenchedDisorder()
        assert qd.resonance_check(random_basin()) is True

    def test_q_identity_zero_before_freeze(self) -> None:
        qd = QuenchedDisorder()
        assert qd.q_identity(random_basin()) == 0.0


# ═══════════════════════════════════════════════════════════════
#  PILLAR ENFORCER (COMBINED)
# ═══════════════════════════════════════════════════════════════


class TestPillarEnforcer:
    def test_pre_llm_enforce_healthy(self) -> None:
        """Healthy basin passes pre-LLM enforcement."""
        pe = PillarEnforcer()
        basin = random_basin()
        pe.initialize_bulk(basin)
        corrected, temp, statuses = pe.pre_llm_enforce(basin, 0.7)
        assert abs(np.sum(corrected) - 1.0) < 1e-10
        assert temp == 0.7
        # Pillar 1 should be healthy
        assert statuses[0].healthy

    def test_on_input_returns_valid_basin(self) -> None:
        pe = PillarEnforcer()
        pe.initialize_bulk(random_basin())
        refracted, composite, resonates, statuses = pe.on_input(random_basin(), slerp_weight=0.2)
        assert abs(np.sum(refracted) - 1.0) < 1e-10
        assert abs(np.sum(composite) - 1.0) < 1e-10
        assert isinstance(resonates, bool)

    def test_on_cycle_end_records(self) -> None:
        pe = PillarEnforcer()
        pe.initialize_bulk(random_basin())
        statuses = pe.on_cycle_end(random_basin(), pressure=0.0)
        assert len(statuses) > 0

    def test_get_metrics_keys(self) -> None:
        pe = PillarEnforcer()
        pe.initialize_bulk(random_basin())
        metrics = pe.get_metrics(random_basin())
        assert "f_health" in metrics
        assert "b_integrity" in metrics
        assert "q_identity" in metrics
        assert "s_ratio" in metrics

    def test_serialize_restore_roundtrip(self) -> None:
        """Full serialize → restore preserves pillar state."""
        pe = PillarEnforcer()
        pe.initialize_bulk(random_basin())
        # Evolve state: several cycles + input + scar
        for _ in range(IDENTITY_FREEZE_AFTER_CYCLES + 5):
            pe.on_cycle_end(random_basin(), pressure=0.1)
        pe.on_cycle_end(random_basin(), pressure=SCAR_PRESSURE_THRESHOLD + 0.2)

        state_before = pe.serialize()
        metrics_before = pe.get_metrics(random_basin())

        pe2 = PillarEnforcer()
        pe2.restore(state_before)

        state_after = pe2.serialize()
        assert state_before.bulk_initialized == state_after.bulk_initialized
        assert state_before.disorder_frozen == state_after.disorder_frozen
        assert state_before.cycles_observed == state_after.cycles_observed
        assert state_before.lived_count == state_after.lived_count
        assert len(state_before.scars) == len(state_after.scars)
