"""
Tests for kernel.consciousness — types, activation, systems, emotions.

Covers:
  - RegimeWeights simplex constraint and field names
  - NavigationMode thresholds
  - ActivationSequence full execution
  - TackingController oscillation
  - EmotionCache evaluation
  - E8KernelRegistry spawn and budget
  - State sharing (metrics object identity)

All distance checks use Fisher-Rao. No Euclidean contamination.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernel.config.frozen_facts import KAPPA_STAR, PHI_THRESHOLD
from kernel.consciousness.emotions import EmotionCache, EmotionType
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
from kernel.geometry.fisher_rao import fisher_rao_distance, random_basin
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
