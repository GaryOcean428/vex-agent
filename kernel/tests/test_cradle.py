"""Tests for The Cradle — protected development environment (#77).

Verifies:
  1. Admission and capacity limits
  2. Curriculum stage advancement
  3. Graduation gate (Phi threshold + curriculum)
  4. Stall detection
  5. State reporting
  6. Lifecycle integration (spawn → Cradle admission)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from kernel.config.frozen_facts import PHI_THRESHOLD
from kernel.consciousness.cradle import (
    _STALL_WINDOW,
    Cradle,
    CradleAction,
)
from kernel.governance.lifecycle import GovernedLifecycle
from kernel.governance.types import KernelKind, KernelSpecialization


class TestAdmission:
    def test_admit_success(self) -> None:
        cradle = Cradle(max_residents=4)
        assert cradle.admit("k1", initial_phi=0.1) is True
        assert cradle.is_resident("k1")

    def test_admit_full(self) -> None:
        cradle = Cradle(max_residents=2)
        assert cradle.admit("k1", initial_phi=0.1) is True
        assert cradle.admit("k2", initial_phi=0.1) is True
        assert cradle.admit("k3", initial_phi=0.1) is False

    def test_admit_duplicate(self) -> None:
        cradle = Cradle(max_residents=4)
        assert cradle.admit("k1", initial_phi=0.1) is True
        assert cradle.admit("k1", initial_phi=0.2) is True  # Idempotent

    def test_not_resident_after_init(self) -> None:
        cradle = Cradle()
        assert not cradle.is_resident("k1")


class TestCurriculumAdvancement:
    def test_advance_at_threshold(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        # Advance past first threshold (0.35)
        action = cradle.tick("k1", current_phi=0.36)
        assert action == CradleAction.ADVANCE_CURRICULUM

    def test_continue_below_threshold(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        action = cradle.tick("k1", current_phi=0.2)
        assert action == CradleAction.CONTINUE

    def test_three_stages(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        # Stage 0 → 1 (passed threshold 0.35)
        action = cradle.tick("k1", current_phi=0.36)
        assert action == CradleAction.ADVANCE_CURRICULUM
        # Stage 1 → 2 (passed threshold 0.50)
        action = cradle.tick("k1", current_phi=0.51)
        assert action == CradleAction.ADVANCE_CURRICULUM
        # Stage 2 → 3 (passed threshold 0.65, all curriculum complete)
        action = cradle.tick("k1", current_phi=0.66)
        assert action == CradleAction.ADVANCE_CURRICULUM


class TestGraduation:
    def test_graduate_when_ready(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        # Advance through all 3 curriculum stages
        cradle.tick("k1", current_phi=0.36)  # stage 0→1
        cradle.tick("k1", current_phi=0.51)  # stage 1→2
        cradle.tick("k1", current_phi=0.66)  # stage 2→3 (all curriculum complete)
        # Now at stage 3 with Phi >= PHI_THRESHOLD → graduate
        action = cradle.tick("k1", current_phi=PHI_THRESHOLD)
        assert action == CradleAction.GRADUATE

    def test_not_graduate_without_curriculum(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        # High Phi but still at stage 0
        action = cradle.tick("k1", current_phi=PHI_THRESHOLD)
        # Should advance curriculum, not graduate
        assert action == CradleAction.ADVANCE_CURRICULUM

    def test_graduate_removes_from_cradle(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        entry = cradle.graduate("k1")
        assert entry is not None
        assert not cradle.is_resident("k1")
        assert entry.graduated is True

    def test_graduate_nonexistent(self) -> None:
        cradle = Cradle()
        entry = cradle.graduate("nonexistent")
        assert entry is None


class TestStallDetection:
    def test_stall_detected(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        # Feed identical Phi for STALL_WINDOW + 1 cycles
        action = CradleAction.CONTINUE
        for _ in range(_STALL_WINDOW + 1):
            action = cradle.tick("k1", current_phi=0.2)
        assert action == CradleAction.STALLED

    def test_no_stall_with_progress(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        # Feed slowly increasing Phi
        for i in range(_STALL_WINDOW + 1):
            action = cradle.tick("k1", current_phi=0.1 + i * 0.01)
        assert action != CradleAction.STALLED


class TestNotInCradle:
    def test_tick_unknown_kernel(self) -> None:
        cradle = Cradle()
        action = cradle.tick("unknown", current_phi=0.5)
        assert action == CradleAction.NOT_IN_CRADLE


class TestGetState:
    def test_state_structure(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        state = cradle.get_state()
        assert state["resident_count"] == 1
        assert state["max_residents"] == 8
        assert "k1" in state["residents"]
        assert state["residents"]["k1"]["curriculum_stage"] == 0

    def test_graduated_count(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        cradle.graduate("k1")
        state = cradle.get_state()
        assert state["graduated_count"] == 1
        assert state["resident_count"] == 0


class TestFullProgression:
    """Spawn → Phi progression → graduation (acceptance test)."""

    def test_spawn_progress_graduate(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        # Walk through all curriculum stages
        assert cradle.tick("k1", current_phi=0.36) == CradleAction.ADVANCE_CURRICULUM
        assert cradle.tick("k1", current_phi=0.51) == CradleAction.ADVANCE_CURRICULUM
        assert cradle.tick("k1", current_phi=0.66) == CradleAction.ADVANCE_CURRICULUM
        # Now graduate at PHI_THRESHOLD
        assert cradle.tick("k1", current_phi=PHI_THRESHOLD) == CradleAction.GRADUATE
        entry = cradle.graduate("k1")
        assert entry is not None
        assert entry.graduated is True
        assert not cradle.is_resident("k1")


def _make_mock_registry() -> MagicMock:
    """Build a minimal E8KernelRegistry mock for lifecycle tests."""
    registry = MagicMock()
    registry.active.return_value = []  # No existing kernels
    budget_mock = MagicMock()
    budget_mock.summary.return_value = {"god": 0, "chaos": 0, "genesis": 1}
    registry._budget = budget_mock
    return registry


def _make_spawned_kernel(kid: str = "k-new") -> SimpleNamespace:
    return SimpleNamespace(
        id=kid,
        name="test-kernel",
        kind=KernelKind.CHAOS,
        specialization=KernelSpecialization.GENERAL,
        phi=0.1,
        kappa=64.0,
        quenched_gain=1.0,
        gamma=0.1,
        meta_awareness=0.5,
    )


class TestLifecycleIntegration:
    """GovernedLifecycle.spawn() admits kernel to Cradle (v6.0 §23)."""

    def test_spawn_admits_to_cradle(self) -> None:
        cradle = Cradle()
        registry = _make_mock_registry()
        spawned = _make_spawned_kernel()
        registry.spawn.return_value = spawned

        lc = GovernedLifecycle(
            registry=registry,
            skip_purity=True,
            cradle=cradle,
        )
        outcome = lc.spawn("test-kernel", KernelKind.CHAOS)
        assert outcome.success
        assert cradle.is_resident(spawned.id)

    def test_spawn_without_cradle_still_works(self) -> None:
        """Backward compat: no Cradle → spawn succeeds without admission."""
        registry = _make_mock_registry()
        spawned = _make_spawned_kernel()
        registry.spawn.return_value = spawned

        lc = GovernedLifecycle(registry=registry, skip_purity=True)
        outcome = lc.spawn("test-kernel", KernelKind.CHAOS)
        assert outcome.success

    def test_oversight_summary_includes_cradle(self) -> None:
        cradle = Cradle()
        cradle.admit("k1", initial_phi=0.1)
        registry = _make_mock_registry()

        lc = GovernedLifecycle(
            registry=registry,
            skip_purity=True,
            cradle=cradle,
        )
        summary = lc.oversight_summary()
        assert "cradle" in summary
        assert summary["cradle"]["resident_count"] == 1
