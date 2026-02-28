"""Tests for The Cradle — protected development environment (#77).

Verifies:
  1. Admission and capacity limits
  2. Curriculum stage advancement
  3. Graduation gate (Phi threshold + curriculum)
  4. Stall detection
  5. State reporting
"""

from __future__ import annotations

from kernel.config.frozen_facts import PHI_THRESHOLD
from kernel.consciousness.cradle import (
    _STALL_WINDOW,
    Cradle,
    CradleAction,
)


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
