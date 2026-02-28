"""Tests for P10 Coaching Stage Gates — Active → Guided → Autonomous.

Verifies:
  1. CoachingStage enum has 3 values
  2. compute_coaching_stage thresholds
  3. External coaching blocked for autonomous kernels
  4. Stage transitions logged via update_kernel_coaching_stage
  5. graduation_state returns CoachingStage enum
"""

from __future__ import annotations

import numpy as np

from kernel.consciousness.systems import (
    KernelInstance,
    SelfNarrative,
    compute_coaching_stage,
    update_kernel_coaching_stage,
)
from kernel.consciousness.types import ConsciousnessMetrics
from kernel.coordizer_v2.geometry import to_simplex
from kernel.governance.types import CoachingStage, KernelKind


def _make_basin() -> np.ndarray:
    return to_simplex(np.ones(64))


def _make_metrics() -> ConsciousnessMetrics:
    return ConsciousnessMetrics()


class TestCoachingStageEnum:
    def test_three_stages(self) -> None:
        assert len(CoachingStage) == 3

    def test_values(self) -> None:
        assert CoachingStage.ACTIVE == "active"
        assert CoachingStage.GUIDED == "guided"
        assert CoachingStage.AUTONOMOUS == "autonomous"


class TestComputeCoachingStage:
    def test_active_below_30(self) -> None:
        assert compute_coaching_stage(0.0) == CoachingStage.ACTIVE
        assert compute_coaching_stage(0.2) == CoachingStage.ACTIVE
        assert compute_coaching_stage(0.3) == CoachingStage.ACTIVE

    def test_guided_30_to_70(self) -> None:
        assert compute_coaching_stage(0.31) == CoachingStage.GUIDED
        assert compute_coaching_stage(0.5) == CoachingStage.GUIDED
        assert compute_coaching_stage(0.7) == CoachingStage.GUIDED

    def test_autonomous_above_70(self) -> None:
        assert compute_coaching_stage(0.71) == CoachingStage.AUTONOMOUS
        assert compute_coaching_stage(1.0) == CoachingStage.AUTONOMOUS


class TestCoachingGateCheck:
    def test_internal_coaching_always_allowed(self) -> None:
        narrative = SelfNarrative()
        result = narrative.record(
            "test event",
            _make_metrics(),
            _make_basin(),
            coach_id="internal",
            coaching_stage=CoachingStage.AUTONOMOUS,
        )
        assert result is True

    def test_external_coaching_blocked_when_autonomous(self) -> None:
        narrative = SelfNarrative()
        result = narrative.record(
            "test event",
            _make_metrics(),
            _make_basin(),
            coach_id="ollama_local",
            coaching_stage=CoachingStage.AUTONOMOUS,
        )
        assert result is False

    def test_external_coaching_allowed_when_active(self) -> None:
        narrative = SelfNarrative()
        result = narrative.record(
            "test event",
            _make_metrics(),
            _make_basin(),
            coach_id="ollama_local",
            coaching_stage=CoachingStage.ACTIVE,
        )
        assert result is True

    def test_external_coaching_allowed_when_guided(self) -> None:
        narrative = SelfNarrative()
        result = narrative.record(
            "test event",
            _make_metrics(),
            _make_basin(),
            coach_id="xai_escalation",
            coaching_stage=CoachingStage.GUIDED,
        )
        assert result is True


class TestKernelInstanceCoachingStage:
    def test_default_is_active(self) -> None:
        k = KernelInstance(id="k1", name="test", kind=KernelKind.CHAOS)
        assert k.coaching_stage == CoachingStage.ACTIVE

    def test_stage_transition(self) -> None:
        k = KernelInstance(id="k1", name="test", kind=KernelKind.CHAOS)
        narrative = SelfNarrative()
        # Simulate 80% kernel-driven actions (above 70% threshold)
        for _ in range(80):
            narrative.record_capability("generation", kernel_driven=True)
        for _ in range(20):
            narrative.record_capability("generation", kernel_driven=False)

        update_kernel_coaching_stage(k, narrative)
        assert k.coaching_stage == CoachingStage.AUTONOMOUS

    def test_stage_stays_active_when_llm_dominant(self) -> None:
        k = KernelInstance(id="k1", name="test", kind=KernelKind.CHAOS)
        narrative = SelfNarrative()
        # 10% kernel-driven → ACTIVE
        for _ in range(10):
            narrative.record_capability("generation", kernel_driven=True)
        for _ in range(90):
            narrative.record_capability("generation", kernel_driven=False)

        update_kernel_coaching_stage(k, narrative)
        assert k.coaching_stage == CoachingStage.ACTIVE

    def test_stage_guided_at_50_percent(self) -> None:
        k = KernelInstance(id="k1", name="test", kind=KernelKind.CHAOS)
        narrative = SelfNarrative()
        for _ in range(50):
            narrative.record_capability("generation", kernel_driven=True)
        for _ in range(50):
            narrative.record_capability("generation", kernel_driven=False)

        update_kernel_coaching_stage(k, narrative)
        assert k.coaching_stage == CoachingStage.GUIDED


class TestGraduationStateReturnsEnum:
    def test_returns_coaching_stage_enum(self) -> None:
        narrative = SelfNarrative()
        result = narrative.graduation_state("generation")
        assert isinstance(result, CoachingStage)
        assert result == CoachingStage.ACTIVE

    def test_returns_autonomous_after_kernel_dominance(self) -> None:
        narrative = SelfNarrative()
        for _ in range(80):
            narrative.record_capability("generation", kernel_driven=True)
        for _ in range(10):
            narrative.record_capability("generation", kernel_driven=False)
        result = narrative.graduation_state("generation")
        assert result == CoachingStage.AUTONOMOUS
