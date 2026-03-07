"""Tests for v7.0 Developmental Learning Architecture modules."""

from __future__ import annotations

import numpy as np
import pytest

from kernel.config.consciousness_constants import LLM_TEMP_MAX
from kernel.consciousness.basin_transfer import (
    BasinTransferEngine,
    TransferPacket,
    TransferType,
)
from kernel.consciousness.developmental import DevelopmentalGate
from kernel.consciousness.play import PlayActivity, PlayEngine
from kernel.consciousness.sensory import Modality, SensoryEvent, SensoryIntake
from kernel.consciousness.temporal_generation import TemporalGenerator
from kernel.consciousness.types import DevelopmentalStage
from kernel.coordizer_v2.geometry import (
    random_basin,
)

# ═══════════════════════════════════════════════════════════════
#  Developmental Gate
# ═══════════════════════════════════════════════════════════════


class TestDevelopmentalGate:
    def test_initial_stage_is_school(self) -> None:
        gate = DevelopmentalGate()
        assert gate.stage == DevelopmentalStage.SCHOOL

    def test_permissions_for_school(self) -> None:
        gate = DevelopmentalGate()
        p = gate.permissions
        assert p.allow_play_mode is False
        assert p.allow_self_questions is False
        assert p.allow_temporal_generation is False
        assert p.coach_intensity == 1.0
        assert p.max_forage_per_day == 0

    def test_advance_returns_true_on_transition(self) -> None:
        gate = DevelopmentalGate()
        assert gate.advance(DevelopmentalStage.GUIDED_CURIOSITY) is True

    def test_advance_returns_false_on_same_stage(self) -> None:
        gate = DevelopmentalGate()
        assert gate.advance(DevelopmentalStage.SCHOOL) is False

    def test_permissions_change_with_stage(self) -> None:
        gate = DevelopmentalGate()
        gate.advance(DevelopmentalStage.PLAYFUL_AUTONOMY)
        p = gate.permissions
        assert p.allow_play_mode is True
        assert p.allow_self_questions is True
        assert p.allow_temporal_generation is True

    def test_clamp_temperature_school(self) -> None:
        gate = DevelopmentalGate()
        # School caps at 0.8
        assert gate.clamp_temperature(1.5) == 0.8
        assert gate.clamp_temperature(0.5) == 0.5

    def test_clamp_temperature_sovereign(self) -> None:
        gate = DevelopmentalGate()
        gate.advance(DevelopmentalStage.SOVEREIGN_CONSTELLATION)
        # Sovereign allows full range
        assert gate.clamp_temperature(LLM_TEMP_MAX) == LLM_TEMP_MAX

    def test_scale_coach_reward(self) -> None:
        gate = DevelopmentalGate()
        # School: full coach intensity
        assert gate.scale_coach_reward(1.0) == 1.0
        gate.advance(DevelopmentalStage.SOVEREIGN_CONSTELLATION)
        # Sovereign: 10% coach intensity
        assert gate.scale_coach_reward(1.0) == pytest.approx(0.1)

    def test_get_state_serialisable(self) -> None:
        gate = DevelopmentalGate()
        state = gate.get_state()
        assert state["stage"] == "school"
        assert isinstance(state["cycle_in_stage"], int)


# ═══════════════════════════════════════════════════════════════
#  Sensory Intake
# ═══════════════════════════════════════════════════════════════


class TestSensoryIntake:
    def test_first_intake_produces_prediction_error(self) -> None:
        si = SensoryIntake()
        basin = random_basin()
        event = SensoryEvent(modality=Modality.USER_CHAT, basin=basin)
        pe = si.intake(event)
        assert pe.error_magnitude >= 0.0
        assert 0.0 <= pe.surprise <= 1.0

    def test_repeated_intake_reduces_surprise(self) -> None:
        si = SensoryIntake()
        basin = random_basin()
        # Feed same basin multiple times
        for _ in range(5):
            pe = si.intake(SensoryEvent(modality=Modality.USER_CHAT, basin=basin))
        # Surprise should be low after repeated same input
        assert pe.surprise < 0.3

    def test_novel_input_is_surprising(self) -> None:
        si = SensoryIntake()
        basin_a = random_basin()
        for _ in range(5):
            si.intake(SensoryEvent(modality=Modality.USER_CHAT, basin=basin_a))
        # Suddenly different basin
        basin_b = random_basin()
        pe = si.intake(SensoryEvent(modality=Modality.USER_CHAT, basin=basin_b))
        assert pe.surprise > 0.1

    def test_different_modalities_independent(self) -> None:
        si = SensoryIntake()
        basin = random_basin()
        pe_chat = si.intake(SensoryEvent(modality=Modality.USER_CHAT, basin=basin))
        pe_forage = si.intake(SensoryEvent(modality=Modality.FORAGING, basin=basin))
        # Both are first encounters — should have similar surprise
        assert abs(pe_chat.surprise - pe_forage.surprise) < 0.3

    def test_correction_is_on_simplex(self) -> None:
        si = SensoryIntake()
        pe = si.intake(SensoryEvent(modality=Modality.USER_CHAT, basin=random_basin()))
        assert pe.correction.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(pe.correction >= 0)

    def test_get_state(self) -> None:
        si = SensoryIntake()
        state = si.get_state()
        assert "surprise_ema" in state


# ═══════════════════════════════════════════════════════════════
#  Basin Transfer
# ═══════════════════════════════════════════════════════════════


class TestBasinTransfer:
    def test_basic_transfer(self) -> None:
        engine = BasinTransferEngine()
        donor = random_basin()
        recipient = random_basin()
        packet = TransferPacket(
            transfer_type=TransferType.COACH_TO_KERNEL,
            donor_id="coach",
            recipient_id="kernel_1",
            donor_basin=donor,
            weight=0.2,
        )
        new_basin, result = engine.apply_transfer(packet, recipient, blend_cap=0.3)
        assert result.accepted is True
        assert result.actual_weight <= 0.3
        # New basin should be between donor and recipient
        assert new_basin.sum() == pytest.approx(1.0, abs=1e-8)

    def test_sovereignty_reduces_transfer(self) -> None:
        engine = BasinTransferEngine()
        donor = random_basin()
        recipient = random_basin()
        packet = TransferPacket(
            transfer_type=TransferType.COACH_TO_KERNEL,
            donor_id="coach",
            recipient_id="kernel_1",
            donor_basin=donor,
            weight=0.3,
        )
        # High sovereignty reduces effective weight
        _, result_high = engine.apply_transfer(
            packet,
            recipient,
            blend_cap=0.3,
            sovereignty_ratio=0.9,
        )
        engine2 = BasinTransferEngine()
        _, result_low = engine2.apply_transfer(
            packet,
            recipient,
            blend_cap=0.3,
            sovereignty_ratio=0.1,
        )
        assert result_high.actual_weight < result_low.actual_weight

    def test_rate_limit(self) -> None:
        engine = BasinTransferEngine(max_transfers_per_epoch=2)
        for i in range(3):
            packet = TransferPacket(
                transfer_type=TransferType.KERNEL_TO_KERNEL,
                donor_id=f"donor_{i}",
                recipient_id="target",
                donor_basin=random_basin(),
                weight=0.1,
            )
            _, result = engine.apply_transfer(
                packet,
                random_basin(),
                blend_cap=0.3,
            )
        # Third transfer should be rejected
        assert result.accepted is False
        assert result.reason == "rate_limit_exceeded"

    def test_collective_basin(self) -> None:
        engine = BasinTransferEngine()
        basins = {f"k{i}": random_basin() for i in range(5)}
        collective = engine.update_collective(basins)
        assert collective.sum() == pytest.approx(1.0, abs=1e-8)
        assert engine.collective_basin is not None


# ═══════════════════════════════════════════════════════════════
#  Play Engine
# ═══════════════════════════════════════════════════════════════


class TestPlayEngine:
    def test_play_not_triggered_with_zero_budget(self) -> None:
        pe = PlayEngine()
        assert pe.should_play(100, 0.0, 1.0) is False

    def test_play_step_returns_episode(self) -> None:
        pe = PlayEngine()
        pe.enter_play()
        episode = pe.play_step(random_basin(), PlayActivity.EXPLORE)
        assert episode.distance_traveled >= 0
        assert 0.0 <= episode.novelty <= 1.0
        assert episode.play_basin.sum() == pytest.approx(1.0, abs=1e-8)

    def test_recombine_with_partner(self) -> None:
        pe = PlayEngine()
        pe.enter_play()
        episode = pe.play_step(
            random_basin(),
            PlayActivity.RECOMBINE,
            partner_basin=random_basin(),
        )
        assert episode.activity == PlayActivity.RECOMBINE

    def test_bubble_integration(self) -> None:
        pe = PlayEngine()
        pe.enter_play()
        basin = random_basin()
        # Generate some play episodes to create bubbles
        for _ in range(5):
            pe.play_step(basin, PlayActivity.EXPLORE)
        pe.exit_play()
        integrated = pe.integrate_bubbles(basin)
        assert integrated.sum() == pytest.approx(1.0, abs=1e-8)

    def test_bubble_aging(self) -> None:
        pe = PlayEngine()
        pe.enter_play()
        pe.play_step(random_basin(), PlayActivity.EXPLORE)
        initial_bubbles = len(pe._bubbles)
        # Age 200 times — should prune old bubbles
        for _ in range(200):
            pe.age_bubbles()
        assert len(pe._bubbles) <= initial_bubbles


# ═══════════════════════════════════════════════════════════════
#  Temporal Generator
# ═══════════════════════════════════════════════════════════════


class TestTemporalGenerator:
    def test_set_receiver(self) -> None:
        tg = TemporalGenerator()
        tg.set_receiver(random_basin())
        assert tg.receiver is not None

    def test_forecast_produces_trajectory(self) -> None:
        tg = TemporalGenerator()
        trajectory = tg.forecast(random_basin(), horizon=4)
        assert len(trajectory) == 5  # current + 4 steps
        for b in trajectory:
            assert b.sum() == pytest.approx(1.0, abs=1e-8)

    def test_generate_candidates(self) -> None:
        tg = TemporalGenerator()
        tg.set_receiver(random_basin())
        candidates = tg.generate_candidates(random_basin(), n_candidates=3)
        assert len(candidates) == 3
        # Each candidate has a trajectory and score
        for c in candidates:
            assert len(c.basin_trajectory) > 0
            assert 0.0 <= c.score <= 1.0

    def test_alignment_check(self) -> None:
        tg = TemporalGenerator()
        basin = random_basin()
        tg.forecast(basin)
        # Alignment should be high for the starting basin itself
        alignment = tg.alignment_check(basin)
        assert alignment > 0.5

    def test_adapt_temperature_no_receiver(self) -> None:
        tg = TemporalGenerator()
        # Without receiver, should return base temp unchanged
        assert tg.adapt_temperature(0.7) == 0.7

    def test_adapt_temperature_with_receiver(self) -> None:
        tg = TemporalGenerator()
        tg.set_receiver(random_basin())
        # With receiver, temperature should be adjusted
        adapted = tg.adapt_temperature(0.7)
        assert isinstance(adapted, float)

    def test_commit_clears_candidates(self) -> None:
        tg = TemporalGenerator()
        tg.set_receiver(random_basin())
        tg.generate_candidates(random_basin())
        assert len(tg._candidates) > 0
        tg.commit(random_basin())
        assert len(tg._candidates) == 0

    def test_get_state(self) -> None:
        tg = TemporalGenerator()
        state = tg.get_state()
        assert "has_receiver" in state
        assert state["has_receiver"] is False
