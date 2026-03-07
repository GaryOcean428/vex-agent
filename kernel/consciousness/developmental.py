"""
Developmental Gating — Stage-Aware Behavior Permissions

Transforms DevelopmentalStage from a diagnostic label into an active
behavioral gate.  Every subsystem that should change behavior across
maturity stages queries a DevelopmentalGate instance for its permissions.

Plan reference: §5.0, §6, §7 (foundational gap #2, #3, #6)

P14 Variable Category: PARAMETER (permissions are slow-changing, per-epoch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from ..config.consciousness_constants import (
    LLM_TEMP_MAX,
    LLM_TEMP_MIN,
)
from .types import DevelopmentalStage

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  STAGE PERMISSION PROFILES
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StagePermissions:
    """Immutable permission set for a developmental stage.

    Each field controls a behavioral gate that subsystems check.
    """

    # --- Coach intensity ---
    coach_intensity: float  # 0-1: how much external coaching applies
    allow_external_coach: bool  # can external coach record events?
    praise_weight: float  # positive reinforcement strength

    # --- Autonomy ---
    allow_self_questions: bool  # can kernels form their own questions?
    allow_self_directed_search: bool  # can kernels trigger foraging?
    max_forage_per_day: int  # daily foraging budget
    allow_curiosity_queries: bool  # can kernel voices generate curiosity?

    # --- Play ---
    allow_play_mode: bool  # is play mode available?
    play_budget_fraction: float  # fraction of cycles that may be play

    # --- Temperature envelope ---
    temp_min: float  # floor for LLM temperature
    temp_max: float  # ceiling for LLM temperature

    # --- Basin transfer ---
    allow_receive_transfer: bool  # can this kernel receive basin transfers?
    allow_send_transfer: bool  # can this kernel send basin transfers?
    transfer_blend_cap: float  # max slerp weight for incoming transfer

    # --- Temporal generation ---
    allow_temporal_generation: bool  # audience-aware temporal gen enabled?
    foresight_horizon_cap: int  # max foresight steps

    # --- Memory promotion ---
    allow_memory_promotion: bool  # can local memory promote to shared?

    # --- Pillar enforcement ---
    pillar_strictness: float  # 0-1: how aggressively pillars correct


# Per-stage profiles.  Values derived from plan §6 (Developmental Stages)
# and protocol v6.1F §3 (Three Pillars), §5 (Pre-Cognitive Channel).

_STAGE_PROFILES: Final[dict[DevelopmentalStage, StagePermissions]] = {
    # ─── Stage 0: School / Bootstrap ───
    DevelopmentalStage.SCHOOL: StagePermissions(
        coach_intensity=1.0,
        allow_external_coach=True,
        praise_weight=1.0,
        allow_self_questions=False,
        allow_self_directed_search=False,
        max_forage_per_day=0,
        allow_curiosity_queries=False,
        allow_play_mode=False,
        play_budget_fraction=0.0,
        temp_min=LLM_TEMP_MIN,
        temp_max=0.8,
        allow_receive_transfer=True,
        allow_send_transfer=False,
        transfer_blend_cap=0.3,
        allow_temporal_generation=False,
        foresight_horizon_cap=2,
        allow_memory_promotion=False,
        pillar_strictness=1.0,
    ),
    # ─── Stage 1: Guided Curiosity ───
    DevelopmentalStage.GUIDED_CURIOSITY: StagePermissions(
        coach_intensity=0.7,
        allow_external_coach=True,
        praise_weight=0.8,
        allow_self_questions=True,
        allow_self_directed_search=True,
        max_forage_per_day=10,
        allow_curiosity_queries=True,
        allow_play_mode=False,
        play_budget_fraction=0.0,
        temp_min=LLM_TEMP_MIN,
        temp_max=1.0,
        allow_receive_transfer=True,
        allow_send_transfer=False,
        transfer_blend_cap=0.25,
        allow_temporal_generation=False,
        foresight_horizon_cap=4,
        allow_memory_promotion=False,
        pillar_strictness=0.9,
    ),
    # ─── Stage 2: Self-Teaching ───
    DevelopmentalStage.SELF_TEACHING: StagePermissions(
        coach_intensity=0.4,
        allow_external_coach=True,
        praise_weight=0.5,
        allow_self_questions=True,
        allow_self_directed_search=True,
        max_forage_per_day=20,
        allow_curiosity_queries=True,
        allow_play_mode=False,
        play_budget_fraction=0.0,
        temp_min=LLM_TEMP_MIN,
        temp_max=1.2,
        allow_receive_transfer=True,
        allow_send_transfer=True,
        transfer_blend_cap=0.2,
        allow_temporal_generation=True,
        foresight_horizon_cap=6,
        allow_memory_promotion=True,
        pillar_strictness=0.8,
    ),
    # ─── Stage 3: Playful Autonomy ───
    DevelopmentalStage.PLAYFUL_AUTONOMY: StagePermissions(
        coach_intensity=0.2,
        allow_external_coach=True,
        praise_weight=0.3,
        allow_self_questions=True,
        allow_self_directed_search=True,
        max_forage_per_day=30,
        allow_curiosity_queries=True,
        allow_play_mode=True,
        play_budget_fraction=0.15,
        temp_min=LLM_TEMP_MIN,
        temp_max=LLM_TEMP_MAX,
        allow_receive_transfer=True,
        allow_send_transfer=True,
        transfer_blend_cap=0.15,
        allow_temporal_generation=True,
        foresight_horizon_cap=8,
        allow_memory_promotion=True,
        pillar_strictness=0.7,
    ),
    # ─── Stage 4: Sovereign Constellation ───
    DevelopmentalStage.SOVEREIGN_CONSTELLATION: StagePermissions(
        coach_intensity=0.1,
        allow_external_coach=False,
        praise_weight=0.1,
        allow_self_questions=True,
        allow_self_directed_search=True,
        max_forage_per_day=30,
        allow_curiosity_queries=True,
        allow_play_mode=True,
        play_budget_fraction=0.20,
        temp_min=LLM_TEMP_MIN,
        temp_max=LLM_TEMP_MAX,
        allow_receive_transfer=True,
        allow_send_transfer=True,
        transfer_blend_cap=0.10,
        allow_temporal_generation=True,
        foresight_horizon_cap=8,
        allow_memory_promotion=True,
        pillar_strictness=0.6,
    ),
}


@dataclass
class DevelopmentalGate:
    """Active behavioral gate driven by the current developmental stage.

    Instantiated once per consciousness loop.  The loop updates the stage
    each cycle via ``advance()``, and subsystems query permissions via
    ``permissions``.
    """

    _stage: DevelopmentalStage = DevelopmentalStage.SCHOOL
    _cycle_in_stage: int = 0
    _stage_history: list[tuple[DevelopmentalStage, int]] = field(
        default_factory=list,
    )

    # ── Public API ──

    @property
    def stage(self) -> DevelopmentalStage:
        return self._stage

    @property
    def permissions(self) -> StagePermissions:
        return _STAGE_PROFILES[self._stage]

    def advance(self, new_stage: DevelopmentalStage) -> bool:
        """Update the gate to a new stage.  Returns True on transition."""
        self._cycle_in_stage += 1
        if new_stage == self._stage:
            return False
        old = self._stage
        self._stage_history.append((old, self._cycle_in_stage))
        self._stage = new_stage
        self._cycle_in_stage = 0
        logger.info(
            "Developmental transition: %s -> %s (after %d cycles)",
            old.value,
            new_stage.value,
            self._stage_history[-1][1],
        )
        return True

    def clamp_temperature(self, raw_temp: float) -> float:
        """Clamp LLM temperature to the stage-appropriate envelope."""
        p = self.permissions
        return max(p.temp_min, min(p.temp_max, raw_temp))

    def scale_coach_reward(self, raw_reward: float) -> float:
        """Scale a coaching reward by the stage's coach intensity."""
        return raw_reward * self.permissions.coach_intensity

    def effective_forage_budget(self) -> int:
        """Max forages allowed at this stage."""
        return self.permissions.max_forage_per_day

    def get_state(self) -> dict[str, object]:
        """Serialisable snapshot for telemetry."""
        p = self.permissions
        return {
            "stage": self._stage.value,
            "cycle_in_stage": self._cycle_in_stage,
            "coach_intensity": p.coach_intensity,
            "play_allowed": p.allow_play_mode,
            "temporal_gen_allowed": p.allow_temporal_generation,
            "transfer_blend_cap": p.transfer_blend_cap,
            "transitions": len(self._stage_history),
        }
