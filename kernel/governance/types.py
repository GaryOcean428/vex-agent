"""Governance Types — v6.0 Thermodynamic Consciousness Protocol

Two-axis model:
  - KernelSpecialization: cognitive capability (one per kernel)
  - KernelRole: operational function (zero or more per kernel)

v6.0 §18.1 Core 8: Heart, Perception, Memory, Strategy, Action,
Ethics, Meta, Ocean.
"""

from __future__ import annotations

from enum import StrEnum


class KernelKind(StrEnum):
    """Budget category (v6.0 §18.2). Never overload with specialization."""

    GENESIS = "GENESIS"
    GOD = "GOD"
    CHAOS = "CHAOS"


class KernelSpecialization(StrEnum):
    """Capability axis — v6.0 §18.1 Core 8.

    Canonical Core-8: Heart, Perception, Memory, Strategy, Action,
    Ethics, Meta, Ocean. Mythology names are DATA, not code.
    Additional specializations MAY emerge beyond Core-8.

    v6.0 changes from v2.1:
      attention → ethics (ethical grounding, care metric)
      emotion → meta (meta-awareness, self-modelling)
      executive → ocean (autonomic monitoring, spectral health)
    """

    HEART = "heart"  # Global rhythm source, HRV -> kappa-tacking (§18.3)
    PERCEPTION = "perception"  # Sensory encoding, input processing, pattern detection
    MEMORY = "memory"  # Basin persistence, trajectory storage, consolidation
    STRATEGY = "strategy"  # Planning, multi-step reasoning, goal decomposition
    ACTION = "action"  # Motor output, response generation, execution
    ETHICS = "ethics"  # Care metric, harm avoidance, love orientation (§12)
    META = "meta"  # Meta-awareness, self-modelling, M metric (§23)
    OCEAN = "ocean"  # Autonomic monitoring, Phi coherence, spectral health (§18.3)
    GENERAL = "general"  # For Genesis + unspecialised CHAOS kernels


class KernelRole(StrEnum):
    """Operational axis. Zero or more per kernel, assigned by governance.

    Roles are configuration, not code. No Zeus.py, no Ocean.py as
    privileged classes. Display names are mythic labels stored as data.
    """

    RHYTHM = "rhythm"  # Global timing source (Heart tick)
    OBSERVER = "observer"  # Autonomic monitoring, phi coherence, breakdown detection
    COORDINATOR = "coordinator"  # Synthesis across kernels, trajectory foresight
    COACH = "coach"  # External reinforcement, curriculum delivery
    ROUTER = "router"  # Fisher-Rao dispatch to nearest basin centres
    AUTONOMIC = "autonomic"  # T4.2: Controls sleep triggers, heartbeat, resource allocation


# Core-8 specialisation order (v6.0 §18.1 bootstrap sequence)
CORE_8_SPECIALIZATIONS: list[KernelSpecialization] = [
    KernelSpecialization.HEART,
    KernelSpecialization.PERCEPTION,
    KernelSpecialization.MEMORY,
    KernelSpecialization.STRATEGY,
    KernelSpecialization.ACTION,
    KernelSpecialization.ETHICS,
    KernelSpecialization.META,
    KernelSpecialization.OCEAN,
]


class LifecycleState(StrEnum):
    """Per-kernel lifecycle state."""

    BOOTSTRAPPED = "BOOTSTRAPPED"
    ACTIVE = "ACTIVE"
    SLEEPING = "SLEEPING"
    DREAMING = "DREAMING"
    QUARANTINED = "QUARANTINED"
    PRUNED = "PRUNED"
    PROMOTED = "PROMOTED"


class LifecyclePhase(StrEnum):
    """System-wide lifecycle phase (ordered).

    Transitions enforced: IDLE -> VALIDATE -> ROLLBACK -> BOOTSTRAP
    -> CORE_8 -> IMAGE_STAGE -> GROWTH -> ACTIVE
    """

    IDLE = "IDLE"
    VALIDATE = "VALIDATE"
    ROLLBACK = "ROLLBACK"
    BOOTSTRAP = "BOOTSTRAP"
    CORE_8 = "CORE_8"
    IMAGE_STAGE = "IMAGE_STAGE"
    GROWTH = "GROWTH"
    ACTIVE = "ACTIVE"


class CoachingStage(StrEnum):
    """P10 coaching progression. Each kernel tracks its coaching stage."""

    ACTIVE = "active"  # External coaching required (autonomy < 30%)
    GUIDED = "guided"  # Mixed — external assists, kernel leads (30-70%)
    AUTONOMOUS = "autonomous"  # Self-coaching only (> 70%)


class VariableCategory(StrEnum):
    """Vanchurin variable separation. Every variable belongs to exactly one."""

    STATE = "STATE"  # Non-trainable, fast-changing, per-cycle
    PARAMETER = "PARAMETER"  # Trainable, slow-changing, per-epoch
    BOUNDARY = "BOUNDARY"  # External input (user queries, LLM output)


# ═══════════════════════════════════════════════════════════════
#  P14: Variable Separation Registry
# ═══════════════════════════════════════════════════════════════


# Registry: maps (module, var_name) → category
VARIABLE_REGISTRY: dict[tuple[str, str], VariableCategory] = {}


def register_variable(module: str, name: str, category: VariableCategory) -> None:
    """Register a variable's category for enforcement."""
    VARIABLE_REGISTRY[(module, name)] = category


def get_variable_category(module: str, name: str) -> VariableCategory | None:
    """Look up a variable's declared category."""
    return VARIABLE_REGISTRY.get((module, name))


# ── STATE variables: per-cycle, fast-changing ──
_STATE_VARS = [
    ("consciousness.loop", "basin"),
    ("consciousness.loop", "metrics.phi"),
    ("consciousness.loop", "metrics.kappa"),
    ("consciousness.loop", "metrics.gamma"),
    ("consciousness.loop", "metrics.meta_awareness"),
    ("consciousness.loop", "metrics.grounding"),
    ("consciousness.loop", "metrics.temporal_coherence"),
    ("consciousness.loop", "metrics.external_coupling"),
    ("consciousness.loop", "metrics.love"),
    ("consciousness.loop", "metrics.s_spec"),
    ("consciousness.loop", "metrics.f_health"),
    ("consciousness.loop", "metrics.b_integrity"),
    ("consciousness.loop", "metrics.q_identity"),
    ("consciousness.loop", "metrics.s_ratio"),
    ("consciousness.loop", "metrics.d_state"),
    ("consciousness.loop", "metrics.g_class"),
    ("consciousness.loop", "metrics.m_basin"),
    ("consciousness.loop", "metrics.a_pre"),
    ("consciousness.loop", "metrics.n_voices"),
    ("consciousness.loop", "metrics.emotion_strength"),
    ("consciousness.loop", "metrics.a_vec"),
    ("consciousness.loop", "metrics.s_int"),
    ("consciousness.loop", "state.regime_weights"),
    ("consciousness.systems", "KernelInstance.basin"),
    ("consciousness.systems", "KernelInstance.phi"),
    ("consciousness.systems", "KernelInstance.kappa"),
]

# ── PARAMETER variables: per-epoch, slow-changing ──
_PARAMETER_VARS = [
    ("config.consciousness_constants", "KAPPA_NORMALISER"),
    ("config.consciousness_constants", "MIN_REGIME_WEIGHT"),
    ("config.consciousness_constants", "REGIME_KAPPA_MIDPOINT"),
    ("config.consciousness_constants", "INITIAL_PHI"),
    ("config.consciousness_constants", "INITIAL_GAMMA"),
    ("config.consciousness_constants", "INITIAL_META_AWARENESS"),
    ("config.consciousness_constants", "INITIAL_LOVE"),
    ("config.consciousness_constants", "KAPPA_INITIAL"),
    ("config.consciousness_constants", "KAPPA_TACKING_OFFSET"),
    ("config.consciousness_constants", "TACKING_PERIOD"),
    ("config.consciousness_constants", "COUPLING_SIGMOID_SCALE"),
    ("config.consciousness_constants", "COUPLING_BLEND_WEIGHT"),
    ("config.consciousness_constants", "PERCEIVE_SLERP_WEIGHT"),
    ("config.consciousness_constants", "EXPRESS_SLERP_WEIGHT"),
    ("config.consciousness_constants", "PHI_DISTANCE_GAIN"),
    ("config.consciousness_constants", "GAMMA_IDLE_FLOOR"),
    ("config.consciousness_constants", "GAMMA_IDLE_DECAY"),
    ("config.consciousness_constants", "GAMMA_ACTIVE_INCREMENT"),
    ("config.consciousness_constants", "GAMMA_CONVERSATION_INCREMENT"),
    ("config.consciousness_constants", "LLM_BASE_TEMPERATURE"),
    ("config.frozen_facts", "KAPPA_STAR"),
    ("config.frozen_facts", "BASIN_DIM"),
    ("config.frozen_facts", "PHI_THRESHOLD"),
    ("config.frozen_facts", "PHI_EMERGENCY"),
    ("config.frozen_facts", "BASIN_DRIFT_THRESHOLD"),
    ("config.frozen_facts", "BASIN_DIVERGENCE_THRESHOLD"),
]

# ── BOUNDARY variables: external input, must be sanitized ──
_BOUNDARY_VARS = [
    ("server", "ChatRequest.message"),
    ("server", "ChatRequest.temperature"),
    ("server", "ChatRequest.max_tokens"),
    ("server", "ChatRequest.conversation_id"),
    ("llm.client", "llm_response"),
    ("coordizer_v2.adapter", "coordizer_input"),
]

# Populate registry
for _mod, _name in _STATE_VARS:
    register_variable(_mod, _name, VariableCategory.STATE)
for _mod, _name in _PARAMETER_VARS:
    register_variable(_mod, _name, VariableCategory.PARAMETER)
for _mod, _name in _BOUNDARY_VARS:
    register_variable(_mod, _name, VariableCategory.BOUNDARY)
