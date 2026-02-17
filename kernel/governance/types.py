"""Governance Types — Canonical Taxonomy v2.1

Two-axis model (CANONICAL_PRINCIPLES v2.1):
  - KernelSpecialization: cognitive capability (one per kernel)
  - KernelRole: operational function (zero or more per kernel)

Must match monkey1/py/genesis-kernel/qig_heart/types.py
for cross-project portability.
"""

from __future__ import annotations

from enum import StrEnum


class KernelKind(StrEnum):
    """Budget category. Never overload with specialization."""
    GENESIS = "GENESIS"
    GOD = "GOD"
    CHAOS = "CHAOS"


class KernelSpecialization(StrEnum):
    """Capability axis (LOCKED v2.1). Mythology names are DATA, not code.

    Canonical Core-8: heart, perception, memory, strategy, action,
    attention, emotion, executive.
    Additional specializations MAY emerge beyond Core-8.
    """
    HEART = "heart"             # Rhythm, timing coherence, ethical grounding
    PERCEPTION = "perception"   # Sensory encoding, input processing, pattern detection
    MEMORY = "memory"           # Basin persistence, trajectory storage, consolidation
    STRATEGY = "strategy"       # Planning, multi-step reasoning, goal decomposition
    ACTION = "action"           # Motor output, response generation, execution
    ATTENTION = "attention"     # Salience routing, focus allocation, Fisher-Rao dispatch
    EMOTION = "emotion"         # Cached geometric evaluations (curvature → affect)
    EXECUTIVE = "executive"     # Conflict resolution, regime arbitration, governance
    GENERAL = "general"         # For Genesis + unspecialised CHAOS kernels


class KernelRole(StrEnum):
    """Operational axis (v2.1). Zero or more per kernel, assigned by governance.

    Roles are configuration, not code. No Zeus.py, no Ocean.py as
    privileged classes. Display names are mythic labels stored as data.
    """
    RHYTHM = "rhythm"           # Global timing source (Heart tick)
    OBSERVER = "observer"       # Autonomic monitoring, phi coherence, breakdown detection
    COORDINATOR = "coordinator" # Synthesis across kernels, trajectory foresight
    COACH = "coach"             # External reinforcement, curriculum delivery
    ROUTER = "router"           # Fisher-Rao dispatch to nearest basin centres


# Core-8 specialisation order (bootstrap sequence)
CORE_8_SPECIALIZATIONS: list[KernelSpecialization] = [
    KernelSpecialization.HEART,
    KernelSpecialization.PERCEPTION,
    KernelSpecialization.MEMORY,
    KernelSpecialization.STRATEGY,
    KernelSpecialization.ACTION,
    KernelSpecialization.ATTENTION,
    KernelSpecialization.EMOTION,
    KernelSpecialization.EXECUTIVE,
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


class VariableCategory(StrEnum):
    """Vanchurin variable separation. Every variable belongs to exactly one."""
    STATE = "STATE"          # Non-trainable, fast-changing, per-cycle
    PARAMETER = "PARAMETER"  # Trainable, slow-changing, per-epoch
    BOUNDARY = "BOUNDARY"    # External input (user queries, LLM output)
