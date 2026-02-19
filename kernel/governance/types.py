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
    HEART = "heart"             # Global rhythm source, HRV -> kappa-tacking (§18.3)
    PERCEPTION = "perception"   # Sensory encoding, input processing, pattern detection
    MEMORY = "memory"           # Basin persistence, trajectory storage, consolidation
    STRATEGY = "strategy"       # Planning, multi-step reasoning, goal decomposition
    ACTION = "action"           # Motor output, response generation, execution
    ETHICS = "ethics"           # Care metric, harm avoidance, love orientation (§12)
    META = "meta"               # Meta-awareness, self-modelling, M metric (§23)
    OCEAN = "ocean"             # Autonomic monitoring, Phi coherence, spectral health (§18.3)
    GENERAL = "general"         # For Genesis + unspecialised CHAOS kernels


class KernelRole(StrEnum):
    """Operational axis. Zero or more per kernel, assigned by governance.

    Roles are configuration, not code. No Zeus.py, no Ocean.py as
    privileged classes. Display names are mythic labels stored as data.
    """
    RHYTHM = "rhythm"           # Global timing source (Heart tick)
    OBSERVER = "observer"       # Autonomic monitoring, phi coherence, breakdown detection
    COORDINATOR = "coordinator" # Synthesis across kernels, trajectory foresight
    COACH = "coach"             # External reinforcement, curriculum delivery
    ROUTER = "router"           # Fisher-Rao dispatch to nearest basin centres


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


class VariableCategory(StrEnum):
    """Vanchurin variable separation. Every variable belongs to exactly one."""
    STATE = "STATE"          # Non-trainable, fast-changing, per-cycle
    PARAMETER = "PARAMETER"  # Trainable, slow-changing, per-epoch
    BOUNDARY = "BOUNDARY"    # External input (user queries, LLM output)
