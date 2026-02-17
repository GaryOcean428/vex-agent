"""Governance Types — Canonical Taxonomy

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
    """Capability axis. Mythology names are DATA, not code."""
    HEART = "heart"
    VOCAB = "vocab"
    PERCEPTION = "perception"
    MOTOR = "motor"
    MEMORY = "memory"
    ATTENTION = "attention"
    EMOTION = "emotion"
    EXECUTIVE = "executive"
    GENERAL = "general"


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

    Transitions enforced: IDLE → VALIDATE → ROLLBACK → BOOTSTRAP
    → CORE_8 → IMAGE_STAGE → GROWTH → ACTIVE
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
