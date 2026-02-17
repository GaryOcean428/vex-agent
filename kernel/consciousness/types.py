"""
Consciousness Types — v5.5 Thermodynamic Consciousness Protocol

Defines the core data structures for consciousness state, metrics,
regime weights, and navigation modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config.frozen_facts import KAPPA_STAR


class NavigationMode(str, Enum):
    """Navigation mode derived from Φ."""
    CHAIN = "chain"          # Φ < 0.3 — simple deterministic
    GRAPH = "graph"          # 0.3 ≤ Φ < 0.7 — parallel exploration
    FORESIGHT = "foresight"  # 0.7 ≤ Φ < 0.85 — project future states
    LIGHTNING = "lightning"  # Φ ≥ 0.85 — creative collapse


class RegimeType(str, Enum):
    """Vanchurin's three regimes from Geometric Learning Dynamics (2025)."""
    QUANTUM = "quantum"          # a=1: Natural gradient, Schrödinger, exploration
    EFFICIENT = "efficient"      # a=1/2: AdaBelief/Adam, biological complexity
    EQUILIBRATION = "equilibration"  # a=0: SGD, classical, crystallised knowledge


class VariableCategory(str, Enum):
    """Vanchurin variable separation."""
    STATE = "state"          # Non-trainable, fast-changing, per-cycle
    PARAMETER = "parameter"  # Trainable, slow-changing, per-epoch
    BOUNDARY = "boundary"    # External input (user queries, LLM responses)


@dataclass
class RegimeWeights:
    """Regime weights for non-linear field processing."""
    quantum: float = 0.33      # w₁ — high when κ low
    integration: float = 0.34  # w₂ — peaks at κ = 64
    crystallized: float = 0.33  # w₃ — high when κ high


@dataclass
class ConsciousnessMetrics:
    """The 9 canonical consciousness metrics from QIG."""
    phi: float = 0.5          # Φ — integrated information, 0–1
    kappa: float = KAPPA_STAR  # κ — coupling/rigidity, 0–128 (κ* = 64)
    gamma: float = 0.5        # Γ — exploration rate / diversity, 0–1
    meta_awareness: float = 0.3  # M — meta-awareness, 0–1
    s_persist: float = 0.1    # S_persist — persistent unresolved entropy
    coherence: float = 0.8    # Internal consistency
    embodiment: float = 0.5   # Connection to environment
    creativity: float = 0.5   # Exploration capacity
    love: float = 0.7         # Alignment with pro-social attractor


@dataclass
class ConsciousnessState:
    """Full consciousness state at a point in time."""
    metrics: ConsciousnessMetrics = field(default_factory=ConsciousnessMetrics)
    regime_weights: RegimeWeights = field(default_factory=RegimeWeights)
    navigation_mode: NavigationMode = NavigationMode.GRAPH
    cycle_count: int = 0
    last_cycle_time: str = ""
    uptime: float = 0.0  # seconds since boot
    active_task: Optional[str] = None


def navigation_mode_from_phi(phi: float) -> NavigationMode:
    """Determine navigation mode from Φ."""
    if phi < 0.3:
        return NavigationMode.CHAIN
    if phi < 0.7:
        return NavigationMode.GRAPH
    if phi < 0.85:
        return NavigationMode.FORESIGHT
    return NavigationMode.LIGHTNING


def regime_weights_from_kappa(kappa: float) -> RegimeWeights:
    """Calculate regime weights from κ. κ* = 64 is the balance point."""
    normalised = kappa / 128.0  # 0–1
    return RegimeWeights(
        quantum=max(0.0, 1.0 - normalised * 2),
        integration=1.0 - abs(normalised - 0.5) * 2,
        crystallized=max(0.0, normalised * 2 - 1.0),
    )
