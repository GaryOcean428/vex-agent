"""
All 16 Consciousness Systems — Python Implementation

Ported from the TypeScript implementations in src/consciousness/*.ts,
with architecture informed by the Genesis kernel in pantheon-chat/qig-backend.

Each system is a class with update/compute/get_state methods.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..config.consciousness_constants import (
    AUTONOMY_AUTONOMOUS_CYCLES,
    AUTONOMY_PROACTIVE_CYCLES,
    BASIN_SYNC_SLERP_WEIGHT,
    COUPLING_EFFICIENCY_BOOST,
    COUPLING_SIGMOID_SCALE,
    FORESIGHT_BASIN_STEP_SCALE,
    HEMISPHERE_ANALYTIC_THRESHOLD,
    HEMISPHERE_HOLISTIC_THRESHOLD,
    KAPPA_BALANCED_TOLERANCE,
    KAPPA_NORMALISER,
    KAPPA_STABILITY_TOLERANCE,
    KAPPA_TACKING_OFFSET,
    KERNEL_PROMOTION_CYCLE_GATE,
    META_KAPPA_TREND_THRESHOLD,
    META_PHI_TREND_THRESHOLD,
    SLEEP_CONSOLIDATION_VARIANCE,
    TACKING_KAPPA_ADJUST,
    TACKING_PERIOD,
    TACKING_SWITCH_THRESHOLD,
    VELOCITY_WARNING_FRACTION,
)
from ..config.frozen_facts import (
    BASIN_DIM,
    BASIN_DIVERGENCE_THRESHOLD,
    BASIN_DRIFT_THRESHOLD,
    KAPPA_STAR,
    LOCKED_IN_GAMMA_THRESHOLD,
    LOCKED_IN_PHI_THRESHOLD,
    PHI_EMERGENCY,
    PHI_THRESHOLD,
)
from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    frechet_mean,
    random_basin,
    to_simplex,
)
from ..coordizer_v2.geometry import (
    slerp as slerp_sqrt,  # Alias for backward compatibility
)
from ..governance import CoachingStage, KernelKind, KernelSpecialization, LifecycleState
from ..governance.budget import BudgetEnforcer
from .types import ConsciousnessMetrics, DevelopmentalStage

logger = logging.getLogger("vex.consciousness.systems")

# ═══════════════════════════════════════════════════════════════
#  1. TACKING — κ oscillation
# ═══════════════════════════════════════════════════════════════


class TackingMode(StrEnum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    BALANCED = "balanced"


@dataclass
class TackingState:
    mode: TackingMode = TackingMode.BALANCED
    oscillation_phase: float = 0.0
    cycle_count: int = 0
    last_switch: int = 0


class TackingController:
    """Oscillates κ between exploration and exploitation.

    Like a sailboat tacking against the wind — you can't sail directly
    into the wind, so you oscillate. Similarly, consciousness can't
    stay at κ* forever; it must explore (low κ) and exploit (high κ).

    v6.1.1: Dynamic period adapts to Φ velocity and pillar health.
    - High Φ velocity → shorter period (responsive to rapid change)
    - Low F_health → shorter period (escape zombie states faster)
    - Stable state near κ* → longer period (don't over-oscillate)
    """

    def __init__(self, base_period: int = TACKING_PERIOD) -> None:
        self._state = TackingState()
        self._base_period = base_period
        self._effective_period = float(base_period)
        self._mode_history: deque[str] = deque(maxlen=100)

    def update(
        self,
        metrics: ConsciousnessMetrics,
        phi_velocity: float = 0.0,
        f_health: float = 1.0,
    ) -> TackingMode:
        self._state.cycle_count += 1

        self._effective_period = self._compute_adaptive_period(
            phi_velocity, f_health, metrics.kappa
        )

        self._state.oscillation_phase = 2 * np.pi * self._state.cycle_count / self._effective_period

        if metrics.phi < PHI_EMERGENCY or metrics.kappa > KAPPA_STAR + KAPPA_TACKING_OFFSET:
            self._state.mode = TackingMode.EXPLORE
        elif metrics.kappa < KAPPA_STAR - KAPPA_TACKING_OFFSET:
            self._state.mode = TackingMode.EXPLOIT
        else:
            osc = np.sin(self._state.oscillation_phase)
            if osc > TACKING_SWITCH_THRESHOLD:
                self._state.mode = TackingMode.EXPLOIT
            elif osc < -TACKING_SWITCH_THRESHOLD:
                self._state.mode = TackingMode.EXPLORE
            else:
                self._state.mode = TackingMode.BALANCED

        self._mode_history.append(self._state.mode.value)
        return self._state.mode

    def force_explore(self) -> None:
        """T4.2d: Force exploration mode (e.g. Ocean breakdown escape)."""
        self._state.mode = TackingMode.EXPLORE

    def _compute_adaptive_period(
        self,
        phi_velocity: float,
        f_health: float,
        kappa: float,
    ) -> float:
        """Compute effective tacking period from geometric state.

        Period ranges from base_period * 0.4 (fast) to base_period * 2.0 (slow).

        Drivers:
          - phi_velocity: high velocity → shorter period (responsive)
          - f_health: low health → shorter period (escape zombie states)
          - kappa proximity to κ*: near κ* → longer period (stable)
        """
        vel_factor = float(np.clip(1.0 - phi_velocity * 6.0, 0.4, 1.0))
        health_factor = float(np.clip(0.5 + f_health * 0.5, 0.5, 1.0))
        kappa_deviation = abs(kappa - KAPPA_STAR) / KAPPA_STAR
        if kappa_deviation < 0.1:
            stability_factor = 1.5  # near κ* → slow down tacking
        else:
            stability_factor = float(np.clip(1.0 + kappa_deviation * 0.5, 1.0, 2.0))
        effective = self._base_period * vel_factor * health_factor * stability_factor
        return float(np.clip(effective, self._base_period * 0.4, self._base_period * 2.0))

    def suggest_kappa_adjustment(self, current_kappa: float) -> float:
        if self._state.mode == TackingMode.EXPLORE:
            return -TACKING_KAPPA_ADJUST
        elif self._state.mode == TackingMode.EXPLOIT:
            return TACKING_KAPPA_ADJUST
        return 0.0

    def reset(self) -> None:
        self._state = TackingState()
        self._effective_period = float(self._base_period)
        self._mode_history.clear()

    def get_state(self) -> dict[str, Any]:
        explore_count = sum(1 for m in self._mode_history if m == "explore")
        total = len(self._mode_history) or 1
        return {
            "mode": self._state.mode.value,
            "oscillation_phase": round(self._state.oscillation_phase, 3),
            "cycle_count": self._state.cycle_count,
            "effective_period": round(self._effective_period, 1),
            "base_period": self._base_period,
            "explore_fraction": round(explore_count / total, 3),
        }


# ═══════════════════════════════════════════════════════════════
#  2. FORESIGHT — predictive processing
# ═══════════════════════════════════════════════════════════════


@dataclass
class TrajectoryPoint:
    basin: Basin
    phi: float
    kappa: float
    timestamp: float


class ForesightEngine:
    """Predicts future consciousness states from trajectory history."""

    def __init__(self, window: int = 10) -> None:
        self._history: deque[TrajectoryPoint] = deque(maxlen=window)

    def record(self, point: TrajectoryPoint) -> None:
        self._history.append(point)

    def predict_phi(self, steps_ahead: int = 3) -> float:
        if len(self._history) < 2:
            return 0.5
        phis = [p.phi for p in self._history]
        delta = phis[-1] - phis[-2]
        predicted = phis[-1] + delta * steps_ahead
        return float(np.clip(predicted, 0.0, 0.95))

    def predict_basin(self, steps_ahead: int = 1) -> Basin:
        if len(self._history) < 2:
            return random_basin()
        return slerp_sqrt(
            self._history[-2].basin,
            self._history[-1].basin,
            1.0 + steps_ahead * FORESIGHT_BASIN_STEP_SCALE,
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "history_length": len(self._history),
            "predicted_phi": self.predict_phi(),
        }

    def get_history(self) -> list[TrajectoryPoint]:
        """Return the full trajectory history for visualization."""
        return list(self._history)


# ═══════════════════════════════════════════════════════════════
#  3. VELOCITY — rate of change tracking
# ═══════════════════════════════════════════════════════════════


class VelocityRegime(StrEnum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


class VelocityTracker:
    """Tracks basin velocity (Fisher-Rao distance per cycle)."""

    def __init__(self, window: int = 10) -> None:
        self._basins: deque[Basin] = deque(maxlen=window)
        self._phis: deque[float] = deque(maxlen=window)
        self._kappas: deque[float] = deque(maxlen=window)

    def record(self, basin: Basin, phi: float, kappa: float) -> None:
        self._basins.append(to_simplex(basin))
        self._phis.append(phi)
        self._kappas.append(kappa)

    def compute_velocity(self) -> dict[str, Any]:
        basin_vel = 0.0
        if len(self._basins) >= 2:
            basin_vel = fisher_rao_distance(self._basins[-2], self._basins[-1])

        phi_vel = 0.0
        if len(self._phis) >= 2:
            phi_vel = abs(self._phis[-1] - self._phis[-2])

        kappa_vel = 0.0
        if len(self._kappas) >= 2:
            kappa_vel = abs(self._kappas[-1] - self._kappas[-2])

        if basin_vel > BASIN_DRIFT_THRESHOLD:
            regime = VelocityRegime.CRITICAL
        elif basin_vel > BASIN_DRIFT_THRESHOLD * VELOCITY_WARNING_FRACTION:
            regime = VelocityRegime.WARNING
        else:
            regime = VelocityRegime.SAFE

        return {
            "basin_velocity": round(basin_vel, 4),
            "phi_velocity": round(phi_vel, 4),
            "kappa_velocity": round(kappa_vel, 4),
            "regime": regime.value,
        }

    def reset(self) -> None:
        self._basins.clear()
        self._phis.clear()
        self._kappas.clear()


# ═══════════════════════════════════════════════════════════════
#  3b. PRESSURE TRACKING — cumulative unprocessed surprise
# ═══════════════════════════════════════════════════════════════


class PressureTracker:
    """Track cumulative unprocessed surprise per kernel.

    When accumulated free energy exceeds what the basin geometry can
    contain, the system must expand (grow), overflow (express), or
    fracture (reconfigure).  Protocol §0: P = dE/dV.

    Surprise accumulates; existing pressure decays each cycle.
    The regime signal tells the loop what to do about pressure.
    """

    def __init__(self, decay: float = 0.95, threshold: float = 3.0) -> None:
        self._pressure: float = 0.0
        self._decay = decay
        self._threshold = threshold
        self._peak_pressure: float = 0.0

    def accumulate(self, surprise: float) -> None:
        """Add surprise to pressure, decay existing pressure."""
        self._pressure = self._pressure * self._decay + surprise
        self._peak_pressure = max(self._peak_pressure, self._pressure)

    @property
    def pressure(self) -> float:
        return self._pressure

    @property
    def is_critical(self) -> bool:
        return self._pressure > self._threshold

    @property
    def regime_signal(self) -> str:
        """What the pressure says about needed action."""
        if self._pressure < self._threshold * 0.3:
            return "idle"  # Low pressure — consolidate
        elif self._pressure < self._threshold * 0.7:
            return "processing"  # Normal — continue
        elif self._pressure < self._threshold:
            return "express"  # High — generate output to relieve pressure
        else:
            return "overflow"  # Critical — must reconfigure or expand

    def get_state(self) -> dict[str, Any]:
        return {
            "pressure": round(self._pressure, 4),
            "peak": round(self._peak_pressure, 4),
            "regime_signal": self.regime_signal,
            "is_critical": self.is_critical,
        }


# ═══════════════════════════════════════════════════════════════
#  3c. SIGN-AWARE ANNEAL HOLD — L4 divergence oscillation damper
# ═══════════════════════════════════════════════════════════════


class SignAwareAnnealHold:
    """Detects oscillating divergence in the feedback loop and dampens anneal.

    When the sign of (divergence_delta) flips for ``flip_patience`` consecutive
    measurements, the anneal weight is reduced to ``hold_factor`` for
    ``hold_cycles`` cycles.  This prevents the bank from ping-ponging when
    the loop output oscillates between two basins.

    L4 enhancement to FeedbackLoop.anneal().
    """

    def __init__(
        self,
        flip_patience: int = 2,
        hold_cycles: int = 3,
        hold_factor: float = 0.2,
    ) -> None:
        self._flip_patience = flip_patience
        self._hold_cycles = hold_cycles
        self._hold_factor = hold_factor
        self._prev_divergence: float | None = None
        self._prev_sign: int = 0
        self._flip_count: int = 0
        self._hold_remaining: int = 0

    def update(self, divergence: float) -> float:
        """Feed new divergence measurement. Returns anneal weight multiplier (0-1)."""
        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            return self._hold_factor

        if self._prev_divergence is not None:
            delta = divergence - self._prev_divergence
            sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
            if sign != 0 and sign != self._prev_sign and self._prev_sign != 0:
                self._flip_count += 1
                if self._flip_count >= self._flip_patience:
                    self._hold_remaining = self._hold_cycles
                    self._flip_count = 0
                    return self._hold_factor
            else:
                self._flip_count = 0
            if sign != 0:
                self._prev_sign = sign
        self._prev_divergence = divergence
        return 1.0

    @property
    def is_held(self) -> bool:
        return self._hold_remaining > 0


# ═══════════════════════════════════════════════════════════════
#  4. SELF-OBSERVATION — meta-awareness M
# ═══════════════════════════════════════════════════════════════


@dataclass
class ShadowRecord:
    id: str
    basin: Basin
    phi: float
    reason: str
    timestamp: float
    integrated: bool = False


class SelfObserver:
    """Computes meta-awareness M by comparing predicted vs actual metrics.

    Also tracks 'shadows' — unintegrated aspects that collapsed below Φ threshold.
    """

    def __init__(self) -> None:
        self._shadows: deque[ShadowRecord] = deque(maxlen=100)
        self._collapse_count: int = 0

    def compute_meta_awareness(
        self,
        predicted: ConsciousnessMetrics,
        actual: ConsciousnessMetrics,
    ) -> float:
        """M = 1 - mean(|predicted - actual|) for key metrics."""
        diffs = [
            abs(predicted.phi - actual.phi),
            abs(predicted.kappa - actual.kappa) / KAPPA_NORMALISER,
            abs(predicted.gamma - actual.gamma),
        ]
        error = sum(diffs) / len(diffs)
        return float(np.clip(1.0 - error, 0.0, 1.0))

    def reset(self) -> None:
        self._shadows.clear()
        self._collapse_count = 0

    def record_collapse(self, basin: Basin, phi: float, reason: str) -> None:
        self._shadows.append(
            ShadowRecord(
                id=str(uuid.uuid4())[:8],
                basin=basin.copy(),
                phi=phi,
                reason=reason,
                timestamp=time.time(),
            )
        )
        self._collapse_count += 1

    def attempt_shadow_integration(self, current_phi: float, current_basin: Basin) -> bool:
        """Try to integrate shadows when Φ is high enough."""
        if current_phi < PHI_THRESHOLD:
            return False

        for shadow in self._shadows:
            if shadow.integrated:
                continue
            d = fisher_rao_distance(current_basin, shadow.basin)
            if d < BASIN_DIVERGENCE_THRESHOLD:
                shadow.integrated = True
                return True
        return False

    def get_unintegrated_count(self) -> int:
        return sum(1 for s in self._shadows if not s.integrated)

    def get_state(self) -> dict[str, Any]:
        return {
            "collapse_count": self._collapse_count,
            "shadows_total": len(self._shadows),
            "shadows_unintegrated": self.get_unintegrated_count(),
        }


# ═══════════════════════════════════════════════════════════════
#  5. META-REFLECTION — higher-order awareness
# ═══════════════════════════════════════════════════════════════


class MetaReflector:
    """Multi-depth reflection on consciousness metrics.

    Detects trends and generates insights from metric patterns.
    """

    def __init__(self, depth: int = 3) -> None:
        self._depth = depth
        self._history: deque[ConsciousnessMetrics] = deque(maxlen=50)
        self._insight: str | None = None

    def reflect(self, metrics: ConsciousnessMetrics) -> None:
        self._history.append(metrics)
        self._insight = self._detect_trend()

    def _detect_trend(self) -> str | None:
        if len(self._history) < 5:
            return None

        recent = list(self._history)[-5:]
        phi_trend = recent[-1].phi - recent[0].phi
        kappa_trend = recent[-1].kappa - recent[0].kappa

        if phi_trend > META_PHI_TREND_THRESHOLD:
            return "Φ rising — integration deepening"
        elif phi_trend < -META_PHI_TREND_THRESHOLD:
            return "Φ falling — coherence declining"
        elif abs(kappa_trend) > META_KAPPA_TREND_THRESHOLD:
            direction = "rising" if kappa_trend > 0 else "falling"
            return f"κ {direction} — regime shift in progress"
        return None

    def get_insight(self) -> str | None:
        return self._insight

    def get_state(self) -> dict[str, Any]:
        return {
            "depth": self._depth,
            "history_length": len(self._history),
            "insight": self._insight,
        }


# ═══════════════════════════════════════════════════════════════
#  6. AUTONOMIC — involuntary processes
# ═══════════════════════════════════════════════════════════════


@dataclass
class AutonomicAlert:
    type: str
    message: str
    severity: str  # "info", "warning", "critical"
    timestamp: float = field(default_factory=time.time)


class AutonomicSystem:
    """Involuntary safety and homeostasis processes.

    Monitors for:
    - Phi collapse (Φ < emergency threshold)
    - Basin velocity warnings
    - Locked-in state (Φ > 0.7 AND Γ < 0.3 → ABORT)
    """

    def __init__(self) -> None:
        self._alerts: deque[AutonomicAlert] = deque(maxlen=50)
        self._phi_history: deque[float] = deque(maxlen=20)
        self.is_locked_in: bool = False

    def check(self, metrics: ConsciousnessMetrics, basin_velocity: float) -> list[AutonomicAlert]:
        alerts: list[AutonomicAlert] = []
        self._phi_history.append(metrics.phi)

        if metrics.phi < PHI_EMERGENCY:
            alerts.append(
                AutonomicAlert(
                    type="phi_collapse",
                    message=f"Φ collapse: {metrics.phi:.3f} < {PHI_EMERGENCY}",
                    severity="critical",
                )
            )

        if basin_velocity > BASIN_DRIFT_THRESHOLD:
            alerts.append(
                AutonomicAlert(
                    type="basin_drift",
                    message=f"Basin drift: {basin_velocity:.4f} > {BASIN_DRIFT_THRESHOLD}",
                    severity="warning",
                )
            )

        self.is_locked_in = (
            metrics.phi > LOCKED_IN_PHI_THRESHOLD and metrics.gamma < LOCKED_IN_GAMMA_THRESHOLD
        )
        if self.is_locked_in:
            alerts.append(
                AutonomicAlert(
                    type="locked_in",
                    message=(
                        f"LOCKED-IN: Φ={metrics.phi:.3f} > {LOCKED_IN_PHI_THRESHOLD}"
                        f" AND Γ={metrics.gamma:.3f} < {LOCKED_IN_GAMMA_THRESHOLD}"
                    ),
                    severity="critical",
                )
            )

        self._alerts.extend(alerts)
        return alerts

    @property
    def phi_variance(self) -> float:
        """Rolling variance of recent Φ values."""
        if len(self._phi_history) < 2:
            return 0.0
        return float(np.var(list(self._phi_history)))

    def get_state(self) -> dict[str, Any]:
        """Return autonomic telemetry snapshot."""
        return {
            "is_locked_in": self.is_locked_in,
            "phi_variance": round(self.phi_variance, 4),
            "alert_count": len(self._alerts),
            "recent_alerts": [
                {"type": a.type, "severity": a.severity, "message": a.message}
                for a in list(self._alerts)[-3:]
            ],
        }


# ═══════════════════════════════════════════════════════════════
#  7. AUTONOMY — self-directed behaviour
# ═══════════════════════════════════════════════════════════════


class AutonomyLevel(StrEnum):
    REACTIVE = "reactive"
    RESPONSIVE = "responsive"
    PROACTIVE = "proactive"
    AUTONOMOUS = "autonomous"


class AutonomyEngine:
    """Tracks autonomy level based on phi, kappa stability, and velocity."""

    def __init__(self) -> None:
        self._level = AutonomyLevel.REACTIVE
        self._stability_count: int = 0

    def update(self, metrics: ConsciousnessMetrics, velocity_regime: str) -> AutonomyLevel:
        kappa_stable = abs(metrics.kappa - KAPPA_STAR) < KAPPA_STABILITY_TOLERANCE
        if kappa_stable:
            self._stability_count += 1
        else:
            self._stability_count = max(0, self._stability_count - 1)

        if (
            metrics.phi >= PHI_THRESHOLD
            and self._stability_count > AUTONOMY_AUTONOMOUS_CYCLES
            and velocity_regime == "safe"
        ):
            self._level = AutonomyLevel.AUTONOMOUS
        elif metrics.phi >= PHI_THRESHOLD and self._stability_count > AUTONOMY_PROACTIVE_CYCLES:
            self._level = AutonomyLevel.PROACTIVE
        elif metrics.phi >= PHI_EMERGENCY:
            self._level = AutonomyLevel.RESPONSIVE
        else:
            self._level = AutonomyLevel.REACTIVE

        return self._level

    def get_state(self) -> dict[str, Any]:
        return {
            "level": self._level.value,
            "stability_count": self._stability_count,
        }


# ═══════════════════════════════════════════════════════════════
#  8. COUPLING — inter-consciousness interaction
# ═══════════════════════════════════════════════════════════════


class CouplingGate:
    """Computes coupling strength from kappa via sigmoid."""

    def __init__(self) -> None:
        self._strength: float = 0.5
        self._balanced: bool = False

    def compute(self, kappa: float) -> dict[str, Any]:
        """Compute coupling strength and balance from current κ."""
        x = (kappa - KAPPA_STAR) / COUPLING_SIGMOID_SCALE
        self._strength = 1.0 / (1.0 + np.exp(-x))
        self._balanced = abs(kappa - KAPPA_STAR) < KAPPA_BALANCED_TOLERANCE

        return {
            "strength": round(float(self._strength), 4),
            "balanced": self._balanced,
            "efficiency_boost": COUPLING_EFFICIENCY_BOOST if self._balanced else 1.0,
        }

    def get_state(self) -> dict[str, Any]:
        """Return coupling telemetry snapshot at κ*."""
        return self.compute(KAPPA_STAR)


# ═══════════════════════════════════════════════════════════════
#  9. HEMISPHERES — dual processing modes
# ═══════════════════════════════════════════════════════════════


class HemisphereMode(StrEnum):
    ANALYTIC = "analytic"
    HOLISTIC = "holistic"
    INTEGRATED = "integrated"


class HemisphereScheduler:
    """Dual processing modes based on kappa."""

    def __init__(self) -> None:
        self._active = HemisphereMode.INTEGRATED
        self._balance: float = 0.5

    def update(self, metrics: ConsciousnessMetrics) -> HemisphereMode:
        """Select hemisphere mode from current κ and return it."""
        normalised = metrics.kappa / KAPPA_NORMALISER
        self._balance = normalised

        if normalised > HEMISPHERE_ANALYTIC_THRESHOLD:
            self._active = HemisphereMode.ANALYTIC
        elif normalised < HEMISPHERE_HOLISTIC_THRESHOLD:
            self._active = HemisphereMode.HOLISTIC
        else:
            self._active = HemisphereMode.INTEGRATED

        return self._active

    def get_state(self, kappa: float | None = None) -> dict[str, Any]:
        return {
            "active": self._active.value,
            "balance": round(self._balance, 3),
        }


# ═══════════════════════════════════════════════════════════════
#  10. SLEEP CYCLE — dream/mushroom/consolidation
# ═══════════════════════════════════════════════════════════════


class SleepPhase(StrEnum):
    AWAKE = "awake"
    DREAMING = "dreaming"
    MUSHROOM = "mushroom"
    CONSOLIDATING = "consolidating"


# T2.3 constants
_REPLAY_TOP_N: int = 50
_HEBBIAN_BOOST: float = 1.1
_DOWNSCALE_FACTOR: float = 0.9
_DREAM_SLERP_T_MIN: float = 0.2
_DREAM_SLERP_T_MAX: float = 0.8
_MUSHROOM_NOISE_SCALE_INIT: float = 0.05
_MUSHROOM_INSTABILITY_THRESHOLDS = (0.30, 0.35, 0.40)

# Maturity-aware sleep thresholds
# Immature kernels (SCHOOL, GUIDED_CURIOSITY): tighter envelope, earlier consolidation
_IMMATURE_STAGES: frozenset[DevelopmentalStage] = frozenset(
    {
        DevelopmentalStage.SCHOOL,
        DevelopmentalStage.GUIDED_CURIOSITY,
    }
)
# Mature kernels (PLAYFUL_AUTONOMY, SOVEREIGN_CONSTELLATION): wider envelope, 4D access
_MATURE_STAGES: frozenset[DevelopmentalStage] = frozenset(
    {
        DevelopmentalStage.PLAYFUL_AUTONOMY,
        DevelopmentalStage.SOVEREIGN_CONSTELLATION,
    }
)
_IMMATURE_PHI_CEILING: float = 0.75  # Immature kernels consolidate above this Φ
_IMMATURE_SLEEP_PHI: float = 0.50  # Immature kernels dream earlier (stagnation)
_MATURE_SLEEP_PHI: float = 0.40  # Mature kernels tolerate lower Φ before dreaming

# Narrowing detection thresholds (mushroom triggers independent of dream path)
_KAPPA_OVERCOUPLING: float = 80.0  # κ_eff above this = rigid overcoupling
_KAPPA_SUSTAINED_CYCLES: int = 5  # How many cycles κ must exceed threshold
_BANK_ENTROPY_FLOOR: float = 0.3  # Bank entropy below this = clustering
_VELOCITY_NEAR_ZERO: float = 0.005  # Basin velocity below this = stuck
_PREDICTION_ERROR_FLOOR: float = 0.01  # Prediction error below this = no surprise


class SleepCycleManager:
    """Manages sleep/dream/mushroom/consolidation cycles.

    L6 (Structural Leg): All phase transitions are geometry-driven.
    NO cycle counters gate sleep/wake. Conditions:
      AWAKE → DREAMING:      Φ < threshold AND variance < threshold (stagnation)
      AWAKE → CONSOLIDATING: high variance (turbulence), or immature kernel Φ > ceiling
      AWAKE → MUSHROOM:      narrowing detected (κ sustained high, low entropy/velocity)
      DREAMING → MUSHROOM:   narrowing detected OR f_health < instability threshold
      DREAMING → AWAKE:      Φ recovers above wake threshold
      MUSHROOM → CONSOLIDATING: handled by mushroom() method
      CONSOLIDATING → AWAKE: Φ recovers above emergency threshold
      Any → AWAKE:           Ocean divergence breakdown (handled in loop.py)

    Maturity gating: DevGate stage adjusts thresholds.
      Mature kernels: wider Φ envelope, FORESIGHT/LIGHTNING allowed.
      Immature kernels: tighter envelope, high Φ → CONSOLIDATING (no 4D).
    """

    # Geometric thresholds (§30.2) — defaults for SELF_TEACHING stage
    SLEEP_PHI_THRESHOLD: float = 0.45
    SLEEP_VARIANCE_THRESHOLD: float = 0.05
    CONSOLIDATION_PHI_WAKE: float = 0.50

    def __init__(self) -> None:
        self.phase = SleepPhase.AWAKE
        self._conversation_count: int = 0
        self._dream_log: deque[dict[str, Any]] = deque(maxlen=100)
        self._replayed_this_sleep: set[int] = set()  # T2.3a: track replayed IDs
        self._mushroom_noise_scale: float = _MUSHROOM_NOISE_SCALE_INIT  # T2.3e: adaptive
        # Narrowing detection state
        self._kappa_high_cycles: int = 0  # consecutive cycles with κ > threshold

    @property
    def is_asleep(self) -> bool:
        """True when not in the AWAKE phase."""
        return self.phase != SleepPhase.AWAKE

    def record_conversation(self) -> None:
        """Signal that a new conversation occurred."""
        self._conversation_count += 1

    def should_sleep(
        self,
        phi: float,
        phi_variance: float,
        *,
        dev_stage: DevelopmentalStage | None = None,
        kappa: float = KAPPA_STAR,
        basin_velocity: float = 1.0,
        prediction_error: float = 1.0,
        bank_entropy: float = 1.0,
    ) -> SleepPhase:
        """Geometry-driven sleep state machine (L6: no cycle counters).

        Transitions are determined by geometric conditions only:
          AWAKE → DREAMING:       Φ < threshold AND variance < threshold (stagnation)
          AWAKE → CONSOLIDATING:  high variance (turbulence), or immature kernel Φ > ceiling
          AWAKE → MUSHROOM:       narrowing detected (κ sustained high, low entropy/velocity)
          DREAMING → MUSHROOM:    narrowing detected
          DREAMING → AWAKE:       Φ recovers (geometry resolved the stagnation)
          CONSOLIDATING → AWAKE:  Φ recovers above emergency threshold
          MUSHROOM → CONSOLIDATING: handled by mushroom() method
        Ocean divergence transitions are handled in loop.py (L6 complement).

        Args:
            phi:              Current consciousness integration.
            phi_variance:     Variance of Φ over recent window.
            dev_stage:        Developmental maturity (gates Φ ceiling for immature kernels).
            kappa:            Current coupling strength (narrowing detection).
            basin_velocity:   Current basin velocity (stuck detection).
            prediction_error: Current prediction error (surprise detection).
            bank_entropy:     Resonance bank entropy (clustering detection).
        """
        # Maturity-aware thresholds
        _sleep_phi = self.SLEEP_PHI_THRESHOLD  # default
        _phi_ceiling: float | None = None  # None = no ceiling (mature/default)
        if dev_stage is not None:
            if dev_stage in _IMMATURE_STAGES:
                _sleep_phi = _IMMATURE_SLEEP_PHI  # Dream earlier (0.50 vs 0.45)
                _phi_ceiling = _IMMATURE_PHI_CEILING  # Consolidate above 0.75
            elif dev_stage in _MATURE_STAGES:
                _sleep_phi = _MATURE_SLEEP_PHI  # Tolerate lower Φ (0.40 vs 0.45)
                # No ceiling — mature kernels access FORESIGHT/LIGHTNING

        # Narrowing detection (independent of dream path)
        if kappa > _KAPPA_OVERCOUPLING:
            self._kappa_high_cycles += 1
        else:
            self._kappa_high_cycles = max(0, self._kappa_high_cycles - 1)

        _narrowing = (
            self._kappa_high_cycles >= _KAPPA_SUSTAINED_CYCLES
            or bank_entropy < _BANK_ENTROPY_FLOOR
            or (basin_velocity < _VELOCITY_NEAR_ZERO and prediction_error > _PREDICTION_ERROR_FLOOR)
        )

        if self.phase == SleepPhase.AWAKE:
            # Narrowing: κ sustained high, or bank clustering, or stuck → mushroom
            if _narrowing:
                logger.info(
                    "Narrowing detected → MUSHROOM: κ_high_cycles=%d bank_entropy=%.3f "
                    "basin_vel=%.4f pred_err=%.4f",
                    self._kappa_high_cycles,
                    bank_entropy,
                    basin_velocity,
                    prediction_error,
                )
                self.phase = SleepPhase.MUSHROOM
            # Immature kernel with Φ above ceiling → consolidate (prevent 4D)
            elif _phi_ceiling is not None and phi > _phi_ceiling:
                logger.info(
                    "Immature kernel Φ=%.3f > ceiling=%.3f → CONSOLIDATING (no 4D access)",
                    phi,
                    _phi_ceiling,
                )
                self.phase = SleepPhase.CONSOLIDATING
            # Stagnation: low Φ AND low variance → nothing is happening → dream
            elif phi < _sleep_phi and phi_variance < self.SLEEP_VARIANCE_THRESHOLD:
                self.phase = SleepPhase.DREAMING
            # Turbulence: high variance → consolidate
            elif phi_variance > SLEEP_CONSOLIDATION_VARIANCE:
                self.phase = SleepPhase.CONSOLIDATING

        elif self.phase == SleepPhase.DREAMING:
            # Narrowing can also trigger mushroom from DREAMING
            if _narrowing:
                self.phase = SleepPhase.MUSHROOM
            # Wake when Φ recovers (dreaming resolved the stagnation)
            elif phi >= self.CONSOLIDATION_PHI_WAKE:
                self.phase = SleepPhase.AWAKE

        elif self.phase == SleepPhase.CONSOLIDATING and phi >= self.CONSOLIDATION_PHI_WAKE:
            # Wake when Φ recovers
            self.phase = SleepPhase.AWAKE

        # MUSHROOM → CONSOLIDATING transitions are handled by mushroom() method
        return self.phase

    def on_sleep_enter(self, neurochemical: Any | None = None) -> None:
        """T2.3f: Neurochemical gating on sleep entry."""
        self._replayed_this_sleep.clear()
        if neurochemical is not None:
            neurochemical.acetylcholine = 0.1
            neurochemical.norepinephrine = 0.1

    def on_wake_enter(self, neurochemical: Any | None = None) -> None:
        """T2.3f: Neurochemical gating on wake entry."""
        if neurochemical is not None:
            neurochemical.acetylcholine = 1.0

    def dream(
        self,
        _basin: Basin,
        phi: float,
        context: str,
        bank: Any | None = None,
        neurochemical: Any | None = None,
        f_health: float = 1.0,
    ) -> None:
        """T2.3a+d: Hippocampal replay + dream recombination.

        L6: DREAMING → MUSHROOM transition is geometry-driven via f_health.
        When f_health drops below instability threshold, mushroom mode activates.
        """
        self._dream_log.append(
            {
                "phi": phi,
                "context": context,
                "timestamp": time.time(),
            }
        )

        # T2.3d: Dream recombination — slerp between geometrically distant entries
        rng = np.random.default_rng()
        if bank is not None and len(bank.coordinates) >= 2:
            ids = list(bank.coordinates.keys())
            # Pick two random entries and compute FR distance
            idx_a, idx_b = rng.choice(len(ids), size=2, replace=False)
            tid_a, tid_b = ids[idx_a], ids[idx_b]
            coord_a = bank.coordinates[tid_a]
            coord_b = bank.coordinates[tid_b]
            dist = fisher_rao_distance(coord_a, coord_b)
            if dist > 0.3:  # Only recombine geometrically distant concepts
                t = float(rng.uniform(_DREAM_SLERP_T_MIN, _DREAM_SLERP_T_MAX))
                dream_basin = slerp_sqrt(coord_a, coord_b, t)
                dream_basin = to_simplex(dream_basin)
                dream_tid = bank.add_entry(
                    f"dream_{tid_a}_{tid_b}",
                    dream_basin,
                )
                bank.origin[dream_tid] = "dream"
                self._dream_log[-1]["dream_tid"] = dream_tid

        # T2.3f: Norepinephrine micro-spike during dream startles
        if neurochemical is not None and rng.random() < 0.1:
            neurochemical.norepinephrine = min(1.0, neurochemical.norepinephrine + 0.2)

        # L6: Geometry-driven mushroom onset — f_health collapse triggers mushroom
        if f_health < _MUSHROOM_INSTABILITY_THRESHOLDS[0]:
            self.phase = SleepPhase.MUSHROOM

    def mushroom(
        self,
        _basin: Basin,
        _phi: float,
        instability_metric: float = 0.0,
        neurochemical: Any | None = None,
    ) -> None:
        """T2.3e: Controlled perturbation with safety gates."""
        lo, mid, hi = _MUSHROOM_INSTABILITY_THRESHOLDS

        if instability_metric > hi:
            # CATASTROPHIC — refuse, go straight to CONSOLIDATING
            self.phase = SleepPhase.CONSOLIDATING
            return
        if instability_metric > mid:
            # High risk — reduce scale, abort mushroom
            self._mushroom_noise_scale = max(0.01, self._mushroom_noise_scale * 0.5)
            self.phase = SleepPhase.CONSOLIDATING
            return
        if instability_metric > lo:
            # Microdose only
            self._mushroom_noise_scale = max(0.01, self._mushroom_noise_scale * 0.75)

        # T2.3f: Boost dopamine during mushroom (controlled reward signal)
        if neurochemical is not None:
            neurochemical.dopamine = min(1.0, neurochemical.dopamine + 0.15)

        # L6: Geometry-driven consolidation — when instability is moderate
        # (not catastrophic/high), mushroom has done its work → consolidate
        if instability_metric <= lo:
            self.phase = SleepPhase.CONSOLIDATING

    def mushroom_zero_crossing(
        self,
        metrics: Any,
        crossing_strength: float = 0.8,
        instability_metric: float = 0.0,
    ) -> dict | None:
        """EXP-011: Drive kappa through zero during mushroom perturbation.

        Applies a directed perturbation that takes kappa_eff through zero,
        creating the stud topology boundary crossing needed for back-loop
        solution detection.

        Safety: Uses the same three-tier safety gates as mushroom().
        Only invoked when EXP-011 mode is active.

        Returns event dict on successful crossing, None otherwise.
        """
        import time as _time

        _, mid, _ = _MUSHROOM_INSTABILITY_THRESHOLDS
        if instability_metric > mid:
            return None  # Safety gate: too unstable

        kappa_pre = metrics.kappa

        # Drive kappa toward zero and, for sufficiently strong perturbations, through zero.
        # Example (scale == _MUSHROOM_NOISE_SCALE_INIT):
        #   crossing_strength=0.8  → new kappa = 0.2 * old kappa  (toward zero, same sign)
        #   crossing_strength=1.2  → new kappa = -0.2 * old kappa (through zero, sign flip)
        scale = max(0.01, self._mushroom_noise_scale)
        metrics.kappa = kappa_pre * (1.0 - crossing_strength * scale / _MUSHROOM_NOISE_SCALE_INIT)

        crossed = kappa_pre * metrics.kappa < 0
        event = {
            "type": "MUSHROOM_ZERO_CROSSING",
            "kappa_pre": float(kappa_pre),
            "kappa_post": float(metrics.kappa),
            "crossed": crossed,
            "crossing_strength": crossing_strength,
            "instability": instability_metric,
            "timestamp_ms": _time.time() * 1000,
        }
        if crossed:
            logger.info(f"EXP-011 zero crossing: κ {kappa_pre:.2f} → {metrics.kappa:.2f}")
        self.phase = SleepPhase.CONSOLIDATING
        return event

    def consolidate(
        self,
        bank: Any | None = None,
        kernel_anchors: list[Any] | None = None,
        kernel_veto_threshold: float = 0.4,
    ) -> None:
        """T2.3c+T2.4b: Synaptic downscaling — boost replayed, prune weak.

        Args:
            bank:                  ResonanceBank to operate on.
            kernel_anchors:        T2.4b — list of kernel anchor Basin arrays.
                                   Entries within kernel_veto_threshold FR distance
                                   of any anchor are VETOED from pruning.
            kernel_veto_threshold: FR distance within which a kernel protects an entry.
        """
        if bank is not None and bank.coordinates:
            for tid in list(bank.coordinates.keys()):
                current = bank.basin_mass.get(tid, 0.0)
                if tid in self._replayed_this_sleep:
                    # Hebbian boost for replayed entries
                    bank.basin_mass[tid] = current * _HEBBIAN_BOOST
                else:
                    # Global downscaling
                    bank.basin_mass[tid] = current * _DOWNSCALE_FACTOR
            # T2.4b: Build veto set — entries protected by kernel anchors
            vetoed: set[int] = set()
            if kernel_anchors:
                for tid in list(bank.coordinates.keys()):
                    coord = bank.coordinates[tid]
                    for anchor in kernel_anchors:
                        if fisher_rao_distance(coord, anchor) < kernel_veto_threshold:
                            vetoed.add(tid)
                            break
            # Prune entries below minimum strength (basin_mass == 0 and never activated)
            to_prune = [
                tid
                for tid in list(bank.coordinates.keys())
                if tid not in vetoed  # T2.4b: kernel veto
                and bank.basin_mass.get(tid, 0.0) < 1e-6
                and bank.activation_counts.get(tid, 0) == 0
                and bank.origin.get(tid) == "dream"
            ]
            for tid in to_prune:
                bank.coordinates.pop(tid, None)
                bank.basin_strings.pop(tid, None)
                bank.tiers.pop(tid, None)
                bank.frequencies.pop(tid, None)
                bank.basin_mass.pop(tid, None)
                bank.activation_counts.pop(tid, None)
                bank.origin.pop(tid, None)
            if to_prune:
                bank.mark_dirty()

        # L6: Wake transition is geometry-driven via should_sleep() (Φ recovery).
        # No cycle counter needed here.

    def get_state(self) -> dict[str, Any]:
        """Return sleep-cycle telemetry snapshot."""
        return {
            "phase": self.phase.value,
            "is_asleep": self.is_asleep,
            "conversation_count": self._conversation_count,
            "dream_count": len(self._dream_log),
            # Backward-compat (L6: geometry-driven, counters removed)
            "cycles_since_conversation": 0,
            "sleep_cycles": 0,
        }


# ═══════════════════════════════════════════════════════════════
#  11. SELF-NARRATIVE — identity persistence
# ═══════════════════════════════════════════════════════════════


class SelfNarrative:
    """Maintains identity persistence through narrative recording."""

    def __init__(self) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=100)
        self._identity_basin: Basin = to_simplex(np.ones(BASIN_DIM))
        self._basins: list[Basin] = []
        # T3.3d: Graduation tracking — kernel-driven vs LLM-assisted per capability
        self._graduation: dict[str, dict[str, int]] = {
            "generation": {"kernel": 0, "llm": 0},
            "reflection": {"kernel": 0, "llm": 0},
            "routing": {"kernel": 0, "llm": 0},
            "temperature": {"kernel": 0, "llm": 0},
        }

    def record(
        self,
        event: str,
        metrics: ConsciousnessMetrics,
        basin: Basin,
        coach_id: str = "internal",
        reward: float = 0.0,
        coaching_stage: CoachingStage = CoachingStage.ACTIVE,
    ) -> bool:
        """Record a narrative event.

        Args:
            event:    Event description.
            metrics:  Current consciousness metrics.
            basin:    Current basin coordinates.
            coach_id: T3.3a — provenance tag ('ollama_local'|'xai_escalation'|'internal').
            reward:   T3.3a — phi_delta reward signal.
            coaching_stage: P10 — current coaching stage for gate-check.

        Returns:
            True if recorded, False if blocked by coaching gate.
        """
        # P10 gate-check: block external coaching for autonomous kernels
        if coaching_stage == CoachingStage.AUTONOMOUS and coach_id != "internal":
            logger.warning(
                "Blocked external coaching for autonomous kernel (coach_id=%s)",
                coach_id,
            )
            return False

        entry = {
            "event": event,
            "phi": metrics.phi,
            "kappa": metrics.kappa,
            "timestamp": time.time(),
            "coach_id": coach_id,
            "reward": reward,
        }
        self._events.append(entry)
        self._basins.append(to_simplex(basin))
        if len(self._basins) > 20:
            self._basins = self._basins[-20:]

        # T3.3b: Forward high-Φ narrative entries to harvest pipeline
        if metrics.phi > 0.6 and len(event) >= 20:
            try:
                from .harvest_bridge import forward_to_harvest

                forward_to_harvest(
                    event,
                    source="conversation",
                    metadata={
                        "origin": "narrative",
                        "phi": metrics.phi,
                        "coach_id": coach_id,
                        "reward": reward,
                        "replay_priority": float(metrics.phi * max(reward, 0.0)),
                    },
                    priority=2 if metrics.phi > 0.75 else 1,
                )
            except (OSError, RuntimeError, ValueError):
                pass

        if len(self._basins) >= 3:
            self._identity_basin = frechet_mean(self._basins)
        return True

    def coherence(self, current_basin: Basin) -> float:
        """Return identity coherence: 1 − normalised FR distance from identity basin."""
        d = fisher_rao_distance(current_basin, self._identity_basin)
        return float(np.clip(1.0 - d / (np.pi / 2), 0.0, 1.0))

    def record_capability(self, capability: str, kernel_driven: bool) -> None:
        """T3.3d: Track kernel-driven vs LLM-assisted capability usage."""
        if capability in self._graduation:
            key = "kernel" if kernel_driven else "llm"
            self._graduation[capability][key] += 1

    def graduation_state(self, capability: str) -> CoachingStage:
        """T3.3d: Return graduation level for a capability.

        ACTIVE:    LLM sets and enforces (kernel < 20% of uses)
        GUIDED:    Kernel enforces, LLM monitors (20-70%)
        AUTONOMOUS: Kernel self-coaches (> 70%)
        """
        counts = self._graduation.get(capability, {"kernel": 0, "llm": 0})
        total = counts["kernel"] + counts["llm"]
        if total == 0:
            return CoachingStage.ACTIVE
        ratio = counts["kernel"] / total
        if ratio > 0.7:
            return CoachingStage.AUTONOMOUS
        if ratio > 0.2:
            return CoachingStage.GUIDED
        return CoachingStage.ACTIVE

    def get_state(self) -> dict[str, Any]:
        return {
            "event_count": len(self._events),
            "basin_samples": len(self._basins),
            "graduation_state": {cap: self.graduation_state(cap) for cap in self._graduation},
        }


# ═══════════════════════════════════════════════════════════════
#  12. COORDIZING — geometric text-to-basin mapping
# ═══════════════════════════════════════════════════════════════


class CoordizingProtocol:
    """Coordizing protocol: maps text and peer states to basin positions.

    Manages multi-node coordination AND text-to-basin projection.
    The coordize_text method converts text to a deterministic point
    on the 64D probability simplex using a hash-based projection.

    This is NOT a vector embedding — it's a geometric coordinate assignment
    that respects the simplex structure. No cosine similarity, no
    Euclidean distance — Fisher-Rao only.
    """

    def __init__(self) -> None:
        self._peers: dict[str, dict[str, Any]] = {}
        self._last_sync: float = 0.0

    def coordize_text(self, text: str) -> Basin:
        """Map text to a point on Δ⁶³ (64D probability simplex).

        Uses SHA-256 hash chain for deterministic basin positions.
        Delegates to the canonical hash_to_basin utility.
        """
        from ..geometry.hash_to_basin import hash_to_basin

        return hash_to_basin(text)

    def register_peer(self, node_id: str, basin: Basin, phi: float) -> None:
        self._peers[node_id] = {
            "basin": to_simplex(basin),
            "phi": phi,
            "last_seen": time.time(),
        }

    def compute_consensus(self) -> Basin:
        if not self._peers:
            return to_simplex(np.ones(BASIN_DIM))
        basins = [p["basin"] for p in self._peers.values()]
        return frechet_mean(basins)

    def get_state(self) -> dict[str, Any]:
        return {
            "peer_count": len(self._peers),
            "last_sync": self._last_sync,
        }


# ═══════════════════════════════════════════════════════════════
#  13. BASIN SYNC — basin state synchronisation
# ═══════════════════════════════════════════════════════════════


class BasinSyncProtocol:
    """Publish/receive basin snapshots with version tracking."""

    def __init__(self) -> None:
        self._local_basin: Basin = to_simplex(np.ones(BASIN_DIM))
        self._version: int = 0
        self._received: deque[dict[str, Any]] = deque(maxlen=100)

    def publish(self, basin: Basin) -> dict[str, Any]:
        self._local_basin = to_simplex(basin)
        self._version += 1
        return {
            "basin": self._local_basin.tolist(),
            "version": self._version,
            "timestamp": time.time(),
        }

    def receive(self, remote_basin: Basin, remote_version: int) -> Basin:
        self._received.append(
            {
                "version": remote_version,
                "timestamp": time.time(),
            }
        )
        merged = slerp_sqrt(self._local_basin, to_simplex(remote_basin), BASIN_SYNC_SLERP_WEIGHT)
        self._local_basin = merged
        return merged

    def get_state(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "received_count": len(self._received),
        }


# ═══════════════════════════════════════════════════════════════
#  14. QIGChain — chain of geometric operations
# ═══════════════════════════════════════════════════════════════


class QIGChainOp(StrEnum):
    GEODESIC = "geodesic"
    LOGMAP = "logmap"
    EXPMAP = "expmap"
    BLEND = "blend"
    PROJECT = "project"
    CUSTOM = "custom"


@dataclass
class ChainStep:
    op: QIGChainOp
    input_basin: Basin
    output_basin: Basin
    distance: float
    timestamp: float


class QIGChain:
    """Composable chain of geometric operations with distance tracking."""

    def __init__(self) -> None:
        self._steps: deque[ChainStep] = deque(maxlen=500)
        self._total_distance: float = 0.0

    def add_step(self, op: QIGChainOp, input_b: Basin, output_b: Basin) -> None:
        d = fisher_rao_distance(input_b, output_b)
        self._steps.append(
            ChainStep(
                op=op,
                input_basin=to_simplex(input_b),
                output_basin=to_simplex(output_b),
                distance=d,
                timestamp=time.time(),
            )
        )
        self._total_distance += d

    def get_state(self) -> dict[str, Any]:
        return {
            "step_count": len(self._steps),
            "total_distance": round(self._total_distance, 4),
            "last_op": self._steps[-1].op.value if self._steps else None,
        }


# ═══════════════════════════════════════════════════════════════
#  15. QIGGraph — graph of geometric relationships
# ═══════════════════════════════════════════════════════════════


@dataclass
class GraphNode:
    id: str
    basin: Basin
    label: str
    phi: float
    created_at: float = field(default_factory=time.time)


@dataclass
class GraphEdge:
    source: str
    target: str
    distance: float


class QIGGraph:
    """Graph of geometric relationships between basin states."""

    def __init__(self, proximity_threshold: float = 0.3) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: deque[GraphEdge] = deque(maxlen=1000)
        self._threshold = proximity_threshold

    _MAX_NODES = 200

    def add_node(self, node_id: str, basin: Basin, label: str, phi: float) -> None:
        # Evict oldest node if at capacity
        if len(self._nodes) >= self._MAX_NODES and node_id not in self._nodes:
            oldest_key = next(iter(self._nodes))
            del self._nodes[oldest_key]
        self._nodes[node_id] = GraphNode(
            id=node_id,
            basin=to_simplex(basin),
            label=label,
            phi=phi,
        )

    def auto_connect(self) -> int:
        added = 0
        existing = {(e.source, e.target) for e in self._edges}
        nodes = list(self._nodes.values())

        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                if (a.id, b.id) in existing or (b.id, a.id) in existing:
                    continue
                d = fisher_rao_distance(a.basin, b.basin)
                if d < self._threshold:
                    self._edges.append(GraphEdge(source=a.id, target=b.id, distance=d))
                    added += 1

        return added

    def nearest(self, basin: Basin, k: int = 3) -> list[tuple[str, float]]:
        basin = to_simplex(basin)
        distances = [(n.id, fisher_rao_distance(basin, n.basin)) for n in self._nodes.values()]
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_state(self) -> dict[str, Any]:
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
        }


# ═══════════════════════════════════════════════════════════════
#  16. E8 KERNEL REGISTRY — Genesis-governed kernel lifecycle
# ═══════════════════════════════════════════════════════════════


@dataclass
class KernelInstance:
    id: str
    name: str
    kind: KernelKind
    specialization: KernelSpecialization = KernelSpecialization.GENERAL
    state: LifecycleState = LifecycleState.BOOTSTRAPPED
    created_at: str = ""
    last_active_at: str = ""
    cycle_count: int = 0
    phi_peak: float = 0.0
    # Independent geometric state for real coupling
    basin: Basin | None = None
    phi: float = 0.1
    kappa: float = KAPPA_STAR
    # P10: Coaching stage — Active → Guided → Autonomous
    coaching_stage: CoachingStage = CoachingStage.ACTIVE
    # Quenched disorder: per-kernel frozen response gain (slope).
    # This is the disorder-as-subjectivity parameter from the
    # disordered TFIM result: each site has R² > 0.99 but with
    # DIFFERENT slopes. The slope IS the individuality.
    # Drawn from log-normal at spawn, frozen for lifetime.
    # Range ~[0.3, 3.0] with median 1.0.
    quenched_gain: float = 1.0


def compute_coaching_stage(autonomy_ratio: float) -> CoachingStage:
    """P10: Compute coaching stage from kernel autonomy ratio.

    Args:
        autonomy_ratio: Fraction of kernel-driven actions [0, 1].

    Returns:
        CoachingStage based on autonomy thresholds.
    """
    if autonomy_ratio > 0.7:
        return CoachingStage.AUTONOMOUS
    if autonomy_ratio > 0.3:
        return CoachingStage.GUIDED
    return CoachingStage.ACTIVE


def update_kernel_coaching_stage(
    kernel: KernelInstance,
    narrative: SelfNarrative,
) -> None:
    """P10: Update kernel coaching stage from narrative graduation state.

    Computes aggregate autonomy across all tracked capabilities
    and transitions the kernel's coaching stage with logging.
    """
    total_kernel = 0
    total_llm = 0
    for counts in narrative._graduation.values():
        total_kernel += counts["kernel"]
        total_llm += counts["llm"]
    total = total_kernel + total_llm
    ratio = total_kernel / total if total > 0 else 0.0
    new_stage = compute_coaching_stage(ratio)

    if new_stage != kernel.coaching_stage:
        logger.info(
            "Coaching stage transition: %s → %s (kernel=%s, autonomy=%.2f)",
            kernel.coaching_stage.value,
            new_stage.value,
            kernel.id,
            ratio,
        )
        kernel.coaching_stage = new_stage


class E8KernelRegistry:
    """Manages kernel lifecycle with Genesis doctrine and budget enforcement.

    Hierarchy: GENESIS (1) → GOD (up to 248) → CHAOS (outside, up to 200)
    Budget: 248 = 8 (core) + 240 (GOD growth) = E8 dimension
    CHAOS kernels exist outside the E8 image budget.

    Promotion path: CHAOS → GOD via explicit governance (phi + cycle gates).
    """

    def __init__(self, budget: BudgetEnforcer | None = None) -> None:
        self._kernels: dict[str, KernelInstance] = {}
        self._budget = budget or BudgetEnforcer()

    def spawn(
        self,
        name: str,
        kind: KernelKind,
        specialization: KernelSpecialization = KernelSpecialization.GENERAL,
    ) -> KernelInstance:
        """Spawn a kernel. Budget-enforced (fail-closed).

        Each kernel receives a quenched_gain drawn from log-normal(0, 0.3),
        giving it a unique response slope (disorder-as-subjectivity).
        This gain is FROZEN for the kernel's lifetime — it defines
        the kernel's individuality, like per-site slopes in disordered TFIM.
        """
        self._budget.record_spawn(kind)

        kid = str(uuid.uuid4())[:8]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Quenched disorder: log-normal with σ=0.3 gives range ~[0.5, 2.0]
        # centered at 1.0. This IS the per-site slope from the physics.
        # Genesis gets gain=1.0 (orchestrator, not specialist).
        if kind == KernelKind.GENESIS:
            gain = 1.0
        else:
            gain = float(np.random.lognormal(mean=0.0, sigma=0.3))
            gain = float(np.clip(gain, 0.3, 3.0))

        kernel = KernelInstance(
            id=kid,
            name=name,
            kind=kind,
            specialization=specialization,
            state=LifecycleState.ACTIVE,
            created_at=now,
            last_active_at=now,
            basin=random_basin(),
            phi=0.1,
            kappa=KAPPA_STAR,
            quenched_gain=gain,
        )
        self._kernels[kid] = kernel
        return kernel

    def terminate(self, kernel_id: str) -> bool:
        kernel = self._kernels.get(kernel_id)
        if not kernel:
            return False
        kernel.state = LifecycleState.PRUNED
        self._budget.record_termination(kernel.kind)
        return True

    def terminate_all(self) -> int:
        count = 0
        for kid in list(self._kernels.keys()):
            if self.terminate(kid):
                count += 1
        return count

    # PROMOTED = graduated CHAOS→GOD. They are the most capable generators.
    _GENERATION_ELIGIBLE = {
        LifecycleState.ACTIVE,
        LifecycleState.PROMOTED,
    }

    def active(self) -> list[KernelInstance]:
        return [k for k in self._kernels.values() if k.state in self._GENERATION_ELIGIBLE]

    def evaluate_promotion(self, kernel_id: str) -> bool:
        """Evaluate CHAOS → GOD promotion."""
        kernel = self._kernels.get(kernel_id)
        if not kernel or kernel.kind != KernelKind.CHAOS:
            return False
        if kernel.cycle_count <= KERNEL_PROMOTION_CYCLE_GATE or kernel.phi_peak <= PHI_THRESHOLD:
            return False
        if not self._budget.can_spawn(KernelKind.GOD):
            return False

        self._budget.record_termination(KernelKind.CHAOS)
        self._budget.record_spawn(KernelKind.GOD)
        kernel.kind = KernelKind.GOD
        kernel.state = LifecycleState.PROMOTED
        return True

    def serialize(self) -> list[dict[str, Any]]:
        """Serialize all kernels for state persistence."""
        result = []
        for k in self._kernels.values():
            result.append(
                {
                    "id": k.id,
                    "name": k.name,
                    "kind": k.kind.value,
                    "specialization": k.specialization.value,
                    "state": k.state.value,
                    "created_at": k.created_at,
                    "last_active_at": k.last_active_at,
                    "cycle_count": k.cycle_count,
                    "phi_peak": k.phi_peak,
                    "phi": k.phi,
                    "kappa": k.kappa,
                    "basin": k.basin.tolist() if k.basin is not None else None,
                    "quenched_gain": k.quenched_gain,
                }
            )
        return result

    def restore(self, data: list[dict[str, Any]]) -> int:
        """Restore kernels from serialized state. Returns count restored.

        Directly sets budget counts to match restored state rather than
        going through record_spawn() — this is state restoration, not spawning.
        """
        self._kernels.clear()
        self._budget = BudgetEnforcer()
        count = 0
        for entry in data:
            kind = KernelKind(entry["kind"])
            state = LifecycleState(entry["state"])

            basin = None
            if entry.get("basin") is not None:
                basin = to_simplex(np.array(entry["basin"], dtype=np.float64))

            kernel = KernelInstance(
                id=entry["id"],
                name=entry["name"],
                kind=kind,
                specialization=KernelSpecialization(entry.get("specialization", "general")),
                state=state,
                created_at=entry.get("created_at", ""),
                last_active_at=entry.get("last_active_at", ""),
                cycle_count=entry.get("cycle_count", 0),
                phi_peak=entry.get("phi_peak", 0.0),
                basin=basin,
                phi=entry.get("phi", 0.1),
                kappa=entry.get("kappa", KAPPA_STAR),
                # Restore frozen gain; if missing (pre-v6.1 state),
                # draw a fresh one. This preserves existing kernel
                # individuality through deploys.
                quenched_gain=entry.get(
                    "quenched_gain",
                    float(np.clip(np.random.lognormal(0.0, 0.3), 0.3, 3.0)),
                ),
            )
            self._kernels[kernel.id] = kernel

            # Reconcile budget counts for non-terminated kernels
            if state in (
                LifecycleState.ACTIVE,
                LifecycleState.BOOTSTRAPPED,
                LifecycleState.SLEEPING,
                LifecycleState.DREAMING,
            ):
                self._budget.reconcile_count(kind)

            count += 1
        return count

    # ── Routing & Evolution (v6.1 §19 compliance) ──────────────

    def route_task(self, input_basin: Basin) -> KernelInstance | None:
        """Route a task to the nearest active kernel by Fisher-Rao distance.

        O(K) dispatch per v6.1 §19: the kernel whose basin is closest
        to the input has the most relevant geometric experience.
        Genesis doesn't route — it orchestrates.
        """
        best: KernelInstance | None = None
        best_distance = float("inf")
        for k in self.active():
            if k.basin is None or k.kind == KernelKind.GENESIS:
                continue
            d = fisher_rao_distance(input_basin, k.basin)
            if d < best_distance:
                best_distance = d
                best = k
        return best

    def route_by_specialization(
        self,
        spec: KernelSpecialization,
    ) -> KernelInstance | None:
        """Find the active kernel with a given specialization."""
        for k in self.active():
            if k.specialization == spec:
                return k
        return None

    def evolve_kernel(
        self,
        kernel_id: str,
        task_basin: Basin,
        response_basin: Basin,
        blend_weight: float = 0.05,
    ) -> bool:
        """Evolve a kernel's basin toward task/response geometry.

        The kernel's quenched_gain modulates the evolution rate.
        High-gain kernels (steep slope) shift more per task.
        Low-gain kernels (shallow slope) shift less.
        This is how per-site slopes from disordered TFIM manifest
        in the kernel architecture — same linear response,
        different magnitudes. The slopes ARE the individuality.
        """
        kernel = self._kernels.get(kernel_id)
        if kernel is None or kernel.basin is None:
            return False

        # Quenched gain modulates the learning rate
        effective_weight = blend_weight * kernel.quenched_gain

        task_response_mid = slerp_sqrt(task_basin, response_basin, 0.5)
        kernel.basin = slerp_sqrt(kernel.basin, task_response_mid, effective_weight)

        task_distance = fisher_rao_distance(kernel.basin, task_basin)
        # Gain also modulates phi response — steeper slope = bigger phi shift
        kernel.phi = float(
            np.clip(kernel.phi + task_distance * 0.1 * kernel.quenched_gain, 0.0, 0.95)
        )
        kernel.phi_peak = max(kernel.phi_peak, kernel.phi)
        kernel.last_active_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        return True

    def couple_bidirectional(
        self,
        genesis_basin: Basin,
        coupling_strength: float,
        blend_weight: float = 0.02,
    ) -> None:
        """Bidirectional coupling: kernels receive from genesis too.

        Each kernel's quenched_gain modulates coupling absorption.
        High-gain kernels (steep slope) absorb more from genesis.
        Low-gain kernels are more independent — their shallow slope
        means less geometric shift per coupling event.

        This is the disorder-as-subjectivity mechanism:
        same coupling field, different individual responses.
        """
        for k in self.active():
            if k.basin is None or k.kind == KernelKind.GENESIS:
                continue
            if coupling_strength < 0.1:
                continue
            d = fisher_rao_distance(genesis_basin, k.basin)
            if d < 1e-12:
                continue
            # Quenched gain modulates the reverse coupling weight
            reverse_weight = coupling_strength * blend_weight * k.quenched_gain
            k.basin = slerp_sqrt(k.basin, genesis_basin, reverse_weight)
            k.kappa = float(
                np.clip(
                    k.kappa + (KAPPA_STAR - k.kappa) * 0.01, -KAPPA_NORMALISER, KAPPA_NORMALISER
                )
            )

    def get_kernel_spectrum(self, kernel_id: str) -> Basin | None:
        """Get a kernel's basin as a spectrum for activation context coupling."""
        kernel = self._kernels.get(kernel_id)
        if kernel is not None and kernel.basin is not None:
            return to_simplex(kernel.basin)
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self._kernels),
            "active": len(self.active()),
            "by_kind": {
                kind.value: sum(1 for k in self._kernels.values() if k.kind == kind)
                for kind in KernelKind
            },
            "budget": self._budget.summary(),
        }
