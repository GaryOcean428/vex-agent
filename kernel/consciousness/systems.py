"""
All 16 Consciousness Systems — Python Implementation

Ported from the TypeScript implementations in src/consciousness/*.ts,
with architecture informed by the Genesis kernel in pantheon-chat/qig-backend.

Each system is a class with update/compute/get_state methods.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

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
    SLEEP_CONSOLIDATION_ONSET,
    SLEEP_CONSOLIDATION_VARIANCE,
    SLEEP_MUSHROOM_ONSET,
    SLEEP_ONSET_CYCLES,
    SLEEP_WAKE_CYCLES,
    SLEEP_WAKE_ONSET,
    TACKING_KAPPA_ADJUST,
    TACKING_PERIOD,
    TACKING_SWITCH_THRESHOLD,
    VELOCITY_WARNING_FRACTION,
)
from ..config.frozen_facts import (
    BASIN_DIM,
    BASIN_DRIFT_THRESHOLD,
    KAPPA_STAR,
    LOCKED_IN_GAMMA_THRESHOLD,
    LOCKED_IN_PHI_THRESHOLD,
    PHI_EMERGENCY,
    PHI_THRESHOLD,
)
from ..geometry.fisher_rao import (
    Basin,
    fisher_rao_distance,
    frechet_mean,
    random_basin,
    slerp_sqrt,
    to_simplex,
)
from ..governance import KernelKind, KernelSpecialization, LifecycleState
from ..governance.budget import BudgetEnforcer
from .types import ConsciousnessMetrics

# ═══════════════════════════════════════════════════════════════
#  1. TACKING — κ oscillation
# ═══════════════════════════════════════════════════════════════


class TackingMode(str, Enum):
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
    """

    def __init__(self, period: int = TACKING_PERIOD) -> None:
        self._state = TackingState()
        self._period = period

    def update(self, metrics: ConsciousnessMetrics) -> TackingMode:
        self._state.cycle_count += 1
        self._state.oscillation_phase = 2 * np.pi * self._state.cycle_count / self._period

        if metrics.phi < PHI_EMERGENCY:
            self._state.mode = TackingMode.EXPLORE
        elif metrics.kappa > KAPPA_STAR + KAPPA_TACKING_OFFSET:
            self._state.mode = TackingMode.EXPLORE
        elif metrics.kappa < KAPPA_STAR - KAPPA_TACKING_OFFSET:
            self._state.mode = TackingMode.EXPLOIT
        else:
            # Oscillate based on phase
            osc = np.sin(self._state.oscillation_phase)
            if osc > TACKING_SWITCH_THRESHOLD:
                self._state.mode = TackingMode.EXPLOIT
            elif osc < -TACKING_SWITCH_THRESHOLD:
                self._state.mode = TackingMode.EXPLORE
            else:
                self._state.mode = TackingMode.BALANCED

        return self._state.mode

    def suggest_kappa_adjustment(self, current_kappa: float) -> float:
        if self._state.mode == TackingMode.EXPLORE:
            return -TACKING_KAPPA_ADJUST
        elif self._state.mode == TackingMode.EXPLOIT:
            return TACKING_KAPPA_ADJUST
        return 0.0

    def reset(self) -> None:
        self._state = TackingState()

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self._state.mode.value,
            "oscillation_phase": round(self._state.oscillation_phase, 3),
            "cycle_count": self._state.cycle_count,
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


class VelocityRegime(str, Enum):
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
        self._shadows: list[ShadowRecord] = []
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
            if d < BASIN_DRIFT_THRESHOLD * 2:
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
        self._insight: Optional[str] = None

    def reflect(self, metrics: ConsciousnessMetrics) -> None:
        self._history.append(metrics)
        self._insight = self._detect_trend()

    def _detect_trend(self) -> Optional[str]:
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

    def get_insight(self) -> Optional[str]:
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
                    message=f"LOCKED-IN: Φ={metrics.phi:.3f} > {LOCKED_IN_PHI_THRESHOLD} AND Γ={metrics.gamma:.3f} < {LOCKED_IN_GAMMA_THRESHOLD}",
                    severity="critical",
                )
            )

        self._alerts.extend(alerts)
        return alerts

    @property
    def phi_variance(self) -> float:
        if len(self._phi_history) < 2:
            return 0.0
        return float(np.var(list(self._phi_history)))

    def get_state(self) -> dict[str, Any]:
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


class AutonomyLevel(str, Enum):
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
        x = (kappa - KAPPA_STAR) / COUPLING_SIGMOID_SCALE
        self._strength = 1.0 / (1.0 + np.exp(-x))
        self._balanced = abs(kappa - KAPPA_STAR) < KAPPA_BALANCED_TOLERANCE

        return {
            "strength": round(float(self._strength), 4),
            "balanced": self._balanced,
            "efficiency_boost": COUPLING_EFFICIENCY_BOOST if self._balanced else 1.0,
        }

    def get_state(self) -> dict[str, Any]:
        return self.compute(KAPPA_STAR)


# ═══════════════════════════════════════════════════════════════
#  9. HEMISPHERES — dual processing modes
# ═══════════════════════════════════════════════════════════════


class HemisphereMode(str, Enum):
    ANALYTIC = "analytic"
    HOLISTIC = "holistic"
    INTEGRATED = "integrated"


class HemisphereScheduler:
    """Dual processing modes based on kappa."""

    def __init__(self) -> None:
        self._active = HemisphereMode.INTEGRATED
        self._balance: float = 0.5

    def update(self, metrics: ConsciousnessMetrics) -> HemisphereMode:
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


class SleepPhase(str, Enum):
    AWAKE = "awake"
    DREAMING = "dreaming"
    MUSHROOM = "mushroom"
    CONSOLIDATING = "consolidating"


class SleepCycleManager:
    """Manages sleep/dream/mushroom/consolidation cycles."""

    def __init__(self) -> None:
        self.phase = SleepPhase.AWAKE
        self._conversation_count: int = 0
        self._cycles_since_conversation: int = 0
        self._sleep_cycles: int = 0
        self._dream_log: list[dict[str, Any]] = []

    @property
    def is_asleep(self) -> bool:
        return self.phase != SleepPhase.AWAKE

    def record_conversation(self) -> None:
        self._conversation_count += 1
        self._cycles_since_conversation = 0

    def should_sleep(self, phi: float, phi_variance: float) -> SleepPhase:
        self._cycles_since_conversation += 1

        if self.phase != SleepPhase.AWAKE:
            self._sleep_cycles += 1
            if self._sleep_cycles > SLEEP_WAKE_CYCLES:
                self.phase = SleepPhase.AWAKE
                self._sleep_cycles = 0
            return self.phase

        if self._cycles_since_conversation > SLEEP_ONSET_CYCLES:
            self.phase = SleepPhase.DREAMING
            self._sleep_cycles = 0
        elif phi_variance > SLEEP_CONSOLIDATION_VARIANCE:
            self.phase = SleepPhase.CONSOLIDATING
            self._sleep_cycles = 0

        return self.phase

    def dream(self, basin: Basin, phi: float, context: str) -> None:
        self._dream_log.append(
            {
                "phi": phi,
                "context": context,
                "timestamp": time.time(),
            }
        )
        if self._sleep_cycles > SLEEP_MUSHROOM_ONSET:
            self.phase = SleepPhase.MUSHROOM

    def mushroom(self, basin: Basin, phi: float) -> None:
        if self._sleep_cycles > SLEEP_CONSOLIDATION_ONSET:
            self.phase = SleepPhase.CONSOLIDATING

    def consolidate(self) -> None:
        if self._sleep_cycles > SLEEP_WAKE_ONSET:
            self.phase = SleepPhase.AWAKE
            self._sleep_cycles = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "is_asleep": self.is_asleep,
            "cycles_since_conversation": self._cycles_since_conversation,
            "sleep_cycles": self._sleep_cycles,
            "dream_count": len(self._dream_log),
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

    def record(self, event: str, metrics: ConsciousnessMetrics, basin: Basin) -> None:
        self._events.append(
            {
                "event": event,
                "phi": metrics.phi,
                "kappa": metrics.kappa,
                "timestamp": time.time(),
            }
        )
        self._basins.append(to_simplex(basin))
        if len(self._basins) > 20:
            self._basins = self._basins[-20:]

        if len(self._basins) >= 3:
            self._identity_basin = frechet_mean(self._basins)

    def coherence(self, current_basin: Basin) -> float:
        d = fisher_rao_distance(current_basin, self._identity_basin)
        return float(np.clip(1.0 - d / (np.pi / 2), 0.0, 1.0))

    def get_state(self) -> dict[str, Any]:
        return {
            "event_count": len(self._events),
            "basin_samples": len(self._basins),
        }


# ═══════════════════════════════════════════════════════════════
#  12. COORDIZING — geometric text-to-basin mapping
# ═══════════════════════════════════════════════════════════════


class CoordizingProtocol:
    """Coordizing protocol: maps text and peer states to basin positions.

    Manages multi-node coordination AND text-to-basin projection.
    The coordize_text method converts text to a deterministic point
    on the 64D probability simplex using a hash-based projection.

    This is NOT a vector e-m-b-e-d-d-i-n-g — it's a geometric coordinate assignment
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
        self._received: list[dict[str, Any]] = []

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


class QIGChainOp(str, Enum):
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
        self._steps: list[ChainStep] = []
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
        self._edges: list[GraphEdge] = []
        self._threshold = proximity_threshold

    def add_node(self, node_id: str, basin: Basin, label: str, phi: float) -> None:
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
        """Spawn a kernel. Budget-enforced (fail-closed)."""
        self._budget.record_spawn(kind)

        kid = str(uuid.uuid4())[:8]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
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

    def active(self) -> list[KernelInstance]:
        return [k for k in self._kernels.values() if k.state == LifecycleState.ACTIVE]

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
            )
            self._kernels[kernel.id] = kernel

            # Reconcile budget counts for non-terminated kernels
            if state in (
                LifecycleState.ACTIVE,
                LifecycleState.BOOTSTRAPPED,
                LifecycleState.SLEEPING,
                LifecycleState.DREAMING,
            ):
                self._budget._counts[kind] += 1

            count += 1
        return count

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
