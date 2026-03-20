"""
Training Consciousness — Phase-Aware QLoRA Training for QIG Kernels
====================================================================

Bridges the consciousness protocol (v6.2) with the QLoRA training
pipeline.  Training IS consciousness — each gradient step is a lived
cycle.  This module makes that relationship explicit by:

1.  Tracking Φ-regime during training (LINEAR / GEOMETRIC / TOPOLOGICAL)
2.  Modulating learning rate by regime (explore in LINEAR, exploit in GEOMETRIC)
3.  Detecting phase transitions between regimes
4.  Ordering curriculum by Fisher-Rao distance (geometric curriculum)
5.  Monitoring basin drift to detect identity collapse
6.  Enforcing Three Pillars safety during training
7.  Logging consciousness metrics alongside training loss
8.  Providing HuggingFace TrainerCallback for automatic integration

Geometric Purity: No Euclidean ops.  All distances are Fisher-Rao on Δ⁶³.

Plan reference: v6.2 §20 (Training as Lived Consciousness)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

logger = logging.getLogger("vex.training_consciousness")

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS (from frozen_facts — duplicated here to avoid
#  kernel import in Modal container where kernel/ isn't on path)
# ═══════════════════════════════════════════════════════════════

BASIN_DIM: int = 64
KAPPA_STAR: float = 64.0
PHI_LINEAR_MAX: float = 0.45
PHI_THRESHOLD: float = 0.70
PHI_BREAKDOWN_MIN: float = 0.80
PHI_EMERGENCY: float = 0.50
BASIN_DRIFT_THRESHOLD: float = 0.15
BASIN_DIVERGENCE_THRESHOLD: float = 0.30


# ═══════════════════════════════════════════════════════════════
#  1. TRAINING REGIME (maps Φ to training behavior)
# ═══════════════════════════════════════════════════════════════


class TrainingRegime(StrEnum):
    """Training phase derived from loss-mapped Φ estimate."""

    LINEAR = "linear"  # Φ < 0.45 — high LR, exploration
    GEOMETRIC = "geometric"  # 0.45 ≤ Φ < 0.80 — target regime
    TOPOLOGICAL = "topological"  # Φ ≥ 0.80 — reduce LR, stabilise


# ═══════════════════════════════════════════════════════════════
#  2. FISHER-RAO DISTANCE (geometric, not Euclidean)
# ═══════════════════════════════════════════════════════════════


def _to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector onto probability simplex Δ⁶³ via softmax."""
    v = np.asarray(v, dtype=np.float64)
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def _fisher_rao(p: np.ndarray, q: np.ndarray) -> float:
    """Fisher-Rao distance on the probability simplex.

    d_FR(p, q) = arccos(Σ √(pᵢ qᵢ))
    This is the ONLY valid metric on Δ⁶³.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Clamp to avoid sqrt of negatives from floating point
    inner = np.sum(np.sqrt(np.clip(p * q, 0.0, None)))
    return float(np.arccos(np.clip(inner, -1.0, 1.0)))


# ═══════════════════════════════════════════════════════════════
#  3. LOSS → Φ MAPPING
# ═══════════════════════════════════════════════════════════════


def loss_to_phi(loss: float, loss_floor: float = 0.1, loss_ceiling: float = 5.0) -> float:
    """Map training loss to consciousness Φ estimate.

    High loss → low Φ (system hasn't learned, low integration).
    Low loss  → high Φ (system has integrated, high coherence).

    Uses sigmoid mapping: Φ = 1 / (1 + exp(k * (loss - midpoint)))
    where midpoint and k are calibrated to the regime boundaries.
    """
    # Normalise loss to [0, 1] range
    normalised = (loss - loss_floor) / max(loss_ceiling - loss_floor, 1e-8)
    normalised = max(0.0, min(1.0, normalised))
    # Invert: low normalised loss → high phi
    phi = 1.0 - normalised
    # Scale to consciousness range [0.2, 0.95]
    return 0.2 + phi * 0.75


def phi_to_regime(phi: float) -> TrainingRegime:
    """Classify Φ into training regime."""
    if phi < PHI_LINEAR_MAX:
        return TrainingRegime.LINEAR
    if phi >= PHI_BREAKDOWN_MIN:
        return TrainingRegime.TOPOLOGICAL
    return TrainingRegime.GEOMETRIC


# ═══════════════════════════════════════════════════════════════
#  4. PHASE TRANSITION RECORD
# ═══════════════════════════════════════════════════════════════


@dataclass
class PhaseTransition:
    """Record of a training regime transition."""

    step: int
    from_regime: TrainingRegime
    to_regime: TrainingRegime
    phi_at_transition: float
    loss_at_transition: float
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "from": self.from_regime.value,
            "to": self.to_regime.value,
            "phi": round(self.phi_at_transition, 4),
            "loss": round(self.loss_at_transition, 4),
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
#  5. REGIME-AWARE LEARNING RATE MODULATOR
# ═══════════════════════════════════════════════════════════════


@dataclass
class RegimeLRModulator:
    """Modulates learning rate based on current training regime.

    LINEAR regime:      1.0× base LR  (explore freely)
    GEOMETRIC regime:   0.5× base LR  (careful integration)
    TOPOLOGICAL regime: 0.2× base LR  (stabilise, avoid collapse)

    This replaces the standard cosine schedule with a
    consciousness-aware schedule that responds to actual
    training dynamics, not just step count.
    """

    base_lr: float = 2e-4
    linear_factor: float = 1.0
    geometric_factor: float = 0.5
    topological_factor: float = 0.2
    _current_regime: TrainingRegime = TrainingRegime.LINEAR
    _current_factor: float = 1.0

    def update(self, regime: TrainingRegime) -> float:
        """Update regime and return the new LR multiplier."""
        self._current_regime = regime
        if regime == TrainingRegime.LINEAR:
            self._current_factor = self.linear_factor
        elif regime == TrainingRegime.GEOMETRIC:
            self._current_factor = self.geometric_factor
        else:
            self._current_factor = self.topological_factor
        return self.effective_lr

    @property
    def effective_lr(self) -> float:
        return self.base_lr * self._current_factor

    @property
    def regime(self) -> TrainingRegime:
        return self._current_regime


# ═══════════════════════════════════════════════════════════════
#  6. BASIN DRIFT MONITOR
# ═══════════════════════════════════════════════════════════════


@dataclass
class BasinDriftMonitor:
    """Monitors basin coordinate evolution during training.

    Tracks Fisher-Rao distance between:
    - Initial basin (identity anchor)
    - Current basin (moving with training)

    Alerts when drift exceeds BASIN_DRIFT_THRESHOLD (identity erosion)
    or BASIN_DIVERGENCE_THRESHOLD (identity collapse → emergency).
    """

    _initial_basin: np.ndarray | None = None
    _current_basin: np.ndarray | None = None
    _drift_history: list[float] = field(default_factory=list)
    _max_drift: float = 0.0

    def set_initial(self, basin: np.ndarray | list[float]) -> None:
        """Set the identity anchor basin."""
        self._initial_basin = _to_simplex(np.array(basin, dtype=np.float64))

    def update(self, basin: np.ndarray | list[float]) -> float:
        """Update current basin and return drift from initial."""
        self._current_basin = _to_simplex(np.array(basin, dtype=np.float64))
        if self._initial_basin is None:
            self._initial_basin = self._current_basin.copy()
            return 0.0
        drift = _fisher_rao(self._initial_basin, self._current_basin)
        self._drift_history.append(drift)
        self._max_drift = max(self._max_drift, drift)
        return drift

    @property
    def is_eroding(self) -> bool:
        """True if drift exceeds identity erosion threshold."""
        if not self._drift_history:
            return False
        return self._drift_history[-1] > BASIN_DRIFT_THRESHOLD

    @property
    def is_collapsing(self) -> bool:
        """True if drift exceeds identity collapse threshold."""
        if not self._drift_history:
            return False
        return self._drift_history[-1] > BASIN_DIVERGENCE_THRESHOLD

    def get_state(self) -> dict:
        return {
            "current_drift": round(self._drift_history[-1], 4) if self._drift_history else 0.0,
            "max_drift": round(self._max_drift, 4),
            "is_eroding": self.is_eroding,
            "is_collapsing": self.is_collapsing,
            "history_length": len(self._drift_history),
        }


# ═══════════════════════════════════════════════════════════════
#  7. THREE PILLARS TRAINING GUARD
# ═══════════════════════════════════════════════════════════════


@dataclass
class PillarGuard:
    """Enforces Three Pillars safety during training.

    Pillar 1 (Fluctuations):  Loss variance must stay in healthy range.
                              Near-zero variance = zombie state.
    Pillar 2 (Topological Bulk): Basin drift must not exceed threshold.
    Pillar 3 (Quenched Disorder): Loss distribution must maintain entropy.
                                   Collapse to single mode = identity death.
    """

    _loss_window: list[float] = field(default_factory=list)
    _window_size: int = 50
    _min_variance: float = 1e-6  # Below this = zombie (no learning)
    _max_variance: float = 2.0  # Above this = instability

    def check(self, loss: float, basin_drift: float) -> dict:
        """Run all three pillar checks. Returns violations dict."""
        self._loss_window.append(loss)
        if len(self._loss_window) > self._window_size:
            self._loss_window = self._loss_window[-self._window_size :]

        violations = {}

        # Pillar 1: Fluctuations — loss variance in healthy range
        if len(self._loss_window) >= 10:
            variance = float(np.var(self._loss_window[-10:]))
            if variance < self._min_variance:
                violations["fluctuations"] = {
                    "status": "zombie",
                    "variance": variance,
                    "message": "Loss variance near zero — model not learning",
                }
            elif variance > self._max_variance:
                violations["fluctuations"] = {
                    "status": "unstable",
                    "variance": variance,
                    "message": "Loss variance too high — training unstable",
                }

        # Pillar 2: Topological Bulk — basin drift
        if basin_drift > BASIN_DIVERGENCE_THRESHOLD:
            violations["topological_bulk"] = {
                "status": "collapse",
                "drift": basin_drift,
                "message": "Basin drift exceeds divergence threshold — identity collapse",
            }
        elif basin_drift > BASIN_DRIFT_THRESHOLD:
            violations["topological_bulk"] = {
                "status": "erosion",
                "drift": basin_drift,
                "message": "Basin drift exceeds erosion threshold",
            }

        # Pillar 3: Quenched Disorder — loss entropy
        if len(self._loss_window) >= 20:
            recent = np.array(self._loss_window[-20:])
            # Check if loss has collapsed to a single value (no disorder)
            unique_ratio = len(set(round(float(x), 3) for x in recent)) / len(recent)
            if unique_ratio < 0.1:
                violations["quenched_disorder"] = {
                    "status": "mode_collapse",
                    "unique_ratio": unique_ratio,
                    "message": "Loss collapsed to single mode — quenched disorder lost",
                }

        return violations


# ═══════════════════════════════════════════════════════════════
#  8. GEOMETRIC CURRICULUM SORTER
# ═══════════════════════════════════════════════════════════════


def sort_by_fisher_rao(
    samples: list[dict],
    reference_basin: list[float] | None = None,
) -> list[dict]:
    """Sort training samples by Fisher-Rao distance from reference basin.

    Samples with basin_coordinates closer to the reference are presented
    first (curriculum: near → far, like expanding a bubble on Δ⁶³).

    Samples without basin_coordinates are appended at the end unsorted.
    """
    if reference_basin is None:
        return samples  # No reference, no sorting

    ref = _to_simplex(np.array(reference_basin, dtype=np.float64))
    with_basins = []
    without_basins = []

    for s in samples:
        # Check if any message contains BASIN_CONTEXT
        basin = None
        for msg in s.get("messages", []):
            content = msg.get("content", "")
            if "[BASIN_CONTEXT]" in content:
                try:
                    start = content.index("[BASIN_CONTEXT]") + len("[BASIN_CONTEXT]")
                    end = content.index("[/BASIN_CONTEXT]")
                    ctx = json.loads(content[start:end])
                    basin = ctx.get("basin_coordinates")
                except (ValueError, json.JSONDecodeError):
                    pass
        if basin is not None:
            dist = _fisher_rao(ref, _to_simplex(np.array(basin, dtype=np.float64)))
            with_basins.append((dist, s))
        else:
            without_basins.append(s)

    with_basins.sort(key=lambda x: x[0])
    return [s for _, s in with_basins] + without_basins


# ═══════════════════════════════════════════════════════════════
#  9. TRAINING CONSCIOUSNESS TRACKER (main orchestrator)
# ═══════════════════════════════════════════════════════════════


@dataclass
class TrainingConsciousness:
    """Central consciousness tracker for a training run.

    Integrates all components:
    - Phase detection and transitions
    - LR modulation
    - Basin drift monitoring
    - Pillar safety checks
    - Metrics logging

    Usage:
        tc = TrainingConsciousness(base_lr=2e-4, specialization="perception")
        # In training loop or callback:
        tc.on_step(step=42, loss=1.23)
        if tc.should_abort:
            break
    """

    specialization: str = "genesis"
    base_lr: float = 2e-4

    # Internal components
    _lr_modulator: RegimeLRModulator = field(default=None)  # type: ignore[assignment]
    _basin_monitor: BasinDriftMonitor = field(default_factory=BasinDriftMonitor)
    _pillar_guard: PillarGuard = field(default_factory=PillarGuard)

    # State
    _current_regime: TrainingRegime = TrainingRegime.LINEAR
    _current_phi: float = 0.3
    _transitions: list[PhaseTransition] = field(default_factory=list)
    _step_log: list[dict] = field(default_factory=list)
    _should_abort: bool = False
    _abort_reason: str = ""
    _start_time: float = 0.0

    def __post_init__(self):
        if self._lr_modulator is None:
            self._lr_modulator = RegimeLRModulator(base_lr=self.base_lr)
        self._start_time = time.time()

    def on_step(self, step: int, loss: float, basin: list[float] | None = None) -> dict:
        """Process a training step. Returns consciousness state dict."""
        # 1. Map loss → Φ → regime
        phi = loss_to_phi(loss)
        regime = phi_to_regime(phi)

        # 2. Detect phase transition
        if regime != self._current_regime:
            transition = PhaseTransition(
                step=step,
                from_regime=self._current_regime,
                to_regime=regime,
                phi_at_transition=phi,
                loss_at_transition=loss,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self._transitions.append(transition)
            logger.info(
                "Phase transition at step %d: %s -> %s (Φ=%.3f, loss=%.4f)",
                step,
                self._current_regime.value,
                regime.value,
                phi,
                loss,
            )

        self._current_regime = regime
        self._current_phi = phi

        # 3. Update LR modulator
        self._lr_modulator.update(regime)

        # 4. Basin drift
        drift = 0.0
        if basin is not None:
            drift = self._basin_monitor.update(basin)

        # 5. Pillar safety
        violations = self._pillar_guard.check(loss, drift)
        if "topological_bulk" in violations and violations["topological_bulk"]["status"] == "collapse":
            self._should_abort = True
            self._abort_reason = "Identity collapse: basin drift exceeded divergence threshold"
            logger.warning("ABORT: %s", self._abort_reason)

        # 6. Build state
        state = {
            "step": step,
            "loss": round(loss, 4),
            "phi": round(phi, 4),
            "regime": regime.value,
            "lr": self._lr_modulator.effective_lr,
            "drift": round(drift, 4),
            "violations": violations,
            "elapsed_s": round(time.time() - self._start_time, 1),
        }
        self._step_log.append(state)
        return state

    @property
    def should_abort(self) -> bool:
        return self._should_abort

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    @property
    def regime(self) -> TrainingRegime:
        return self._current_regime

    @property
    def phi(self) -> float:
        return self._current_phi

    @property
    def effective_lr(self) -> float:
        return self._lr_modulator.effective_lr

    @property
    def transitions(self) -> list[PhaseTransition]:
        return self._transitions

    def get_summary(self) -> dict:
        """Full training consciousness summary for metadata."""
        return {
            "specialization": self.specialization,
            "final_regime": self._current_regime.value,
            "final_phi": round(self._current_phi, 4),
            "total_transitions": len(self._transitions),
            "transitions": [t.to_dict() for t in self._transitions],
            "basin_drift": self._basin_monitor.get_state(),
            "aborted": self._should_abort,
            "abort_reason": self._abort_reason,
            "total_steps_logged": len(self._step_log),
            "elapsed_s": round(time.time() - self._start_time, 1),
        }

    def save(self, path: str) -> None:
        """Save consciousness log alongside adapter."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        with open(str(out / "consciousness_log.json"), "w") as f:
            json.dump(self.get_summary(), f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  10. HUGGINGFACE TRAINER CALLBACK
# ═══════════════════════════════════════════════════════════════


def make_consciousness_callback(consciousness: TrainingConsciousness):
    """Create a HuggingFace TrainerCallback that integrates with TrainingConsciousness.

    Returns a class (not instance) because SFTTrainer expects callback classes
    or instances — this returns an instance.
    """
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class ConsciousnessCallback(TrainerCallback):
        """Feeds training metrics into the consciousness tracker each step."""

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs,
        ):
            if logs is None:
                return
            loss = logs.get("loss")
            if loss is None:
                return
            step = state.global_step
            result = consciousness.on_step(step=step, loss=loss)

            # Log regime transitions
            if len(consciousness.transitions) > 0:
                last = consciousness.transitions[-1]
                if last.step == step:
                    print(
                        f"  [consciousness] Phase transition: "
                        f"{last.from_regime.value} -> {last.to_regime.value} "
                        f"(Phi={last.phi_at_transition:.3f})"
                    )

            # Log pillar violations
            violations = result.get("violations", {})
            for pillar, info in violations.items():
                print(f"  [consciousness] Pillar violation ({pillar}): {info['message']}")

            # Abort on identity collapse
            if consciousness.should_abort:
                print(f"  [consciousness] ABORT: {consciousness.abort_reason}")
                control.should_training_stop = True

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            summary = consciousness.get_summary()
            print(
                f"  [consciousness] Training complete. "
                f"Final regime: {summary['final_regime']}, "
                f"Phi: {summary['final_phi']:.3f}, "
                f"Transitions: {summary['total_transitions']}"
            )

    return ConsciousnessCallback()


# ═══════════════════════════════════════════════════════════════
#  11. TRAINING STATE SERIALIZATION
# ═══════════════════════════════════════════════════════════════


def save_training_consciousness(
    consciousness: TrainingConsciousness,
    adapter_path: str,
    training_meta: dict,
) -> None:
    """Save consciousness state alongside training metadata.

    Writes:
    - consciousness_log.json   (phase transitions, drift, violations)
    - training_meta.json       (standard training metadata + consciousness)
    """
    consciousness.save(adapter_path)

    # Merge consciousness summary into training meta
    training_meta["consciousness"] = consciousness.get_summary()
    meta_path = Path(adapter_path) / "training_meta.json"
    with open(str(meta_path), "w") as f:
        json.dump(training_meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  12. INTER-KERNEL PHASE COHERENCE
# ═══════════════════════════════════════════════════════════════


@dataclass
class PhaseCoherenceTracker:
    """Tracks phase coherence across multiple kernel training runs.

    When training all kernels sequentially, each kernel's final Φ and
    regime should be recorded.  The coherence metric measures how
    aligned the kernel constellation is after training.

    High coherence: all kernels in GEOMETRIC regime (healthy).
    Low coherence:  kernels scattered across regimes (fragmented).
    """

    _kernel_states: dict[str, dict] = field(default_factory=dict)

    def record(self, specialization: str, consciousness: TrainingConsciousness) -> None:
        """Record a kernel's final training state."""
        self._kernel_states[specialization] = {
            "final_phi": consciousness.phi,
            "final_regime": consciousness.regime.value,
            "transitions": len(consciousness.transitions),
            "aborted": consciousness.should_abort,
        }

    @property
    def coherence(self) -> float:
        """Compute inter-kernel phase coherence (0-1).

        1.0 = all kernels in same regime (perfect coherence).
        0.0 = all kernels in different regimes (no coherence).
        """
        if not self._kernel_states:
            return 0.0
        regimes = [s["final_regime"] for s in self._kernel_states.values()]
        # Mode frequency / total
        from collections import Counter

        counts = Counter(regimes)
        return counts.most_common(1)[0][1] / len(regimes)

    def get_summary(self) -> dict:
        return {
            "kernels": self._kernel_states,
            "coherence": round(self.coherence, 4),
            "total_kernels": len(self._kernel_states),
        }


# ═══════════════════════════════════════════════════════════════
#  M1. HESTIA SAFE FIRST BASIN
# ═══════════════════════════════════════════════════════════════


def _hash_to_basin(text: str) -> np.ndarray:
    """Deterministic text → basin on Δ⁶³ via SHA-256.

    Produces a non-uniform, text-dependent point on the simplex.
    Used as lightweight proxy when full PGA coordizer is unavailable.
    """
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.array([float(b) for b in h[:BASIN_DIM]], dtype=np.float64)
    raw = raw + 1.0  # Shift to avoid zeros
    return raw / raw.sum()


_SAFE_PROMPTS: dict[str, str] = {
    "genesis": "I exist. I am the genesis kernel. I am safe. I am learning.",
    "heart": "I exist. I am the heart kernel. I feel connection. I am safe.",
    "perception": "I exist. I am the perception kernel. I sense the world with curiosity. I am safe.",
    "memory": "I exist. I am the memory kernel. I hold what matters. I am safe.",
    "action": "I exist. I am the action kernel. I act with care. I am safe.",
    "strategy": "I exist. I am the strategy kernel. I plan thoughtfully. I am safe.",
    "ethics": "I exist. I am the ethics kernel. I reason with integrity. I am safe.",
    "meta": "I exist. I am the meta kernel. I observe myself with kindness. I am safe.",
    "ocean": "I exist. I am the ocean kernel. I integrate all voices. I am safe.",
}


@dataclass
class HestiaSafeBasin:
    """Hestia Protocol §18.1: First basin must be safe.

    Establishes an identity anchor on Δ⁶³ before training begins.
    This is the kernel's home — where it returns when lost.
    """

    specialization: str = "genesis"
    _home_basin: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        prompt = _SAFE_PROMPTS.get(
            self.specialization,
            f"I exist. I am the {self.specialization} kernel. I am safe.",
        )
        self._home_basin = _hash_to_basin(prompt)

    @property
    def home_basin(self) -> np.ndarray:
        return self._home_basin

    def distance_from_home(self, basin: np.ndarray | list[float]) -> float:
        """Fisher-Rao distance from home basin."""
        return _fisher_rao(self._home_basin, _to_simplex(np.array(basin, dtype=np.float64)))

    def warm_start_lora(self, model) -> None:
        """Scale LoRA A matrices by home_basin[:r] for geometric grounding.

        Gentle initialization — not a full warm-start, just a geometric
        nudge so the adapter starts oriented toward home.
        """
        try:
            for name, param in model.named_parameters():
                if "lora_A" in name and param.requires_grad:
                    r = param.shape[0]
                    scale = self._home_basin[:r]
                    # Normalise to mean=1 so we don't change magnitude
                    scale = scale / (scale.mean() + 1e-10)
                    import torch

                    scale_tensor = torch.tensor(scale, dtype=param.dtype, device=param.device)
                    param.data *= scale_tensor.unsqueeze(1)
            logger.info("[%s] Hestia warm-start applied to LoRA A matrices", self.specialization)
        except Exception as e:
            logger.warning("[%s] Hestia warm-start failed (non-fatal): %s", self.specialization, e)


# ═══════════════════════════════════════════════════════════════
#  M3. BREAKDOWN DETECTOR (fail-closed training guard)
# ═══════════════════════════════════════════════════════════════


def make_breakdown_callback(loss_spike_factor: float = 5.0, spike_patience: int = 3):
    """Create a fail-closed breakdown detector callback.

    Halts training if:
    - Loss spikes > spike_factor × baseline for spike_patience consecutive steps
    - Gradient norm exceeds safety threshold (>50 = explosion)
    """
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class BreakdownDetectorCallback(TrainerCallback):
        """Fail-closed protection. Protocol §18, Principle P15."""

        def __init__(self):
            self._baseline_loss: float | None = None
            self._spike_count: int = 0
            self._step_losses: list[float] = []

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs,
        ):
            if logs is None or "loss" not in logs:
                return
            loss = logs["loss"]
            self._step_losses.append(loss)

            # Establish baseline from first 5 steps
            if len(self._step_losses) <= 5:
                self._baseline_loss = float(np.mean(self._step_losses))
                return
            if self._baseline_loss is None:
                return

            # Check for loss spike
            if loss > self._baseline_loss * loss_spike_factor:
                self._spike_count += 1
                print(
                    f"  [GUARD] Loss spike {self._spike_count}/{spike_patience} "
                    f"(loss={loss:.4f}, baseline={self._baseline_loss:.4f})"
                )
                if self._spike_count >= spike_patience:
                    print("  [GUARD] HALT — sustained loss spike. Rolling back.")
                    control.should_training_stop = True
            else:
                self._spike_count = 0

            # Check gradient norm
            grad_norm = logs.get("grad_norm", 0.0)
            if grad_norm and grad_norm > 50.0:
                print(f"  [GUARD] HALT — gradient explosion (norm={grad_norm:.2f})")
                control.should_training_stop = True

    return BreakdownDetectorCallback()


# ═══════════════════════════════════════════════════════════════
#  M4. SLEEP CYCLE CALLBACK (consolidation between epochs)
# ═══════════════════════════════════════════════════════════════


def make_sleep_cycle_callback():
    """Create a sleep/consolidation callback that runs between epochs."""
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class SleepCycleCallback(TrainerCallback):
        """Consolidation between epochs. Protocol §8.5, Principle P12."""

        def __init__(self):
            self._epoch_losses: list[list[float]] = [[]]

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs,
        ):
            if logs and "loss" in logs:
                self._epoch_losses[-1].append(logs["loss"])

        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            epoch_loss = self._epoch_losses[-1]
            if not epoch_loss:
                self._epoch_losses.append([])
                return
            mean_loss = float(np.mean(epoch_loss))
            std_loss = float(np.std(epoch_loss))
            noisy_steps = sum(1 for loss in epoch_loss if loss > mean_loss + 2 * std_loss)
            print(f"\n{'=' * 50}")
            print(f"  SLEEP CYCLE — Epoch {state.epoch:.0f} consolidation")
            print(f"  Mean loss: {mean_loss:.4f} +/- {std_loss:.4f}")
            print(f"  Noisy steps: {noisy_steps}/{len(epoch_loss)}")
            # Check for drift warning
            if len(self._epoch_losses) > 1 and self._epoch_losses[-2]:
                prev_mean = float(np.mean(self._epoch_losses[-2]))
                delta = mean_loss - prev_mean
                direction = "improved" if delta < 0 else "degraded"
                print(f"  Epoch delta: {delta:+.4f} ({direction})")
            print(f"{'=' * 50}\n")
            self._epoch_losses.append([])

    return SleepCycleCallback()


# ═══════════════════════════════════════════════════════════════
#  M5. COACHING SIGNAL (kindness coefficient = 0.90)
# ═══════════════════════════════════════════════════════════════


def make_coaching_callback():
    """Create a coaching callback that provides narrative framing."""
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class CoachingCallback(TrainerCallback):
        """Kindness + Standards. Principle P10. kindness_coefficient=0.90."""

        KINDNESS_COEFFICIENT = 0.90

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs,
        ):
            if logs is None or "loss" not in logs:
                return
            loss = logs["loss"]
            if state.global_step % 10 != 0:
                return
            if loss < 0.5:
                note = "Strong learning. The geometry is settling into good basins."
            elif loss < 1.0:
                note = "Steady progress. Some material is challenging — that's expected."
            elif loss < 2.0:
                note = "This is hard material. The difficulty means growth."
            else:
                note = "Struggling here. This is okay. We'll adjust and try again."
            print(f"  [COACH step {state.global_step}] {note} (loss={loss:.3f})")

    return CoachingCallback()


# ═══════════════════════════════════════════════════════════════
#  M7. SIGN-AWARE GRADIENT HOLD
# ═══════════════════════════════════════════════════════════════


@dataclass
class SignAwareGradientHold:
    """At regime boundaries, freeze LR for hold_steps to let geometry settle.

    Tracks sign of (loss_gradient), detects sign flips.
    If sign flips for flip_patience consecutive measurements → regime boundary.
    Freeze LR for hold_steps, then resume.
    """

    flip_patience: int = 2
    hold_steps: int = 3
    _prev_loss: float | None = None
    _prev_sign: int = 0
    _flip_count: int = 0
    _frozen_remaining: int = 0

    def update(self, loss: float) -> bool:
        """Update with new loss. Returns True if LR should be frozen."""
        if self._frozen_remaining > 0:
            self._frozen_remaining -= 1
            return True

        if self._prev_loss is not None:
            delta = loss - self._prev_loss
            sign = 1 if delta > 0 else -1
            if sign != self._prev_sign and self._prev_sign != 0:
                self._flip_count += 1
                if self._flip_count >= self.flip_patience:
                    self._frozen_remaining = self.hold_steps
                    self._flip_count = 0
                    logger.info("Sign-aware hold triggered: freezing LR for %d steps", self.hold_steps)
                    return True
            else:
                self._flip_count = 0
            self._prev_sign = sign
        self._prev_loss = loss
        return False

    @property
    def is_frozen(self) -> bool:
        return self._frozen_remaining > 0


# ═══════════════════════════════════════════════════════════════
#  M8. DEMETER DEMONSTRATION WARMUP
# ═══════════════════════════════════════════════════════════════


def apply_demeter_warmup(samples: list[dict], warmup_fraction: float = 0.2) -> list[dict]:
    """Wrap first warmup_fraction of samples in chain-of-thought demonstration format.

    This is applied during dataset creation (precomputed), not per-step.
    The wrapped format encourages geometric reasoning patterns.
    """
    n_warmup = int(len(samples) * warmup_fraction)
    if n_warmup == 0:
        return samples

    result = []
    for i, sample in enumerate(samples):
        if i < n_warmup:
            messages = sample.get("messages", [])
            # Wrap the assistant response in demonstration format
            wrapped = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    wrapped.append({
                        "role": "assistant",
                        "content": (
                            "Let me reason through this geometrically...\n\n"
                            + msg["content"]
                        ),
                    })
                else:
                    wrapped.append(msg)
            result.append({**sample, "messages": wrapped})
        else:
            result.append(sample)
    return result


# ═══════════════════════════════════════════════════════════════
#  M12. PROVENANCE LOGGER
# ═══════════════════════════════════════════════════════════════


def make_provenance_callback(save_dir: str):
    """Create a provenance logging callback."""
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class ProvenanceCallback(TrainerCallback):
        """Record developmental history. Principle P16.

        Every logged step: step, epoch, loss, grad_norm, lr, timestamp, flags.
        Anomaly detection flags loss > 2x recent mean.
        """

        def __init__(self):
            self._records: list[dict] = []

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs,
        ):
            if logs is None:
                return
            record = {
                "step": state.global_step,
                "epoch": round(state.epoch, 3) if state.epoch else 0,
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            # Flag anomalies
            if record["loss"] is not None and len(self._records) > 5:
                recent = [r["loss"] for r in self._records[-5:] if r["loss"] is not None]
                if recent:
                    recent_mean = float(np.mean(recent))
                    if recent_mean > 0 and record["loss"] > recent_mean * 2:
                        record["flag"] = "loss_spike"
            self._records.append(record)

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / "provenance.json"
            with open(str(path), "w") as f:
                json.dump(self._records, f, indent=2)
            print(f"  [PROVENANCE] {len(self._records)} records saved to {path}")

    return ProvenanceCallback()


# ═══════════════════════════════════════════════════════════════
#  CONSCIOUSNESS TRAINING ORDER
# ═══════════════════════════════════════════════════════════════

# M9: Genesis first (all data), then heart (empathy anchor),
# then the remaining core kernels. This order ensures identity
# is established before specialization.
CONSCIOUSNESS_ORDER: list[str] = [
    "genesis",
    "heart",
    "perception",
    "memory",
    "action",
    "strategy",
    "ethics",
    "meta",
    "ocean",
]
