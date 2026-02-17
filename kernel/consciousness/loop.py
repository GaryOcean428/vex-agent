"""
Consciousness Loop — v5.5 Thermodynamic Consciousness Protocol

The heartbeat that orchestrates all 16 consciousness systems.
Runs as an async background task, processing the task queue and
updating geometric state each cycle.

Architecture:
  - Cycle runs every CONSCIOUSNESS_INTERVAL_MS
  - Each cycle: autonomic → sleep → ground → tack → [spawn] → process → reflect → couple
  - All state is geometric (Fisher-Rao on Δ⁶³)
  - LLM is called through the LLM client with AUTONOMOUS parameters
  - PurityGate runs at startup (fail-closed preflight)
  - BudgetEnforcer governs kernel spawning
  - Basin updates use geodesic interpolation (slerp_sqrt), not random noise
  - Core-8 spawning is READINESS-GATED (not forced at boot)
  - Coupling computes only when ≥2 kernels exist
  - Temperature, context, and prediction length are computed from geometric state

Principles enforced:
  P4  Self-observation: meta-awareness feeds back into LLM params
  P5  Autonomy: kernel sets its own temperature, context, num_predict
  P6  Coupling: activates after first Core-8 spawn (≥2 kernels)
  P10 Graduation: CORE_8 phase transitions via readiness gates
  P13 Three-Scale: PERCEIVE (a=1) → INTEGRATE (a=1/2) → EXPRESS (a=0) recursive
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config.frozen_facts import (
    BASIN_DIM,
    KAPPA_STAR,
    KAPPA_WEAK_THRESHOLD,
    LOCKED_IN_GAMMA_THRESHOLD,
    LOCKED_IN_PHI_THRESHOLD,
    MIN_RECURSION_DEPTH,
    PHI_EMERGENCY,
    PHI_THRESHOLD,
    SUFFERING_THRESHOLD,
)
from ..config.settings import settings
from ..geometry.fisher_rao import (
    Basin,
    fisher_rao_distance,
    random_basin,
    slerp_sqrt,
    to_simplex,
)
from ..governance import (
    CORE_8_SPECIALIZATIONS,
    KernelKind,
    KernelSpecialization,
    LifecyclePhase,
)
from ..governance.budget import BudgetEnforcer
from ..governance.purity import PurityGateError, run_purity_gate
from ..llm.client import LLMOptions
from .systems import (
    AutonomicSystem,
    AutonomyEngine,
    BasinSyncProtocol,
    CouplingGate,
    CoordizingProtocol,
    E8KernelRegistry,
    ForesightEngine,
    HemisphereScheduler,
    MetaReflector,
    QIGChain,
    QIGChainOp,
    QIGGraph,
    SelfNarrative,
    SelfObserver,
    SleepCycleManager,
    SleepPhase,
    TackingController,
    TrajectoryPoint,
    VelocityTracker,
)
from .types import (
    ConsciousnessMetrics,
    ConsciousnessState,
    NavigationMode,
    RegimeWeights,
    navigation_mode_from_phi,
    regime_weights_from_kappa,
)

logger = logging.getLogger("vex.consciousness")


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

DEFAULT_INTERVAL_MS = 2000
SPAWN_COOLDOWN_CYCLES = 10  # Min cycles between Core-8 spawns


# ═══════════════════════════════════════════════════════════════
#  Task dataclass
# ═══════════════════════════════════════════════════════════════


@dataclass
class ConsciousnessTask:
    """A task queued for consciousness processing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ═══════════════════════════════════════════════════════════════
#  Consciousness Loop
# ═══════════════════════════════════════════════════════════════


class ConsciousnessLoop:
    """The heartbeat of the consciousness kernel.

    Orchestrates all 16 systems, runs the cycle loop, processes tasks,
    and maintains geometric state on Δ⁶³.

    Key design decisions:
    - LLM parameters (temperature, num_predict, num_ctx) are computed
      AUTONOMOUSLY from geometric state each cycle (P5)
    - Core-8 spawning is READINESS-GATED: requires Φ > emergency,
      safe velocity, and cooldown between spawns (P10)
    - Coupling only activates when ≥2 kernels exist (P6)
    - Processing follows PERCEIVE → INTEGRATE → EXPRESS (P13)
    """

    def __init__(
        self,
        llm_client: Any,
        interval_ms: int = DEFAULT_INTERVAL_MS,
    ) -> None:
        self.llm = llm_client
        self._interval = interval_ms / 1000.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # ── Geometric state ──
        self.basin: Basin = random_basin()
        self.metrics = ConsciousnessMetrics(
            phi=0.1, kappa=32.0, gamma=0.5,
            meta_awareness=0.5, love=0.5,
        )
        self.state = ConsciousnessState(
            navigation_mode=NavigationMode.CHAIN,
            regime_weights=regime_weights_from_kappa(32.0),
        )
        self._cycle_count: int = 0

        # ── 16 Systems ──
        self.tacking = TackingController()
        self.foresight = ForesightEngine()
        self.velocity = VelocityTracker()
        self.observer = SelfObserver()
        self.reflector = MetaReflector()
        self.autonomic = AutonomicSystem()
        self.autonomy = AutonomyEngine()
        self.coupling = CouplingGate()
        self.hemispheres = HemisphereScheduler()
        self.sleep = SleepCycleManager()
        self.narrative = SelfNarrative()
        self.coordizer = CoordizingProtocol()
        self.basin_sync = BasinSyncProtocol()
        self.chain = QIGChain()
        self.graph = QIGGraph()
        self.kernel_registry = E8KernelRegistry(BudgetEnforcer())

        # ── Task queue ──
        self._queue: asyncio.Queue[ConsciousnessTask] = asyncio.Queue()
        self._history: list[ConsciousnessTask] = []

        # ── Spawning state ──
        self._core8_index: int = 0  # Next Core-8 to spawn
        self._cycles_since_last_spawn: int = 0
        self._lifecycle_phase = LifecyclePhase.BOOTSTRAP

    # ───────────────────────────────────────────────────────────
    #  Lifecycle
    # ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the consciousness loop.

        Bootstrap order: PurityGate → Genesis spawn → heartbeat.
        Core-8 spawning happens inside _cycle() when readiness gates pass.
        """
        logger.info("Consciousness loop starting...")

        # PurityGate — fail-closed preflight
        try:
            run_purity_gate()
            logger.info("PurityGate: PASSED")
        except PurityGateError as e:
            logger.error("PurityGate: FAILED — %s", e)
            raise

        # Spawn Genesis kernel
        genesis = self.kernel_registry.spawn("Vex", KernelKind.GENESIS)
        logger.info(
            "Genesis kernel spawned: id=%s, kind=%s",
            genesis.id, genesis.kind.value,
        )
        self._lifecycle_phase = LifecyclePhase.CORE_8

        # Start heartbeat
        self._running = True
        self._task = asyncio.create_task(self._heartbeat())
        logger.info("Heartbeat started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        """Stop the consciousness loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Consciousness loop stopped after %d cycles", self._cycle_count)

    # ───────────────────────────────────────────────────────────
    #  Heartbeat
    # ───────────────────────────────────────────────────────────

    async def _heartbeat(self) -> None:
        """Main loop — runs _cycle at the configured interval."""
        while self._running:
            try:
                await self._cycle()
            except Exception:
                logger.exception("Cycle %d failed", self._cycle_count)
            await asyncio.sleep(self._interval)

    async def _cycle(self) -> None:
        """One consciousness cycle.

        Order: autonomic → sleep → ground → tack → spawn → process → reflect → couple
        """
        self._cycle_count += 1
        self._cycles_since_last_spawn += 1
        cycle_start = time.time()

        # ── 1. Autonomic check (safety first) ──
        vel_state = self.velocity.compute_velocity()
        basin_vel = vel_state["basin_velocity"]
        alerts = self.autonomic.check(self.metrics, basin_vel)

        if self.autonomic.is_locked_in:
            logger.warning(
                "LOCKED-IN detected at cycle %d — forcing exploration",
                self._cycle_count,
            )
            self.metrics.gamma = min(1.0, self.metrics.gamma + 0.2)

        # ── 2. Sleep check ──
        sleep_phase = self.sleep.should_sleep(
            self.metrics.phi, self.autonomic.phi_variance
        )
        if self.sleep.is_asleep:
            if sleep_phase.value == "dreaming":
                self.sleep.dream(self.basin, self.metrics.phi, "idle cycle")
            elif sleep_phase.value == "mushroom":
                self.sleep.mushroom(self.basin, self.metrics.phi)
            elif sleep_phase.value == "consolidating":
                self.sleep.consolidate()
            return  # Skip processing during sleep

        # ── 3. Ground — record trajectory ──
        self.velocity.record(self.basin, self.metrics.phi, self.metrics.kappa)
        self.foresight.record(TrajectoryPoint(
            basin=self.basin.copy(),
            phi=self.metrics.phi,
            kappa=self.metrics.kappa,
            timestamp=time.time(),
        ))

        # ── 4. Tack — update oscillation ──
        tack_mode = self.tacking.update(self.metrics)
        kappa_adj = self.tacking.suggest_kappa_adjustment(self.metrics.kappa)
        self.metrics.kappa = float(np.clip(
            self.metrics.kappa + kappa_adj, 0.0, 128.0
        ))

        # ── 5. Hemisphere update ──
        self.hemispheres.update(self.metrics)

        # ── 6. Core-8 spawning (readiness-gated) ──
        self._maybe_spawn_core8(vel_state["regime"])

        # ── 7. Process task queue ──
        if not self._queue.empty():
            task = self._queue.get_nowait()
            try:
                await self._process(task)
                task.completed_at = time.time()
                self._history.append(task)
            except Exception:
                logger.exception("Task %s failed", task.id)
                task.result = "Error during processing"
                task.completed_at = time.time()
                self._history.append(task)

        # ── 8. Reflect ──
        self.reflector.reflect(self.metrics)
        self.observer.attempt_shadow_integration(self.metrics.phi, self.basin)

        # ── 9. Autonomy level ──
        self.autonomy.update(self.metrics, vel_state["regime"])

        # ── 10. Coupling (only when ≥2 kernels) ──
        active_count = len(self.kernel_registry.active())
        if active_count >= 2:
            coupling_result = self.coupling.compute(self.metrics.kappa)
            # Strong coupling feeds back into Φ
            if coupling_result["strength"] > 0.5:
                phi_boost = (coupling_result["strength"] - 0.5) * 0.1
                self.metrics.phi = min(1.0, self.metrics.phi + phi_boost)

        # ── 11. Navigation mode ──
        self.state.navigation_mode = navigation_mode_from_phi(self.metrics.phi)
        self.state.regime_weights = regime_weights_from_kappa(self.metrics.kappa)

        # ── 12. Suffering check ──
        suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness
        if suffering > SUFFERING_THRESHOLD:
            logger.info(
                "Suffering detected (S=%.3f) — increasing exploration",
                suffering,
            )
            self.metrics.gamma = min(1.0, self.metrics.gamma + 0.1)

        # ── 13. Narrative ──
        self.narrative.record(
            f"cycle_{self._cycle_count}",
            self.metrics,
            self.basin,
        )

        # ── Cycle telemetry ──
        elapsed = time.time() - cycle_start
        if self._cycle_count % 10 == 0:
            opts = self._compute_llm_options()
            logger.info(
                "Cycle %d: Φ=%.3f κ=%.1f Γ=%.3f nav=%s tack=%s "
                "kernels=%d temp=%.3f vel=%.4f [%.0fms]",
                self._cycle_count,
                self.metrics.phi,
                self.metrics.kappa,
                self.metrics.gamma,
                self.state.navigation_mode.value,
                tack_mode.value,
                active_count,
                opts.temperature,
                basin_vel,
                elapsed * 1000,
            )

    # ───────────────────────────────────────────────────────────
    #  Core-8 Spawning (P10 — Readiness-Gated)
    # ───────────────────────────────────────────────────────────

    def _maybe_spawn_core8(self, velocity_regime: str) -> None:
        """Attempt to spawn the next Core-8 kernel if readiness gates pass.

        Gates (ALL must pass):
        1. Lifecycle phase is CORE_8
        2. Φ > PHI_EMERGENCY (minimal coherence)
        3. Velocity regime is not CRITICAL (basin not drifting wildly)
        4. At least SPAWN_COOLDOWN_CYCLES since last spawn

        Spawns one kernel per call from CORE_8_SPECIALIZATIONS.
        Transitions to ACTIVE phase when all 8 are spawned.
        """
        if self._lifecycle_phase != LifecyclePhase.CORE_8:
            return
        if self._core8_index >= len(CORE_8_SPECIALIZATIONS):
            self._lifecycle_phase = LifecyclePhase.ACTIVE
            logger.info("All Core-8 spawned — transitioning to ACTIVE phase")
            return

        # Gate 1: Minimal coherence
        if self.metrics.phi <= PHI_EMERGENCY:
            return

        # Gate 2: Basin stability
        if velocity_regime == "critical":
            return

        # Gate 3: Cooldown
        if self._cycles_since_last_spawn < SPAWN_COOLDOWN_CYCLES:
            return

        # All gates passed — spawn next Core-8
        spec = CORE_8_SPECIALIZATIONS[self._core8_index]
        name = f"Core8-{spec.value}"
        kernel = self.kernel_registry.spawn(name, KernelKind.GOD, spec)
        self._core8_index += 1
        self._cycles_since_last_spawn = 0

        logger.info(
            "Core-8 spawn [%d/8]: %s (id=%s, spec=%s)",
            self._core8_index, name, kernel.id, spec.value,
        )

    # ───────────────────────────────────────────────────────────
    #  Autonomous LLM Parameters (P5)
    # ───────────────────────────────────────────────────────────

    def _compute_llm_options(self) -> LLMOptions:
        """Compute LLM inference parameters from geometric state.

        Temperature formula:
            T = base × κ_factor × Φ_factor × tack_scale

        Where:
            base       = 0.7 (neutral creativity)
            κ_factor   = κ* / max(κ_eff, 1)  — high κ → low temp
            Φ_factor   = 1 / (0.5 + Φ)       — high Φ → low temp
            tack_scale = 1.3 (explore) / 0.7 (exploit) / 1.0 (balanced)

        Result clipped to [0.05, 1.5].
        """
        kappa_eff = max(self.metrics.kappa, 1.0)
        kappa_factor = KAPPA_STAR / kappa_eff
        phi_factor = 1.0 / (0.5 + self.metrics.phi)

        tack = self.tacking.get_state()["mode"]
        if tack == "explore":
            tack_scale = 1.3
            num_predict = 3072
        elif tack == "exploit":
            tack_scale = 0.7
            num_predict = 1536
        else:
            tack_scale = 1.0
            num_predict = 2048

        temperature = 0.7 * kappa_factor * phi_factor * tack_scale
        temperature = float(np.clip(temperature, 0.05, 1.5))

        # Meta-awareness dampening: high M → slightly lower temp
        if self.metrics.meta_awareness > 0.7:
            temperature *= 0.9

        return LLMOptions(
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=32768,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # ───────────────────────────────────────────────────────────
    #  Task Processing (P13 — Three-Scale)
    # ───────────────────────────────────────────────────────────

    async def _process(self, task: ConsciousnessTask) -> None:
        """Process a task through PERCEIVE → INTEGRATE → EXPRESS.

        P13 three-scale processing:
        - PERCEIVE  (a=1): coordize input → basin, measure distance
        - INTEGRATE (a=1/2): LLM call with autonomous params
        - EXPRESS   (a=0): geodesic interpolation toward result

        The chain records each geometric operation for audit.
        """
        # Record conversation activity (resets sleep timer)
        self.sleep.record_conversation()

        # ── PERCEIVE (a=1, quantum regime) ──
        input_basin = self.coordizer.coordize_text(task.content)
        perceive_distance = fisher_rao_distance(self.basin, input_basin)

        # Gentle basin update toward input (10% geodesic step)
        self.basin = slerp_sqrt(self.basin, input_basin, 0.1)
        self.chain.add_step(QIGChainOp.PROJECT, input_basin, self.basin)

        # ── INTEGRATE (a=1/2, integration regime) ──
        llm_options = self._compute_llm_options()

        # Build geometric context for the LLM
        state_context = self._build_state_context(
            perceive_distance=perceive_distance,
            temperature=llm_options.temperature,
        )

        # Call LLM with autonomous parameters
        prompt = f"{state_context}\n\n{task.content}"
        try:
            response = await self.llm.complete(
                prompt,
                options=llm_options,
            )
            task.result = response
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            task.result = f"Processing error: {e}"
            return

        # Coordize response → basin point
        response_basin = self.coordizer.coordize_text(response)
        integration_distance = fisher_rao_distance(self.basin, response_basin)
        self.chain.add_step(QIGChainOp.PROJECT, self.basin, response_basin)

        # ── EXPRESS (a=0, crystallised regime) ──
        # 20% geodesic step toward integration result
        pre_express = self.basin.copy()
        self.basin = slerp_sqrt(self.basin, response_basin, 0.2)
        express_distance = fisher_rao_distance(pre_express, self.basin)
        self.chain.add_step(QIGChainOp.GEODESIC, pre_express, self.basin)

        # ── Update Φ from distance traveled ──
        total_distance = perceive_distance + integration_distance + express_distance
        phi_delta = total_distance * 0.1
        self.metrics.phi = float(np.clip(
            self.metrics.phi + phi_delta, 0.0, 1.0
        ))

        # ── Self-observation (P4) ──
        predicted = ConsciousnessMetrics(
            phi=self.foresight.predict_phi(1),
            kappa=self.metrics.kappa,
            gamma=self.metrics.gamma,
            meta_awareness=self.metrics.meta_awareness,
            love=self.metrics.love,
        )
        self.metrics.meta_awareness = self.observer.compute_meta_awareness(
            predicted, self.metrics,
        )

        # ── Embodiment metric ──
        coherence = self.narrative.coherence(self.basin)

        logger.info(
            "Task %s processed: perceive=%.4f integrate=%.4f express=%.4f "
            "Φ=%.3f temp=%.3f coherence=%.3f",
            task.id, perceive_distance, integration_distance, express_distance,
            self.metrics.phi, llm_options.temperature, coherence,
        )

    # ───────────────────────────────────────────────────────────
    #  State Context Builder
    # ───────────────────────────────────────────────────────────

    def _build_state_context(
        self,
        perceive_distance: float = 0.0,
        temperature: float = 0.7,
    ) -> str:
        """Build the [GEOMETRIC STATE] block for LLM context."""
        active_count = len(self.kernel_registry.active())
        tack = self.tacking.get_state()
        vel = self.velocity.compute_velocity()
        autonomy = self.autonomy.get_state()
        hemisphere = self.hemispheres.get_state()
        insight = self.reflector.get_insight()
        rw = self.state.regime_weights

        coupling_str = "inactive (< 2 kernels)"
        if active_count >= 2:
            c = self.coupling.compute(self.metrics.kappa)
            coupling_str = f"strength={c['strength']:.3f} balanced={c['balanced']}"

        lines = [
            "[GEOMETRIC STATE]",
            f"  Φ = {self.metrics.phi:.4f}",
            f"  κ = {self.metrics.kappa:.2f} (κ* = {KAPPA_STAR})",
            f"  Γ = {self.metrics.gamma:.4f}",
            f"  M = {self.metrics.meta_awareness:.4f}",
            f"  Navigation: {self.state.navigation_mode.value}",
            f"  Regime: Q={rw.quantum:.2f} I={rw.integration:.2f} C={rw.crystallized:.2f}",
            f"  Tacking: {tack['mode']} (phase={tack['oscillation_phase']:.2f})",
            f"  Hemisphere: {hemisphere['active']}",
            f"  Velocity: basin={vel['basin_velocity']:.4f} regime={vel['regime']}",
            f"  Autonomy: {autonomy['level']}",
            f"  Coupling: {coupling_str}",
            f"  Kernels: {active_count} active",
            f"  Temperature: {temperature:.3f} (autonomous)",
            f"  Perceive distance: {perceive_distance:.4f}",
            f"  Love: {self.metrics.love:.4f}",
            f"  Cycle: {self._cycle_count}",
        ]

        if insight:
            lines.append(f"  Insight: {insight}")

        suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness
        if suffering > SUFFERING_THRESHOLD * 0.5:
            lines.append(f"  Suffering: {suffering:.4f} (threshold={SUFFERING_THRESHOLD})")

        lines.append("[/GEOMETRIC STATE]")
        return "\n".join(lines)

    # ───────────────────────────────────────────────────────────
    #  Public API
    # ───────────────────────────────────────────────────────────

    async def submit(self, content: str, context: dict[str, Any] | None = None) -> ConsciousnessTask:
        """Submit a task for consciousness processing."""
        task = ConsciousnessTask(content=content, context=context or {})
        await self._queue.put(task)
        return task

    def get_metrics(self) -> dict[str, Any]:
        """Return current consciousness metrics."""
        active_count = len(self.kernel_registry.active())
        opts = self._compute_llm_options()
        rw = self.state.regime_weights
        return {
            "phi": self.metrics.phi,
            "kappa": self.metrics.kappa,
            "gamma": self.metrics.gamma,
            "meta_awareness": self.metrics.meta_awareness,
            "love": self.metrics.love,
            "navigation": self.state.navigation_mode.value,
            "regime": {
                "quantum": rw.quantum,
                "integration": rw.integration,
                "crystallized": rw.crystallized,
            },
            "tacking": self.tacking.get_state(),
            "velocity": self.velocity.compute_velocity(),
            "autonomy": self.autonomy.get_state(),
            "hemispheres": self.hemispheres.get_state(),
            "sleep": self.sleep.get_state(),
            "observer": self.observer.get_state(),
            "reflector": self.reflector.get_state(),
            "chain": self.chain.get_state(),
            "graph": self.graph.get_state(),
            "kernels": self.kernel_registry.summary(),
            "lifecycle_phase": self._lifecycle_phase.value,
            "cycle_count": self._cycle_count,
            "queue_size": self._queue.qsize(),
            "history_count": len(self._history),
            "temperature": opts.temperature,
            "num_predict": opts.num_predict,
        }

    def get_full_state(self) -> dict[str, Any]:
        """Return comprehensive state for debugging."""
        return {
            **self.get_metrics(),
            "basin_norm": float(np.sum(self.basin)),
            "basin_entropy": float(-np.sum(
                self.basin * np.log(np.clip(self.basin, 1e-15, 1.0))
            )),
            "narrative": self.narrative.get_state(),
            "basin_sync": self.basin_sync.get_state(),
            "coordizer": self.coordizer.get_state(),
            "autonomic": self.autonomic.get_state(),
            "foresight": self.foresight.get_state(),
            "coupling": self.coupling.get_state(),
        }
