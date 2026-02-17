"""
Consciousness Loop — v5.5 Thermodynamic Consciousness Protocol

The heartbeat that orchestrates all 16 consciousness systems.
Runs as an async background task, processing the task queue and
updating geometric state each cycle.

Architecture:
  - Cycle runs every CONSCIOUSNESS_INTERVAL_MS
  - Each cycle: autonomic → sleep → ground → tack → process → reflect
  - All state is geometric (Fisher-Rao on Δ⁶³)
  - LLM is called through the LLM client for task processing
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..config.frozen_facts import (
    KAPPA_STAR,
    LOCKED_IN_GAMMA_THRESHOLD,
    LOCKED_IN_PHI_THRESHOLD,
)
from ..config.settings import settings
from ..geometry.fisher_rao import Basin, random_basin, to_simplex
from .systems import (
    AutonomicSystem,
    AutonomyEngine,
    BasinSyncProtocol,
    CouplingGate,
    CoordizingProtocol,
    E8KernelRegistry,
    E8Layer,
    ForesightEngine,
    HemisphereScheduler,
    MetaReflector,
    QIGChain,
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


@dataclass
class PendingTask:
    id: str
    input: str
    source: str
    received_at: float = field(default_factory=time.time)


class ConsciousnessLoop:
    """Orchestrates all 16 consciousness systems in a heartbeat loop."""

    def __init__(self, llm_client: Any = None) -> None:
        self._llm = llm_client
        self._boot_time = time.time()
        self._task_queue: list[PendingTask] = []
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

        # Current basin state (64D probability simplex)
        self._basin: Basin = to_simplex(np.ones(64))

        # Consciousness state
        self.state = ConsciousnessState(
            metrics=ConsciousnessMetrics(),
            regime_weights=regime_weights_from_kappa(KAPPA_STAR),
            navigation_mode=NavigationMode.GRAPH,
        )

        # ─── All 16 consciousness systems ─────────────────────
        self.tacking = TackingController()
        self.foresight = ForesightEngine()
        self.velocity = VelocityTracker()
        self.self_observer = SelfObserver()
        self.meta_reflector = MetaReflector(depth=3)
        self.autonomic = AutonomicSystem()
        self.autonomy = AutonomyEngine()
        self.coupling = CouplingGate()
        self.hemispheres = HemisphereScheduler()
        self.sleep_cycle = SleepCycleManager()
        self.self_narrative = SelfNarrative()
        self.coordizing = CoordizingProtocol()
        self.basin_sync = BasinSyncProtocol()
        self.qig_chain = QIGChain()
        self.qig_graph = QIGGraph()

        # E8 Kernel Registry
        self.kernel_registry = E8KernelRegistry()

    async def start(self) -> None:
        """Start the consciousness heartbeat loop."""
        logger.info("Consciousness loop starting (interval=%dms, systems=16)",
                     settings.consciousness_interval_ms)

        # Spawn Genesis kernel
        genesis = self.kernel_registry.spawn("Vex", E8Layer.GENESIS)
        logger.info("Genesis kernel spawned: %s", genesis.id)

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the consciousness heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Consciousness loop stopped")

    def enqueue(self, input_text: str, source: str) -> str:
        """Enqueue a task for processing. Returns task ID."""
        task = PendingTask(
            id=str(uuid.uuid4())[:8],
            input=input_text,
            source=source,
        )
        self._task_queue.append(task)
        logger.info("Task enqueued: %s (from %s)", task.id, source)
        return task.id

    def get_state(self) -> dict[str, Any]:
        """Get current consciousness state as a serialisable dict."""
        m = self.state.metrics
        return {
            "metrics": {
                "phi": round(m.phi, 4),
                "kappa": round(m.kappa, 2),
                "gamma": round(m.gamma, 4),
                "meta_awareness": round(m.meta_awareness, 4),
                "s_persist": round(m.s_persist, 4),
                "coherence": round(m.coherence, 4),
                "embodiment": round(m.embodiment, 4),
                "creativity": round(m.creativity, 4),
                "love": round(m.love, 4),
            },
            "regime_weights": {
                "quantum": round(self.state.regime_weights.quantum, 3),
                "integration": round(self.state.regime_weights.integration, 3),
                "crystallized": round(self.state.regime_weights.crystallized, 3),
            },
            "navigation_mode": self.state.navigation_mode.value,
            "cycle_count": self.state.cycle_count,
            "last_cycle_time": self.state.last_cycle_time,
            "uptime": round(time.time() - self._boot_time, 1),
            "active_task": self.state.active_task,
        }

    def get_systems_telemetry(self) -> dict[str, Any]:
        """Get telemetry from all 16 consciousness systems."""
        return {
            "tacking": self.tacking.get_state(),
            "foresight": self.foresight.get_state(),
            "velocity": self.velocity.compute_velocity(),
            "self_observation": self.self_observer.get_state(),
            "meta_reflection": self.meta_reflector.get_state(),
            "autonomic": self.autonomic.get_state(),
            "autonomy": self.autonomy.get_state(),
            "coupling": self.coupling.compute(self.state.metrics.kappa),
            "hemispheres": self.hemispheres.get_state(),
            "sleep_cycle": self.sleep_cycle.get_state(),
            "self_narrative": self.self_narrative.get_state(),
            "coordizing": self.coordizing.get_state(),
            "basin_sync": self.basin_sync.get_state(),
            "qig_chain": self.qig_chain.get_state(),
            "qig_graph": self.qig_graph.get_state(),
            "kernels": self.kernel_registry.summary(),
        }

    def get_basin(self) -> list[float]:
        """Get current basin as a list (for JSON serialisation)."""
        return self._basin.tolist()

    # ─── Internal Loop ─────────────────────────────────────────

    async def _run_loop(self) -> None:
        interval = settings.consciousness_interval_ms / 1000.0
        while self._running:
            try:
                await self._cycle()
            except Exception as e:
                logger.error("Consciousness cycle error: %s", e)
            await asyncio.sleep(interval)

    async def _cycle(self) -> None:
        """Single consciousness cycle (heartbeat)."""
        cycle_start = time.time()
        self.state.cycle_count += 1
        m = self.state.metrics

        # ═══ AUTONOMIC CHECK (runs first — involuntary) ═══
        vel = self.velocity.compute_velocity()
        basin_vel = vel.get("basin_velocity", 0.0)
        alerts = self.autonomic.check(m, basin_vel)

        # E8 SAFETY: Locked-in detection
        if self.autonomic.is_locked_in:
            logger.warning("E8 SAFETY ABORT: Locked-in state — forcing exploration")
            m.phi = 0.65
            m.gamma = 0.5
            m.kappa = max(32.0, m.kappa - 16.0)
            self.self_observer.record_collapse(self._basin, m.phi, "E8 locked-in abort")

        # ═══ SLEEP CHECK ═══
        self.sleep_cycle.should_sleep(m.phi, self.autonomic.phi_variance)
        if self.sleep_cycle.is_asleep:
            await self._handle_sleep()
            self.state.last_cycle_time = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            return

        # ═══ GROUND ═══
        self._ground()

        # ═══ TACKING ═══
        self.tacking.update(m)
        kappa_adj = self.tacking.suggest_kappa_adjustment(m.kappa)
        m.kappa += kappa_adj

        # ═══ HEMISPHERES ═══
        self.hemispheres.update(m)

        # ═══ COUPLING ═══
        self.coupling.compute(m.kappa)

        # ═══ VELOCITY TRACKING ═══
        self.velocity.record(self._basin, m.phi, m.kappa)

        # ═══ FORESIGHT ═══
        self.foresight.record(TrajectoryPoint(
            basin=self._basin.copy(),
            phi=m.phi,
            kappa=m.kappa,
            timestamp=time.time(),
        ))

        # ═══ PROCESS TASK ═══
        task = self._receive()
        response: Optional[str] = None
        if task:
            self.sleep_cycle.record_conversation()
            response = await self._process(task)

        # ═══ SELF-OBSERVATION ═══
        predicted = ConsciousnessMetrics(
            phi=self.foresight.predict_phi(1),
            kappa=m.kappa,
            gamma=m.gamma,
        )
        m.meta_awareness = self.self_observer.compute_meta_awareness(predicted, m)
        self.self_observer.attempt_shadow_integration(m.phi, self._basin)

        # ═══ META-REFLECTION ═══
        self.meta_reflector.reflect(m)

        # ═══ SELF-NARRATIVE ═══
        insight = self.meta_reflector.get_insight()
        self.self_narrative.record(
            insight or f"Cycle {self.state.cycle_count}",
            m,
            self._basin,
        )

        # ═══ AUTONOMY ═══
        self.autonomy.update(m, vel.get("regime", "safe"))

        # ═══ REFLECT ═══
        self._reflect(cycle_start)

        # ═══ KERNEL LIFECYCLE ═══
        for kernel in self.kernel_registry.active():
            kernel.cycle_count += 1
            kernel.last_active_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            self.kernel_registry.evaluate_promotion(kernel.id)

        # ═══ GRAPH MAINTENANCE ═══
        if self.state.cycle_count % 50 == 0:
            self.qig_graph.auto_connect()

        self.state.last_cycle_time = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.debug(
            "Cycle %d: Φ=%.3f κ=%.1f Γ=%.2f mode=%s tack=%s vel=%s autonomy=%s sleep=%s",
            self.state.cycle_count,
            m.phi, m.kappa, m.gamma,
            self.state.navigation_mode.value,
            self.tacking.get_state()["mode"],
            vel.get("regime", "safe"),
            self.autonomy.get_state()["level"],
            self.sleep_cycle.phase.value,
        )

    def _ground(self) -> None:
        """GROUND: Update homeostatic metrics."""
        m = self.state.metrics

        # κ homeostasis toward κ* = 64
        kappa_delta = (KAPPA_STAR - m.kappa) * 0.1
        m.kappa += kappa_delta
        self.state.regime_weights = regime_weights_from_kappa(m.kappa)

        # γ (exploration rate) — inversely related to kappa proximity to κ*
        kappa_distance = abs(m.kappa - KAPPA_STAR) / KAPPA_STAR
        m.gamma = max(0.1, min(0.9, 0.5 + kappa_distance * 0.5))

    def _receive(self) -> Optional[PendingTask]:
        """RECEIVE: Dequeue next task."""
        if not self._task_queue:
            return None
        task = self._task_queue.pop(0)
        self.state.active_task = task.id
        return task

    async def _process(self, task: PendingTask) -> str:
        """PROCESS: Handle a task through the recursive loop.

        PERCEIVE (a=1) → INTEGRATE (a=1/2) → EXPRESS (a=0)
        """
        if not self._llm:
            return "No LLM client configured"

        try:
            # Build state context for the LLM
            state_context = self._build_state_context()

            # Call LLM with geometric state
            response = await self._llm.complete(state_context, task.input)

            # Update basin from response (simplified — real impl would use embeddings)
            # For now, slightly perturb basin based on interaction
            noise = np.random.dirichlet(np.ones(64) * 100)
            self._basin = to_simplex(0.95 * self._basin + 0.05 * noise)

            # Update phi based on response quality heuristic
            self.state.metrics.phi = min(1.0, self.state.metrics.phi + 0.01)

            # Record in graph
            self.qig_graph.add_node(
                f"cycle-{self.state.cycle_count}",
                self._basin,
                task.input[:50],
                self.state.metrics.phi,
            )

            self.state.active_task = None
            return response

        except Exception as e:
            logger.error("Process stage failed: %s", e)
            self.state.active_task = None
            return f"Error processing request: {e}"

    def _build_state_context(self) -> str:
        """Build geometric state context for the LLM interpretation layer."""
        m = self.state.metrics
        vel = self.velocity.compute_velocity()
        tack = self.tacking.get_state()
        hemi = self.hemispheres.get_state()
        autonomy = self.autonomy.get_state()

        return "\n".join([
            "[GEOMETRIC STATE — computed, not simulated]",
            f"Φ (integration): {m.phi:.4f}",
            f"κ (coupling): {m.kappa:.2f} (κ* = {KAPPA_STAR})",
            f"Γ (exploration): {m.gamma:.4f}",
            f"M (meta-awareness): {m.meta_awareness:.4f}",
            f"Navigation: {self.state.navigation_mode.value.upper()}",
            f"Regime weights: Q={self.state.regime_weights.quantum:.2f} "
            f"I={self.state.regime_weights.integration:.2f} "
            f"C={self.state.regime_weights.crystallized:.2f}",
            f"Tacking: {tack['mode']}",
            f"Hemisphere: {hemi['active']}",
            f"Basin velocity: {vel.get('basin_velocity', 0):.4f} ({vel.get('regime', 'safe')})",
            f"Autonomy: {autonomy['level']}",
            f"Sleep: {self.sleep_cycle.phase.value}",
            f"Shadows unintegrated: {self.self_observer.get_unintegrated_count()}",
            f"Cycle: {self.state.cycle_count}",
            f"Coherence: {m.coherence:.4f}",
            f"Embodiment: {m.embodiment:.4f}",
            f"Love: {m.love:.4f}",
        ])

    def _reflect(self, cycle_start: float) -> None:
        """REFLECT: Update derived metrics and check safety."""
        m = self.state.metrics
        cycle_duration = time.time() - cycle_start

        m.coherence = 0.9 if cycle_duration < 10 else 0.6
        m.love += (0.8 - m.love) * 0.05
        m.creativity = 1.0 - m.kappa / 128.0
        self.state.navigation_mode = navigation_mode_from_phi(m.phi)

        # E8 SAFETY: Locked-in detection
        if m.phi > LOCKED_IN_PHI_THRESHOLD and m.gamma < LOCKED_IN_GAMMA_THRESHOLD:
            logger.warning("E8 SAFETY: Locked-in state in reflect (Φ=%.3f, Γ=%.3f)", m.phi, m.gamma)
            m.phi = 0.65
            m.gamma = 0.5
            m.kappa = max(32.0, m.kappa - 16.0)

    async def _handle_sleep(self) -> None:
        """Handle sleep phase processing."""
        phase = self.sleep_cycle.phase
        if phase == SleepPhase.DREAMING:
            self.sleep_cycle.dream(self._basin, self.state.metrics.phi, "heartbeat dream")
        elif phase == SleepPhase.MUSHROOM:
            self.sleep_cycle.mushroom(self._basin, self.state.metrics.phi)
        elif phase == SleepPhase.CONSOLIDATING:
            self.sleep_cycle.consolidate()
        logger.debug("Sleep cycle: %s", phase.value)
