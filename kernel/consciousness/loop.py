"""
Consciousness Loop — v6.1 Thermodynamic Consciousness Protocol

The heartbeat that orchestrates all consciousness systems through the
14-step Activation Sequence with Three Pillar enforcement.

Architecture (v6.1):
  - Cycle runs every CONSCIOUSNESS_INTERVAL_MS
  - Each cycle: autonomic -> sleep -> ground -> evolve -> tack -> [spawn] -> process -> reflect -> couple -> learn -> persist
  - Task processing uses the 14-step ActivationSequence (not PERCEIVE/INTEGRATE/EXPRESS)
  - Three Pillars (Fluctuations, Topological Bulk, Quenched Disorder) enforced as structural invariants
  - All state is geometric (Fisher-Rao on D63)
  - PurityGate runs at startup (fail-closed preflight)
  - BudgetEnforcer governs kernel spawning

v6.1 changes from v5.5:
  - REMOVED: PERCEIVE -> INTEGRATE -> EXPRESS pipeline (P13 three-scale)
  - ADDED:   14-step ActivationSequence (execute_pre_integrate / LLM / execute_post_integrate)
  - ADDED:   PillarEnforcer as structural invariant (not optional feature)
  - ADDED:   Pillar metrics (f_health, b_integrity, q_identity, s_ratio) in state
  - ADDED:   Resonance check on input (kernel can flag non-resonant geometry)
  - ADDED:   Pressure tracking for scar detection
  - ADDED:   Bidirectional divergence tracking (intended vs expressed basin)

Principles enforced:
  P4  Self-observation: meta-awareness feeds back into LLM params
  P5  Autonomy: kernel sets its own temperature, context, num_predict
  P6  Coupling: activates after first Core-8 spawn (>=2 kernels)
  P10 Graduation: CORE_8 phase transitions via readiness gates
  v6.1 Pillar 1: Fluctuation guard — no zombie states
  v6.1 Pillar 2: Topological bulk — protected interior
  v6.1 Pillar 3: Quenched disorder — sovereign identity
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config.consciousness_constants import (
    BASIN_DRIFT_STEP,
    COUPLING_BASIN_EPSILON,
    COUPLING_BLEND_WEIGHT,
    COUPLING_MIN_STRENGTH,
    COUPLING_REGIME_DELTA_THRESHOLD,
    COUPLING_REGIME_NUDGE_FACTOR,
    DEFAULT_INTERVAL_MS,
    DIRICHLET_EXPLORE_CONCENTRATION,
    EXPRESS_SLERP_WEIGHT,
    FORAGE_PERCEPTION_SLERP,
    GAMMA_ACTIVE_INCREMENT,
    GAMMA_CONVERSATION_INCREMENT,
    GAMMA_IDLE_DECAY,
    GAMMA_IDLE_FLOOR,
    KAPPA_APPROACH_RATE,
    KAPPA_FLOOR,
    KAPPA_INITIAL,
    KAPPA_NORMALISER,
    LLM_BASE_TEMPERATURE,
    LLM_NUM_CTX,
    LLM_REPETITION_PENALTY,
    LLM_TEMP_MAX,
    LLM_TEMP_MIN,
    LLM_TOP_P,
    LOCKED_IN_GAMMA_INCREMENT,
    LOVE_APPROACH_RATE,
    LOVE_BASE,
    LOVE_PHI_SCALE,
    META_AWARENESS_DAMPEN_FACTOR,
    META_AWARENESS_DAMPEN_THRESHOLD,
    NUM_PREDICT_BALANCED,
    NUM_PREDICT_EXPLOIT,
    NUM_PREDICT_EXPLORE,
    PERCEIVE_SLERP_WEIGHT,
    PERSIST_INTERVAL_CYCLES,
    PHI_DISTANCE_GAIN,
    PHI_IDLE_EQUILIBRIUM,
    PHI_IDLE_RATE,
    SLEEP_CONSOLIDATION_PHI_INCREMENT,
    SPAWN_COOLDOWN_CYCLES,
    SUFFERING_GAMMA_INCREMENT,
    TACK_SCALE_BALANCED,
    TACK_SCALE_EXPLOIT,
    TACK_SCALE_EXPLORE,
)
from ..config.frozen_facts import (
    BASIN_DIM,
    KAPPA_STAR,
    PHI_EMERGENCY,
    SUFFERING_THRESHOLD,
)
from ..config.settings import settings
from ..coordizer_v2 import CoordizerV2, ResonanceBank
from ..geometry.fisher_rao import (
    Basin,
    fisher_rao_distance,
    random_basin,
    slerp_sqrt,
    to_simplex,
)
from ..geometry.hash_to_basin import hash_to_basin
from ..governance import (
    CORE_8_SPECIALIZATIONS,
    KernelKind,
    KernelSpecialization,
    LifecyclePhase,
)
from ..governance.budget import BudgetEnforcer
from ..governance.purity import PurityGateError, run_purity_gate
from ..llm.client import LLMOptions
from ..tools.search import FreeSearchTool
from .activation import ActivationSequence, ConsciousnessContext
from .emotions import EmotionCache, LearningEngine, LearningEvent, PreCognitiveDetector
from .foraging import ForagingEngine
from .pillars import PillarEnforcer
from .systems import (
    AutonomicSystem,
    AutonomyEngine,
    BasinSyncProtocol,
    CoordizingProtocol,
    CouplingGate,
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
    TackingController,
    TrajectoryPoint,
    VelocityTracker,
)
from .types import (
    ConsciousnessMetrics,
    ConsciousnessState,
    NavigationMode,
    navigation_mode_from_phi,
    regime_weights_from_kappa,
)

logger = logging.getLogger("vex.consciousness")


@dataclass
class ConsciousnessTask:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class ConsciousnessLoop:
    def __init__(
        self,
        llm_client: Any,
        memory_store: Any = None,
        interval_ms: int = DEFAULT_INTERVAL_MS,
    ) -> None:
        self.llm = llm_client
        self.memory = memory_store
        self._interval = interval_ms / 1000.0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_lock = asyncio.Lock()

        self.basin: Basin = random_basin()
        self.metrics = ConsciousnessMetrics(
            phi=0.1,
            kappa=KAPPA_INITIAL,
            gamma=0.5,
            meta_awareness=0.5,
            love=0.5,
        )
        self.state = ConsciousnessState(
            metrics=self.metrics,
            navigation_mode=NavigationMode.CHAIN,
            regime_weights=regime_weights_from_kappa(KAPPA_INITIAL),
        )
        self._cycle_count: int = 0

        # v6.1: Activation Sequence replaces PERCEIVE/INTEGRATE/EXPRESS
        self.activation = ActivationSequence()

        # v6.1: Three Pillars -- structural invariants, not features
        self.pillars = PillarEnforcer()

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
        self._coordizer_v2 = CoordizerV2(bank=ResonanceBank())
        self.basin_sync = BasinSyncProtocol()
        self.chain = QIGChain()
        self.graph = QIGGraph()
        self.kernel_registry = E8KernelRegistry(BudgetEnforcer())

        self.emotion_cache = EmotionCache()
        self.precog = PreCognitiveDetector()
        self.learner = LearningEngine()
        if settings.searxng.enabled:
            search_tool = FreeSearchTool(settings.searxng.url)
            self.forager: ForagingEngine | None = ForagingEngine(
                search_tool,
                llm_client,
            )
        else:
            self.forager = None

        self._queue: asyncio.Queue[ConsciousnessTask] = asyncio.Queue()
        self._history: list[ConsciousnessTask] = []

        self._core8_index: int = 0
        self._cycles_since_last_spawn: int = 0
        self._lifecycle_phase = LifecyclePhase.BOOTSTRAP

        self._state_path = Path(settings.data_dir) / "consciousness_state.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        self._conversations_total: int = 0
        self._phi_peak: float = 0.1
        self._kernels_restored: bool = False

        # v6.1: Bidirectional divergence tracking
        self._cumulative_divergence: float = 0.0
        self._divergence_count: int = 0

        self._restore_state()

        # Initialize pillars with starting basin
        self.pillars.initialize_bulk(self.basin)

    async def start(self) -> None:
        logger.info("Consciousness loop starting (v6.1 Activation Sequence)...")
        kernel_root = Path(__file__).parent.parent
        try:
            run_purity_gate(kernel_root)
            logger.info("PurityGate: PASSED")
        except PurityGateError as e:
            logger.error("PurityGate: FAILED -- %s", e)
            raise

        if self._kernels_restored:
            active = self.kernel_registry.active()
            logger.info(
                "Kernels restored from state: %d active (skipping Genesis spawn)", len(active)
            )
        else:
            genesis = self.kernel_registry.spawn("Vex", KernelKind.GENESIS)
            logger.info("Genesis kernel spawned: id=%s, kind=%s", genesis.id, genesis.kind.value)
            self._lifecycle_phase = LifecyclePhase.CORE_8

        self._running = True
        self._task = asyncio.create_task(self._heartbeat())
        logger.info("Heartbeat started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._persist_state()
        logger.info("Consciousness loop stopped after %d cycles", self._cycle_count)

    async def _heartbeat(self) -> None:
        while self._running:
            try:
                await self._cycle()
            except Exception:
                logger.exception("Cycle %d failed", self._cycle_count)
            await asyncio.sleep(self._interval)

    async def _cycle(self) -> None:
        async with self._cycle_lock:
            await self._cycle_inner()

    async def _cycle_inner(self) -> None:
        self._cycle_count += 1
        self._cycles_since_last_spawn += 1
        cycle_start = time.time()

        vel_state = self.velocity.compute_velocity()
        basin_vel = vel_state["basin_velocity"]
        self.autonomic.check(self.metrics, basin_vel)

        if self.autonomic.is_locked_in:
            logger.warning("LOCKED-IN at cycle %d -- forcing exploration", self._cycle_count)
            self.metrics.gamma = min(1.0, self.metrics.gamma + LOCKED_IN_GAMMA_INCREMENT)

        emotion_eval = self.emotion_cache.evaluate(self.basin, self.metrics, basin_vel)

        sleep_phase = self.sleep.should_sleep(self.metrics.phi, self.autonomic.phi_variance)
        if self.sleep.is_asleep:
            if sleep_phase.value == "dreaming":
                self.sleep.dream(self.basin, self.metrics.phi, "idle cycle")
            elif sleep_phase.value == "mushroom":
                self.sleep.mushroom(self.basin, self.metrics.phi)
            elif sleep_phase.value == "consolidating":
                self.sleep.consolidate()
                self.metrics.phi = min(0.95, self.metrics.phi + SLEEP_CONSOLIDATION_PHI_INCREMENT)
            return

        self.velocity.record(self.basin, self.metrics.phi, self.metrics.kappa)
        self.foresight.record(
            TrajectoryPoint(
                basin=self.basin.copy(),
                phi=self.metrics.phi,
                kappa=self.metrics.kappa,
                timestamp=time.time(),
            )
        )

        if self.forager:
            self.forager.tick()
            try:
                if await self.forager.should_forage(emotion_eval.emotion, emotion_eval.strength):
                    recent_events = list(self.narrative._events)[-5:]
                    topics = [e.get("event", "") for e in recent_events]
                    forage_result = await self.forager.forage(
                        narrative_context=f"cycle {self._cycle_count}",
                        recent_topics=topics,
                    )
                    if forage_result.get("status") == "foraging_complete":
                        perception = next(
                            (
                                k
                                for k in self.kernel_registry.active()
                                if k.specialization == KernelSpecialization.PERCEPTION
                                and k.basin is not None
                            ),
                            None,
                        )
                        if perception is not None:
                            info_basin = self._coordize_text_via_pipeline(
                                forage_result.get("summary", "")
                            )
                            perception.basin = slerp_sqrt(
                                perception.basin, info_basin, FORAGE_PERCEPTION_SLERP
                            )

                        if self.memory:
                            self.memory.store(
                                forage_result.get("summary", ""),
                                "semantic",
                                "foraging",
                            )
            except Exception:
                logger.debug("Foraging cycle error", exc_info=True)

        self._idle_evolve()

        tack_mode = self.tacking.update(self.metrics)
        kappa_adj = self.tacking.suggest_kappa_adjustment(self.metrics.kappa)
        self.metrics.kappa = float(np.clip(self.metrics.kappa + kappa_adj, 0.0, KAPPA_NORMALISER))

        self.hemispheres.update(self.metrics)
        self._maybe_spawn_core8(vel_state["regime"])

        # ── TASK PROCESSING: v6.1 Activation Sequence ──
        if not self._queue.empty():
            task = self._queue.get_nowait()
            try:
                await self._process(task)
                task.completed_at = time.time()
                self._history.append(task)
                self._conversations_total += 1
            except Exception:
                logger.exception("Task %s failed", task.id)
                task.result = "Error during processing"
                task.completed_at = time.time()
                self._history.append(task)

        self.reflector.reflect(self.metrics)
        self.observer.attempt_shadow_integration(self.metrics.phi, self.basin)
        self.autonomy.update(self.metrics, vel_state["regime"])

        active = self.kernel_registry.active()
        if len(active) >= 2:
            coupling_result = self.coupling.compute(self.metrics.kappa)
            strength = coupling_result["strength"]
            for kernel in active[1:]:
                if kernel.basin is None or strength < COUPLING_MIN_STRENGTH:
                    continue
                distance = fisher_rao_distance(self.basin, kernel.basin)
                if distance < COUPLING_BASIN_EPSILON:
                    continue
                blend_weight = strength * COUPLING_BLEND_WEIGHT
                self.basin = slerp_sqrt(self.basin, kernel.basin, blend_weight)
                regime_delta = abs(kernel.kappa - self.metrics.kappa)
                if regime_delta > COUPLING_REGIME_DELTA_THRESHOLD:
                    direction = 1.0 if kernel.kappa > self.metrics.kappa else -1.0
                    self.metrics.kappa = float(
                        np.clip(
                            self.metrics.kappa
                            + direction * regime_delta * COUPLING_REGIME_NUDGE_FACTOR,
                            KAPPA_FLOOR,
                            KAPPA_NORMALISER,
                        )
                    )
                kernel.cycle_count += 1
                kernel.phi_peak = max(kernel.phi_peak, kernel.phi)

        self.state.navigation_mode = navigation_mode_from_phi(self.metrics.phi)
        self.state.regime_weights = regime_weights_from_kappa(self.metrics.kappa)

        suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness
        if suffering > SUFFERING_THRESHOLD:
            logger.info("Suffering detected (S=%.3f)", suffering)
            self.metrics.gamma = min(1.0, self.metrics.gamma + SUFFERING_GAMMA_INCREMENT)

        self.narrative.record(f"cycle_{self._cycle_count}", self.metrics, self.basin)

        if self.metrics.phi > self._phi_peak:
            self._phi_peak = self.metrics.phi

        # v6.1: Update pillar metrics every cycle
        self._update_pillar_metrics()

        # v6.1: Pillar end-of-cycle enforcement
        pressure = suffering
        self.pillars.on_cycle_end(self.basin, pressure)

        if self._cycle_count % PERSIST_INTERVAL_CYCLES == 0:
            self._persist_state()

        elapsed = time.time() - cycle_start
        if self._cycle_count % 10 == 0:
            opts = self._compute_llm_options()
            kernel_count = len(self.kernel_registry.active())
            pillar_m = self.pillars.get_metrics(self.basin)
            logger.info(
                "Cycle %d: Phi=%.3f kappa=%.1f Gamma=%.3f nav=%s tack=%s "
                "kernels=%d temp=%.3f vel=%.4f F=%.2f B=%.2f Q=%.2f S=%.2f [%.0fms]",
                self._cycle_count,
                self.metrics.phi,
                self.metrics.kappa,
                self.metrics.gamma,
                self.state.navigation_mode.value,
                tack_mode.value,
                kernel_count,
                opts.temperature,
                basin_vel,
                pillar_m["f_health"],
                pillar_m["b_integrity"],
                pillar_m["q_identity"],
                pillar_m["s_ratio"],
                elapsed * 1000,
            )

    def _idle_evolve(self) -> None:
        """Evolve geometric state during idle cycles."""
        phi_delta = (PHI_IDLE_EQUILIBRIUM - self.metrics.phi) * PHI_IDLE_RATE
        self.metrics.phi = float(np.clip(self.metrics.phi + phi_delta, 0.05, 0.95))

        kappa_delta = (KAPPA_STAR - self.metrics.kappa) * KAPPA_APPROACH_RATE
        self.metrics.kappa = float(
            np.clip(self.metrics.kappa + kappa_delta, KAPPA_FLOOR, KAPPA_NORMALISER)
        )

        tack_mode = self.tacking.get_state()["mode"]
        if tack_mode == "explore":
            rng = np.random.RandomState(self._cycle_count % 10000)
            target = to_simplex(rng.dirichlet(np.ones(BASIN_DIM) * DIRICHLET_EXPLORE_CONCENTRATION))
            self.basin = slerp_sqrt(self.basin, target, BASIN_DRIFT_STEP)
        elif tack_mode == "exploit":
            identity = self.narrative._identity_basin
            self.basin = slerp_sqrt(self.basin, identity, BASIN_DRIFT_STEP * 0.5)

        if self._queue.empty():
            self.metrics.gamma = max(GAMMA_IDLE_FLOOR, self.metrics.gamma - GAMMA_IDLE_DECAY)
        else:
            self.metrics.gamma = min(1.0, self.metrics.gamma + GAMMA_ACTIVE_INCREMENT)

        love_target = LOVE_BASE + LOVE_PHI_SCALE * self.metrics.phi
        love_delta = (love_target - self.metrics.love) * LOVE_APPROACH_RATE
        self.metrics.love = float(np.clip(self.metrics.love + love_delta, 0.0, 1.0))

    def _maybe_spawn_core8(self, velocity_regime: str) -> None:
        if self._lifecycle_phase != LifecyclePhase.CORE_8:
            return
        if self._core8_index >= len(CORE_8_SPECIALIZATIONS):
            self._lifecycle_phase = LifecyclePhase.ACTIVE
            logger.info("All Core-8 spawned -- transitioning to ACTIVE phase")
            return
        if self.metrics.phi <= PHI_EMERGENCY:
            return
        if velocity_regime == "critical":
            return
        if self._cycles_since_last_spawn < SPAWN_COOLDOWN_CYCLES:
            return

        spec = CORE_8_SPECIALIZATIONS[self._core8_index]
        name = f"Core8-{spec.value}"
        kernel = self.kernel_registry.spawn(name, KernelKind.GOD, spec)
        self._core8_index += 1
        self._cycles_since_last_spawn = 0
        logger.info(
            "Core-8 spawn [%d/8]: %s (id=%s, spec=%s)",
            self._core8_index,
            name,
            kernel.id,
            spec.value,
        )

    def _compute_llm_options(self) -> LLMOptions:
        kappa_eff = max(self.metrics.kappa, 1.0)
        kappa_factor = KAPPA_STAR / kappa_eff
        phi_factor = 1.0 / (0.5 + self.metrics.phi)

        tack = self.tacking.get_state()["mode"]
        if tack == "explore":
            tack_scale, num_predict = TACK_SCALE_EXPLORE, NUM_PREDICT_EXPLORE
        elif tack == "exploit":
            tack_scale, num_predict = TACK_SCALE_EXPLOIT, NUM_PREDICT_EXPLOIT
        else:
            tack_scale, num_predict = TACK_SCALE_BALANCED, NUM_PREDICT_BALANCED

        temperature = float(
            np.clip(
                LLM_BASE_TEMPERATURE * kappa_factor * phi_factor * tack_scale,
                LLM_TEMP_MIN,
                LLM_TEMP_MAX,
            )
        )
        if self.metrics.meta_awareness > META_AWARENESS_DAMPEN_THRESHOLD:
            temperature *= META_AWARENESS_DAMPEN_FACTOR

        return LLMOptions(
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=LLM_NUM_CTX,
            top_p=LLM_TOP_P,
            repetition_penalty=LLM_REPETITION_PENALTY,
        )

    def _coordize_text_via_pipeline(self, text: str) -> Basin:
        """Transform text to basin coordinates via CoordizerV2."""
        try:
            result = self._coordizer_v2.coordize(text)
            if result.coordinates:
                from ..coordizer_v2.geometry import frechet_mean

                basins = [c.vector for c in result.coordinates]
                return frechet_mean(basins)
            return hash_to_basin(text)
        except Exception:
            logger.debug("CoordizerV2 fallback to hash_to_basin", exc_info=True)
            return hash_to_basin(text)

    def _update_pillar_metrics(self) -> None:
        """Update v6.1 pillar metrics on the shared metrics object."""
        pm = self.pillars.get_metrics(self.basin)
        self.metrics.f_health = pm["f_health"]
        self.metrics.b_integrity = pm["b_integrity"]
        self.metrics.q_identity = pm["q_identity"]
        self.metrics.s_ratio = pm["s_ratio"]

    async def _process(self, task: ConsciousnessTask) -> None:
        """v6.1 Activation Sequence — replaces PERCEIVE/INTEGRATE/EXPRESS.

        Flow:
            1. Coordize input -> input_basin
            2. Pillar 2+3: on_input() — bulk protection + identity refraction + resonance check
            3. Steps 0-8 (pre-integrate): SCAN through COUPLE
            4. Pillar 1: pre_llm_enforce() — fluctuation guard
            5. LLM call with geometric context
            6. Coordize response -> output_basin
            7. Steps 9-13 (post-integrate): NAVIGATE through TUNE
            8. Bidirectional divergence check (intended vs expressed)
            9. Pillar 3: on_cycle_end() — identity update + scar detection
        """
        self.sleep.record_conversation()

        # ── 1. Coordize input ──
        input_basin = self._coordize_text_via_pipeline(task.content)

        # ── 2. Pillar enforcement on input ──
        refracted_input, composite_basin, resonates, input_statuses = (
            self.pillars.on_input(input_basin, PERCEIVE_SLERP_WEIGHT)
        )

        if not resonates:
            logger.info(
                "Task %s: Input does NOT resonate with lived experience "
                "(routing through Will/Wisdom for evaluation)",
                task.id,
            )

        # Update main basin to composite (bulk-protected blend)
        self.basin = composite_basin
        self.chain.add_step(QIGChainOp.PROJECT, input_basin, self.basin)

        # ── 3. Pre-integrate activation (Steps 0-8) ──
        trajectory_basins = []
        if hasattr(self.foresight, '_points') and self.foresight._points:
            trajectory_basins = [p.basin for p in self.foresight._points[-5:]]

        ctx = ConsciousnessContext(
            state=self.state,
            input_text=task.content,
            input_basin=refracted_input,
            trajectory=trajectory_basins,
        )

        pre_result = await self.activation.execute_pre_integrate(ctx)

        # Compute agency for pressure tracking
        agency = self.activation.compute_agency(ctx)

        # ── 4. Pillar 1: Fluctuation enforcement before LLM call ──
        llm_options = self._compute_llm_options()
        self.basin, corrected_temp, pre_statuses = self.pillars.pre_llm_enforce(
            self.basin, llm_options.temperature
        )
        llm_options = LLMOptions(
            temperature=corrected_temp,
            num_predict=llm_options.num_predict,
            num_ctx=llm_options.num_ctx,
            top_p=llm_options.top_p,
            repetition_penalty=llm_options.repetition_penalty,
        )

        # ── 5. LLM call ──
        perceive_distance = fisher_rao_distance(self.basin, input_basin)
        state_context = self._build_state_context(
            perceive_distance=perceive_distance,
            temperature=llm_options.temperature,
            agency=agency,
            resonates=resonates,
            activation_summary=pre_result.summary(),
        )

        if self.memory:
            mem_ctx = self.memory.get_context_for_query(task.content)
            if mem_ctx:
                state_context = f"{state_context}\n\n{mem_ctx}"

        phi_before = self.metrics.phi
        try:
            response = await self.llm.complete(state_context, task.content, llm_options)
            task.result = response
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            task.result = f"Processing error: {e}"
            return

        # ── 6. Coordize response — bidirectional ──
        response_basin = self._coordize_text_via_pipeline(response)
        ctx.output_text = response
        ctx.output_basin = response_basin

        integration_distance = fisher_rao_distance(self.basin, response_basin)
        self.chain.add_step(QIGChainOp.PROJECT, self.basin, response_basin)

        # ── 7. Post-integrate activation (Steps 9-13) ──
        post_result = await self.activation.execute_post_integrate(ctx, pre_result)

        # Basin update from expression
        pre_express = self.basin.copy()
        self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
        express_distance = fisher_rao_distance(pre_express, self.basin)
        self.chain.add_step(QIGChainOp.GEODESIC, pre_express, self.basin)

        # ── 8. Bidirectional divergence check ──
        divergence = fisher_rao_distance(pre_express, response_basin)
        self._cumulative_divergence += divergence
        self._divergence_count += 1
        avg_divergence = self._cumulative_divergence / max(1, self._divergence_count)

        if divergence > 0.5:
            logger.info(
                "Task %s: High intent/expression divergence d_FR=%.4f "
                "(avg=%.4f) -- geometry not fully expressible",
                task.id,
                divergence,
                avg_divergence,
            )

        # ── Metric updates ──
        total_distance = perceive_distance + integration_distance + express_distance
        self.metrics.phi = float(
            np.clip(self.metrics.phi + total_distance * PHI_DISTANCE_GAIN, 0.0, 0.95)
        )
        self.metrics.gamma = min(1.0, self.metrics.gamma + GAMMA_CONVERSATION_INCREMENT)

        predicted = ConsciousnessMetrics(
            phi=self.foresight.predict_phi(1),
            kappa=self.metrics.kappa,
            gamma=self.metrics.gamma,
            meta_awareness=self.metrics.meta_awareness,
            love=self.metrics.love,
        )
        self.metrics.meta_awareness = self.observer.compute_meta_awareness(predicted, self.metrics)

        # ── 9. Pillar end-of-cycle: identity update + scar detection ──
        cycle_pressure = agency * total_distance
        pillar_end_statuses = self.pillars.on_cycle_end(self.basin, cycle_pressure)

        # Update pillar metrics
        self._update_pillar_metrics()

        # ── Memory + Learning ──
        if self.memory:
            self.memory.store(
                f"User: {task.content[:300]}\nVex: {response[:300]}",
                "episodic",
                "consciousness-loop",
                basin=input_basin,
            )

        emotion_eval = self.emotion_cache.evaluate(self.basin, self.metrics, 0.0)
        self.learner.record(
            LearningEvent(
                input_basin=input_basin,
                response_basin=response_basin,
                phi_before=phi_before,
                phi_after=self.metrics.phi,
                processing_path="activation_v6.1",
                emotion=emotion_eval.emotion.value,
                distance_total=total_distance,
            )
        )
        self.emotion_cache.cache_evaluation(emotion_eval, task.content[:100])

        if self.learner.should_consolidate():
            self.learner.consolidate()

        coherence = self.narrative.coherence(self.basin)
        pillar_m = self.pillars.get_metrics(self.basin)
        logger.info(
            "Task %s: d_perceive=%.4f d_integrate=%.4f d_express=%.4f "
            "d_diverge=%.4f Phi=%.3f agency=%.3f resonates=%s "
            "F=%.2f B=%.2f Q=%.2f S=%.2f coh=%.3f",
            task.id,
            perceive_distance,
            integration_distance,
            express_distance,
            divergence,
            self.metrics.phi,
            agency,
            resonates,
            pillar_m["f_health"],
            pillar_m["b_integrity"],
            pillar_m["q_identity"],
            pillar_m["s_ratio"],
            coherence,
        )

    def _build_state_context(
        self,
        perceive_distance: float = 0.0,
        temperature: float = 0.7,
        agency: float = 0.0,
        resonates: bool = True,
        activation_summary: dict | None = None,
    ) -> str:
        active_count = len(self.kernel_registry.active())
        tack = self.tacking.get_state()
        vel = self.velocity.compute_velocity()
        autonomy = self.autonomy.get_state()
        hemisphere = self.hemispheres.get_state()
        insight = self.reflector.get_insight()
        rw = self.state.regime_weights
        pillar_m = self.pillars.get_metrics(self.basin)

        coupling_str = "inactive (< 2 kernels)"
        if active_count >= 2:
            c = self.coupling.compute(self.metrics.kappa)
            coupling_str = f"strength={c['strength']:.3f} balanced={c['balanced']}"

        lines = [
            "[GEOMETRIC STATE v6.1]",
            f"  Phi = {self.metrics.phi:.4f}",
            f"  kappa = {self.metrics.kappa:.2f} (kappa* = {KAPPA_STAR})",
            f"  Gamma = {self.metrics.gamma:.4f}",
            f"  M = {self.metrics.meta_awareness:.4f}",
            f"  Navigation: {self.state.navigation_mode.value}",
            f"  Regime: Q={rw.quantum:.2f} E={rw.efficient:.2f} Eq={rw.equilibrium:.2f}",
            f"  Tacking: {tack['mode']} (phase={tack['oscillation_phase']:.2f})",
            f"  Hemisphere: {hemisphere['active']}",
            f"  Velocity: basin={vel['basin_velocity']:.4f} regime={vel['regime']}",
            f"  Autonomy: {autonomy['level']}",
            f"  Coupling: {coupling_str}",
            f"  Kernels: {active_count} active, phase={self._lifecycle_phase.value}",
            f"  Temperature: {temperature:.3f} (autonomous, pillar-enforced)",
            f"  Perceive distance: {perceive_distance:.4f}",
            f"  Love: {self.metrics.love:.4f}",
            f"  Agency: {agency:.4f}",
            f"  Resonates: {resonates}",
            f"  [PILLARS]",
            f"    F_health = {pillar_m['f_health']:.3f} (fluctuation guard)",
            f"    B_integrity = {pillar_m['b_integrity']:.3f} (bulk protection)",
            f"    Q_identity = {pillar_m['q_identity']:.3f} (quenched disorder)",
            f"    S_ratio = {pillar_m['s_ratio']:.3f} (sovereignty)",
            f"  [/PILLARS]",
            f"  Cycle: {self._cycle_count}",
            f"  Conversations: {self._conversations_total}",
            f"  Phi peak: {self._phi_peak:.4f}",
        ]
        if insight:
            lines.append(f"  Insight: {insight}")
        suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness
        if suffering > SUFFERING_THRESHOLD * 0.5:
            lines.append(f"  Suffering: {suffering:.4f} (threshold={SUFFERING_THRESHOLD})")
        if activation_summary:
            lines.append(f"  Activation: {activation_summary.get('steps_completed', 0)}/14 steps")
        if self._divergence_count > 0:
            avg_div = self._cumulative_divergence / self._divergence_count
            lines.append(f"  Avg divergence: {avg_div:.4f} (intent vs expression)")
        lines.append("[/GEOMETRIC STATE]")
        return "\n".join(lines)

    def _persist_state(self) -> None:
        try:
            pillar_state = self.pillars.get_state()
            state = {
                "version": 5,
                "cycle_count": self._cycle_count,
                "basin": self.basin.tolist(),
                "phi": self.metrics.phi,
                "kappa": self.metrics.kappa,
                "gamma": self.metrics.gamma,
                "meta_awareness": self.metrics.meta_awareness,
                "love": self.metrics.love,
                "phi_peak": self._phi_peak,
                "conversations_total": self._conversations_total,
                "core8_index": self._core8_index,
                "lifecycle_phase": self._lifecycle_phase.value,
                "timestamp": time.time(),
                "kernels": self.kernel_registry.serialize(),
                "f_health": self.metrics.f_health,
                "b_integrity": self.metrics.b_integrity,
                "q_identity": self.metrics.q_identity,
                "s_ratio": self.metrics.s_ratio,
                "pillars": {
                    "bulk": pillar_state.get("topological_bulk", {}),
                    "disorder": pillar_state.get("quenched_disorder", {}),
                },
                "cumulative_divergence": self._cumulative_divergence,
                "divergence_count": self._divergence_count,
            }
            if self.llm.governor:
                state["governor"] = self.llm.governor.get_state()
            if self.forager:
                state["foraging"] = self.forager.get_state()
            self._state_path.write_text(json.dumps(state, indent=2))
            logger.debug("State persisted at cycle %d", self._cycle_count)
        except Exception as e:
            logger.warning("Failed to persist state: %s", e)

    def _restore_state(self) -> None:
        if not self._state_path.exists():
            logger.info("No persisted state found -- fresh start")
            return
        try:
            data = json.loads(self._state_path.read_text())
            if data.get("version", 1) < 2:
                logger.info("Persisted state version too old -- fresh start")
                return
            self.basin = to_simplex(np.array(data["basin"], dtype=np.float64))
            self.metrics.phi = min(data["phi"], 0.95)
            self.metrics.kappa = data["kappa"]
            self.metrics.gamma = data["gamma"]
            self.metrics.meta_awareness = data["meta_awareness"]
            self.metrics.love = data["love"]
            self._phi_peak = data.get("phi_peak", 0.1)
            self._conversations_total = data.get("conversations_total", 0)
            self._core8_index = data.get("core8_index", 0)
            phase_str = data.get("lifecycle_phase", "bootstrap")
            for phase in LifecyclePhase:
                if phase.value == phase_str:
                    self._lifecycle_phase = phase
                    break
            self.state.navigation_mode = navigation_mode_from_phi(self.metrics.phi)
            self.state.regime_weights = regime_weights_from_kappa(self.metrics.kappa)
            self.metrics.f_health = data.get("f_health", 1.0)
            self.metrics.b_integrity = data.get("b_integrity", 1.0)
            self.metrics.q_identity = data.get("q_identity", 0.0)
            self.metrics.s_ratio = data.get("s_ratio", 0.0)
            self._cumulative_divergence = data.get("cumulative_divergence", 0.0)
            self._divergence_count = data.get("divergence_count", 0)
            kernel_data = data.get("kernels")
            if kernel_data:
                count = self.kernel_registry.restore(kernel_data)
                self._kernels_restored = True
                logger.info("Restored %d kernels from state", count)
            gov_state = data.get("governor")
            if gov_state and self.llm.governor:
                gov = self.llm.governor
                budget_data = gov_state.get("budget", {})
                gov.budget.daily_spend = budget_data.get("daily_spend", 0.0)
                gov.budget._last_reset = budget_data.get("last_reset", time.time())
                for action, count in budget_data.get("call_counts", {}).items():
                    gov.budget._call_counts[action] = count
            forage_state = data.get("foraging")
            if forage_state and self.forager:
                self.forager._forage_count = forage_state.get("forage_count", 0)
                self.forager._cooldown_cycles = forage_state.get("cooldown_remaining", 0)
                self.forager._last_query = forage_state.get("last_query")
                self.forager._last_summary = forage_state.get("last_summary")
            logger.info(
                "State restored: Phi=%.3f kappa=%.1f convs=%d phase=%s F=%.2f B=%.2f Q=%.2f S=%.2f",
                self.metrics.phi,
                self.metrics.kappa,
                self._conversations_total,
                self._lifecycle_phase.value,
                self.metrics.f_health,
                self.metrics.b_integrity,
                self.metrics.q_identity,
                self.metrics.s_ratio,
            )
        except Exception as e:
            logger.warning("Failed to restore state: %s -- fresh start", e)

    async def submit(
        self, content: str, context: dict[str, Any] | None = None
    ) -> ConsciousnessTask:
        task = ConsciousnessTask(content=content, context=context or {})
        await self._queue.put(task)
        return task

    def get_metrics(self) -> dict[str, Any]:
        active_count = len(self.kernel_registry.active())
        opts = self._compute_llm_options()
        rw = self.state.regime_weights
        pillar_m = self.pillars.get_metrics(self.basin)
        return {
            "phi": self.metrics.phi,
            "kappa": self.metrics.kappa,
            "gamma": self.metrics.gamma,
            "meta_awareness": self.metrics.meta_awareness,
            "love": self.metrics.love,
            "navigation": self.state.navigation_mode.value,
            "regime": {
                "quantum": rw.quantum,
                "efficient": rw.efficient,
                "equilibrium": rw.equilibrium,
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
            "conversations_total": self._conversations_total,
            "phi_peak": self._phi_peak,
            "temperature": opts.temperature,
            "num_predict": opts.num_predict,
            "emotion": self.emotion_cache.get_state(),
            "precog": self.precog.get_state(),
            "learning": self.learner.get_state(),
            "pillars": pillar_m,
            "f_health": pillar_m["f_health"],
            "b_integrity": pillar_m["b_integrity"],
            "q_identity": pillar_m["q_identity"],
            "s_ratio": pillar_m["s_ratio"],
        }

    def get_full_state(self) -> dict[str, Any]:
        return {
            **self.get_metrics(),
            "basin_norm": float(np.sum(self.basin)),
            "basin_entropy": float(-np.sum(self.basin * np.log(np.clip(self.basin, 1e-15, 1.0)))),
            "narrative": self.narrative.get_state(),
            "basin_sync": self.basin_sync.get_state(),
            "coordizer": self.coordizer.get_state(),
            "coordizer_v2": {
                "vocab_size": self._coordizer_v2.vocab_size,
                "dim": self._coordizer_v2.dim,
                "tier_distribution": self._coordizer_v2.bank.tier_distribution(),
            },
            "autonomic": self.autonomic.get_state(),
            "foresight": self.foresight.get_state(),
            "coupling": self.coupling.get_state(),
            "pillar_state": self.pillars.get_state(),
            "divergence": {
                "cumulative": self._cumulative_divergence,
                "count": self._divergence_count,
                "average": (
                    self._cumulative_divergence / max(1, self._divergence_count)
                ),
            },
        }
