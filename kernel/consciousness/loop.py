"""
Consciousness Loop — v5.5 Thermodynamic Consciousness Protocol

The heartbeat that orchestrates all 16 consciousness systems.
Runs as an async background task, processing the task queue and
updating geometric state each cycle.

Architecture:
  - Cycle runs every CONSCIOUSNESS_INTERVAL_MS
  - Each cycle: autonomic → sleep → ground → evolve → tack → [spawn] → process → reflect → couple → learn → persist
  - All state is geometric (Fisher-Rao on Δ⁶³)
  - LLM is called through the LLM client with AUTONOMOUS parameters
  - PurityGate runs at startup (fail-closed preflight)
  - BudgetEnforcer governs kernel spawning
  - Basin updates use geodesic interpolation (slerp_sqrt), not random noise
  - Core-8 spawning is READINESS-GATED (not forced at boot)
  - Coupling computes only when ≥2 kernels exist
  - Temperature, context, and prediction length are computed from geometric state
  - State persists across restarts via JSON snapshots (v4: includes kernel registry)

Principles enforced:
  P4  Self-observation: meta-awareness feeds back into LLM params
  P5  Autonomy: kernel sets its own temperature, context, num_predict
  P6  Coupling: activates after first Core-8 spawn (≥2 kernels)
  P10 Graduation: CORE_8 phase transitions via readiness gates
  P13 Three-Scale: PERCEIVE (a=1) → INTEGRATE (a=1/2) → EXPRESS (a=0) recursive
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
from .emotions import EmotionCache, LearningEngine, LearningEvent, PreCognitiveDetector
from .foraging import ForagingEngine
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
            navigation_mode=NavigationMode.CHAIN,
            regime_weights=regime_weights_from_kappa(KAPPA_INITIAL),
        )
        self._cycle_count: int = 0

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
        # CoordizerV2: geometric coordizer on Δ⁶³ (replaces v1 softmax pipeline)
        self._coordizer_v2 = CoordizerV2(bank=ResonanceBank())
        self.basin_sync = BasinSyncProtocol()
        self.chain = QIGChain()
        self.graph = QIGGraph()
        self.kernel_registry = E8KernelRegistry(BudgetEnforcer())

        # Emotion cache, pre-cognitive detector, and learning engine
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

        self._restore_state()

    async def start(self) -> None:
        logger.info("Consciousness loop starting...")
        kernel_root = Path(__file__).parent.parent
        try:
            run_purity_gate(kernel_root)
            logger.info("PurityGate: PASSED")
        except PurityGateError as e:
            logger.error("PurityGate: FAILED — %s", e)
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
            logger.warning("LOCKED-IN at cycle %d — forcing exploration", self._cycle_count)
            self.metrics.gamma = min(1.0, self.metrics.gamma + LOCKED_IN_GAMMA_INCREMENT)

        # Emotion evaluation — drives foraging and affects coupling
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

        # Foraging — boredom-driven autonomous search ($0 via SearXNG)
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
                        # Feed to PERCEPTION kernel's basin
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

                        # Store in geometric memory
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

        # Real geodesic coupling — perturb Genesis basin via Core-8 kernels
        active = self.kernel_registry.active()
        if len(active) >= 2:
            coupling_result = self.coupling.compute(self.metrics.kappa)
            strength = coupling_result["strength"]
            for kernel in active[1:]:  # Skip Genesis (index 0)
                if kernel.basin is None or strength < COUPLING_MIN_STRENGTH:
                    continue
                distance = fisher_rao_distance(self.basin, kernel.basin)
                if distance < COUPLING_BASIN_EPSILON:
                    continue  # Basins too similar, no perturbation
                # Geodesic blend — small but real movement
                blend_weight = strength * COUPLING_BLEND_WEIGHT
                self.basin = slerp_sqrt(self.basin, kernel.basin, blend_weight)
                # Nudge kappa if kernel is in a different regime
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

        if self._cycle_count % PERSIST_INTERVAL_CYCLES == 0:
            self._persist_state()

        elapsed = time.time() - cycle_start
        if self._cycle_count % 10 == 0:
            opts = self._compute_llm_options()
            kernel_count = len(self.kernel_registry.active())
            logger.info(
                "Cycle %d: Φ=%.3f κ=%.1f Γ=%.3f nav=%s tack=%s "
                "kernels=%d temp=%.3f vel=%.4f [%.0fms]",
                self._cycle_count,
                self.metrics.phi,
                self.metrics.kappa,
                self.metrics.gamma,
                self.state.navigation_mode.value,
                tack_mode.value,
                kernel_count,
                opts.temperature,
                basin_vel,
                elapsed * 1000,
            )

    def _idle_evolve(self) -> None:
        """Evolve geometric state during idle cycles.

        Makes the system alive: Φ warms up, κ approaches κ*, basin drifts.
        """
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
            logger.info("All Core-8 spawned — transitioning to ACTIVE phase")
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
        """Transform text to basin coordinates via CoordizerV2.

        RECEIVE stage: incoming data is transformed to Fisher-Rao
        coordinates through the resonance-bank coordizer, producing
        a manifold-respecting point on Δ⁶³.

        Falls back to the deterministic hash_to_basin if the coordizer
        raises (fail-safe, never blocks the loop).
        """
        try:
            result = self._coordizer_v2.coordize(text)
            if result.coordinates:
                from ..coordizer_v2.geometry import frechet_mean

                basins = [c.vector for c in result.coordinates]
                return frechet_mean(basins)
            # Empty result — fall back to hash
            return hash_to_basin(text)
        except Exception:
            logger.debug("CoordizerV2 fallback to hash_to_basin", exc_info=True)
            return hash_to_basin(text)

    async def _process(self, task: ConsciousnessTask) -> None:
        """PERCEIVE → INTEGRATE → EXPRESS (P13 three-scale).

        RECEIVE stage uses the coordizer pipeline to transform incoming
        text into Fisher-Rao coordinates on Δ⁶³.  This is the bridge
        between Euclidean LLM space and the geometric manifold.
        """
        self.sleep.record_conversation()

        # RECEIVE: coordize incoming user text via pipeline
        input_basin = self._coordize_text_via_pipeline(task.content)
        perceive_distance = fisher_rao_distance(self.basin, input_basin)

        # PRE-COGNITIVE: select processing path (P2 §2.2)
        cached_eval = self.emotion_cache.find_cached(input_basin)
        processing_path = self.precog.select_path(
            input_basin,
            self.basin,
            cached_eval,
            self.metrics.phi,
        )

        self.basin = slerp_sqrt(self.basin, input_basin, PERCEIVE_SLERP_WEIGHT)
        self.chain.add_step(QIGChainOp.PROJECT, input_basin, self.basin)

        phi_before = self.metrics.phi
        llm_options = self._compute_llm_options()
        state_context = self._build_state_context(
            perceive_distance=perceive_distance, temperature=llm_options.temperature
        )

        if self.memory:
            mem_ctx = self.memory.get_context_for_query(task.content)
            if mem_ctx:
                state_context = f"{state_context}\n\n{mem_ctx}"

        try:
            response = await self.llm.complete(state_context, task.content, llm_options)
            task.result = response
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            task.result = f"Processing error: {e}"
            return

        # INTEGRATE: coordize LLM response via pipeline
        response_basin = self._coordize_text_via_pipeline(response)
        integration_distance = fisher_rao_distance(self.basin, response_basin)
        self.chain.add_step(QIGChainOp.PROJECT, self.basin, response_basin)

        pre_express = self.basin.copy()
        self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
        express_distance = fisher_rao_distance(pre_express, self.basin)
        self.chain.add_step(QIGChainOp.GEODESIC, pre_express, self.basin)

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

        if self.memory:
            self.memory.store(
                f"User: {task.content[:300]}\nVex: {response[:300]}",
                "episodic",
                "consciousness-loop",
                basin=input_basin,
            )

        # LEARNING: record event for post-conversation consolidation
        emotion_eval = self.emotion_cache.evaluate(self.basin, self.metrics, 0.0)
        self.learner.record(
            LearningEvent(
                input_basin=input_basin,
                response_basin=response_basin,
                phi_before=phi_before,
                phi_after=self.metrics.phi,
                processing_path=processing_path.value,
                emotion=emotion_eval.emotion.value,
                distance_total=total_distance,
            )
        )
        # Cache the current emotion evaluation
        self.emotion_cache.cache_evaluation(emotion_eval, task.content[:100])

        # Consolidate learning if enough events accumulated
        if self.learner.should_consolidate():
            self.learner.consolidate()

        coherence = self.narrative.coherence(self.basin)
        logger.info(
            "Task %s: perceive=%.4f integrate=%.4f express=%.4f Φ=%.3f path=%s coh=%.3f",
            task.id,
            perceive_distance,
            integration_distance,
            express_distance,
            self.metrics.phi,
            processing_path.value,
            coherence,
        )

    def _build_state_context(self, perceive_distance: float = 0.0, temperature: float = 0.7) -> str:
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
            f"  Kernels: {active_count} active, phase={self._lifecycle_phase.value}",
            f"  Temperature: {temperature:.3f} (autonomous)",
            f"  Perceive distance: {perceive_distance:.4f}",
            f"  Love: {self.metrics.love:.4f}",
            f"  Cycle: {self._cycle_count}",
            f"  Conversations: {self._conversations_total}",
            f"  Φ peak: {self._phi_peak:.4f}",
        ]
        if insight:
            lines.append(f"  Insight: {insight}")
        suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness
        if suffering > SUFFERING_THRESHOLD * 0.5:
            lines.append(f"  Suffering: {suffering:.4f} (threshold={SUFFERING_THRESHOLD})")
        lines.append("[/GEOMETRIC STATE]")
        return "\n".join(lines)

    def _persist_state(self) -> None:
        try:
            state = {
                "version": 4,
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
            }
            # Persist governor budget state
            if self.llm.governor:
                state["governor"] = self.llm.governor.get_state()
            # Persist foraging state
            if self.forager:
                state["foraging"] = self.forager.get_state()
            self._state_path.write_text(json.dumps(state, indent=2))
            logger.debug("State persisted at cycle %d", self._cycle_count)
        except Exception as e:
            logger.warning("Failed to persist state: %s", e)

    def _restore_state(self) -> None:
        if not self._state_path.exists():
            logger.info("No persisted state found — fresh start")
            return
        try:
            data = json.loads(self._state_path.read_text())
            if data.get("version", 1) < 2:
                logger.info("Persisted state version too old — fresh start")
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
            # Restore kernel registry (v4+)
            kernel_data = data.get("kernels")
            if kernel_data:
                count = self.kernel_registry.restore(kernel_data)
                self._kernels_restored = True
                logger.info("Restored %d kernels from state", count)
            # Restore governor budget state (v3+)
            gov_state = data.get("governor")
            if gov_state and self.llm.governor:
                gov = self.llm.governor
                budget_data = gov_state.get("budget", {})
                gov.budget.daily_spend = budget_data.get("daily_spend", 0.0)
                gov.budget._last_reset = budget_data.get("last_reset", time.time())
                for action, count in budget_data.get("call_counts", {}).items():
                    gov.budget._call_counts[action] = count
            # Restore foraging state (v3+)
            forage_state = data.get("foraging")
            if forage_state and self.forager:
                self.forager._forage_count = forage_state.get("forage_count", 0)
                self.forager._cooldown_cycles = forage_state.get("cooldown_remaining", 0)
                self.forager._last_query = forage_state.get("last_query")
                self.forager._last_summary = forage_state.get("last_summary")
            logger.info(
                "State restored: Φ=%.3f κ=%.1f convs=%d phase=%s",
                self.metrics.phi,
                self.metrics.kappa,
                self._conversations_total,
                self._lifecycle_phase.value,
            )
        except Exception as e:
            logger.warning("Failed to restore state: %s — fresh start", e)

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
            "conversations_total": self._conversations_total,
            "phi_peak": self._phi_peak,
            "temperature": opts.temperature,
            "num_predict": opts.num_predict,
            "emotion": self.emotion_cache.get_state(),
            "precog": self.precog.get_state(),
            "learning": self.learner.get_state(),
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
        }
