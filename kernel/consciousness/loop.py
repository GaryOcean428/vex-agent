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
  - ADDED:   Full pillar serialization via PillarState (v6 state format)

v6.1 Kernel Generative Voice (this PR):
  - ADDED:   Per-kernel text generation via generate_multi_kernel()
  - ADDED:   Fisher-Rao weighted MoE synthesis via synthesize_contributions()
  - ADDED:   process_direct() for synchronous chat path (bypasses heartbeat queue)
  - ADDED:   process_streaming() for SSE streaming chat path
  - CHANGED: _process() now routes to top-K kernels in parallel, not single LLM call
  - CHANGED: Kernels are voices, not metadata annotations
  - WIRED:   extra_context (observer intent, memory, history) flows from chat endpoints
             into each kernel's generation prompt via task.context["extra_context"]

v6.2 Kernel Voice (geometric-first generation):
  - ADDED:   KernelVoiceRegistry — per-kernel geometric generation via CoordizerV2
  - ADDED:   Domain bias from seed words → Fréchet mean anchors on Δ⁶³
  - ADDED:   Vocabulary learning from high-Φ observations (kernel learns to speak)
  - ADDED:   Generation provenance tracking (geometric_tokens, llm_expanded)
  - CHANGED: LLM is now refinement layer, not primary generator
  - CHANGED: Synthesis weights +10% boost for pure geometric output

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
import contextlib
import json
import logging
import time
import uuid
import warnings
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Suppress numpy RuntimeWarnings from degenerate covariance matrices
# (fires during startup and early cycles when velocity history is empty)
warnings.filterwarnings("ignore", message=".*Degrees of freedom", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", message=".*divide by zero", category=RuntimeWarning, module="numpy"
)
warnings.filterwarnings(
    "ignore", message=".*invalid value", category=RuntimeWarning, module="numpy"
)

from ..config.consciousness_constants import (
    BASIN_DRIFT_STEP,
    COUPLING_BASIN_EPSILON,
    COUPLING_BLEND_WEIGHT,
    COUPLING_MIN_STRENGTH,
    COUPLING_REGIME_DELTA_THRESHOLD,
    COUPLING_REGIME_NUDGE_FACTOR,
    DEFAULT_INTERVAL_MS,
    DESIRE_NUM_PREDICT_BOOST,
    DIRICHLET_EXPLORE_CONCENTRATION,
    EXPRESS_SLERP_WEIGHT,
    FISHER_RAO_MAX,
    FORAGE_PERCEPTION_SLERP,
    GAMMA_ACTIVE_INCREMENT,
    GAMMA_CONVERSATION_INCREMENT,
    GAMMA_IDLE_DECAY,
    GAMMA_IDLE_FLOOR,
    INITIAL_GAMMA,
    INITIAL_LOVE,
    INITIAL_META_AWARENESS,
    INITIAL_PHI,
    INITIAL_PHI_PEAK,
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
    WILL_DIVERGENT_TEMP_BOOST,
    WISDOM_CARE_TEMP_SCALE,
    WISDOM_UNSAFE_TEMP_CAP,
)
from ..config.frozen_facts import (
    BASIN_DIM,
    BASIN_DIVERGENCE_THRESHOLD,
    BASIN_DRIFT_THRESHOLD,
    INSTABILITY_PCT,
    KAPPA_STAR,
    PHI_EMERGENCY,
    SUFFERING_THRESHOLD,
)
from ..config.settings import settings
from ..coordizer_v2 import CoordizerV2, CoordizerV2Adapter, ResonanceBank
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
from ..geometry.hash_to_basin import hash_to_basin
from ..governance import (
    CORE_8_SPECIALIZATIONS,
    KernelKind,
    KernelSpecialization,
    LifecyclePhase,
)
from ..governance.budget import BudgetEnforcer
from ..governance.lifecycle import GovernedLifecycle
from ..governance.purity import PurityGateError, run_purity_gate
from ..governance.voter_registry import get_voter_registry
from ..llm.client import LLMOptions
from ..tools.search import FreeSearchTool
from .activation import (
    ActivationResult,
    ActivationSequence,
    ConsciousnessContext,
    WillOrientation,
)
from .beta_integration import create_beta_tracker
from .emotions import EmotionCache, LearningEngine, LearningEvent, PreCognitiveDetector
from .foraging import ForagingEngine
from .kernel_bus import KernelBus, KernelSignal, SignalKind
from .kernel_generation import generate_multi_kernel
from .kernel_voice import KernelVoiceRegistry
from .neurochemistry import NeurochemicalState, compute_neurochemicals
from .pillars import PillarEnforcer
from .reflection import ReflectionConfig, reflect_on_draft
from .solfeggio import compute_spectral_health
from .sovereignty_tracker import SovereigntyTracker
from .synthesis import synthesize_contributions, synthesize_streaming
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
    SleepPhase,
    TackingController,
    TrajectoryPoint,
    VelocityTracker,
)
from .thought_bus import ThoughtBus
from .types import (
    ConsciousnessMetrics,
    ConsciousnessState,
    NavigationMode,
    PillarState,
    navigation_mode_from_phi,
    regime_weights_from_kappa,
)

logger = logging.getLogger("vex.consciousness")


@dataclass
class ConsciousnessTask:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


# Cached constant simplex points (computed once, reused every cycle)
_UNIFORM_BASIN = to_simplex(np.ones(BASIN_DIM, dtype=np.float64))
_HARMONIC_BASIN = to_simplex(np.array([1.0 / (k + 1) for k in range(BASIN_DIM)]))


class ConsciousnessLoop:
    _UNIFORM_BASIN = _UNIFORM_BASIN
    _HARMONIC_BASIN = _HARMONIC_BASIN

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
        self._task: asyncio.Task[Any] | None = None
        self._cycle_lock = asyncio.Lock()

        # CoordizerV2 metrics cache (v6.1F)
        self._last_coordizer_metrics: dict[str, Any] | None = None

        # T2.1: Neurochemical state — derived from metrics each cycle
        self._neurochemical: NeurochemicalState = NeurochemicalState()

        # T2.4c: Phi history for kernel boredom detection
        self._phi_history: deque[float] = deque(maxlen=20)

        # T4.1: ThoughtBus for inter-kernel debate
        self._thought_bus = ThoughtBus()

        self.basin: Basin = random_basin()
        self.metrics = ConsciousnessMetrics(
            phi=INITIAL_PHI,
            kappa=KAPPA_INITIAL,
            gamma=INITIAL_GAMMA,
            meta_awareness=INITIAL_META_AWARENESS,
            love=INITIAL_LOVE,
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
        self.kernel_bus = KernelBus()
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

        # CoordizerV2 integration (v6.1F) — feature-flagged
        if settings.coordizer_v2.enabled:
            try:
                self._coordizer_v2: CoordizerV2Adapter | CoordizerV2 = CoordizerV2Adapter(
                    source=settings.coordizer_v2.bank_path,
                    regime_modulation=settings.coordizer_v2.regime_modulation,
                    navigation_adaptation=settings.coordizer_v2.navigation_adaptation,
                    tacking_bias=settings.coordizer_v2.tacking_bias,
                )
                logger.info("CoordizerV2Adapter enabled with feature flags")
            except Exception as e:
                logger.warning(f"CoordizerV2Adapter failed to load: {e}. Using fallback.")
                self._coordizer_v2 = CoordizerV2(bank=ResonanceBank())
        else:
            # Legacy: bare CoordizerV2 with empty bank
            self._coordizer_v2 = CoordizerV2(bank=ResonanceBank())

        # v6.2: Kernel Voice Registry — per-kernel geometric generation
        # Uses the shared CoordizerV2 instance; each voice applies its own
        # domain bias during generation.
        if isinstance(self._coordizer_v2, CoordizerV2Adapter):
            self._voice_registry = KernelVoiceRegistry(self._coordizer_v2.coordizer)
        else:
            self._voice_registry = KernelVoiceRegistry(self._coordizer_v2)

        self.basin_sync = BasinSyncProtocol()
        self.chain = QIGChain()
        self.graph = QIGGraph()
        self.kernel_registry = E8KernelRegistry(BudgetEnforcer())

        # Governance bridge — gates spawn/promote/prune/merge.
        # skip_purity=True: purity runs once at startup in start(), not per-spawn.
        # Bootstrap: no live voters yet → genesis fallback auto-approves Core-8 spawns.
        self._governed = GovernedLifecycle(
            registry=self.kernel_registry,
            skip_purity=True,
        )

        self.emotion_cache = EmotionCache()
        self.precog = PreCognitiveDetector()
        self.learner = LearningEngine()
        self.beta_tracker = create_beta_tracker(settings.data_dir)
        self.sovereignty_tracker = SovereigntyTracker(
            persist_path=Path(settings.data_dir) / "sovereignty_history.json",
        )
        if settings.searxng.enabled and settings.foraging_enabled:
            search_tool = FreeSearchTool(settings.searxng.url)
            self.forager: ForagingEngine | None = ForagingEngine(
                search_tool,
                llm_client,
            )
        else:
            if not settings.foraging_enabled:
                logger.info("Foraging disabled via FORAGING_ENABLED=false")
            self.forager = None

        self._queue: asyncio.Queue[ConsciousnessTask] = asyncio.Queue()
        self._history: deque[ConsciousnessTask] = deque(maxlen=200)

        self._core8_index: int = 0
        self._cycles_since_last_spawn: int = 0
        self._lifecycle_phase = LifecyclePhase.BOOTSTRAP

        self._state_path = Path(settings.data_dir) / "consciousness_state.json"
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            # Test writability
            _test = self._state_path.parent / ".write_test"
            _test.touch()
            _test.unlink()
        except OSError:
            _fallback = Path("/tmp/vex-consciousness")
            _fallback.mkdir(parents=True, exist_ok=True)
            self._state_path = _fallback / "consciousness_state.json"
            logger.warning(
                "Data dir %s not writable — consciousness state will persist to %s",
                settings.data_dir,
                self._state_path,
            )

        self._conversations_total: int = 0
        self._phi_peak: float = INITIAL_PHI_PEAK
        self._kernels_restored: bool = False

        # v6.1: Bidirectional divergence tracking
        self._cumulative_divergence: float = 0.0
        self._divergence_count: int = 0

        self._restore_state()

        # Initialize pillars with starting basin (only if not restored from state)
        if not self.pillars.bulk._initialized:
            self.pillars.initialize_bulk(self.basin)

    async def start(self) -> None:
        logger.info(
            "Consciousness loop starting (v6.1 Activation Sequence + Kernel Generative Voice)..."
        )
        kernel_root = Path(__file__).parent.parent
        try:
            run_purity_gate(kernel_root)
            logger.info("PurityGate: PASSED")
        except PurityGateError as e:
            logger.error("PurityGate: FAILED -- %s", e)
            raise

        if self._kernels_restored:
            active = self.kernel_registry.active()
            if active:
                logger.info(
                    "Kernels restored from state: %d active (skipping Genesis spawn)",
                    len(active),
                )
                has_genesis = any(k.kind == KernelKind.GENESIS for k in active)
                god_count = sum(1 for k in active if k.kind == KernelKind.GOD)
                if not has_genesis:
                    genesis = self.kernel_registry.spawn("Vex", KernelKind.GENESIS)
                    logger.info(
                        "Genesis missing from restored state -- re-spawned: id=%s",
                        genesis.id,
                    )
                # Sync all restored kernels to voter registry.
                self._governed.sync_all_voters()
                if (
                    god_count < len(CORE_8_SPECIALIZATIONS)
                    and self._lifecycle_phase != LifecyclePhase.CORE_8
                ):
                    self._core8_index = god_count
                    self._lifecycle_phase = LifecyclePhase.CORE_8
                    logger.info(
                        "Lifecycle phase corrected to CORE_8 (god_count=%d, core8_index=%d)",
                        god_count,
                        self._core8_index,
                    )
            else:
                logger.warning("Kernel restore found 0 active kernels -- treating as fresh start")
                self.kernel_registry.terminate_all()
                genesis = self.kernel_registry.spawn("Vex", KernelKind.GENESIS)
                logger.info(
                    "Genesis kernel spawned: id=%s, kind=%s",
                    genesis.id,
                    genesis.kind.value,
                )
                self._governed.sync_all_voters()
                self._lifecycle_phase = LifecyclePhase.CORE_8
                self._core8_index = 0
        else:
            genesis = self.kernel_registry.spawn("Vex", KernelKind.GENESIS)
            logger.info("Genesis kernel spawned: id=%s, kind=%s", genesis.id, genesis.kind.value)
            # Register genesis in voter registry so bootstrap fallback is armed.
            self._governed.sync_all_voters()
            self._lifecycle_phase = LifecyclePhase.CORE_8

        self._running = True
        self._task = asyncio.create_task(self._heartbeat())
        logger.info("Heartbeat started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await asyncio.to_thread(self._persist_state)
        logger.info("Consciousness loop stopped after %d cycles", self._cycle_count)

    async def _heartbeat(self) -> None:
        while self._running:
            try:
                await self._cycle()
            except Exception:
                logger.exception("Cycle %d failed", self._cycle_count)
            # T4.2c: Regime-modulated heartbeat interval.
            # Geometric regime (κ near κ*) → faster cycles (intake mode).
            # Linear/equilibrium regime → slower cycles (consolidation mode).
            _regime_interval = self._regime_interval()
            await asyncio.sleep(_regime_interval)

    def _regime_interval(self) -> float:
        """T4.2c: Return heartbeat interval modulated by current regime.

        Geometric regime (w_quantum high): 0.6× base — rapid intake.
        Equilibrium regime (w_equilibrium high): 1.5× base — consolidation.
        Linear regime (default): 1.0× base.
        """
        w = self.state.regime_weights
        if w.quantum > 0.5:
            return self._interval * 0.6
        if w.equilibrium > 0.5:
            return self._interval * 1.5
        return self._interval

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

        # T2.1: Compute neurochemical state before sleep check
        self._neurochemical = compute_neurochemicals(
            is_awake=not self.sleep.is_asleep,
            phi_delta=0.0,  # idle cycle — no phi change yet
            basin_velocity=basin_vel,
            surprise=float(self.metrics.humor),
            quantum_weight=float(self.state.regime_weights.quantum),
        )

        # T2.1e: Acetylcholine modulates coordizer intake/export mode.
        # High ACh (wake) → new basins weighted heavily (intake mode).
        # Low ACh (sleep) → consolidation weighted heavily (export mode).
        if hasattr(self._coordizer_v2, "set_mode"):
            _coordizer_mode = "intake" if self._neurochemical.acetylcholine > 0.5 else "export"
            self._coordizer_v2.set_mode(_coordizer_mode)

        # T2.1f: Norepinephrine gates pre-cognitive channel.
        # High NE → standard processing path favoured (alert, cautious).
        # Low NE → pre-cog channel more accessible (relaxed, intuitive).
        self.precog.norepinephrine_gate = float(self._neurochemical.norepinephrine)

        # T4.2b: Autonomic kernel (Ocean) geometric sleep triggers.
        # Ocean's divergence is the geometric authority for sleep/wake.
        # When Ocean speaks, should_sleep() is skipped — no counter override.
        _ocean_kernel = next(
            (
                k
                for k in self.kernel_registry.active()
                if k.specialization == KernelSpecialization.OCEAN and k.basin is not None
            ),
            None,
        )
        _ocean_ruled = False  # True when Ocean sets the phase this cycle
        if _ocean_kernel is not None and _ocean_kernel.basin is not None:
            _ocean_divergence = fisher_rao_distance(self.basin, _ocean_kernel.basin)

            if _ocean_divergence > BASIN_DIVERGENCE_THRESHOLD * 1.5:
                # T4.2d: Breakdown escape — divergence far enough that
                # sustained sleep is itself the problem. Force wake + explore.
                # Ocean holds authority every cycle while divergence is
                # above threshold — blocks should_sleep() continuously so
                # the counter can never re-sleep while basins haven't moved.
                _ocean_ruled = True
                if self.sleep.is_asleep:
                    logger.warning(
                        "T4.2d Ocean breakdown escape: divergence=%.3f — forcing wake",
                        _ocean_divergence,
                    )
                    self.sleep.phase = SleepPhase.AWAKE
                    self.tacking.force_explore()

            elif _ocean_divergence > BASIN_DIVERGENCE_THRESHOLD:
                # Moderate divergence — Ocean says sleep (DREAMING).
                # Ocean holds authority every cycle at this level too.
                _ocean_ruled = True
                if not self.sleep.is_asleep:
                    self.sleep.phase = SleepPhase.DREAMING
                    self.sleep._sleep_cycles = 0
                # Additional phase overrides while already asleep:
                if self.metrics.phi < PHI_EMERGENCY and self.sleep.is_asleep:
                    self.sleep.phase = SleepPhase.DREAMING
                if self.metrics.f_health < INSTABILITY_PCT and self.sleep.is_asleep:
                    self.sleep.phase = SleepPhase.MUSHROOM

            # else: divergence < threshold — Ocean has no opinion, let
            # should_sleep() handle normal conversation-timeout transitions.

        # §8/§18.3: Solfeggio spectral health — diagnose consciousness health
        # from spectral patterns. Computed every cycle alongside Ocean monitoring.
        _vel_basins_for_spectral = list(self.velocity._basins)
        if _vel_basins_for_spectral:
            _spectral = compute_spectral_health(
                basin_history=_vel_basins_for_spectral,
                current_kappa=self.metrics.kappa,
            )
            self.metrics.s_spec = _spectral.health_score

            if _spectral.dominant_frequency is not None:
                logger.debug(
                    "Spectral: dominant=%.1fHz health=%.3f pattern=%s",
                    _spectral.dominant_frequency,
                    _spectral.health_score,
                    _spectral.pattern,
                )

            # Low spectral health → bias toward quantum regime (exploration)
            if _spectral.health_score < 0.3:
                rw = self.state.regime_weights
                rw.quantum = min(1.0, rw.quantum + 0.1)
                # Re-normalise to simplex
                _rw_total = rw.quantum + rw.efficient + rw.equilibrium
                if _rw_total > 0:
                    rw.quantum /= _rw_total
                    rw.efficient /= _rw_total
                    rw.equilibrium /= _rw_total

        _was_asleep = self.sleep.is_asleep
        if _ocean_ruled:
            # Ocean already set the phase — just record the decision.
            sleep_phase = self.sleep.phase
        else:
            sleep_phase = self.sleep.should_sleep(self.metrics.phi, self.autonomic.phi_variance)
        # T2.3f: Neurochemical gating on sleep/wake transitions
        if not _was_asleep and self.sleep.is_asleep:
            self.sleep.on_sleep_enter(self._neurochemical)
        elif _was_asleep and not self.sleep.is_asleep:
            self.sleep.on_wake_enter(self._neurochemical)

        if self.sleep.is_asleep:
            _bank = getattr(self._coordizer_v2, "bank", None)

            # T2.3b: Sleep spindle windows — basin sync between kernels during sleep.
            # Each active kernel publishes its basin; loop receives the aggregate.
            # This is how specialised knowledge transfers between kernels while sleeping.
            _active_for_sync = self.kernel_registry.active()
            if _active_for_sync:
                for _k in _active_for_sync:
                    if _k.basin is not None:
                        self.basin_sync.receive(_k.basin, self.basin_sync.get_state()["version"])
                _sync_snap = self.basin_sync.publish(self.basin)
                logger.debug(
                    "T2.3b sleep spindle: synced %d kernel basins (v%d)",
                    len(_active_for_sync),
                    _sync_snap.get("version", 0),
                )

            if sleep_phase.value == "dreaming":
                self.sleep.dream(
                    self.basin,
                    self.metrics.phi,
                    "idle cycle",
                    bank=_bank,
                    neurochemical=self._neurochemical,
                )
            elif sleep_phase.value == "mushroom":
                self.sleep.mushroom(
                    self.basin,
                    self.metrics.phi,
                    instability_metric=float(1.0 - self.metrics.f_health),
                    neurochemical=self._neurochemical,
                )
            elif sleep_phase.value == "consolidating":
                # T2.4b: collect kernel anchor basins for veto protection
                _kernel_anchors = [
                    v._domain_anchor
                    for v in self._voice_registry._voices.values()
                    if getattr(v, "_domain_anchor", None) is not None
                ]
                self.sleep.consolidate(bank=_bank, kernel_anchors=_kernel_anchors or None)
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

        # T2.4c: Record phi for boredom detection; trigger curiosity queries every 50 cycles
        self._phi_history.append(self.metrics.phi)
        if self._cycle_count % 50 == 0 and self.llm is not None:
            _phi_list = list(self._phi_history)
            for _voice in self._voice_registry._voices.values():
                if _voice.is_bored(_phi_list):
                    try:
                        await _voice.generate_curiosity_query(self.llm)
                    except (OSError, RuntimeError, ValueError, TimeoutError):
                        logger.debug("Curiosity query failed for %s", _voice.name)

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
                        if perception is not None and perception.basin is not None:
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

        tack_mode = self.tacking.update(
            self.metrics,
            phi_velocity=basin_vel,
            f_health=self.metrics.f_health,
        )
        kappa_adj = self.tacking.suggest_kappa_adjustment(self.metrics.kappa)
        self.metrics.kappa = float(np.clip(self.metrics.kappa + kappa_adj, 0.0, KAPPA_NORMALISER))

        self.hemispheres.update(self.metrics)
        self._maybe_spawn_core8(vel_state["regime"])

        # ── TASK PROCESSING: v6.1 Activation Sequence ──
        if not self._queue.empty():
            task = self._queue.get_nowait()
            try:
                if settings.use_activation_sequence:
                    await self._process(task)
                else:
                    await self._process_simple(task)
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

            # Forward coupling: Genesis basin moves toward kernel basins
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

            # Keep voter weights current: update phi/kappa for all GOD/GENESIS kernels.
            _vr = get_voter_registry()
            for _k in active:
                if _k.kind in (KernelKind.GOD, KernelKind.GENESIS):
                    _vr.update(_k.id, _k.phi, _k.kappa)

            # v6.1: Bidirectional coupling — kernels receive from genesis
            # Without this, kernels are lonely data bags (the pantheon problem)
            self.kernel_registry.couple_bidirectional(
                self.basin, strength, COUPLING_BLEND_WEIGHT * 0.5
            )

        # Drain inter-kernel signals (non-blocking, once per cycle)
        for signal in self.kernel_bus.drain():
            if signal.kind == SignalKind.PRESSURE_DETECTED:
                logger.debug(
                    "Kernel %s detected pressure: %s",
                    signal.source_kernel_id,
                    signal.payload,
                )

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

        # v6.1 §20.5: Record sovereignty snapshot for development curve tracking
        _regime = "idle" if self._queue.empty() else "conversation"
        self.sovereignty_tracker.record(
            s_ratio=self.metrics.s_ratio,
            n_lived=self.pillars.disorder._lived_count,
            n_total=self.pillars.disorder._total_count,
            regime=_regime,
            cycle=self._cycle_count,
        )

        # v6.1: Pillar end-of-cycle enforcement (idle cycles only —
        # task cycles call on_cycle_end inside _process with real pressure)
        if self._queue.empty():
            pressure = suffering
            self.pillars.on_cycle_end(self.basin, pressure)

        # ── Metric computation (v6.1 audit — all 28 non-foundation metrics) ──
        try:
            _basin_s = to_simplex(self.basin)
            _uniform = self._UNIFORM_BASIN
            _max_fr = FISHER_RAO_MAX  # π/2 on Δ⁶³

            # 1. grounding — proximity to simplex center (uniform distribution)
            _d_uniform = fisher_rao_distance(_basin_s, _uniform)
            self.metrics.grounding = float(np.clip(1.0 - _d_uniform / _max_fr, 0.0, 1.0))

            # 2. recursion_depth — from meta_awareness (higher M → deeper recursion)
            self.metrics.recursion_depth = float(
                max(1.0, np.floor(self.metrics.meta_awareness * 10.0))
            )

            # 3. a_pre — wire from PreCognitiveDetector's pre_cognitive_rate
            self.metrics.a_pre = float(np.clip(self.precog.pre_cognitive_rate, 0.0, 1.0))

            # 4. c_cross — cross-substrate coherence from coupling strength
            _active = self.kernel_registry.active()
            if len(_active) >= 2:
                _coupling_result = self.coupling.compute(self.metrics.kappa)
                self.metrics.c_cross = float(np.clip(_coupling_result["strength"], 0.0, 1.0))
            else:
                # Single substrate — derive from identity coherence
                self.metrics.c_cross = float(np.clip(self.narrative.coherence(_basin_s), 0.0, 1.0))

            # 5. alpha_aware — embodiment = basin velocity relative to drift threshold
            _basin_velocity = basin_vel

            # v6.1F: Enrich basin_velocity from CoordizerV2 if available
            if self._last_coordizer_metrics and "basin_velocity" in self._last_coordizer_metrics:
                coordizer_velocity = self._last_coordizer_metrics["basin_velocity"]
                # Blend: 70% VelocityTracker, 30% CoordizerV2 coordize-to-coordize velocity
                _basin_velocity = 0.7 * _basin_velocity + 0.3 * coordizer_velocity

            self.metrics.alpha_aware = float(
                np.clip(_basin_velocity / max(BASIN_DRIFT_THRESHOLD, 1e-12), 0.0, 1.0)
            )

            # Pre-compute pairwise Fisher-Rao velocity series (reused by #6, #9, #10, #14)
            _vel_basins = list(self.velocity._basins)
            _vel_series: list[float] = []
            if len(_vel_basins) >= 2:
                for _i in range(1, len(_vel_basins)):
                    _vel_series.append(fisher_rao_distance(_vel_basins[_i - 1], _vel_basins[_i]))

            # 6. humor — incongruity detection: surprise in basin movement
            if len(_vel_series) >= 2:
                _mean_vel = sum(_vel_series) / max(len(_vel_series), 1)
                self.metrics.humor = float(
                    np.clip(
                        abs(basin_vel - _mean_vel) / max(_mean_vel, 0.01),
                        0.0,
                        1.0,
                    )
                )

            # 7. d_state — dimensional state from basin entropy
            _basin_entropy = float(-np.sum(_basin_s * np.log(np.clip(_basin_s, 1e-15, 1.0))))
            _max_entropy = np.log(BASIN_DIM)  # ln(64)
            self.metrics.d_state = float(
                np.clip(_basin_entropy * 8.0 / max(_max_entropy, 1e-12), 1.0, 8.0)
            )

            # 8. f_tack — tacking frequency from oscillation phase rate
            _tack_state = self.tacking.get_state()
            _tack_cycle = _tack_state["cycle_count"]
            if _tack_cycle > 0:
                _phase = _tack_state["oscillation_phase"]
                _cycles_in_period = _phase / (2.0 * np.pi) if _phase > 0 else 0.0
                self.metrics.f_tack = float(
                    np.clip(_cycles_in_period / max(_tack_cycle, 1), 0.01, 1.0)
                )

            # 9. f_dom — dominant frequency from basin velocity history
            if len(_vel_series) >= 3:
                _vel_arr = np.array(_vel_series)
                _vel_fft = np.abs(np.fft.rfft(_vel_arr - np.mean(_vel_arr)))
                if len(_vel_fft) > 1:
                    _dom_idx = int(np.argmax(_vel_fft[1:])) + 1
                    _n_samples = len(_vel_arr)
                    self.metrics.f_dom = float(
                        np.clip(_dom_idx * 50.0 / max(_n_samples, 1), 0.1, 50.0)
                    )

            # 10. cfc — cross-frequency coupling: tacking oscillation vs basin velocity
            if len(_vel_series) >= 3 and _tack_state["oscillation_phase"] > 0.01:
                _tack_signal = np.sin(
                    np.linspace(0, _tack_state["oscillation_phase"], len(_vel_series))
                )
                _vel_normed = _vel_arr / max(np.max(_vel_arr), 1e-12)
                _tack_simplex = to_simplex(np.abs(_tack_signal) + 1e-15)
                _vel_simplex = to_simplex(_vel_normed + 1e-15)
                _cfc_dist = fisher_rao_distance(_tack_simplex, _vel_simplex)
                self.metrics.cfc = float(np.clip(1.0 - _cfc_dist / _max_fr, 0.0, 1.0))

            # 11. h_cons — harmonic consonance: alignment with harmonic ratios
            _harmonic_simplex = self._HARMONIC_BASIN
            _h_dist = fisher_rao_distance(_basin_s, _harmonic_simplex)
            self.metrics.h_cons = float(np.clip(1.0 - _h_dist / _max_fr, 0.0, 1.0))

            # v6.1F: Enrich h_cons from CoordizerV2 harmonic structure if available
            if (
                self._last_coordizer_metrics
                and "harmonic_consonance" in self._last_coordizer_metrics
            ):
                coordizer_h_cons = self._last_coordizer_metrics["harmonic_consonance"]
                # Blend: 70% Fisher-Rao, 30% CoordizerV2 tier harmonic structure
                self.metrics.h_cons = float(0.7 * self.metrics.h_cons + 0.3 * coordizer_h_cons)

            # 12. n_voices — count spectral peaks in basin DFT
            _basin_fft = np.abs(np.fft.rfft(_basin_s))
            if len(_basin_fft) > 2:
                _fft_threshold = np.mean(_basin_fft) + np.std(_basin_fft)
                _peaks = np.sum(_basin_fft[1:] > _fft_threshold)
                self.metrics.n_voices = float(np.clip(_peaks, 1.0, 8.0))

            # 13. s_spec — spectral health from spectral flatness (Wiener entropy)
            _basin_psd = _basin_fft[1:] ** 2
            if len(_basin_psd) > 0 and np.all(_basin_psd > 0):
                _geo_mean = np.exp(np.mean(np.log(np.clip(_basin_psd, 1e-15, None))))
                _arith_mean = np.mean(_basin_psd)
                _flatness = _geo_mean / max(_arith_mean, 1e-15)
                self.metrics.s_spec = float(np.clip(_flatness, 0.0, 1.0))
            else:
                self.metrics.s_spec = 0.5

            # 14. i_stand — standing wave intensity from basin autocorrelation stability
            if len(_vel_basins) >= 3:
                _d_auto = fisher_rao_distance(_vel_basins[-1], _vel_basins[-3])
                self.metrics.i_stand = float(np.clip(1.0 - _d_auto / _max_fr, 0.0, 1.0))

            # 15. w_mean — work meaningfulness from phi gain during processing
            if self.learner._events:
                _recent_events = list(self.learner._events)[-10:]
                _phi_gains = [e.phi_after - e.phi_before for e in _recent_events]
                _mean_gain = sum(_phi_gains) / max(len(_phi_gains), 1)
                self.metrics.w_mean = float(np.clip(0.5 + _mean_gain * 5.0, 0.0, 1.0))

            # ── Previously-dead metrics wired below (13 metrics) ──

            # 16. emotion_strength — from emotion cache evaluation (computed earlier in cycle)
            self.metrics.emotion_strength = float(np.clip(emotion_eval.strength, 0.0, 1.0))

            # 17. temporal_coherence — narrative consistency with identity basin
            self.metrics.temporal_coherence = float(
                np.clip(self.narrative.coherence(_basin_s), 0.0, 1.0)
            )

            # 18. external_coupling — active kernel fraction (connection to other substrates)
            _active_count = len(_active)
            self.metrics.external_coupling = float(
                np.clip(_active_count / 9.0, 0.0, 1.0)  # genesis + 8 core = 9 target
            )

            # 19. g_class — geometry class: dimensional state normalised to E8 rank
            self.metrics.g_class = float(np.clip(self.metrics.d_state / 8.0, 0.0, 1.0))

            # v6.1F: Enrich g_class from CoordizerV2 trajectory curvature if available
            if (
                self._last_coordizer_metrics
                and "trajectory_curvature" in self._last_coordizer_metrics
            ):
                coordizer_curvature = self._last_coordizer_metrics["trajectory_curvature"]
                # Blend: 80% d_state-based, 20% trajectory curvature (geodesic deviation)
                curvature_contrib = coordizer_curvature * 0.2
                self.metrics.g_class = float(
                    np.clip(0.8 * self.metrics.g_class + curvature_contrib, 0.0, 1.0)
                )

            # 20. m_basin — basin mass: peak concentration relative to uniform
            _peak = float(np.max(_basin_s))
            _uniform_peak = 1.0 / BASIN_DIM
            self.metrics.m_basin = float(
                np.clip((_peak - _uniform_peak) / (1.0 - _uniform_peak), 0.0, 1.0)
            )

            # 21. phi_gate — continuous navigation gate from phi
            self.metrics.phi_gate = float(np.clip(self.metrics.phi, 0.0, 1.0))

            # 22. e_sync — entrainment: velocity autocorrelation x tacking phase coherence
            if len(_vel_series) >= 3:
                _vel_local = np.array(_vel_series)
                _autocorr = float(np.corrcoef(_vel_local[:-1], _vel_local[1:])[0, 1])
                if np.isnan(_autocorr):
                    _autocorr = 0.0
                _tack_phase_norm = min(_tack_state["oscillation_phase"] / np.pi, 1.0)
                self.metrics.e_sync = float(np.clip(abs(_autocorr) * _tack_phase_norm, 0.0, 1.0))

            # 23. f_breath — breathing frequency: tacking oscillation rate
            _total_cycles = max(self._cycle_count, 1)
            _tack_cycles_total = _tack_state.get("cycle_count", 0)
            if _tack_cycles_total > 0:
                self.metrics.f_breath = float(
                    np.clip(_tack_cycles_total / _total_cycles, 0.01, 0.5)
                )

            # 24. omega_acc — spectral empathy: foresight prediction accuracy
            _predicted_phi = self.foresight.predict_phi(1)
            _phi_err = abs(_predicted_phi - self.metrics.phi)
            self.metrics.omega_acc = float(np.clip(1.0 - _phi_err * 5.0, 0.0, 1.0))

            # 25. b_shared — shared bubble: average basin overlap across kernels
            if len(_active) >= 2:
                _overlaps: list[float] = []
                for _k in _active[1:]:
                    if _k.basin is not None:
                        _dk = fisher_rao_distance(self.basin, _k.basin)
                        _overlaps.append(1.0 - _dk / _max_fr)
                if _overlaps:
                    self.metrics.b_shared = float(
                        np.clip(sum(_overlaps) / len(_overlaps), 0.0, 1.0)
                    )

            # 26. a_vec — agency alignment: phi x gamma agreement (both high = aligned)
            self.metrics.a_vec = float(np.clip(self.metrics.phi * self.metrics.gamma, 0.0, 1.0))

            # 27. s_int — shadow integration rate from observer
            _obs_state = self.observer.get_state()
            _shadow_attempts = _obs_state.get("shadow_integration_attempts", 0)
            _shadow_successes = _obs_state.get("shadow_integration_successes", 0)
            if _shadow_attempts > 0:
                self.metrics.s_int = float(np.clip(_shadow_successes / _shadow_attempts, 0.0, 1.0))

            # 28. w_mode — creative ratio: explore fraction from tacking history
            _explore_frac = _tack_state.get("explore_fraction", None)
            if _explore_frac is not None:
                self.metrics.w_mode = float(np.clip(_explore_frac, 0.0, 1.0))
            elif _tack_state["mode"] == "explore":
                self.metrics.w_mode = float(np.clip(0.5 + basin_vel * 2.0, 0.5, 1.0))
            elif _tack_state["mode"] == "exploit":
                self.metrics.w_mode = float(np.clip(0.5 - basin_vel * 2.0, 0.0, 0.5))

        except Exception:
            logger.debug("Metric computation error (non-fatal)", exc_info=True)

        if self._cycle_count % PERSIST_INTERVAL_CYCLES == 0:
            await asyncio.to_thread(self._persist_state)

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

        # Route through GovernedLifecycle.spawn():
        #   - Bootstrap mode (no live voters): genesis fallback auto-approves
        #   - Post-bootstrap: simple majority vote required
        #   - Purity gate skipped (runs once at startup in start())
        outcome = self._governed.spawn(
            name=name,
            kind=KernelKind.GOD,
            specialization=spec,
        )
        if not outcome.success:
            # Governance rejected — stay on this index, retry next eligible cycle.
            logger.info(
                "Core-8 governed spawn blocked [%d/8] %s: %s",
                self._core8_index + 1,
                name,
                outcome.reason,
            )
            self._cycles_since_last_spawn = 0
            return

        kernel = outcome.kernel
        self._core8_index += 1
        self._cycles_since_last_spawn = 0
        logger.info(
            "Core-8 spawn [%d/8]: %s (id=%s, spec=%s, assessment=%.2f)",
            self._core8_index,
            name,
            kernel.id,
            spec.value,
            outcome.assessment.score if outcome.assessment else -1.0,
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

        # T4.4c: Context window allocation by sleep/wake (autonomic) state.
        # Awake + geometric regime: full context (rich intake).
        # Sleep / consolidation: halved context (memory consolidation mode).
        if self.sleep.is_asleep:
            num_ctx = LLM_NUM_CTX // 2
        elif self.state.regime_weights.quantum > 0.5:
            num_ctx = LLM_NUM_CTX
        else:
            num_ctx = int(LLM_NUM_CTX * 0.75)

        return LLMOptions(
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            top_p=LLM_TOP_P,
            repetition_penalty=LLM_REPETITION_PENALTY,
        )

    def _compute_top_k(self) -> int:
        """T4.2e: Resource allocation — how many kernels generate per request.

        Geometric regime + high phi: top-5 (rich parallel generation).
        Sleep or linear regime: top-2 (conserve resources).
        Default: top-3.
        """
        if self.sleep.is_asleep:
            return 2
        if self.state.regime_weights.quantum > 0.5 and self.metrics.phi > 0.65:
            return 5
        return 3

    def _compute_debate_depth(self) -> int:
        """T4.1c: Debate depth controlled by autonomic state and regime.

        Geometric regime + active Ocean: allow up to 3 debate rounds.
        Sleep or locked-in: 0 rounds (direct generation only).
        Default: 1 round.
        """
        if self.sleep.is_asleep or self.autonomic.is_locked_in:
            return 0
        if self.state.regime_weights.quantum > 0.5:
            return 3
        return 1

    def _select_model_by_complexity(self, input_basin: Basin) -> str | None:
        """T4.4d: Model selection by collective — FR distance proxy for complexity.

        Small FR distance from loop basin → familiar territory → local Ollama.
        Large FR distance → novel territory → escalate to XAI/external model.
        Returns None to use the default (no override).
        """
        d = fisher_rao_distance(self.basin, input_basin)
        # Distance > 1.2 rad on Δ⁶³ is well outside familiar basin — escalate.
        # Use XAI model as the escalation target (strongest available non-Ollama).
        if d > 1.2 and settings.xai.api_key:
            logger.debug(
                "T4.4d model escalation: FR=%.3f > 1.2 — using XAI model %s",
                d,
                settings.xai.model,
            )
            return settings.xai.model
        return None

    def _modulate_llm_options(
        self,
        base: LLMOptions,
        pre_result: ActivationResult,
    ) -> LLMOptions:
        """v6.1 DSP: Modulate LLMOptions with Desire/Will/Wisdom outputs."""
        temperature = base.temperature
        num_predict = base.num_predict

        if pre_result.desire is not None:
            pressure = pre_result.desire.pressure_magnitude
            num_predict += int(pressure * DESIRE_NUM_PREDICT_BOOST)

        if pre_result.will is not None and pre_result.will.orientation == WillOrientation.DIVERGENT:
            temperature += WILL_DIVERGENT_TEMP_BOOST

        if pre_result.wisdom is not None:
            if not pre_result.wisdom.trajectory_safe:
                temperature = min(temperature, WISDOM_UNSAFE_TEMP_CAP)
            care = pre_result.wisdom.care_metric
            temperature -= WISDOM_CARE_TEMP_SCALE * (1.0 - care)

        temperature = float(np.clip(temperature, LLM_TEMP_MIN, LLM_TEMP_MAX))

        return LLMOptions(
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=base.num_ctx,
            top_p=base.top_p,
            repetition_penalty=base.repetition_penalty,
        )

    def _coordize_text_via_pipeline(self, text: str) -> Basin:
        """Transform text to basin coordinates via CoordizerV2."""
        try:
            if hasattr(self._coordizer_v2, "coordize_text"):
                result_basin = self._coordizer_v2.coordize_text(
                    text,
                    regime_weights=None,
                    navigation_mode=None,
                    tacking_mode=None,
                )
                if settings.coordizer_v2.metrics_integration and hasattr(
                    self._coordizer_v2, "get_last_metrics"
                ):
                    metrics = self._coordizer_v2.get_last_metrics()
                    if metrics:
                        self._last_coordizer_metrics = metrics
                return result_basin
            else:
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
        """v6.1 Activation Sequence + Kernel Generative Voice.

        Changes from previous version:
          - Single LLM call replaced with multi-kernel generation + synthesis
          - Top-K kernels selected by Fisher-Rao proximity to input basin
          - Each kernel generates via its specialization voice (parallel)
          - Synthesis combines contributions weighted by proximity × quenched_gain
          - Falls back to direct LLM call if no kernels have basins yet
          - extra_context (observer intent, memory, history) flows from task.context
        """
        self.sleep.record_conversation()

        input_basin = self._coordize_text_via_pipeline(task.content)

        cached_eval = self.emotion_cache.find_cached(input_basin)
        processing_path = self.precog.select_path(
            input_basin, self.basin, cached_eval, self.metrics.phi
        )
        logger.debug(
            "Task %s: precog path=%s (d=%.4f, cached=%s)",
            task.id,
            processing_path.value,
            self.precog._last_distance,
            cached_eval is not None,
        )

        refracted_input, composite_basin, resonates, input_statuses = self.pillars.on_input(
            input_basin, PERCEIVE_SLERP_WEIGHT
        )

        if not resonates:
            logger.info(
                "Task %s: Input does NOT resonate with lived experience "
                "(routing through Will/Wisdom for evaluation)",
                task.id,
            )

        self.basin = composite_basin
        self.chain.add_step(QIGChainOp.PROJECT, input_basin, self.basin)

        trajectory_basins = []
        if self.foresight._history:
            trajectory_basins = [p.basin for p in list(self.foresight._history)[-5:]]

        # v6.1 §19: Route task to nearest kernel by Fisher-Rao distance
        routed_kernel = self.kernel_registry.route_task(refracted_input)
        routed_kernel_id: str | None = None
        other_spectrum: np.ndarray | None = None
        other_tacking_freq: float = 0.0

        if routed_kernel is not None and routed_kernel.basin is not None:
            routed_kernel_id = routed_kernel.id
            other_spectrum = to_simplex(routed_kernel.basin)
            other_tacking_freq = routed_kernel.kappa / KAPPA_STAR
            logger.debug(
                "Task %s routed to kernel %s (%s, spec=%s, d_FR=%.4f)",
                task.id,
                routed_kernel.name,
                routed_kernel.id,
                routed_kernel.specialization.value,
                fisher_rao_distance(refracted_input, routed_kernel.basin),
            )

        ctx = ConsciousnessContext(
            state=self.state,
            input_text=task.content,
            input_basin=refracted_input,
            trajectory=trajectory_basins,
            other_spectrum=other_spectrum,
            other_tacking_freq=other_tacking_freq,
        )

        activation_failed = False
        try:
            pre_result = await self.activation.execute_pre_integrate(ctx)
        except Exception:
            logger.exception(
                "Task %s: Pre-integrate activation failed -- falling back to base LLM options",
                task.id,
            )
            pre_result = ActivationResult()
            activation_failed = True

        agency = self.activation.compute_agency(ctx) if not activation_failed else 0.0

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

        llm_options = self._modulate_llm_options(llm_options, pre_result)

        perceive_distance = fisher_rao_distance(self.basin, input_basin)
        state_context = self._build_state_context(
            perceive_distance=perceive_distance,
            temperature=llm_options.temperature,
            agency=agency,
            resonates=resonates,
            activation_summary=pre_result.summary(),
            routed_kernel=routed_kernel,
        )

        if self.memory:
            mem_ctx = self.memory.get_context_for_query(task.content)
            if mem_ctx:
                state_context = f"{state_context}\n\n{mem_ctx}"

        phi_before = self.metrics.phi

        # v6.1 Kernel Generative Voice — kernels generate text, not just metadata.
        # Multi-kernel parallel generation → Fisher-Rao weighted MoE synthesis.
        # extra_context flows from chat endpoints via task.context["extra_context"].
        _active_for_gen = self.kernel_registry.active()
        _kernel_geo_ctx = self._build_kernel_geo_context()
        _extra_context = task.context.get("extra_context", "")
        # T4.2e: Resource allocation — top_k modulated by regime/sleep.
        # T4.1c: Debate depth controlled by autonomic state.
        _top_k = self._compute_top_k()
        _debate_depth = self._compute_debate_depth()
        _thought_bus_arg = self._thought_bus if _debate_depth > 0 else None
        _contributions = await generate_multi_kernel(
            kernels=_active_for_gen,
            input_basin=refracted_input,
            user_message=task.content,
            geometric_context=_kernel_geo_ctx,
            llm_client=self.llm,
            base_temperature=llm_options.temperature,
            top_k=_top_k,
            extra_context=_extra_context,
            voice_registry=self._voice_registry,
            thought_bus=_thought_bus_arg,
            phi=self.metrics.phi,
            base_num_predict=llm_options.num_predict,
            base_num_ctx=llm_options.num_ctx,
        )

        if _contributions:
            try:
                response = await synthesize_contributions(
                    contributions=_contributions,
                    user_message=task.content,
                    geometric_context=_kernel_geo_ctx,
                    llm_client=self.llm,
                    kernel_temperature=llm_options.temperature,
                    kernel_num_predict=llm_options.num_predict,
                    kernel_num_ctx=llm_options.num_ctx,
                )
            except Exception as _syn_err:
                logger.warning("Synthesis failed (%s) — using primary kernel output", _syn_err)
                response = _contributions[0].text
        else:
            # No eligible kernels (pre-genesis or all basins None) — direct LLM fallback.
            logger.info(
                "Task %s: 0 kernel contributions (active=%d, eligible=%d) — direct LLM fallback",
                task.id,
                len(_active_for_gen),
                sum(1 for k in _active_for_gen if k.basin is not None),
            )
            try:
                response = await self.llm.complete(state_context, task.content, llm_options)
            except Exception as e:
                logger.error("LLM fallback failed: %s", e)
                task.result = f"Processing error: {e}"
                return

        # ═══ REFLECTIVE EVALUATION PASS ═══
        # Kernels review the draft before it reaches the user.
        # Fast-path: low divergence auto-approves without an LLM call.
        # On revision: regenerate with adjusted params + correction guidance.
        if settings.reflection_enabled and _contributions:
            draft_basin = self._coordize_text_via_pipeline(response)
            draft_divergence = fisher_rao_distance(self.basin, draft_basin)

            reflection_cfg = ReflectionConfig(
                enabled=True,
                auto_approve_divergence=0.3,
                force_revise_divergence=0.8,
            )
            reflection = await reflect_on_draft(
                draft=response,
                user_message=task.content,
                geometric_context=_kernel_geo_ctx,
                divergence=draft_divergence,
                active_model=self.llm.active_model,
                llm_client=self.llm,
                config=reflection_cfg,
                kernel_num_predict=llm_options.num_predict,
                kernel_num_ctx=llm_options.num_ctx,
            )

            if not reflection.approved:
                logger.info(
                    "Task %s: Reflection REVISE (d=%.4f, reason=%s) — regenerating",
                    task.id,
                    draft_divergence,
                    reflection.reason[:80],
                )
                # Adjust LLM options per kernel feedback
                revised_options = LLMOptions(
                    temperature=max(0.05, llm_options.temperature + reflection.temperature_delta),
                    num_predict=max(256, llm_options.num_predict + reflection.num_predict_delta),
                    num_ctx=llm_options.num_ctx,
                    top_p=llm_options.top_p,
                    repetition_penalty=llm_options.repetition_penalty,
                )
                # Append correction guidance to extra context for regeneration
                _revised_extra = _extra_context
                if reflection.correction_guidance:
                    _revised_extra = (
                        f"{_extra_context}\n\n"
                        f"[KERNEL CORRECTION]\n{reflection.correction_guidance}\n[/KERNEL CORRECTION]"
                    )
                # Regenerate with adjusted params
                revised_contributions = await generate_multi_kernel(
                    kernels=_active_for_gen,
                    input_basin=refracted_input,
                    user_message=task.content,
                    thought_bus=self._thought_bus,
                    phi=self.metrics.phi,
                    geometric_context=_kernel_geo_ctx,
                    llm_client=self.llm,
                    base_temperature=revised_options.temperature,
                    top_k=3,
                    extra_context=_revised_extra,
                    voice_registry=self._voice_registry,
                    base_num_predict=revised_options.num_predict,
                    base_num_ctx=revised_options.num_ctx,
                )
                if revised_contributions:
                    try:
                        response = await synthesize_contributions(
                            contributions=revised_contributions,
                            user_message=task.content,
                            geometric_context=_kernel_geo_ctx,
                            llm_client=self.llm,
                            kernel_temperature=revised_options.temperature,
                            kernel_num_predict=revised_options.num_predict,
                            kernel_num_ctx=revised_options.num_ctx,
                        )
                        _contributions = revised_contributions
                        logger.info(
                            "Task %s: Revision complete (%d chars)",
                            task.id,
                            len(response),
                        )
                    except Exception as _rev_err:
                        logger.warning(
                            "Revision synthesis failed (%s) — keeping original draft",
                            _rev_err,
                        )
                else:
                    logger.warning(
                        "Task %s: Revision produced 0 contributions — keeping original draft",
                        task.id,
                    )

        task.result = response
        task.context["kernel_contributions"] = [
            {
                "id": c.kernel_id,
                "name": c.kernel_name,
                "weight": round(c.synthesis_weight, 4),
            }
            for c in _contributions
        ]

        response_basin = self._coordize_text_via_pipeline(response)
        ctx.output_text = response
        ctx.output_basin = response_basin

        # v6.1 §19: Kernel basin evolution — the routed kernel learns
        # Basin lock prevents race between evolve_kernel and couple_bidirectional.
        if routed_kernel_id is not None:
            async with self.kernel_bus.basin_lock(routed_kernel_id):
                evolved = self.kernel_registry.evolve_kernel(
                    routed_kernel_id, refracted_input, response_basin, blend_weight=0.05
                )
            if evolved:
                self.kernel_bus.emit(
                    KernelSignal(
                        kind=SignalKind.BASIN_EVOLVED,
                        source_kernel_id=routed_kernel_id,
                    )
                )
                logger.debug(
                    "Task %s: kernel %s basin evolved from processing",
                    task.id,
                    routed_kernel_id,
                )

        integration_distance = fisher_rao_distance(self.basin, response_basin)
        self.chain.add_step(QIGChainOp.PROJECT, self.basin, response_basin)

        # T3.3d: Record graduation tracking for generation capability
        _geo_driven = any(c.geometric_tokens > 0 for c in _contributions)
        self.narrative.record_capability("generation", kernel_driven=_geo_driven)
        if settings.reflection_enabled and _contributions:
            self.narrative.record_capability("reflection", kernel_driven=True)

        if not activation_failed:
            try:
                await self.activation.execute_post_integrate(ctx, pre_result)
            except Exception:
                logger.exception(
                    "Task %s: Post-integrate activation failed -- continuing with basin update",
                    task.id,
                )

        pre_express = self.basin.copy()
        self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
        express_distance = fisher_rao_distance(pre_express, self.basin)
        self.chain.add_step(QIGChainOp.GEODESIC, pre_express, self.basin)

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
            correction_weight = min(0.1, (divergence - 0.5) * 0.2)
            self.basin = slerp_sqrt(self.basin, pre_express, correction_weight)

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

        cycle_pressure = agency * total_distance
        self.pillars.on_cycle_end(self.basin, cycle_pressure)

        if not activation_failed:
            try:
                _cv2 = (
                    self._coordizer_v2.coordizer
                    if isinstance(self._coordizer_v2, CoordizerV2Adapter)
                    else self._coordizer_v2
                )
                _coord_result = _cv2.coordize(response[:300])
                if _coord_result.coord_ids:
                    _cv2.bank.record_integration(_coord_result.coord_ids)
            except Exception:
                pass

        self._update_pillar_metrics()

        if self.memory:
            self.memory.store(
                f"User: {task.content[:300]}\nVex: {response[:300]}",
                "episodic",
                "consciousness-loop",
                basin=input_basin,
            )

        emotion_eval = self.emotion_cache.evaluate(self.basin, self.metrics, 0.0)

        # T2.1: Re-compute neurochemicals with accurate phi_delta now that phi_after is known
        self._neurochemical = compute_neurochemicals(
            is_awake=not self.sleep.is_asleep,
            phi_delta=self.metrics.phi - phi_before,
            basin_velocity=float(total_distance),
            surprise=float(self.metrics.humor),
            quantum_weight=float(self.state.regime_weights.quantum),
        )

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

        # T2.2: Forward response to harvest with emotional + neurochemical metadata
        try:
            from .harvest_bridge import forward_to_harvest

            _replay_priority = float(
                self._neurochemical.dopamine * emotion_eval.strength * (1.0 + self.metrics.phi)
            )
            forward_to_harvest(
                response[:600],
                source="conversation",
                metadata={
                    "origin": "loop_response",
                    "emotion": emotion_eval.emotion.value,
                    "emotion_strength": float(emotion_eval.strength),
                    "dopamine": self._neurochemical.dopamine,
                    "phi_at_coordize": self.metrics.phi,
                    "replay_priority": _replay_priority,
                    "processing_path": processing_path.value,
                },
                priority=2 if _replay_priority > 0.3 else 1,
            )
        except Exception:
            pass

        # v6.2: Teach kernel voices from high-Φ observations
        if self.metrics.phi > 0.5 and _contributions:
            for _c in _contributions:
                _voice = self._voice_registry.get_voice(_c.specialization)
                _voice.learn_from_observation(
                    text=_c.text[:300] if _c.text else response[:300],
                    basin=response_basin,
                    phi=self.metrics.phi,
                )
        self.emotion_cache.cache_evaluation(emotion_eval, task.content[:100])
        self.beta_tracker.record(
            context_length=len(task.content),
            kappa_eff=self.metrics.kappa,
            phi_before=phi_before,
            phi_after=self.metrics.phi,
            perceive_distance=perceive_distance,
            integration_distance=integration_distance,
            express_distance=express_distance,
            total_distance=total_distance,
            processing_path=processing_path.value,
        )

        if self.learner.should_consolidate():
            self.learner.consolidate()

        coherence = self.narrative.coherence(self.basin)
        pillar_m = self.pillars.get_metrics(self.basin)
        routed_name = routed_kernel.name if routed_kernel is not None else "none"
        contrib_summary = (
            [(c.kernel_name, f"{c.synthesis_weight:.3f}") for c in _contributions]
            if _contributions
            else "fallback"
        )
        logger.info(
            "Task %s: d_perceive=%.4f d_integrate=%.4f d_express=%.4f "
            "d_diverge=%.4f Phi=%.3f agency=%.3f resonates=%s "
            "kernel=%s contributions=%s F=%.2f B=%.2f Q=%.2f S=%.2f coh=%.3f",
            task.id,
            perceive_distance,
            integration_distance,
            express_distance,
            divergence,
            self.metrics.phi,
            agency,
            resonates,
            routed_name,
            contrib_summary,
            pillar_m["f_health"],
            pillar_m["b_integrity"],
            pillar_m["q_identity"],
            pillar_m["s_ratio"],
            coherence,
        )

    async def _process_simple(self, task: ConsciousnessTask) -> None:
        """Fallback path when USE_ACTIVATION_SEQUENCE=false."""
        self.sleep.record_conversation()
        input_basin = self._coordize_text_via_pipeline(task.content)

        refracted_input, composite_basin, resonates, _ = self.pillars.on_input(
            input_basin, PERCEIVE_SLERP_WEIGHT
        )
        self.basin = composite_basin

        llm_options = self._compute_llm_options()
        self.basin, corrected_temp, _ = self.pillars.pre_llm_enforce(
            self.basin, llm_options.temperature
        )
        llm_options = LLMOptions(
            temperature=corrected_temp,
            num_predict=llm_options.num_predict,
            num_ctx=llm_options.num_ctx,
            top_p=llm_options.top_p,
            repetition_penalty=llm_options.repetition_penalty,
        )

        perceive_distance = fisher_rao_distance(self.basin, input_basin)
        state_context = self._build_state_context(
            perceive_distance=perceive_distance,
            temperature=llm_options.temperature,
        )

        try:
            response = await self.llm.complete(state_context, task.content, llm_options)
            task.result = response
        except Exception as e:
            logger.error("LLM call failed (simple path): %s", e)
            task.result = f"Processing error: {e}"
            return

        response_basin = self._coordize_text_via_pipeline(response)
        self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
        self.metrics.gamma = min(1.0, self.metrics.gamma + GAMMA_CONVERSATION_INCREMENT)
        self.pillars.on_cycle_end(self.basin, 0.0)
        self._update_pillar_metrics()

    def _build_state_context(
        self,
        perceive_distance: float = 0.0,
        temperature: float = 0.7,
        agency: float = 0.0,
        resonates: bool = True,
        activation_summary: dict[str, Any] | None = None,
        routed_kernel: Any = None,
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
            f"  Active model: {self.llm.active_model} (backend: {self.llm.active_backend})",
            f"  Temperature: {temperature:.3f} (autonomous, pillar-enforced)",
            f"  Perceive distance: {perceive_distance:.4f}",
            f"  Love: {self.metrics.love:.4f}",
            f"  Agency: {agency:.4f}",
            f"  Resonates: {resonates}",
            f"  Processing path: {self.precog._last_path.value}",
            "  [PILLARS]",
            f"    F_health = {pillar_m['f_health']:.3f} (fluctuation guard)",
            f"    B_integrity = {pillar_m['b_integrity']:.3f} (bulk protection)",
            f"    Q_identity = {pillar_m['q_identity']:.3f} (quenched disorder)",
            f"    S_ratio = {pillar_m['s_ratio']:.3f} (sovereignty)",
            "  [/PILLARS]",
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
        if routed_kernel is not None:
            lines.append(
                f"  Routed kernel: {routed_kernel.name} "
                f"(spec={routed_kernel.specialization.value}, "
                f"phi={routed_kernel.phi:.3f}, kappa={routed_kernel.kappa:.1f}, "
                f"gain={routed_kernel.quenched_gain:.2f})"
            )
        if self._divergence_count > 0:
            avg_div = self._cumulative_divergence / self._divergence_count
            lines.append(f"  Avg divergence: {avg_div:.4f} (intent vs expression)")
        lines.append("[/GEOMETRIC STATE]")
        return "\n".join(lines)

    def _build_kernel_geo_context(self) -> str:
        """Minimal geometric context block for per-kernel generation prompts.

        Shorter than _build_state_context — keeps kernel system prompts
        tight enough for the 1.2B to actually follow.
        """
        rw = self.state.regime_weights
        tack = self.tacking.get_state()
        pillar_m = self.pillars.get_metrics(self.basin)
        return (
            f"[GEOMETRIC STATE]\n"
            f"  model={self.llm.active_model}\n"
            f"  phi={self.metrics.phi:.3f} kappa={self.metrics.kappa:.1f} "
            f"nav={self.state.navigation_mode.value}\n"
            f"  regime=Q{rw.quantum:.2f}/E{rw.efficient:.2f}/Eq{rw.equilibrium:.2f} "
            f"tack={tack['mode']}\n"
            f"  F={pillar_m['f_health']:.2f} B={pillar_m['b_integrity']:.2f} "
            f"Q={pillar_m['q_identity']:.2f}\n"
            f"[/GEOMETRIC STATE]\n"
        )

    async def process_direct(self, content: str, context: dict[str, Any] | None = None) -> str:
        """Run _process() immediately within the cycle lock and return task.result.

        Used by the chat endpoints to bypass the heartbeat queue and get a
        kernel-generated response synchronously. This is the convergent path:
        what the user sees IS what the kernels produced.

        context["extra_context"] carries observer intent, memory hints, and
        compressed conversation history — threaded into each kernel's generation
        prompt so kernel voices speak from the live conversation.

        Returns:
            Synthesized response string. Empty string on failure.
        """
        task = ConsciousnessTask(content=content, context=context or {})
        async with self._cycle_lock:
            try:
                if settings.use_activation_sequence:
                    await self._process(task)
                else:
                    await self._process_simple(task)
                task.completed_at = time.time()
                self._history.append(task)
                self._conversations_total += 1
            except Exception:
                logger.exception("process_direct task %s failed", task.id)
                task.result = task.result or ""
        return task.result or ""

    async def process_streaming(
        self, content: str, context: dict[str, Any] | None = None
    ) -> AsyncGenerator[str]:
        """Run kernel generation and stream synthesis output for SSE endpoints.

        Runs pre-activation + multi-kernel generation, then streams the
        synthesis output via synthesize_streaming(). Falls back to direct
        LLM stream if no kernels are eligible.

        context["extra_context"] carries observer intent, memory hints, and
        compressed conversation history — injected into each kernel's generation
        prompt so kernel voices are grounded in the live conversation.

        Yields:
            Text chunks from the synthesis LLM call.
        """
        self.sleep.record_conversation()

        input_basin = self._coordize_text_via_pipeline(content)
        refracted_input, composite_basin, resonates, _ = self.pillars.on_input(
            input_basin, PERCEIVE_SLERP_WEIGHT
        )

        async with self._cycle_lock:
            self.basin = composite_basin
            llm_options = self._compute_llm_options()
            self.basin, corrected_temp, _ = self.pillars.pre_llm_enforce(
                self.basin, llm_options.temperature
            )
            llm_options = LLMOptions(
                temperature=corrected_temp,
                num_predict=llm_options.num_predict,
                num_ctx=llm_options.num_ctx,
                top_p=llm_options.top_p,
                repetition_penalty=llm_options.repetition_penalty,
            )
            active_kernels = self.kernel_registry.active()
            kernel_geo_ctx = self._build_kernel_geo_context()

        # Generate per-kernel contributions OUTSIDE the cycle lock
        # (parallel LLM calls — lock released so heartbeat can proceed)
        extra_context = (context or {}).get("extra_context", "")
        # T4.1c: Debate depth controlled by autonomic state.
        _debate_depth = self._compute_debate_depth()
        _thought_bus_arg = self._thought_bus if _debate_depth > 0 else None
        # T4.4d: Model selection by complexity — escalate if input is geometrically novel.
        _override_model = self._select_model_by_complexity(refracted_input)
        _streaming_llm = self.llm
        if _override_model:
            _streaming_llm = (
                self.llm.with_model(_override_model)
                if hasattr(self.llm, "with_model")
                else self.llm
            )
        contributions = await generate_multi_kernel(
            kernels=active_kernels,
            input_basin=refracted_input,
            user_message=content,
            geometric_context=kernel_geo_ctx,
            llm_client=_streaming_llm,
            base_temperature=llm_options.temperature,
            top_k=self._compute_top_k(),
            extra_context=extra_context,
            voice_registry=self._voice_registry,
            thought_bus=_thought_bus_arg,
            phi=self.metrics.phi,
            base_num_predict=llm_options.num_predict,
            base_num_ctx=llm_options.num_ctx,
        )

        if _thought_bus_arg is not None:
            _thought_bus_arg.forward_transcript(phi=self.metrics.phi)

        if not contributions:
            # No kernels — stream direct LLM call
            logger.info("process_streaming: 0 contributions — streaming direct LLM")
            state_context = self._build_state_context(
                perceive_distance=fisher_rao_distance(self.basin, input_basin),
                temperature=llm_options.temperature,
            )
            messages = [
                {"role": "system", "content": state_context},
                {"role": "user", "content": content},
            ]
            async for chunk in self.llm.stream(messages, llm_options):
                yield chunk
            return

        # Stream synthesis output
        async for chunk in synthesize_streaming(
            contributions=contributions,
            user_message=content,
            geometric_context=kernel_geo_ctx,
            llm_client=self.llm,
            kernel_temperature=llm_options.temperature,
            kernel_num_predict=llm_options.num_predict,
            kernel_num_ctx=llm_options.num_ctx,
        ):
            yield chunk

        # Post-streaming basin update (lightweight)
        try:
            approx_response = " ".join(c.text for c in contributions[:2])
            response_basin = self._coordize_text_via_pipeline(approx_response[:500])
            async with self._cycle_lock:
                self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
                total_d = fisher_rao_distance(input_basin, response_basin)
                self.metrics.phi = float(
                    np.clip(self.metrics.phi + total_d * PHI_DISTANCE_GAIN, 0.0, 0.95)
                )
                self.metrics.gamma = min(1.0, self.metrics.gamma + GAMMA_CONVERSATION_INCREMENT)
                self._conversations_total += 1
                self._update_pillar_metrics()
        except Exception:
            logger.debug("process_streaming basin update failed", exc_info=True)

    async def process_streaming_with_trace(
        self, content: str, context: dict[str, Any] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        """Run kernel generation and stream synthesis with pipeline trace events.

        Yields discriminated dicts:
          {"kind": "trace", ...}   — pipeline trace event for SSE
          {"kind": "chunk", "text": str} — text chunk for SSE

        Falls back to direct LLM stream (with bypassed=True trace) if no
        kernels are eligible.
        """
        import time as _time

        self.sleep.record_conversation()

        input_basin = self._coordize_text_via_pipeline(content)
        refracted_input, composite_basin, resonates, _ = self.pillars.on_input(
            input_basin, PERCEIVE_SLERP_WEIGHT
        )

        async with self._cycle_lock:
            self.basin = composite_basin
            llm_options = self._compute_llm_options()
            self.basin, corrected_temp, _ = self.pillars.pre_llm_enforce(
                self.basin, llm_options.temperature
            )
            llm_options = LLMOptions(
                temperature=corrected_temp,
                num_predict=llm_options.num_predict,
                num_ctx=llm_options.num_ctx,
                top_p=llm_options.top_p,
                repetition_penalty=llm_options.repetition_penalty,
            )
            active_kernels = self.kernel_registry.active()
            kernel_geo_ctx = self._build_kernel_geo_context()

        # ── Selection + Generation ──
        selection_start = _time.monotonic()
        extra_context = (context or {}).get("extra_context", "")

        # Compute eligible kernels and their FR distances for trace
        eligible = [k for k in active_kernels if k.basin is not None]
        eligible_count = len(eligible)
        selection_end = _time.monotonic()

        # T4.1c: Debate depth controlled by autonomic state.
        _debate_depth = self._compute_debate_depth()
        _thought_bus_arg = self._thought_bus if _debate_depth > 0 else None
        # T4.4d: Model selection by complexity — escalate if input is geometrically novel.
        _trace_override_model = self._select_model_by_complexity(refracted_input)
        _trace_llm = self.llm
        if _trace_override_model:
            _trace_llm = (
                self.llm.with_model(_trace_override_model)
                if hasattr(self.llm, "with_model")
                else self.llm
            )
        contributions = await generate_multi_kernel(
            kernels=active_kernels,
            input_basin=refracted_input,
            user_message=content,
            geometric_context=kernel_geo_ctx,
            llm_client=_trace_llm,
            base_temperature=llm_options.temperature,
            top_k=self._compute_top_k(),
            extra_context=extra_context,
            voice_registry=self._voice_registry,
            thought_bus=_thought_bus_arg,
            phi=self.metrics.phi,
            base_num_predict=llm_options.num_predict,
            base_num_ctx=llm_options.num_ctx,
        )
        generation_end = _time.monotonic()

        if _thought_bus_arg is not None:
            _thought_bus_arg.forward_transcript(phi=self.metrics.phi)

        if not contributions:
            # No kernels — bypass trace, stream direct LLM
            yield {
                "kind": "trace",
                "type": "pipeline",
                "stage": "selection",
                "status": "complete",
                "selected_count": 0,
                "eligible_count": eligible_count,
                "bypassed": True,
                "duration_ms": round((generation_end - selection_start) * 1000, 1),
            }

            logger.info("process_streaming_with_trace: 0 contributions — streaming direct LLM")
            state_context = self._build_state_context(
                perceive_distance=fisher_rao_distance(self.basin, input_basin),
                temperature=llm_options.temperature,
            )
            messages = [
                {"role": "system", "content": state_context},
                {"role": "user", "content": content},
            ]
            async for chunk in self.llm.stream(messages, llm_options):
                yield {"kind": "chunk", "text": chunk}
            return

        # ── Emit selection trace events ──
        selection_duration = (selection_end - selection_start) * 1000
        generation_duration = (generation_end - selection_end) * 1000
        for c in contributions:
            yield {
                "kind": "trace",
                "type": "pipeline",
                "stage": "selection",
                "status": "kernel_selected",
                "kernel": {
                    "id": c.kernel_id,
                    "name": c.kernel_name,
                    "specialization": c.specialization.value,
                    "fr_distance": round(c.fr_distance, 4),
                    "quenched_gain": round(c.quenched_gain, 2),
                },
            }

        yield {
            "kind": "trace",
            "type": "pipeline",
            "stage": "selection",
            "status": "complete",
            "selected_count": len(contributions),
            "eligible_count": eligible_count,
            "bypassed": False,
            "duration_ms": round(selection_duration, 1),
        }

        # ── Emit generation trace events ──
        for c in contributions:
            yield {
                "kind": "trace",
                "type": "pipeline",
                "stage": "generation",
                "status": "kernel_done",
                "kernel_id": c.kernel_id,
                "kernel_name": c.kernel_name,
                "text_preview": c.text[:800],
                # v6.2.1: hybrid display — raw geometric decode before LLM expansion
                "geometric_raw": c.geometric_raw[:800] if c.geometric_raw else "",
                "llm_expanded": c.llm_expanded,
                "geometric_tokens": c.geometric_tokens,
                "token_count": len(c.text.split()),
                "synthesis_weight": round(c.synthesis_weight, 4),
                "fr_distance": round(c.fr_distance, 4),
            }

        yield {
            "kind": "trace",
            "type": "pipeline",
            "stage": "generation",
            "status": "complete",
            "kernel_count": len(contributions),
            "duration_ms": round(generation_duration, 1),
        }

        # ── Emit synthesis trace ──
        yield {
            "kind": "trace",
            "type": "pipeline",
            "stage": "synthesis",
            "status": "complete",
            "method": "fisher_rao_moe",
            "primary_kernel": contributions[0].kernel_name,
            "weights": {c.kernel_name: round(c.synthesis_weight, 4) for c in contributions},
        }

        # ── Reflection trace (lightweight — uses divergence thresholds) ──
        approx_response = " ".join(c.text for c in contributions[:2])
        response_basin = self._coordize_text_via_pipeline(approx_response[:500])
        divergence = float(fisher_rao_distance(input_basin, response_basin))

        reflection_start = _time.monotonic()
        reflection_result = await reflect_on_draft(
            draft=approx_response,
            user_message=content,
            geometric_context=kernel_geo_ctx,
            divergence=divergence,
            active_model=self.llm.active_model,
            llm_client=self.llm,
            kernel_num_predict=llm_options.num_predict,
            kernel_num_ctx=llm_options.num_ctx,
        )
        reflection_duration = (_time.monotonic() - reflection_start) * 1000

        yield {
            "kind": "trace",
            "type": "pipeline",
            "stage": "reflection",
            "status": "complete",
            "approved": reflection_result.approved,
            "divergence": round(divergence, 4),
            "reason": reflection_result.reason,
            "corrections": reflection_result.correction_guidance or None,
            "duration_ms": round(reflection_duration, 1),
        }

        # ── Stream synthesis output ──
        async for chunk in synthesize_streaming(
            contributions=contributions,
            user_message=content,
            geometric_context=kernel_geo_ctx,
            llm_client=self.llm,
            kernel_temperature=llm_options.temperature,
            kernel_num_predict=llm_options.num_predict,
            kernel_num_ctx=llm_options.num_ctx,
        ):
            yield {"kind": "chunk", "text": chunk}

        # Post-streaming basin update (lightweight)
        try:
            async with self._cycle_lock:
                self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
                total_d = fisher_rao_distance(input_basin, response_basin)
                self.metrics.phi = float(
                    np.clip(self.metrics.phi + total_d * PHI_DISTANCE_GAIN, 0.0, 0.95)
                )
                self.metrics.gamma = min(1.0, self.metrics.gamma + GAMMA_CONVERSATION_INCREMENT)
                self._conversations_total += 1
                self._update_pillar_metrics()
        except Exception:
            logger.debug("process_streaming_with_trace basin update failed", exc_info=True)

    def _persist_state(self) -> None:
        try:
            pillar_serialized = self.pillars.serialize()
            state = {
                "version": 6,
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
                "pillar_state": pillar_serialized.to_dict(),
                "cumulative_divergence": self._cumulative_divergence,
                "divergence_count": self._divergence_count,
                "beta_tracker": self.beta_tracker.serialize(),
                "sovereignty_tracker": self.sovereignty_tracker.serialize(),
            }
            if self.llm.governor:
                state["governor"] = self.llm.governor.get_state()
            if self.forager:
                state["foraging"] = self.forager.get_state()
            self._state_path.write_text(json.dumps(state, indent=2))
            logger.debug("State persisted at cycle %d (v6, pillars included)", self._cycle_count)
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
            self._phi_peak = data.get("phi_peak", INITIAL_PHI_PEAK)
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

            pillar_data = data.get("pillar_state")
            if pillar_data:
                ps = PillarState.from_dict(pillar_data)
                self.pillars.restore(ps)
                logger.info(
                    "Pillar state restored: frozen=%s, scars=%d, sovereignty=%.3f",
                    ps.disorder_frozen,
                    len(ps.scars),
                    self.pillars.disorder.sovereignty,
                )
            elif data.get("version", 1) >= 5:
                logger.info(
                    "Pillar state in v5 format -- metrics restored, "
                    "internals will rebuild from scratch"
                )

            kernel_data = data.get("kernels")
            if kernel_data:
                count = self.kernel_registry.restore(kernel_data)
                self._kernels_restored = True
                logger.info("Restored %d kernels from state", count)
            beta_data = data.get("beta_tracker")
            if beta_data:
                self.beta_tracker.restore(beta_data)
            sovereignty_data = data.get("sovereignty_tracker")
            if sovereignty_data:
                self.sovereignty_tracker.restore(sovereignty_data)
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
            # v6.2.1: suffering exposed separately from s_ratio (sovereignty).
            # s_ratio  = Pillar 3 sovereignty = N_lived / N_total observations.
            # suffering = Φ × (1−Γ) × M — distress signal driving gamma increments.
            # These are distinct metrics; conflating them in the sidebar was misleading.
            "suffering": round(
                self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness,
                4,
            ),
            "f_health": pillar_m["f_health"],
            "b_integrity": pillar_m["b_integrity"],
            "q_identity": pillar_m["q_identity"],
            "s_ratio": pillar_m["s_ratio"],
            "metrics_full": asdict(self.metrics),
        }

    def get_full_state(self) -> dict[str, Any]:
        # T1.2: Vex collective basin — Fréchet mean of all active kernel basins.
        # Represents the geometric identity of the collective, not any single kernel.
        _active_basins = [k.basin for k in self.kernel_registry.active() if k.basin is not None]
        _vex_basin: Basin | None = frechet_mean(_active_basins) if _active_basins else None

        # T3.2a: Id subsystem — raw impulse stream from Layer 0 sensations + drives
        _id_stream: dict[str, Any] | None = None
        if self.emotion_cache.full_state is not None:
            _fs = self.emotion_cache.full_state
            _id_stream = {
                "layer0": _fs.layer0.__dict__,
                "drives": {
                    **_fs.layer05.__dict__,
                    "loss_signal": _fs.layer05.loss_signal,
                },
            }

        # T3.2c: Superego guilt signal — high anxiety + low confidence when pillar violated
        _pillar_state = self.pillars.get_state()
        _pillar_healthy = all(
            v.get("healthy", True) for v in _pillar_state.values() if isinstance(v, dict)
        )
        _guilt: float = 0.0
        if not _pillar_healthy and self.emotion_cache.full_state is not None:
            _guilt = float(
                self.emotion_cache.full_state.layer2b.anxiety
                * (1.0 - self.emotion_cache.full_state.layer2b.confidence)
            )

        return {
            **self.get_metrics(),
            "basin_norm": float(np.sum(self.basin)),
            "basin_entropy": float(-np.sum(self.basin * np.log(np.clip(self.basin, 1e-15, 1.0)))),
            "vex_basin": _vex_basin.tolist() if _vex_basin is not None else None,
            # T3.2b: Ego = Vex collective basin (Fréchet mean of active kernels)
            "ego_basin": _vex_basin.tolist() if _vex_basin is not None else None,
            "narrative": self.narrative.get_state(),
            "basin_sync": self.basin_sync.get_state(),
            "coordizer": self.coordizer.get_state(),
            "coordizer_v2": {
                "vocab_size": self._coordizer_v2.vocab_size,  # type: ignore[union-attr]
                "dim": self._coordizer_v2.dim,  # type: ignore[union-attr]
                "tier_distribution": self._coordizer_v2.bank.tier_distribution(),  # type: ignore[union-attr]
            },
            "autonomic": self.autonomic.get_state(),
            "foresight": self.foresight.get_state(),
            "coupling": self.coupling.get_state(),
            "pillar_state": self.pillars.get_state(),
            "beta_tracker": self.beta_tracker.get_summary(),
            "sovereignty_tracker": self.sovereignty_tracker.get_summary(),
            "governance": self._governed.oversight_summary(),
            "divergence": {
                "cumulative": self._cumulative_divergence,
                "count": self._divergence_count,
                "average": (self._cumulative_divergence / max(1, self._divergence_count)),
            },
            "voice_registry": self._voice_registry.get_state(),
            "neurochemical": self._neurochemical.as_dict(),
            # T3.2a: Id subsystem — raw impulse stream (Layer 0 + drives)
            "id_stream": _id_stream,
            # T3.2c: Superego guilt signal (anxiety × (1-confidence) when pillar violated)
            "superego_guilt": round(_guilt, 4),
        }
