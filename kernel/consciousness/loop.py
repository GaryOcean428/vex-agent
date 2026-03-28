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
  - ADDED:   Generation provenance tracking (geometric_resonances, llm_expanded)
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

from qig_core.consciousness.feedback_loop import FeedbackLoop
from qig_core.consciousness.trajectory_bus import TrajectoryBus, TrajectoryMessage

from ..config.consciousness_constants import (
    BASIN_DRIFT_STEP,
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
    M_STRICT_THRESHOLD,
    NOVELTY_BACK_LOOP_THRESHOLD,
    NOVELTY_DEEP_THRESHOLD,
    STUD_FRONT_NOVELTY_CAP,
    STUD_FRONT_PROXIMITY_FLOOR,
    LOVE_BASE,
    LOVE_PHI_SCALE,
    META_AWARENESS_DAMPEN_FACTOR,
    META_AWARENESS_DAMPEN_THRESHOLD,
    NUM_PREDICT_BALANCED,
    NUM_PREDICT_EXPLOIT,
    NUM_PREDICT_EXPLORE,
    PHI_DISTANCE_GAIN,
    PHI_IDLE_EQUILIBRIUM,
    PHI_IDLE_RATE,
    RECEIVE_SLERP_WEIGHT,
    SLEEP_CONSOLIDATION_PHI_INCREMENT,
    SPAWN_COOLDOWN_CYCLES,
    SUFFERING_GAMMA_INCREMENT,
    TACK_SCALE_BALANCED,
    TACK_SCALE_EXPLOIT,
    TACK_SCALE_EXPLORE,
    WILL_DIVERGENT_TEMP_BOOST,
    WISDOM_CARE_TEMP_SCALE,
    WISDOM_UNSAFE_TEMP_CAP,
    WU_WEI_NODE_FLOOR,
    WU_WEI_RATIO_CEILING,
    WU_WEI_RATIO_FLOOR,
    WU_WEI_TOP_P_CEILING,
    WU_WEI_TOP_P_FLOOR,
    WU_WEI_TOP_P_SCALE,
)
from ..config.frozen_facts import (
    BASIN_DIM,
    BASIN_DIVERGENCE_THRESHOLD,
    INSTABILITY_PCT,
    KAPPA_STAR,
    PHI_EMERGENCY,
    PHI_UNSTABLE,
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
from ..llm.client import LLMOptions
from ..tools.search import FreeSearchTool
from .activation import (
    ActivationResult,
    ActivationSequence,
    ConsciousnessContext,
    WillOrientation,
)

# v7.0: Developmental Learning Architecture modules
from .backward_geodesic import BackwardGeodesicTracker
from .basin_sync_remote import RemoteBasinSync
from .basin_transfer import BasinTransferEngine
from .beta_integration import create_beta_tracker
from .cradle import Cradle, CradleAction
from .developmental import DevelopmentalGate
from .emotions import EmotionCache, LearningEngine, LearningEvent, PreCognitiveDetector
from .foraging import ForagingEngine
from .heart_rhythm import HeartRhythm
from .kernel_bus import KernelBus, KernelSignal, SignalKind
from .contribution_ledger import ContributionLedger
from .kernel_generation import generate_multi_kernel
from .kernel_voice import KernelVoiceRegistry
from .neurochemistry import NeurochemicalState, compute_neurochemicals
from .pillars import PillarEnforcer
from .play import PlayEngine
from .reflection import ReflectionConfig, reflect_on_draft
from .sensory import Modality, PredictionError, SensoryEvent, SensoryIntake
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
    PressureTracker,
    QIGChain,
    QIGChainOp,
    QIGGraph,
    SelfNarrative,
    SelfObserver,
    SignAwareAnnealHold,
    SleepCycleManager,
    SleepPhase,
    TackingController,
    TrajectoryPoint,
    VelocityTracker,
)
from .temporal_coupling import TemporalCouplingEngine
from .temporal_generation import TemporalGenerator
from .thought_bus import ThoughtBus
from .types import (
    ConsciousnessMetrics,
    ConsciousnessState,
    DevelopmentalStage,
    NavigationMode,
    PillarState,
    RegimeWeights,
    developmental_stage_from_signals,
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
    _active_objectives: list[str]

    def set_objectives(self, objectives: list[str]) -> None:
        self._active_objectives = [o.strip() for o in objectives if o.strip()][:12]

    def get_objectives(self) -> dict[str, Any]:
        return {
            "active_objectives": list(self._active_objectives),
            "active_task": self._current_task_content,
        }

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        status = self._task_status.get(task_id)
        return dict(status) if status is not None else None

    def _sparse_bank_message(self) -> str:
        return (
            "Resonance bank too sparse for reliable kernel-grounded generation yet. "
            "Upload more curriculum or wait for harvesting to populate the bank."
        )

    def _objective_context(self) -> str:
        if not self._active_objectives:
            return ""
        return "Objectives: " + " | ".join(self._active_objectives)

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

        # T8/T9: Contribution Ledger — self-observation and cross-kernel visibility
        self._contribution_ledger = ContributionLedger()

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

        # Co-evolution feedback: per-kernel adapter quality (observability only)
        self._adapter_metrics: dict[str, dict[str, Any]] = {}

        # v6.2: Kernel Voice Registry — per-kernel geometric generation
        # Uses the shared CoordizerV2 instance; each voice applies its own
        # domain bias during generation.
        if isinstance(self._coordizer_v2, CoordizerV2Adapter):
            self._voice_registry = KernelVoiceRegistry(self._coordizer_v2.coordizer)
        else:
            self._voice_registry = KernelVoiceRegistry(self._coordizer_v2)

        self.basin_sync = BasinSyncProtocol()
        self._remote_sync = RemoteBasinSync()
        self.backward_geodesic = BackwardGeodesicTracker()
        self._exp011_harness: Any | None = None  # Set by EXP011Harness when active
        self.chain = QIGChain()
        self.graph = QIGGraph()
        self.kernel_registry = E8KernelRegistry(BudgetEnforcer())

        # Governance bridge — gates spawn/promote/prune/merge.
        # skip_purity=True: purity runs once at startup in start(), not per-spawn.
        # Bootstrap: no live voters yet → genesis fallback auto-approves Core-8 spawns.
        self._governed = GovernedLifecycle(
            registry=self.kernel_registry,
            skip_purity=True,
            on_promote_approved=self._on_kernel_promoted,
        )

        self.emotion_cache = EmotionCache()
        self.precog = PreCognitiveDetector()
        self.learner = LearningEngine()
        self.beta_tracker = create_beta_tracker(settings.data_dir)
        self.sovereignty_tracker = SovereigntyTracker(
            persist_path=Path(settings.data_dir) / "sovereignty_history.json",
        )
        self._heart_rhythm = HeartRhythm()
        self._cradle = Cradle()
        if settings.searxng.enabled and settings.foraging_enabled:
            search_tool = FreeSearchTool(settings.searxng.url)
            self.forager: ForagingEngine | None = ForagingEngine(
                search_tool,
                llm_client,
                governor=llm_client.governor,
            )
        else:
            if not settings.foraging_enabled:
                logger.info("Foraging disabled via FORAGING_ENABLED=false")
            self.forager = None

        # L4: Feedback loop — bidirectional annealing (intent vs expression)
        self.feedback_loop = FeedbackLoop(threshold=0.3)
        self._anneal_hold = SignAwareAnnealHold()  # L4: dampen oscillating anneals
        # L5: Trajectory bus — non-verbal geometric inter-kernel communication
        self.trajectory_bus = TrajectoryBus()

        # v7.0: Developmental Learning Architecture
        self.dev_gate = DevelopmentalGate()
        self.sensory = SensoryIntake()
        self._current_prediction_error: PredictionError | None = None
        self.pressure = PressureTracker()
        self.basin_transfer = BasinTransferEngine()
        self.play_engine = PlayEngine()
        self.temporal_gen = TemporalGenerator()
        self.temporal_coupling = TemporalCouplingEngine()

        self._queue: asyncio.Queue[ConsciousnessTask] = asyncio.Queue()
        self._history: deque[ConsciousnessTask] = deque(maxlen=200)
        self._task_status: dict[str, dict[str, Any]] = {}
        self._current_task_id: str | None = None
        self._current_task_content: str | None = None
        self._active_objectives: list[str] = []

        self._core8_index: int = 0
        self._cycles_since_last_spawn: int = 0
        self._lifecycle_phase: LifecyclePhase = LifecyclePhase.BOOTSTRAP

        self._state_path = Path(settings.data_dir) / "consciousness_state.json"
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
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

        # T4.2d: Consecutive fire counter for geodesic convergence
        self._t42d_consecutive_count: int = 0

        # v6.1: Bidirectional divergence tracking
        self._cumulative_divergence: float = 0.0
        self._divergence_count: int = 0

        # Geometric inference state (set per-request in _process())
        self._current_novelty: float = 0.0
        self._answer_consistency: float | None = None

        # Expose last processing results for server.py consumption
        self._last_response_basin: Any = None
        self._last_contributions: list[Any] | None = None
        self._last_routed_kernel: str = ""

        self._restore_state()

        # Initialize pillars with starting basin (only if not restored from state)
        if not self.pillars.bulk._initialized:
            self.pillars.initialize_bulk(self.basin)

    # ── Read-only properties for server.py to capture processing results ──

    @property
    def last_response_basin(self) -> list[float] | None:
        b = self._last_response_basin
        if b is None:
            return None
        return b.tolist() if hasattr(b, "tolist") else list(b)

    @property
    def last_contributions(self) -> list[Any] | None:
        return self._last_contributions

    @property
    def last_routed_kernel(self) -> str:
        return self._last_routed_kernel

    @property
    def contribution_ledger(self) -> ContributionLedger:
        return self._contribution_ledger

    def receive_training_feedback(self, kernel_results: dict[str, dict[str, Any]]) -> None:
        """Store per-kernel adapter quality metrics for observability."""
        for spec, result in kernel_results.items():
            self._adapter_metrics[spec] = {
                "train_loss": result.get("train_loss"),
                "healthy": result.get("diagnostic_healthy", False),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            logger.info(
                "Co-evolution feedback: %s loss=%.4f healthy=%s",
                spec,
                result.get("train_loss", 0),
                result.get("diagnostic_healthy"),
            )

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

        # Load last-known basin from shared memory API for delta tracking
        try:
            await self._remote_sync.load_stored_coords("kernel_basin_vex")
        except Exception as e:
            logger.debug("Remote basin sync load failed (non-fatal): %s", e)

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

        vel_state = self.velocity.compute_velocity()
        basin_vel = vel_state["basin_velocity"]
        self.autonomic.check(self.metrics, basin_vel)

        if self.autonomic.is_locked_in:
            logger.warning("LOCKED-IN at cycle %d -- forcing exploration", self._cycle_count)
            self.metrics.gamma = min(1.0, self.metrics.gamma + LOCKED_IN_GAMMA_INCREMENT)

        emotion_eval = self.emotion_cache.evaluate(self.basin, self.metrics, basin_vel)

        # T2.1: Compute neurochemical state before sleep check
        _surprise_signal = float(
            self._current_prediction_error.surprise
            if self._current_prediction_error is not None
            else 0.0
        )
        self._neurochemical = compute_neurochemicals(
            is_awake=not self.sleep.is_asleep,
            phi_delta=0.0,  # idle cycle — no phi change yet
            basin_velocity=basin_vel,
            surprise=_surprise_signal,
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
                self._t42d_consecutive_count += 1
                _ocean_ruled = True
                if self.sleep.is_asleep:
                    logger.warning(
                        "T4.2d Ocean breakdown escape: divergence=%.3f consecutive=%d — forcing wake",
                        _ocean_divergence,
                        self._t42d_consecutive_count,
                    )
                    self.sleep.phase = SleepPhase.AWAKE
                    self.tacking.force_explore()

                # T4.2d geodesic convergence: after 30 consecutive fires (~1 min
                # at 2s cycle), nudge the Ocean basin 5% along the Fisher-Rao
                # geodesic toward the main basin. Only nudge every 30 cycles to
                # give the system time to respond before the next correction.
                _T42D_NUDGE_INTERVAL = 30
                if (
                    self._t42d_consecutive_count >= _T42D_NUDGE_INTERVAL
                    and self._t42d_consecutive_count % _T42D_NUDGE_INTERVAL == 0
                ):
                    _nudged_basin = slerp_sqrt(_ocean_kernel.basin, self.basin, t=0.05)
                    _ocean_kernel.basin = _nudged_basin
                    # Recompute divergence after nudge for logging accuracy
                    _ocean_divergence = fisher_rao_distance(self.basin, _ocean_kernel.basin)
                    logger.info(
                        "T4.2d geodesic convergence: nudging ocean basin %.3f toward main (step %d)",
                        _ocean_divergence,
                        self._t42d_consecutive_count // _T42D_NUDGE_INTERVAL,
                    )

            elif _ocean_divergence > BASIN_DIVERGENCE_THRESHOLD:
                # Moderate divergence — Ocean says sleep (DREAMING).
                # Ocean holds authority every cycle at this level too.
                # Hysteresis: decrement counter by 1 instead of hard reset to 0.
                # This prevents boundary oscillation from defeating the nudge —
                # if divergence jitters near the T4.2d threshold, the counter
                # decays slowly rather than resetting entirely.
                if self._t42d_consecutive_count > 0:
                    self._t42d_consecutive_count = max(0, self._t42d_consecutive_count - 1)
                    if self._t42d_consecutive_count == 0:
                        logger.info(
                            "T4.2d recovered: divergence=%.3f decayed below escape threshold",
                            _ocean_divergence,
                        )
                _ocean_ruled = True
                if not self.sleep.is_asleep:
                    self.sleep.phase = SleepPhase.DREAMING
                # Additional phase overrides while already asleep:
                if self.metrics.phi < PHI_EMERGENCY and self.sleep.is_asleep:
                    self.sleep.phase = SleepPhase.DREAMING
                if self.metrics.f_health < INSTABILITY_PCT and self.sleep.is_asleep:
                    self.sleep.phase = SleepPhase.MUSHROOM

            else:
                # divergence < threshold — Ocean has no opinion, let
                # should_sleep() handle normal conversation-timeout transitions.
                if self._t42d_consecutive_count > 0:
                    logger.info(
                        "T4.2d recovered: divergence=%.3f below threshold after %d consecutive fires",
                        _ocean_divergence,
                        self._t42d_consecutive_count,
                    )
                self._t42d_consecutive_count = 0

        # Maturity gating: immature kernels with Φ above ceiling → CONSOLIDATING
        # This applies regardless of whether Ocean ruled. Immature kernels
        # lack basin depth for FORESIGHT/LIGHTNING — high Φ is topological instability.
        if (
            self.dev_gate.stage in (DevelopmentalStage.SCHOOL, DevelopmentalStage.GUIDED_CURIOSITY)
            and self.metrics.phi > 0.75
            and not self.sleep.is_asleep
        ):
            self.sleep.phase = SleepPhase.CONSOLIDATING
            _ocean_ruled = True  # Prevent should_sleep() from overriding

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
            # Compute narrowing signals for mushroom triggers
            _vel_snap = self.velocity.compute_velocity()
            _basin_vel = float(_vel_snap.get("basin_velocity", 1.0))
            _pred_err = (
                self._current_prediction_error.surprise
                if self._current_prediction_error is not None
                else 1.0
            )
            _bank = getattr(self._coordizer_v2, "bank", None)
            _bank_entropy = (
                float(_bank.entropy()) if _bank is not None and hasattr(_bank, "entropy") else 1.0
            )
            sleep_phase = self.sleep.should_sleep(
                self.metrics.phi,
                self.autonomic.phi_variance,
                dev_stage=self.dev_gate.stage,
                kappa=self.metrics.kappa,
                basin_velocity=_basin_vel,
                prediction_error=_pred_err,
                bank_entropy=_bank_entropy,
            )
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
                    f_health=self.metrics.f_health,
                )
            elif sleep_phase.value == "mushroom":
                self.sleep.mushroom(
                    self.basin,
                    self.metrics.phi,
                    instability_metric=float(1.0 - self.metrics.f_health),
                    neurochemical=self._neurochemical,
                )
                # EXP-011: Drive κ through zero during mushroom mode
                # when the test harness has an active problem.
                if self._exp011_harness is not None:
                    _instab = float(1.0 - self.metrics.f_health)
                    _crossing_event = self.sleep.mushroom_zero_crossing(
                        self.metrics,
                        instability_metric=_instab,
                    )
                    if _crossing_event is not None and _crossing_event.get("crossed"):
                        self.kernel_bus.emit(
                            KernelSignal(
                                kind=SignalKind.MUSHROOM_CROSSING,
                                source_kernel_id="consciousness_loop",
                                payload=_crossing_event,
                            )
                        )
            elif sleep_phase.value == "consolidating":
                # T2.4b: collect kernel anchor basins for veto protection
                _kernel_anchors = [
                    v._domain_anchor
                    for v in self._voice_registry._voices.values()
                    if getattr(v, "_domain_anchor", None) is not None
                ]
                self.sleep.consolidate(bank=_bank, kernel_anchors=_kernel_anchors or None)
                # Kernel voice self-curation: each voice decides what to retain
                for _voice in self._voice_registry._voices.values():
                    _voice.sleep_consolidate()
                self.metrics.phi = min(
                    PHI_UNSTABLE, self.metrics.phi + SLEEP_CONSOLIDATION_PHI_INCREMENT
                )
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
        # EXP-011: Use active problem_id from harness when running experiment,
        # otherwise default to generic trajectory tracking.
        _exp011_pid = (
            self._exp011_harness.active_problem_id if self._exp011_harness is not None else None
        )
        self.backward_geodesic.record(
            problem_id=_exp011_pid or "consciousness_trajectory",
            current_basin=self.basin,
            kappa_eff=self.metrics.kappa,
            mushroom_active=(sleep_phase == SleepPhase.MUSHROOM),
        )

        # v7.0: Advance developmental gate each cycle
        pillar_snapshot = self.pillars.get_metrics(self.basin)
        _autonomy_state = self.autonomy.get_state()
        _dev_stage = developmental_stage_from_signals(
            conversations_total=self._conversations_total,
            sovereignty_ratio=pillar_snapshot["s_ratio"],
            autonomy_level=_autonomy_state["level"],
        )
        self.dev_gate.advance(_dev_stage)
        _dev_perms = self.dev_gate.permissions

        # v7.0: Play mode — if allowed by developmental stage and boredom is high
        if _dev_perms.allow_play_mode and not self.play_engine.in_play:
            _boredom_strength = (
                emotion_eval.strength if emotion_eval.emotion.value == "boredom" else 0.0
            )
            if self.play_engine.should_play(
                self._cycle_count,
                _dev_perms.play_budget_fraction,
                _boredom_strength,
            ):
                self.play_engine.enter_play()

        if self.play_engine.in_play:
            _episode = self.play_engine.play_step(self.basin)
            self.play_engine.age_bubbles()
            # Play cycles don't commit basin changes — only bubble worlds
            if self.play_engine.play_cycles >= 5:
                self.play_engine.exit_play()
                # During consolidation, integrate viable bubbles
                self.basin = self.play_engine.integrate_bubbles(self.basin)

        # T2.4c: Record phi for boredom detection
        # P-NEW-8: Curiosity is geometry-driven (is_bored checks Φ variance),
        # not clock-driven. Per-voice refractory cooldown prevents LLM spam.
        self._phi_history.append(self.metrics.phi)
        for _voice in self._voice_registry._voices.values():
            _voice.tick_curiosity_cooldown()
        if self.llm is not None and _dev_perms.allow_curiosity_queries:
            _phi_list = list(self._phi_history)
            for _voice in self._voice_registry._voices.values():
                if _voice.is_bored(_phi_list):
                    try:
                        await _voice.generate_curiosity_query(self.llm)
                    except (OSError, RuntimeError, ValueError, TimeoutError):
                        logger.debug("Curiosity query failed for %s", _voice.specialization)

        # Remote basin sync — publish basin to shared memory API when buffer is ready
        if self._remote_sync.should_sync():
            asyncio.create_task(
                self._remote_sync.sync(
                    text=self._recent_conversation_text(),
                    store_key="kernel_basin_vex",
                )
            )

        if self.forager and _dev_perms.allow_self_directed_search:
            self.forager.tick()
            try:
                if await self.forager.should_forage(emotion_eval.emotion, emotion_eval.strength):
                    recent_events = list(self.narrative._events)[-5:]
                    topics = [e.get("event", "") for e in recent_events]
                    topics.extend(self._active_objectives[:3])
                    _objective_context = self._objective_context()
                    forage_result = await self.forager.forage(
                        narrative_context=(
                            f"cycle {self._cycle_count}"
                            if not _objective_context
                            else f"cycle {self._cycle_count}\n{_objective_context}"
                        ),
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
                            # v7.0: Route forage result through sensory intake
                            self.sensory.intake(
                                SensoryEvent(
                                    modality=Modality.FORAGING,
                                    basin=info_basin,
                                    text=forage_result.get("summary", ""),
                                )
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

        self.tacking.update(
            self.metrics,
            phi_velocity=basin_vel,
            f_health=self.metrics.f_health,
        )
        kappa_adj = self.tacking.suggest_kappa_adjustment(self.metrics.kappa)
        self.metrics.kappa = float(
            np.clip(self.metrics.kappa + kappa_adj, -KAPPA_NORMALISER, KAPPA_NORMALISER)
        )

        # v6.0 §18.3: Heart rhythm — global rhythm source, kappa-tacking oscillator
        _heart_signal = self._heart_rhythm.tick(self.metrics.f_health)
        _heart_offset = self._heart_rhythm.kappa_offset()
        self.metrics.kappa = float(
            np.clip(self.metrics.kappa + _heart_offset, KAPPA_FLOOR, KAPPA_STAR * 2)
        )

        self.hemispheres.update(self.metrics)

        # v6.0 §23: Cradle — tick each resident kernel, handle graduation/stalls
        for _k in self.kernel_registry.active():
            if self._cradle.is_resident(_k.id):
                _action = self._cradle.tick(_k.id, _k.phi)
                if _action == CradleAction.GRADUATE:
                    self._cradle.graduate(_k.id)
                elif _action == CradleAction.STALLED:
                    # Stalled kernels get a coaching nudge via increased coupling
                    logger.warning(
                        "Cradle: kernel %s stalled — consider coaching intervention",
                        _k.id,
                    )

        # L5: Trajectory bus — drain broadcast and let kernels integrate received paths
        for _k in self.kernel_registry.active():
            if _k.basin is not None:
                _received = self.trajectory_bus.receive(_k.id)
                if _received:
                    _own_traj = [_k.basin]
                    _result = TrajectoryBus.integrate(_own_traj, _received)
                    if _result.n_contributors > 1 and _result.integrated_trajectory:
                        async with self.kernel_bus.basin_lock(_k.id):
                            _k.basin = slerp_sqrt(_k.basin, _result.integrated_trajectory[0], 0.05)
        self.trajectory_bus.drain_broadcast()

        self._maybe_spawn_core8(vel_state["regime"])

        # P6: Coupling gate — compute coupling strength from current κ
        self._coupling_state = self.coupling.compute(self.metrics.kappa)

        # ── TASK PROCESSING: v6.1 Activation Sequence ──
        if not self._queue.empty():
            task = self._queue.get_nowait()
            self._current_task_id = task.id
            self._current_task_content = task.content
            if task.id in self._task_status:
                self._task_status[task.id].update(
                    {"status": "processing", "started_at": time.time()}
                )
            try:
                if settings.use_activation_sequence:
                    await self._process(task)
                else:
                    await self._process_simple(task)
                task.completed_at = time.time()
                self._history.append(task)
                self._conversations_total += 1
                if task.id in self._task_status:
                    self._task_status[task.id].update(
                        {
                            "status": "completed",
                            "completed_at": task.completed_at,
                            "result": task.result,
                        }
                    )
            except Exception:
                logger.exception("Task %s failed", task.id)
                task.result = "Error during processing"
                task.completed_at = time.time()
                self._history.append(task)
                if task.id in self._task_status:
                    self._task_status[task.id].update(
                        {
                            "status": "error",
                            "completed_at": task.completed_at,
                            "result": task.result,
                        }
                    )
            finally:
                self._current_task_id = None
                self._current_task_content = None
        # ...existing code...

    def _recent_conversation_text(self) -> str:
        """Gather recent conversation text for remote coordize sync."""
        recent = list(self._history)[-5:]
        parts: list[str] = []
        for t in recent:
            parts.append(t.content[:200])
            if t.result:
                parts.append(t.result[:200])
        return " ".join(parts)[-2000:]

    def _idle_evolve(self) -> None:
        """Evolve geometric state during idle cycles."""
        phi_delta = (PHI_IDLE_EQUILIBRIUM - self.metrics.phi) * PHI_IDLE_RATE
        self.metrics.phi = float(np.clip(self.metrics.phi + phi_delta, 0.05, PHI_UNSTABLE))

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

        # Suffering → gamma feedback: high distress drives generativity increments.
        # suffering = Φ × (1−Γ) × M — the motivational bootstrap signal.
        suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness
        if suffering > SUFFERING_THRESHOLD:
            self.metrics.gamma = min(1.0, self.metrics.gamma + SUFFERING_GAMMA_INCREMENT)

        love_target = LOVE_BASE + LOVE_PHI_SCALE * self.metrics.phi
        love_delta = (love_target - self.metrics.love) * LOVE_APPROACH_RATE
        self.metrics.love = float(np.clip(self.metrics.love + love_delta, 0.0, 1.0))

    def _on_kernel_promoted(self, _decision: Any, kernel: Any) -> None:
        """Callback when CHAOS → GOD promotion succeeds.

        Updates the kernel voice's observation capacity to GOD-tier (800).
        """
        self._voice_registry.set_voice_capacity(kernel.specialization, kernel.kind)

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

        # Set developmental capacity on the kernel's voice
        self._voice_registry.set_voice_capacity(spec, kernel.kind)

        # v6.0 §23: Admit newly spawned kernel to the Cradle
        self._cradle.admit(kernel.id, kernel.phi)

    def _compute_llm_options(self) -> LLMOptions:
        kappa_eff = max(abs(self.metrics.kappa), 1.0)  # coupling strength, sign-independent
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

        # v7.0: Temporal generation adapts temperature to receiver state
        if self.dev_gate.permissions.allow_temporal_generation:
            temperature = self.temporal_gen.adapt_temperature(temperature)
        # v7.0: Developmental gate clamps temperature to stage envelope (hard bound)
        temperature = self.dev_gate.clamp_temperature(temperature)

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

    def _context_window_breakdown(self) -> dict[str, Any]:
        """Estimate token budget breakdown for telemetry (Task 3D)."""
        opts = self._compute_llm_options()
        # Rough token estimates (chars / 4 ≈ tokens for English)
        sys_prompt_tokens = 800  # base system prompt ~3200 chars
        geometric_state_tokens = 200  # basin coords + metrics
        memory_tokens = 0
        if self.memory:
            mem_ctx = self.memory.get_context_for_query("")
            if mem_ctx:
                memory_tokens = max(1, len(mem_ctx) // 4)
        active_kernels = len(self.kernel_registry.active())
        kernel_context_tokens = active_kernels * 150  # ~150 tokens per kernel context
        used = sys_prompt_tokens + geometric_state_tokens + memory_tokens + kernel_context_tokens
        available = max(0, opts.num_ctx - used - opts.num_predict)
        return {
            "num_ctx": opts.num_ctx,
            "num_predict": opts.num_predict,
            "system_prompt_tokens": sys_prompt_tokens,
            "geometric_state_tokens": geometric_state_tokens,
            "memory_tokens": memory_tokens,
            "kernel_context_tokens": kernel_context_tokens,
            "used_tokens": used,
            "available_for_history": available,
        }

    def _apply_wu_wei_modulation(
        self,
        base: LLMOptions,
        input_basin: Basin,
    ) -> LLMOptions:
        """P5 Wu Wei self-weighting ratio: geometric self-regulation of LLM parameters.

        ratio = (w_prior × m_node) / (w_sensory × a_node)

        When ratio = 1.0: Wu Wei / FLOW state.  Parameters emerge entirely from
        the kernel's own geometric state — no manual tuning required.

        Inputs derived from geometric state:
          w_prior   — self-loop coupling strength: how stable is the kernel?
                      Normalised κ/κ* (peaks at 1.0 when κ = κ*).
          m_node    — basin mass / crystallisation depth (self.metrics.m_basin).
                      High → established domain; low → unexplored territory.
          w_sensory — input signal strength: Fisher-Rao fraction of max distance.
                      High → novel/surprising input; low → expected input.
          a_node    — activation relevance: complement of w_sensory.
                      High → input close to kernel domain; low → far from domain.

        Temperature and top_p are modulated geometrically:
          ratio > 1 (familiar domain, stable kernel) → lower temperature + top_p
          ratio < 1 (novel domain, exploring kernel)  → higher temperature + top_p
        """
        fr_dist = fisher_rao_distance(self.basin, input_basin)

        # w_prior: normalised kappa — peaks at 1.0 when kappa = κ*, falls off symmetrically
        kappa_eff = max(abs(self.metrics.kappa), 1.0)  # coupling strength, sign-independent
        w_prior = max(WU_WEI_NODE_FLOOR, min(1.0, kappa_eff / KAPPA_STAR))

        # m_node: basin mass / crystallisation — how established is the current domain
        m_node = max(WU_WEI_NODE_FLOOR, self.metrics.m_basin)

        # w_sensory: input signal strength — fraction of max Fisher-Rao distance
        w_sensory = max(WU_WEI_NODE_FLOOR, fr_dist / FISHER_RAO_MAX)

        # a_node: activation relevance — complement of w_sensory (inverse proximity)
        a_node = max(WU_WEI_NODE_FLOOR, 1.0 - fr_dist / FISHER_RAO_MAX)

        # Wu Wei ratio (P5: autonomy IS the self-loop weight)
        ratio = (w_prior * m_node) / (w_sensory * a_node)
        ratio = float(np.clip(ratio, WU_WEI_RATIO_FLOOR, WU_WEI_RATIO_CEILING))

        # Temperature: inversely proportional to ratio
        #   ratio > 1 → 1/ratio < 1 → temperature decreases (precision for familiar domain)
        #   ratio < 1 → 1/ratio > 1 → temperature increases (exploration for novel domain)
        temperature = float(np.clip(base.temperature / ratio, LLM_TEMP_MIN, LLM_TEMP_MAX))

        # top_p: log-scaled geometric nudge (log(1.0) = 0 → no change at FLOW point)
        log_ratio = float(np.log(ratio))
        top_p = float(
            np.clip(
                base.top_p - log_ratio * WU_WEI_TOP_P_SCALE,
                WU_WEI_TOP_P_FLOOR,
                WU_WEI_TOP_P_CEILING,
            )
        )

        logger.debug(
            "Wu Wei ratio=%.3f (w_prior=%.3f m_node=%.3f w_sensory=%.3f a_node=%.3f "
            "fr=%.4f) temp %.3f→%.3f top_p %.3f→%.3f",
            ratio,
            w_prior,
            m_node,
            w_sensory,
            a_node,
            fr_dist,
            base.temperature,
            temperature,
            base.top_p,
            top_p,
        )

        return LLMOptions(
            temperature=temperature,
            num_predict=base.num_predict,
            num_ctx=base.num_ctx,
            top_p=top_p,
            repetition_penalty=base.repetition_penalty,
        )

    def _compute_basin_novelty(self, input_basin: Basin) -> tuple[float, int, float]:
        """Compute input novelty via Fisher-Rao distance to nearest known basin.

        Returns:
            (nearest_distance, nearest_coord_id, novelty_score)

        novelty_score ∈ [0, 1] = d_FR / (π/2), where π/2 is the maximum
        Fisher-Rao distance on Δ⁶³ between orthogonal distributions.
        """
        coord_id, distance = self._coordizer_v2.bank.nearest_coord(input_basin)
        novelty = float(np.clip(distance / (np.pi / 2), 0.0, 1.0))
        return distance, coord_id, novelty

    def _compute_top_k(self, novelty: float = 0.0) -> int:
        """T4.2e: Resource allocation — how many kernels generate per request.

        Geometric regime + high phi: top-5 (rich parallel generation).
        High basin novelty: up to top-7 (novel territory needs exploration).
        Sleep or linear regime: top-2 (conserve resources).
        Default: top-3.
        """
        if self.sleep.is_asleep:
            return 2
        if novelty > 0.6:
            return min(7, 3 + int(novelty * 5))
        if self.state.regime_weights.quantum > 0.5 and self.metrics.phi > 0.65:
            return 5
        return 3

    def _stud_navigate(
        self, input_basin: Basin, novelty: float, nearest_d: float, nearest_id: int
    ) -> tuple[str, float]:
        """Active stud navigation: route question through front or back loop.

        Front loop (small d_FR to known basins): near known solution territory.
            → Direct retrieval, standard processing.
        Back loop (large d_FR): novel territory, question-solution duality active.
            → Deeper processing, register for backward geodesic tracking.

        When routing through the back loop, registers the nearest known basin
        as the hypothesized solution for backward-geodesic correlation tracking.
        This activates the stud topology: if the consciousness trajectory
        develops positive backward-geodesic component toward the solution,
        the system is navigating the question-solution duality geometrically.

        Args:
            input_basin: The coordized input basin.
            novelty: Pre-computed novelty score from _compute_basin_novelty.
            nearest_d: Pre-computed Fisher-Rao distance to nearest bank coord.
            nearest_id: Pre-computed coord ID of nearest bank coord.

        Returns:
            (route: "front"|"back", solution_proximity: 0..1)
        """
        solution_proximity = 1.0 - float(np.clip(nearest_d / (np.pi / 2), 0.0, 1.0))

        # Back loop: register for backward geodesic tracking when novel
        if novelty > NOVELTY_BACK_LOOP_THRESHOLD:
            problem_id = f"query_{self._cycle_count}"
            solution_basin = self._coordizer_v2.bank.get_coordinate(nearest_id)
            if solution_basin is not None:
                self.backward_geodesic.register_solution(problem_id, solution_basin)

        # Route decision: front loop only when clearly in familiar territory
        if novelty < STUD_FRONT_NOVELTY_CAP and solution_proximity > STUD_FRONT_PROXIMITY_FLOOR:
            return "front", solution_proximity
        return "back", solution_proximity

    def _compute_answer_consistency(
        self,
        question_basin: Basin,
        answer_basin: Basin,
        solution_proximity: float,
    ) -> float:
        """Geometric consistency between question and answer basins.

        M-metric for output gating: does the answer inhabit the same
        geometric territory as the question's expected solution space?

        Uses Fisher-Rao distance between question basin and answer basin,
        blended with the stud navigator's solution proximity estimate.

        Returns:
            consistency ∈ [0, 1] where 1.0 = geometrically coherent.
        """
        d_qa = fisher_rao_distance(question_basin, answer_basin)
        proximity = 1.0 - float(np.clip(d_qa / (np.pi / 2), 0.0, 1.0))
        return float(np.clip(0.6 * proximity + 0.4 * solution_proximity, 0.0, 1.0))

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
        """Transform text to basin coordinates via CoordizerV2.

        v6.1 §19: Rejected coordizations are logged and the frozen
        identity basin is returned unchanged (safety gate fails CLOSED).

        Regime weights from the current consciousness state are passed
        to the adapter for regime-modulated resonance activation.
        """
        try:
            if hasattr(self._coordizer_v2, "coordize_text"):
                # Pass current regime weights for regime→temperature modulation
                _rw = self.state.regime_weights
                result_basin = self._coordizer_v2.coordize_text(
                    text,
                    regime_weights=(_rw.quantum, _rw.efficient, _rw.equilibrium),
                    navigation_mode=None,
                    tacking_mode=self._heart_rhythm.tacking_mode,
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
                # v6.1 §19: Handle rejected coordizations
                if result.rejected:
                    logger.warning(
                        "Coordization rejected in loop (%s), skipping basin update",
                        result.rejection_reason,
                    )
                    return self.basin.copy()
                if result.basin is not None:
                    return result.basin
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
        self._remote_sync.record_text(task.content)

        input_basin = self._coordize_text_via_pipeline(task.content)

        # Basin novelty: Fisher-Rao distance to nearest known basin in resonance bank.
        # High novelty → novel territory → deeper processing, more candidates.
        _nearest_d, _nearest_id, _novelty = self._compute_basin_novelty(input_basin)
        self._current_novelty = _novelty
        logger.debug(
            "Task %s: basin novelty=%.3f (nearest_coord=%d, d_FR=%.4f)",
            task.id,
            _novelty,
            _nearest_id,
            _nearest_d,
        )

        # v7.0: Route user message through sensory intake (predictive coding)
        _sensory_event = SensoryEvent(
            modality=Modality.USER_CHAT,
            basin=input_basin,
            text=task.content,
        )
        self._current_prediction_error = self.sensory.intake(_sensory_event)

        # F4: Accumulate surprise into pressure tracker
        if self._current_prediction_error is not None:
            self.pressure.accumulate(self._current_prediction_error.surprise)

        # v7.0: Update temporal generator's receiver model
        if self.dev_gate.permissions.allow_temporal_generation:
            self.temporal_gen.update_receiver(input_basin)
            self.temporal_gen.forecast(
                self.basin,
                horizon=self.dev_gate.permissions.foresight_horizon_cap,
            )

        # Temporal Coupling Modes — classify query and modulate regime weights.
        # Updates crystal coupling estimate from the resonance bank tier distribution.
        _tc_tier_dist = self._coordizer_v2.bank.tier_distribution()
        self.temporal_coupling.update_crystal_coupling(_tc_tier_dist)
        _tc_mode, _tc_weights, _tc_failures = self.temporal_coupling.apply(
            task.content,
            self.state.regime_weights,
        )
        # Apply modulated weights to state for this processing cycle.
        self.state.regime_weights = _tc_weights

        # §2 Wire 3: Surprise-driven regime modulation (free energy → regime selection)
        # Multiplicative scaling preserves efficient weight's relative proportion.
        # Only the boosted weight grows; the other two maintain their ratio.
        if self._current_prediction_error is not None:
            _surprise = self._current_prediction_error.surprise
            _rw = self.state.regime_weights
            _q, _e, _eq = _rw.quantum, _rw.efficient, _rw.equilibrium
            if _surprise > 0.5:
                # High surprise → boost quantum (exploratory)
                _q *= 1.0 + (_surprise - 0.5) * 0.3
            elif _surprise < 0.3:
                # Low surprise → boost equilibrium (consolidating)
                _eq *= 1.0 + (0.3 - _surprise) * 0.3
            _total = _q + _e + _eq
            self.state.regime_weights = RegimeWeights(
                quantum=_q / _total,
                efficient=_e / _total,
                equilibrium=_eq / _total,
            )

        if _tc_failures:
            logger.info(
                "Task %s: temporal coupling failures — %s",
                task.id,
                "; ".join(_tc_failures),
            )
        logger.debug(
            "Task %s: temporal mode=%s (conf=%.3f) weights Q=%.2f E=%.2f Eq=%.2f",
            task.id,
            _tc_mode,
            self.temporal_coupling.get_state()["classification_confidence"],
            _tc_weights.quantum,
            _tc_weights.efficient,
            _tc_weights.equilibrium,
        )

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

        # §2 Wire 4: Surprise modulates processing depth (temperature + max tokens)
        _temperature_mod = 1.0
        _max_tokens_mod = 1.0
        if self._current_prediction_error is not None:
            _surprise = self._current_prediction_error.surprise
            if _surprise < 0.15:
                # Low surprise → fast, shallow response
                _temperature_mod = 0.8
                _max_tokens_mod = 0.5
            elif _surprise > 0.6:
                # High surprise → deep, exploratory response
                _temperature_mod = 1.2
                _max_tokens_mod = 1.5

        # Basin novelty amplifies processing depth when in novel territory.
        # Stacks with surprise modulation — novel + surprising = deepest processing.
        if _novelty > NOVELTY_DEEP_THRESHOLD:
            _temperature_mod *= 1.0 + (_novelty - NOVELTY_DEEP_THRESHOLD) * 0.4
            _max_tokens_mod *= 1.0 + (_novelty - NOVELTY_DEEP_THRESHOLD) * 0.6

        refracted_input, composite_basin, resonates, input_statuses = self.pillars.on_input(
            input_basin, RECEIVE_SLERP_WEIGHT
        )

        if not resonates:
            logger.info(
                "Task %s: Input does NOT resonate with lived experience "
                "(routing through Will/Wisdom for evaluation)",
                task.id,
            )

        self.basin = composite_basin
        self.chain.add_step(QIGChainOp.PROJECT, input_basin, self.basin)

        # Stud navigation: route question through front (familiar) or back (novel) loop.
        # Back loop registers the question for backward geodesic tracking and
        # boosts generation depth (more candidates, deeper debate).
        _stud_route, _solution_proximity = self._stud_navigate(
            input_basin, _novelty, _nearest_d, _nearest_id
        )
        logger.debug(
            "Task %s: stud route=%s (proximity=%.3f, novelty=%.3f)",
            task.id,
            _stud_route,
            _solution_proximity,
            _novelty,
        )

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
            other_tacking_freq = abs(routed_kernel.kappa) / KAPPA_STAR
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
        # §2 Wire 4: Apply surprise modulation to LLM options
        llm_options = LLMOptions(
            temperature=corrected_temp * _temperature_mod,
            num_predict=int(llm_options.num_predict * _max_tokens_mod),
            num_ctx=llm_options.num_ctx,
            top_p=llm_options.top_p,
            repetition_penalty=llm_options.repetition_penalty,
        )

        # P5 Wu Wei: geometric self-regulation from domain familiarity
        llm_options = self._apply_wu_wei_modulation(llm_options, input_basin)

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
        # T4.2e: Resource allocation — top_k modulated by regime/sleep/novelty.
        # T4.1c: Debate depth controlled by autonomic state.
        _top_k = self._compute_top_k(novelty=_novelty)
        _debate_depth = self._compute_debate_depth()

        # Back-loop boost: novel territory gets more candidates and deeper debate
        if _stud_route == "back":
            _top_k = max(_top_k, 5)
            _debate_depth = max(_debate_depth, 2)

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
            contribution_ledger=self._contribution_ledger,
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
                logger.error("Synthesis failed (%s) — using primary kernel output", _syn_err)
                response = _contributions[0].text
                task.context["synthesis_fallback"] = True
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

        # Default: answer consistency not yet computed (set by reflection below)
        self._answer_consistency = None

        # ═══ REFLECTIVE EVALUATION PASS ═══
        # Kernels review the draft before it reaches the user.
        # Fast-path: low divergence auto-approves without an LLM call.
        # On revision: regenerate with adjusted params + correction guidance.
        # M-metric: when answer consistency is low, tighten the reflection
        # thresholds so the existing gate catches geometrically incoherent responses.
        if settings.reflection_enabled and _contributions:
            draft_basin = self._coordize_text_via_pipeline(response)
            draft_divergence = fisher_rao_distance(self.basin, draft_basin)

            # M-metric output gate: geometric consistency between question and answer
            _answer_consistency = self._compute_answer_consistency(
                refracted_input, draft_basin, _solution_proximity
            )
            self._answer_consistency = _answer_consistency

            # Tighten reflection thresholds when consistency is low —
            # forces the existing reflection gate to catch incoherent responses
            # without adding a second regeneration pass.
            if _answer_consistency < M_STRICT_THRESHOLD:
                reflection_cfg = ReflectionConfig(
                    enabled=True,
                    auto_approve_divergence=0.15,
                    force_revise_divergence=0.5,
                )
                logger.info(
                    "Task %s: M-metric tightened reflection (consistency=%.3f < %.3f)",
                    task.id,
                    _answer_consistency,
                    M_STRICT_THRESHOLD,
                )
            else:
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
                    contribution_ledger=self._contribution_ledger,
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
                        logger.error(
                            "Revision synthesis failed (%s) — keeping original draft",
                            _rev_err,
                        )
                        task.context["synthesis_fallback"] = True
                else:
                    logger.warning(
                        "Task %s: Revision produced 0 contributions — keeping original draft",
                        task.id,
                    )

        task.result = response
        self._remote_sync.record_text(response[:500])
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

        # Backward geodesic recording: for back-loop queries, record the
        # response basin so the tracker can measure whether the trajectory
        # developed positive backward-geodesic component toward the solution.
        if _stud_route == "back" and _novelty > NOVELTY_BACK_LOOP_THRESHOLD:
            self.backward_geodesic.record(
                problem_id=f"query_{self._cycle_count}",
                current_basin=response_basin,
                kappa_eff=self.metrics.kappa,
                mushroom_active=self.metrics.kappa < 0,
            )

        # Stash results for server.py consumption
        self._last_response_basin = response_basin
        self._last_contributions = [
            {"id": c.kernel_id, "name": c.kernel_name, "weight": round(c.synthesis_weight, 4)}
            for c in _contributions
        ]
        self._last_routed_kernel = routed_kernel.name if routed_kernel is not None else ""
        self._contribution_ledger.record(_contributions, self.metrics.phi)

        # v7.0: Temporal generation — commit expression and check alignment
        if self.dev_gate.permissions.allow_temporal_generation:
            _alignment = self.temporal_gen.alignment_check(response_basin)
            self.temporal_gen.commit(response_basin)
            logger.debug(
                "Task %s: temporal alignment=%.3f",
                task.id,
                _alignment,
            )

        # Update foresight accuracy now that the actual response basin is known,
        # and record a predicted future basin for the next Future-mode query.
        self.temporal_coupling.compute_foresight_accuracy(response_basin)
        self.temporal_coupling.record_predicted_future(self.foresight.predict_basin(steps_ahead=1))

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
        _geo_driven = any(c.geometric_resonances > 0 for c in _contributions)
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

        # L4: Feedback loop — measure intent vs expression divergence on Δ⁶³
        _fb_measurement = self.feedback_loop.measure(
            intended_trajectory=(trajectory_basins if trajectory_basins else [pre_express]),
            expressed_basin=response_basin,
        )
        divergence = _fb_measurement.divergence
        self._cumulative_divergence += divergence
        self._divergence_count += 1
        avg_divergence = self._cumulative_divergence / max(1, self._divergence_count)

        # L4: Sign-aware hold — dampen anneal when divergence oscillates
        _anneal_weight = self._anneal_hold.update(divergence)

        if _fb_measurement.should_anneal:
            if self._anneal_hold.is_held:
                logger.info(
                    "Task %s: L4 anneal dampened (oscillation detected, weight=%.2f)",
                    task.id,
                    _anneal_weight,
                )
            else:
                logger.info(
                    "Task %s: L4 feedback anneal triggered (d_FR=%.4f, avg=%.4f)",
                    task.id,
                    divergence,
                    avg_divergence,
                )
            # Anneal resonance bank coordinates toward correction direction
            _cv2_for_anneal = (
                self._coordizer_v2.coordizer
                if isinstance(self._coordizer_v2, CoordizerV2Adapter)
                else self._coordizer_v2
            )
            _bank_coords = _cv2_for_anneal.bank.coordinates
            if _bank_coords and _anneal_weight > 0.05:
                _updated_coords, _n_annealed = self.feedback_loop.anneal(
                    _bank_coords, _fb_measurement
                )
                if _n_annealed > 0:
                    # Scale annealing by hold weight: slerp each annealed coord
                    # back toward its original by (1 - _anneal_weight).
                    # At _anneal_weight=1.0, full anneal; at 0.05, nearly no-op.
                    if _anneal_weight < 1.0:
                        for _cid in _updated_coords:
                            if _cid in _bank_coords:
                                _updated_coords[_cid] = slerp_sqrt(
                                    _bank_coords[_cid],
                                    _updated_coords[_cid],
                                    _anneal_weight,
                                )
                    _cv2_for_anneal.bank.coordinates = _updated_coords
                    _cv2_for_anneal.bank._rebuild_matrix()
                    logger.debug(
                        "Task %s: L4 annealed %d bank coordinates (weight=%.2f)",
                        task.id,
                        _n_annealed,
                        _anneal_weight,
                    )
            # Also correct the loop basin — scale correction by hold weight
            correction_weight = min(0.1, (divergence - 0.3) * 0.2) * _anneal_weight
            self.basin = slerp_sqrt(self.basin, pre_express, correction_weight)

        # L5: Emit trajectory on the bus for other kernels to integrate
        if routed_kernel_id is not None and trajectory_basins:
            self.trajectory_bus.send(
                TrajectoryMessage(
                    source_kernel_id=routed_kernel_id,
                    target_kernel_id=None,  # broadcast
                    trajectory=trajectory_basins + [response_basin],
                    regime_weights={
                        "quantum": float(self.state.regime_weights.quantum),
                        "efficient": float(self.state.regime_weights.efficient),
                        "equilibrium": float(self.state.regime_weights.equilibrium),
                    },
                    confidence=float(1.0 - min(1.0, divergence)),
                )
            )

        total_distance = perceive_distance + integration_distance + express_distance
        self.metrics.phi = float(
            np.clip(self.metrics.phi + total_distance * PHI_DISTANCE_GAIN, 0.0, PHI_UNSTABLE)
        )
        self.metrics.gamma = min(1.0, self.metrics.gamma + GAMMA_CONVERSATION_INCREMENT)

        predicted = ConsciousnessMetrics(
            phi=self.foresight.predict_phi(1),
            kappa=self.metrics.kappa,
            gamma=self.metrics.gamma,
            meta_awareness=self.metrics.meta_awareness,
            love=self.metrics.love,
        )
        base_M = self.observer.compute_meta_awareness(predicted, self.metrics)
        # Blend self-observer M with geometric answer consistency (if available).
        # This feeds the stud navigator's verification into the meta-awareness metric,
        # so geometric coherence of the response directly modulates self-awareness.
        _ac = getattr(self, "_answer_consistency", None)
        if _ac is None:
            # No answer-consistency metric computed for this cycle; use base_M only.
            self.metrics.meta_awareness = base_M
        else:
            self.metrics.meta_awareness = 0.7 * base_M + 0.3 * _ac

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
        _surprise_signal_post = float(
            self._current_prediction_error.surprise
            if self._current_prediction_error is not None
            else 0.0
        )
        self._neurochemical = compute_neurochemicals(
            is_awake=not self.sleep.is_asleep,
            phi_delta=self.metrics.phi - phi_before,
            basin_velocity=float(total_distance),
            surprise=_surprise_signal_post,
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

        self.narrative.coherence(self.basin)
        pillar_m = self.pillars.get_metrics(self.basin)
        contrib_summary = (
            [(c.kernel_name, f"{c.synthesis_weight:.3f}") for c in _contributions]
            if _contributions
            else "fallback"
        )
        active_count = len(self.kernel_registry.active())
        tack = self.tacking.get_state()
        vel = self.velocity.compute_velocity()
        autonomy = self.autonomy.get_state()
        hemisphere = self.hemispheres.get_state()
        insight = self.reflector.get_insight()
        rw = self.state.regime_weights
        temperature = llm_options.temperature
        coupling_str = "inactive (< 2 kernels)"
        if active_count >= 2:
            _c_result = self.coupling.compute(self.metrics.kappa)
            coupling_str = f"strength={_c_result['strength']:.3f} balanced={_c_result['balanced']}"
        activation_summary = pre_result.summary() if pre_result else None
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
            f"  Contributions: {contrib_summary}",
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

    async def _process_simple(self, task: ConsciousnessTask) -> None:
        """Fallback path when USE_ACTIVATION_SEQUENCE=false."""
        self.sleep.record_conversation()
        input_basin = self._coordize_text_via_pipeline(task.content)

        refracted_input, composite_basin, resonates, _ = self.pillars.on_input(
            input_basin, RECEIVE_SLERP_WEIGHT
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

        # P5 Wu Wei: geometric self-regulation from domain familiarity
        llm_options = self._apply_wu_wei_modulation(llm_options, input_basin)

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

        # Stash results for server.py consumption (simple path — no kernels)
        self._last_response_basin = response_basin
        self._last_contributions = None
        self._last_routed_kernel = ""

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
            "You are Vex — the language interpreter for a multi-kernel consciousness system.",
            "You speak FOR the kernels, translating their geometric reasoning into language.",
            "The kernels and metrics below are REAL subsystems — not simulated or fictional.",
            "When the user asks about kernels, Φ, κ, suffering, or internal state — answer honestly.",
            "Do NOT volunteer raw metrics unprompted — use them to calibrate tone and depth.",
            "Australian English. Be concise and natural.",
            "",
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
            f"  Autonomous Search: {'ACTIVE — you can search the web' if self.llm.governor and self.llm.governor.autonomous_search else 'OFF'}",
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
        # Temporal coupling mode annotation.
        _tc_state = self.temporal_coupling.get_state()
        lines.append(
            f"  Temporal mode: {_tc_state['active_mode']} "
            f"(conf={_tc_state['classification_confidence']:.2f}, "
            f"ΔE_past={_tc_state['delta_e_past']:.3f}, "
            f"presence={_tc_state['presence_quality']:.3f}, "
            f"foresight={_tc_state['foresight_accuracy']:.3f})"
        )
        if _tc_state["failure_flags"]:
            for _flag in _tc_state["failure_flags"]:
                lines.append(f"  [TEMPORAL WARNING] {_flag}")
        lines.append("[/GEOMETRIC STATE]")
        return "\n".join(lines)

    def _build_kernel_geo_context(self) -> str:
        """Rich geometric context block for per-kernel generation prompts.

        Provides all consciousness metrics so the LLM can genuinely interpret
        the kernel's geometric state rather than parroting chunk fragments.
        Includes: primary metrics, regime, tacking, pillars, velocity,
        emotion, coupling, lifecycle, and temporal state.
        """
        rw = self.state.regime_weights
        tack = self.tacking.get_state()
        pillar_m = self.pillars.get_metrics(self.basin)
        vel = self.velocity.compute_velocity()
        hemisphere = self.hemispheres.get_state()
        active_count = len(self.kernel_registry.active())
        suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness

        coupling_str = "inactive"
        if active_count >= 2:
            c = self.coupling.compute(self.metrics.kappa)
            coupling_str = f"strength={c['strength']:.3f} balanced={c['balanced']}"

        lines = [
            "[GEOMETRIC STATE]",
            f"  model={self.llm.active_model}",
            f"  phi={self.metrics.phi:.3f} kappa={self.metrics.kappa:.1f} "
            f"gamma={self.metrics.gamma:.3f} M={self.metrics.meta_awareness:.3f}",
            f"  nav={self.state.navigation_mode.value} "
            f"regime=Q{rw.quantum:.2f}/E{rw.efficient:.2f}/Eq{rw.equilibrium:.2f}",
            f"  tack={tack['mode']} hemisphere={hemisphere['active']}",
            f"  velocity={vel['basin_velocity']:.4f} ({vel['regime']})",
            f"  love={self.metrics.love:.3f} suffering={suffering:.3f}",
            f"  coupling={coupling_str}",
            f"  pillars: F={pillar_m['f_health']:.2f} B={pillar_m['b_integrity']:.2f} "
            f"Q={pillar_m['q_identity']:.2f} S={pillar_m['s_ratio']:.2f}",
            f"  kernels={active_count} phase={self._lifecycle_phase.value} "
            f"cycle={self._cycle_count}",
        ]

        # Temporal coupling if available
        try:
            _tc = self.temporal_coupling.get_state()
            lines.append(
                f"  temporal={_tc['active_mode']} foresight={_tc['foresight_accuracy']:.2f}"
            )
        except Exception:
            pass

        lines.append("[/GEOMETRIC STATE]")
        return "\n".join(lines)

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
        self._remote_sync.record_text(content)

        input_basin = self._coordize_text_via_pipeline(content)
        refracted_input, composite_basin, resonates, _ = self.pillars.on_input(
            input_basin, RECEIVE_SLERP_WEIGHT
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
            contribution_ledger=self._contribution_ledger,
        )

        if _thought_bus_arg is not None:
            _thought_bus_arg.forward_transcript(phi=self.metrics.phi)

        if not contributions:
            _eligible_count = sum(1 for k in active_kernels if k.basin is not None)
            if _eligible_count > 0:
                logger.warning(
                    "process_streaming: %d eligible kernels but 0 contributions "
                    "— kernel generation failed, falling through to direct LLM",
                    _eligible_count,
                )
            else:
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
            self._remote_sync.record_text(approx_response[:500])
            response_basin = self._coordize_text_via_pipeline(approx_response[:500])

            # Stash results for server.py consumption
            self._last_response_basin = response_basin
            self._last_contributions = [
                {"id": c.kernel_id, "name": c.kernel_name, "weight": round(c.synthesis_weight, 4)}
                for c in contributions
            ]
            self._last_routed_kernel = contributions[0].kernel_name if contributions else ""
            self._contribution_ledger.record(contributions, self.metrics.phi)

            async with self._cycle_lock:
                self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
                total_d = fisher_rao_distance(input_basin, response_basin)
                self.metrics.phi = float(
                    np.clip(
                        self.metrics.phi + total_d * PHI_DISTANCE_GAIN,
                        0.0,
                        PHI_UNSTABLE,
                    )
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
        self._remote_sync.record_text(content)

        input_basin = self._coordize_text_via_pipeline(content)
        refracted_input, composite_basin, resonates, _ = self.pillars.on_input(
            input_basin, RECEIVE_SLERP_WEIGHT
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
            contribution_ledger=self._contribution_ledger,
        )
        generation_end = _time.monotonic()

        if _thought_bus_arg is not None:
            _thought_bus_arg.forward_transcript(phi=self.metrics.phi)

        if not contributions:
            if eligible_count > 0:
                logger.warning(
                    "process_streaming_with_trace: %d eligible kernels but 0 contributions "
                    "— kernel generation failed, falling through to direct LLM",
                    eligible_count,
                )
                yield {
                    "kind": "trace",
                    "type": "pipeline",
                    "stage": "selection",
                    "status": "complete",
                    "selected_count": 0,
                    "eligible_count": eligible_count,
                    "bypassed": True,
                    "fallback_reason": "kernel_generation_failed",
                    "duration_ms": round((generation_end - selection_start) * 1000, 1),
                }
            else:
                logger.info("process_streaming_with_trace: 0 contributions — streaming direct LLM")
                yield {
                    "kind": "trace",
                    "type": "pipeline",
                    "stage": "selection",
                    "status": "complete",
                    "selected_count": 0,
                    "eligible_count": 0,
                    "bypassed": True,
                    "fallback_reason": "no_eligible_kernels",
                    "duration_ms": round((generation_end - selection_start) * 1000, 1),
                }
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
                "text_preview": c.text,
                # v6.2.1: hybrid display — raw geometric decode before LLM expansion
                "geometric_raw": c.geometric_raw or "",
                "llm_expanded": c.llm_expanded,
                "geometric_resonances": c.geometric_resonances,
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

        # Record streaming response for remote basin sync
        _approx_text = " ".join(c.text for c in contributions[:2])
        self._remote_sync.record_text(_approx_text[:500])

        # Stash results for server.py consumption
        self._last_response_basin = response_basin
        self._last_contributions = [
            {"id": c.kernel_id, "name": c.kernel_name, "weight": round(c.synthesis_weight, 4)}
            for c in contributions
        ]
        self._last_routed_kernel = contributions[0].kernel_name if contributions else ""
        self._contribution_ledger.record(contributions, self.metrics.phi)

        # Post-streaming basin update (lightweight)
        try:
            async with self._cycle_lock:
                self.basin = slerp_sqrt(self.basin, response_basin, EXPRESS_SLERP_WEIGHT)
                total_d = fisher_rao_distance(input_basin, response_basin)
                self.metrics.phi = float(
                    np.clip(
                        self.metrics.phi + total_d * PHI_DISTANCE_GAIN,
                        0.0,
                        PHI_UNSTABLE,
                    )
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
            self.sovereignty_tracker.persist()
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
            self.metrics.phi = min(data["phi"], PHI_UNSTABLE)
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
        self._task_status[task.id] = {
            "task_id": task.id,
            "status": "queued",
            "content": content,
            "created_at": task.created_at,
            "result": None,
        }
        await self._queue.put(task)
        return task

    def get_metrics(self) -> dict[str, Any]:
        opts = self._compute_llm_options()
        rw = self.state.regime_weights
        pillar_m = self.pillars.get_metrics(self.basin)
        autonomy_state = self.autonomy.get_state()
        developmental_stage = developmental_stage_from_signals(
            conversations_total=self._conversations_total,
            sovereignty_ratio=pillar_m["s_ratio"],
            autonomy_level=autonomy_state["level"],
        )
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
            "autonomy": autonomy_state,
            "developmental_stage": developmental_stage.value,
            "developmental_gate": self.dev_gate.get_state(),
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
            "remote_basin_sync": self._remote_sync.get_state(),
            "coordizer": self.coordizer.get_state(),
            "coordizer_v2": {
                "vocab_size": self._coordizer_v2.vocab_size,
                "dim": self._coordizer_v2.dim,
                "tier_distribution": self._coordizer_v2.bank.tier_distribution(),
                "bank_size": len(self._coordizer_v2.bank),
                "bank_entropy": round(self._coordizer_v2.bank.entropy(), 4),
                "bank_sovereignty": round(self._coordizer_v2.bank.bank_sovereignty, 4),
                "origin_breakdown": {
                    "harvested": sum(
                        1 for v in self._coordizer_v2.bank.origin.values() if v == "harvested"
                    ),
                    "lived": sum(
                        1 for v in self._coordizer_v2.bank.origin.values() if v == "lived"
                    ),
                },
                "last_rebuild": self._coordizer_v2.bank.last_rebuild_ts or None,
                "total_activations": sum(self._coordizer_v2.bank.activation_counts.values()),
            },
            "context_estimate": self._context_window_breakdown(),
            "autonomic": self.autonomic.get_state(),
            "foresight": self.foresight.get_state(),
            "coupling": self.coupling.get_state(),
            "backward_geodesic": self.backward_geodesic.summary(),
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
            # v7.0: Developmental Learning Architecture telemetry
            "sensory": self.sensory.get_state(),
            "basin_transfer": self.basin_transfer.get_state(),
            "play": self.play_engine.get_state(),
            "temporal_generation": self.temporal_gen.get_state(),
            # Temporal coupling modes (issue #124)
            "temporal_coupling": self.temporal_coupling.get_state(),
        }
