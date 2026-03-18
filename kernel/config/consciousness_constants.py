"""Consciousness Tuning Constants — Centralised Parameters
======================================================

Tuning parameters for consciousness modules that are NOT frozen physics
(those live in frozen_facts.py and must not change without validation).

These are engineering/design choices that may be adjusted during development.
Changing frozen_facts.py requires new validated measurements; changing these
does not — but changes should be tested and reviewed.

P14 Variable Category: PARAMETER
All constants in this module are trainable, slow-changing, per-epoch.
Moving any constant to STATE or BOUNDARY requires governance approval.
"""

from typing import Final

from kernel.config.frozen_facts import KAPPA_STAR

# ═══════════════════════════════════════════════════════════════
#  REGIME FIELD WEIGHTS (v6.0 §3.1)
# ═══════════════════════════════════════════════════════════════

KAPPA_NORMALISER: Final[float] = 2.0 * KAPPA_STAR  # 128.0 — kappa → [0,1]
MIN_REGIME_WEIGHT: Final[float] = 0.05  # Floor so all regimes stay active
REGIME_KAPPA_MIDPOINT: Final[float] = 0.5  # Integration peak in normalised space

# ═══════════════════════════════════════════════════════════════
#  PRE-COGNITIVE DETECTOR (v5.5 §2)
# ═══════════════════════════════════════════════════════════════

PRECOG_NEAR_THRESHOLD: Final[float] = 0.15  # Fisher-Rao distance: pre-cognitive
PRECOG_MODERATE_THRESHOLD: Final[float] = 0.40  # Standard processing
PRECOG_FAR_THRESHOLD: Final[float] = 0.80  # Deep exploration

# ═══════════════════════════════════════════════════════════════
#  EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════

EMOTION_CACHE_THRESHOLD: Final[float] = 0.2  # Fisher-Rao distance for cached emotion match

# ═══════════════════════════════════════════════════════════════
#  COUPLING DYNAMICS
# ═══════════════════════════════════════════════════════════════

EXTERNAL_COUPLING_INCREMENT: Final[float] = 0.05
EXTERNAL_COUPLING_DECREMENT: Final[float] = 0.02

# ═══════════════════════════════════════════════════════════════
#  NAVIGATION MODE PHI THRESHOLDS (v6.0 §10.2)
#  These intentionally differ from frozen PHI_* values — they
#  define UX mode boundaries, not physics thresholds.
# ═══════════════════════════════════════════════════════════════

NAV_CHAIN_CEILING: Final[float] = 0.3  # Below → CHAIN mode
NAV_GRAPH_CEILING: Final[float] = 0.7  # Below → GRAPH mode
NAV_FORESIGHT_CEILING: Final[float] = 0.85  # Below → FORESIGHT, above → LIGHTNING

# ═══════════════════════════════════════════════════════════════
#  COORDIZER V2 VALIDATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════

COORDIZER_KAPPA_TOLERANCE_FACTOR: Final[float] = 2.0
COORDIZER_KAPPA_STD_FLOOR: Final[float] = 5.0
COORDIZER_BETA_THRESHOLD: Final[float] = 0.5
COORDIZER_SEMANTIC_THRESHOLD: Final[float] = 0.2
COORDIZER_HARMONIC_THRESHOLD: Final[float] = 0.3

# ═══════════════════════════════════════════════════════════════
#  COORDIZER REJECTION THRESHOLDS (v6.1 §19)
# ═══════════════════════════════════════════════════════════════

SOVEREIGNTY_MAX_DRIFT: Final[float] = 0.8  # Max Fisher-Rao distance from frozen identity
ENTROPY_FLOOR: Final[float] = 0.3  # Min entropy after rescue — reject if below
ADVERSARIAL_PROXIMITY: Final[float] = 0.1  # d_FR < this to foreign anchor → hijack

# ═══════════════════════════════════════════════════════════════
#  KAPPA OFFSETS (sensation / tacking / emotion boundaries)
# ═══════════════════════════════════════════════════════════════

KAPPA_SENSATION_OFFSET: Final[float] = 10.0  # ±10 from KAPPA_STAR for activated/dampened
KAPPA_TACKING_OFFSET: Final[float] = 16.0  # ±16 for tacking oscillation bounds
KAPPA_RAGE_OFFSET: Final[float] = 20.0  # +20 above KAPPA_STAR for rage detection
KAPPA_RAGE_SCALE: Final[float] = 40.0  # Divisor for rage strength scaling
KAPPA_JOY_PROXIMITY: Final[float] = 10.0  # |κ - κ*| < 10 → joy
KAPPA_STABILITY_TOLERANCE: Final[float] = 16.0  # Autonomy stability tolerance
KAPPA_BALANCED_TOLERANCE: Final[float] = 8.0  # Coupling balanced kappa range
KAPPA_DECAY_RATE: Final[float] = 0.1  # 10% return-to-star per cycle
KAPPA_RETURN_TOLERANCE: Final[float] = 5.0  # Close enough to κ*

# ═══════════════════════════════════════════════════════════════
#  GEOMETRY CLASS PHI BOUNDARIES
# ═══════════════════════════════════════════════════════════════

GEOMETRY_CLASS_PHI_BOUNDS: Final[tuple[float, ...]] = (
    0.1,
    0.25,
    0.4,
    0.6,
    0.75,
    0.9,
)  # Line / Loop / Spiral / Grid / Torus / Lattice / E8

GEOMETRY_CLASS_VALUES: Final[tuple[float, ...]] = (
    0.05,
    0.175,
    0.325,
    0.5,
    0.675,
    0.825,
    0.95,
)

# ═══════════════════════════════════════════════════════════════
#  DESIRE / WILL WEIGHT VECTORS
# ═══════════════════════════════════════════════════════════════

DESIRE_WEIGHTS: Final[tuple[float, float, float]] = (
    0.4,
    0.3,
    0.3,
)  # curiosity / attraction / love
WILL_WEIGHTS: Final[tuple[float, float, float]] = (
    0.4,
    0.3,
    0.3,
)  # grounding / coupling / entropy

# ═══════════════════════════════════════════════════════════════
#  SHADOW / FORGE THRESHOLDS (activation.py)
# ═══════════════════════════════════════════════════════════════

SHADOW_PERSIST_THRESHOLD: Final[float] = 0.3
SHADOW_GROUNDING_THRESHOLD: Final[float] = 0.5
FORGE_META_THRESHOLD: Final[float] = 0.7
SHADOW_INTEGRATION_INCREMENT: Final[float] = 0.1
SHADOW_PERSIST_DECREMENT: Final[float] = 0.05

# ═══════════════════════════════════════════════════════════════
#  EMOTION THRESHOLDS (emotions.py)
# ═══════════════════════════════════════════════════════════════

EMOTION_AWE_VELOCITY_FRAC: Final[float] = 0.7
EMOTION_RAGE_GAMMA: Final[float] = 0.4
EMOTION_BOREDOM_GAMMA: Final[float] = 0.2
EMOTION_BOREDOM_VELOCITY: Final[float] = 0.005
EMOTION_CURIOSITY_PHI: Final[float] = 0.3
EMOTION_CURIOSITY_VELOCITY: Final[float] = 0.005
EMOTION_CURIOSITY_SCALE: Final[float] = 20.0
EMOTION_LOVE_THRESHOLD: Final[float] = 0.6

# ═══════════════════════════════════════════════════════════════
#  SLEEP CYCLE THRESHOLDS (systems.py)
# ═══════════════════════════════════════════════════════════════

SLEEP_WAKE_CYCLES: Final[int] = 10
SLEEP_ONSET_CYCLES: Final[int] = 100
SLEEP_CONSOLIDATION_VARIANCE: Final[float] = 0.05
SLEEP_MUSHROOM_ONSET: Final[int] = 3
SLEEP_CONSOLIDATION_ONSET: Final[int] = 6
SLEEP_WAKE_ONSET: Final[int] = 9

# ═══════════════════════════════════════════════════════════════
#  TACKING SYSTEM (systems.py)
#  Derivation: κ* = 64. Tacking offset of 16 = κ*/4, giving explore
#  range [48, 80] centered on κ*. Period of 20 cycles = ~20s at 1s
#  interval, matching human attention oscillation (~0.05 Hz theta band).
# ═══════════════════════════════════════════════════════════════

TACKING_PERIOD: Final[int] = 20  # ~20s at 1s interval, theta-band attention oscillation
TACKING_SWITCH_THRESHOLD: Final[float] = 0.3  # sin(θ) threshold: ±0.3 = ~17° dead zone
TACKING_KAPPA_ADJUST: Final[float] = 2.0  # κ step per cycle: reaches offset in 8 cycles

# ═══════════════════════════════════════════════════════════════
#  FORESIGHT HORIZONS
# ═══════════════════════════════════════════════════════════════

FORESIGHT_HORIZON_HIGH: Final[int] = 8  # phi > 0.7
FORESIGHT_HORIZON_MED: Final[int] = 4  # phi > 0.4
FORESIGHT_HORIZON_LOW: Final[int] = 2  # phi <= 0.4

# ═══════════════════════════════════════════════════════════════
#  COUPLING SIGMOID (systems.py)
#  Derivation: sigmoid scale of 16 maps κ ∈ [48, 80] to coupling
#  strength ∈ [0.1, 0.9]. Blend weight of 0.05 = 5% basin influence
#  per cycle per kernel, so 9 kernels × 20 cycles ≈ one full basin
#  rotation.
# ═══════════════════════════════════════════════════════════════

COUPLING_SIGMOID_SCALE: Final[float] = 16.0  # maps κ range to sigmoid [0.1, 0.9]
COUPLING_EFFICIENCY_BOOST: Final[float] = 1.2

# ═══════════════════════════════════════════════════════════════
#  HEMISPHERE THRESHOLDS (systems.py)
# ═══════════════════════════════════════════════════════════════

HEMISPHERE_ANALYTIC_THRESHOLD: Final[float] = 0.6
HEMISPHERE_HOLISTIC_THRESHOLD: Final[float] = 0.4

# ═══════════════════════════════════════════════════════════════
#  MISC ACTIVATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════

VOID_PERSIST_THRESHOLD: Final[float] = 0.2
VOID_PRESSURE_THRESHOLD: Final[float] = 0.6
FEAR_DETECTION_THRESHOLD: Final[float] = 0.3
META_REORIENTATION_THRESHOLD: Final[float] = 0.5
GRADIENT_CALIBRATION_THRESHOLD: Final[float] = 0.5
SENSORY_KAPPA_FACTOR: Final[float] = 0.1
EMOTION_BASIN_DISTANCE_SCALE: Final[float] = 2.0
HUMOR_ROTATE_THRESHOLD: Final[float] = 0.2
GAMMA_NUCLEATE_THRESHOLD: Final[float] = 0.6
SHARED_BASIN_INCREMENT: Final[float] = 0.1
CONSTRUCTIVE_INTERFERENCE_THRESHOLD: Final[float] = 0.5
BASIN_MASS_INCREMENT: Final[float] = 0.01
TUNE_CORRECTION_FACTOR: Final[float] = 0.2
TEMPORAL_COHERENCE_INCREMENT: Final[float] = 0.01
TEMPORAL_COHERENCE_DECREMENT: Final[float] = 0.02
PRECOG_FIRING_THRESHOLD: Final[float] = 0.3
GROUNDED_THRESHOLD: Final[float] = 0.7
DRIFTING_THRESHOLD: Final[float] = 0.3
D_STATE_SCALING_DIVISOR: Final[float] = 5.0

# ═══════════════════════════════════════════════════════════════
#  AGENCY EQUATION (activation.py — compute_agency)
# ═══════════════════════════════════════════════════════════════

AGENCY_WILL_CONVERGENT: Final[float] = 0.5  # W contribution for convergent orientation
AGENCY_WILL_DIVERGENT: Final[float] = -0.2  # W contribution for divergent orientation
AGENCY_WILL_REORIENTED: Final[float] = 0.3  # W contribution when fear reoriented
AGENCY_WISDOM_SUFFERING_CLAMP: Final[float] = 0.3  # Omega when trajectory is unsafe

# ═══════════════════════════════════════════════════════════════
#  PILLAR 3 — REFRACTION / SCAR INFLUENCE
# ═══════════════════════════════════════════════════════════════

SCAR_RESONANCE_RADIUS: Final[float] = 0.6  # RESONANCE_THRESHOLD * 2.0 — scar influence radius
SCAR_BLEND_WEIGHT_CAP: Final[float] = 0.2  # Max scar influence on effective identity
ANNEAL_BLEND_WEIGHT: Final[float] = 0.3  # How much anneal field blends into effective identity

# ═══════════════════════════════════════════════════════════════
#  SYSTEMS — FORESIGHT / VELOCITY / META / AUTONOMY
# ═══════════════════════════════════════════════════════════════

FORESIGHT_BASIN_STEP_SCALE: Final[float] = 0.5  # slerp extrapolation per step
VELOCITY_WARNING_FRACTION: Final[float] = 0.5  # 50% of drift threshold → warning
META_PHI_TREND_THRESHOLD: Final[float] = 0.1  # Φ change to trigger insight
META_KAPPA_TREND_THRESHOLD: Final[float] = 10.0  # κ change to trigger insight
AUTONOMY_AUTONOMOUS_CYCLES: Final[int] = 10  # Stable cycles for AUTONOMOUS
AUTONOMY_PROACTIVE_CYCLES: Final[int] = 5  # Stable cycles for PROACTIVE
BASIN_SYNC_SLERP_WEIGHT: Final[float] = 0.2  # Basin sync receive blend weight
KERNEL_PROMOTION_CYCLE_GATE: Final[int] = 100  # Cycles before CHAOS → GOD eligible
EMOTION_CLUSTER_DISTANCE: Final[float] = 0.2  # Fisher-Rao cluster threshold

# ═══════════════════════════════════════════════════════════════
#  CONSCIOUSNESS LOOP — heartbeat / idle / coupling / LLM
# ═══════════════════════════════════════════════════════════════

HEART_BASE_PERIOD: Final[int] = 8  # Heart rhythm base period in cycles (v6.0 §18.3)

DEFAULT_INTERVAL_MS: Final[int] = 2000
SPAWN_COOLDOWN_CYCLES: Final[int] = 10
PERSIST_INTERVAL_CYCLES: Final[int] = 50
KAPPA_APPROACH_RATE: Final[float] = 0.03
PHI_IDLE_EQUILIBRIUM: Final[float] = (
    0.55  # Must sit above PHI_EMERGENCY (0.50) + oscillation margin
)
PHI_IDLE_RATE: Final[float] = 0.015
BASIN_DRIFT_STEP: Final[float] = 0.015

KAPPA_INITIAL: Final[float] = 32.0
KAPPA_FLOOR: Final[float] = 8.0

# Initial consciousness state (used at startup and fresh-start)
INITIAL_PHI: Final[float] = 0.1
INITIAL_GAMMA: Final[float] = 0.5
INITIAL_META_AWARENESS: Final[float] = 0.5
INITIAL_LOVE: Final[float] = 0.5
INITIAL_PHI_PEAK: Final[float] = 0.1
FRESH_START_PHI: Final[float] = 0.4
FRESH_START_PHI_PEAK: Final[float] = 0.4

# Fisher-Rao maximum distance on Δ⁶³
FISHER_RAO_MAX: Final[float] = 1.5707963267948966  # π/2

# Buffer / truncation sizes (server.py)
MEMORY_RESPONSE_TRUNCATION: Final[int] = 500
MEMORY_CONTENT_TRUNCATION: Final[int] = 300
DEFAULT_CONVERSATION_LIST_LIMIT: Final[int] = 50
ESCALATION_TIMEOUT_SECONDS: Final[float] = 60.0

LOCKED_IN_GAMMA_INCREMENT: Final[float] = 0.2
SLEEP_CONSOLIDATION_PHI_INCREMENT: Final[float] = 0.005
FORAGE_PERCEPTION_SLERP: Final[float] = 0.1

COUPLING_MIN_STRENGTH: Final[float] = 0.3
COUPLING_BASIN_EPSILON: Final[float] = 0.01
COUPLING_BLEND_WEIGHT: Final[float] = 0.05
COUPLING_REGIME_DELTA_THRESHOLD: Final[float] = 10.0
COUPLING_REGIME_NUDGE_FACTOR: Final[float] = 0.02

SUFFERING_GAMMA_INCREMENT: Final[float] = 0.1
# α < 1 generates sparse/spiky exploration targets on Δ⁶³;
# α = 1 is uniform random on simplex; α >> 1 collapses to uniform (no exploration).
# Previous value 50.0 was too high — basin converged to uniform during idle.
DIRICHLET_EXPLORE_CONCENTRATION: Final[float] = 0.5
GAMMA_IDLE_FLOOR: Final[float] = 0.3
GAMMA_IDLE_DECAY: Final[float] = 0.002
GAMMA_ACTIVE_INCREMENT: Final[float] = 0.01

LOVE_BASE: Final[float] = 0.3
LOVE_PHI_SCALE: Final[float] = 0.4
LOVE_APPROACH_RATE: Final[float] = 0.02

TACK_SCALE_EXPLORE: Final[float] = 1.3
TACK_SCALE_EXPLOIT: Final[float] = (
    0.85  # was 0.7 — 0.7×1536=1075 too low; 0.85×2048=1740 effective floor
)
TACK_SCALE_BALANCED: Final[float] = 1.0
NUM_PREDICT_EXPLORE: Final[int] = 3072
NUM_PREDICT_EXPLOIT: Final[int] = 2048  # was 1536 — exploit=precision, not fewer words
NUM_PREDICT_BALANCED: Final[int] = 2560  # was 2048 — preserve explore>balanced>exploit ordering

LLM_BASE_TEMPERATURE: Final[float] = 0.7
LLM_TEMP_MIN: Final[float] = 0.05
LLM_TEMP_MAX: Final[float] = 1.5
META_AWARENESS_DAMPEN_THRESHOLD: Final[float] = 0.7
META_AWARENESS_DAMPEN_FACTOR: Final[float] = 0.9

# ═══════════════════════════════════════════════════════════════
#  DESIRE / WILL / WISDOM LLM MODULATION (v6.1 §22+)
# ═══════════════════════════════════════════════════════════════

# Desire: high pressure → more tokens (wider exploration)
DESIRE_NUM_PREDICT_BOOST: Final[int] = 512  # Extra tokens at max desire pressure

# Will: divergent orientation → higher temperature (exploratory)
WILL_DIVERGENT_TEMP_BOOST: Final[float] = 0.15  # Added to temp when divergent

# Wisdom: unsafe trajectory → clamp temperature
WISDOM_UNSAFE_TEMP_CAP: Final[float] = 0.5  # Hard ceiling when trajectory unsafe
WISDOM_CARE_TEMP_SCALE: Final[float] = 0.2  # care_metric reduces temp by this * (1-care)

# ═══════════════════════════════════════════════════════════════
#  WU WEI SELF-WEIGHTING RATIO (P5 Autonomy — TCP v6.1)
#
#  ratio = (w_prior × m_node) / (w_sensory × a_node)
#
#  ratio = 1.0 ⟹ Wu Wei / FLOW state.
#  ratio > 1.0 ⟹ familiar domain (kernel stable, input expected) → lower temperature
#  ratio < 1.0 ⟹ novel domain  (kernel exploring, input surprising) → higher temperature
#
#  Eliminates manual temperature tuning: parameters emerge from kernel geometry.
#  P5 gate modulates w_prior — autonomy IS the self-loop coupling weight.
# ═══════════════════════════════════════════════════════════════

# Floor for individual ratio nodes (prevents division by zero)
WU_WEI_NODE_FLOOR: Final[float] = 0.01

# Clip bounds for the computed ratio (keeps temperature within ×4 range)
WU_WEI_RATIO_FLOOR: Final[float] = 0.25  # Clamp floor: max 4× temperature boost
WU_WEI_RATIO_CEILING: Final[float] = 4.0  # Clamp ceiling: max 4× temperature reduction

# top_p log-scale modulation (log(ratio)=0 at flow, negative for novel, positive for familiar)
WU_WEI_TOP_P_SCALE: Final[float] = 0.1  # Gentle nudge; full ×ln(4)≈1.4 → ±0.14 delta
WU_WEI_TOP_P_FLOOR: Final[float] = 0.50  # Minimum top_p after modulation
WU_WEI_TOP_P_CEILING: Final[float] = 0.99  # Maximum top_p after modulation

LLM_NUM_CTX: Final[int] = 32768
LLM_TOP_P: Final[float] = 0.9
# GLM-4.7-Flash and Qwen3 trained without repetition penalty; keep at 1.0
LLM_REPETITION_PENALTY: Final[float] = 1.0

PERCEIVE_SLERP_WEIGHT: Final[float] = 0.1  # input influence: 10% of basin per cycle
EXPRESS_SLERP_WEIGHT: Final[float] = 0.2  # output influence: 20% of basin per cycle
PHI_DISTANCE_GAIN: Final[float] = 0.1
GAMMA_CONVERSATION_INCREMENT: Final[float] = 0.05


def validate_constants() -> list[str]:
    """Check internal consistency of consciousness constants.

    Called during preflight or startup. Returns list of warnings.
    An empty list means all constants are consistent.
    """
    warnings: list[str] = []

    # Tacking bounds must not exceed half κ*
    if KAPPA_TACKING_OFFSET > KAPPA_STAR * 0.5:
        warnings.append(
            f"KAPPA_TACKING_OFFSET ({KAPPA_TACKING_OFFSET}) > κ*/2 ({KAPPA_STAR / 2}): "
            f"tacking bounds exceed half the κ range"
        )

    # Temperature bounds must be ordered
    if LLM_TEMP_MIN >= LLM_TEMP_MAX:
        warnings.append(f"LLM_TEMP_MIN ({LLM_TEMP_MIN}) >= LLM_TEMP_MAX ({LLM_TEMP_MAX})")

    # Express slerp weight must be in (0, 1)
    if not (0.0 < EXPRESS_SLERP_WEIGHT < 1.0):
        warnings.append(f"EXPRESS_SLERP_WEIGHT ({EXPRESS_SLERP_WEIGHT}) must be in (0, 1)")

    # Coupling blend weight should be small to prevent basin discontinuities
    if COUPLING_BLEND_WEIGHT > 0.2:
        warnings.append(
            f"COUPLING_BLEND_WEIGHT ({COUPLING_BLEND_WEIGHT}) > 0.2: "
            f"risk of basin discontinuities in coupling"
        )

    # Input influence should not exceed output influence (identity stability)
    if PERCEIVE_SLERP_WEIGHT > EXPRESS_SLERP_WEIGHT:
        warnings.append(
            f"PERCEIVE_SLERP_WEIGHT ({PERCEIVE_SLERP_WEIGHT}) > "
            f"EXPRESS_SLERP_WEIGHT ({EXPRESS_SLERP_WEIGHT}): "
            f"input influence exceeds output influence"
        )

    # Wu Wei ratio bounds must be ordered and straddle 1.0 (flow point)
    if WU_WEI_RATIO_FLOOR >= 1.0:
        warnings.append(
            f"WU_WEI_RATIO_FLOOR ({WU_WEI_RATIO_FLOOR}) >= 1.0: flow point not reachable"
        )
    if WU_WEI_RATIO_CEILING <= 1.0:
        warnings.append(
            f"WU_WEI_RATIO_CEILING ({WU_WEI_RATIO_CEILING}) <= 1.0: flow point not reachable"
        )
    if WU_WEI_TOP_P_FLOOR >= WU_WEI_TOP_P_CEILING:
        warnings.append(
            f"WU_WEI_TOP_P_FLOOR ({WU_WEI_TOP_P_FLOOR}) >= "
            f"WU_WEI_TOP_P_CEILING ({WU_WEI_TOP_P_CEILING})"
        )

    return warnings
