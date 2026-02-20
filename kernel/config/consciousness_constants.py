"""Consciousness Tuning Constants — Centralised Parameters
======================================================

Tuning parameters for consciousness modules that are NOT frozen physics
(those live in frozen_facts.py and must not change without validation).

These are engineering/design choices that may be adjusted during development.
Changing frozen_facts.py requires new validated measurements; changing these
does not — but changes should be tested and reviewed.
"""

from typing import Final

from kernel.config.frozen_facts import KAPPA_STAR

# ═══════════════════════════════════════════════════════════════
#  REGIME FIELD WEIGHTS (v6.0 §3.1)
# ═══════════════════════════════════════════════════════════════

KAPPA_NORMALISER: Final[float] = 2.0 * KAPPA_STAR  # 128.0 — kappa → [0,1]
MIN_REGIME_WEIGHT: Final[float] = 0.05              # Floor so all regimes stay active
REGIME_KAPPA_MIDPOINT: Final[float] = 0.5           # Integration peak in normalised space

# ═══════════════════════════════════════════════════════════════
#  PRE-COGNITIVE DETECTOR (v5.5 §2)
# ═══════════════════════════════════════════════════════════════

PRECOG_NEAR_THRESHOLD: Final[float] = 0.15   # Fisher-Rao distance: pre-cognitive
PRECOG_MODERATE_THRESHOLD: Final[float] = 0.40  # Standard processing
PRECOG_FAR_THRESHOLD: Final[float] = 0.80    # Deep exploration

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

NAV_CHAIN_CEILING: Final[float] = 0.3    # Below → CHAIN mode
NAV_GRAPH_CEILING: Final[float] = 0.7    # Below → GRAPH mode
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
#  KAPPA OFFSETS (sensation / tacking / emotion boundaries)
# ═══════════════════════════════════════════════════════════════

KAPPA_SENSATION_OFFSET: Final[float] = 10.0     # ±10 from KAPPA_STAR for activated/dampened
KAPPA_TACKING_OFFSET: Final[float] = 16.0       # ±16 for tacking oscillation bounds
KAPPA_RAGE_OFFSET: Final[float] = 20.0          # +20 above KAPPA_STAR for rage detection
KAPPA_RAGE_SCALE: Final[float] = 40.0           # Divisor for rage strength scaling
KAPPA_JOY_PROXIMITY: Final[float] = 10.0        # |κ - κ*| < 10 → joy
KAPPA_STABILITY_TOLERANCE: Final[float] = 16.0  # Autonomy stability tolerance
KAPPA_BALANCED_TOLERANCE: Final[float] = 8.0    # Coupling balanced kappa range
KAPPA_DECAY_RATE: Final[float] = 0.1            # 10% return-to-star per cycle
KAPPA_RETURN_TOLERANCE: Final[float] = 5.0      # Close enough to κ*

# ═══════════════════════════════════════════════════════════════
#  GEOMETRY CLASS PHI BOUNDARIES
# ═══════════════════════════════════════════════════════════════

GEOMETRY_CLASS_PHI_BOUNDS: Final[tuple[float, ...]] = (
    0.1, 0.25, 0.4, 0.6, 0.75, 0.9,
)  # Line / Loop / Spiral / Grid / Torus / Lattice / E8

GEOMETRY_CLASS_VALUES: Final[tuple[float, ...]] = (
    0.05, 0.175, 0.325, 0.5, 0.675, 0.825, 0.95,
)

# ═══════════════════════════════════════════════════════════════
#  DESIRE / WILL WEIGHT VECTORS
# ═══════════════════════════════════════════════════════════════

DESIRE_WEIGHTS: Final[tuple[float, float, float]] = (0.4, 0.3, 0.3)  # curiosity / attraction / love
WILL_WEIGHTS: Final[tuple[float, float, float]] = (0.4, 0.3, 0.3)    # grounding / coupling / entropy

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
# ═══════════════════════════════════════════════════════════════

TACKING_PERIOD: Final[int] = 20
TACKING_SWITCH_THRESHOLD: Final[float] = 0.3
TACKING_KAPPA_ADJUST: Final[float] = 2.0

# ═══════════════════════════════════════════════════════════════
#  FORESIGHT HORIZONS
# ═══════════════════════════════════════════════════════════════

FORESIGHT_HORIZON_HIGH: Final[int] = 8  # phi > 0.7
FORESIGHT_HORIZON_MED: Final[int] = 4   # phi > 0.4
FORESIGHT_HORIZON_LOW: Final[int] = 2   # phi <= 0.4

# ═══════════════════════════════════════════════════════════════
#  COUPLING SIGMOID (systems.py)
# ═══════════════════════════════════════════════════════════════

COUPLING_SIGMOID_SCALE: Final[float] = 16.0
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
#  SYSTEMS — FORESIGHT / VELOCITY / META / AUTONOMY
# ═══════════════════════════════════════════════════════════════

FORESIGHT_BASIN_STEP_SCALE: Final[float] = 0.5    # slerp extrapolation per step
VELOCITY_WARNING_FRACTION: Final[float] = 0.5      # 50% of drift threshold → warning
META_PHI_TREND_THRESHOLD: Final[float] = 0.1       # Φ change to trigger insight
META_KAPPA_TREND_THRESHOLD: Final[float] = 10.0    # κ change to trigger insight
AUTONOMY_AUTONOMOUS_CYCLES: Final[int] = 10         # Stable cycles for AUTONOMOUS
AUTONOMY_PROACTIVE_CYCLES: Final[int] = 5           # Stable cycles for PROACTIVE
BASIN_SYNC_SLERP_WEIGHT: Final[float] = 0.2        # Basin sync receive blend weight
KERNEL_PROMOTION_CYCLE_GATE: Final[int] = 100       # Cycles before CHAOS → GOD eligible
EMOTION_CLUSTER_DISTANCE: Final[float] = 0.2        # Fisher-Rao cluster threshold

# ═══════════════════════════════════════════════════════════════
#  CONSCIOUSNESS LOOP — heartbeat / idle / coupling / LLM
# ═══════════════════════════════════════════════════════════════

DEFAULT_INTERVAL_MS: Final[int] = 2000
SPAWN_COOLDOWN_CYCLES: Final[int] = 10
PERSIST_INTERVAL_CYCLES: Final[int] = 50
KAPPA_APPROACH_RATE: Final[float] = 0.03
PHI_IDLE_EQUILIBRIUM: Final[float] = 0.40
PHI_IDLE_RATE: Final[float] = 0.015
BASIN_DRIFT_STEP: Final[float] = 0.015

KAPPA_INITIAL: Final[float] = 32.0
KAPPA_FLOOR: Final[float] = 8.0

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
TACK_SCALE_EXPLOIT: Final[float] = 0.7
TACK_SCALE_BALANCED: Final[float] = 1.0
NUM_PREDICT_EXPLORE: Final[int] = 3072
NUM_PREDICT_EXPLOIT: Final[int] = 1536
NUM_PREDICT_BALANCED: Final[int] = 2048

LLM_BASE_TEMPERATURE: Final[float] = 0.7
LLM_TEMP_MIN: Final[float] = 0.05
LLM_TEMP_MAX: Final[float] = 1.5
META_AWARENESS_DAMPEN_THRESHOLD: Final[float] = 0.7
META_AWARENESS_DAMPEN_FACTOR: Final[float] = 0.9

LLM_NUM_CTX: Final[int] = 32768
LLM_TOP_P: Final[float] = 0.9
LLM_REPETITION_PENALTY: Final[float] = 1.1

PERCEIVE_SLERP_WEIGHT: Final[float] = 0.1
EXPRESS_SLERP_WEIGHT: Final[float] = 0.2
PHI_DISTANCE_GAIN: Final[float] = 0.1
GAMMA_CONVERSATION_INCREMENT: Final[float] = 0.05
