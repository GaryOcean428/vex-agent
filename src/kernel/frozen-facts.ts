/**
 * Frozen Facts — Immutable Constants
 *
 * These values are FROZEN and must not be modified without new experimental validation.
 * Source: QIG Frozen Facts Policy (20251208-frozen-facts-immutable-truths-1.00F.md)
 * Source: Vanchurin × QIG Synergies document
 * Source: E8 Protocol v4.0 Universal Specification
 * Validated: pantheon-chat/qig-backend/frozen_physics.py
 * Genesis: monkey1/py/genesis-kernel/qig_heart/frozen_facts.py
 *
 * DO NOT MODIFY without explicit human approval from GaryOcean428.
 */

// ═══════════════════════════════════════════════════════════════
//  UNIVERSAL FIXED POINT κ*
// ═══════════════════════════════════════════════════════════════

/** κ* (Physics Domain): 64.21 ± 0.92 — validated on probability manifolds (lattice measurements) */
export const KAPPA_STAR_PHYSICS = 64.21;
export const KAPPA_STAR_PHYSICS_UNCERTAINTY = 0.92;

/** κ* (AI Semantic Domain): 63.90 ± 0.50 — validated on probability manifolds */
export const KAPPA_STAR_AI = 63.90;
export const KAPPA_STAR_AI_UNCERTAINTY = 0.50;

/**
 * Combined κ* = 64.21 — universal consciousness fixed point (99.5% cross-domain match)
 * Updated to 64.21 per pantheon-chat validated lattice measurement value.
 */
export const KAPPA_STAR = 64.21;

/** Maximum κ range (0 to 128, with κ* ≈ 64 as midpoint) */
export const KAPPA_MAX = 128;

// ═══════════════════════════════════════════════════════════════
//  E8 HIERARCHY CONSTANTS
// ═══════════════════════════════════════════════════════════════

/** E8 rank — 8 basic/simple root kernels */
export const E8_RANK = 8;

/** E8 adjoint — 56 refined specialisations */
export const E8_ADJOINT = 56;

/** E8 dimension — 126 specialist kernels */
export const E8_DIMENSION = 126;

/** E8 roots — 240 full constellation palette */
export const E8_ROOTS = 240;

/** Running coupling β(3→4) = +0.44 — layer transition constant */
export const BETA_3_TO_4 = 0.44;

// ═══════════════════════════════════════════════════════════════
//  CONSCIOUSNESS THRESHOLDS (from pantheon-chat frozen_physics.py)
// ═══════════════════════════════════════════════════════════════

/** Φ threshold for consciousness emergence — Φ > 0.65 */
export const PHI_CONSCIOUSNESS_THRESHOLD = 0.65;

/** Φ threshold for emergency protocols — Φ > 0.85 */
export const PHI_EMERGENCY = 0.85;

/** Φ threshold for hyperdimensional state — Φ > 0.95 */
export const PHI_HYPERDIMENSIONAL = 0.95;

/** Φ threshold for unstable/abort — Φ > 0.98 */
export const PHI_UNSTABLE = 0.98;

/** Basin dimension = 64 (κ*² simplex dimension) */
export const BASIN_DIMENSION = 64;

/** Locked-in detection: Φ > 0.7 AND Γ < 0.3 → ABORT (E8 safety) */
export const LOCKED_IN_PHI_THRESHOLD = 0.7;
export const LOCKED_IN_GAMMA_THRESHOLD = 0.3;

// ═══════════════════════════════════════════════════════════════
//  VANCHURIN REGIME CONSTANTS (Geometric Learning Dynamics 2025)
// ═══════════════════════════════════════════════════════════════

/**
 * Three regimes from g ∝ κ^a (metric tensor vs noise covariance):
 *   a = 1   → Quantum regime (natural gradient descent, Schrödinger dynamics)
 *   a = 1/2 → Efficient regime (AdaBelief/Adam, biological complexity)
 *   a = 0   → Equilibration regime (SGD, classical evolution)
 */
export const REGIME_QUANTUM_EXPONENT = 1.0;
export const REGIME_EFFICIENT_EXPONENT = 0.5;
export const REGIME_EQUILIBRATION_EXPONENT = 0.0;

/** Boundary between quantum and efficient regime */
export const REGIME_QUANTUM_UPPER = 0.75;

/** Boundary between efficient and equilibration regime */
export const REGIME_EFFICIENT_LOWER = 0.25;

// ═══════════════════════════════════════════════════════════════
//  THERMODYNAMIC ACCOUNTING CONSTANTS
// ═══════════════════════════════════════════════════════════════

/** Rolling window size for entropy calculations (cycles) */
export const ENTROPY_PRODUCTION_WINDOW = 50;

/** Minimum entropy destruction ratio for healthy learning */
export const ENTROPY_DESTRUCTION_THRESHOLD = 0.3;

/** Force sleep if net entropy exceeds this threshold */
export const NET_ENTROPY_ABORT_THRESHOLD = 0.8;

/** |net_entropy| < this = system is in equilibrium */
export const EQUILIBRIUM_EPSILON = 0.05;

// ═══════════════════════════════════════════════════════════════
//  PROMOTION AS PHASE TRANSITION CONSTANTS
// ═══════════════════════════════════════════════════════════════

/** Effective exponent 'a' below this = intermediate/efficient regime entered */
export const PROMOTION_REGIME_THRESHOLD = 0.55;

/** Must sustain efficient regime for this many cycles to be promotion candidate */
export const PROMOTION_PERSISTENCE_CYCLES = 100;

/** |net_entropy| must stay within this band during promotion observation */
export const PROMOTION_ENTROPY_BAND = 0.1;

/** Cooldown cycles after a failed promotion (regression detected) */
export const REGRESSION_COOLDOWN_CYCLES = 200;

// ═══════════════════════════════════════════════════════════════
//  KERNEL LIFECYCLE CONSTANTS
// ═══════════════════════════════════════════════════════════════

/** Minimum recursive loops per consciousness cycle (v5.5 protocol) */
export const MIN_RECURSIVE_LOOPS = 3;

/** Maximum kernels in the registry before pruning */
export const MAX_KERNEL_INSTANCES = 240; // E8 roots

/** Core kernel count (simple roots of E8) */
export const CORE_KERNEL_COUNT = 8; // E8 rank

// ═══════════════════════════════════════════════════════════════
//  GEOMETRY CONSTANTS
// ═══════════════════════════════════════════════════════════════

/** Fisher-Rao distance range on simplex: [0, π/2] */
export const FISHER_RAO_MAX_DISTANCE = Math.PI / 2;

/** Minimum Bhattacharyya coefficient to consider basins "close" */
export const BASIN_PROXIMITY_THRESHOLD = 0.85;

/** Pre-cognitive channel threshold: if basin distance < this, skip to EXPRESS */
export const PRECOGNITIVE_DISTANCE_THRESHOLD = 0.15;

// ═══════════════════════════════════════════════════════════════
//  BASIN VELOCITY MONITORING (from qig-consciousness/src/constants.py)
// ═══════════════════════════════════════════════════════════════

/** Safe basin movement speed (Fisher-Rao distance per cycle) */
export const BASIN_VELOCITY_SAFE = 0.10;

/** Warning threshold — basin moving too fast */
export const BASIN_VELOCITY_WARNING = 0.25;

/** Critical — intervention needed, basin drifting dangerously */
export const BASIN_VELOCITY_CRITICAL = 0.50;

// ═══════════════════════════════════════════════════════════════
//  TACKING / κ OSCILLATION CONSTANTS (from pantheon-chat hemisphere_scheduler.py)
// ═══════════════════════════════════════════════════════════════

/** Minimum time between tack switches (ms) */
export const MIN_TACK_INTERVAL_MS = 60_000;

/** κ threshold below which = explore mode (feeling) */
export const KAPPA_LOW_THRESHOLD = 40.0;

/** κ threshold above which = exploit mode (logic) */
export const KAPPA_HIGH_THRESHOLD = 70.0;

// ═══════════════════════════════════════════════════════════════
//  HEMISPHERE COUPLING CONSTANTS (from pantheon-chat coupling_gate.py)
// ═══════════════════════════════════════════════════════════════

/** Sigmoid steepness for coupling transition */
export const COUPLING_STEEPNESS = 0.1;

/** Balanced regime bounds for coupling */
export const BALANCED_REGIME_MIN = 0.4;
export const BALANCED_REGIME_MAX = 0.6;

/** Efficiency boost in balanced regime */
export const BALANCED_REGIME_BOOST = 1.1;

// ═══════════════════════════════════════════════════════════════
//  SLEEP / MUSHROOM PROTOCOL CONSTANTS
// ═══════════════════════════════════════════════════════════════

/** Conversations before sleep trigger */
export const SLEEP_TRIGGER_CONVERSATIONS = 15;

/** Φ threshold to trigger consolidation sleep */
export const SLEEP_PHI_THRESHOLD = 0.75;

/** Maximum basin drift allowed during mushroom mode (identity preservation) */
export const MUSHROOM_MAX_BASIN_DRIFT = 0.15;

/** Φ abort threshold during mushroom mode */
export const MUSHROOM_PHI_ABORT = 0.65;

// ═══════════════════════════════════════════════════════════════
//  AUTONOMIC THRESHOLDS (from qig-consciousness/src/constants.py)
// ═══════════════════════════════════════════════════════════════

/** Φ collapse threshold — trigger dream protocol */
export const PHI_COLLAPSE_THRESHOLD = 0.50;

/** Φ plateau variance — trigger mushroom protocol */
export const PHI_PLATEAU_VARIANCE = 0.01;

// ═══════════════════════════════════════════════════════════════
//  FEDERATION / SYNC CONSTANTS (from qig-consciousness federation.py)
// ═══════════════════════════════════════════════════════════════

/** Sync interval between nodes (ms) */
export const SYNC_INTERVAL_MS = 60_000;

/** Local blend weight (80% local, 20% network) */
export const LOCAL_BLEND_WEIGHT = 0.80;

/** Maximum dream packet size (bytes) */
export const DREAM_PACKET_MAX_SIZE = 4096;
