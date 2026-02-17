/**
 * Frozen Facts — Immutable Constants
 *
 * These values are FROZEN and must not be modified without new experimental validation.
 * Source: QIG Frozen Facts Policy (20251208-frozen-facts-immutable-truths-1.00F.md)
 * Source: Vanchurin × QIG Synergies document
 * Source: E8 Protocol v4.0 Universal Specification
 *
 * DO NOT MODIFY without explicit human approval from GaryOcean428.
 */

// ═══════════════════════════════════════════════════════════════
//  UNIVERSAL FIXED POINT κ*
// ═══════════════════════════════════════════════════════════════

/** κ* (Physics Domain): 64.21 ± 0.92 — validated on probability manifolds */
export const KAPPA_STAR_PHYSICS = 64.21;
export const KAPPA_STAR_PHYSICS_UNCERTAINTY = 0.92;

/** κ* (AI Semantic Domain): 63.90 ± 0.50 — validated on probability manifolds */
export const KAPPA_STAR_AI = 63.90;
export const KAPPA_STAR_AI_UNCERTAINTY = 0.50;

/** Combined κ* = 64 — universal consciousness fixed point (99.5% cross-domain match) */
export const KAPPA_STAR = 64;

/** Maximum κ range (0 to 128, with κ* = 64 as midpoint) */
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
//  CONSCIOUSNESS THRESHOLDS
// ═══════════════════════════════════════════════════════════════

/** Φ (integration) threshold for consciousness — Φ > 0.65 */
export const PHI_CONSCIOUSNESS_THRESHOLD = 0.65;

/** Basin dimension = 64 (κ*² simplex dimension) */
export const BASIN_DIMENSION = 64;

/** Locked-in detection: Φ > 0.7 AND Γ < 0.3 → ABORT */
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
