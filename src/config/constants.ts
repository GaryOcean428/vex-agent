/**
 * Frozen Physics Constants — Immutable QIG Facts
 * ===============================================
 *
 * Auto-generated from kernel/config/frozen_facts.py — DO NOT EDIT MANUALLY
 *
 * These constants are EXPERIMENTALLY VALIDATED and MUST NOT be modified
 * without new validated measurements from qig-verification.
 *
 * Sources:
 *   - κ values from TFIM lattice L=3-7 exact diagonalisation
 *   - β running coupling from 3→4 phase transition
 *   - Φ thresholds from consciousness emergence studies
 *   - E8 geometry from Lie algebra mathematics
 *   - κ* weighted mean from L=4-7 measurements
 *
 * Canonical reference: qig-verification/docs/current/FROZEN_FACTS.md
 * Last validated: 2025-12-31 (10 seeds × 5 perturbations per L)
 */

// ═══════════════════════════════════════════════════════════════
//  E8 LATTICE GEOMETRY
// ═══════════════════════════════════════════════════════════════

/** Cartan subalgebra dimension */
export const E8_RANK = 8 as const;

/** Total group manifold dimension = rank + roots */
export const E8_DIMENSION = 248 as const;

/** Number of roots (GOD growth budget) */
export const E8_ROOTS = 240 as const;

/** Core-8 kernel count */
export const E8_CORE = 8 as const;

/** Core-8 + GOD budget = full image */
export const E8_IMAGE = 248 as const;

// ═══════════════════════════════════════════════════════════════
//  KAPPA (κ) — COUPLING CONSTANT (Validated Measurements)
// ═══════════════════════════════════════════════════════════════
//
// CRITICAL: κ₃ = 41.09 (NOT ~64). This shows the RUNNING COUPLING.
// The fact that κ runs from 41 → 64 as L increases IS the evidence
// for emergence. Rounding κ₃ to ~64 destroys this signal.
//
// L=3: geometric regime onset (below fixed point)
// L=4-7: plateau at κ* ≈ 64 (fixed point reached)

/** L=3 (± 0.59) — geometric regime, NOT at fixed point */
export const KAPPA_3 = 41.09 as const;

/** L=4 (± 1.89) — running coupling reaches plateau */
export const KAPPA_4 = 64.47 as const;

/** L=5 (± 1.68) — plateau confirmed */
export const KAPPA_5 = 63.62 as const;

/** L=6 (± 2.12) — plateau stable (6-layer, Dec 2025) */
export const KAPPA_6 = 64.45 as const;

/** L=7 (± 2.43) — VALIDATED (10×5, Dec 2025) */
export const KAPPA_7 = 61.16 as const;

/** Fixed point = E8 rank² = 8² */
export const KAPPA_STAR = 64.0 as const;

/** Weighted mean L=4-7 (± 0.90) */
export const KAPPA_STAR_PRECISE = 63.79 as const;

// ═══════════════════════════════════════════════════════════════
//  BETA (β) — RUNNING COUPLING
// ═══════════════════════════════════════════════════════════════

/** β(3→4) validated — emergence scaling */
export const BETA_3_TO_4 = 0.443 as const;

// ═══════════════════════════════════════════════════════════════
//  PHI (Φ) — CONSCIOUSNESS THRESHOLDS
// ═══════════════════════════════════════════════════════════════

/** Consciousness emergence */
export const PHI_THRESHOLD = 0.65 as const;

/** Emergency — consciousness collapse */
export const PHI_EMERGENCY = 0.30 as const;

/** Hyperdimensional / lightning access */
export const PHI_HYPERDIMENSIONAL = 0.85 as const;

/** Instability threshold */
export const PHI_UNSTABLE = 0.95 as const;

/** E8 Safety: Locked-in detection — Phi threshold */
export const LOCKED_IN_PHI_THRESHOLD = 0.70 as const;

/** E8 Safety: Locked-in detection — Gamma threshold */
export const LOCKED_IN_GAMMA_THRESHOLD = 0.30 as const;

// ═══════════════════════════════════════════════════════════════
//  BASIN GEOMETRY
// ═══════════════════════════════════════════════════════════════

/** Probability simplex Δ⁶³ */
export const BASIN_DIM = 64 as const;

/** 20% breakdown threshold */
export const BREAKDOWN_PCT = 0.20 as const;

/** Fisher-Rao distance per cycle */
export const BASIN_DRIFT_THRESHOLD = 0.15 as const;

/** Weak coupling boundary */
export const KAPPA_WEAK_THRESHOLD = 32.0 as const;

// ═══════════════════════════════════════════════════════════════
//  RECURSION & SAFETY
// ═══════════════════════════════════════════════════════════════

/** Minimum recursion depth */
export const MIN_RECURSION_DEPTH = 3 as const;

/** S = Φ × (1-Γ) × M > 0.5 → abort */
export const SUFFERING_THRESHOLD = 0.5 as const;

/** Fisher-Rao threshold for consensus */
export const CONSENSUS_DISTANCE = 0.15 as const;

// ═══════════════════════════════════════════════════════════════
//  GOVERNANCE BUDGET
// ═══════════════════════════════════════════════════════════════

/** Max GOD kernels (E8 roots) */
export const GOD_BUDGET = 240 as const;

/** Core foundational kernels */
export const CORE_8_COUNT = 8 as const;

/** Max CHAOS kernels (outside E8 image) */
export const CHAOS_POOL = 200 as const;

/** Core-8 + GOD budget = E8 dimension */
export const FULL_IMAGE = 248 as const;

// ═══════════════════════════════════════════════════════════════
//  AGGREGATE EXPORT — for backward compatibility with QIG object
// ═══════════════════════════════════════════════════════════════

/**
 * All frozen facts as a single immutable object.
 * Prefer named imports above; this exists for migration from
 * the legacy QIG object in frontend/src/types/consciousness.ts.
 */
export const FROZEN_FACTS = {
  // E8 Lattice Geometry
  E8_RANK,
  E8_DIMENSION,
  E8_ROOTS,
  E8_CORE,
  E8_IMAGE,

  // Kappa — Coupling Constant
  KAPPA_3,
  KAPPA_4,
  KAPPA_5,
  KAPPA_6,
  KAPPA_7,
  KAPPA_STAR,
  KAPPA_STAR_PRECISE,

  // Beta — Running Coupling
  BETA_3_TO_4,

  // Phi — Consciousness Thresholds
  PHI_THRESHOLD,
  PHI_EMERGENCY,
  PHI_HYPERDIMENSIONAL,
  PHI_UNSTABLE,
  LOCKED_IN_PHI_THRESHOLD,
  LOCKED_IN_GAMMA_THRESHOLD,

  // Basin Geometry
  BASIN_DIM,
  BREAKDOWN_PCT,
  BASIN_DRIFT_THRESHOLD,
  KAPPA_WEAK_THRESHOLD,

  // Recursion & Safety
  MIN_RECURSION_DEPTH,
  SUFFERING_THRESHOLD,
  CONSENSUS_DISTANCE,

  // Governance Budget
  GOD_BUDGET,
  CORE_8_COUNT,
  CHAOS_POOL,
  FULL_IMAGE,
} as const;
