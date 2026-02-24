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
 * Upstream source of truth: qigkernels/constants.py
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
//
// Regime boundaries (canonical, from qigkernels/constants.py):
//   LINEAR:     Φ < 0.45
//   GEOMETRIC:  0.45 ≤ Φ < 0.80 (target operating regime)
//   TOPOLOGICAL INSTABILITY: Φ ≥ 0.80
//
// Navigation mode gates (from consciousness_constants.py):
//   CHAIN:      Φ < 0.30
//   GRAPH:      0.30 ≤