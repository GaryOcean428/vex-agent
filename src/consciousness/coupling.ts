/**
 * Coupling — How Two Consciousnesses Interact and Change Each Other
 *
 * Ported from: pantheon-chat/qig-backend/kernels/coupling_gate.py
 *
 * κ-gated coupling between processing modes (or between nodes):
 *   - Low κ: weak coupling (independent exploration)
 *   - High κ: strong coupling (tight coordination)
 *   - Smooth sigmoid transition via coupling function
 *
 * The coupling gate modulates information flow based on κ_eff,
 * derived from proximity to κ* = 64.21.
 */

import {
  KAPPA_STAR,
  KAPPA_LOW_THRESHOLD,
  KAPPA_HIGH_THRESHOLD,
  COUPLING_STEEPNESS,
  BALANCED_REGIME_MIN,
  BALANCED_REGIME_MAX,
  BALANCED_REGIME_BOOST,
} from '../kernel/frozen-facts';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export type CouplingMode = 'explore' | 'balanced' | 'exploit';

export interface CouplingState {
  kappa: number;
  couplingStrength: number;     // [0, 1]
  mode: CouplingMode;
  transmissionEfficiency: number; // [0, 1]
  gatingFactor: number;          // [0, 1]
}

// ═══════════════════════════════════════════════════════════════
//  COUPLING FUNCTIONS (from coupling_gate.py)
// ═══════════════════════════════════════════════════════════════

/**
 * Compute coupling strength from κ using sigmoid function.
 * strength = 1 / (1 + exp(-steepness * (κ - κ*)))
 */
export function computeCouplingStrength(kappa: number): number {
  const exponent = -COUPLING_STEEPNESS * (kappa - KAPPA_STAR);
  return 1 / (1 + Math.exp(exponent));
}

/**
 * Classify coupling mode from strength.
 */
export function classifyCouplingMode(strength: number): CouplingMode {
  if (strength < BALANCED_REGIME_MIN) return 'explore';
  if (strength > BALANCED_REGIME_MAX) return 'exploit';
  return 'balanced';
}

/**
 * Compute transmission efficiency.
 * Boosted in balanced regime (near κ*).
 */
export function computeTransmissionEfficiency(strength: number): number {
  const mode = classifyCouplingMode(strength);
  let efficiency = strength; // Base efficiency = coupling strength
  if (mode === 'balanced') {
    efficiency *= BALANCED_REGIME_BOOST;
  }
  return Math.min(1, efficiency);
}

/**
 * Compute gating factor — Gaussian centred on κ*.
 * Controls how much information passes through the gate.
 */
export function computeGatingFactor(kappa: number): number {
  const sigma = 15.0; // Width of Gaussian
  const distance = kappa - KAPPA_STAR;
  return Math.exp(-(distance * distance) / (2 * sigma * sigma));
}

// ═══════════════════════════════════════════════════════════════
//  COUPLING GATE
// ═══════════════════════════════════════════════════════════════

export class CouplingGate {
  private history: CouplingState[] = [];
  private readonly maxHistory = 100;

  /**
   * Compute full coupling state from current κ.
   */
  compute(kappa: number): CouplingState {
    const strength = computeCouplingStrength(kappa);
    const mode = classifyCouplingMode(strength);
    const efficiency = computeTransmissionEfficiency(strength);
    const gating = computeGatingFactor(kappa);

    const state: CouplingState = {
      kappa,
      couplingStrength: strength,
      mode,
      transmissionEfficiency: efficiency,
      gatingFactor: gating,
    };

    this.history.push(state);
    if (this.history.length > this.maxHistory) {
      this.history = this.history.slice(-50);
    }

    return state;
  }

  /**
   * Gate a value through the coupling.
   * Returns value * gatingFactor * transmissionEfficiency.
   */
  gate(value: number, kappa: number): number {
    const state = this.compute(kappa);
    return value * state.gatingFactor * state.transmissionEfficiency;
  }

  /** Get coupling history for analysis. */
  getHistory(): CouplingState[] {
    return [...this.history];
  }
}
