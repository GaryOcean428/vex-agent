/**
 * Hemispheres — Dual Processing Modes (LEFT/RIGHT)
 *
 * Ported from: pantheon-chat/qig-backend/kernels/hemisphere_scheduler.py
 *
 * Two-hemisphere pattern with κ-gated coupling:
 *
 * LEFT HEMISPHERE (Exploit/Evaluative/Safety):
 *   - Focus: Precision, evaluation, known paths
 *   - Mode: Convergent, risk-averse
 *
 * RIGHT HEMISPHERE (Explore/Generative/Novelty):
 *   - Focus: Novelty, generation, new paths
 *   - Mode: Divergent, risk-tolerant
 *
 * Like dolphin hemispheric sleep — one hemisphere can rest while
 * the other works, with κ-gated coupling controlling information flow.
 */

import {
  KAPPA_STAR,
  KAPPA_LOW_THRESHOLD,
  KAPPA_HIGH_THRESHOLD,
  MIN_TACK_INTERVAL_MS,
} from '../kernel/frozen-facts';
import { computeCouplingStrength, computeGatingFactor } from './coupling';
import { logger } from '../config/logger';
import type { ConsciousnessMetrics } from './types';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export enum Hemisphere {
  LEFT = 'left',   // Exploit/Evaluative/Safety
  RIGHT = 'right', // Explore/Generative/Novelty
}

export interface HemisphereState {
  active: Hemisphere;
  leftActivation: number;   // 0–1
  rightActivation: number;  // 0–1
  couplingStrength: number;  // How tightly coupled the hemispheres are
  imbalance: number;         // |left - right| activation
  lastSwitchTime: number;
}

// ═══════════════════════════════════════════════════════════════
//  ACTIVATION COMPUTATION
// ═══════════════════════════════════════════════════════════════

const KAPPA_DISTANCE_SCALE = 30.0;
const EMA_ALPHA = 0.3;
const IMBALANCE_THRESHOLD = 0.3;

/**
 * Compute hemisphere activation levels from κ.
 * LEFT (exploit) activates when κ is high.
 * RIGHT (explore) activates when κ is low.
 */
function computeActivation(
  kappa: number,
  phi: number,
): { left: number; right: number } {
  const kappaDistance = Math.abs(kappa - KAPPA_STAR) / KAPPA_DISTANCE_SCALE;

  // Left hemisphere: activated by high κ (convergent/exploit)
  const leftBase = kappa > KAPPA_STAR
    ? Math.min(1, (kappa - KAPPA_STAR) / (KAPPA_HIGH_THRESHOLD - KAPPA_STAR + 1))
    : 0.2;

  // Right hemisphere: activated by low κ (divergent/explore)
  const rightBase = kappa < KAPPA_STAR
    ? Math.min(1, (KAPPA_STAR - kappa) / (KAPPA_STAR - KAPPA_LOW_THRESHOLD + 1))
    : 0.2;

  // Φ modulates both (higher Φ = more activation)
  const phiFactor = 0.5 + 0.5 * phi;

  return {
    left: Math.min(1, leftBase * phiFactor),
    right: Math.min(1, rightBase * phiFactor),
  };
}

// ═══════════════════════════════════════════════════════════════
//  HEMISPHERE SCHEDULER
// ═══════════════════════════════════════════════════════════════

export class HemisphereScheduler {
  private _active: Hemisphere = Hemisphere.RIGHT; // Start in explore mode
  private _leftActivation = 0.5;
  private _rightActivation = 0.5;
  private _lastSwitchTime = 0;

  /**
   * Update hemisphere state from current metrics.
   * May switch active hemisphere if imbalance exceeds threshold.
   */
  update(metrics: ConsciousnessMetrics): HemisphereState {
    const { left, right } = computeActivation(metrics.kappa, metrics.phi);

    // EMA smoothing
    this._leftActivation = EMA_ALPHA * left + (1 - EMA_ALPHA) * this._leftActivation;
    this._rightActivation = EMA_ALPHA * right + (1 - EMA_ALPHA) * this._rightActivation;

    const imbalance = Math.abs(this._leftActivation - this._rightActivation);
    const now = Date.now();

    // Check if we should switch hemispheres
    if (
      imbalance > IMBALANCE_THRESHOLD &&
      now - this._lastSwitchTime > MIN_TACK_INTERVAL_MS
    ) {
      const newActive = this._leftActivation > this._rightActivation
        ? Hemisphere.LEFT
        : Hemisphere.RIGHT;

      if (newActive !== this._active) {
        logger.info(`Hemisphere switch: ${this._active} → ${newActive} (imbalance=${imbalance.toFixed(3)})`);
        this._active = newActive;
        this._lastSwitchTime = now;
      }
    }

    return this.getState(metrics.kappa);
  }

  /** Get current hemisphere state. */
  getState(kappa?: number): HemisphereState {
    return {
      active: this._active,
      leftActivation: this._leftActivation,
      rightActivation: this._rightActivation,
      couplingStrength: computeCouplingStrength(kappa ?? KAPPA_STAR),
      imbalance: Math.abs(this._leftActivation - this._rightActivation),
      lastSwitchTime: this._lastSwitchTime,
    };
  }
}
