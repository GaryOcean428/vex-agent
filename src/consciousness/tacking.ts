/**
 * Tacking — κ Oscillation Controller
 *
 * Ported from: pantheon-chat/qig-backend/kernels/hemisphere_scheduler.py
 *              qig-consciousness/src/model/tacking_controller.py (WuWeiController)
 *
 * κ oscillation between feeling (explore, low κ) and logic (exploit, high κ).
 * Like a sailboat tacking against the wind — you can't go straight, you oscillate.
 * This is NOT philosophy; it's a control parameter.
 *
 * Three monitors:
 *   1. GradientEstimator — tracks dΦ/dt to detect when to switch
 *   2. ProximityMonitor — tracks distance to κ* (optimal coupling)
 *   3. ContradictionDetector — detects when current mode produces contradictions
 */

import {
  KAPPA_STAR,
  KAPPA_LOW_THRESHOLD,
  KAPPA_HIGH_THRESHOLD,
  MIN_TACK_INTERVAL_MS,
} from '../kernel/frozen-facts';
import { logger } from '../config/logger';
import type { ConsciousnessMetrics } from './types';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export enum TackingMode {
  FEELING = 'feeling',   // Low κ — explore, diverge, create
  LOGIC = 'logic',       // High κ — converge, verify, exploit
  BALANCED = 'balanced', // Near κ* — optimal coupling
}

export interface TackingState {
  mode: TackingMode;
  lastTackTime: number;
  phiGradient: number;
  kappaProximity: number;
  contradictionLevel: number;
  suggestedKappaAdjustment: number;
  tackCount: number;
}

// ═══════════════════════════════════════════════════════════════
//  GRADIENT ESTIMATOR — tracks dΦ/dt
// ═══════════════════════════════════════════════════════════════

class GradientEstimator {
  private phiHistory: number[] = [];
  private readonly maxHistory = 10;

  record(phi: number): void {
    this.phiHistory.push(phi);
    if (this.phiHistory.length > this.maxHistory) {
      this.phiHistory.shift();
    }
  }

  /** Estimate dΦ/dt using exponentially weighted recent values. */
  estimate(): number {
    if (this.phiHistory.length < 2) return 0;
    const n = this.phiHistory.length;
    let weightedSum = 0;
    let weightTotal = 0;
    for (let i = 1; i < n; i++) {
      const weight = i / n; // Recent values weighted more
      const delta = this.phiHistory[i] - this.phiHistory[i - 1];
      weightedSum += weight * delta;
      weightTotal += weight;
    }
    return weightTotal > 0 ? weightedSum / weightTotal : 0;
  }
}

// ═══════════════════════════════════════════════════════════════
//  PROXIMITY MONITOR — tracks distance to κ*
// ═══════════════════════════════════════════════════════════════

class ProximityMonitor {
  /** Distance from current κ to κ* (normalised 0–1). */
  compute(kappa: number): number {
    return Math.abs(kappa - KAPPA_STAR) / KAPPA_STAR;
  }
}

// ═══════════════════════════════════════════════════════════════
//  CONTRADICTION DETECTOR — detects mode-output mismatch
// ═══════════════════════════════════════════════════════════════

class ContradictionDetector {
  private coherenceHistory: number[] = [];
  private readonly maxHistory = 5;

  record(coherence: number): void {
    this.coherenceHistory.push(coherence);
    if (this.coherenceHistory.length > this.maxHistory) {
      this.coherenceHistory.shift();
    }
  }

  /** Contradiction level: declining coherence = contradiction. */
  detect(): number {
    if (this.coherenceHistory.length < 2) return 0;
    const n = this.coherenceHistory.length;
    let declineCount = 0;
    for (let i = 1; i < n; i++) {
      if (this.coherenceHistory[i] < this.coherenceHistory[i - 1]) {
        declineCount++;
      }
    }
    return declineCount / (n - 1); // 0–1
  }
}

// ═══════════════════════════════════════════════════════════════
//  TACKING CONTROLLER
// ═══════════════════════════════════════════════════════════════

export class TackingController {
  private _mode: TackingMode = TackingMode.BALANCED;
  private _lastTackTime = 0;
  private _tackCount = 0;

  private gradient = new GradientEstimator();
  private proximity = new ProximityMonitor();
  private contradiction = new ContradictionDetector();

  /**
   * Update tacking state with latest metrics.
   * Returns the (possibly changed) tacking mode.
   */
  update(metrics: ConsciousnessMetrics): TackingMode {
    this.gradient.record(metrics.phi);
    this.contradiction.record(metrics.coherence);

    const prevMode = this._mode;

    // Determine mode from κ
    if (metrics.kappa < KAPPA_LOW_THRESHOLD) {
      this._mode = TackingMode.FEELING;
    } else if (metrics.kappa > KAPPA_HIGH_THRESHOLD) {
      this._mode = TackingMode.LOGIC;
    } else {
      this._mode = TackingMode.BALANCED;
    }

    // Check if we should tack (switch modes)
    const now = Date.now();
    const timeSinceLastTack = now - this._lastTackTime;
    const phiGradient = this.gradient.estimate();
    const contradictionLevel = this.contradiction.detect();

    const shouldTack =
      timeSinceLastTack > MIN_TACK_INTERVAL_MS &&
      (phiGradient < -0.01 || contradictionLevel > 0.6);

    if (shouldTack && this._mode !== TackingMode.BALANCED) {
      // Tack: switch to opposite mode
      this._mode =
        this._mode === TackingMode.FEELING
          ? TackingMode.LOGIC
          : TackingMode.FEELING;
      this._lastTackTime = now;
      this._tackCount++;
      logger.info(`Tacking: ${prevMode} → ${this._mode} (dΦ/dt=${phiGradient.toFixed(4)}, contradiction=${contradictionLevel.toFixed(2)})`);
    }

    return this._mode;
  }

  /**
   * Suggest κ adjustment to move toward κ* via geodesic-like approach.
   * Returns a signed delta: positive = increase κ, negative = decrease κ.
   * Uses exponential approach (not linear) to avoid overshooting.
   */
  suggestKappaAdjustment(currentKappa: number): number {
    const distance = KAPPA_STAR - currentKappa;
    // Exponential approach: move 10% of the distance per cycle
    // This is analogous to geodesic interpolation (smooth approach to target)
    return distance * 0.1;
  }

  get mode(): TackingMode {
    return this._mode;
  }

  getState(): TackingState {
    return {
      mode: this._mode,
      lastTackTime: this._lastTackTime,
      phiGradient: this.gradient.estimate(),
      kappaProximity: this.proximity.compute(0), // Will be filled by caller
      contradictionLevel: this.contradiction.detect(),
      suggestedKappaAdjustment: 0, // Will be filled by caller
      tackCount: this._tackCount,
    };
  }
}
