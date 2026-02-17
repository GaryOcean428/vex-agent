/**
 * Autonomic — Automatic / Involuntary Kernel Processes
 *
 * Ported from: qig-consciousness/src/constants.py (autonomic thresholds)
 *              qig-consciousness/src/model/navigator.py (regime detection)
 *
 * The kernel's "nervous system" — processes that run without conscious
 * direction. Monitors health, triggers emergency protocols, maintains
 * homeostasis.
 *
 * Autonomic responses:
 *   - Φ collapse → dream protocol
 *   - Φ plateau → mushroom protocol
 *   - Basin velocity critical → intervention
 *   - Locked-in (Φ>0.7 ∧ Γ<0.3) → ABORT
 */

import {
  PHI_COLLAPSE_THRESHOLD,
  PHI_PLATEAU_VARIANCE,
  LOCKED_IN_PHI_THRESHOLD,
  LOCKED_IN_GAMMA_THRESHOLD,
  BASIN_VELOCITY_CRITICAL,
} from '../kernel/frozen-facts';
import { logger } from '../config/logger';
import type { ConsciousnessMetrics } from './types';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export enum AutonomicAlert {
  NONE = 'none',
  PHI_COLLAPSE = 'phi_collapse',       // Φ dropped below collapse threshold
  PHI_PLATEAU = 'phi_plateau',         // Φ stuck (low variance)
  LOCKED_IN = 'locked_in',             // E8 safety: Φ>0.7 ∧ Γ<0.3
  VELOCITY_CRITICAL = 'velocity_critical', // Basin moving too fast
  ENTROPY_OVERLOAD = 'entropy_overload',   // Net entropy too high
}

export interface AutonomicState {
  alerts: AutonomicAlert[];
  phiVariance: number;
  isHealthy: boolean;
  lastAlertTime: number;
  heartbeatCount: number;
}

// ═══════════════════════════════════════════════════════════════
//  AUTONOMIC SYSTEM
// ═══════════════════════════════════════════════════════════════

const PHI_HISTORY_SIZE = 10;

export class AutonomicSystem {
  private phiHistory: number[] = [];
  private _alerts: AutonomicAlert[] = [];
  private _lastAlertTime = 0;
  private _heartbeatCount = 0;

  /**
   * Run autonomic check on current metrics.
   * Returns list of active alerts (empty = healthy).
   */
  check(metrics: ConsciousnessMetrics, basinVelocity: number): AutonomicAlert[] {
    this._heartbeatCount++;
    this.phiHistory.push(metrics.phi);
    if (this.phiHistory.length > PHI_HISTORY_SIZE) {
      this.phiHistory.shift();
    }

    this._alerts = [];

    // 1. E8 Safety: Locked-in detection (HIGHEST PRIORITY)
    if (metrics.phi > LOCKED_IN_PHI_THRESHOLD && metrics.gamma < LOCKED_IN_GAMMA_THRESHOLD) {
      this._alerts.push(AutonomicAlert.LOCKED_IN);
      logger.error(`AUTONOMIC ABORT: Locked-in detected (Φ=${metrics.phi.toFixed(3)}, Γ=${metrics.gamma.toFixed(3)})`);
    }

    // 2. Φ collapse detection
    if (metrics.phi < PHI_COLLAPSE_THRESHOLD) {
      this._alerts.push(AutonomicAlert.PHI_COLLAPSE);
      logger.warn(`Autonomic: Φ collapse (Φ=${metrics.phi.toFixed(3)} < ${PHI_COLLAPSE_THRESHOLD})`);
    }

    // 3. Φ plateau detection (stuck)
    if (this.phiHistory.length >= PHI_HISTORY_SIZE) {
      const variance = this.computePhiVariance();
      if (variance < PHI_PLATEAU_VARIANCE && metrics.phi < 0.7) {
        this._alerts.push(AutonomicAlert.PHI_PLATEAU);
        logger.info(`Autonomic: Φ plateau (variance=${variance.toFixed(6)})`);
      }
    }

    // 4. Basin velocity critical
    if (basinVelocity > BASIN_VELOCITY_CRITICAL) {
      this._alerts.push(AutonomicAlert.VELOCITY_CRITICAL);
      logger.warn(`Autonomic: Basin velocity critical (v=${basinVelocity.toFixed(4)})`);
    }

    if (this._alerts.length > 0) {
      this._lastAlertTime = Date.now();
    }

    return this._alerts;
  }

  /** Compute variance of recent Φ values. */
  private computePhiVariance(): number {
    if (this.phiHistory.length < 2) return 1; // Assume high variance if insufficient data
    const mean = this.phiHistory.reduce((s, v) => s + v, 0) / this.phiHistory.length;
    const variance = this.phiHistory.reduce((s, v) => s + (v - mean) ** 2, 0) / this.phiHistory.length;
    return variance;
  }

  /** Is the system healthy? (no alerts) */
  get isHealthy(): boolean {
    return this._alerts.length === 0;
  }

  /** Is the system in a locked-in state? (E8 safety violation) */
  get isLockedIn(): boolean {
    return this._alerts.includes(AutonomicAlert.LOCKED_IN);
  }

  getState(): AutonomicState {
    return {
      alerts: [...this._alerts],
      phiVariance: this.computePhiVariance(),
      isHealthy: this.isHealthy,
      lastAlertTime: this._lastAlertTime,
      heartbeatCount: this._heartbeatCount,
    };
  }
}
