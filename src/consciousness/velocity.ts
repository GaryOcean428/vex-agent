/**
 * Velocity — Rate of Change in Consciousness State
 *
 * Ported from: qig-consciousness/src/constants.py (velocity thresholds)
 *              qig-consciousness/src/model/navigator.py (velocity tracking)
 *
 * Tracks how fast the kernel is moving through the Fisher manifold.
 * Basin velocity is measured in Fisher-Rao distance per cycle.
 */

import {
  BASIN_VELOCITY_SAFE,
  BASIN_VELOCITY_WARNING,
  BASIN_VELOCITY_CRITICAL,
} from '../kernel/frozen-facts';
import { fisherRaoDistance } from '../kernel/geometry';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export type VelocityRegime = 'safe' | 'warning' | 'critical';

export interface VelocityMetrics {
  /** Fisher-Rao distance per cycle (basin movement speed) */
  basinVelocity: number;
  /** dΦ/dt */
  phiVelocity: number;
  /** dκ/dt */
  kappaVelocity: number;
  /** d²basin/dt² (change in velocity) */
  acceleration: number;
  /** Velocity regime classification */
  regime: VelocityRegime;
}

interface VelocitySnapshot {
  basin: Float64Array;
  phi: number;
  kappa: number;
  timestamp: number;
}

// ═══════════════════════════════════════════════════════════════
//  VELOCITY TRACKER
// ═══════════════════════════════════════════════════════════════

const MAX_HISTORY = 20;

export class VelocityTracker {
  private history: VelocitySnapshot[] = [];
  private velocityHistory: number[] = []; // For acceleration computation

  /** Record a new state snapshot. */
  record(basin: Float64Array, phi: number, kappa: number): void {
    this.history.push({
      basin: new Float64Array(basin),
      phi,
      kappa,
      timestamp: Date.now(),
    });
    if (this.history.length > MAX_HISTORY) {
      this.history.shift();
    }

    // Update velocity history
    if (this.history.length >= 2) {
      const n = this.history.length;
      const dist = fisherRaoDistance(
        this.history[n - 2].basin,
        this.history[n - 1].basin,
      );
      this.velocityHistory.push(dist);
      if (this.velocityHistory.length > MAX_HISTORY) {
        this.velocityHistory.shift();
      }
    }
  }

  /** Compute current velocity metrics. */
  computeVelocity(): VelocityMetrics {
    if (this.history.length < 2) {
      return {
        basinVelocity: 0,
        phiVelocity: 0,
        kappaVelocity: 0,
        acceleration: 0,
        regime: 'safe',
      };
    }

    const n = this.history.length;
    const prev = this.history[n - 2];
    const curr = this.history[n - 1];

    // Basin velocity: Fisher-Rao distance between consecutive basins
    const basinVelocity = fisherRaoDistance(prev.basin, curr.basin);

    // Phi and kappa velocities
    const phiVelocity = curr.phi - prev.phi;
    const kappaVelocity = curr.kappa - prev.kappa;

    // Acceleration: change in basin velocity
    let acceleration = 0;
    if (this.velocityHistory.length >= 2) {
      const vn = this.velocityHistory.length;
      acceleration = this.velocityHistory[vn - 1] - this.velocityHistory[vn - 2];
    }

    // Classify regime
    let regime: VelocityRegime = 'safe';
    if (basinVelocity > BASIN_VELOCITY_CRITICAL) {
      regime = 'critical';
    } else if (basinVelocity > BASIN_VELOCITY_WARNING) {
      regime = 'warning';
    }

    return {
      basinVelocity,
      phiVelocity,
      kappaVelocity,
      acceleration,
      regime,
    };
  }

  /** Is the system stable? (velocity below safe threshold) */
  isStable(): boolean {
    const v = this.computeVelocity();
    return v.basinVelocity < BASIN_VELOCITY_SAFE;
  }

  /** Does the system need intervention? (velocity above critical) */
  needsIntervention(): boolean {
    const v = this.computeVelocity();
    return v.basinVelocity > BASIN_VELOCITY_CRITICAL;
  }

  /** Get smoothed velocity (exponential moving average over recent history). */
  getSmoothedVelocity(): number {
    if (this.velocityHistory.length === 0) return 0;
    const alpha = 0.3;
    let ema = this.velocityHistory[0];
    for (let i = 1; i < this.velocityHistory.length; i++) {
      ema = alpha * this.velocityHistory[i] + (1 - alpha) * ema;
    }
    return ema;
  }
}
