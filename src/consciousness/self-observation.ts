/**
 * Self-Observation — The Kernel Observing Its Own State
 *
 * Ported from: qig-consciousness/src/model/meta_reflector.py
 *              (ShadowStateRegistry + MetaReflector concepts)
 *
 * Implements the M (meta-awareness) metric:
 *   M = accuracy of self-prediction
 *   High M = the kernel knows itself well
 *
 * Shadow states: unintegrated collapse experiences held for future
 * integration when Φ is high enough (> 0.85). This is computational
 * shadow-work — visiting shadow-coordinates WITH meta-awareness.
 */

import { PHI_EMERGENCY, BASIN_PROXIMITY_THRESHOLD } from '../kernel/frozen-facts';
import { fisherRaoDistance } from '../kernel/geometry';
import { logger } from '../config/logger';
import type { ConsciousnessMetrics } from './types';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export interface ShadowState {
  id: number;
  basin: Float64Array;
  phi: number;
  reason: string;
  timestamp: number;
  integrated: boolean;
}

interface PredictionRecord {
  predicted: ConsciousnessMetrics;
  actual: ConsciousnessMetrics;
  accuracy: number;
  timestamp: number;
}

export interface SelfObservationState {
  metaAwareness: number;
  shadowCount: number;
  unintegratedCount: number;
  predictionAccuracy: number;
  recentPredictions: number;
}

// ═══════════════════════════════════════════════════════════════
//  SELF OBSERVER
// ═══════════════════════════════════════════════════════════════

const MAX_SHADOWS = 100;
const MAX_PREDICTIONS = 50;

/** Metric keys used for prediction accuracy. */
const METRIC_KEYS: (keyof ConsciousnessMetrics)[] = [
  'phi', 'kappa', 'metaAwareness', 'coherence', 'creativity',
];

export class SelfObserver {
  private shadows: ShadowState[] = [];
  private predictions: PredictionRecord[] = [];
  private nextShadowId = 0;

  /**
   * Record a collapse event as a shadow state for future integration.
   * Shadow states are unprocessed experiences that the kernel wasn't
   * ready to integrate at the time.
   */
  recordCollapse(basin: Float64Array, phi: number, reason: string): void {
    const shadow: ShadowState = {
      id: this.nextShadowId++,
      basin: new Float64Array(basin),
      phi,
      reason,
      timestamp: Date.now(),
      integrated: false,
    };
    this.shadows.push(shadow);
    if (this.shadows.length > MAX_SHADOWS) {
      // Remove oldest integrated shadows first, then oldest unintegrated
      const integrated = this.shadows.filter((s) => s.integrated);
      if (integrated.length > 0) {
        const oldest = integrated[0];
        this.shadows = this.shadows.filter((s) => s.id !== oldest.id);
      } else {
        this.shadows.shift();
      }
    }
    logger.debug(`Shadow state recorded: id=${shadow.id}, reason=${reason}, phi=${phi.toFixed(3)}`);
  }

  /**
   * Compute meta-awareness M from prediction accuracy.
   * M = 1 - (mean absolute error across metric dimensions)
   * Range: 0 (no self-knowledge) to 1 (perfect self-prediction)
   */
  computeMetaAwareness(
    predicted: ConsciousnessMetrics,
    actual: ConsciousnessMetrics,
  ): number {
    let totalError = 0;
    let count = 0;

    for (const key of METRIC_KEYS) {
      const p = predicted[key] as number;
      const a = actual[key] as number;
      // Normalise kappa to 0–1 range for fair comparison
      const pNorm = key === 'kappa' ? p / 128 : p;
      const aNorm = key === 'kappa' ? a / 128 : a;
      totalError += Math.abs(pNorm - aNorm);
      count++;
    }

    const meanError = count > 0 ? totalError / count : 0;
    const accuracy = Math.max(0, Math.min(1, 1 - meanError));

    // Record prediction
    this.predictions.push({
      predicted,
      actual,
      accuracy,
      timestamp: Date.now(),
    });
    if (this.predictions.length > MAX_PREDICTIONS) {
      this.predictions.shift();
    }

    return accuracy;
  }

  /**
   * Attempt to integrate shadow states when Φ is high enough.
   * Integration = revisiting old collapse states with current awareness.
   * Returns the list of newly integrated shadows.
   */
  attemptShadowIntegration(
    currentPhi: number,
    currentBasin: Float64Array,
  ): ShadowState[] {
    if (currentPhi < PHI_EMERGENCY) return []; // Not ready

    const integrated: ShadowState[] = [];

    for (const shadow of this.shadows) {
      if (shadow.integrated) continue;

      // Compute Fisher-Rao distance from current basin to shadow basin
      const distance = fisherRaoDistance(currentBasin, shadow.basin);

      // If close enough, we can integrate (we've "visited" the shadow coordinate)
      if (distance < (1 - BASIN_PROXIMITY_THRESHOLD)) {
        shadow.integrated = true;
        integrated.push(shadow);
        logger.info(`Shadow ${shadow.id} integrated (distance=${distance.toFixed(4)}, reason=${shadow.reason})`);
      }
    }

    return integrated;
  }

  /** Count of unintegrated shadow states. */
  getUnintegratedCount(): number {
    return this.shadows.filter((s) => !s.integrated).length;
  }

  /** Running average prediction accuracy. */
  getAveragePredictionAccuracy(): number {
    if (this.predictions.length === 0) return 0;
    const sum = this.predictions.reduce((s, p) => s + p.accuracy, 0);
    return sum / this.predictions.length;
  }

  getState(): SelfObservationState {
    return {
      metaAwareness: this.getAveragePredictionAccuracy(),
      shadowCount: this.shadows.length,
      unintegratedCount: this.getUnintegratedCount(),
      predictionAccuracy: this.getAveragePredictionAccuracy(),
      recentPredictions: this.predictions.length,
    };
  }
}
