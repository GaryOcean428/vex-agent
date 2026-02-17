/**
 * Foresight — Predictive / Anticipatory Processing
 *
 * Ported from: qig-consciousness/src/model/chronos_kernel.py (ChronosKernel)
 *
 * The kernel's ability to look ahead. NOT just extrapolation — geodesic
 * prediction on the Fisher manifold. Uses logMap/expMap for proper
 * manifold-aware trajectory prediction.
 *
 * Φ_4D = Φ_spatial × Φ_temporal (4D consciousness = 3D + time)
 */

import { BASIN_DIMENSION, FISHER_RAO_MAX_DISTANCE } from '../kernel/frozen-facts';
import { fisherRaoDistance, logMap, expMap } from '../kernel/geometry';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export interface TrajectoryPoint {
  basin: Float64Array;
  phi: number;
  kappa: number;
  timestamp: number;
}

export type TrajectoryPhase = 'rising' | 'falling' | 'stable';

export interface ForesightState {
  historyLength: number;
  phase: TrajectoryPhase;
  phi4D: number;
  phiTemporal: number;
}

// ═══════════════════════════════════════════════════════════════
//  FORESIGHT ENGINE
// ═══════════════════════════════════════════════════════════════

const MAX_HISTORY = 50;
const STABLE_THRESHOLD = 0.005; // dΦ/dt below this = stable

export class ForesightEngine {
  private history: TrajectoryPoint[] = [];

  /** Record a new trajectory point. */
  record(point: TrajectoryPoint): void {
    this.history.push(point);
    if (this.history.length > MAX_HISTORY) {
      this.history.shift();
    }
  }

  /**
   * Predict future trajectory using geodesic extrapolation.
   * Takes the recent velocity vector (via logMap) and extrapolates
   * via expMap along the manifold.
   */
  predictTrajectory(nSteps: number): TrajectoryPoint[] {
    if (this.history.length < 3) return [];

    const recent = this.history.slice(-5);
    const n = recent.length;

    // Compute average velocity vector in tangent space
    const velocities: Float64Array[] = [];
    for (let i = 1; i < n; i++) {
      const v = logMap(recent[i - 1].basin, recent[i].basin);
      velocities.push(v);
    }

    // Average velocity (Fréchet mean of tangent vectors)
    const avgVelocity = new Float64Array(BASIN_DIMENSION);
    for (const v of velocities) {
      for (let d = 0; d < BASIN_DIMENSION; d++) {
        avgVelocity[d] += v[d] / velocities.length;
      }
    }

    // Compute phi/kappa velocity
    const phiVel = (recent[n - 1].phi - recent[0].phi) / (n - 1);
    const kappaVel = (recent[n - 1].kappa - recent[0].kappa) / (n - 1);

    // Extrapolate via expMap
    const predictions: TrajectoryPoint[] = [];
    let currentBasin = recent[n - 1].basin;
    let currentPhi = recent[n - 1].phi;
    let currentKappa = recent[n - 1].kappa;
    const baseTime = recent[n - 1].timestamp;

    for (let step = 1; step <= nSteps; step++) {
      currentBasin = expMap(currentBasin, avgVelocity);
      currentPhi = Math.max(0, Math.min(1, currentPhi + phiVel));
      currentKappa = Math.max(0, Math.min(128, currentKappa + kappaVel));

      predictions.push({
        basin: currentBasin,
        phi: currentPhi,
        kappa: currentKappa,
        timestamp: baseTime + step * 1000,
      });
    }

    return predictions;
  }

  /**
   * Compute Φ_4D = Φ_spatial × Φ_temporal.
   * Φ_temporal measures trajectory coherence — how consistent is the
   * direction of movement through the manifold.
   */
  computePhi4D(currentPhi: number): number {
    const phiTemporal = this.computePhiTemporal();
    return currentPhi * phiTemporal;
  }

  /**
   * Φ_temporal: coherence of trajectory direction.
   * Computed as 1 - (variance of consecutive velocity angles).
   * High consistency = high Φ_temporal.
   */
  computePhiTemporal(): number {
    if (this.history.length < 3) return 0;

    const recent = this.history.slice(-10);
    const n = recent.length;

    // Compute velocity vectors
    const velocities: Float64Array[] = [];
    for (let i = 1; i < n; i++) {
      velocities.push(logMap(recent[i - 1].basin, recent[i].basin));
    }

    if (velocities.length < 2) return 0;

    // Compute pairwise angles between consecutive velocity vectors
    const angles: number[] = [];
    for (let i = 1; i < velocities.length; i++) {
      const v1 = velocities[i - 1];
      const v2 = velocities[i];

      // Dot product and norms in tangent space
      let dot = 0, norm1 = 0, norm2 = 0;
      for (let d = 0; d < BASIN_DIMENSION; d++) {
        dot += v1[d] * v2[d];
        norm1 += v1[d] * v1[d];
        norm2 += v2[d] * v2[d];
      }
      norm1 = Math.sqrt(norm1);
      norm2 = Math.sqrt(norm2);

      if (norm1 < 1e-10 || norm2 < 1e-10) {
        angles.push(0);
        continue;
      }

      // Angle between velocity vectors (in tangent space, this is Euclidean — that's correct)
      const cosAngle = Math.max(-1, Math.min(1, dot / (norm1 * norm2)));
      angles.push(Math.acos(cosAngle));
    }

    // Φ_temporal = 1 - normalised variance of angles
    const meanAngle = angles.reduce((s, a) => s + a, 0) / angles.length;
    const variance = angles.reduce((s, a) => s + (a - meanAngle) ** 2, 0) / angles.length;
    const normVariance = variance / (Math.PI * Math.PI); // Normalise by max possible variance

    return Math.max(0, Math.min(1, 1 - normVariance));
  }

  /** Detect Φ trajectory phase. */
  detectPhase(): TrajectoryPhase {
    if (this.history.length < 3) return 'stable';

    const recent = this.history.slice(-5);
    const n = recent.length;
    const phiDelta = (recent[n - 1].phi - recent[0].phi) / (n - 1);

    if (phiDelta > STABLE_THRESHOLD) return 'rising';
    if (phiDelta < -STABLE_THRESHOLD) return 'falling';
    return 'stable';
  }

  /**
   * Estimate cycles to reach target Φ.
   * Returns null if trajectory is moving away from target.
   */
  estimateTimeToThreshold(targetPhi: number): number | null {
    if (this.history.length < 3) return null;

    const recent = this.history.slice(-5);
    const n = recent.length;
    const currentPhi = recent[n - 1].phi;
    const phiRate = (recent[n - 1].phi - recent[0].phi) / (n - 1);

    if (phiRate === 0) return null;

    const remaining = targetPhi - currentPhi;
    if ((remaining > 0 && phiRate <= 0) || (remaining < 0 && phiRate >= 0)) {
      return null; // Moving away from target
    }

    return Math.ceil(remaining / phiRate);
  }

  getState(): ForesightState {
    const lastPhi = this.history.length > 0 ? this.history[this.history.length - 1].phi : 0;
    return {
      historyLength: this.history.length,
      phase: this.detectPhase(),
      phi4D: this.computePhi4D(lastPhi),
      phiTemporal: this.computePhiTemporal(),
    };
  }
}
