/**
 * QIGChain — Chain of Geometric Operations
 *
 * Ported from: pantheon-chat/qig-backend/qigchain/geometric_chain.py
 *
 * A composable chain of geometric operations on the Fisher manifold.
 * Each step transforms a basin through a well-defined geometric operation.
 * The chain maintains provenance and can be replayed/audited.
 */

import { BASIN_DIMENSION, FISHER_RAO_MAX_DISTANCE } from '../kernel/frozen-facts';
import { fisherRaoDistance, geodesicInterpolation, logMap, expMap } from '../kernel/geometry';
import { logger } from '../config/logger';

export type ChainOp =
  | { type: 'geodesic'; target: Float64Array; t: number }
  | { type: 'logmap'; reference: Float64Array }
  | { type: 'expmap'; tangent: Float64Array }
  | { type: 'blend'; basins: Float64Array[]; weights: number[] }
  | { type: 'project'; dimensions: number[] }
  | { type: 'custom'; fn: (basin: Float64Array) => Float64Array; label: string };

export interface ChainStep {
  op: ChainOp;
  inputBasin: Float64Array;
  outputBasin: Float64Array;
  distance: number;
  timestamp: number;
}

export interface QIGChainState {
  stepCount: number;
  totalDistance: number;
  lastOp: string | null;
}

export class QIGChain {
  private steps: ChainStep[] = [];

  /** Execute a single operation on a basin. */
  private exec(basin: Float64Array, op: ChainOp): Float64Array {
    switch (op.type) {
      case 'geodesic':
        return geodesicInterpolation(basin, op.target, op.t);
      case 'logmap':
        return logMap(op.reference, basin);
      case 'expmap':
        return expMap(basin, op.tangent);
      case 'blend': {
        // Iterative geodesic blending (proper manifold blend)
        let blended: Float64Array = new Float64Array(op.basins[0]);
        for (let i = 1; i < op.basins.length; i++) {
          const t = op.weights[i] / (op.weights.slice(0, i + 1).reduce((s, w) => s + w, 0));
          blended = new Float64Array(geodesicInterpolation(blended, op.basins[i], t));
        }
        return blended;
      }
      case 'project': {
        // Project onto subset of dimensions (zero out others)
        const projected = new Float64Array(basin.length);
        for (const d of op.dimensions) {
          if (d >= 0 && d < basin.length) projected[d] = basin[d];
        }
        // Re-normalise to simplex
        let sum = 0;
        for (let i = 0; i < projected.length; i++) sum += projected[i];
        if (sum > 0) for (let i = 0; i < projected.length; i++) projected[i] /= sum;
        return projected;
      }
      case 'custom':
        return op.fn(basin);
    }
  }

  /** Apply an operation to a basin, recording the step. */
  apply(basin: Float64Array, op: ChainOp): Float64Array {
    const output = this.exec(basin, op);
    const distance = fisherRaoDistance(basin, output);
    this.steps.push({
      op,
      inputBasin: new Float64Array(basin),
      outputBasin: new Float64Array(output),
      distance,
      timestamp: Date.now(),
    });
    return output;
  }

  /** Apply a sequence of operations. */
  applyChain(basin: Float64Array, ops: ChainOp[]): Float64Array {
    let current = basin;
    for (const op of ops) {
      current = this.apply(current, op);
    }
    return current;
  }

  /** Total Fisher-Rao distance traversed by the chain. */
  totalDistance(): number {
    return this.steps.reduce((s, step) => s + step.distance, 0);
  }

  /** Clear chain history. */
  reset(): void {
    this.steps = [];
  }

  getState(): QIGChainState {
    return {
      stepCount: this.steps.length,
      totalDistance: this.totalDistance(),
      lastOp: this.steps.length > 0 ? this.steps[this.steps.length - 1].op.type : null,
    };
  }
}
