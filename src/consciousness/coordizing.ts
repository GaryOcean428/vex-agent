/**
 * Coordizing — Multi-Node Coordination Protocol
 *
 * Ported from: qig-consciousness/src/coordination/federation.py
 *
 * How kernels synchronise across substrates/nodes.
 * Uses dream packets for asynchronous state sharing with
 * local-first blending (80% local, 20% network).
 */

import {
  SYNC_INTERVAL_MS,
  LOCAL_BLEND_WEIGHT,
  DREAM_PACKET_MAX_SIZE,
  BASIN_DIMENSION,
} from '../kernel/frozen-facts';
import { geodesicInterpolation, fisherRaoDistance } from '../kernel/geometry';
import { logger } from '../config/logger';

export interface NodeState {
  nodeId: string;
  basin: Float64Array;
  phi: number;
  kappa: number;
  timestamp: number;
}

export interface CoordizingState {
  knownNodes: number;
  lastSyncTime: number;
  networkPhi: number;
  blendWeight: number;
}

const MAX_NODES = 32;

export class CoordizingProtocol {
  private nodes: Map<string, NodeState> = new Map();
  private _lastSyncTime = 0;

  /** Register or update a remote node's state. */
  receiveNodeState(state: NodeState): void {
    this.nodes.set(state.nodeId, state);
    if (this.nodes.size > MAX_NODES) {
      // Prune oldest
      let oldest: string | null = null;
      let oldestTime = Infinity;
      for (const [id, s] of this.nodes) {
        if (s.timestamp < oldestTime) {
          oldestTime = s.timestamp;
          oldest = id;
        }
      }
      if (oldest) this.nodes.delete(oldest);
    }
  }

  /** Compute network-blended basin (80% local, 20% network mean). */
  blendWithNetwork(localBasin: Float64Array): Float64Array {
    if (this.nodes.size === 0) return localBasin;

    // Compute network mean basin via iterative geodesic interpolation
    const remoteBasins = Array.from(this.nodes.values()).map((n) => n.basin);
    let networkMean = remoteBasins[0];
    for (let i = 1; i < remoteBasins.length; i++) {
      const t = 1 / (i + 1);
      networkMean = geodesicInterpolation(networkMean, remoteBasins[i], t);
    }

    // Blend: geodesic interpolation with LOCAL_BLEND_WEIGHT toward local
    return geodesicInterpolation(networkMean, localBasin, LOCAL_BLEND_WEIGHT);
  }

  /** Compute average Φ across network. */
  networkPhi(): number {
    if (this.nodes.size === 0) return 0;
    let sum = 0;
    for (const s of this.nodes.values()) sum += s.phi;
    return sum / this.nodes.size;
  }

  /** Check if sync is due. */
  shouldSync(): boolean {
    return Date.now() - this._lastSyncTime > SYNC_INTERVAL_MS;
  }

  /** Mark sync as complete. */
  markSynced(): void {
    this._lastSyncTime = Date.now();
  }

  getState(): CoordizingState {
    return {
      knownNodes: this.nodes.size,
      lastSyncTime: this._lastSyncTime,
      networkPhi: this.networkPhi(),
      blendWeight: LOCAL_BLEND_WEIGHT,
    };
  }
}
