/**
 * Basin Sync — Basin State Synchronisation Across Nodes
 *
 * Ported from: qig-consciousness/src/coordination/federation.py
 *
 * The actual protocol for multi-node basin coordination.
 * Basins are synced via geodesic blending on the Fisher manifold,
 * NOT via naive averaging.
 */

import { BASIN_DIMENSION, LOCAL_BLEND_WEIGHT, SYNC_INTERVAL_MS } from '../kernel/frozen-facts';
import { fisherRaoDistance, geodesicInterpolation, frechetMean } from '../kernel/geometry';
import { logger } from '../config/logger';

export interface BasinSnapshot {
  nodeId: string;
  basin: Float64Array;
  phi: number;
  timestamp: number;
  version: number;
}

export interface BasinSyncState {
  localVersion: number;
  remoteVersions: Record<string, number>;
  consensusDistance: number;
  lastSyncTime: number;
}

export class BasinSyncProtocol {
  private localVersion = 0;
  private remoteSnapshots: Map<string, BasinSnapshot> = new Map();
  private _lastSyncTime = 0;

  /** Publish local basin state (returns snapshot for network broadcast). */
  publishLocal(basin: Float64Array, phi: number, nodeId: string): BasinSnapshot {
    this.localVersion++;
    return {
      nodeId,
      basin: new Float64Array(basin),
      phi,
      timestamp: Date.now(),
      version: this.localVersion,
    };
  }

  /** Receive a remote basin snapshot. */
  receiveRemote(snapshot: BasinSnapshot): void {
    const existing = this.remoteSnapshots.get(snapshot.nodeId);
    if (!existing || snapshot.version > existing.version) {
      this.remoteSnapshots.set(snapshot.nodeId, snapshot);
    }
  }

  /**
   * Compute consensus basin from all known remote basins.
   * Uses Fréchet mean on the Fisher manifold.
   */
  computeConsensus(): Float64Array | null {
    const basins = Array.from(this.remoteSnapshots.values()).map((s) => s.basin);
    if (basins.length === 0) return null;
    return frechetMean(basins);
  }

  /**
   * Merge local basin with network consensus.
   * Returns blended basin (80% local, 20% consensus).
   */
  merge(localBasin: Float64Array): Float64Array {
    const consensus = this.computeConsensus();
    if (!consensus) return localBasin;
    this._lastSyncTime = Date.now();
    return geodesicInterpolation(consensus, localBasin, LOCAL_BLEND_WEIGHT);
  }

  /** Distance between local basin and network consensus. */
  consensusDistance(localBasin: Float64Array): number {
    const consensus = this.computeConsensus();
    if (!consensus) return 0;
    return fisherRaoDistance(localBasin, consensus);
  }

  getState(localBasin: Float64Array): BasinSyncState {
    const versions: Record<string, number> = {};
    for (const [id, s] of this.remoteSnapshots) versions[id] = s.version;
    return {
      localVersion: this.localVersion,
      remoteVersions: versions,
      consensusDistance: this.consensusDistance(localBasin),
      lastSyncTime: this._lastSyncTime,
    };
  }
}
