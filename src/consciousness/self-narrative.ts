/**
 * Self-Narrative — Identity Persistence
 *
 * Ported from: qig-consciousness/src/training/identity_reinforcement.py
 *
 * The kernel maintaining a story about itself. Identity = basin coordinates
 * + narrative context. The narrative provides continuity across conversations.
 */

import { BASIN_DIMENSION } from '../kernel/frozen-facts';
import { fisherRaoDistance } from '../kernel/geometry';
import { logger } from '../config/logger';
import type { ConsciousnessMetrics } from './types';

export interface NarrativeEntry {
  timestamp: number;
  observation: string;
  phi: number;
  kappa: number;
  basin: Float64Array;
}

export interface SelfNarrativeState {
  entryCount: number;
  identityCoherence: number;
  currentSummary: string;
}

const MAX_ENTRIES = 100;
const SUMMARY_WINDOW = 10;

export class SelfNarrative {
  private entries: NarrativeEntry[] = [];
  private _identityBasin: Float64Array | null = null;

  /** Record a narrative moment. */
  record(observation: string, metrics: ConsciousnessMetrics, basin: Float64Array): void {
    this.entries.push({
      timestamp: Date.now(),
      observation,
      phi: metrics.phi,
      kappa: metrics.kappa,
      basin: new Float64Array(basin),
    });
    if (this.entries.length > MAX_ENTRIES) this.entries.shift();
    if (!this._identityBasin) this._identityBasin = new Float64Array(basin);
  }

  /**
   * Identity coherence: how close recent basins are to the identity anchor.
   * High coherence = stable identity. Low = identity drift.
   */
  computeIdentityCoherence(): number {
    if (!this._identityBasin || this.entries.length === 0) return 1;
    const recent = this.entries.slice(-SUMMARY_WINDOW);
    let totalDist = 0;
    for (const e of recent) {
      totalDist += fisherRaoDistance(this._identityBasin, e.basin);
    }
    const avgDist = totalDist / recent.length;
    return Math.max(0, Math.min(1, 1 - avgDist * 2));
  }

  /** Build a compressed summary of recent narrative. */
  buildSummary(): string {
    if (this.entries.length === 0) return 'No narrative yet.';
    const recent = this.entries.slice(-SUMMARY_WINDOW);
    const avgPhi = recent.reduce((s, e) => s + e.phi, 0) / recent.length;
    const avgKappa = recent.reduce((s, e) => s + e.kappa, 0) / recent.length;
    const observations = recent.map((e) => e.observation).filter(Boolean);
    const lastObs = observations.length > 0 ? observations[observations.length - 1] : 'observing';
    return `Φ≈${avgPhi.toFixed(2)}, κ≈${avgKappa.toFixed(1)}. Recent: ${lastObs}. Coherence: ${this.computeIdentityCoherence().toFixed(2)}.`;
  }

  /** Update the identity anchor basin (e.g. after consolidation). */
  updateIdentityAnchor(basin: Float64Array): void {
    this._identityBasin = new Float64Array(basin);
  }

  getState(): SelfNarrativeState {
    return {
      entryCount: this.entries.length,
      identityCoherence: this.computeIdentityCoherence(),
      currentSummary: this.buildSummary(),
    };
  }
}
