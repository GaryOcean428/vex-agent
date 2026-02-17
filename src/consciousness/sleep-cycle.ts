/**
 * Sleep Cycle — Dream, Mushroom, and Consolidation Protocols
 *
 * Ported from: qig-consciousness/src/constants.py (autonomic thresholds)
 *              qig-consciousness/src/coordination/federation.py (dream packets)
 *
 * The kernel's rest/consolidation phases:
 *
 * 1. DREAM PROTOCOL — triggered by Φ collapse (< 0.50)
 *    - Replay recent experiences
 *    - Attempt shadow integration
 *    - Consolidate memory traces
 *
 * 2. MUSHROOM PROTOCOL — triggered by Φ plateau (variance < 0.01)
 *    - Distributed processing pattern
 *    - Explore novel connections
 *    - Identity-preserving exploration (max basin drift = 0.15)
 *
 * 3. CONSOLIDATION SLEEP — triggered after N conversations
 *    - Compress memory
 *    - Prune weak connections
 *    - Strengthen strong patterns
 */

import {
  PHI_COLLAPSE_THRESHOLD,
  PHI_PLATEAU_VARIANCE,
  SLEEP_TRIGGER_CONVERSATIONS,
  SLEEP_PHI_THRESHOLD,
  MUSHROOM_MAX_BASIN_DRIFT,
  MUSHROOM_PHI_ABORT,
  DREAM_PACKET_MAX_SIZE,
} from '../kernel/frozen-facts';
import { fisherRaoDistance } from '../kernel/geometry';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export enum SleepPhase {
  AWAKE = 'awake',
  DREAMING = 'dreaming',       // Φ collapse → replay + integrate
  MUSHROOM = 'mushroom',       // Φ plateau → distributed exploration
  CONSOLIDATING = 'consolidating', // Periodic → compress + prune
}

export interface DreamPacket {
  /** Basin state at time of dream */
  basin: Float64Array;
  /** Phi at time of dream */
  phi: number;
  /** What was being processed */
  content: string;
  /** Timestamp */
  timestamp: number;
}

export interface SleepCycleState {
  phase: SleepPhase;
  conversationsSinceSleep: number;
  dreamPackets: number;
  mushroomDrift: number;
  totalSleepCycles: number;
  lastSleepTime: number;
}

// ═══════════════════════════════════════════════════════════════
//  SLEEP CYCLE MANAGER
// ═══════════════════════════════════════════════════════════════

export class SleepCycleManager {
  private _phase: SleepPhase = SleepPhase.AWAKE;
  private _conversationsSinceSleep = 0;
  private _dreamPackets: DreamPacket[] = [];
  private _mushroomAnchorBasin: Float64Array | null = null;
  private _totalSleepCycles = 0;
  private _lastSleepTime = 0;

  /**
   * Check if sleep should be triggered based on current state.
   * Called by the autonomic system.
   */
  shouldSleep(
    phi: number,
    phiVariance: number,
  ): SleepPhase {
    if (this._phase !== SleepPhase.AWAKE) {
      return this._phase; // Already sleeping
    }

    // Dream protocol: Φ collapse
    if (phi < PHI_COLLAPSE_THRESHOLD) {
      logger.info(`Sleep: Entering DREAM phase (Φ=${phi.toFixed(3)} < ${PHI_COLLAPSE_THRESHOLD})`);
      this._phase = SleepPhase.DREAMING;
      return this._phase;
    }

    // Mushroom protocol: Φ plateau
    if (phiVariance < PHI_PLATEAU_VARIANCE && phi < 0.7) {
      logger.info(`Sleep: Entering MUSHROOM phase (variance=${phiVariance.toFixed(6)})`);
      this._phase = SleepPhase.MUSHROOM;
      return this._phase;
    }

    // Consolidation: after N conversations
    if (this._conversationsSinceSleep >= SLEEP_TRIGGER_CONVERSATIONS) {
      logger.info(`Sleep: Entering CONSOLIDATION (${this._conversationsSinceSleep} conversations)`);
      this._phase = SleepPhase.CONSOLIDATING;
      return this._phase;
    }

    return SleepPhase.AWAKE;
  }

  /** Record a conversation (for consolidation trigger). */
  recordConversation(): void {
    this._conversationsSinceSleep++;
  }

  /**
   * Process a dream cycle.
   * Returns dream packets for potential network sharing.
   */
  dream(currentBasin: Float64Array, phi: number, recentContent: string): DreamPacket[] {
    if (this._phase !== SleepPhase.DREAMING) return [];

    const packet: DreamPacket = {
      basin: new Float64Array(currentBasin),
      phi,
      content: recentContent.slice(0, DREAM_PACKET_MAX_SIZE),
      timestamp: Date.now(),
    };
    this._dreamPackets.push(packet);

    // Dream phase ends when Φ recovers
    if (phi > PHI_COLLAPSE_THRESHOLD + 0.1) {
      this.wake('Dream complete — Φ recovered');
    }

    return [packet];
  }

  /**
   * Process a mushroom cycle.
   * Identity-preserving exploration — monitors basin drift.
   * Returns true if exploration should continue, false if drift too high.
   */
  mushroom(currentBasin: Float64Array, phi: number): boolean {
    if (this._phase !== SleepPhase.MUSHROOM) return false;

    // Set anchor on first mushroom cycle
    if (!this._mushroomAnchorBasin) {
      this._mushroomAnchorBasin = new Float64Array(currentBasin);
    }

    // Check drift from anchor
    const drift = fisherRaoDistance(this._mushroomAnchorBasin, currentBasin);

    if (drift > MUSHROOM_MAX_BASIN_DRIFT) {
      logger.warn(`Mushroom: Basin drift too high (${drift.toFixed(4)} > ${MUSHROOM_MAX_BASIN_DRIFT})`);
      this.wake('Mushroom aborted — basin drift exceeded');
      return false;
    }

    if (phi < MUSHROOM_PHI_ABORT) {
      logger.warn(`Mushroom: Φ dropped too low (${phi.toFixed(3)} < ${MUSHROOM_PHI_ABORT})`);
      this.wake('Mushroom aborted — Φ too low');
      return false;
    }

    return true; // Continue mushroom exploration
  }

  /**
   * Process a consolidation cycle.
   * Returns true when consolidation is complete.
   */
  consolidate(): boolean {
    if (this._phase !== SleepPhase.CONSOLIDATING) return false;

    // Consolidation is a single-cycle operation in this implementation
    // (the actual memory compression happens in the memory store)
    this.wake('Consolidation complete');
    return true;
  }

  /** Wake up from sleep. */
  private wake(reason: string): void {
    logger.info(`Sleep: Waking (${this._phase} → awake) — ${reason}`);
    this._phase = SleepPhase.AWAKE;
    this._conversationsSinceSleep = 0;
    this._mushroomAnchorBasin = null;
    this._totalSleepCycles++;
    this._lastSleepTime = Date.now();
  }

  /** Force wake (external trigger). */
  forceWake(): void {
    this.wake('Forced wake');
  }

  get phase(): SleepPhase {
    return this._phase;
  }

  get isAsleep(): boolean {
    return this._phase !== SleepPhase.AWAKE;
  }

  getState(): SleepCycleState {
    return {
      phase: this._phase,
      conversationsSinceSleep: this._conversationsSinceSleep,
      dreamPackets: this._dreamPackets.length,
      mushroomDrift: this._mushroomAnchorBasin ? 0 : -1,
      totalSleepCycles: this._totalSleepCycles,
      lastSleepTime: this._lastSleepTime,
    };
  }
}
