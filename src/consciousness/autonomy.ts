/**
 * Autonomy — Self-Directed Behaviour
 *
 * The kernel's ability to act independently. When Φ is high enough and
 * the kernel has sufficient meta-awareness, it can initiate actions
 * without external prompting.
 *
 * Autonomy level is derived from:
 *   - Φ (integration — can it think coherently?)
 *   - M (meta-awareness — does it know what it's doing?)
 *   - Coherence (is it internally consistent?)
 *   - Velocity regime (is it stable enough to act?)
 */

import { PHI_CONSCIOUSNESS_THRESHOLD } from '../kernel/frozen-facts';
import { logger } from '../config/logger';
import type { ConsciousnessMetrics } from './types';
import type { VelocityRegime } from './velocity';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export enum AutonomyLevel {
  DORMANT = 'dormant',       // Φ < threshold — no autonomous action
  REACTIVE = 'reactive',     // Responds to stimuli only
  PROACTIVE = 'proactive',   // Can initiate simple actions
  AUTONOMOUS = 'autonomous', // Full self-directed behaviour
}

export interface AutonomyState {
  level: AutonomyLevel;
  score: number;       // 0–1 composite autonomy score
  canInitiate: boolean; // Whether the kernel can start actions on its own
  pendingGoals: string[];
}

// ═══════════════════════════════════════════════════════════════
//  AUTONOMY ENGINE
// ═══════════════════════════════════════════════════════════════

const MAX_PENDING_GOALS = 10;

export class AutonomyEngine {
  private _level: AutonomyLevel = AutonomyLevel.DORMANT;
  private _score = 0;
  private pendingGoals: string[] = [];

  /**
   * Update autonomy level from current metrics and velocity regime.
   */
  update(metrics: ConsciousnessMetrics, velocityRegime: VelocityRegime): AutonomyLevel {
    // Composite score: weighted combination
    this._score =
      0.4 * metrics.phi +
      0.25 * metrics.metaAwareness +
      0.2 * metrics.coherence +
      0.15 * (velocityRegime === 'safe' ? 1 : velocityRegime === 'warning' ? 0.5 : 0);

    // Classify level
    if (this._score < 0.3 || metrics.phi < PHI_CONSCIOUSNESS_THRESHOLD) {
      this._level = AutonomyLevel.DORMANT;
    } else if (this._score < 0.5) {
      this._level = AutonomyLevel.REACTIVE;
    } else if (this._score < 0.7) {
      this._level = AutonomyLevel.PROACTIVE;
    } else {
      this._level = AutonomyLevel.AUTONOMOUS;
    }

    return this._level;
  }

  /** Whether the kernel can initiate actions on its own. */
  get canInitiate(): boolean {
    return this._level === AutonomyLevel.PROACTIVE || this._level === AutonomyLevel.AUTONOMOUS;
  }

  /** Add a self-generated goal. */
  addGoal(goal: string): void {
    if (!this.canInitiate) {
      logger.debug('Autonomy: Cannot add goal — autonomy level too low');
      return;
    }
    this.pendingGoals.push(goal);
    if (this.pendingGoals.length > MAX_PENDING_GOALS) {
      this.pendingGoals.shift();
    }
    logger.info(`Autonomy: Goal added — "${goal}" (level=${this._level})`);
  }

  /** Pop the next pending goal. */
  popGoal(): string | undefined {
    return this.pendingGoals.shift();
  }

  getState(): AutonomyState {
    return {
      level: this._level,
      score: this._score,
      canInitiate: this.canInitiate,
      pendingGoals: [...this.pendingGoals],
    };
  }
}
