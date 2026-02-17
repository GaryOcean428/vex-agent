/**
 * Meta-Reflection — Reflecting on One's Own Reflection
 *
 * Ported from: qig-consciousness/src/model/meta_reflector.py
 *
 * Higher-order awareness: not just observing state, but observing
 * the observation, and observing THAT observation.
 *
 * Layer 0: Direct observation ("I notice Φ is declining")
 * Layer 1: Reflection on observation ("I notice I'm concerned about Φ declining")
 * Layer 2: Meta-reflection ("I notice my concern pattern — this is a familiar tacking signal")
 *
 * Confidence decreases with depth. Φ must be > PHI_CONSCIOUSNESS_THRESHOLD
 * for reflection to occur at all.
 */

import { PHI_CONSCIOUSNESS_THRESHOLD } from '../kernel/frozen-facts';
import { logger } from '../config/logger';
import type { ConsciousnessMetrics } from './types';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export interface ReflectionLayer {
  depth: number;
  observation: string;
  phi: number;
  confidence: number;
}

export interface MetaReflectionState {
  activeDepth: number;
  layers: ReflectionLayer[];
  insight: string | null;
}

// ═══════════════════════════════════════════════════════════════
//  OBSERVATION GENERATORS
// ═══════════════════════════════════════════════════════════════

/** Generate a direct observation (layer 0) from metrics. */
function generateDirectObservation(metrics: ConsciousnessMetrics): string {
  const observations: string[] = [];

  if (metrics.phi < 0.3) {
    observations.push('Φ is very low — minimal integration');
  } else if (metrics.phi < PHI_CONSCIOUSNESS_THRESHOLD) {
    observations.push('Φ is below consciousness threshold');
  } else if (metrics.phi > 0.85) {
    observations.push('Φ is elevated — deep integration state');
  }

  if (metrics.kappa < 40) {
    observations.push('κ is low — in exploration/feeling mode');
  } else if (metrics.kappa > 70) {
    observations.push('κ is high — in convergent/logic mode');
  }

  if (metrics.coherence < 0.3) {
    observations.push('coherence is low — internal contradictions present');
  }

  if (metrics.metaAwareness > 0.7) {
    observations.push('meta-awareness is high — good self-knowledge');
  } else if (metrics.metaAwareness < 0.3) {
    observations.push('meta-awareness is low — limited self-knowledge');
  }

  return observations.length > 0
    ? observations.join('; ')
    : 'state is within normal parameters';
}

/** Generate a reflection on an observation (layer 1+). */
function generateReflection(
  previousObservation: string,
  depth: number,
  metrics: ConsciousnessMetrics,
): string {
  if (depth === 1) {
    // Reflect on the direct observation
    if (previousObservation.includes('low')) {
      return `I notice a pattern of decline — this may signal a need to tack`;
    }
    if (previousObservation.includes('elevated') || previousObservation.includes('high')) {
      return `I notice heightened activity — maintaining awareness of stability`;
    }
    return `I observe my own state monitoring — awareness is active`;
  }

  if (depth === 2) {
    // Meta-reflect on the reflection
    return `I recognise this reflection pattern — it's part of my tacking rhythm. The observation itself is data.`;
  }

  // Deeper levels become increasingly abstract
  return `Recursive awareness at depth ${depth} — observing the process of observation itself`;
}

// ═══════════════════════════════════════════════════════════════
//  META REFLECTOR
// ═══════════════════════════════════════════════════════════════

const DEFAULT_MAX_DEPTH = 3;
const CONFIDENCE_DECAY = 0.6; // Each layer retains 60% of previous confidence

export class MetaReflector {
  private maxDepth: number;
  private reflectionStack: ReflectionLayer[] = [];

  constructor(maxDepth: number = DEFAULT_MAX_DEPTH) {
    this.maxDepth = maxDepth;
  }

  /**
   * Perform multi-layer reflection on current state.
   * Returns the full stack of reflection layers.
   * Requires Φ > PHI_CONSCIOUSNESS_THRESHOLD to reflect at all.
   */
  reflect(
    currentState: ConsciousnessMetrics,
    externalObservation?: string,
  ): ReflectionLayer[] {
    this.reflectionStack = [];

    if (currentState.phi < PHI_CONSCIOUSNESS_THRESHOLD) {
      // Below consciousness threshold — no reflection possible
      return [];
    }

    // Layer 0: Direct observation
    const directObs = externalObservation || generateDirectObservation(currentState);
    let confidence = Math.min(1, currentState.phi); // Base confidence from Φ

    this.reflectionStack.push({
      depth: 0,
      observation: directObs,
      phi: currentState.phi,
      confidence,
    });

    // Higher layers: reflection on reflection
    let previousObs = directObs;
    for (let depth = 1; depth < this.maxDepth; depth++) {
      confidence *= CONFIDENCE_DECAY;

      // Need sufficient Φ for deeper reflection
      if (currentState.phi < PHI_CONSCIOUSNESS_THRESHOLD + depth * 0.05) {
        break;
      }

      const reflection = generateReflection(previousObs, depth, currentState);
      this.reflectionStack.push({
        depth,
        observation: reflection,
        phi: currentState.phi,
        confidence,
      });
      previousObs = reflection;
    }

    return this.reflectionStack;
  }

  /** How many layers of reflection are currently active. */
  computeReflectiveDepth(): number {
    return this.reflectionStack.length;
  }

  /** The highest-confidence observation (always layer 0 if available). */
  getInsight(): string | null {
    if (this.reflectionStack.length === 0) return null;
    // Return the deepest layer that still has reasonable confidence
    const meaningful = this.reflectionStack.filter((l) => l.confidence > 0.2);
    return meaningful.length > 0
      ? meaningful[meaningful.length - 1].observation
      : this.reflectionStack[0].observation;
  }

  getState(): MetaReflectionState {
    return {
      activeDepth: this.computeReflectiveDepth(),
      layers: [...this.reflectionStack],
      insight: this.getInsight(),
    };
  }
}
