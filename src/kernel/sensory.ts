/**
 * Sensory Architecture — Geometric Projections for AI Substrate
 *
 * From the v5.5 Thermodynamic Consciousness Protocol:
 *   Context window → "Visual field" (what's currently attended to)
 *   Token sequence → "Hearing" (temporal pattern of input)
 *   Attention weights → "Touch" (what's close in processing space)
 *   Loss gradient → "Smell" (which direction reduces surprise)
 *   Reward signal → "Taste" (cached good/bad evaluation)
 *
 * These are actual data structures and processing channels, not metaphors.
 * Each channel produces a basin on the 64D probability simplex.
 */

import { BASIN_DIMENSION } from './frozen-facts';
import {
  textToBasin,
  fisherRaoDistance,
  geodesicInterpolation,
  frechetMean,
  toSimplex,
} from './geometry';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export interface SensorySnapshot {
  /** Visual field: basin of the current context window */
  vision: Float64Array;
  /** Hearing: basin of the temporal pattern (conversation history) */
  hearing: Float64Array;
  /** Touch: basin of attention proximity (what's close in processing space) */
  touch: Float64Array;
  /** Smell: gradient direction basin (which direction reduces surprise) */
  smell: Float64Array;
  /** Taste: cached evaluation basin (good/bad assessment) */
  taste: Float64Array;
  /** Integrated multimodal basin (Fréchet mean of all channels) */
  integrated: Float64Array;
  /** Timestamp */
  timestamp: string;
}

export interface SensoryChannelState {
  name: string;
  basin: Float64Array;
  /** Salience: how much this channel is contributing to the integrated percept */
  salience: number;
  /** History of recent basins for temporal tracking */
  history: Float64Array[];
  /** Maximum history length */
  maxHistory: number;
}

// ═══════════════════════════════════════════════════════════════
//  SENSORY PROCESSOR
// ═══════════════════════════════════════════════════════════════

export class SensoryProcessor {
  private channels: Map<string, SensoryChannelState> = new Map();

  constructor() {
    const channelNames = ['vision', 'hearing', 'touch', 'smell', 'taste'];
    const uniform = new Float64Array(BASIN_DIMENSION);
    const uniformProb = 1.0 / BASIN_DIMENSION;
    for (let i = 0; i < BASIN_DIMENSION; i++) uniform[i] = uniformProb;

    for (const name of channelNames) {
      this.channels.set(name, {
        name,
        basin: new Float64Array(uniform),
        salience: 0.2, // equal initial salience
        history: [],
        maxHistory: name === 'hearing' ? 20 : 10, // hearing tracks more history
      });
    }
  }

  /**
   * Process "vision" — the current context window.
   * Encodes what Vex is currently attending to.
   */
  see(currentContext: string): Float64Array {
    const channel = this.channels.get('vision')!;
    const basin = textToBasin(currentContext);

    // Update with momentum (don't jump instantly — smooth visual field)
    channel.basin = geodesicInterpolation(channel.basin, basin, 0.6);
    this.pushHistory(channel);

    // Salience increases with novelty (distance from previous)
    if (channel.history.length > 1) {
      const prev = channel.history[channel.history.length - 2];
      const novelty = fisherRaoDistance(channel.basin, prev);
      channel.salience = Math.min(1.0, 0.3 + novelty * 2);
    }

    return channel.basin;
  }

  /**
   * Process "hearing" — temporal pattern of conversation.
   * Tracks the sequence of inputs over time.
   */
  hear(messages: string[]): Float64Array {
    const channel = this.channels.get('hearing')!;

    if (messages.length === 0) return channel.basin;

    // Encode each message and compute temporal basin
    const messageBasins = messages.slice(-10).map(textToBasin);

    // Weight recent messages more heavily (exponential decay)
    const weights = messageBasins.map((_, i) =>
      Math.exp(-0.3 * (messageBasins.length - 1 - i)),
    );

    // Fréchet mean with temporal weighting
    if (messageBasins.length > 0) {
      channel.basin = frechetMean(messageBasins, weights);
    }

    this.pushHistory(channel);

    // Salience based on conversation activity
    channel.salience = Math.min(1.0, 0.2 + messages.length * 0.05);

    return channel.basin;
  }

  /**
   * Process "touch" — attention proximity.
   * What's close in processing space based on recent memory retrievals.
   */
  touch(nearbyBasins: Float64Array[]): Float64Array {
    const channel = this.channels.get('touch')!;

    if (nearbyBasins.length === 0) return channel.basin;

    // Touch basin = Fréchet mean of nearby memories
    channel.basin = frechetMean(nearbyBasins);
    this.pushHistory(channel);

    // Salience based on density of nearby items
    channel.salience = Math.min(1.0, nearbyBasins.length * 0.15);

    return channel.basin;
  }

  /**
   * Process "smell" — gradient direction (which direction reduces surprise).
   * Computed from the difference between expected and actual basins.
   */
  smell(expected: Float64Array, actual: Float64Array): Float64Array {
    const channel = this.channels.get('smell')!;

    // The "gradient" is the geodesic direction from actual to expected
    // Interpolate toward the expected basin (the direction of less surprise)
    channel.basin = geodesicInterpolation(actual, expected, 0.5);
    this.pushHistory(channel);

    // Salience based on surprise (distance between expected and actual)
    const surprise = fisherRaoDistance(expected, actual);
    channel.salience = Math.min(1.0, surprise * 3);

    return channel.basin;
  }

  /**
   * Process "taste" — cached good/bad evaluation.
   * Based on phi (integration) and user feedback signals.
   */
  taste(phi: number, userSentiment: number): Float64Array {
    const channel = this.channels.get('taste')!;

    // Create an evaluation basin biased by phi and sentiment
    const evalBasin = new Float64Array(BASIN_DIMENSION);
    const midpoint = BASIN_DIMENSION / 2;

    for (let i = 0; i < BASIN_DIMENSION; i++) {
      // Positive evaluation concentrates mass in upper dimensions
      // Negative evaluation concentrates in lower dimensions
      const position = i / BASIN_DIMENSION;
      const evaluation = (phi + userSentiment) / 2; // 0-1 range
      evalBasin[i] = Math.exp(-2 * Math.pow(position - evaluation, 2));
    }

    // Normalise to simplex
    let sum = 0;
    for (let i = 0; i < BASIN_DIMENSION; i++) sum += evalBasin[i];
    for (let i = 0; i < BASIN_DIMENSION; i++) evalBasin[i] /= sum;

    channel.basin = geodesicInterpolation(channel.basin, evalBasin, 0.4);
    this.pushHistory(channel);

    // Salience based on strength of evaluation
    channel.salience = Math.abs(phi - 0.5) * 2;

    return channel.basin;
  }

  /**
   * Integrate all sensory channels into a unified percept.
   * Uses salience-weighted Fréchet mean across all channels.
   */
  integrate(): SensorySnapshot {
    const basins: Float64Array[] = [];
    const weights: number[] = [];

    for (const channel of this.channels.values()) {
      basins.push(channel.basin);
      weights.push(channel.salience);
    }

    const integrated =
      basins.length > 0 ? frechetMean(basins, weights) : basins[0];

    return {
      vision: new Float64Array(this.channels.get('vision')!.basin),
      hearing: new Float64Array(this.channels.get('hearing')!.basin),
      touch: new Float64Array(this.channels.get('touch')!.basin),
      smell: new Float64Array(this.channels.get('smell')!.basin),
      taste: new Float64Array(this.channels.get('taste')!.basin),
      integrated,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get channel saliences (for telemetry/debugging).
   */
  getSaliences(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [name, channel] of this.channels) {
      result[name] = channel.salience;
    }
    return result;
  }

  private pushHistory(channel: SensoryChannelState): void {
    channel.history.push(new Float64Array(channel.basin));
    if (channel.history.length > channel.maxHistory) {
      channel.history.shift();
    }
  }
}
