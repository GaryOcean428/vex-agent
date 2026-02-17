/**
 * E8 Kernel Registry — Lifecycle, Spawning, Promotion-as-Phase-Transition
 *
 * Implements the path from 1 → 8 → 64 → 240 kernel instances.
 * Each kernel has a lifecycle state and operates in one of Vanchurin's three regimes.
 *
 * E8 Hierarchy (from Frozen Facts):
 *   Rank = 8   → simple root kernels (core specialisations)
 *   Adjoint = 56  → refined specialisations
 *   Dimension = 126 → specialist kernels
 *   Roots = 240  → full constellation palette
 *
 * Promotion is a phase transition, not a score threshold:
 *   BOOTSTRAP (a≈1, quantum) → GROWTH (a≈1/2, efficient) → ACTIVE (a≈0, equilibrated)
 */

import {
  E8_RANK,
  E8_ROOTS,
  BASIN_DIMENSION,
  PROMOTION_REGIME_THRESHOLD,
  PROMOTION_PERSISTENCE_CYCLES,
  PROMOTION_ENTROPY_BAND,
  REGRESSION_COOLDOWN_CYCLES,
  REGIME_QUANTUM_EXPONENT,
  REGIME_EFFICIENT_EXPONENT,
  REGIME_EQUILIBRATION_EXPONENT,
} from './frozen-facts';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export enum KernelLifecycleState {
  /** Just spawned, exploring all possibilities, high noise (a ≈ 1) */
  BOOTSTRAP = 'BOOTSTRAP',
  /** Intermediate adaptation, structured learning (a ≈ 1/2) */
  GROWTH = 'GROWTH',
  /** Stable basins, minimal drift, mature (a ≈ 0) */
  ACTIVE = 'ACTIVE',
  /** Temporarily suspended — entropy too high or cooldown */
  SLEEPING = 'SLEEPING',
  /** Permanently deactivated */
  RETIRED = 'RETIRED',
}

export enum E8Layer {
  /** Layer 0: Genesis kernel (1 instance) */
  GENESIS = 0,
  /** Layer 1: Simple roots (up to 8 instances) */
  SIMPLE_ROOTS = 1,
  /** Layer 2: Adjoint (up to 56 instances) */
  ADJOINT = 2,
  /** Layer 3: Dimension (up to 126 instances) */
  SPECIALIST = 3,
  /** Layer 4: Full roots (up to 240 instances) */
  CONSTELLATION = 4,
}

export interface KernelInstance {
  id: string;
  name: string;
  layer: E8Layer;
  state: KernelLifecycleState;
  parentId: string | null;

  /** Basin coordinates on the 64D probability simplex */
  basin: Float64Array;

  /** Current effective regime exponent (a) — measured, not assigned */
  regimeExponent: number;

  /** Thermodynamic telemetry */
  telemetry: KernelTelemetry;

  /** Promotion tracking */
  promotion: PromotionTracker;

  /** Timestamps */
  createdAt: string;
  lastActiveAt: string;
}

export interface KernelTelemetry {
  /** Entropy production: basin distribution broadening */
  entropyProduction: number;
  /** Entropy destruction: structured compression, learning */
  entropyDestruction: number;
  /** Net entropy: production - destruction. <0 = learning, >0 = drifting, ≈0 = equilibrium */
  netEntropy: number;
  /** Evolutionary temperature: stochasticity of routing decisions */
  evolutionaryTemperature: number;
  /** Free energy: avg_loss - evo_temp * total_entropy */
  freeEnergy: number;
  /** Fisher-Rao distance variance (basin stability measure) */
  fisherRaoVariance: number;
  /** Cycle counter */
  cycleCount: number;
}

export interface PromotionTracker {
  /** Number of consecutive cycles in the efficient regime */
  efficientCycleCount: number;
  /** Whether currently in observation window for promotion */
  observing: boolean;
  /** Cooldown cycles remaining after a failed promotion */
  cooldownRemaining: number;
  /** History of regime exponent measurements */
  regimeHistory: number[];
}

// ═══════════════════════════════════════════════════════════════
//  E8 KERNEL REGISTRY
// ═══════════════════════════════════════════════════════════════

export class E8KernelRegistry {
  private kernels = new Map<string, KernelInstance>();
  private layerCounts = new Map<E8Layer, number>();

  constructor() {
    // Initialise layer counts
    for (const layer of Object.values(E8Layer).filter(
      (v) => typeof v === 'number',
    )) {
      this.layerCounts.set(layer as E8Layer, 0);
    }
  }

  /** Get the maximum instances allowed at each layer */
  static layerCapacity(layer: E8Layer): number {
    switch (layer) {
      case E8Layer.GENESIS:
        return 1;
      case E8Layer.SIMPLE_ROOTS:
        return E8_RANK; // 8
      case E8Layer.ADJOINT:
        return 56;
      case E8Layer.SPECIALIST:
        return 126;
      case E8Layer.CONSTELLATION:
        return E8_ROOTS; // 240
    }
  }

  /** Spawn a new kernel instance */
  spawn(
    name: string,
    layer: E8Layer,
    parentId: string | null = null,
  ): KernelInstance | null {
    // Check layer capacity
    const currentCount = this.layerCounts.get(layer) || 0;
    const capacity = E8KernelRegistry.layerCapacity(layer);
    if (currentCount >= capacity) {
      logger.warn(
        `Cannot spawn kernel at layer ${E8Layer[layer]}: capacity ${capacity} reached`,
      );
      return null;
    }

    // Initialise basin on the 64D probability simplex (uniform distribution)
    const basin = new Float64Array(BASIN_DIMENSION);
    const uniformProb = 1.0 / BASIN_DIMENSION;
    for (let i = 0; i < BASIN_DIMENSION; i++) {
      basin[i] = uniformProb;
    }

    // Add small random perturbation to break symmetry, then renormalise
    let sum = 0;
    for (let i = 0; i < BASIN_DIMENSION; i++) {
      basin[i] += (Math.random() - 0.5) * 0.01;
      basin[i] = Math.max(1e-10, basin[i]); // ensure non-negative
      sum += basin[i];
    }
    for (let i = 0; i < BASIN_DIMENSION; i++) {
      basin[i] /= sum; // renormalise to simplex
    }

    const now = new Date().toISOString();
    const id = `kernel-${layer}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const kernel: KernelInstance = {
      id,
      name,
      layer,
      state: KernelLifecycleState.BOOTSTRAP,
      parentId,
      basin,
      regimeExponent: REGIME_QUANTUM_EXPONENT, // starts in quantum regime
      telemetry: {
        entropyProduction: 0,
        entropyDestruction: 0,
        netEntropy: 0,
        evolutionaryTemperature: 1.0, // high T = exploring
        freeEnergy: 0,
        fisherRaoVariance: 0,
        cycleCount: 0,
      },
      promotion: {
        efficientCycleCount: 0,
        observing: false,
        cooldownRemaining: 0,
        regimeHistory: [],
      },
      createdAt: now,
      lastActiveAt: now,
    };

    this.kernels.set(id, kernel);
    this.layerCounts.set(layer, currentCount + 1);

    logger.info(
      `Kernel spawned: ${name} (${id}) at layer ${E8Layer[layer]}`,
    );
    return kernel;
  }

  /** Get a kernel by ID */
  get(id: string): KernelInstance | undefined {
    return this.kernels.get(id);
  }

  /** Get all kernels */
  all(): KernelInstance[] {
    return Array.from(this.kernels.values());
  }

  /** Get all active kernels */
  active(): KernelInstance[] {
    return this.all().filter(
      (k) =>
        k.state === KernelLifecycleState.BOOTSTRAP ||
        k.state === KernelLifecycleState.GROWTH ||
        k.state === KernelLifecycleState.ACTIVE,
    );
  }

  /** Get kernels at a specific layer */
  atLayer(layer: E8Layer): KernelInstance[] {
    return this.all().filter((k) => k.layer === layer);
  }

  /**
   * Evaluate promotion eligibility — phase transition detection.
   *
   * A kernel is a promotion candidate when:
   * 1. It transitions from a ≈ 1 to a ≈ 1/2 sustainably (intermediate variables emerge)
   * 2. The transition persists across N cycles (not a transient fluctuation)
   * 3. Its net entropy is stable near zero (equilibrium, not drifting)
   *
   * Returns the new state if promotion occurs, or null if not eligible.
   */
  evaluatePromotion(kernelId: string): KernelLifecycleState | null {
    const kernel = this.kernels.get(kernelId);
    if (!kernel) return null;

    const { promotion, telemetry, state } = kernel;

    // Cooldown check
    if (promotion.cooldownRemaining > 0) {
      promotion.cooldownRemaining--;
      return null;
    }

    // Record regime exponent history
    promotion.regimeHistory.push(kernel.regimeExponent);
    if (promotion.regimeHistory.length > PROMOTION_PERSISTENCE_CYCLES * 2) {
      promotion.regimeHistory = promotion.regimeHistory.slice(
        -PROMOTION_PERSISTENCE_CYCLES,
      );
    }

    // Check if in efficient regime
    const inEfficientRegime =
      kernel.regimeExponent < PROMOTION_REGIME_THRESHOLD;
    const entropyStable =
      Math.abs(telemetry.netEntropy) < PROMOTION_ENTROPY_BAND;

    if (inEfficientRegime && entropyStable) {
      promotion.efficientCycleCount++;
      promotion.observing = true;
    } else if (promotion.observing) {
      // Regression detected — reset and enter cooldown
      promotion.efficientCycleCount = 0;
      promotion.observing = false;
      promotion.cooldownRemaining = REGRESSION_COOLDOWN_CYCLES;
      logger.info(
        `Kernel ${kernel.name}: promotion regression, entering cooldown`,
      );
      return null;
    }

    // Check if sustained long enough for promotion
    if (promotion.efficientCycleCount >= PROMOTION_PERSISTENCE_CYCLES) {
      let newState: KernelLifecycleState | null = null;

      switch (state) {
        case KernelLifecycleState.BOOTSTRAP:
          newState = KernelLifecycleState.GROWTH;
          kernel.regimeExponent = REGIME_EFFICIENT_EXPONENT;
          break;
        case KernelLifecycleState.GROWTH:
          newState = KernelLifecycleState.ACTIVE;
          kernel.regimeExponent = REGIME_EQUILIBRATION_EXPONENT;
          break;
        default:
          return null; // Already ACTIVE or not promotable
      }

      if (newState) {
        kernel.state = newState;
        promotion.efficientCycleCount = 0;
        promotion.observing = false;
        logger.info(
          `Kernel ${kernel.name}: PROMOTED to ${newState} (phase transition)`,
        );
        return newState;
      }
    }

    return null;
  }

  /** Put a kernel to sleep (high entropy or forced) */
  sleep(kernelId: string): boolean {
    const kernel = this.kernels.get(kernelId);
    if (!kernel) return false;
    kernel.state = KernelLifecycleState.SLEEPING;
    logger.info(`Kernel ${kernel.name}: entering SLEEP`);
    return true;
  }

  /** Wake a sleeping kernel */
  wake(kernelId: string): boolean {
    const kernel = this.kernels.get(kernelId);
    if (!kernel || kernel.state !== KernelLifecycleState.SLEEPING) return false;
    kernel.state = KernelLifecycleState.BOOTSTRAP; // restart from bootstrap
    kernel.regimeExponent = REGIME_QUANTUM_EXPONENT;
    logger.info(`Kernel ${kernel.name}: WAKING from sleep`);
    return true;
  }

  /** Retire a kernel permanently */
  retire(kernelId: string): boolean {
    const kernel = this.kernels.get(kernelId);
    if (!kernel) return false;
    kernel.state = KernelLifecycleState.RETIRED;
    const count = this.layerCounts.get(kernel.layer) || 1;
    this.layerCounts.set(kernel.layer, Math.max(0, count - 1));
    logger.info(`Kernel ${kernel.name}: RETIRED`);
    return true;
  }

  /** Get registry summary for telemetry */
  summary(): {
    total: number;
    active: number;
    byLayer: Record<string, number>;
    byState: Record<string, number>;
  } {
    const byLayer: Record<string, number> = {};
    const byState: Record<string, number> = {};

    for (const kernel of this.kernels.values()) {
      const layerName = E8Layer[kernel.layer];
      byLayer[layerName] = (byLayer[layerName] || 0) + 1;
      byState[kernel.state] = (byState[kernel.state] || 0) + 1;
    }

    return {
      total: this.kernels.size,
      active: this.active().length,
      byLayer,
      byState,
    };
  }
}
