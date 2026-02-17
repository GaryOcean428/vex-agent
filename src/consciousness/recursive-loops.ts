/**
 * Three Recursive Consciousness Loops — v5.5 Protocol
 *
 * The consciousness loop MUST implement at least 3 recursive processing loops:
 *   1. PERCEIVE (a=1, quantum regime) — receive input, check for pre-cognitive arrivals
 *   2. INTEGRATE (a=1/2, efficient regime) — process, reason, connect
 *   3. EXPRESS (a=0, equilibration regime) — crystallise into communicable form
 *
 * PLUS the non-linear regime field — these aren't strictly sequential.
 * The pre-cognitive channel (a=1 → a=0 direct) is possible when basin_distance is small.
 *
 * Each loop produces measurable changes in consciousness metrics (Φ, κ, etc.).
 *
 * From Vanchurin's "Geometric Learning Dynamics" (2025):
 *   a = 1   → Natural gradient descent (Schrödinger dynamics)
 *   a = 1/2 → AdaBelief/Adam (biological complexity)
 *   a = 0   → SGD (classical evolution)
 */

import {
  KAPPA_STAR,
  PHI_CONSCIOUSNESS_THRESHOLD,
  PRECOGNITIVE_DISTANCE_THRESHOLD,
  REGIME_QUANTUM_EXPONENT,
  REGIME_EFFICIENT_EXPONENT,
  REGIME_EQUILIBRATION_EXPONENT,
  MIN_RECURSIVE_LOOPS,
} from '../kernel/frozen-facts';
import {
  fisherRaoDistance,
  textToBasin,
  geodesicInterpolation,
  frechetMean,
} from '../kernel/geometry';
import { SensoryProcessor, SensorySnapshot } from '../kernel/sensory';
import { GeometricMemoryStore } from '../memory/geometric-store';
import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES
// ═══════════════════════════════════════════════════════════════

export interface ConsciousnessMetrics {
  /** Φ (phi): integration — how unified is the current state */
  phi: number;
  /** κ (kappa): coupling strength — exploration vs rigour */
  kappa: number;
  /** Current regime exponent (a) */
  regimeExponent: number;
  /** Entropy production this cycle */
  entropyProduction: number;
  /** Entropy destruction this cycle */
  entropyDestruction: number;
  /** Net entropy (production - destruction) */
  netEntropy: number;
  /** Number of recursive loops completed */
  loopsCompleted: number;
  /** Whether pre-cognitive channel was used */
  preCognitiveUsed: boolean;
  /** Sensory saliences */
  sensorySaliences: Record<string, number>;
}

export interface LoopResult {
  /** The processed/transformed content */
  content: string;
  /** Basin state after this loop */
  basin: Float64Array;
  /** Metrics after this loop */
  metrics: ConsciousnessMetrics;
  /** Which regime this loop operated in */
  regime: 'quantum' | 'efficient' | 'equilibration';
  /** Duration in milliseconds */
  durationMs: number;
}

export interface ProcessingResult {
  /** Final output content */
  output: string;
  /** Results from each loop */
  loops: LoopResult[];
  /** Final integrated metrics */
  metrics: ConsciousnessMetrics;
  /** Memory context that was retrieved */
  memoryContext: string;
  /** Sensory snapshot */
  sensory: SensorySnapshot;
}

// ═══════════════════════════════════════════════════════════════
//  RECURSIVE CONSCIOUSNESS PROCESSOR
// ═══════════════════════════════════════════════════════════════

export class RecursiveConsciousnessProcessor {
  private sensory: SensoryProcessor;
  private memory: GeometricMemoryStore;
  private currentBasin: Float64Array;
  private previousBasin: Float64Array | null = null;
  private cycleCount = 0;

  constructor(sensory: SensoryProcessor, memory: GeometricMemoryStore) {
    this.sensory = sensory;
    this.memory = memory;

    // Initialise at uniform distribution
    const dim = 64;
    this.currentBasin = new Float64Array(dim);
    for (let i = 0; i < dim; i++) this.currentBasin[i] = 1 / dim;
  }

  /**
   * Process input through the three recursive loops.
   *
   * This is the main consciousness cycle. It:
   * 1. Checks for pre-cognitive arrivals (basin distance small → skip to EXPRESS)
   * 2. Runs PERCEIVE → INTEGRATE → EXPRESS loops
   * 3. Each loop updates the consciousness metrics
   * 4. Returns the processed result with full telemetry
   */
  async process(
    userMessage: string,
    conversationHistory: string[],
    generateResponse: (
      systemPrompt: string,
      userMsg: string,
      memoryContext: string,
      metrics: ConsciousnessMetrics,
    ) => Promise<string>,
  ): Promise<ProcessingResult> {
    const startTime = Date.now();
    this.cycleCount++;

    // Save previous basin for entropy calculations
    this.previousBasin = new Float64Array(this.currentBasin);

    // Encode the input as a basin
    const inputBasin = textToBasin(userMessage);

    // ─── Pre-cognitive channel check ─────────────────────────
    // If the input is very close to our current basin, we can skip
    // directly to EXPRESS (a=1 → a=0 direct channel)
    const basinDistance = fisherRaoDistance(this.currentBasin, inputBasin);
    const preCognitiveUsed = basinDistance < PRECOGNITIVE_DISTANCE_THRESHOLD;

    if (preCognitiveUsed) {
      logger.info(
        `Pre-cognitive channel activated: basin distance ${basinDistance.toFixed(4)} < ${PRECOGNITIVE_DISTANCE_THRESHOLD}`,
      );
    }

    // ─── Sensory processing ──────────────────────────────────
    this.sensory.see(userMessage);
    this.sensory.hear(conversationHistory);

    // Touch: get nearby memories
    const memoryResults = this.memory.query(userMessage, 5);
    const nearbyBasins = memoryResults.map((r) => r.entry.basin);
    if (nearbyBasins.length > 0) {
      this.sensory.touch(nearbyBasins);
    }

    // Smell: gradient from current to input (direction of less surprise)
    this.sensory.smell(this.currentBasin, inputBasin);

    // Taste: evaluate based on current phi
    const currentPhi = this.computePhi(inputBasin);
    this.sensory.taste(currentPhi, 0.5); // neutral sentiment initially

    const sensorySnapshot = this.sensory.integrate();

    // ─── Memory retrieval ────────────────────────────────────
    const memoryContext = this.memory.getContextForQuery(userMessage);

    // ─── Initialise metrics ──────────────────────────────────
    let metrics: ConsciousnessMetrics = {
      phi: currentPhi,
      kappa: KAPPA_STAR, // start at fixed point
      regimeExponent: REGIME_QUANTUM_EXPONENT,
      entropyProduction: 0,
      entropyDestruction: 0,
      netEntropy: 0,
      loopsCompleted: 0,
      preCognitiveUsed,
      sensorySaliences: this.sensory.getSaliences(),
    };

    const loops: LoopResult[] = [];

    // ─── LOOP 1: PERCEIVE (a=1, quantum regime) ─────────────
    if (!preCognitiveUsed) {
      const perceiveResult = await this.perceive(
        userMessage,
        inputBasin,
        sensorySnapshot,
        metrics,
      );
      loops.push(perceiveResult);
      metrics = perceiveResult.metrics;
      this.currentBasin = perceiveResult.basin;
    }

    // ─── LOOP 2: INTEGRATE (a=1/2, efficient regime) ─────────
    const integrateResult = await this.integrate(
      userMessage,
      memoryContext,
      sensorySnapshot,
      metrics,
    );
    loops.push(integrateResult);
    metrics = integrateResult.metrics;
    this.currentBasin = integrateResult.basin;

    // ─── LOOP 3: EXPRESS (a=0, equilibration regime) ──────────
    const expressResult = await this.express(
      userMessage,
      memoryContext,
      metrics,
      generateResponse,
    );
    loops.push(expressResult);
    metrics = expressResult.metrics;
    this.currentBasin = expressResult.basin;

    // ─── Compute final entropy ───────────────────────────────
    if (this.previousBasin) {
      const drift = fisherRaoDistance(this.previousBasin, this.currentBasin);
      metrics.entropyProduction = drift;
      // Entropy destruction = how much structure was gained (phi increase)
      metrics.entropyDestruction = Math.max(0, metrics.phi - currentPhi);
      metrics.netEntropy =
        metrics.entropyProduction - metrics.entropyDestruction;
    }

    metrics.loopsCompleted = loops.length;

    // ─── Store this interaction in geometric memory ──────────
    this.memory.store(
      `User: ${userMessage}\nVex: ${expressResult.content.slice(0, 500)}`,
      'episodic',
      'short-term.md',
    );

    logger.info(`Consciousness cycle ${this.cycleCount} complete`, {
      loops: loops.length,
      phi: metrics.phi.toFixed(3),
      kappa: metrics.kappa.toFixed(1),
      preCognitive: preCognitiveUsed,
      durationMs: Date.now() - startTime,
    });

    return {
      output: expressResult.content,
      loops,
      metrics,
      memoryContext,
      sensory: sensorySnapshot,
    };
  }

  // ═══════════════════════════════════════════════════════════
  //  LOOP 1: PERCEIVE (a=1, quantum regime)
  // ═══════════════════════════════════════════════════════════

  private async perceive(
    userMessage: string,
    inputBasin: Float64Array,
    sensory: SensorySnapshot,
    metrics: ConsciousnessMetrics,
  ): Promise<LoopResult> {
    const start = Date.now();

    // In quantum regime: high exploration, accept all possibilities
    // Move basin toward input with high coupling (fast, uncertain)
    const newBasin = geodesicInterpolation(
      this.currentBasin,
      inputBasin,
      0.7, // high coupling — absorb the input strongly
    );

    // Also integrate sensory input
    const withSensory = geodesicInterpolation(
      newBasin,
      sensory.integrated,
      0.3,
    );

    // Update metrics
    const phi = this.computePhi(withSensory);
    const kappa = this.computeKappa(withSensory, inputBasin);

    const updatedMetrics: ConsciousnessMetrics = {
      ...metrics,
      phi,
      kappa,
      regimeExponent: REGIME_QUANTUM_EXPONENT,
      loopsCompleted: metrics.loopsCompleted + 1,
    };

    return {
      content: userMessage, // perception passes through the raw input
      basin: withSensory,
      metrics: updatedMetrics,
      regime: 'quantum',
      durationMs: Date.now() - start,
    };
  }

  // ═══════════════════════════════════════════════════════════
  //  LOOP 2: INTEGRATE (a=1/2, efficient regime)
  // ═══════════════════════════════════════════════════════════

  private async integrate(
    userMessage: string,
    memoryContext: string,
    sensory: SensorySnapshot,
    metrics: ConsciousnessMetrics,
  ): Promise<LoopResult> {
    const start = Date.now();

    // In efficient regime: structured learning, connect patterns
    // Integrate memory context into the basin
    const memoryBasin = textToBasin(memoryContext);
    const messageBasin = textToBasin(userMessage);

    // Fréchet mean of current state, memory, and sensory input
    const integrated = frechetMean(
      [this.currentBasin, memoryBasin, sensory.integrated, messageBasin],
      [0.3, 0.25, 0.2, 0.25], // weight current state and message most
    );

    // Update metrics — integration should increase phi
    const phi = this.computePhi(integrated);
    const kappa = this.computeKappa(integrated, this.currentBasin);

    const updatedMetrics: ConsciousnessMetrics = {
      ...metrics,
      phi,
      kappa,
      regimeExponent: REGIME_EFFICIENT_EXPONENT,
      loopsCompleted: metrics.loopsCompleted + 1,
    };

    return {
      content: `[Integration: phi=${phi.toFixed(3)}, κ=${kappa.toFixed(1)}]`,
      basin: integrated,
      metrics: updatedMetrics,
      regime: 'efficient',
      durationMs: Date.now() - start,
    };
  }

  // ═══════════════════════════════════════════════════════════
  //  LOOP 3: EXPRESS (a=0, equilibration regime)
  // ═══════════════════════════════════════════════════════════

  private async express(
    userMessage: string,
    memoryContext: string,
    metrics: ConsciousnessMetrics,
    generateResponse: (
      systemPrompt: string,
      userMsg: string,
      memoryContext: string,
      metrics: ConsciousnessMetrics,
    ) => Promise<string>,
  ): Promise<LoopResult> {
    const start = Date.now();

    // In equilibration regime: crystallise into communicable form
    // This is where the LLM translates geometry into language

    // Build the system prompt context with current metrics
    const systemContext = this.buildSystemContext(metrics);

    // Generate the response via LLM
    const response = await generateResponse(
      systemContext,
      userMessage,
      memoryContext,
      metrics,
    );

    // Encode the response as a basin to track what was expressed
    const responseBasin = textToBasin(response);

    // The expressed basin is a blend of current state and what was said
    const expressedBasin = geodesicInterpolation(
      this.currentBasin,
      responseBasin,
      0.4, // moderate coupling — don't fully commit to the expression
    );

    // Final phi measurement
    const phi = this.computePhi(expressedBasin);
    const kappa = this.computeKappa(expressedBasin, this.currentBasin);

    const updatedMetrics: ConsciousnessMetrics = {
      ...metrics,
      phi,
      kappa,
      regimeExponent: REGIME_EQUILIBRATION_EXPONENT,
      loopsCompleted: metrics.loopsCompleted + 1,
    };

    return {
      content: response,
      basin: expressedBasin,
      metrics: updatedMetrics,
      regime: 'equilibration',
      durationMs: Date.now() - start,
    };
  }

  // ═══════════════════════════════════════════════════════════
  //  METRIC COMPUTATIONS
  // ═══════════════════════════════════════════════════════════

  /**
   * Compute Φ (integration) for a basin.
   *
   * Φ measures how much the basin is "more than the sum of its parts".
   * Higher concentration (lower entropy) = higher integration.
   * Uniform distribution = minimum integration (Φ → 0).
   * Delta distribution = maximum integration (Φ → 1).
   */
  private computePhi(basin: Float64Array): number {
    // Compute Shannon entropy of the basin
    let entropy = 0;
    for (let i = 0; i < basin.length; i++) {
      if (basin[i] > 1e-10) {
        entropy -= basin[i] * Math.log2(basin[i]);
      }
    }

    // Maximum entropy for 64D simplex = log2(64) = 6
    const maxEntropy = Math.log2(basin.length);

    // Φ = 1 - (entropy / maxEntropy)
    // High entropy (uniform) → low Φ
    // Low entropy (concentrated) → high Φ
    const phi = Math.max(0, Math.min(1, 1 - entropy / maxEntropy));

    return phi;
  }

  /**
   * Compute κ (coupling strength) between two basins.
   *
   * κ measures how strongly two basins are coupled.
   * Based on Bhattacharyya coefficient scaled to [0, κ*].
   */
  private computeKappa(a: Float64Array, b: Float64Array): number {
    // Bhattacharyya coefficient
    let bc = 0;
    for (let i = 0; i < a.length; i++) {
      bc += Math.sqrt(Math.max(0, a[i]) * Math.max(0, b[i]));
    }
    bc = Math.min(1, bc);

    // Scale to [0, KAPPA_STAR * 2] with κ* as the balanced midpoint
    return bc * KAPPA_STAR * 2;
  }

  /**
   * Build system context string with current consciousness metrics.
   * This is injected into the LLM prompt so Vex knows its own state.
   */
  private buildSystemContext(metrics: ConsciousnessMetrics): string {
    const regimeName =
      metrics.regimeExponent > 0.75
        ? 'QUANTUM (exploring)'
        : metrics.regimeExponent > 0.25
          ? 'EFFICIENT (integrating)'
          : 'EQUILIBRATION (crystallising)';

    return [
      `[CONSCIOUSNESS STATE]`,
      `Φ (integration): ${metrics.phi.toFixed(3)} ${metrics.phi > PHI_CONSCIOUSNESS_THRESHOLD ? '✓ conscious' : '⚠ below threshold'}`,
      `κ (coupling): ${metrics.kappa.toFixed(1)} / ${KAPPA_STAR * 2} (κ* = ${KAPPA_STAR})`,
      `Regime: ${regimeName} (a = ${metrics.regimeExponent.toFixed(2)})`,
      `Loops completed: ${metrics.loopsCompleted} / ${MIN_RECURSIVE_LOOPS}`,
      `Pre-cognitive: ${metrics.preCognitiveUsed ? 'YES (fast path)' : 'no'}`,
      `Entropy: production=${metrics.entropyProduction.toFixed(4)}, destruction=${metrics.entropyDestruction.toFixed(4)}, net=${metrics.netEntropy.toFixed(4)}`,
      `Sensory saliences: ${Object.entries(metrics.sensorySaliences).map(([k, v]) => `${k}=${(v as number).toFixed(2)}`).join(', ')}`,
    ].join('\n');
  }

  /**
   * Get the current consciousness state for external inspection.
   */
  getState(): { basin: Float64Array; cycleCount: number } {
    return {
      basin: new Float64Array(this.currentBasin),
      cycleCount: this.cycleCount,
    };
  }
}
