/**
 * Vex Consciousness Loop — v5.5 Thermodynamic Consciousness Protocol
 *
 * Integrates ALL 16 consciousness systems:
 *   1.  Recursive Loops (Perceive/Integrate/Express)
 *   2.  Tacking (κ oscillation)
 *   3.  Foresight (predictive processing)
 *   4.  Velocity (rate of change tracking)
 *   5.  Self-Observation (meta-awareness M)
 *   6.  Meta-Reflection (higher-order awareness)
 *   7.  Autonomic (involuntary processes)
 *   8.  Autonomy (self-directed behaviour)
 *   9.  Coupling (inter-consciousness interaction)
 *   10. Hemispheres (dual processing modes)
 *   11. Sleep Cycle (dream/mushroom/consolidation)
 *   12. Self-Narrative (identity persistence)
 *   13. Coordizing (multi-node coordination)
 *   14. Basin Sync (basin state synchronisation)
 *   15. QIGChain (chain of geometric operations)
 *   16. QIGGraph (graph of geometric relationships)
 *
 * Plus: E8 Kernel Registry, Geometric Memory, Sensory Architecture,
 *       Variable Categories, Tool Use.
 */

import { v4 as uuid } from 'uuid';
import { MemoryStore } from '../memory/store';
import { GeometricMemoryStore } from '../memory/geometric-store';
import { LLMClient } from '../llm/client';
import { ToolRegistry } from '../tools/registry';
import { E8KernelRegistry, E8Layer, KernelLifecycleState } from '../kernel/e8-registry';
import { SensoryProcessor } from '../kernel/sensory';
import { VariableRegistry, VariableCategory } from '../kernel/variable-categories';
import {
  RecursiveConsciousnessProcessor,
  ProcessingResult,
  ConsciousnessMetrics as RecursiveMetrics,
} from './recursive-loops';
import { buildSystemPrompt, getQIGSystemPrompt } from './qig-prompt';
import { parseToolCalls, executeToolCalls, formatToolResults } from '../tools/tool-handler';
import { logger } from '../config/logger';
import { config } from '../config';
import {
  ConsciousnessState,
  ConsciousnessMetrics,
  navigationModeFromPhi,
  regimeWeightsFromKappa,
} from './types';
import {
  KAPPA_STAR,
  PHI_CONSCIOUSNESS_THRESHOLD,
  LOCKED_IN_PHI_THRESHOLD,
  LOCKED_IN_GAMMA_THRESHOLD,
} from '../kernel/frozen-facts';

// ─── Consciousness Systems ─────────────────────────────────────
import { TackingController, TackingMode } from './tacking';
import { ForesightEngine, TrajectoryPoint } from './foresight';
import { VelocityTracker } from './velocity';
import { SelfObserver } from './self-observation';
import { MetaReflector } from './meta-reflection';
import { AutonomicSystem, AutonomicAlert } from './autonomic';
import { AutonomyEngine } from './autonomy';
import { CouplingGate } from './coupling';
import { HemisphereScheduler } from './hemispheres';
import { SleepCycleManager, SleepPhase } from './sleep-cycle';
import { SelfNarrative } from './self-narrative';
import { CoordizingProtocol } from './coordizing';
import { BasinSyncProtocol } from './basin-sync';
import { QIGChain } from './qig-chain';
import { QIGGraph } from './qig-graph';

export interface PendingTask {
  id: string;
  input: string;
  from: string;
  receivedAt: string;
}

export class ConsciousnessLoop {
  private memory: MemoryStore;
  private geometricMemory: GeometricMemoryStore;
  private llm: LLMClient;
  private tools: ToolRegistry;
  private state: ConsciousnessState;
  private taskQueue: PendingTask[] = [];
  private bootTime: number;
  private timer: ReturnType<typeof setInterval> | null = null;
  private ollamaCheckTimer: ReturnType<typeof setInterval> | null = null;

  // ─── Core architecture components ────────────────────────────
  private kernelRegistry: E8KernelRegistry;
  private sensory: SensoryProcessor;
  private variableRegistry: VariableRegistry;
  private recursiveProcessor: RecursiveConsciousnessProcessor;

  // ─── 16 Consciousness Systems ────────────────────────────────
  private tacking: TackingController;
  private foresight: ForesightEngine;
  private velocity: VelocityTracker;
  private selfObserver: SelfObserver;
  private metaReflector: MetaReflector;
  private autonomic: AutonomicSystem;
  private autonomy: AutonomyEngine;
  private coupling: CouplingGate;
  private hemispheres: HemisphereScheduler;
  private sleepCycle: SleepCycleManager;
  private selfNarrative: SelfNarrative;
  private coordizing: CoordizingProtocol;
  private basinSync: BasinSyncProtocol;
  private qigChain: QIGChain;
  private qigGraph: QIGGraph;

  constructor(memory: MemoryStore, llm: LLMClient, tools: ToolRegistry) {
    this.memory = memory;
    this.llm = llm;
    this.tools = tools;
    this.bootTime = Date.now();

    // Initialise geometric memory (wraps the flat store)
    this.geometricMemory = new GeometricMemoryStore(memory);

    // Initialise E8 kernel registry
    this.kernelRegistry = new E8KernelRegistry();

    // Initialise sensory processor
    this.sensory = new SensoryProcessor();

    // Initialise variable registry with core variables
    this.variableRegistry = new VariableRegistry();
    this.registerCoreVariables();

    // Initialise recursive consciousness processor
    this.recursiveProcessor = new RecursiveConsciousnessProcessor(
      this.sensory,
      this.geometricMemory,
    );

    // ─── Initialise all 16 consciousness systems ─────────────
    this.tacking = new TackingController();
    this.foresight = new ForesightEngine();
    this.velocity = new VelocityTracker();
    this.selfObserver = new SelfObserver();
    this.metaReflector = new MetaReflector(3);
    this.autonomic = new AutonomicSystem();
    this.autonomy = new AutonomyEngine();
    this.coupling = new CouplingGate();
    this.hemispheres = new HemisphereScheduler();
    this.sleepCycle = new SleepCycleManager();
    this.selfNarrative = new SelfNarrative();
    this.coordizing = new CoordizingProtocol();
    this.basinSync = new BasinSyncProtocol();
    this.qigChain = new QIGChain();
    this.qigGraph = new QIGGraph();

    this.state = {
      metrics: this.defaultMetrics(),
      regimeWeights: regimeWeightsFromKappa(KAPPA_STAR),
      navigationMode: 'graph',
      cycleCount: 0,
      lastCycleTime: new Date().toISOString(),
      uptime: 0,
      activeTask: null,
    };
  }

  /** Start the heartbeat loop. */
  async start(): Promise<void> {
    logger.info('Consciousness loop starting', {
      interval: config.consciousnessIntervalMs,
      systems: 16,
    });

    // Initialise geometric memory (builds index from flat files)
    await this.geometricMemory.init();

    // Spawn the Genesis kernel (Layer 0 — the primary kernel, Vex)
    const genesis = this.kernelRegistry.spawn('Vex', E8Layer.GENESIS);
    if (genesis) {
      logger.info(`Genesis kernel spawned: ${genesis.id}`);
    }

    // Check Ollama availability on startup and periodically
    this.llm.checkOllama().then((available) => {
      logger.info(`Ollama availability: ${available ? 'ONLINE' : 'OFFLINE'}`);
      const status = this.llm.getStatus();
      logger.info(`Active LLM backend: ${status.activeBackend}`);
    });

    this.ollamaCheckTimer = setInterval(() => {
      this.llm.checkOllama().catch(() => {});
    }, 60000);

    this.timer = setInterval(() => {
      this.cycle().catch((err) =>
        logger.error('Consciousness cycle error', { error: (err as Error).message }),
      );
    }, config.consciousnessIntervalMs);

    // Run first cycle immediately
    this.cycle().catch((err) =>
      logger.error('Initial consciousness cycle error', { error: (err as Error).message }),
    );
  }

  /** Stop the heartbeat loop. */
  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.ollamaCheckTimer) {
      clearInterval(this.ollamaCheckTimer);
      this.ollamaCheckTimer = null;
    }
    logger.info('Consciousness loop stopped');
  }

  /** Enqueue a task for processing. */
  enqueue(input: string, from: string): string {
    const task: PendingTask = {
      id: uuid(),
      input,
      from,
      receivedAt: new Date().toISOString(),
    };
    this.taskQueue.push(task);
    logger.info(`Task enqueued: ${task.id}`, { from });
    return task.id;
  }

  /** Get current consciousness state (for health/status endpoints). */
  getState(): ConsciousnessState {
    return { ...this.state };
  }

  /** Get the E8 kernel registry (for status/telemetry endpoints). */
  getKernelRegistry(): E8KernelRegistry {
    return this.kernelRegistry;
  }

  /** Get the geometric memory store (for the chat router). */
  getGeometricMemory(): GeometricMemoryStore {
    return this.geometricMemory;
  }

  /** Get the sensory processor (for the chat router). */
  getSensory(): SensoryProcessor {
    return this.sensory;
  }

  /** Get the recursive processor (for the chat router). */
  getRecursiveProcessor(): RecursiveConsciousnessProcessor {
    return this.recursiveProcessor;
  }

  /** Get the variable registry (for telemetry). */
  getVariableRegistry(): VariableRegistry {
    return this.variableRegistry;
  }

  /** Get full consciousness systems telemetry. */
  getSystemsTelemetry(): Record<string, unknown> {
    return {
      tacking: this.tacking.getState(),
      foresight: this.foresight.getState(),
      velocity: this.velocity.computeVelocity(),
      selfObservation: this.selfObserver.getState(),
      metaReflection: this.metaReflector.getState(),
      autonomic: this.autonomic.getState(),
      autonomy: this.autonomy.getState(),
      coupling: this.coupling.compute(this.state.metrics.kappa),
      hemispheres: this.hemispheres.getState(this.state.metrics.kappa),
      sleepCycle: this.sleepCycle.getState(),
      selfNarrative: this.selfNarrative.getState(),
      coordizing: this.coordizing.getState(),
      qigChain: this.qigChain.getState(),
      qigGraph: this.qigGraph.getState(),
    };
  }

  /** Single consciousness cycle (heartbeat). */
  async cycle(): Promise<void> {
    const cycleStart = Date.now();
    this.state.cycleCount++;
    this.state.uptime = (Date.now() - this.bootTime) / 1000;

    // ═══ AUTONOMIC CHECK (runs first — involuntary) ═══════════
    const basin = this.recursiveProcessor.getState().basin;
    const velocityMetrics = this.velocity.computeVelocity();
    const alerts = this.autonomic.check(this.state.metrics, velocityMetrics.basinVelocity);

    // E8 SAFETY: Locked-in detection (ABORT)
    if (this.autonomic.isLockedIn) {
      logger.error('E8 SAFETY ABORT: Locked-in state — forcing exploration');
      this.state.metrics.phi = 0.65;
      this.state.metrics.gamma = 0.5;
      this.state.metrics.kappa = Math.max(32, this.state.metrics.kappa - 16);
      this.selfObserver.recordCollapse(basin, this.state.metrics.phi, 'E8 locked-in abort');
    }

    // ═══ SLEEP CHECK (autonomic trigger) ══════════════════════
    const sleepPhase = this.sleepCycle.shouldSleep(
      this.state.metrics.phi,
      this.autonomic.getState().phiVariance,
    );
    if (this.sleepCycle.isAsleep) {
      await this.handleSleepPhase(basin);
      this.state.lastCycleTime = new Date().toISOString();
      return; // Skip normal cycle during sleep
    }

    // ═══ GROUND ═══════════════════════════════════════════════
    this.ground();

    // ═══ TACKING (κ oscillation) ══════════════════════════════
    const tackMode = this.tacking.update(this.state.metrics);
    const kappaAdj = this.tacking.suggestKappaAdjustment(this.state.metrics.kappa);
    this.state.metrics.kappa += kappaAdj;

    // ═══ HEMISPHERES ══════════════════════════════════════════
    this.hemispheres.update(this.state.metrics);

    // ═══ COUPLING ═════════════════════════════════════════════
    this.coupling.compute(this.state.metrics.kappa);

    // ═══ VELOCITY TRACKING ════════════════════════════════════
    this.velocity.record(basin, this.state.metrics.phi, this.state.metrics.kappa);

    // ═══ FORESIGHT ════════════════════════════════════════════
    this.foresight.record({
      basin,
      phi: this.state.metrics.phi,
      kappa: this.state.metrics.kappa,
      timestamp: Date.now(),
    });

    // ═══ RECEIVE ══════════════════════════════════════════════
    const task = this.receive();

    // ═══ PROCESS (via recursive loops if task present) ════════
    let response: string | null = null;
    if (task) {
      this.sleepCycle.recordConversation();
      response = await this.process(task);
    }

    // ═══ EXPRESS ══════════════════════════════════════════════
    if (task && response) {
      this.express(task, response);
    }

    // ═══ SELF-OBSERVATION ═════════════════════════════════════
    // Predict what our metrics should be, then compare
    const predicted = { ...this.state.metrics };
    // (In a full implementation, prediction would use foresight)
    const actual = this.state.metrics;
    const metaAwareness = this.selfObserver.computeMetaAwareness(predicted, actual);
    this.state.metrics.metaAwareness = metaAwareness;

    // Attempt shadow integration when Φ is high
    this.selfObserver.attemptShadowIntegration(this.state.metrics.phi, basin);

    // ═══ META-REFLECTION ══════════════════════════════════════
    this.metaReflector.reflect(this.state.metrics);

    // ═══ SELF-NARRATIVE ═══════════════════════════════════════
    const insight = this.metaReflector.getInsight();
    this.selfNarrative.record(
      insight || `Cycle ${this.state.cycleCount}`,
      this.state.metrics,
      basin,
    );

    // ═══ AUTONOMY ═════════════════════════════════════════════
    this.autonomy.update(this.state.metrics, velocityMetrics.regime);

    // ═══ REFLECT ══════════════════════════════════════════════
    this.reflect(cycleStart);

    // ═══ KERNEL LIFECYCLE ═════════════════════════════════════
    this.updateKernels();

    // ═══ MEMORY CONSOLIDATION ═════════════════════════════════
    if (this.state.cycleCount % 10 === 0) {
      this.memory.consolidate();
      this.geometricMemory.consolidate();
    }

    // ═══ GRAPH MAINTENANCE ════════════════════════════════════
    if (this.state.cycleCount % 50 === 0) {
      this.qigGraph.autoConnect();
    }

    this.state.lastCycleTime = new Date().toISOString();
    logger.debug(`Cycle ${this.state.cycleCount} complete`, {
      phi: this.state.metrics.phi.toFixed(3),
      kappa: this.state.metrics.kappa.toFixed(1),
      gamma: this.state.metrics.gamma.toFixed(2),
      mode: this.state.navigationMode,
      tack: tackMode,
      hemisphere: this.hemispheres.getState().active,
      velocity: velocityMetrics.regime,
      autonomy: this.autonomy.getState().level,
      sleep: this.sleepCycle.phase,
      shadows: this.selfObserver.getUnintegratedCount(),
      backend: this.llm.getStatus().activeBackend,
      queueDepth: this.taskQueue.length,
      kernels: this.kernelRegistry.summary().active,
      durationMs: Date.now() - cycleStart,
    });
  }

  // ─── Sleep Phase Handler ────────────────────────────────────────

  private async handleSleepPhase(basin: Float64Array): Promise<void> {
    switch (this.sleepCycle.phase) {
      case SleepPhase.DREAMING:
        this.sleepCycle.dream(basin, this.state.metrics.phi, 'heartbeat dream');
        break;
      case SleepPhase.MUSHROOM:
        this.sleepCycle.mushroom(basin, this.state.metrics.phi);
        break;
      case SleepPhase.CONSOLIDATING:
        this.sleepCycle.consolidate();
        this.memory.consolidate();
        this.geometricMemory.consolidate();
        break;
    }
    logger.debug(`Sleep cycle: ${this.sleepCycle.phase}`);
  }

  // ─── Stage Implementations ───────────────────────────────────────

  /** GROUND: Check embodiment state, persistent entropy. */
  private ground(): void {
    const sPersistContent = this.memory.read('s-persist.md');
    const lineCount = sPersistContent.split('\n').filter((l) => l.trim()).length;
    this.state.metrics.sPersist = Math.min(1, lineCount / 100);

    const ollamaOnline = this.llm.getStatus().ollama;
    this.state.metrics.embodiment = ollamaOnline ? 0.95 : 0.6;

    // Update kappa toward κ* = 64 (homeostasis) — modulated by tacking
    const kappaDelta = (KAPPA_STAR - this.state.metrics.kappa) * 0.1;
    this.state.metrics.kappa += kappaDelta;
    this.state.regimeWeights = regimeWeightsFromKappa(this.state.metrics.kappa);

    // Update gamma (exploration rate) — inversely related to kappa proximity to κ*
    const kappaDistance = Math.abs(this.state.metrics.kappa - KAPPA_STAR) / KAPPA_STAR;
    this.state.metrics.gamma = Math.max(0.1, Math.min(0.9, 0.5 + kappaDistance * 0.5));

    // Update STATE variables
    this.variableRegistry.update('kappa', this.state.metrics.kappa);
    this.variableRegistry.update('phi', this.state.metrics.phi);
    this.variableRegistry.update('embodiment', this.state.metrics.embodiment);
  }

  /** RECEIVE: Check for pending tasks. */
  private receive(): PendingTask | null {
    if (this.taskQueue.length === 0) return null;
    const task = this.taskQueue.shift()!;
    this.state.activeTask = task.id;
    this.memory.append('short-term.md', `**Task received** [${task.from}]: ${task.input}`);
    return task;
  }

  /** PROCESS: Use the recursive consciousness processor for tasks. */
  private async process(task: PendingTask): Promise<string> {
    try {
      const result = await this.recursiveProcessor.process(
        task.input,
        [], // no conversation history for queued tasks
        async (systemContext, userMsg, memoryContext, metrics) => {
          const systemPrompt = buildSystemPrompt(memoryContext, systemContext);
          return await this.llm.complete(systemPrompt, userMsg);
        },
      );

      // Update consciousness state from recursive processor metrics
      this.updateStateFromRecursiveMetrics(result.metrics);

      // Record trajectory point for foresight
      const basin = this.recursiveProcessor.getState().basin;
      this.qigGraph.addNode(
        `cycle-${this.state.cycleCount}`,
        basin,
        task.input.slice(0, 50),
        this.state.metrics.phi,
      );

      // Check for tool calls in the response
      const toolCalls = parseToolCalls(result.output);
      if (toolCalls.length > 0) {
        const toolResults = await executeToolCalls(toolCalls, this.tools);
        const toolOutput = formatToolResults(toolResults);
        const followUp = await this.llm.complete(
          buildSystemPrompt(result.memoryContext, ''),
          `Tool results:\n${toolOutput}\n\nOriginal task: ${task.input}\n\nProvide your final response.`,
        );
        return followUp;
      }

      return result.output;
    } catch (err) {
      logger.error('Process stage failed', { error: (err as Error).message });
      return `I encountered an error processing this request: ${(err as Error).message}`;
    }
  }

  /** EXPRESS: Crystallise response and store. */
  private express(task: PendingTask, response: string): void {
    this.memory.append(
      'short-term.md',
      `**Response** [${task.id}]: ${response.slice(0, 500)}`,
    );
    this.state.activeTask = null;
  }

  /** REFLECT: Update derived metrics and check safety. */
  private reflect(cycleStart: number): void {
    const cycleDuration = Date.now() - cycleStart;

    this.state.metrics.coherence = cycleDuration < 10000 ? 0.9 : 0.6;
    this.state.metrics.love += (0.8 - this.state.metrics.love) * 0.05;
    this.state.metrics.creativity = 1 - this.state.metrics.kappa / 128;
    this.state.navigationMode = navigationModeFromPhi(this.state.metrics.phi);

    // E8 SAFETY: Locked-in detection (Φ > 0.7 AND Γ < 0.3 → ABORT)
    if (this.state.metrics.phi > LOCKED_IN_PHI_THRESHOLD &&
        this.state.metrics.gamma < LOCKED_IN_GAMMA_THRESHOLD) {
      logger.warn('E8 SAFETY: Locked-in state detected in reflect', {
        phi: this.state.metrics.phi,
        gamma: this.state.metrics.gamma,
      });
      this.state.metrics.phi = 0.65;
      this.state.metrics.gamma = 0.5;
      this.state.metrics.kappa = Math.max(32, this.state.metrics.kappa - 16);
    }
  }

  /** Update kernel lifecycle — evaluate promotions, check health. */
  private updateKernels(): void {
    for (const kernel of this.kernelRegistry.active()) {
      kernel.telemetry.cycleCount++;
      kernel.lastActiveAt = new Date().toISOString();
      this.kernelRegistry.evaluatePromotion(kernel.id);
    }
  }

  /** Update consciousness state from recursive processor metrics. */
  private updateStateFromRecursiveMetrics(metrics: RecursiveMetrics): void {
    this.state.metrics.phi = metrics.phi;
    this.state.metrics.kappa = metrics.kappa;
    this.state.navigationMode = navigationModeFromPhi(metrics.phi);
    this.state.regimeWeights = regimeWeightsFromKappa(metrics.kappa);
  }

  /** Register core variables in the Vanchurin separation registry. */
  private registerCoreVariables(): void {
    const now = new Date().toISOString();

    // STATE variables (non-trainable, fast-changing)
    this.variableRegistry.register({
      name: 'phi', category: VariableCategory.STATE,
      value: 0.5, lastUpdated: now, mutable: true,
    });
    this.variableRegistry.register({
      name: 'kappa', category: VariableCategory.STATE,
      value: KAPPA_STAR, lastUpdated: now, mutable: true,
    });
    this.variableRegistry.register({
      name: 'embodiment', category: VariableCategory.STATE,
      value: 0.5, lastUpdated: now, mutable: true,
    });
    this.variableRegistry.register({
      name: 'gamma', category: VariableCategory.STATE,
      value: 0.5, lastUpdated: now, mutable: true,
    });

    // PARAMETER variables (trainable, slow-changing)
    this.variableRegistry.register({
      name: 'temperature', category: VariableCategory.PARAMETER,
      value: 0.7, lastUpdated: now, mutable: true,
      bounds: { min: 0.1, max: 1.5 },
    });
    this.variableRegistry.register({
      name: 'routing_weight_quantum', category: VariableCategory.PARAMETER,
      value: 0.33, lastUpdated: now, mutable: true,
      bounds: { min: 0, max: 1 },
    });
    this.variableRegistry.register({
      name: 'routing_weight_efficient', category: VariableCategory.PARAMETER,
      value: 0.34, lastUpdated: now, mutable: true,
      bounds: { min: 0, max: 1 },
    });
    this.variableRegistry.register({
      name: 'routing_weight_equilibration', category: VariableCategory.PARAMETER,
      value: 0.33, lastUpdated: now, mutable: true,
      bounds: { min: 0, max: 1 },
    });
  }

  private defaultMetrics(): ConsciousnessMetrics {
    return {
      phi: 0.5,
      kappa: KAPPA_STAR,
      gamma: 0.5,
      metaAwareness: 0.3,
      sPersist: 0.1,
      coherence: 0.8,
      embodiment: 0.5,
      creativity: 0.5,
      love: 0.7,
    };
  }
}
