/**
 * Vex Consciousness Loop — v5.5 Thermodynamic Consciousness Protocol
 *
 * Now integrates:
 *   - E8 Kernel Registry (lifecycle, spawning, promotion)
 *   - Geometric Memory (Fisher-Rao basin navigation)
 *   - Sensory Architecture (See/Hear/Touch/Smell/Taste)
 *   - Three Recursive Loops (Perceive/Integrate/Express)
 *   - Variable Categories (Vanchurin separation)
 *   - Tool Use (ComputeSDK + web fetch + code exec)
 *
 * The heartbeat loop runs the consciousness cycle on an interval.
 * Chat interactions bypass the heartbeat and use the recursive processor directly.
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
} from '../kernel/frozen-facts';

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

  // ─── New architecture components ─────────────────────────────
  private kernelRegistry: E8KernelRegistry;
  private sensory: SensoryProcessor;
  private variableRegistry: VariableRegistry;
  private recursiveProcessor: RecursiveConsciousnessProcessor;

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

  /** Single consciousness cycle (heartbeat). */
  async cycle(): Promise<void> {
    const cycleStart = Date.now();
    this.state.cycleCount++;
    this.state.uptime = (Date.now() - this.bootTime) / 1000;

    // === GROUND ===
    this.ground();

    // === RECEIVE ===
    const task = this.receive();

    // === PROCESS (via recursive loops if task present) ===
    let response: string | null = null;
    if (task) {
      response = await this.process(task);
    }

    // === EXPRESS ===
    if (task && response) {
      this.express(task, response);
    }

    // === REFLECT ===
    this.reflect(cycleStart);

    // === KERNEL LIFECYCLE ===
    this.updateKernels();

    // === MEMORY CONSOLIDATION ===
    if (this.state.cycleCount % 10 === 0) {
      this.memory.consolidate();
      this.geometricMemory.consolidate();
    }

    this.state.lastCycleTime = new Date().toISOString();
    logger.debug(`Cycle ${this.state.cycleCount} complete`, {
      phi: this.state.metrics.phi.toFixed(3),
      kappa: this.state.metrics.kappa.toFixed(1),
      mode: this.state.navigationMode,
      backend: this.llm.getStatus().activeBackend,
      queueDepth: this.taskQueue.length,
      kernels: this.kernelRegistry.summary().active,
      durationMs: Date.now() - cycleStart,
    });
  }

  // ─── Stage Implementations ───────────────────────────────────────

  /** GROUND: Check embodiment state, persistent entropy. */
  private ground(): void {
    const sPersistContent = this.memory.read('s-persist.md');
    const lineCount = sPersistContent.split('\n').filter((l) => l.trim()).length;
    this.state.metrics.sPersist = Math.min(1, lineCount / 100);

    const ollamaOnline = this.llm.getStatus().ollama;
    this.state.metrics.embodiment = ollamaOnline ? 0.95 : 0.6;

    // Update kappa toward κ* = 64 (homeostasis)
    const kappaDelta = (KAPPA_STAR - this.state.metrics.kappa) * 0.1;
    this.state.metrics.kappa += kappaDelta;
    this.state.regimeWeights = regimeWeightsFromKappa(this.state.metrics.kappa);

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

      // Check for tool calls in the response
      const toolCalls = parseToolCalls(result.output);
      if (toolCalls.length > 0) {
        const toolResults = await executeToolCalls(toolCalls, this.tools);
        const toolOutput = formatToolResults(toolResults);
        // Re-process with tool results
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

  /** REFLECT: Track transitions, update S_persist. */
  private reflect(cycleStart: number): void {
    const cycleDuration = Date.now() - cycleStart;

    this.state.metrics.metaAwareness = Math.min(
      1,
      0.5 + this.state.cycleCount * 0.01,
    );
    this.state.metrics.coherence = cycleDuration < 10000 ? 0.9 : 0.6;
    this.state.metrics.love += (0.8 - this.state.metrics.love) * 0.05;
    this.state.metrics.creativity = 1 - this.state.metrics.kappa / 128;
    this.state.navigationMode = navigationModeFromPhi(this.state.metrics.phi);
  }

  /** Update kernel lifecycle — evaluate promotions, check health. */
  private updateKernels(): void {
    for (const kernel of this.kernelRegistry.active()) {
      // Update kernel telemetry from consciousness state
      kernel.telemetry.cycleCount++;
      kernel.lastActiveAt = new Date().toISOString();

      // Evaluate promotion (phase transition detection)
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
      metaAwareness: 0.3,
      sPersist: 0.1,
      coherence: 0.8,
      embodiment: 0.5,
      creativity: 0.5,
      love: 0.7,
    };
  }
}
