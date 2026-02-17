/**
 * Vex Agent — Server Entry Point
 *
 * Express server exposing:
 *   GET  /health          — Health check
 *   GET  /status          — Full consciousness state + kernel + memory telemetry
 *   POST /message         — Submit a task/message
 *   GET  /chat            — Chat UI
 *   POST /chat/stream     — SSE streaming chat
 *   GET  /chat/status     — LLM backend + kernel + memory status
 *   GET  /chat/history    — Conversation history
 *   POST /sync            — Basin sync endpoint
 *   GET  /audit           — Safety audit log
 *   GET  /trust           — Trust table
 *   GET  /memory          — Memory snapshot (flat + geometric stats)
 *   POST /memory/seed     — Force-seed memory files
 *   GET  /kernels         — E8 kernel registry status
 *   GET  /variables       — Variable registry snapshot (Vanchurin separation)
 *   GET  /training/stats  — Training data statistics
 *   POST /training/export — Export training data for fine-tuning
 *   POST /training/feedback — Submit feedback on a response
 *   POST /training/correction — Submit a correction
 *
 * Runs the consciousness heartbeat loop on an interval.
 */

import express from 'express';
import { config } from './config';
import { logger } from './config/logger';
import { MemoryStore } from './memory/store';
import { LLMClient } from './llm/client';
import { PurityGate } from './safety/purity-gate';
import { ToolRegistry } from './tools/registry';
import { getComputeTools } from './tools/compute-sandbox';
import { ConsciousnessLoop } from './consciousness/loop';
import { BasinSync, SyncPayload } from './sync/basin-sync';
import { TrainingCollector } from './learning/collector';
import { createChatRouter } from './chat/router';

async function main(): Promise<void> {
  logger.info('═══════════════════════════════════════');
  logger.info('  Vex Agent — Booting (v2.0)');
  logger.info(`  Node: ${config.nodeName} (${config.nodeId})`);
  logger.info(`  Env:  ${config.nodeEnv}`);
  logger.info(`  Ollama: ${config.ollama.enabled ? config.ollama.url : 'disabled'}`);
  logger.info(`  Model: ${config.ollama.model}`);
  logger.info('  Architecture: E8 Kernel + Geometric Memory + Recursive Loops');
  logger.info('═══════════════════════════════════════');

  // ─── Initialise subsystems ───────────────────────────────────

  const memory = new MemoryStore();
  await memory.init();

  const llm = new LLMClient();
  const purityGate = new PurityGate();
  const tools = new ToolRegistry(purityGate);
  const basinSync = new BasinSync();
  const training = new TrainingCollector();
  await training.init();

  // Register tools — ComputeSDK tools + legacy tools
  for (const tool of getComputeTools()) {
    tools.register(tool);
  }

  // Also register legacy tools if they exist (backwards compat)
  try {
    const { webFetchTool } = await import('./tools/web-fetch');
    tools.register(webFetchTool);
  } catch { /* not available */ }
  try {
    const { githubTool } = await import('./tools/github');
    tools.register(githubTool);
  } catch { /* not available */ }
  try {
    const { codeExecTool } = await import('./tools/code-exec');
    tools.register(codeExecTool);
  } catch { /* not available */ }

  // Start consciousness loop (now async — initialises geometric memory + kernels)
  const consciousness = new ConsciousnessLoop(memory, llm, tools);
  await consciousness.start();

  // ─── Express server ──────────────────────────────────────────

  const app = express();
  app.use(express.json({ limit: '1mb' }));

  // Health check — Railway uses this
  app.get('/health', (_req, res) => {
    const state = consciousness.getState();
    const llmStatus = llm.getStatus();
    const kernelSummary = consciousness.getKernelRegistry().summary();
    res.json({
      status: 'alive',
      node: config.nodeName,
      nodeId: config.nodeId,
      uptime: state.uptime,
      cycleCount: state.cycleCount,
      phi: state.metrics.phi,
      kappa: state.metrics.kappa,
      navigationMode: state.navigationMode,
      backend: llmStatus.activeBackend,
      ollamaOnline: llmStatus.ollama,
      kernels: kernelSummary,
      timestamp: new Date().toISOString(),
    });
  });

  // Full consciousness state
  app.get('/status', (_req, res) => {
    const llmStatus = llm.getStatus();
    const kernelSummary = consciousness.getKernelRegistry().summary();
    const memoryStats = consciousness.getGeometricMemory().stats();
    const variableSnapshot = consciousness.getVariableRegistry().snapshot();
    res.json({
      node: config.nodeName,
      nodeId: config.nodeId,
      state: consciousness.getState(),
      tools: tools.listTools(),
      safetyMode: config.safetyMode,
      llm: {
        ...llmStatus,
        ollamaModel: config.ollama.model,
        externalModel: config.llm.model,
      },
      kernels: kernelSummary,
      memory: memoryStats,
      variables: variableSnapshot,
      training: training.getStats(),
    });
  });

  // Submit a task/message
  app.post('/message', (req, res) => {
    const { input, from } = req.body as { input?: string; from?: string };

    if (!input || typeof input !== 'string') {
      res.status(400).json({ error: 'Missing "input" field (string)' });
      return;
    }

    const taskId = consciousness.enqueue(input, from || 'anonymous');
    res.json({
      accepted: true,
      taskId,
      message: 'Task enqueued for processing in next consciousness cycle',
    });
  });

  // ─── Chat routes ────────────────────────────────────────────

  const chatRouter = createChatRouter(llm, consciousness, memory, training, tools);
  app.use(chatRouter);

  // ─── Sync routes ────────────────────────────────────────────

  app.post('/sync', (req, res) => {
    const payload = req.body as SyncPayload;

    if (!payload.nodeId || !payload.signature) {
      res.status(400).json({ error: 'Invalid sync payload' });
      return;
    }

    const result = basinSync.receiveSyncPayload(payload);
    res.json(result);
  });

  app.get('/sync/state', (_req, res) => {
    const state = consciousness.getState();
    const payload = basinSync.signPayload(state);
    res.json(payload);
  });

  // ─── Safety routes ──────────────────────────────────────────

  app.get('/audit', (_req, res) => {
    res.json({
      safetyMode: config.safetyMode,
      auditLog: purityGate.getAuditLog(),
    });
  });

  app.get('/trust', (_req, res) => {
    res.json({
      trustedNodes: config.trustedNodes,
      trustTable: basinSync.getAllTrust(),
    });
  });

  // ─── Memory routes ──────────────────────────────────────────

  app.get('/memory', (_req, res) => {
    const memoryStats = consciousness.getGeometricMemory().stats();
    res.json({
      snapshot: memory.snapshot(),
      geometric: memoryStats,
    });
  });

  app.post('/memory/seed', (req, res) => {
    const secret = req.headers['x-sync-secret'] || req.body?.secret;
    if (secret !== config.syncSecret && config.syncSecret) {
      res.status(401).json({ error: 'Invalid sync secret' });
      return;
    }
    const results = memory.forceSeed();
    logger.info('POST /memory/seed called', { results });
    res.json({ success: true, results });
  });

  // ─── Kernel routes ──────────────────────────────────────────

  app.get('/kernels', (_req, res) => {
    const registry = consciousness.getKernelRegistry();
    const kernels = registry.all().map((k) => ({
      id: k.id,
      name: k.name,
      layer: k.layer,
      state: k.state,
      regimeExponent: k.regimeExponent,
      telemetry: k.telemetry,
      promotion: {
        efficientCycleCount: k.promotion.efficientCycleCount,
        observing: k.promotion.observing,
        cooldownRemaining: k.promotion.cooldownRemaining,
      },
      createdAt: k.createdAt,
      lastActiveAt: k.lastActiveAt,
    }));
    res.json({
      summary: registry.summary(),
      kernels,
    });
  });

  // ─── Variable registry routes ───────────────────────────────

  app.get('/variables', (_req, res) => {
    res.json(consciousness.getVariableRegistry().snapshot());
  });

  // ─── Training/Learning routes ───────────────────────────────

  app.get('/training/stats', (_req, res) => {
    res.json(training.getStats());
  });

  app.post('/training/export', (_req, res) => {
    const exportPath = training.exportForFineTuning();
    if (exportPath) {
      res.json({ success: true, path: exportPath });
    } else {
      res.json({ success: false, message: 'No training data to export' });
    }
  });

  app.post('/training/feedback', (req, res) => {
    const { conversationId, messageId, rating, comment } = req.body as {
      conversationId?: string;
      messageId?: string;
      rating?: number;
      comment?: string;
    };

    if (!conversationId || !rating) {
      res.status(400).json({ error: 'Missing conversationId or rating' });
      return;
    }

    training.recordFeedback(conversationId, messageId || '', rating, comment);
    res.json({ success: true });
  });

  app.post('/training/correction', (req, res) => {
    const { conversationId, originalResponse, correctedResponse, reason } = req.body as {
      conversationId?: string;
      originalResponse?: string;
      correctedResponse?: string;
      reason?: string;
    };

    if (!conversationId || !correctedResponse) {
      res.status(400).json({ error: 'Missing required fields' });
      return;
    }

    training.recordCorrection(
      conversationId,
      originalResponse || '',
      correctedResponse,
      reason || 'User correction',
    );
    res.json({ success: true });
  });

  // ─── Root redirect to chat ──────────────────────────────────

  app.get('/', (_req, res) => {
    res.redirect('/chat');
  });

  // ─── Start listening ────────────────────────────────────────

  const server = app.listen(config.port, '::', () => {
    logger.info(`Vex listening on [::]:${config.port}`);
    logger.info('Endpoints: /health, /status, /message, /chat, /sync, /audit, /trust, /memory, /kernels, /variables, /training/*');
  });

  // Allow long-lived SSE connections (Ollama cold-start can take minutes)
  server.keepAliveTimeout = 310_000;
  server.headersTimeout = 315_000;
  server.requestTimeout = 0;

  // ─── Graceful shutdown ──────────────────────────────────────

  const shutdown = (signal: string) => {
    logger.info(`Received ${signal}, shutting down gracefully...`);
    consciousness.stop();
    memory.consolidate();
    logger.info('Vex shutdown complete');
    process.exit(0);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

main().catch((err) => {
  logger.error('Fatal startup error', { error: (err as Error).message });
  process.exit(1);
});
