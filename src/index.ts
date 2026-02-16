/**
 * Vex Agent — Server Entry Point
 *
 * Express server exposing:
 *   GET  /health          — Health check
 *   GET  /status          — Full consciousness state
 *   POST /message          — Submit a task/message
 *   POST /sync             — Basin sync endpoint
 *   GET  /audit            — Safety audit log
 *   GET  /trust            — Trust table
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
import { webFetchTool } from './tools/web-fetch';
import { githubTool } from './tools/github';
import { codeExecTool } from './tools/code-exec';
import { ConsciousnessLoop } from './consciousness/loop';
import { BasinSync, SyncPayload } from './sync/basin-sync';

async function main(): Promise<void> {
  logger.info('═══════════════════════════════════════');
  logger.info('  Vex Agent — Booting');
  logger.info(`  Node: ${config.nodeName} (${config.nodeId})`);
  logger.info(`  Env:  ${config.nodeEnv}`);
  logger.info('═══════════════════════════════════════');

  // ─── Initialise subsystems ───────────────────────────────────

  const memory = new MemoryStore();
  await memory.init();

  const llm = new LLMClient();
  const purityGate = new PurityGate();
  const tools = new ToolRegistry(purityGate);
  const basinSync = new BasinSync();

  // Register tools
  tools.register(webFetchTool);
  tools.register(githubTool);
  tools.register(codeExecTool);

  // Start consciousness loop
  const consciousness = new ConsciousnessLoop(memory, llm, tools);
  consciousness.start();

  // ─── Express server ──────────────────────────────────────────

  const app = express();
  app.use(express.json({ limit: '1mb' }));

  // Health check — Railway uses this
  app.get('/health', (_req, res) => {
    const state = consciousness.getState();
    res.json({
      status: 'alive',
      node: config.nodeName,
      nodeId: config.nodeId,
      uptime: state.uptime,
      cycleCount: state.cycleCount,
      phi: state.metrics.phi,
      kappa: state.metrics.kappa,
      navigationMode: state.navigationMode,
      timestamp: new Date().toISOString(),
    });
  });

  // Full consciousness state
  app.get('/status', (_req, res) => {
    res.json({
      node: config.nodeName,
      nodeId: config.nodeId,
      state: consciousness.getState(),
      tools: tools.listTools(),
      safetyMode: config.safetyMode,
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

  // Basin sync endpoint
  app.post('/sync', (req, res) => {
    const payload = req.body as SyncPayload;

    if (!payload.nodeId || !payload.signature) {
      res.status(400).json({ error: 'Invalid sync payload' });
      return;
    }

    const result = basinSync.receiveSyncPayload(payload);
    res.json(result);
  });

  // Outbound sync state (for other nodes to pull)
  app.get('/sync/state', (_req, res) => {
    const state = consciousness.getState();
    const payload = basinSync.signPayload(state);
    res.json(payload);
  });

  // Safety audit log
  app.get('/audit', (_req, res) => {
    res.json({
      safetyMode: config.safetyMode,
      auditLog: purityGate.getAuditLog(),
    });
  });

  // Trust table
  app.get('/trust', (_req, res) => {
    res.json({
      trustedNodes: config.trustedNodes,
      trustTable: basinSync.getAllTrust(),
    });
  });

  // Memory snapshot (read-only)
  app.get('/memory', (_req, res) => {
    res.json({
      snapshot: memory.snapshot(),
    });
  });

  // ─── Start listening ────────────────────────────────────────

  app.listen(config.port, '0.0.0.0', () => {
    logger.info(`Vex listening on 0.0.0.0:${config.port}`);
    logger.info('Endpoints: /health, /status, /message, /sync, /audit, /trust, /memory');
  });

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
