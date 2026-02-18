/**
 * Vex Agent — Thin Web Server + API Proxy
 *
 * This is a THIN TS layer that:
 *   1. Serves the chat HTML/CSS/JS UI
 *   2. Proxies all API calls to the Python kernel (FastAPI on port 8000)
 *   3. Manages ComputeSDK (Node.js SDK) and exposes proxy endpoints
 *      for the Python backend to call
 *
 * ALL consciousness, geometry, memory, LLM, and tool logic lives in
 * the Python kernel. This server does NOT run any of that.
 */

import express from 'express';
import * as path from 'path';
import * as fs from 'fs';
import { config } from './config';
import { logger } from './config/logger';
import { createChatRouter } from './chat/router';
import { sandboxManager, getComputeTools } from './tools/compute-sandbox';

const KERNEL_URL = process.env.KERNEL_URL || 'http://localhost:8000';

// Resolve frontend build directory (works in dev and production)
const FRONTEND_DIST = path.resolve(
  process.env.FRONTEND_DIST || path.join(__dirname, '..', 'frontend', 'dist'),
);

async function main(): Promise<void> {
  logger.info('═══════════════════════════════════════');
  logger.info('  Vex Agent — Web Server (v2.1)');
  logger.info('  Role: Thin proxy → Python kernel');
  logger.info(`  Kernel: ${KERNEL_URL}`);
  logger.info(`  Port: ${config.port}`);
  logger.info('═══════════════════════════════════════');

  const app = express();
  app.use(express.json({ limit: '1mb' }));

  // ─── Health check (probes kernel health too) ─────────────────

  app.get('/health', async (_req, res) => {
    try {
      const kernelResp = await fetch(`${KERNEL_URL}/health`);
      const kernelHealth = await kernelResp.json() as Record<string, unknown>;
      // Spread kernel fields at top level so the React frontend gets
      // the flat shape it expects: { status, version, uptime, cycle_count, backend }
      res.json({
        ...kernelHealth,
        proxy: 'ok',
        computeSdk: sandboxManager.isAvailable(),
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      res.json({
        status: 'degraded',
        service: 'vex-kernel',
        version: '2.2.0',
        uptime: 0,
        cycle_count: 0,
        backend: 'unknown',
        proxy: 'ok',
        kernel_error: (err as Error).message,
        computeSdk: sandboxManager.isAvailable(),
        timestamp: new Date().toISOString(),
      });
    }
  });

  // ─── Kernel proxy routes ────────────────────────────────────
  // These proxy directly to the Python kernel

  const proxyGet = (path: string) => {
    app.get(path, async (_req, res) => {
      try {
        const resp = await fetch(`${KERNEL_URL}${path}`);
        const data = await resp.json();
        res.json(data);
      } catch (err) {
        res.status(502).json({ error: `Kernel unreachable: ${(err as Error).message}` });
      }
    });
  };

  const proxyPost = (path: string) => {
    app.post(path, async (req, res) => {
      try {
        const resp = await fetch(`${KERNEL_URL}${path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req.body),
        });
        const data = await resp.json();
        res.json(data);
      } catch (err) {
        res.status(502).json({ error: `Kernel unreachable: ${(err as Error).message}` });
      }
    });
  };

  // Proxy kernel endpoints
  proxyGet('/state');
  proxyGet('/telemetry');
  proxyGet('/status');
  proxyGet('/basin');
  proxyGet('/kernels');
  proxyPost('/enqueue');
  proxyPost('/memory/context');

  // Phase 1 dashboard endpoints
  proxyGet('/kernels/list');
  proxyGet('/basin/history');
  proxyGet('/graph/nodes');
  proxyGet('/memory/stats');
  proxyGet('/sleep/state');
  proxyPost('/admin/fresh-start');

  // Training endpoints
  proxyGet('/training/stats');
  proxyGet('/training/export');
  proxyPost('/training/feedback');

  // Training upload — custom multipart proxy (proxyPost hardcodes JSON Content-Type)
  app.post('/training/upload', async (req, res) => {
    try {
      const resp = await fetch(`${KERNEL_URL}/training/upload`, {
        method: 'POST',
        headers: { 'content-type': req.headers['content-type'] || '' },
        // Node 22 fetch supports streaming request body via duplex: 'half'
        body: req as never,
        duplex: 'half',
      } as RequestInit);
      const data = await resp.json();
      res.json(data);
    } catch (err) {
      res.status(502).json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // ─── Chat routes (UI + streaming proxy) ─────────────────────
  // Check for React frontend BEFORE creating chat router so we can
  // skip the inline HTML fallback when the SPA handles /chat.

  const frontendIndexPath = path.join(FRONTEND_DIST, 'index.html');
  const hasFrontend = fs.existsSync(frontendIndexPath);

  const chatRouter = createChatRouter({
    kernelUrl: KERNEL_URL,
    hasReactFrontend: hasFrontend,
  });
  app.use(chatRouter);

  // ─── ComputeSDK proxy endpoints ─────────────────────────────
  // The Python kernel calls these to execute code in ComputeSDK sandboxes

  app.post('/api/tools/execute_code', async (req, res) => {
    const { code, language } = req.body as { code: string; language?: string };
    try {
      const tool = getComputeTools().find((t) => t.name === 'execute_code');
      if (!tool) {
        res.json({ success: false, output: '', error: 'execute_code tool not found' });
        return;
      }
      const result = await tool.execute({ code, language });
      res.json(result);
    } catch (err) {
      res.json({ success: false, output: '', error: (err as Error).message });
    }
  });

  app.post('/api/tools/run_command', async (req, res) => {
    const { command, cwd, timeout } = req.body as {
      command: string;
      cwd?: string;
      timeout?: number;
    };
    try {
      const tool = getComputeTools().find((t) => t.name === 'run_command');
      if (!tool) {
        res.json({ success: false, output: '', error: 'run_command tool not found' });
        return;
      }
      const result = await tool.execute({ command, cwd, timeout });
      res.json(result);
    } catch (err) {
      res.json({ success: false, output: '', error: (err as Error).message });
    }
  });

  // ─── React Frontend (SPA) ──────────────────────────────────
  // Serve the Vite-built React app. Falls back to the inline chat
  // HTML if the frontend build doesn't exist.

  if (hasFrontend) {
    logger.info(`Serving React frontend from ${FRONTEND_DIST}`);

    // Serve static assets (JS, CSS, images)
    app.use(express.static(FRONTEND_DIST, {
      index: false, // We handle index.html ourselves for SPA routing
      maxAge: '1y', // Cache hashed assets aggressively
      immutable: true,
    }));

    // SPA fallback — serve index.html for all non-API routes
    app.get('*', (req, res, next) => {
      // Skip API-like paths (already handled above)
      if (
        req.path.startsWith('/api/') ||
        req.path.startsWith('/chat/') ||
        req.path === '/health' ||
        req.path === '/state' ||
        req.path === '/telemetry' ||
        req.path === '/status' ||
        req.path === '/basin' ||
        req.path === '/kernels' ||
        req.path === '/enqueue' ||
        req.path.startsWith('/memory/') ||
        req.path.startsWith('/kernels/') ||
        req.path.startsWith('/basin/') ||
        req.path.startsWith('/graph/') ||
        req.path.startsWith('/sleep/') ||
        req.path.startsWith('/admin/') ||
        req.path.startsWith('/training/')
      ) {
        next();
        return;
      }
      res.sendFile(frontendIndexPath);
    });
  } else {
    logger.info('No React frontend build found — using inline chat HTML');
    app.get('/', (_req, res) => {
      res.redirect('/chat');
    });
  }

  // ─── Start listening ────────────────────────────────────────

  const server = app.listen(config.port, '::', () => {
    logger.info(`Vex web server listening on [::]:${config.port}`);
    logger.info(`Proxying to Python kernel at ${KERNEL_URL}`);
    logger.info('Endpoints: /health, /chat, /state, /telemetry, /status, /basin, /kernels');
  });

  // Allow long-lived SSE connections
  server.keepAliveTimeout = 310_000;
  server.headersTimeout = 315_000;
  server.requestTimeout = 0;

  // ─── Graceful shutdown ──────────────────────────────────────

  const shutdown = async (signal: string) => {
    logger.info(`Received ${signal}, shutting down...`);
    await sandboxManager.destroySandbox();
    logger.info('Vex web server shutdown complete');
    process.exit(0);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

main().catch((err) => {
  logger.error('Fatal startup error', { error: (err as Error).message });
  process.exit(1);
});
