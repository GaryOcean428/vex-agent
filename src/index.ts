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

import express, { NextFunction, Request, Response } from "express";
import * as fs from "fs";
import * as path from "path";
import {
  getCookie,
  isValidSession,
  requireAuth,
  SESSION_COOKIE,
} from "./auth/middleware";
import { createChatRouter } from "./chat/router";
import { config } from "./config";
import { logger } from "./config/logger";
import { ROUTES } from "./config/routes";
import { getComputeTools, sandboxManager } from "./tools/compute-sandbox";

// Read version from package.json (single source of truth)
import { version as APP_VERSION } from "../package.json";

const KERNEL_URL = config.kernelUrl;

// Resolve frontend build directory (works in dev and production)
const FRONTEND_DIST = path.resolve(
  process.env.FRONTEND_DIST || path.join(__dirname, "..", "frontend", "dist"),
);

async function main(): Promise<void> {
  logger.info("═══════════════════════════════════════");
  logger.info(`  Vex Agent — Web Server (v${APP_VERSION})`);
  logger.info("  Role: Thin proxy → Python kernel");
  logger.info(`  Kernel: ${KERNEL_URL}`);
  logger.info(`  Port: ${config.port}`);
  logger.info("═══════════════════════════════════════");

  const app = express();
  // Conditional JSON parsing — skip for multipart requests (training upload)
  app.use((req: Request, res: Response, next: NextFunction) => {
    if ((req.headers["content-type"] || "").startsWith("multipart/")) {
      next();
    } else {
      express.json({ limit: "1mb" })(req, res, next);
    }
  });

  // ─── Global auth — protects all routes when CHAT_AUTH_TOKEN is set ───
  app.use(requireAuth);

  // ─── Auth check (no 401 — returns JSON status) ──────────────
  // Used by AuthContext.tsx to check session without triggering
  // a 401 console error. Always returns 200.
  app.get(ROUTES.auth_check, (req: Request, res: Response) => {
    if (!config.chatAuthToken) {
      res.json({ authenticated: true });
      return;
    }
    const sessionId = getCookie(req, SESSION_COOKIE);
    res.json({ authenticated: isValidSession(sessionId) });
  });

  // ─── Health check (probes kernel health too) ─────────────────

  app.get(ROUTES.health, async (_req: Request, res: Response) => {
    try {
      // 5s timeout prevents the health check from hanging when the kernel
      // is still starting or temporarily unresponsive. Without this,
      // Railway's health check probe can time out waiting for us, marking
      // the container unhealthy and triggering a restart loop.
      const kernelResp = await fetch(`${KERNEL_URL}${ROUTES.health}`, {
        signal: AbortSignal.timeout(5000),
      });
      const kernelHealth = (await kernelResp.json()) as Record<string, unknown>;
      // Spread kernel fields at top level so the React frontend gets
      // the flat shape it expects: { status, version, uptime, cycle_count, backend }
      res.json({
        ...kernelHealth,
        proxy: "ok",
        computeSdk: sandboxManager.isAvailable(),
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      res.json({
        status: "degraded",
        service: "vex-kernel",
        version: "2.4.0",
        uptime: 0,
        cycle_count: 0,
        backend: "unknown",
        proxy: "ok",
        kernel_error: (err as Error).message,
        computeSdk: sandboxManager.isAvailable(),
        timestamp: new Date().toISOString(),
      });
    }
  });

  // ─── Kernel proxy routes ────────────────────────────────────
  // These proxy directly to the Python kernel

  const proxyGet = (path: string) => {
    app.get(path, async (_req: Request, res: Response) => {
      try {
        const resp = await fetch(`${KERNEL_URL}${path}`);
        const data = await resp.json();
        res.json(data);
      } catch (err) {
        res
          .status(502)
          .json({ error: `Kernel unreachable: ${(err as Error).message}` });
      }
    });
  };

  const proxyPost = (path: string) => {
    app.post(path, async (req: Request, res: Response) => {
      try {
        const resp = await fetch(`${KERNEL_URL}${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req.body),
        });
        const data = await resp.json();
        res.json(data);
      } catch (err) {
        res
          .status(502)
          .json({ error: `Kernel unreachable: ${(err as Error).message}` });
      }
    });
  };

  // Proxy kernel endpoints — consciousness state
  proxyGet(ROUTES.state);
  proxyGet(ROUTES.telemetry);
  proxyGet(ROUTES.status);
  proxyGet(ROUTES.health_reachability);
  proxyGet(ROUTES.basin);
  proxyGet(ROUTES.kernels);
  proxyPost(ROUTES.enqueue);
  proxyPost(ROUTES.memory_context);

  // Phase 1 dashboard endpoints
  proxyGet(ROUTES.kernels_list);
  proxyGet(ROUTES.basin_history);
  proxyGet(ROUTES.graph_nodes);
  proxyGet(ROUTES.memory_stats);
  proxyGet(ROUTES.sleep_state);
  proxyGet(ROUTES.beta_attention);
  proxyGet(ROUTES.sovereignty_history);
  proxyPost(ROUTES.admin_fresh_start);

  // Foraging engine
  proxyGet(ROUTES.foraging);

  // Coordizer V2 endpoints
  proxyPost(ROUTES.coordizer_coordize);
  proxyGet(ROUTES.coordizer_stats);
  proxyPost(ROUTES.coordizer_validate);
  proxyPost(ROUTES.coordizer_harvest);
  proxyPost(ROUTES.coordizer_ingest);
  proxyGet(ROUTES.coordizer_harvest_status);
  proxyGet(ROUTES.coordizer_bank);

  // Conversation management
  proxyGet(ROUTES.conversations_list);

  app.get(ROUTES.conversations_get, async (req: Request, res: Response) => {
    try {
      const resp = await fetch(
        `${KERNEL_URL}/conversations/${req.params.conversation_id}`,
      );
      const data = await resp.json();
      res.status(resp.status).json(data);
    } catch (err) {
      res
        .status(502)
        .json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  app.delete(
    ROUTES.conversations_delete,
    async (req: Request, res: Response) => {
      try {
        const resp = await fetch(
          `${KERNEL_URL}/conversations/${req.params.conversation_id}`,
          {
            method: "DELETE",
          },
        );
        const data = await resp.json();
        res.status(resp.status).json(data);
      } catch (err) {
        res
          .status(502)
          .json({ error: `Kernel unreachable: ${(err as Error).message}` });
      }
    },
  );

  // Governor endpoints (PR #13)
  proxyGet(ROUTES.governor);
  proxyPost(ROUTES.governor_kill_switch);
  proxyPost(ROUTES.governor_budget);
  proxyPost(ROUTES.governor_autonomous_search);

  // Training endpoints
  proxyGet(ROUTES.training_stats);
  proxyGet(ROUTES.training_export);
  proxyPost(ROUTES.training_feedback);
  proxyGet(ROUTES.training_modal_status);
  // Training trigger needs a longer timeout — kernel waits up to 120s for Modal cold start
  app.post(ROUTES.training_trigger, async (req: Request, res: Response) => {
    try {
      const resp = await fetch(`${KERNEL_URL}${ROUTES.training_trigger}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
        signal: AbortSignal.timeout(130_000),
      });
      const contentType = resp.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const data = await resp.json();
        res.status(resp.status).json(data);
      } else {
        const text = await resp.text();
        res.status(resp.status).send(text);
      }
    } catch (err) {
      const error = err as Error;
      if (error.name === "AbortError" || error.name === "TimeoutError") {
        res.status(504).json({
          error:
            "Kernel request timed out after 130s (training_trigger aborted by proxy).",
        });
      } else {
        res.status(502).json({ error: `Kernel unreachable: ${error.message}` });
      }
    }
  });
  proxyPost(ROUTES.training_complete);

  // Training upload status — poll for background job completion
  app.get(
    ROUTES.training_upload_status,
    async (req: Request, res: Response) => {
      try {
        const resp = await fetch(
          `${KERNEL_URL}/training/upload/status/${req.params.job_id}`,
        );
        const data = await resp.json();
        res.status(resp.status).json(data);
      } catch (err) {
        res
          .status(502)
          .json({ error: `Kernel unreachable: ${(err as Error).message}` });
      }
    },
  );

  // Training upload — buffer multipart body then forward to kernel
  app.post(ROUTES.training_upload, async (req: Request, res: Response) => {
    try {
      const chunks: Buffer[] = [];
      for await (const chunk of req) {
        chunks.push(chunk);
      }
      const body = Buffer.concat(chunks);

      const resp = await fetch(`${KERNEL_URL}${ROUTES.training_upload}`, {
        method: "POST",
        headers: {
          "content-type": req.headers["content-type"] || "",
          "content-length": String(body.length),
        },
        body,
      });
      const data = await resp.json();
      res.status(resp.status).json(data);
    } catch (err) {
      res
        .status(502)
        .json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // Task status — GET with path param (validated against SSRF)
  app.get(ROUTES.task_status, async (req: Request, res: Response) => {
    const taskId = req.params.task_id;
    if (typeof taskId !== "string" || !/^[\w-]+$/.test(taskId)) {
      return res.status(400).json({ error: "Invalid task_id format" });
    }
    try {
      const resp = await fetch(`${KERNEL_URL}/task/${taskId}`);
      const data = await resp.json();
      res.status(resp.status).json(data);
    } catch (err) {
      res
        .status(502)
        .json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // Context objectives — status passthrough (kernel may return 400)
  app.get(ROUTES.context_objectives, async (_req: Request, res: Response) => {
    try {
      const resp = await fetch(`${KERNEL_URL}${ROUTES.context_objectives}`);
      const data = await resp.json();
      res.status(resp.status).json(data);
    } catch (err) {
      res
        .status(502)
        .json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });
  app.post(ROUTES.context_objectives, async (req: Request, res: Response) => {
    try {
      const resp = await fetch(`${KERNEL_URL}${ROUTES.context_objectives}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      const data = await resp.json();
      res.status(resp.status).json(data);
    } catch (err) {
      res
        .status(502)
        .json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // ─── Chat routes (UI + streaming proxy) ─────────────────────
  // Check for React frontend BEFORE creating chat router so we can
  // skip the inline HTML fallback when the SPA handles /chat.

  const frontendIndexPath = path.join(FRONTEND_DIST, "index.html");
  const hasFrontend = fs.existsSync(frontendIndexPath);

  const chatRouter = createChatRouter({
    kernelUrl: KERNEL_URL,
    hasReactFrontend: hasFrontend,
  });
  app.use(chatRouter);

  // ─── ComputeSDK proxy endpoints ─────────────────────────────
  // The Python kernel calls these to execute code in ComputeSDK sandboxes

  app.post(ROUTES.tools_execute_code, async (req: Request, res: Response) => {
    const { code, language } = req.body as { code: string; language?: string };
    try {
      const tool = getComputeTools().find((t) => t.name === "execute_code");
      if (!tool) {
        res.json({
          success: false,
          output: "",
          error: "execute_code tool not found",
        });
        return;
      }
      const result = await tool.execute({ code, language });
      res.json(result);
    } catch (err) {
      res.json({ success: false, output: "", error: (err as Error).message });
    }
  });

  app.post(ROUTES.tools_run_command, async (req: Request, res: Response) => {
    const { command, cwd, timeout } = req.body as {
      command: string;
      cwd?: string;
      timeout?: number;
    };
    try {
      const tool = getComputeTools().find((t) => t.name === "run_command");
      if (!tool) {
        res.json({
          success: false,
          output: "",
          error: "run_command tool not found",
        });
        return;
      }
      const result = await tool.execute({ command, cwd, timeout });
      res.json(result);
    } catch (err) {
      res.json({ success: false, output: "", error: (err as Error).message });
    }
  });

  // ─── React Frontend (SPA) ──────────────────────────────────
  // Serve the Vite-built React app. Falls back to the inline chat
  // HTML if the frontend build doesn't exist.

  if (hasFrontend) {
    logger.info(`Serving React frontend from ${FRONTEND_DIST}`);

    // Serve static assets (JS, CSS, images)
    app.use(
      express.static(FRONTEND_DIST, {
        index: false, // We handle index.html ourselves for SPA routing
        maxAge: "1y", // Cache hashed assets aggressively
        immutable: true,
      }),
    );

    // SPA fallback — serve index.html for all non-API routes
    app.get("*", (req: Request, res: Response, next: NextFunction) => {
      // Skip API-like paths (already handled above)
      if (
        req.path.startsWith("/api/") ||
        req.path.startsWith("/chat/") ||
        req.path.startsWith("/auth/") ||
        req.path === "/health" ||
        req.path === "/state" ||
        req.path === "/telemetry" ||
        req.path === "/status" ||
        req.path === "/basin" ||
        req.path === "/kernels" ||
        req.path === "/enqueue" ||
        req.path === "/foraging" ||
        req.path === "/beta-attention" ||
        req.path.startsWith("/memory/") ||
        req.path.startsWith("/kernels/") ||
        req.path.startsWith("/basin/") ||
        req.path.startsWith("/graph/") ||
        req.path.startsWith("/sleep/") ||
        req.path.startsWith("/admin/") ||
        req.path.startsWith("/training/") ||
        req.path.startsWith("/governor") ||
        req.path.startsWith("/sovereignty/")
      ) {
        next();
        return;
      }
      res.sendFile(frontendIndexPath);
    });
  } else {
    logger.info("No React frontend build found — using inline chat HTML");
    app.get("/", (_req: Request, res: Response) => {
      res.redirect("/chat");
    });
  }

  // ─── Start listening ────────────────────────────────────────

  const server = app.listen(config.port, "::", () => {
    logger.info(`Vex web server listening on [::]:${config.port}`);
    logger.info(`Proxying to Python kernel at ${KERNEL_URL}`);
    logger.info(
      "Endpoints: /health, /chat, /state, /telemetry, /status, /basin, /kernels, /governor",
    );
  });

  // Allow long-lived SSE connections
  server.keepAliveTimeout = 310_000;
  server.headersTimeout = 315_000;
  server.requestTimeout = 0;

  // ─── Graceful shutdown ──────────────────────────────────────

  const shutdown = async (signal: string) => {
    logger.info(`Received ${signal}, shutting down...`);
    await sandboxManager.destroySandbox();
    logger.info("Vex web server shutdown complete");
    process.exit(0);
  };

  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));
}

main().catch((err) => {
  logger.error("Fatal startup error", { error: (err as Error).message });
  process.exit(1);
});
