/**
 * Vex Chat Router — Thin Proxy
 *
 * Provides:
 *   GET  /chat          — Serve the chat UI (auth-gated)
 *   POST /chat/auth     — Authenticate with token, set session cookie
 *   POST /chat/stream   — SSE streaming chat (proxied to Python kernel)
 *   GET  /chat/status   — Kernel status (proxied)
 *   GET  /chat/history  — Conversation history (proxied)
 *
 * All actual logic (consciousness, LLM, memory, tools) runs in the
 * Python kernel. This router only handles auth and SSE proxying.
 */

import { Router, Request, Response } from "express";
import { config } from "../config";
import { logger } from "../config/logger";
import { getChatHTML } from "./ui";
import {
  requireAuth,
  createSession,
  SESSION_COOKIE,
  SESSION_TTL_MS,
} from "../auth/middleware";

/** Interval (ms) between SSE keep-alive pings */
const SSE_KEEPALIVE_INTERVAL_MS = 15_000;

export interface ChatRouterOptions {
  kernelUrl: string;
  /** When true, skip serving inline HTML for GET /chat (React SPA handles it) */
  hasReactFrontend?: boolean;
}

export function createChatRouter(options: ChatRouterOptions): Router;
export function createChatRouter(kernelUrl: string): Router;
export function createChatRouter(arg: string | ChatRouterOptions): Router {
  const options: ChatRouterOptions =
    typeof arg === "string" ? { kernelUrl: arg } : arg;
  const { kernelUrl, hasReactFrontend = false } = options;
  const router = Router();

  // ── Auth endpoint ─────────────────────────────────────────────
  router.post("/chat/auth", (req: Request, res: Response) => {
    if (!config.chatAuthToken) {
      const sessionId = createSession();
      res.setHeader(
        "Set-Cookie",
        `${SESSION_COOKIE}=${sessionId}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}`,
      );
      res.json({ ok: true });
      return;
    }

    const { token } = req.body as { token?: string };
    if (!token || token !== config.chatAuthToken) {
      res.status(403).json({ error: "Invalid token" });
      return;
    }

    const sessionId = createSession();
    res.setHeader(
      "Set-Cookie",
      `${SESSION_COOKIE}=${sessionId}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}`,
    );
    res.json({ ok: true });
  });

  // ── Serve chat UI (auth-gated) ────────────────────────────────
  // Only serve inline HTML fallback when the React frontend is not available.
  // When the React SPA exists, /chat is handled by the SPA fallback route.
  if (!hasReactFrontend) {
    router.get("/chat", requireAuth, (_req: Request, res: Response) => {
      res.setHeader("Content-Type", "text/html");
      res.send(getChatHTML());
    });
  }

  // ── Kernel status (proxied) ───────────────────────────────────
  router.get("/chat/status", async (_req: Request, res: Response) => {
    try {
      const resp = await fetch(`${kernelUrl}/status`);
      const data = await resp.json();
      res.json(data);
    } catch (err) {
      res
        .status(502)
        .json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // ── Conversation history (proxied) ────────────────────────────
  router.get("/chat/history", async (req: Request, res: Response) => {
    try {
      const convId = req.query.id ? `?id=${req.query.id}` : "";
      const resp = await fetch(`${kernelUrl}/chat/history${convId}`);
      const data = await resp.json();
      res.json(data);
    } catch (err) {
      res
        .status(502)
        .json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // ── Streaming chat endpoint via SSE (auth-gated) ──────────────
  // Proxies the request to the Python kernel's /chat/stream endpoint
  // and pipes the SSE response back to the client
  router.post(
    "/chat/stream",
    requireAuth,
    async (req: Request, res: Response) => {
      const { message, conversationId } = req.body as {
        message?: string;
        conversationId?: string;
      };

      if (!message || typeof message !== "string") {
        res.status(400).json({ error: 'Missing "message" field' });
        return;
      }

      // ── SSE headers ──────────────────────────────────────────────
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      req.setTimeout(0);
      res.setTimeout(0);

      const keepAlive = setInterval(() => {
        try {
          res.write(": ping\n\n");
        } catch {
          // Connection closed
        }
      }, SSE_KEEPALIVE_INTERVAL_MS);

      try {
        // Proxy to Python kernel's streaming endpoint
        const kernelResp = await fetch(`${kernelUrl}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, conversation_id: conversationId }),
        });

        if (!kernelResp.ok) {
          const errBody = await kernelResp.text();
          res.write(
            `data: ${JSON.stringify({ type: "error", error: errBody })}\n\n`,
          );
          clearInterval(keepAlive);
          res.end();
          return;
        }

        // Pipe the SSE stream from the kernel to the client
        const reader = kernelResp.body?.getReader();
        if (!reader) {
          res.write(
            `data: ${JSON.stringify({ type: "error", error: "No response body from kernel" })}\n\n`,
          );
          clearInterval(keepAlive);
          res.end();
          return;
        }

        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          res.write(chunk);
          if (typeof (res as any).flush === "function") {
            (res as any).flush();
          }
        }
      } catch (err) {
        const errMsg = (err as Error).message;
        logger.error("Chat stream proxy error", { error: errMsg });
        res.write(
          `data: ${JSON.stringify({ type: "error", error: errMsg })}\n\n`,
        );
      } finally {
        clearInterval(keepAlive);
        res.end();
      }
    },
  );

  return router;
}
