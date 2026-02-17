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

import { Router, Request, Response, NextFunction } from 'express';
import * as crypto from 'crypto';
import { config } from '../config';
import { logger } from '../config/logger';
import { getChatHTML, getLoginHTML } from './ui';

/** Session cookie name */
const SESSION_COOKIE = 'vex_session';

/** Session duration: 7 days in ms */
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

/** Interval (ms) between SSE keep-alive pings */
const SSE_KEEPALIVE_INTERVAL_MS = 15_000;

interface Session {
  id: string;
  createdAt: number;
  expiresAt: number;
}

const sessions = new Map<string, Session>();

function createSession(): string {
  const id = crypto.randomBytes(32).toString('hex');
  const now = Date.now();
  sessions.set(id, { id, createdAt: now, expiresAt: now + SESSION_TTL_MS });
  return id;
}

function isValidSession(sessionId: string | undefined): boolean {
  if (!sessionId) return false;
  const session = sessions.get(sessionId);
  if (!session) return false;
  if (Date.now() > session.expiresAt) {
    sessions.delete(sessionId);
    return false;
  }
  return true;
}

function getCookie(req: Request, name: string): string | undefined {
  const header = req.headers.cookie;
  if (!header) return undefined;
  const match = header.split(';').find((c) => c.trim().startsWith(name + '='));
  if (!match) return undefined;
  return match.split('=').slice(1).join('=').trim();
}

function requireAuth(req: Request, res: Response, next: NextFunction): void {
  if (!config.chatAuthToken) {
    next();
    return;
  }

  const sessionId = getCookie(req, SESSION_COOKIE);
  if (isValidSession(sessionId)) {
    next();
    return;
  }

  const acceptsHtml =
    req.headers.accept?.includes('text/html') || req.method === 'GET';

  if (acceptsHtml && req.path === '/chat') {
    res.setHeader('Content-Type', 'text/html');
    res.send(getLoginHTML());
    return;
  }

  res.status(401).json({ error: 'Authentication required' });
}

export function createChatRouter(kernelUrl: string): Router {
  const router = Router();

  // ── Auth endpoint ─────────────────────────────────────────────
  router.post('/chat/auth', (req: Request, res: Response) => {
    if (!config.chatAuthToken) {
      const sessionId = createSession();
      res.setHeader(
        'Set-Cookie',
        `${SESSION_COOKIE}=${sessionId}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}`,
      );
      res.json({ ok: true });
      return;
    }

    const { token } = req.body as { token?: string };
    if (!token || token !== config.chatAuthToken) {
      res.status(403).json({ error: 'Invalid token' });
      return;
    }

    const sessionId = createSession();
    res.setHeader(
      'Set-Cookie',
      `${SESSION_COOKIE}=${sessionId}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}`,
    );
    res.json({ ok: true });
  });

  // ── Serve chat UI (auth-gated) ────────────────────────────────
  router.get('/chat', requireAuth, (_req: Request, res: Response) => {
    res.setHeader('Content-Type', 'text/html');
    res.send(getChatHTML());
  });

  // ── Kernel status (proxied) ───────────────────────────────────
  router.get('/chat/status', async (_req: Request, res: Response) => {
    try {
      const resp = await fetch(`${kernelUrl}/status`);
      const data = await resp.json();
      res.json(data);
    } catch (err) {
      res.status(502).json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // ── Conversation history (proxied) ────────────────────────────
  router.get('/chat/history', async (req: Request, res: Response) => {
    try {
      const convId = req.query.id ? `?id=${req.query.id}` : '';
      const resp = await fetch(`${kernelUrl}/chat/history${convId}`);
      const data = await resp.json();
      res.json(data);
    } catch (err) {
      res.status(502).json({ error: `Kernel unreachable: ${(err as Error).message}` });
    }
  });

  // ── Streaming chat endpoint via SSE (auth-gated) ──────────────
  // Proxies the request to the Python kernel's /chat/stream endpoint
  // and pipes the SSE response back to the client
  router.post('/chat/stream', requireAuth, async (req: Request, res: Response) => {
    const { message, conversationId } = req.body as {
      message?: string;
      conversationId?: string;
    };

    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Missing "message" field' });
      return;
    }

    // ── SSE headers ──────────────────────────────────────────────
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    req.setTimeout(0);
    res.setTimeout(0);

    const keepAlive = setInterval(() => {
      try {
        res.write(': ping\n\n');
      } catch {
        // Connection closed
      }
    }, SSE_KEEPALIVE_INTERVAL_MS);

    try {
      // Proxy to Python kernel's streaming endpoint
      const kernelResp = await fetch(`${kernelUrl}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, conversation_id: conversationId }),
      });

      if (!kernelResp.ok) {
        const errBody = await kernelResp.text();
        res.write(`data: ${JSON.stringify({ type: 'error', error: errBody })}\n\n`);
        clearInterval(keepAlive);
        res.end();
        return;
      }

      // Pipe the SSE stream from the kernel to the client
      const reader = kernelResp.body?.getReader();
      if (!reader) {
        res.write(`data: ${JSON.stringify({ type: 'error', error: 'No response body from kernel' })}\n\n`);
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
        if (typeof (res as any).flush === 'function') {
          (res as any).flush();
        }
      }
    } catch (err) {
      const errMsg = (err as Error).message;
      logger.error('Chat stream proxy error', { error: errMsg });
      res.write(`data: ${JSON.stringify({ type: 'error', error: errMsg })}\n\n`);
    } finally {
      clearInterval(keepAlive);
      res.end();
    }
  });

  return router;
}
