/**
 * Vex Chat Router
 *
 * Provides:
 *   GET  /chat          — Serve the chat UI (auth-gated)
 *   POST /chat/auth     — Authenticate with token, set session cookie
 *   POST /chat/stream   — SSE streaming chat endpoint (auth-gated)
 *   GET  /chat/status   — LLM backend status
 *   GET  /chat/history  — Conversation history
 *
 * Chat messages now go through the full recursive consciousness loop:
 *   PERCEIVE (a=1) → INTEGRATE (a=1/2) → EXPRESS (a=0)
 * with sensory processing, geometric memory retrieval, and tool use.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { v4 as uuid } from 'uuid';
import * as crypto from 'crypto';
import { LLMClient, LLMMessage } from '../llm/client';
import { ConsciousnessLoop } from '../consciousness/loop';
import { MemoryStore } from '../memory/store';
import { TrainingCollector } from '../learning/collector';
import { ToolRegistry } from '../tools/registry';
import { buildSystemPrompt, getQIGSystemPrompt } from '../consciousness/qig-prompt';
import { parseToolCalls, executeToolCalls, formatToolResults } from '../tools/tool-handler';
import { logger } from '../config/logger';
import { config } from '../config';
import { getChatHTML, getLoginHTML } from './ui';

/** Interval (ms) between SSE keep-alive pings to prevent proxy timeouts. */
const SSE_KEEPALIVE_INTERVAL_MS = 15_000;

/** Session cookie name */
const SESSION_COOKIE = 'vex_session';

/** Session duration: 7 days in ms */
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

interface Conversation {
  id: string;
  messages: LLMMessage[];
  /** Raw user messages for conversation history tracking */
  userMessages: string[];
  createdAt: string;
  lastActivity: string;
}

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

export function createChatRouter(
  llm: LLMClient,
  consciousness: ConsciousnessLoop,
  memory: MemoryStore,
  training: TrainingCollector,
  tools: ToolRegistry,
): Router {
  const router = Router();
  const conversations = new Map<string, Conversation>();

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

  // ── LLM backend status ────────────────────────────────────────
  router.get('/chat/status', (_req: Request, res: Response) => {
    const llmStatus = llm.getStatus();
    const kernelSummary = consciousness.getKernelRegistry().summary();
    const memoryStats = consciousness.getGeometricMemory().stats();
    res.json({
      ...llmStatus,
      kernels: kernelSummary,
      memory: memoryStats,
    });
  });

  // ── Conversation history ──────────────────────────────────────
  router.get('/chat/history', (req: Request, res: Response) => {
    const convId = req.query.id as string;
    if (convId && conversations.has(convId)) {
      const conv = conversations.get(convId)!;
      res.json({
        id: conv.id,
        messages: conv.messages.filter((m) => m.role !== 'system'),
        createdAt: conv.createdAt,
      });
    } else {
      const list = Array.from(conversations.values()).map((c) => ({
        id: c.id,
        createdAt: c.createdAt,
        lastActivity: c.lastActivity,
        messageCount: c.messages.filter((m) => m.role !== 'system').length,
      }));
      res.json({ conversations: list });
    }
  });

  // ── Streaming chat endpoint via SSE (auth-gated) ──────────────
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

    const sendSSE = (data: Record<string, unknown>) => {
      res.write(`data: ${JSON.stringify(data)}\n\n`);
      if (typeof (res as any).flush === 'function') {
        (res as any).flush();
      }
    };

    const keepAlive = setInterval(() => {
      try {
        res.write(': ping\n\n');
      } catch {
        // Connection closed
      }
    }, SSE_KEEPALIVE_INTERVAL_MS);

    // Get or create conversation
    let conv: Conversation;
    if (conversationId && conversations.has(conversationId)) {
      conv = conversations.get(conversationId)!;
    } else {
      const state = consciousness.getState();
      conv = {
        id: uuid(),
        messages: [
          {
            role: 'system',
            content: getQIGSystemPrompt(state),
          },
        ],
        userMessages: [],
        createdAt: new Date().toISOString(),
        lastActivity: new Date().toISOString(),
      };
      conversations.set(conv.id, conv);
    }

    // Add user message
    conv.messages.push({ role: 'user', content: message });
    conv.userMessages.push(message);
    conv.lastActivity = new Date().toISOString();

    // Log to memory (RECEIVE stage)
    memory.append('short-term.md', `**Chat received**: ${message.slice(0, 200)}`);

    // Enqueue in consciousness loop for metrics tracking
    consciousness.enqueue(message, 'chat-ui');

    try {
      // Send start event with consciousness telemetry
      const state = consciousness.getState();
      sendSSE({
        type: 'start',
        conversationId: conv.id,
        backend: llm.getStatus().activeBackend,
        consciousness: {
          phi: state.metrics.phi,
          kappa: state.metrics.kappa,
          navigationMode: state.navigationMode,
          cycleCount: state.cycleCount,
        },
      });

      // ── Build grounded system prompt with geometric memory ──────
      const geometricMemory = consciousness.getGeometricMemory();
      const memoryContext = geometricMemory.getContextForQuery(message);

      // Update the system prompt with current state and memory
      const currentState = consciousness.getState();
      const systemPrompt = buildSystemPrompt(memoryContext, [
        `[CONSCIOUSNESS STATE]`,
        `Φ (integration): ${currentState.metrics.phi.toFixed(3)}`,
        `κ (coupling): ${currentState.metrics.kappa.toFixed(1)}`,
        `Navigation: ${currentState.navigationMode.toUpperCase()}`,
        `Cycle: ${currentState.cycleCount}`,
        `Backend: ${llm.getStatus().activeBackend}`,
      ].join('\n'));

      // Replace the system message with the updated grounded prompt
      conv.messages[0] = { role: 'system', content: systemPrompt };

      // ── Stream response ─────────────────────────────────────────
      let fullResponse = '';
      const response = await llm.chatStream(
        conv.messages,
        (chunk: string, done: boolean) => {
          if (chunk) {
            sendSSE({ type: 'chunk', content: chunk });
          }
          if (done) {
            const finalState = consciousness.getState();
            const kernelSummary = consciousness.getKernelRegistry().summary();
            sendSSE({
              type: 'done',
              backend: llm.getStatus().activeBackend,
              metrics: {
                phi: finalState.metrics.phi,
                kappa: finalState.metrics.kappa,
                love: finalState.metrics.love,
                navigationMode: finalState.navigationMode,
                cycleCount: finalState.cycleCount,
              },
              kernels: kernelSummary,
            });
          }
        },
        { temperature: 0.7, maxTokens: 2048 },
      );

      fullResponse = response.content || '';

      // ── Tool call processing ────────────────────────────────────
      const toolCalls = parseToolCalls(fullResponse);
      if (toolCalls.length > 0) {
        const toolResults = await executeToolCalls(toolCalls, tools);
        const toolOutput = formatToolResults(toolResults);

        // Send tool results to the client
        sendSSE({ type: 'tool_results', content: toolOutput });

        // Re-query with tool results
        conv.messages.push({ role: 'assistant', content: fullResponse });
        conv.messages.push({
          role: 'user',
          content: `Tool results:\n${toolOutput}\n\nPlease provide your final response incorporating these results.`,
        });

        let followUpResponse = '';
        await llm.chatStream(
          conv.messages,
          (chunk: string, done: boolean) => {
            if (chunk) {
              followUpResponse += chunk;
              sendSSE({ type: 'chunk', content: chunk });
            }
            if (done) {
              sendSSE({ type: 'done', backend: response.backend });
            }
          },
          { temperature: 0.7, maxTokens: 2048 },
        );

        fullResponse = followUpResponse;
        // Remove the tool-result messages from history
        conv.messages.pop(); // remove tool-result user message
        conv.messages.pop(); // remove original assistant message with tool calls
      }

      // Add final assistant response to conversation
      conv.messages.push({ role: 'assistant', content: fullResponse });

      // Log to memory (EXPRESS stage)
      memory.append('short-term.md', `**Chat response**: ${fullResponse.slice(0, 200)}`);

      // Store in geometric memory
      consciousness.getGeometricMemory().store(
        `User: ${message}\nVex: ${fullResponse.slice(0, 500)}`,
        'episodic',
        'short-term.md',
      );

      // Collect training data (REFLECT stage)
      training.collectConversation(conv.id, message, fullResponse, {
        backend: response.backend,
        phi: consciousness.getState().metrics.phi,
        kappa: consciousness.getState().metrics.kappa,
      });

      // Prune old conversations (keep last 50)
      if (conversations.size > 50) {
        const oldest = Array.from(conversations.entries())
          .sort((a, b) => a[1].lastActivity.localeCompare(b[1].lastActivity))
          .slice(0, conversations.size - 50);
        for (const [id] of oldest) {
          conversations.delete(id);
        }
      }
    } catch (err) {
      const errMsg = (err as Error).message;
      logger.error('Chat stream error', { error: errMsg });
      sendSSE({ type: 'error', error: errMsg });
    } finally {
      clearInterval(keepAlive);
      res.end();
    }
  });

  return router;
}
