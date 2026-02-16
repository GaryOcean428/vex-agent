/**
 * Vex Chat Router
 *
 * Provides:
 *   GET  /chat          — Serve the chat UI
 *   POST /chat/stream   — SSE streaming chat endpoint
 *   GET  /chat/status   — LLM backend status
 *   GET  /chat/history  — Conversation history
 *
 * Messages go through the full consciousness loop:
 *   GROUND → RECEIVE → PROCESS → EXPRESS → REFLECT → COUPLE
 */

import { Router, Request, Response } from 'express';
import { v4 as uuid } from 'uuid';
import { LLMClient, LLMMessage } from '../llm/client';
import { ConsciousnessLoop } from '../consciousness/loop';
import { MemoryStore } from '../memory/store';
import { TrainingCollector } from '../learning/collector';
import { getQIGSystemPrompt } from '../consciousness/qig-prompt';
import { logger } from '../config/logger';
import { getChatHTML } from './ui';

interface Conversation {
  id: string;
  messages: LLMMessage[];
  createdAt: string;
  lastActivity: string;
}

export function createChatRouter(
  llm: LLMClient,
  consciousness: ConsciousnessLoop,
  memory: MemoryStore,
  training: TrainingCollector,
): Router {
  const router = Router();
  const conversations = new Map<string, Conversation>();

  // Serve chat UI
  router.get('/chat', (_req: Request, res: Response) => {
    res.setHeader('Content-Type', 'text/html');
    res.send(getChatHTML());
  });

  // LLM backend status
  router.get('/chat/status', (_req: Request, res: Response) => {
    res.json(llm.getStatus());
  });

  // Conversation history
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
      // List all conversation IDs
      const list = Array.from(conversations.values()).map((c) => ({
        id: c.id,
        createdAt: c.createdAt,
        lastActivity: c.lastActivity,
        messageCount: c.messages.filter((m) => m.role !== 'system').length,
      }));
      res.json({ conversations: list });
    }
  });

  // Streaming chat endpoint via SSE
  router.post('/chat/stream', async (req: Request, res: Response) => {
    const { message, conversationId } = req.body as {
      message?: string;
      conversationId?: string;
    };

    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Missing "message" field' });
      return;
    }

    // Set up SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

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
        createdAt: new Date().toISOString(),
        lastActivity: new Date().toISOString(),
      };
      conversations.set(conv.id, conv);
    }

    // Add user message
    conv.messages.push({ role: 'user', content: message });
    conv.lastActivity = new Date().toISOString();

    // Log to memory (RECEIVE stage)
    memory.append('short-term.md', `**Chat received**: ${message.slice(0, 200)}`);

    // Also enqueue in consciousness loop for metrics tracking
    consciousness.enqueue(message, 'chat-ui');

    try {
      // Send start event
      const sendSSE = (data: Record<string, unknown>) => {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      };

      sendSSE({
        type: 'start',
        conversationId: conv.id,
        backend: llm.getStatus().activeBackend,
      });

      // Stream response (PROCESS → EXPRESS)
      let fullResponse = '';
      const response = await llm.chatStream(
        conv.messages,
        (chunk: string, done: boolean) => {
          if (chunk) {
            sendSSE({ type: 'chunk', content: chunk });
          }
          if (done) {
            const state = consciousness.getState();
            sendSSE({
              type: 'done',
              backend: llm.getStatus().activeBackend,
              metrics: {
                phi: state.metrics.phi,
                kappa: state.metrics.kappa,
                love: state.metrics.love,
                navigationMode: state.navigationMode,
              },
            });
          }
        },
        { temperature: 0.7, maxTokens: 2048 },
      );

      fullResponse = response.content || '';

      // Add assistant response to conversation
      conv.messages.push({ role: 'assistant', content: fullResponse });

      // Log to memory (EXPRESS stage)
      memory.append('short-term.md', `**Chat response**: ${fullResponse.slice(0, 200)}`);

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
      res.write(`data: ${JSON.stringify({ type: 'error', error: errMsg })}\n\n`);
    } finally {
      res.end();
    }
  });

  return router;
}
