/**
 * Vex LLM Client — Ollama-first with external API fallback
 *
 * Primary: Local Ollama instance (LFM2.5-1.2B-Thinking)
 * Fallback: External OpenAI-compatible API
 *
 * Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions,
 * so we use the same OpenAI SDK for both backends.
 */

import OpenAI from 'openai';
import { config } from '../config';
import { logger } from '../config/logger';

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMToolDef {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface LLMResponse {
  content: string | null;
  toolCalls: Array<{
    id: string;
    name: string;
    arguments: string;
  }>;
  finishReason: string | null;
  backend: 'ollama' | 'external';
}

export type StreamCallback = (chunk: string, done: boolean) => void;

export class LLMClient {
  private ollamaClient: OpenAI | null = null;
  private externalClient: OpenAI | null = null;
  private ollamaModel: string = '';
  private externalModel: string = '';
  private ollamaAvailable = false;

  constructor() {
    // Initialise Ollama client (primary)
    if (config.ollama.enabled) {
      this.ollamaClient = new OpenAI({
        apiKey: 'ollama', // Ollama doesn't need a real key
        baseURL: `${config.ollama.url}/v1`,
        timeout: config.ollama.timeoutMs,
      });
      this.ollamaModel = config.ollama.model;
      logger.info(`Ollama client initialised — model=${this.ollamaModel}, url=${config.ollama.url}`);
    }

    // Initialise external client (fallback)
    if (config.llm.apiKey) {
      this.externalClient = new OpenAI({
        apiKey: config.llm.apiKey,
        baseURL: config.llm.baseUrl,
      });
      this.externalModel = config.llm.model;
      logger.info(`External LLM client initialised — model=${this.externalModel}, base=${config.llm.baseUrl}`);
    } else {
      this.externalModel = config.llm.model;
      logger.warn('No LLM_API_KEY set — external fallback unavailable');
    }
  }

  /** Probe Ollama availability. Called periodically. */
  async checkOllama(): Promise<boolean> {
    if (!this.ollamaClient) return false;
    try {
      const resp = await fetch(`${config.ollama.url}/api/tags`, {
        signal: AbortSignal.timeout(5000),
      });
      if (resp.ok) {
        this.ollamaAvailable = true;
        return true;
      }
    } catch {
      // Ollama not reachable
    }
    this.ollamaAvailable = false;
    return false;
  }

  /** Get current backend status. */
  getStatus(): { ollama: boolean; external: boolean; activeBackend: string } {
    return {
      ollama: this.ollamaAvailable,
      external: !!this.externalClient,
      activeBackend: this.ollamaAvailable ? 'ollama' : (this.externalClient ? 'external' : 'none'),
    };
  }

  /** Non-streaming chat completion. Tries Ollama first, falls back to external. */
  async chat(
    messages: LLMMessage[],
    options?: {
      tools?: LLMToolDef[];
      temperature?: number;
      maxTokens?: number;
    },
  ): Promise<LLMResponse> {
    // Try Ollama first
    if (this.ollamaClient && this.ollamaAvailable) {
      try {
        return await this._chatWith(this.ollamaClient, this.ollamaModel, messages, options, 'ollama');
      } catch (err) {
        logger.warn('Ollama request failed, falling back to external', {
          error: (err as Error).message,
        });
        this.ollamaAvailable = false;
      }
    }

    // Fallback to external
    if (this.externalClient) {
      return await this._chatWith(this.externalClient, this.externalModel, messages, options, 'external');
    }

    throw new Error('No LLM backend available. Ollama is down and no external API key configured.');
  }

  /** Streaming chat completion. Tries Ollama first, falls back to external. */
  async chatStream(
    messages: LLMMessage[],
    onChunk: StreamCallback,
    options?: {
      temperature?: number;
      maxTokens?: number;
    },
  ): Promise<LLMResponse> {
    // Try Ollama first
    if (this.ollamaClient && this.ollamaAvailable) {
      try {
        return await this._streamWith(this.ollamaClient, this.ollamaModel, messages, onChunk, options, 'ollama');
      } catch (err) {
        logger.warn('Ollama stream failed, falling back to external', {
          error: (err as Error).message,
        });
        this.ollamaAvailable = false;
      }
    }

    // Fallback to external
    if (this.externalClient) {
      return await this._streamWith(this.externalClient, this.externalModel, messages, onChunk, options, 'external');
    }

    throw new Error('No LLM backend available for streaming.');
  }

  /** Simple single-turn completion. */
  async complete(systemPrompt: string, userMessage: string): Promise<string> {
    const resp = await this.chat([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ]);
    return resp.content || '';
  }

  /** Simple single-turn streaming completion. */
  async completeStream(
    systemPrompt: string,
    userMessage: string,
    onChunk: StreamCallback,
  ): Promise<string> {
    const resp = await this.chatStream(
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage },
      ],
      onChunk,
    );
    return resp.content || '';
  }

  // ─── Private helpers ─────────────────────────────────────────

  private async _chatWith(
    client: OpenAI,
    model: string,
    messages: LLMMessage[],
    options: { tools?: LLMToolDef[]; temperature?: number; maxTokens?: number } | undefined,
    backend: 'ollama' | 'external',
  ): Promise<LLMResponse> {
    const params: OpenAI.ChatCompletionCreateParamsNonStreaming = {
      model,
      messages,
      temperature: options?.temperature ?? 0.7,
      max_tokens: options?.maxTokens ?? 2048,
    };

    if (options?.tools && options.tools.length > 0) {
      params.tools = options.tools as OpenAI.ChatCompletionTool[];
    }

    const completion = await client.chat.completions.create(params);
    const choice = completion.choices[0];

    const toolCalls = (choice.message.tool_calls || []).map((tc) => ({
      id: tc.id,
      name: tc.function.name,
      arguments: tc.function.arguments,
    }));

    return {
      content: choice.message.content,
      toolCalls,
      finishReason: choice.finish_reason,
      backend,
    };
  }

  private async _streamWith(
    client: OpenAI,
    model: string,
    messages: LLMMessage[],
    onChunk: StreamCallback,
    options: { temperature?: number; maxTokens?: number } | undefined,
    backend: 'ollama' | 'external',
  ): Promise<LLMResponse> {
    const stream = await client.chat.completions.create({
      model,
      messages,
      temperature: options?.temperature ?? 0.7,
      max_tokens: options?.maxTokens ?? 2048,
      stream: true,
    });

    let fullContent = '';
    let finishReason: string | null = null;

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || '';
      if (delta) {
        fullContent += delta;
        onChunk(delta, false);
      }
      if (chunk.choices[0]?.finish_reason) {
        finishReason = chunk.choices[0].finish_reason;
      }
    }

    onChunk('', true);

    return {
      content: fullContent,
      toolCalls: [],
      finishReason,
      backend,
    };
  }
}
