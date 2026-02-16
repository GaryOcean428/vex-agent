/**
 * Vex LLM Client
 *
 * Wraps the OpenAI SDK for multi-backend support.
 * Supports structured output via tool/function calling.
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
}

export class LLMClient {
  private client: OpenAI;
  private model: string;

  constructor() {
    this.client = new OpenAI({
      apiKey: config.llm.apiKey,
      baseURL: config.llm.baseUrl,
    });
    this.model = config.llm.model;
    logger.info(`LLM client initialised — model=${this.model}, base=${config.llm.baseUrl}`);
  }

  async chat(
    messages: LLMMessage[],
    options?: {
      tools?: LLMToolDef[];
      temperature?: number;
      maxTokens?: number;
    },
  ): Promise<LLMResponse> {
    try {
      const params: OpenAI.ChatCompletionCreateParamsNonStreaming = {
        model: this.model,
        messages,
        temperature: options?.temperature ?? 0.7,
        max_tokens: options?.maxTokens ?? 2048,
      };

      if (options?.tools && options.tools.length > 0) {
        params.tools = options.tools as OpenAI.ChatCompletionTool[];
      }

      const completion = await this.client.chat.completions.create(params);
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
      };
    } catch (err) {
      logger.error('LLM request failed', { error: (err as Error).message });
      throw err;
    }
  }

  /** Simple single-turn completion. */
  async complete(systemPrompt: string, userMessage: string): Promise<string> {
    const resp = await this.chat([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ]);
    return resp.content || '';
  }
}
