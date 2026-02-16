/**
 * Vex Tool — Web Fetch
 *
 * Fetches a URL and returns the text content (truncated).
 */

import { VexTool, ToolResult } from './registry';

export const webFetchTool: VexTool = {
  name: 'web_fetch',
  description: 'Fetch a URL and return its text content. Useful for research and information gathering.',
  parameters: {
    type: 'object',
    properties: {
      url: { type: 'string', description: 'The URL to fetch' },
      max_length: {
        type: 'number',
        description: 'Maximum characters to return (default 4000)',
      },
    },
    required: ['url'],
  },

  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const url = args.url as string;
    const maxLength = (args.max_length as number) || 4000;

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);

      const resp = await fetch(url, {
        signal: controller.signal,
        headers: {
          'User-Agent': 'VexAgent/1.0',
          Accept: 'text/html, application/json, text/plain',
        },
      });
      clearTimeout(timeout);

      if (!resp.ok) {
        return {
          success: false,
          output: '',
          error: `HTTP ${resp.status}: ${resp.statusText}`,
        };
      }

      let text = await resp.text();
      // Strip HTML tags for readability
      text = text.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '');
      text = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
      text = text.replace(/<[^>]+>/g, ' ');
      text = text.replace(/\s+/g, ' ').trim();

      if (text.length > maxLength) {
        text = text.slice(0, maxLength) + '\n\n[...truncated]';
      }

      return { success: true, output: text };
    } catch (err) {
      return {
        success: false,
        output: '',
        error: `Fetch failed: ${(err as Error).message}`,
      };
    }
  },
};
