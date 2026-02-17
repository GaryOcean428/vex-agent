/**
 * Tool Use Handler — Parses and Executes Tool Calls from LLM Responses
 *
 * Vex's LLM can emit tool call blocks in its responses:
 *   ```tool:execute_code
 *   {"code": "print(2+2)", "language": "python"}
 *   ```
 *
 * This handler:
 * 1. Detects tool call blocks in the response text
 * 2. Parses the tool name and arguments
 * 3. Executes via the ToolRegistry
 * 4. Returns results for inclusion in the conversation
 */

import { ToolRegistry, ToolResult } from './registry';
import { logger } from '../config/logger';

export interface ParsedToolCall {
  toolName: string;
  args: Record<string, unknown>;
  /** The raw block text (for replacement in the response) */
  rawBlock: string;
}

export interface ToolCallResult {
  call: ParsedToolCall;
  result: ToolResult;
}

/**
 * Parse tool call blocks from LLM response text.
 *
 * Format:
 *   ```tool:<tool_name>
 *   <json_args>
 *   ```
 */
export function parseToolCalls(text: string): ParsedToolCall[] {
  const calls: ParsedToolCall[] = [];
  const regex = /```tool:(\w+)\s*\n([\s\S]*?)```/g;

  let match;
  while ((match = regex.exec(text)) !== null) {
    const toolName = match[1];
    const argsText = match[2].trim();

    try {
      const args = JSON.parse(argsText);
      calls.push({
        toolName,
        args,
        rawBlock: match[0],
      });
    } catch (err) {
      logger.warn(`Failed to parse tool call args for ${toolName}`, {
        argsText,
        error: (err as Error).message,
      });
    }
  }

  return calls;
}

/**
 * Execute all parsed tool calls and return results.
 */
export async function executeToolCalls(
  calls: ParsedToolCall[],
  registry: ToolRegistry,
): Promise<ToolCallResult[]> {
  const results: ToolCallResult[] = [];

  for (const call of calls) {
    logger.info(`Executing tool: ${call.toolName}`, { args: call.args });
    const result = await registry.execute(call.toolName, call.args);
    results.push({ call, result });
  }

  return results;
}

/**
 * Format tool results for inclusion in the conversation.
 */
export function formatToolResults(results: ToolCallResult[]): string {
  if (results.length === 0) return '';

  const lines: string[] = [];
  for (const { call, result } of results) {
    lines.push(`**Tool: ${call.toolName}**`);
    if (result.success) {
      lines.push('```');
      lines.push(result.output);
      lines.push('```');
    } else {
      lines.push(`Error: ${result.error}`);
    }
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Process a response that may contain tool calls.
 * Returns the response with tool call blocks replaced by results.
 */
export async function processToolCalls(
  responseText: string,
  registry: ToolRegistry,
): Promise<{ text: string; hadToolCalls: boolean }> {
  const calls = parseToolCalls(responseText);

  if (calls.length === 0) {
    return { text: responseText, hadToolCalls: false };
  }

  const results = await executeToolCalls(calls, registry);

  // Replace tool call blocks with results
  let processed = responseText;
  for (const { call, result } of results) {
    const replacement = result.success
      ? `**\`${call.toolName}\` result:**\n\`\`\`\n${result.output}\n\`\`\``
      : `**\`${call.toolName}\` error:** ${result.error}`;
    processed = processed.replace(call.rawBlock, replacement);
  }

  return { text: processed, hadToolCalls: true };
}
