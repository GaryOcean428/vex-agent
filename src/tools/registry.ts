/**
 * Vex Tool Registry
 *
 * Central registry for all available tools.
 * Each tool exposes a schema (for LLM function calling) and an execute method.
 */

import { LLMToolDef } from '../llm/client';
import { PurityGate } from '../safety/purity-gate';
import { logger } from '../config/logger';

export interface ToolResult {
  success: boolean;
  output: string;
  error?: string;
}

export interface VexTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  execute(args: Record<string, unknown>): Promise<ToolResult>;
}

export class ToolRegistry {
  private tools = new Map<string, VexTool>();
  private purityGate: PurityGate;

  constructor(purityGate: PurityGate) {
    this.purityGate = purityGate;
  }

  register(tool: VexTool): void {
    this.tools.set(tool.name, tool);
    logger.info(`Tool registered: ${tool.name}`);
  }

  get(name: string): VexTool | undefined {
    return this.tools.get(name);
  }

  /** Get all tool definitions for LLM function calling. */
  getToolDefs(): LLMToolDef[] {
    return Array.from(this.tools.values()).map((t) => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      },
    }));
  }

  /** Execute a tool by name with safety checks. */
  async execute(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    const tool = this.tools.get(name);
    if (!tool) {
      return { success: false, output: '', error: `Unknown tool: ${name}` };
    }

    // Safety check
    const argsStr = JSON.stringify(args);
    const check = this.purityGate.check(name, argsStr);
    if (check.verdict === 'block') {
      return {
        success: false,
        output: '',
        error: `Blocked by PurityGate: ${check.reason}`,
      };
    }
    if (check.verdict === 'review') {
      return {
        success: false,
        output: '',
        error: `Requires human review: ${check.reason}`,
      };
    }

    try {
      const result = await tool.execute(args);
      logger.info(`Tool executed: ${name}`, { success: result.success });
      return result;
    } catch (err) {
      const msg = (err as Error).message;
      logger.error(`Tool execution failed: ${name}`, { error: msg });
      return { success: false, output: '', error: msg };
    }
  }

  listTools(): string[] {
    return Array.from(this.tools.keys());
  }
}
