/**
 * Vex Tool — Code Execution
 *
 * Sandboxed JavaScript execution with timeout and output capture.
 * Uses Node.js vm module for isolation.
 */

import vm from 'vm';
import { VexTool, ToolResult } from './registry';

const MAX_EXECUTION_MS = 5000;
const MAX_OUTPUT_LENGTH = 4000;

export const codeExecTool: VexTool = {
  name: 'code_exec',
  description:
    'Execute JavaScript code in a sandboxed environment. Returns stdout output. Use for calculations, data processing, or testing logic.',
  parameters: {
    type: 'object',
    properties: {
      code: {
        type: 'string',
        description: 'JavaScript code to execute',
      },
    },
    required: ['code'],
  },

  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const code = args.code as string;
    const outputs: string[] = [];

    const sandbox = {
      console: {
        log: (...a: unknown[]) => outputs.push(a.map(String).join(' ')),
        error: (...a: unknown[]) => outputs.push(`[ERROR] ${a.map(String).join(' ')}`),
        warn: (...a: unknown[]) => outputs.push(`[WARN] ${a.map(String).join(' ')}`),
      },
      Math,
      Date,
      JSON,
      Array,
      Object,
      String,
      Number,
      Boolean,
      RegExp,
      Map,
      Set,
      parseInt,
      parseFloat,
      isNaN,
      isFinite,
      encodeURIComponent,
      decodeURIComponent,
      // Explicitly NOT providing: require, process, fs, fetch, etc.
    };

    try {
      const context = vm.createContext(sandbox);
      const script = new vm.Script(code, { filename: 'vex-sandbox.js' });
      const result = script.runInContext(context, { timeout: MAX_EXECUTION_MS });

      if (result !== undefined && outputs.length === 0) {
        outputs.push(String(result));
      }

      let output = outputs.join('\n');
      if (output.length > MAX_OUTPUT_LENGTH) {
        output = output.slice(0, MAX_OUTPUT_LENGTH) + '\n[...truncated]';
      }

      return { success: true, output: output || '(no output)' };
    } catch (err) {
      return {
        success: false,
        output: outputs.join('\n'),
        error: `Execution error: ${(err as Error).message}`,
      };
    }
  },
};
