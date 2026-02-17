/**
 * ComputeSDK Sandbox Tool — Code Execution via Railway
 *
 * Provides Vex with isolated sandbox environments for:
 *   - Code execution (Python, Node.js)
 *   - Shell command execution
 *   - File system operations
 *
 * Uses ComputeSDK with Railway provider. The ComputeSDK service
 * is deployed alongside Vex on Railway.
 *
 * The Python kernel calls these tools via the TS proxy's
 * /api/tools/execute_code and /api/tools/run_command endpoints.
 *
 * Docs: https://www.computesdk.com/docs/providers/railway/
 */

import { logger } from '../config/logger';

// ═══════════════════════════════════════════════════════════════
//  TYPES (inlined — no dependency on deleted registry.ts)
// ═══════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════
//  SANDBOX MANAGER — manages ComputeSDK sandbox lifecycle
// ═══════════════════════════════════════════════════════════════

interface SandboxInstance {
  sandbox: any; // ComputeSDK Sandbox type
  createdAt: number;
  lastUsedAt: number;
}

/**
 * Manages a pool of ComputeSDK sandboxes.
 * Reuses sandboxes when possible, creates new ones when needed.
 */
export class SandboxManager {
  private activeSandbox: SandboxInstance | null = null;
  private compute: any = null;
  private initialised = false;
  /** Maximum sandbox idle time before cleanup (5 minutes) */
  private readonly MAX_IDLE_MS = 5 * 60 * 1000;

  /**
   * Lazily initialise the ComputeSDK client.
   * Deferred to avoid import errors if computesdk isn't installed yet.
   */
  private async ensureInit(): Promise<void> {
    if (this.initialised) return;

    try {
      const { compute } = await import('computesdk');
      this.compute = compute;

      // ComputeSDK auto-detects Railway from env vars:
      // COMPUTESDK_API_KEY, RAILWAY_API_KEY, RAILWAY_PROJECT_ID, RAILWAY_ENVIRONMENT_ID
      if (
        process.env.COMPUTESDK_API_KEY &&
        process.env.RAILWAY_API_KEY &&
        process.env.RAILWAY_PROJECT_ID &&
        process.env.RAILWAY_ENVIRONMENT_ID
      ) {
        compute.setConfig({
          computesdkApiKey: process.env.COMPUTESDK_API_KEY,
          provider: 'railway',
          railway: {
            apiToken: process.env.RAILWAY_API_KEY,
            projectId: process.env.RAILWAY_PROJECT_ID,
            environmentId: process.env.RAILWAY_ENVIRONMENT_ID,
          },
        });
      }

      this.initialised = true;
      logger.info('ComputeSDK initialised with Railway provider');
    } catch (err) {
      logger.warn('ComputeSDK not available — tool use will be limited', {
        error: (err as Error).message,
      });
      throw new Error(
        'ComputeSDK is not available. Install with: npm install computesdk',
      );
    }
  }

  /**
   * Get or create a sandbox instance.
   */
  async getSandbox(): Promise<any> {
    await this.ensureInit();

    // Reuse existing sandbox if it's still fresh
    if (this.activeSandbox) {
      const idleTime = Date.now() - this.activeSandbox.lastUsedAt;
      if (idleTime < this.MAX_IDLE_MS) {
        this.activeSandbox.lastUsedAt = Date.now();
        return this.activeSandbox.sandbox;
      }
      // Sandbox is stale — destroy and create new
      await this.destroySandbox();
    }

    // Create new sandbox
    try {
      const sandbox = await this.compute.sandbox.create();
      this.activeSandbox = {
        sandbox,
        createdAt: Date.now(),
        lastUsedAt: Date.now(),
      };
      logger.info('ComputeSDK sandbox created', {
        sandboxId: sandbox.sandboxId,
      });
      return sandbox;
    } catch (err) {
      logger.error('Failed to create ComputeSDK sandbox', {
        error: (err as Error).message,
      });
      throw err;
    }
  }

  /**
   * Destroy the active sandbox.
   */
  async destroySandbox(): Promise<void> {
    if (this.activeSandbox) {
      try {
        await this.activeSandbox.sandbox.destroy();
        logger.info('ComputeSDK sandbox destroyed');
      } catch (err) {
        logger.warn('Failed to destroy sandbox', {
          error: (err as Error).message,
        });
      }
      this.activeSandbox = null;
    }
  }

  /**
   * Check if ComputeSDK is available.
   */
  isAvailable(): boolean {
    return (
      !!process.env.COMPUTESDK_API_KEY ||
      !!process.env.RAILWAY_API_KEY
    );
  }
}

// Singleton sandbox manager
export const sandboxManager = new SandboxManager();

// ═══════════════════════════════════════════════════════════════
//  TOOL: execute_code
// ═══════════════════════════════════════════════════════════════

export const executeCodeTool: VexTool = {
  name: 'execute_code',
  description:
    'Execute Python or Node.js code in an isolated ComputeSDK sandbox. Returns stdout output and exit code.',
  parameters: {
    type: 'object',
    properties: {
      code: {
        type: 'string',
        description: 'The code to execute',
      },
      language: {
        type: 'string',
        enum: ['python', 'node'],
        description: 'Programming language (auto-detected if not specified)',
      },
    },
    required: ['code'],
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const code = args.code as string;
    const language = args.language as string | undefined;

    if (!sandboxManager.isAvailable()) {
      // Fallback: try local execution if ComputeSDK isn't configured
      return executeLocally(code, language);
    }

    try {
      const sandbox = await sandboxManager.getSandbox();
      const result = await sandbox.runCode(code, language);

      return {
        success: result.exitCode === 0,
        output: result.output || '(no output)',
        error:
          result.exitCode !== 0
            ? `Exit code: ${result.exitCode}`
            : undefined,
      };
    } catch (err) {
      // Fallback to local execution
      logger.warn('ComputeSDK execution failed, falling back to local', {
        error: (err as Error).message,
      });
      return executeLocally(code, language);
    }
  },
};

// ═══════════════════════════════════════════════════════════════
//  TOOL: run_command
// ═══════════════════════════════════════════════════════════════

export const runCommandTool: VexTool = {
  name: 'run_command',
  description:
    'Execute a shell command in an isolated ComputeSDK sandbox. Returns stdout, stderr, and exit code.',
  parameters: {
    type: 'object',
    properties: {
      command: {
        type: 'string',
        description: 'The shell command to execute',
      },
      cwd: {
        type: 'string',
        description: 'Working directory (optional)',
      },
      timeout: {
        type: 'number',
        description: 'Timeout in milliseconds (optional, default 30000)',
      },
    },
    required: ['command'],
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const command = args.command as string;
    const cwd = args.cwd as string | undefined;
    const timeout = (args.timeout as number) || 30000;

    if (!sandboxManager.isAvailable()) {
      return {
        success: false,
        output: '',
        error: 'ComputeSDK not configured — shell commands require a sandbox',
      };
    }

    try {
      const sandbox = await sandboxManager.getSandbox();
      const result = await sandbox.runCommand(command, { cwd, timeout });

      return {
        success: result.exitCode === 0,
        output: result.stdout || result.stderr || '(no output)',
        error:
          result.exitCode !== 0
            ? `Exit code ${result.exitCode}: ${result.stderr}`
            : undefined,
      };
    } catch (err) {
      return {
        success: false,
        output: '',
        error: `Command execution failed: ${(err as Error).message}`,
      };
    }
  },
};

// ═══════════════════════════════════════════════════════════════
//  LOCAL FALLBACK (when ComputeSDK isn't available)
// ═══════════════════════════════════════════════════════════════

async function executeLocally(
  code: string,
  language?: string,
): Promise<ToolResult> {
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);
  const { writeFile, unlink } = await import('fs/promises');
  const { join } = await import('path');
  const os = await import('os');

  const isPython =
    language === 'python' || (!language && code.includes('import '));
  const ext = isPython ? '.py' : '.js';
  const cmd = isPython ? 'python3' : 'node';
  const tmpFile = join(os.tmpdir(), `vex-exec-${Date.now()}${ext}`);

  try {
    await writeFile(tmpFile, code, 'utf-8');
    const { stdout, stderr } = await execAsync(`${cmd} ${tmpFile}`, {
      timeout: 30000,
      maxBuffer: 1024 * 1024,
    });
    return {
      success: true,
      output: stdout || stderr || '(no output)',
    };
  } catch (err: any) {
    return {
      success: false,
      output: err.stdout || '',
      error: err.stderr || err.message,
    };
  } finally {
    try {
      await unlink(tmpFile);
    } catch {
      // ignore cleanup errors
    }
  }
}

// ═══════════════════════════════════════════════════════════════
//  REGISTER ALL TOOLS
// ═══════════════════════════════════════════════════════════════

/**
 * Get all ComputeSDK tools for registration.
 */
export function getComputeTools(): VexTool[] {
  return [executeCodeTool, runCommandTool];
}
