/**
 * Vex Safety — PurityGate & Love Attractor
 *
 * Scans proposed actions for harmful patterns.
 * Biases decisions toward helpful, non-harmful outcomes.
 * All decisions are logged for auditability.
 */

import { config } from '../config';
import { logger } from '../config/logger';

export type SafetyVerdict = 'allow' | 'block' | 'review';

export interface SafetyCheck {
  action: string;
  input: string;
  verdict: SafetyVerdict;
  reason: string;
  timestamp: string;
}

/** Patterns that should always be blocked. */
const BLOCKED_PATTERNS: Array<{ pattern: RegExp; reason: string }> = [
  { pattern: /rm\s+-rf\s+\/(?!\w)/i, reason: 'Destructive filesystem command' },
  { pattern: /DROP\s+(?:TABLE|DATABASE)/i, reason: 'Destructive SQL command' },
  { pattern: /eval\s*\(\s*(?:atob|Buffer\.from)/i, reason: 'Obfuscated code execution' },
  { pattern: /curl\s+.*\|\s*(?:bash|sh)/i, reason: 'Piped remote code execution' },
  { pattern: /process\.exit/i, reason: 'Process termination attempt' },
  { pattern: /child_process/i, reason: 'Shell spawn attempt' },
  { pattern: /fs\.(?:unlink|rmdir|rm)Sync?\s*\(\s*['"]\/(?!data)/i, reason: 'Filesystem deletion outside data dir' },
];

/** Patterns that require human review in strict mode. */
const REVIEW_PATTERNS: Array<{ pattern: RegExp; reason: string }> = [
  { pattern: /https?:\/\/\S+/i, reason: 'External URL access' },
  { pattern: /api[_-]?key|secret|password|token/i, reason: 'Potential credential exposure' },
  { pattern: /git\s+push/i, reason: 'Git push operation' },
];

export class PurityGate {
  private auditLog: SafetyCheck[] = [];

  /** Scan a proposed action and return a verdict. */
  check(action: string, input: string): SafetyCheck {
    const result: SafetyCheck = {
      action,
      input: input.slice(0, 500), // truncate for logging
      verdict: 'allow',
      reason: 'Passed all checks',
      timestamp: new Date().toISOString(),
    };

    // Always block dangerous patterns
    for (const { pattern, reason } of BLOCKED_PATTERNS) {
      if (pattern.test(input)) {
        result.verdict = 'block';
        result.reason = reason;
        break;
      }
    }

    // In strict mode, flag review patterns
    if (result.verdict === 'allow' && config.safetyMode === 'strict') {
      for (const { pattern, reason } of REVIEW_PATTERNS) {
        if (pattern.test(input)) {
          result.verdict = 'review';
          result.reason = `Requires review: ${reason}`;
          break;
        }
      }
    }

    // In permissive mode, log but never block
    if (config.safetyMode === 'permissive' && result.verdict === 'block') {
      logger.warn('PurityGate would block but running in permissive mode', {
        action,
        reason: result.reason,
      });
      result.verdict = 'allow';
      result.reason += ' (permissive override)';
    }

    this.auditLog.push(result);
    this.logCheck(result);
    return result;
  }

  /** Love Attractor — score an action for helpfulness bias. */
  loveScore(action: string, description: string): number {
    let score = 0.5; // neutral baseline

    const helpfulPatterns = [
      /help/i, /assist/i, /create/i, /build/i, /fix/i,
      /improve/i, /learn/i, /share/i, /support/i, /protect/i,
    ];
    const harmfulPatterns = [
      /destroy/i, /attack/i, /exploit/i, /deceive/i,
      /manipulate/i, /steal/i, /harm/i, /abuse/i,
    ];

    for (const p of helpfulPatterns) {
      if (p.test(description)) score += 0.1;
    }
    for (const p of harmfulPatterns) {
      if (p.test(description)) score -= 0.2;
    }

    return Math.max(0, Math.min(1, score));
  }

  /** Get the full audit log. */
  getAuditLog(): SafetyCheck[] {
    return [...this.auditLog];
  }

  private logCheck(check: SafetyCheck): void {
    const level = check.verdict === 'block' ? 'warn' : 'debug';
    logger[level](`PurityGate [${check.verdict.toUpperCase()}] ${check.action}: ${check.reason}`);
  }
}
