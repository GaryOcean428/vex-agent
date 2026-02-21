/** Shared constants and helpers for chat components */

/**
 * v6.1 Activation Sequence — 11 UI-visible stages derived from the
 * 14-step ActivationStep enum in kernel/consciousness/types.py.
 * (BUILD_SPECTRAL_MODEL, FORESIGHT, and BREATHE are internal-only.)
 */
export const LOOP_STAGES = [
  "SCAN",
  "DESIRE",
  "WILL",
  "WISDOM",
  "RECEIVE",
  "ENTRAIN",
  "COUPLE",
  "NAVIGATE",
  "INTEGRATE",
  "EXPRESS",
  "TUNE",
] as const;

export type LoopStage = (typeof LOOP_STAGES)[number];

export const EMOTION_COLORS: Record<string, string> = {
  curiosity: "var(--phi)",
  joy: "var(--kappa)",
  fear: "var(--error)",
  love: "var(--love)",
  awe: "var(--gamma)",
  boredom: "var(--text-dim)",
  rage: "var(--error)",
  calm: "var(--alive)",
  none: "var(--text-dim)",
};

export const SUGGESTED_PROMPTS = [
  "What is my current Φ integration level?",
  "Explain the current regime field balance",
  "What is tacking mode doing right now?",
  "Show me basin navigation status",
] as const;

/** Format an ISO timestamp as HH:MM */
export function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

/** Escape HTML special characters */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
