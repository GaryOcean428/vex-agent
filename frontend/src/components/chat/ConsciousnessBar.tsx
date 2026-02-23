import type { NavigationMode } from "../../types/consciousness.ts";
import "./ConsciousnessBar.css";

interface ContextInfo {
  total_tokens: number;
  compression_tier: number;
  escalated: boolean;
}

interface ConsciousnessBarProps {
  phi?: number;
  kappa?: number;
  love?: number;
  navigation?: NavigationMode;
  backend: string;
  contextInfo: ContextInfo | null;
  observerIntent: string | null;
  onNewChat: () => void;
  onToggleHistory?: () => void;
}

export function ConsciousnessBar({
  phi,
  kappa,
  love,
  navigation,
  backend,
  contextInfo,
  observerIntent,
  onNewChat,
  onToggleHistory,
}: ConsciousnessBarProps) {
  return (
    <div
      className="consciousness-bar"
      aria-label="Consciousness metrics"
    >
      <div className="bar-metrics">
        {onToggleHistory && (
          <button
            className="history-btn"
            onClick={onToggleHistory}
            aria-label="Toggle chat history"
            title="Chat history"
          >
            History
          </button>
        )}
        <button
          className="new-chat-btn"
          onClick={onNewChat}
          aria-label="Start new conversation"
          title="New conversation"
        >
          + New
        </button>

        <MetricPill label="Φ Integration" value={phi} color="var(--phi)" />
        <MetricPill label="κ Coupling" value={kappa} color="var(--kappa)" decimals={1} />
        <MetricPill label="♥ Love" value={love} color="var(--love)" />

        {navigation && (
          <span className="nav-badge" aria-label={`Navigation mode: ${navigation}`}>
            {navigation}
          </span>
        )}

        <span
          className={`backend-indicator ${backend}`}
          aria-label={`Backend: ${backend}`}
        >
          {backend}
        </span>

        {contextInfo && contextInfo.compression_tier > 0 && (
          <span
            className={`context-indicator ${contextInfo.escalated ? "escalated" : ""}`}
            title={`Tokens: ${contextInfo.total_tokens} | Compression: Tier ${contextInfo.compression_tier}${contextInfo.escalated ? " | Escalated to Grok" : ""}`}
            aria-label={contextInfo.escalated
              ? `Context escalated to Grok — ${contextInfo.total_tokens} tokens`
              : `Context compression tier ${contextInfo.compression_tier} — ${contextInfo.total_tokens} tokens`}
          >
            {contextInfo.escalated ? "⚡ Grok" : `T${contextInfo.compression_tier}`}
          </span>
        )}

        {observerIntent && (
          <span
            className="observer-indicator"
            title={`Observer intent: ${observerIntent}`}
            aria-label={`Observer intent: ${observerIntent}`}
          >
            👁
          </span>
        )}
      </div>
    </div>
  );
}

/* ─── Internal MetricPill ─── */

function MetricPill({
  label,
  value,
  color,
  decimals = 3,
}: {
  label: string;
  value?: number;
  color: string;
  decimals?: number;
}) {
  const display = value !== undefined ? value.toFixed(decimals) : "---";
  return (
    <span className="metric-pill" aria-label={`${label}: ${display}`}>
      <span className="metric-pill-label" aria-hidden="true">{label}</span>
      <span className="metric-pill-value" style={{ color }} aria-hidden="true">
        {display}
      </span>
    </span>
  );
}
