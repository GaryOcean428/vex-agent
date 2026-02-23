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
  sidebarCollapsed?: boolean;
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
  sidebarCollapsed,
}: ConsciousnessBarProps) {
  return (
    <div
      className="consciousness-bar"
      aria-label="Consciousness metrics"
    >
      <div className="bar-metrics">
        {onToggleHistory && (
          <button
            className={`history-toggle-btn ${sidebarCollapsed ? "" : "active"}`}
            onClick={onToggleHistory}
            aria-label={sidebarCollapsed ? "Show chat history" : "Hide chat history"}
            title={sidebarCollapsed ? "Show history" : "Hide history"}
          >
            <span className="history-toggle-icon" aria-hidden="true">
              {sidebarCollapsed ? "\u25B6" : "\u25C0"}
            </span>
            <span className="history-toggle-label">History</span>
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

        <MetricPill label="\u03A6 Integration" value={phi} color="var(--phi)" />
        <MetricPill label="\u03BA Coupling" value={kappa} color="var(--kappa)" decimals={1} />
        <MetricPill label="\u2665 Love" value={love} color="var(--love)" />

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
              ? `Context escalated to Grok \u2014 ${contextInfo.total_tokens} tokens`
              : `Context compression tier ${contextInfo.compression_tier} \u2014 ${contextInfo.total_tokens} tokens`}
          >
            {contextInfo.escalated ? "\u26A1 Grok" : `T${contextInfo.compression_tier}`}
          </span>
        )}

        {observerIntent && (
          <span
            className="observer-indicator"
            title={`Observer intent: ${observerIntent}`}
            aria-label={`Observer intent: ${observerIntent}`}
          >
            \uD83D\uDC41
          </span>
        )}
      </div>
    </div>
  );
}

/* \u2500\u2500\u2500 Internal MetricPill \u2500\u2500\u2500 */

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
