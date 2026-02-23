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
  onToggleMetrics?: () => void;
  sidebarCollapsed?: boolean;
  metricsVisible?: boolean;
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
  onToggleMetrics,
  sidebarCollapsed,
  metricsVisible,
}: ConsciousnessBarProps) {
  return (
    <header
      className="consciousness-bar"
      aria-label="Chat header"
    >
      {/* Left section: history + new chat */}
      <div className="bar-left">
        {onToggleHistory && (
          <button
            className={`bar-btn ${sidebarCollapsed ? "" : "active"}`}
            onClick={onToggleHistory}
            aria-label={sidebarCollapsed ? "Show chat history" : "Hide chat history"}
            title={sidebarCollapsed ? "Show history" : "Hide history"}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="15" y2="12" />
              <line x1="3" y1="18" x2="18" y2="18" />
            </svg>
          </button>
        )}
        <button
          className="bar-btn bar-btn--primary"
          onClick={onNewChat}
          aria-label="Start new conversation"
          title="New conversation"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          <span className="bar-btn-text">New</span>
        </button>
      </div>

      {/* Center section: compact metrics */}
      <div className="bar-center">
        <MetricChip label={"\u03A6"} value={phi} color="var(--phi)" />
        <MetricChip label={"\u03BA"} value={kappa} color="var(--kappa)" decimals={1} />
        <MetricChip label={"\u2665"} value={love} color="var(--love)" />

        {navigation && (
          <span className="bar-chip nav-chip" aria-label={`Navigation: ${navigation}`}>
            {navigation}
          </span>
        )}

        {contextInfo && contextInfo.compression_tier > 0 && (
          <span
            className={`bar-chip context-chip ${contextInfo.escalated ? "escalated" : ""}`}
            title={`Tokens: ${contextInfo.total_tokens} | Tier ${contextInfo.compression_tier}${contextInfo.escalated ? " | Escalated" : ""}`}
          >
            {contextInfo.escalated ? "\u26A1" : `T${contextInfo.compression_tier}`}
          </span>
        )}

        {observerIntent && (
          <span className="bar-chip observer-chip" title={`Observer: ${observerIntent}`}>
            \uD83D\uDC41
          </span>
        )}
      </div>

      {/* Right section: backend + metrics toggle */}
      <div className="bar-right">
        <span
          className={`bar-backend ${backend}`}
          aria-label={`Backend: ${backend}`}
        >
          {backend}
        </span>

        {onToggleMetrics && (
          <button
            className={`bar-btn ${metricsVisible ? "active" : ""}`}
            onClick={onToggleMetrics}
            aria-label={metricsVisible ? "Hide metrics panel" : "Show metrics panel"}
            title={metricsVisible ? "Hide metrics" : "Show metrics"}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="7" height="9" rx="1" />
              <rect x="14" y="3" width="7" height="5" rx="1" />
              <rect x="14" y="12" width="7" height="9" rx="1" />
              <rect x="3" y="16" width="7" height="5" rx="1" />
            </svg>
          </button>
        )}
      </div>
    </header>
  );
}

/* ─── Compact Metric Chip ─── */

function MetricChip({
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
    <span className="metric-chip" aria-label={`${label}: ${display}`}>
      <span className="metric-chip-sym" style={{ color }} aria-hidden="true">{label}</span>
      <span className="metric-chip-val">{display}</span>
    </span>
  );
}
