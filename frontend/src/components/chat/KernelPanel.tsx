/**
 * Kernel, Emotion, and Regime panel components extracted from MetricsSidebar
 * to keep that file under 300 lines.
 */
import type {
  EmotionState,
  KernelSummary,
  LearningState,
  PreCogState,
  RegimeWeights,
} from "../../types/consciousness.ts";
import { EMOTION_COLORS } from "./chatUtils.ts";
import { RegimeBar } from "./RegimeBar.tsx";
// Styles are owned by MetricsSidebar.css (imported by MetricsSidebar.tsx)

/* ─── Emotion / PreCog / Learning panel ─── */

export function EmotionPanel({
  emotion,
  precog,
  learning,
}: {
  emotion: EmotionState | null;
  precog: PreCogState | null;
  learning: LearningState | null;
}) {
  const emotionName = emotion?.current_emotion ?? "none";
  const emotionColor = EMOTION_COLORS[emotionName] ?? "var(--text-dim)";
  const strength = emotion?.current_strength ?? 0;

  return (
    <div className="kernel-panel">
      <KernelStateRow label="Emotion">
        <span style={{ color: emotionColor }}>
          {emotionName}{strength > 0 ? ` (${(strength * 100).toFixed(0)}%)` : ""}
        </span>
      </KernelStateRow>

      {precog && (
        <>
          <KernelStateRow label="Path">{precog.last_path.replace("_", "-")}</KernelStateRow>
          <KernelStateRow label="Pre-cog %">{(precog.a_pre * 100).toFixed(1)}%</KernelStateRow>
        </>
      )}

      {learning && (
        <>
          <KernelStateRow label="Patterns">{learning.patterns_found}</KernelStateRow>
          <KernelStateRow label="\u03A6 gain">
            <span style={{ color: learning.total_phi_gain >= 0 ? "var(--alive)" : "var(--error)" }}>
              {learning.total_phi_gain >= 0 ? "+" : ""}
              {learning.total_phi_gain.toFixed(4)}
            </span>
          </KernelStateRow>
        </>
      )}
    </div>
  );
}

/* ─── Kernel balance panel ─── */

export function KernelPanel({
  summary,
  regime,
  tacking,
  temperature,
  hemisphere,
}: {
  summary: KernelSummary | null;
  regime: RegimeWeights | null;
  tacking: string | null;
  temperature: number | null;
  hemisphere: string | null;
}) {
  if (!summary) return <div className="sidebar-placeholder">---</div>;

  const byKind = summary.by_kind ?? {};
  const genesis = byKind.GENESIS ?? 0;
  const god = byKind.GOD ?? 0;
  const chaos = byKind.CHAOS ?? 0;
  const budget = summary.budget;

  return (
    <div className="kernel-panel">
      <div className="kernel-counts">
        <KernelKindRow dotClass="genesis" label="Genesis" count={genesis} />
        <KernelKindRow dotClass="god" label="God" count={god} max={budget?.god_max ?? 248} />
        <KernelKindRow dotClass="chaos" label="Chaos" count={chaos} max={budget?.chaos_max ?? 200} />
      </div>

      <div className="sidebar-sub-header">Regime</div>
      <RegimeBar regime={regime} />
      {regime && (
        <div className="regime-labels">
          <span style={{ color: "var(--regime-quantum)" }}>Q {Math.round(regime.quantum * 100)}%</span>
          <span style={{ color: "var(--regime-efficient)" }}>E {Math.round(regime.efficient * 100)}%</span>
          <span style={{ color: "var(--regime-equilibrium)" }}>Eq {Math.round(regime.equilibrium * 100)}%</span>
        </div>
      )}

      <KernelStateRow label="Tack">{tacking ?? "---"}</KernelStateRow>
      <KernelStateRow label="Temp">{temperature?.toFixed(3) ?? "---"}</KernelStateRow>
      <KernelStateRow label="Hemi">{hemisphere ?? "---"}</KernelStateRow>
    </div>
  );
}

/* ─── Shared sub-components ─── */

export function KernelStateRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="kernel-state-row">
      <span className="kernel-state-label">{label}</span>
      <span className="kernel-state-value">{children}</span>
    </div>
  );
}

function KernelKindRow({
  dotClass,
  label,
  count,
  max,
}: {
  dotClass: string;
  label: string;
  count: number;
  max?: number;
}) {
  return (
    <div
      className="kernel-kind"
      aria-label={`${label}: ${count}${max !== undefined ? ` of ${max}` : ""}`}
    >
      <span className={`kernel-dot ${dotClass}`} aria-hidden="true" />
      <span className="kernel-kind-label">{label}</span>
      <span className="kernel-kind-value">
        {count}
        {max !== undefined && (
          <span className="kernel-budget" aria-hidden="true">/{max}</span>
        )}
      </span>
    </div>
  );
}
