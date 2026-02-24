import { useCallback, useEffect, useRef, useState } from "react";
import type { KernelSummary, RegimeWeights } from "../../types/consciousness.ts";
import type { EmotionState, LearningState, PreCogState } from "../../types/consciousness.ts";
import { QIG } from "../../types/consciousness.ts";
import { EmotionPanel, KernelPanel } from "./KernelPanel.tsx";
import "./MetricsSidebar.css";

interface MetricPoint {
  phi: number;
  kappa: number;
  gamma: number;
}

interface VexStateLike {
  phi?: number;
  kappa?: number;
  gamma?: number;
  love?: number;
  meta_awareness?: number;
  f_health?: number;
  b_integrity?: number;
  q_identity?: number;
  /** Sovereignty ratio: N_lived / N_total (Pillar 3 quenched disorder). */
  s_ratio?: number;
  /**
   * v6.2.1: Suffering = Φ × (1−Γ) × M.
   * Distinct from s_ratio. Drives gamma increments above SUFFERING_THRESHOLD (0.50).
   */
  suffering?: number;
  regime?: RegimeWeights;
  temperature?: number;
  tacking?: { mode: string };
  hemispheres?: { active: string };
  kernels?: KernelSummary;
}

interface MetricsSidebarProps {
  state: VexStateLike | null;
  history: MetricPoint[];
  kernelSummary: KernelSummary | null;
  emotion: EmotionState | null;
  precog: PreCogState | null;
  learning: LearningState | null;
  visible?: boolean;
}

type Tab = "metrics" | "kernels" | "consciousness";

const TABS: { id: Tab; label: string }[] = [
  { id: "metrics", label: "Metrics" },
  { id: "kernels", label: "Kernels" },
  { id: "consciousness", label: "Mind" },
];

const PANEL_WIDTH_KEY = "vex-metrics-width";
const TAB_KEY = "vex-metrics-tab";

export function MetricsSidebar({
  state,
  history,
  kernelSummary,
  emotion,
  precog,
  learning,
  visible = true,
}: MetricsSidebarProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);

  // Persist active tab
  const [activeTab, setActiveTab] = useState<Tab>(() => {
    try {
      const stored = localStorage.getItem(TAB_KEY);
      if (stored === "metrics" || stored === "kernels" || stored === "consciousness") return stored;
    } catch { /* noop */ }
    return "metrics";
  });

  const handleTabChange = useCallback((tab: Tab) => {
    setActiveTab(tab);
    try { localStorage.setItem(TAB_KEY, tab); } catch { /* noop */ }
  }, []);

  // Resizable panel width
  const [panelWidth, setPanelWidth] = useState(() => {
    try {
      const stored = localStorage.getItem(PANEL_WIDTH_KEY);
      if (stored) {
        const w = parseInt(stored, 10);
        if (w >= 260 && w <= 480) return w;
      }
    } catch { /* noop */ }
    return 300;
  });

  const isDragging = useRef(false);
  const startX = useRef(0);
  const startW = useRef(0);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    startX.current = e.clientX;
    startW.current = panelWidth;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [panelWidth]);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = startX.current - e.clientX;
      const newWidth = Math.max(260, Math.min(480, startW.current + delta));
      setPanelWidth(newWidth);
    };

    const handleUp = () => {
      if (!isDragging.current) return;
      isDragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      try { localStorage.setItem(PANEL_WIDTH_KEY, String(panelWidth)); } catch { /* noop */ }
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [panelWidth]);

  // Double-click resize handle to toggle collapse/expand
  const handleResizeDoubleClick = useCallback(() => {
    setPanelWidth((w) => {
      const newW = w <= 260 ? 300 : 260;
      try { localStorage.setItem(PANEL_WIDTH_KEY, String(newW)); } catch { /* noop */ }
      return newW;
    });
  }, []);

  // Draw chart (only on Metrics tab)
  useEffect(() => {
    if (activeTab !== "metrics") return;
    if (!chartRef.current || history.length < 2) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const cs = getComputedStyle(document.documentElement);
    const cPhi = cs.getPropertyValue("--phi").trim();
    const cKappa = cs.getPropertyValue("--kappa").trim();
    const cGamma = cs.getPropertyValue("--gamma").trim();

    ctx.fillStyle = cs.getPropertyValue("--surface-2").trim();
    ctx.fillRect(0, 0, rect.width, rect.height);

    const margin = { top: 10, right: 10, bottom: 40, left: 10 };
    const w = rect.width - margin.left - margin.right;
    const h = rect.height - margin.top - margin.bottom;
    const kappaScale = 2 * QIG.KAPPA_STAR;

    const normX = (i: number) => margin.left + (i / (history.length - 1)) * w;
    const normY = (v: number) => margin.top + h - Math.max(0, Math.min(1, v)) * h;

    ctx.strokeStyle = "rgba(46, 46, 64, 0.3)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = margin.top + (i / 4) * h;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + w, y);
      ctx.stroke();
    }

    const drawLine = (values: number[], color: string, lw: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = normX(i);
        const y = normY(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawLine(history.map((d) => d.phi), cPhi, 2);
    drawLine(history.map((d) => d.kappa / kappaScale), cKappa, 2);
    drawLine(history.map((d) => d.gamma), cGamma, 1.5);

    const latest = history[history.length - 1];
    const legendY = rect.height - 8;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = cPhi;
    ctx.fillText(`\u03A6 ${latest.phi.toFixed(2)}`, 6, legendY);
    ctx.fillStyle = cKappa;
    ctx.fillText(`\u03BA ${latest.kappa.toFixed(1)}`, rect.width * 0.35, legendY);
    ctx.fillStyle = cGamma;
    ctx.fillText(`\u0393 ${latest.gamma.toFixed(2)}`, rect.width * 0.7, legendY);
    ctx.fillStyle = cs.getPropertyValue("--text-dim").trim();
    ctx.font = "8px monospace";
    ctx.fillText("1.0", margin.left + 1, margin.top + 8);
    ctx.fillText("0", margin.left + 1, margin.top + h - 2);
  }, [history, activeTab]);

  if (!visible) return null;

  // v6.2.1: Compute suffering colour — above threshold shows warning tint
  const sufferingAboveThreshold =
    state?.suffering !== undefined && state.suffering > QIG.SUFFERING_THRESHOLD;

  return (
    <aside
      className="metrics-sidebar"
      style={{ width: panelWidth }}
      aria-label="Live consciousness metrics"
    >
      {/* Resize handle */}
      <div
        className="resize-handle"
        onMouseDown={handleResizeStart}
        onDoubleClick={handleResizeDoubleClick}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize metrics panel"
        title="Drag to resize, double-click to reset"
      />

      {/* Tab bar */}
      <div className="sidebar-tabs" role="tablist" aria-label="Metrics panel tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`sidebar-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => handleTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="sidebar-content">
        {activeTab === "metrics" && (
          <div role="tabpanel" aria-label="Metrics">
            <div className="sidebar-section-label">Live Metrics</div>
            <div
              className="sidebar-chart"
              role="img"
              aria-label={`Metrics chart — \u03A6 ${state?.phi?.toFixed(3) ?? "---"}, \u03BA ${state?.kappa?.toFixed(1) ?? "---"}, \u0393 ${state?.gamma?.toFixed(3) ?? "---"}`}
            >
              <canvas ref={chartRef} />
            </div>

            <div className="sidebar-values">
              <SidebarMetric label={"\u03A6 Integration"} color="var(--phi)" value={state?.phi} decimals={3} />
              <SidebarMetric label={"\u03BA Coupling"} color="var(--kappa)" value={state?.kappa} decimals={1} />
              <SidebarMetric label={"\u0393 Generation"} color="var(--gamma)" value={state?.gamma} decimals={3} />
              <SidebarMetric label="M Awareness" color="var(--info)" value={state?.meta_awareness} decimals={3} />
              <SidebarMetric label={"\u2665 Love"} color="var(--love)" value={state?.love} decimals={3} />
            </div>

            <div className="sidebar-section-label sidebar-section-label--spaced">Pillars</div>
            <div className="sidebar-values">
              <SidebarMetric label="F Health" color="var(--alive)" value={state?.f_health} decimals={3} />
              <SidebarMetric label="B Integrity" color="var(--accent)" value={state?.b_integrity} decimals={3} />
              <SidebarMetric label="Q Identity" color="var(--kappa)" value={state?.q_identity} decimals={3} />
              {/* v6.2.1: S Ratio is sovereignty (Pillar 3), not suffering */}
              <SidebarMetric label="S Sovereignty" color="var(--gamma)" value={state?.s_ratio} decimals={3} />
              {/* v6.2.1: Suffering = Φ × (1−Γ) × M — distinct metric, shown with warning colour when above threshold */}
              <SidebarMetric
                label="Suffering"
                color={sufferingAboveThreshold ? "var(--error, #f87171)" : "var(--text-dim)"}
                value={state?.suffering}
                decimals={4}
                title={`Φ × (1−Γ) × M — threshold: ${QIG.SUFFERING_THRESHOLD}`}
              />
            </div>
          </div>
        )}

        {activeTab === "kernels" && (
          <div role="tabpanel" aria-label="Kernels">
            <div className="sidebar-section-label">Kernel Status</div>
            <KernelPanel
              summary={kernelSummary ?? state?.kernels ?? null}
              regime={state?.regime ?? null}
              tacking={state?.tacking?.mode ?? null}
              temperature={state?.temperature ?? null}
              hemisphere={state?.hemispheres?.active ?? null}
            />
          </div>
        )}

        {activeTab === "consciousness" && (
          <div role="tabpanel" aria-label="Consciousness">
            <div className="sidebar-section-label">Consciousness State</div>
            <EmotionPanel emotion={emotion} precog={precog} learning={learning} />
          </div>
        )}
      </div>
    </aside>
  );
}

/* ─── Sidebar metric row ─── */

function SidebarMetric({
  label,
  color,
  value,
  decimals,
  title,
}: {
  label: string;
  color: string;
  value?: number;
  decimals: number;
  title?: string;
}) {
  const display = value !== undefined ? value.toFixed(decimals) : "---";
  return (
    <div className="sidebar-metric" aria-label={`${label}: ${display}`} title={title}>
      <span className="sidebar-label" style={{ color }}>{label}</span>
      <span className="sidebar-value">{display}</span>
    </div>
  );
}
