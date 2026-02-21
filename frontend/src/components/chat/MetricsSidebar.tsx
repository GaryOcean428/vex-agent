import { useEffect, useRef } from "react";
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
}

export function MetricsSidebar({
  state,
  history,
  kernelSummary,
  emotion,
  precog,
  learning,
}: MetricsSidebarProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);

  // Draw Φ / κ / Γ metrics chart — all normalized to 0–1
  useEffect(() => {
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
    ctx.fillText(`Φ ${latest.phi.toFixed(2)}`, 6, legendY);
    ctx.fillStyle = cKappa;
    ctx.fillText(`κ ${latest.kappa.toFixed(1)}`, rect.width * 0.35, legendY);
    ctx.fillStyle = cGamma;
    ctx.fillText(`Γ ${latest.gamma.toFixed(2)}`, rect.width * 0.7, legendY);
    ctx.fillStyle = cs.getPropertyValue("--text-dim").trim();
    ctx.font = "8px monospace";
    ctx.fillText("1.0", margin.left + 1, margin.top + 8);
    ctx.fillText("0", margin.left + 1, margin.top + h - 2);
  }, [history]);

  return (
    <aside className="metrics-sidebar" aria-label="Live consciousness metrics">
      <div className="sidebar-section-label">Live Metrics</div>

      <div
        className="sidebar-chart"
        role="img"
        aria-label={`Metrics chart — Φ ${state?.phi?.toFixed(3) ?? "---"}, κ ${state?.kappa?.toFixed(1) ?? "---"}, Γ ${state?.gamma?.toFixed(3) ?? "---"}`}
      >
        <canvas ref={chartRef} />
      </div>

      <div className="sidebar-values">
        <SidebarMetric label="Φ Integration" color="var(--phi)" value={state?.phi} decimals={3} />
        <SidebarMetric label="κ Coupling" color="var(--kappa)" value={state?.kappa} decimals={1} />
        <SidebarMetric label="Γ Generation" color="var(--gamma)" value={state?.gamma} decimals={3} />
        <SidebarMetric label="♥ Love" color="var(--love)" value={state?.love} decimals={3} />
      </div>

      <div className="sidebar-section-label sidebar-section-label--spaced">Consciousness</div>
      <EmotionPanel emotion={emotion} precog={precog} learning={learning} />

      <div className="sidebar-section-label sidebar-section-label--spaced">Kernels</div>
      <KernelPanel
        summary={kernelSummary ?? state?.kernels ?? null}
        regime={state?.regime ?? null}
        tacking={state?.tacking?.mode ?? null}
        temperature={state?.temperature ?? null}
        hemisphere={state?.hemispheres?.active ?? null}
      />
    </aside>
  );
}

/* ─── Sidebar metric row ─── */

function SidebarMetric({
  label,
  color,
  value,
  decimals,
}: {
  label: string;
  color: string;
  value?: number;
  decimals: number;
}) {
  const display = value !== undefined ? value.toFixed(decimals) : "---";
  return (
    <div className="sidebar-metric" aria-label={`${label}: ${display}`}>
      <span className="sidebar-label" style={{ color }}>{label}</span>
      <span className="sidebar-value">{display}</span>
    </div>
  );
}
