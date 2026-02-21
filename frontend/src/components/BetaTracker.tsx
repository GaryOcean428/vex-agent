import { useEffect, useRef } from 'react';
import MetricCard from './MetricCard.tsx';
import type { BetaTrackerSummary } from '../types/consciousness.ts';

interface Props {
  data: BetaTrackerSummary;
}

/** Resolve a CSS custom property from :root. */
function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Set up a canvas for high-DPI rendering. Returns drawing dimensions. */
function setupCanvas(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return null;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
}

// ═══════════════════════════════════════════════════════
//  β-Trajectory Chart
// ═══════════════════════════════════════════════════════

function drawBetaTrajectory(canvas: HTMLCanvasElement, data: BetaTrackerSummary) {
  const setup = setupCanvas(canvas);
  if (!setup) return;
  const { ctx, w, h } = setup;

  const cAlive = cssVar('--alive');
  const cError = cssVar('--error');
  const cKappa = cssVar('--kappa');
  const cDim = cssVar('--text-dim');
  const cSurface = cssVar('--surface-2');
  const cGrid = 'rgba(46, 46, 64, 0.3)';

  // Background
  ctx.fillStyle = cSurface;
  ctx.fillRect(0, 0, w, h);

  const margin = { top: 16, right: 16, bottom: 36, left: 50 };
  const cw = w - margin.left - margin.right;
  const ch = h - margin.top - margin.bottom;

  const traj = data.trajectory;
  if (traj.length === 0) {
    ctx.fillStyle = cDim;
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting sufficient bin data for β trajectory...', w / 2, h / 2);
    return;
  }

  // Y range: show -0.2 to max(0.7, max_beta + 0.1)
  const betas = traj.map((p) => p.beta);
  const betaErrors = traj.map((p) => p.beta_error);
  const maxBeta = Math.max(0.7, ...betas.map((b, i) => b + betaErrors[i]) , 0.443 + 0.2);
  const minBeta = Math.min(-0.2, ...betas.map((b, i) => b - betaErrors[i]));
  const yRange = maxBeta - minBeta;

  const toX = (i: number) => margin.left + ((i + 0.5) / traj.length) * cw;
  const toY = (beta: number) => margin.top + ch - ((beta - minBeta) / yRange) * ch;

  // Grid lines
  ctx.strokeStyle = cGrid;
  ctx.lineWidth = 0.5;
  const gridStep = yRange > 1.5 ? 0.5 : 0.25;
  const gridStart = Math.ceil(minBeta / gridStep) * gridStep;
  for (let v = gridStart; v <= maxBeta; v += gridStep) {
    const y = toY(v);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + cw, y);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = cDim;
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), margin.left - 4, y + 3);
  }

  // Reference line: β_physics = 0.443 (emergence)
  const yPhysics = toY(0.443);
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = cKappa;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, yPhysics);
  ctx.lineTo(margin.left + cw, yPhysics);
  ctx.stroke();

  // Acceptance band around physics reference (±threshold)
  const threshold = data.acceptance_threshold;
  ctx.fillStyle = `${cKappa}15`;
  ctx.fillRect(
    margin.left,
    toY(0.443 + threshold),
    cw,
    toY(0.443 - threshold) - toY(0.443 + threshold),
  );

  // Reference line: β = 0 (asymptotic freedom)
  const yZero = toY(0);
  ctx.strokeStyle = cDim;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, yZero);
  ctx.lineTo(margin.left + cw, yZero);
  ctx.stroke();

  // Acceptance band around 0 (plateau)
  ctx.fillStyle = `${cDim}10`;
  ctx.fillRect(
    margin.left,
    toY(threshold),
    cw,
    toY(-threshold) - toY(threshold),
  );

  ctx.setLineDash([]);

  // Data points with error bars
  for (let i = 0; i < traj.length; i++) {
    const pt = traj[i];
    const x = toX(i);
    const y = toY(pt.beta);
    const color = pt.within_acceptance ? cAlive : cError;

    // Error bar
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, toY(pt.beta + pt.beta_error));
    ctx.lineTo(x, toY(pt.beta - pt.beta_error));
    ctx.stroke();

    // Error bar caps
    ctx.beginPath();
    ctx.moveTo(x - 3, toY(pt.beta + pt.beta_error));
    ctx.lineTo(x + 3, toY(pt.beta + pt.beta_error));
    ctx.moveTo(x - 3, toY(pt.beta - pt.beta_error));
    ctx.lineTo(x + 3, toY(pt.beta - pt.beta_error));
    ctx.stroke();

    // Point
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();

    // X-axis label
    ctx.fillStyle = cDim;
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${pt.from}→`, x, h - margin.bottom + 12);
    ctx.fillText(pt.to, x, h - margin.bottom + 22);
  }

  // Legend
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = cKappa;
  ctx.fillText('β_physics = 0.443', margin.left + 4, margin.top + 10);
  ctx.fillStyle = cDim;
  ctx.fillText(`threshold = ±${threshold}`, margin.left + cw - 100, margin.top + 10);
}

// ═══════════════════════════════════════════════════════
//  κ Running Coupling Chart
// ═══════════════════════════════════════════════════════

function drawKappaChart(canvas: HTMLCanvasElement, data: BetaTrackerSummary) {
  const setup = setupCanvas(canvas);
  if (!setup) return;
  const { ctx, w, h } = setup;

  const cAccent = cssVar('--accent');
  const cKappa = cssVar('--kappa');
  const cDim = cssVar('--text-dim');
  const cSurface = cssVar('--surface-2');
  const cGrid = 'rgba(46, 46, 64, 0.3)';

  ctx.fillStyle = cSurface;
  ctx.fillRect(0, 0, w, h);

  const margin = { top: 16, right: 16, bottom: 32, left: 50 };
  const cw = w - margin.left - margin.right;
  const ch = h - margin.top - margin.bottom;

  const bins = data.bins.filter((b) => b.count > 0);
  if (bins.length === 0) {
    ctx.fillStyle = cDim;
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No bin data yet...', w / 2, h / 2);
    return;
  }

  // X: log scale of bin centers
  const centers = bins.map((b) => Math.log(b.center));
  const xMin = centers[0] - 0.3;
  const xMax = centers[centers.length - 1] + 0.3;
  const xRange = xMax - xMin;

  // Y: κ range
  const kStar = data.kappa_star_reference;
  const kappas = bins.map((b) => b.kappa_mean);
  const sems = bins.map((b) => b.kappa_sem);
  const yMax = Math.max(kStar * 1.8, ...kappas.map((k, i) => k + sems[i] * 2));
  const yMin = Math.min(0, ...kappas.map((k, i) => k - sems[i] * 2));
  const yRange = yMax - yMin;

  const toX = (logCenter: number) => margin.left + ((logCenter - xMin) / xRange) * cw;
  const toY = (kappa: number) => margin.top + ch - ((kappa - yMin) / yRange) * ch;

  // Grid
  ctx.strokeStyle = cGrid;
  ctx.lineWidth = 0.5;
  const kStep = yRange > 100 ? 20 : 10;
  for (let v = Math.ceil(yMin / kStep) * kStep; v <= yMax; v += kStep) {
    const y = toY(v);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + cw, y);
    ctx.stroke();

    ctx.fillStyle = cDim;
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(0), margin.left - 4, y + 3);
  }

  // Reference line: κ* = 64
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = cKappa;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, toY(kStar));
  ctx.lineTo(margin.left + cw, toY(kStar));
  ctx.stroke();
  ctx.setLineDash([]);

  // Connect points with line
  if (bins.length > 1) {
    ctx.strokeStyle = `${cAccent}60`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < bins.length; i++) {
      const x = toX(centers[i]);
      const y = toY(bins[i].kappa_mean);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Data points with error bars
  for (let i = 0; i < bins.length; i++) {
    const b = bins[i];
    const x = toX(centers[i]);
    const y = toY(b.kappa_mean);
    const color = b.sufficient ? cAccent : cDim;

    // Error bar (±1 SEM)
    if (b.kappa_sem > 0) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, toY(b.kappa_mean + b.kappa_sem));
      ctx.lineTo(x, toY(b.kappa_mean - b.kappa_sem));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 3, toY(b.kappa_mean + b.kappa_sem));
      ctx.lineTo(x + 3, toY(b.kappa_mean + b.kappa_sem));
      ctx.moveTo(x - 3, toY(b.kappa_mean - b.kappa_sem));
      ctx.lineTo(x + 3, toY(b.kappa_mean - b.kappa_sem));
      ctx.stroke();
    }

    // Point
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();

    // X label (bin range)
    ctx.fillStyle = cDim;
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(b.bin, x, h - margin.bottom + 12);
  }

  // Legend
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = cKappa;
  ctx.fillText(`κ* = ${kStar}`, margin.left + 4, margin.top + 10);
  ctx.fillStyle = cAccent;
  ctx.fillText('● sufficient', margin.left + cw - 100, margin.top + 10);
}

// ═══════════════════════════════════════════════════════
//  Component
// ═══════════════════════════════════════════════════════

const VERDICT_COLORS: Record<string, string> = {
  SUBSTRATE_INDEPENDENCE_CONFIRMED: 'var(--alive)',
  PARTIAL_MATCH: 'var(--warning)',
  MISMATCH: 'var(--error)',
  INSUFFICIENT_DATA: 'var(--text-dim)',
};

const VERDICT_LABELS: Record<string, string> = {
  SUBSTRATE_INDEPENDENCE_CONFIRMED: 'Confirmed',
  PARTIAL_MATCH: 'Partial',
  MISMATCH: 'Mismatch',
  INSUFFICIENT_DATA: 'Collecting',
};

export default function BetaTracker({ data }: Props) {
  const betaChartRef = useRef<HTMLCanvasElement>(null);
  const kappaChartRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (betaChartRef.current) drawBetaTrajectory(betaChartRef.current, data);
    if (kappaChartRef.current) drawKappaChart(kappaChartRef.current, data);
  }, [data]);

  const verdictColor = VERDICT_COLORS[data.verdict] ?? 'var(--text-dim)';
  const verdictLabel = VERDICT_LABELS[data.verdict] ?? data.verdict;

  if (data.total_recorded === 0) {
    return (
      <div className="dash-section">
        <div className="dash-section-title">β-Attention Tracker <span style={{ opacity: 0.5 }}>(v6.1)</span></div>
        <div className="dash-empty">
          No β-attention measurements yet. Measurements accumulate from real conversations.
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Summary Cards */}
      <div className="dash-section">
        <div className="dash-section-title">β-Attention Tracker <span style={{ opacity: 0.5 }}>(v6.1)</span></div>
        <div className="dash-grid">
          <MetricCard
            label="Measurements"
            value={data.total_recorded}
            color="var(--accent)"
          />
          <MetricCard
            label="Bins"
            value={`${data.sufficient_bins}/${data.active_bins}`}
            color="var(--phi)"
            subtitle={`${data.min_per_bin}+ per bin needed`}
            progress={data.active_bins > 0 ? data.sufficient_bins / data.active_bins : 0}
          />
          <MetricCard
            label="Verdict"
            value={verdictLabel}
            color={verdictColor}
          />
          <MetricCard
            label="Uptime"
            value={`${data.uptime_hours}h`}
            color="var(--text-secondary)"
          />
        </div>
      </div>

      {/* β-Function Trajectory Chart */}
      <div className="dash-section">
        <div className="dash-section-title">β-Function Trajectory</div>
        <div className="viz-canvas">
          <canvas ref={betaChartRef} />
        </div>
      </div>

      {/* κ Running Coupling Chart */}
      <div className="dash-section">
        <div className="dash-section-title">κ Running Coupling</div>
        <div className="viz-canvas">
          <canvas ref={kappaChartRef} />
        </div>
      </div>

      {/* Bin Details */}
      {data.bins.length > 0 && (
        <div className="dash-section">
          <div className="dash-section-title">Bin Statistics</div>
          <div className="dash-card">
            {/* Header row */}
            <div className="dash-row" style={{ borderBottom: '1px solid var(--border)', paddingBottom: '8px', marginBottom: '4px' }}>
              <span className="dash-row-label" style={{ flex: 1 }}>Bin</span>
              <span className="dash-row-value" style={{ flex: 1, textAlign: 'right' }}>n</span>
              <span className="dash-row-value" style={{ flex: 2, textAlign: 'right' }}>κ mean ± sem</span>
              <span className="dash-row-value" style={{ flex: 1, textAlign: 'right' }}>d̄</span>
              <span className="dash-row-value" style={{ flex: 1, textAlign: 'right' }}>ΔΦ̄</span>
            </div>
            {data.bins.map((b) => (
              <div className="dash-row" key={b.bin} style={{ opacity: b.sufficient ? 1 : 0.5 }}>
                <span className="dash-row-label" style={{ flex: 1 }}>{b.bin}</span>
                <span className="dash-row-value" style={{ flex: 1, textAlign: 'right' }}>{b.count}</span>
                <span className="dash-row-value" style={{ flex: 2, textAlign: 'right', color: b.sufficient ? 'var(--accent)' : 'var(--text-dim)' }}>
                  {b.kappa_mean.toFixed(2)} ± {b.kappa_sem.toFixed(3)}
                </span>
                <span className="dash-row-value" style={{ flex: 1, textAlign: 'right' }}>{b.distance_mean.toFixed(4)}</span>
                <span className="dash-row-value" style={{ flex: 1, textAlign: 'right' }}>{b.phi_gain_mean.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Substrate Match */}
      {data.substrate_match && (
        <div className="dash-section">
          <div className="dash-section-title">Substrate Independence Assessment</div>
          <div className="dash-card">
            <div className="dash-row">
              <span className="dash-row-label">Verdict</span>
              <span className="dash-row-value" style={{ color: verdictColor, fontWeight: 700 }}>
                {data.verdict.replace(/_/g, ' ')}
              </span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">β mean (attention)</span>
              <span className="dash-row-value">{data.substrate_match.beta_mean.toFixed(4)}</span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">β physics (TFIM)</span>
              <span className="dash-row-value">{data.substrate_match.beta_physics.toFixed(4)}</span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Deviation</span>
              <span className="dash-row-value">
                {Math.abs(data.substrate_match.beta_mean - data.substrate_match.beta_physics).toFixed(4)}
              </span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Passing</span>
              <span className="dash-row-value" style={{
                color: data.substrate_match.all_within_threshold ? 'var(--alive)' : 'var(--warning)',
              }}>
                {(data.substrate_match.fraction_passing * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
