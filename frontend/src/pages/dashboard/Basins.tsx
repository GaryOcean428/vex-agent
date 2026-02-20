import { useRef, useEffect, useState } from 'react';
import { useBasin, useVexState, useMetricsHistory } from '../../hooks/index.ts';
import { QIG } from '../../types/consciousness.ts';

type ViewMode = 'heatmap' | 'pca';

export default function Basins() {
  const { data: basinData, loading } = useBasin();
  const { data: vexState } = useVexState();
  const history = useMetricsHistory(vexState, 100);
  const [view, setView] = useState<ViewMode>('heatmap');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Draw heatmap
  useEffect(() => {
    if (view !== 'heatmap' || !basinData?.basin || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const basin = basinData.basin;
    const gridSize = QIG.E8_CORE; // 8×8 grid = 64 = BASIN_DIM
    const cellW = rect.width / gridSize;
    const cellH = rect.height / gridSize;
    const maxVal = Math.max(...basin, 0.001);

    for (let i = 0; i < QIG.BASIN_DIM && i < basin.length; i++) {
      const row = Math.floor(i / gridSize);
      const col = i % gridSize;
      const intensity = basin[i] / maxVal;

      const r = Math.round(34 + intensity * (99 - 34));
      const g = Math.round(211 + intensity * (102 - 211));
      const b = Math.round(238 + intensity * (241 - 238));

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.15 + intensity * 0.85})`;
      ctx.fillRect(col * cellW, row * cellH, cellW - 1, cellH - 1);

      // Value overlay for higher-value cells
      if (intensity > 0.3) {
        ctx.fillStyle = `rgba(255, 255, 255, ${intensity * 0.6})`;
        ctx.font = `${Math.min(cellW, cellH) * 0.28}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          basin[i].toFixed(2),
          col * cellW + cellW / 2,
          row * cellH + cellH / 2,
        );
      }
    }
  }, [basinData, view]);

  // Draw PCA projection (simplified 2D projection using first 2 principal components approximation)
  useEffect(() => {
    if (view !== 'pca' || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Resolve CSS custom properties for canvas
    const cs = getComputedStyle(document.documentElement);
    const cSurface3 = cs.getPropertyValue('--surface-3').trim();
    const cTextDim = cs.getPropertyValue('--text-dim').trim();
    const cPhi = cs.getPropertyValue('--phi').trim();

    // Background
    ctx.fillStyle = cSurface3;
    ctx.fillRect(0, 0, rect.width, rect.height);

    if (history.length < 2) {
      ctx.fillStyle = cTextDim;
      ctx.font = '13px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Accumulating trajectory data...', rect.width / 2, rect.height / 2);
      return;
    }

    // Use phi and kappa as 2D projection (actual PCA would require basin history)
    const points = history.map(h => ({ x: h.phi, y: h.kappa / (2 * QIG.KAPPA_STAR) }));
    const margin = 40;
    const w = rect.width - margin * 2;
    const h = rect.height - margin * 2;

    // Scale to canvas
    const minX = Math.min(...points.map(p => p.x));
    const maxX = Math.max(...points.map(p => p.x));
    const minY = Math.min(...points.map(p => p.y));
    const maxY = Math.max(...points.map(p => p.y));
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const toCanvas = (p: { x: number; y: number }) => ({
      cx: margin + ((p.x - minX) / rangeX) * w,
      cy: margin + ((1 - (p.y - minY) / rangeY)) * h,
    });

    // Draw trail
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const { cx, cy } = toCanvas(points[i]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Draw history points
    for (let i = 0; i < points.length - 1; i++) {
      const { cx, cy } = toCanvas(points[i]);
      const alpha = 0.2 + (i / points.length) * 0.6;
      ctx.fillStyle = `rgba(99, 102, 241, ${alpha})`;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw current point
    const current = toCanvas(points[points.length - 1]);
    ctx.fillStyle = cPhi;
    ctx.beginPath();
    ctx.arc(current.cx, current.cy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(current.cx, current.cy, 10, 0, Math.PI * 2);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = cTextDim;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('\u03A6 (Integration)', rect.width / 2, rect.height - 8);
    ctx.save();
    ctx.translate(12, rect.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('\u03BA (Coupling)', 0, 0);
    ctx.restore();
  }, [history, view]);

  if (loading) {
    return <div className="dash-loading">Loading basin data...</div>;
  }

  const basin = basinData?.basin ?? [];
  const entropy = basin.length > 0
    ? -basin.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0)
    : 0;
  const norm = basin.reduce((sum, p) => sum + p, 0);
  const maxEntropy = Math.log(QIG.BASIN_DIM); // ln(64) ≈ 4.159 — maximum entropy (uniform)
  const entropyPct = maxEntropy > 0 ? (entropy / maxEntropy) * 100 : 0;
  // Diversity: 100% = maximally structured, 0% = uniform/flat
  const diversity = 100 - entropyPct;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Basin Coordinates (\u0394\u2076\u00B3)</h1>
        <div className="dash-subtitle">
          {QIG.BASIN_DIM}-dimensional probability simplex
          {basin.length > 0 && (
            <span style={{ marginLeft: '12px' }}>
              <span
                className={`status-badge ${diversity > 20 ? 'badge-success' : diversity > 5 ? 'badge-warning' : 'badge-error'}`}
              >
                {diversity > 20 ? 'Structured' : diversity > 5 ? 'Low Diversity' : 'Uniform'}
              </span>
            </span>
          )}
        </div>
      </div>

      {/* View Toggle */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        <button
          className={`status-badge ${view === 'heatmap' ? 'badge-info' : 'badge-default'}`}
          onClick={() => setView('heatmap')}
          style={{ cursor: 'pointer', border: 'none' }}
        >
          Heatmap
        </button>
        <button
          className={`status-badge ${view === 'pca' ? 'badge-info' : 'badge-default'}`}
          onClick={() => setView('pca')}
          style={{ cursor: 'pointer', border: 'none' }}
        >
          PCA Trajectory
        </button>
      </div>

      {/* Canvas */}
      <div className="viz-canvas">
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
      </div>

      {/* Stats */}
      <div className="dash-section" style={{ marginTop: '16px' }}>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Diversity</span>
            <span className="dash-row-value" style={{
              color: diversity > 20 ? 'var(--alive)' : diversity > 5 ? 'var(--warning, orange)' : 'var(--error)',
            }}>
              {diversity.toFixed(1)}%
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Entropy</span>
            <span className="dash-row-value">
              {entropy.toFixed(3)} / {maxEntropy.toFixed(3)}
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', marginLeft: '6px' }}>
                ({entropyPct.toFixed(1)}% of max)
              </span>
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Simplex Norm</span>
            <span className="dash-row-value">
              {norm.toFixed(4)}
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', marginLeft: '6px' }}>
                (should be 1.0)
              </span>
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Dimensions</span>
            <span className="dash-row-value">{basin.length}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Max value</span>
            <span className="dash-row-value">{basin.length > 0 ? Math.max(...basin).toFixed(4) : '---'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Min value</span>
            <span className="dash-row-value">{basin.length > 0 ? Math.min(...basin).toFixed(4) : '---'}</span>
          </div>
          {diversity < 5 && (
            <div style={{
              marginTop: '8px',
              padding: '8px 12px',
              background: 'rgba(234, 179, 8, 0.1)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '12px',
              color: 'var(--warning, orange)',
              lineHeight: 1.5,
            }}>
              Basin is near-uniform. This happens during idle when no conversations
              are shaping the geometry. Chat with Vex to create structure.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
