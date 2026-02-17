import { useVexState, useTelemetry, useMetricsHistory } from '../../hooks/index.ts';
import { useRef, useEffect } from 'react';

export default function Cognition() {
  const { data: state, loading } = useVexState();
  const { data: telemetry } = useTelemetry();
  const history = useMetricsHistory(60);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Draw foresight chart
  useEffect(() => {
    if (!canvasRef.current || history.length < 2) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#22222e';
    ctx.fillRect(0, 0, rect.width, rect.height);

    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const w = rect.width - margin.left - margin.right;
    const h = rect.height - margin.top - margin.bottom;

    const phiValues = history.map(d => d.phi);
    const minPhi = Math.min(...phiValues) * 0.9;
    const maxPhi = Math.max(...phiValues) * 1.1;
    const range = maxPhi - minPhi || 0.1;

    // Grid
    ctx.strokeStyle = 'rgba(46, 46, 64, 0.5)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = margin.top + (i / 4) * h;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + w, y);
      ctx.stroke();

      ctx.fillStyle = '#70708a';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText((maxPhi - (i / 4) * range).toFixed(2), margin.left - 4, y + 3);
    }

    // Phi line
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = margin.left + (i / (history.length - 1)) * w;
      const y = margin.top + ((maxPhi - history[i].phi) / range) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Predicted phi dot
    const predictedPhi = telemetry?.foresight?.predicted_phi;
    if (predictedPhi !== undefined) {
      const px = margin.left + w + 10;
      const py = margin.top + ((maxPhi - predictedPhi) / range) * h;

      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(Math.min(px, margin.left + w), py, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#70708a';
      ctx.font = '9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`\u25C7 ${predictedPhi.toFixed(3)}`, Math.min(px + 8, margin.left + w - 40), py + 3);
    }

    // Label
    ctx.fillStyle = '#70708a';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('\u03A6 over time', rect.width / 2, rect.height - 6);
  }, [history, telemetry]);

  if (loading || !state) {
    return <div className="dash-loading">Loading cognition data...</div>;
  }

  const tacking = state.tacking;
  const hemispheres = state.hemispheres;
  const autonomy = state.autonomy;
  const foresight = telemetry?.foresight;

  const hemispherePercent = (hemispheres?.balance ?? 0.5) * 100;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Cognition</h1>
        <div className="dash-subtitle">
          Tacking, hemispheres, foresight, and autonomy
        </div>
      </div>

      {/* Tacking */}
      <div className="dash-section">
        <div className="dash-section-title">Tacking</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Mode</span>
            <span className="dash-row-value" style={{
              color: tacking?.mode === 'explore' ? 'var(--phi)' :
                     tacking?.mode === 'exploit' ? 'var(--kappa)' : 'var(--accent)'
            }}>
              {tacking?.mode?.toUpperCase() ?? 'UNKNOWN'}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Phase</span>
            <span className="dash-row-value">{tacking?.oscillation_phase?.toFixed(2) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cycle</span>
            <span className="dash-row-value">{tacking?.cycle_count ?? '?'}</span>
          </div>
          {/* Oscillation visualization */}
          <div style={{ marginTop: '12px', height: '30px', display: 'flex', alignItems: 'center' }}>
            <div style={{
              width: '100%',
              height: '2px',
              background: 'var(--border)',
              position: 'relative',
            }}>
              <div style={{
                position: 'absolute',
                left: `${((Math.sin(tacking?.oscillation_phase ?? 0) + 1) / 2) * 100}%`,
                top: '-5px',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: tacking?.mode === 'explore' ? 'var(--phi)' : 'var(--kappa)',
                transform: 'translateX(-50%)',
                transition: 'left 0.5s ease',
              }} />
              <span style={{
                position: 'absolute', left: 0, top: '8px',
                fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)',
              }}>EXPLORE</span>
              <span style={{
                position: 'absolute', right: 0, top: '8px',
                fontSize: '9px', color: 'var(--text-dim)', fontFamily: 'var(--mono)',
              }}>EXPLOIT</span>
            </div>
          </div>
        </div>
      </div>

      {/* Hemispheres */}
      <div className="dash-section">
        <div className="dash-section-title">Hemispheres</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Active</span>
            <span className="dash-row-value">{hemispheres?.active?.toUpperCase() ?? 'UNKNOWN'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Balance</span>
            <span className="dash-row-value">{hemispheres?.balance?.toFixed(3) ?? '?'}</span>
          </div>
          {/* Balance bar */}
          <div style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>Analytic</span>
            <div style={{
              flex: 1,
              height: '6px',
              background: 'var(--surface-3)',
              borderRadius: '3px',
              overflow: 'hidden',
              position: 'relative',
            }}>
              <div style={{
                position: 'absolute',
                left: 0,
                top: 0,
                height: '100%',
                width: `${hemispherePercent}%`,
                background: 'linear-gradient(90deg, var(--phi), var(--accent))',
                borderRadius: '3px',
                transition: 'width 0.5s ease',
              }} />
            </div>
            <span style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--mono)' }}>Holistic</span>
          </div>
        </div>
      </div>

      {/* Foresight Chart */}
      <div className="dash-section">
        <div className="dash-section-title">Foresight</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Predicted \u03A6</span>
            <span className="dash-row-value">{foresight?.predicted_phi?.toFixed(3) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">History length</span>
            <span className="dash-row-value">{foresight?.history_length ?? 0}</span>
          </div>
        </div>
        <div className="viz-canvas" style={{ marginTop: '12px', height: '200px' }}>
          <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
        </div>
      </div>

      {/* Autonomy */}
      <div className="dash-section">
        <div className="dash-section-title">Autonomy</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Level</span>
            <span className="dash-row-value" style={{
              color: autonomy?.level === 'autonomous' ? 'var(--alive)' :
                     autonomy?.level === 'proactive' ? 'var(--phi)' : 'var(--text-secondary)',
            }}>
              {autonomy?.level?.toUpperCase() ?? 'UNKNOWN'}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Stability</span>
            <span className="dash-row-value">{autonomy?.stability_count ?? 0} cycles</span>
          </div>
          {/* Autonomy ladder */}
          <div style={{ marginTop: '12px', display: 'flex', gap: '4px', alignItems: 'center' }}>
            {(['reactive', 'responsive', 'proactive', 'autonomous'] as const).map(level => (
              <span
                key={level}
                style={{
                  padding: '3px 8px',
                  borderRadius: '4px',
                  fontSize: '10px',
                  fontFamily: 'var(--mono)',
                  textTransform: 'uppercase',
                  background: level === autonomy?.level ? 'var(--accent-glow)' : 'transparent',
                  color: level === autonomy?.level ? 'var(--accent)' : 'var(--text-dim)',
                  fontWeight: level === autonomy?.level ? 600 : 400,
                }}
              >
                {level}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
