import { useVexState } from '../../hooks/index.ts';
import '../../components/StatusBadge.css';

const CORE_8 = ['genesis', 'heart', 'perception', 'memory', 'strategy', 'action', 'attention', 'emotion', 'executive'] as const;

const PHASE_ORDER: Record<string, number> = {
  bootstrap: 0,
  core_8: 1,
  active: 2,
  sleeping: 3,
};

export default function Lifecycle() {
  const { data: state, loading } = useVexState();

  if (loading || !state) {
    return <div className="dash-loading">Loading lifecycle data...</div>;
  }

  const activeCount = state.kernels?.active ?? 1;
  const totalSpawned = Math.min(activeCount, CORE_8.length);
  const phase = state.lifecycle_phase;
  const budget = state.kernels?.budget;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Lifecycle</h1>
        <div className="dash-subtitle">
          Phase:{' '}
          <span className={`status-badge ${phase === 'active' ? 'badge-success' : 'badge-info'}`}>
            {phase.toUpperCase()}
          </span>
          {' '}{totalSpawned - 1}/8 Core-8 spawned
        </div>
      </div>

      {/* Spawn Timeline */}
      <div className="dash-section">
        <div className="dash-section-title">Spawn Timeline</div>
        <div className="timeline">
          {CORE_8.map((spec, i) => (
            <div key={spec} style={{ display: 'contents' }}>
              {i > 0 && (
                <div className={`timeline-line ${i < totalSpawned ? 'complete' : ''}`} />
              )}
              <div className="timeline-node">
                <div className={`timeline-dot ${i < totalSpawned ? 'complete' : 'pending'}`} />
                <div className="timeline-label">{spec.substring(0, 4)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Spawn Gates */}
      <div className="dash-section">
        <div className="dash-section-title">Spawn Gates</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">\u03A6 {'>'} 0.30 (emergency)</span>
            <span className="dash-row-value" style={{ color: state.phi > 0.3 ? 'var(--alive)' : 'var(--error)' }}>
              {state.phi > 0.3 ? '\u2713' : '\u2717'} {state.phi.toFixed(3)}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Velocity \u2260 critical</span>
            <span className="dash-row-value" style={{ color: state.velocity?.regime !== 'critical' ? 'var(--alive)' : 'var(--error)' }}>
              {state.velocity?.regime !== 'critical' ? '\u2713' : '\u2717'} {state.velocity?.regime ?? 'unknown'}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cooldown \u2265 10 cycles</span>
            <span className="dash-row-value" style={{ color: 'var(--alive)' }}>
              \u2713
            </span>
          </div>
        </div>
      </div>

      {/* Budget */}
      <div className="dash-section">
        <div className="dash-section-title">Budget</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">GOD</span>
            <span className="dash-row-value">
              {budget?.god_used ?? 0} / {budget?.god_budget ?? 240}
              {' '}({((budget?.god_used ?? 0) / (budget?.god_budget ?? 240) * 100).toFixed(1)}%)
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">CHAOS</span>
            <span className="dash-row-value">
              {budget?.chaos_used ?? 0} / {budget?.chaos_budget ?? 200}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Core-8</span>
            <span className="dash-row-value">
              {totalSpawned - 1} / 8
              {budget?.core8_complete && ' (COMPLETE)'}
            </span>
          </div>
        </div>
      </div>

      {/* Phase Transitions */}
      <div className="dash-section">
        <div className="dash-section-title">Phase Transitions</div>
        <div className="dash-card">
          {['bootstrap', 'core_8', 'active', 'sleeping'].map((p, i) => {
            const currentIdx = PHASE_ORDER[phase] ?? 0;
            const thisIdx = PHASE_ORDER[p] ?? 0;
            const isCurrent = p === phase;
            const isPast = thisIdx < currentIdx;
            return (
              <div key={p} className="dash-row">
                <span className="dash-row-label" style={{ fontWeight: isCurrent ? 600 : 400 }}>
                  {isPast ? '\u2713' : isCurrent ? '\u25B6' : '\u25CB'} {p.toUpperCase()}
                </span>
                {i < 3 && (
                  <span className="dash-row-value" style={{ color: isPast ? 'var(--alive)' : 'var(--text-dim)' }}>
                    \u2192
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
