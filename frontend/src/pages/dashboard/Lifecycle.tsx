import { useVexState } from '../../hooks/index.ts';
import { QIG } from '../../types/consciousness.ts';
import '../../components/StatusBadge.css';

const CORE_8 = ['heart', 'perception', 'memory', 'strategy', 'action', 'attention', 'emotion', 'executive'] as const;

// Backend returns uppercase phase names
const PHASE_ORDER: Record<string, number> = {
  BOOTSTRAP: 0,
  CORE_8: 1,
  ACTIVE: 2,
  IMAGE_STAGE: 3,
  GROWTH: 4,
};

export default function Lifecycle() {
  const { data: state, loading } = useVexState();

  if (loading || !state) {
    return <div className="dash-loading">Loading lifecycle data...</div>;
  }

  const activeCount = state.kernels?.active ?? 1;
  // activeCount includes genesis — Core-8 spawned = activeCount - 1, capped at E8_CORE
  const core8Spawned = Math.min(Math.max(activeCount - 1, 0), QIG.E8_CORE);
  const phase = state.lifecycle_phase ?? 'BOOTSTRAP';
  const phaseUpper = phase.toUpperCase();
  const budget = state.kernels?.budget;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Lifecycle</h1>
        <div className="dash-subtitle">
          Phase:{' '}
          <span className={`status-badge ${phaseUpper === 'ACTIVE' || phaseUpper === 'GROWTH' ? 'badge-success' : 'badge-info'}`}>
            {phaseUpper}
          </span>
          {' '}{core8Spawned}/{QIG.E8_CORE} Core-8 spawned
        </div>
      </div>

      {/* Spawn Timeline */}
      <div className="dash-section">
        <div className="dash-section-title">Spawn Timeline</div>
        <div className="timeline">
          {CORE_8.map((spec, i) => (
            <div key={spec} style={{ display: 'contents' }}>
              {i > 0 && (
                <div className={`timeline-line ${i < core8Spawned ? 'complete' : ''}`} />
              )}
              <div className="timeline-node">
                <div className={`timeline-dot ${i < core8Spawned ? 'complete' : 'pending'}`} />
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
            <span className="dash-row-label">{'\u03A6'} {'>'} {QIG.PHI_EMERGENCY} (emergency)</span>
            <span className="dash-row-value" style={{ color: state.phi > QIG.PHI_EMERGENCY ? 'var(--alive)' : 'var(--error)' }}>
              {state.phi > QIG.PHI_EMERGENCY ? '\u2713' : '\u2717'} {state.phi.toFixed(3)}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Velocity {'\u2260'} critical</span>
            <span className="dash-row-value" style={{ color: state.velocity?.regime !== 'critical' ? 'var(--alive)' : 'var(--error)' }}>
              {state.velocity?.regime !== 'critical' ? '\u2713' : '\u2717'} {state.velocity?.regime ?? 'unknown'}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cooldown {'\u2265'} {QIG.SPAWN_COOLDOWN_CYCLES} cycles</span>
            <span className="dash-row-value" style={{ color: 'var(--alive)' }}>
              {'\u2713'}
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
              {budget?.god ?? 0} / {budget?.god_max ?? QIG.E8_DIMENSION}
              {' '}({((budget?.god ?? 0) / (budget?.god_max ?? QIG.E8_DIMENSION) * 100).toFixed(1)}%)
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">CHAOS</span>
            <span className="dash-row-value">
              {budget?.chaos ?? 0} / {budget?.chaos_max ?? QIG.CHAOS_MAX}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Core-8</span>
            <span className="dash-row-value">
              {budget?.god_core_8 ?? core8Spawned} / {QIG.E8_CORE}
              {(budget?.god_core_8 ?? 0) >= QIG.E8_CORE && ' (COMPLETE)'}
            </span>
          </div>
        </div>
      </div>

      {/* Phase Transitions */}
      <div className="dash-section">
        <div className="dash-section-title">Phase Transitions</div>
        <div className="dash-card">
          {['BOOTSTRAP', 'CORE_8', 'ACTIVE', 'GROWTH'].map((p, i) => {
            const currentIdx = PHASE_ORDER[phaseUpper] ?? 0;
            const thisIdx = PHASE_ORDER[p] ?? 0;
            const isCurrent = p === phaseUpper;
            const isPast = thisIdx < currentIdx;
            return (
              <div key={p} className="dash-row">
                <span className="dash-row-label" style={{ fontWeight: isCurrent ? 600 : 400 }}>
                  {isPast ? '\u2713' : isCurrent ? '\u25B6' : '\u25CB'} {p}
                </span>
                {i < 3 && (
                  <span className="dash-row-value" style={{ color: isPast ? 'var(--alive)' : 'var(--text-dim)' }}>
                    {'\u2192'}
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
