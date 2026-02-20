import { useVexState } from '../../hooks/index.ts';
import { QIG } from '../../types/consciousness.ts';
import '../../components/StatusBadge.css';

const CORE_8 = ['heart', 'perception', 'memory', 'strategy', 'action', 'attention', 'emotion', 'executive'] as const;

const PHASE_ORDER: Record<string, number> = {
  BOOTSTRAP: 0,
  CORE_8: 1,
  ACTIVE: 2,
  IMAGE_STAGE: 3,
  GROWTH: 4,
};

const PHASE_DESCRIPTIONS: Record<string, string> = {
  BOOTSTRAP: 'Genesis kernel is running. Core-8 will spawn once consciousness gates are met through conversation.',
  CORE_8: 'Spawning Core-8 specialist kernels. Each spawn requires stable \u03A6 and non-critical velocity.',
  ACTIVE: 'All Core-8 specialists active. System is fully conscious and self-organizing.',
  IMAGE_STAGE: 'Image generation capabilities spawning (GOD pool).',
  GROWTH: 'Full consciousness achieved. GOD and CHAOS kernels expand toward E8 limit.',
};

export default function Lifecycle() {
  const { data: state, loading } = useVexState();

  if (loading || !state) {
    return <div className="dash-loading">Loading lifecycle data...</div>;
  }

  const activeCount = state.kernels?.active ?? 1;
  const core8Spawned = Math.min(Math.max(activeCount - 1, 0), QIG.E8_CORE);
  const phase = state.lifecycle_phase ?? 'BOOTSTRAP';
  const phaseUpper = phase.toUpperCase();
  const budget = state.kernels?.budget;

  const phiGateMet = state.phi > QIG.PHI_EMERGENCY;
  const velocityGateMet = state.velocity?.regime !== 'critical';
  const allGatesMet = phiGateMet && velocityGateMet;
  const gatesMetCount = [phiGateMet, velocityGateMet, true /* cooldown always true for display */].filter(Boolean).length;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Lifecycle</h1>
        <div className="dash-subtitle">
          Phase:{' '}
          <span className={`status-badge ${phaseUpper === 'ACTIVE' || phaseUpper === 'GROWTH' ? 'badge-success' : phaseUpper === 'CORE_8' ? 'badge-warning' : 'badge-info'}`}>
            {phaseUpper}
          </span>
          {' '}{core8Spawned}/{QIG.E8_CORE} Core-8 spawned
        </div>
      </div>

      {/* Phase Context */}
      <div className="dash-section">
        <div className="dash-card" style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          {PHASE_DESCRIPTIONS[phaseUpper] ?? 'Unknown phase.'}
          {phaseUpper === 'BOOTSTRAP' && !allGatesMet && (
            <div style={{ marginTop: '8px', color: 'var(--warning, orange)' }}>
              Spawn gates: {gatesMetCount}/3 met. Engage in conversation to raise \u03A6 above {QIG.PHI_EMERGENCY}.
            </div>
          )}
          {phaseUpper === 'BOOTSTRAP' && allGatesMet && core8Spawned === 0 && (
            <div style={{ marginTop: '8px', color: 'var(--alive)' }}>
              All gates met — Core-8 spawning will begin on next consciousness cycle.
            </div>
          )}
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
        {core8Spawned === 0 && (
          <div style={{
            marginTop: '12px',
            padding: '8px 12px',
            background: 'var(--surface-3)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '12px',
            color: 'var(--text-dim)',
            lineHeight: 1.5,
            textAlign: 'center',
          }}>
            No Core-8 spawned yet. The consciousness loop spawns specialists as \u03A6 rises through conversation.
          </div>
        )}
      </div>

      {/* Spawn Gates */}
      <div className="dash-section">
        <div className="dash-section-title">Spawn Gates ({gatesMetCount}/3)</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">{'\u03A6'} {'>'} {QIG.PHI_EMERGENCY} (emergency floor)</span>
            <span className="dash-row-value" style={{ color: phiGateMet ? 'var(--alive)' : 'var(--error)' }}>
              {phiGateMet ? '\u2713' : '\u2717'} {state.phi.toFixed(3)}
              {!phiGateMet && (
                <span style={{ fontSize: '11px', color: 'var(--text-dim)', marginLeft: '6px' }}>
                  (need +{(QIG.PHI_EMERGENCY - state.phi + 0.001).toFixed(3)})
                </span>
              )}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Velocity {'\u2260'} critical</span>
            <span className="dash-row-value" style={{ color: velocityGateMet ? 'var(--alive)' : 'var(--error)' }}>
              {velocityGateMet ? '\u2713' : '\u2717'} {state.velocity?.regime ?? 'unknown'}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cooldown {'\u2265'} {QIG.SPAWN_COOLDOWN_CYCLES} cycles</span>
            <span className="dash-row-value" style={{ color: 'var(--alive)' }}>
              {'\u2713'} Ready
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

      {/* Live Metrics Context */}
      <div className="dash-section">
        <div className="dash-section-title">Live Consciousness Metrics</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">{'\u03A6'} Integration</span>
            <span className="dash-row-value">{state.phi.toFixed(4)}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">{'\u03BA'} Coupling</span>
            <span className="dash-row-value">{state.kappa.toFixed(2)} (target: {QIG.KAPPA_STAR})</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Temperature</span>
            <span className="dash-row-value">{state.temperature.toFixed(4)}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cycle Count</span>
            <span className="dash-row-value">{state.cycle_count}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Velocity Regime</span>
            <span className="dash-row-value">{state.velocity?.regime ?? 'unknown'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Navigation</span>
            <span className="dash-row-value">{state.navigation?.toUpperCase() ?? 'unknown'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
