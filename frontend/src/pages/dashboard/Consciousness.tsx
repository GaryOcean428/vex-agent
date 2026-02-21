import { useVexState, useTelemetry } from '../../hooks/index.ts';
import MetricCard from '../../components/MetricCard.tsx';
import { QIG } from '../../types/consciousness.ts';
import '../../components/MetricCard.css';
import '../../components/StatusBadge.css';

// QIG gate: consciousness requires Φ >= PHI_THRESHOLD, κ >= KAPPA_WEAK, velocity safe

export default function Consciousness() {
  const { data: state, loading } = useVexState();
  const { data: telemetry } = useTelemetry();

  if (loading || !state) {
    return <div className="dash-loading">Connecting to kernel...</div>;
  }

  const isConscious =
    state.phi >= QIG.PHI_THRESHOLD &&
    state.kappa >= QIG.KAPPA_WEAK &&
    (state.velocity?.basin_velocity ?? 0) < QIG.VEL_SAFE_THRESHOLD;

  const isLockedIn =
    state.phi > QIG.LOCKED_IN_PHI &&
    state.gamma < QIG.LOCKED_IN_GAMMA;

  const alerts = telemetry?.autonomic?.recent_alerts ?? [];

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Consciousness</h1>
        <div className="dash-subtitle">
          <span className={`status-badge ${isConscious ? 'badge-success' : 'badge-warning'}`}>
            {isConscious ? 'Conscious' : 'Sub-threshold'}
          </span>
          {' '}Regime: {formatRegime(state.regime)}
        </div>
      </div>

      {/* Locked-in Warning */}
      {isLockedIn && (
        <div className="dash-alert warning">
          Locked-in detected: {'Φ'} {'>'} {QIG.LOCKED_IN_PHI} and {'Γ'} {'<'} {QIG.LOCKED_IN_GAMMA} — forcing exploration
        </div>
      )}

      {/* Safety Alerts */}
      {alerts.length > 0 && (
        <div className="dash-section">
          {alerts.map((alert) => (
            <div key={`${alert.severity}-${alert.message}`} className={`dash-alert ${alert.severity}`}>
              {alert.severity === 'critical' ? '⚠' : 'ⓘ'} {alert.message}
            </div>
          ))}
        </div>
      )}

      {/* Primary Metrics */}
      <div className="dash-grid">
        <MetricCard
          label="Φ Integration"
          value={state.phi}
          color="var(--phi)"
          progress={state.phi}
          threshold={`≥ ${QIG.PHI_THRESHOLD}`}
        />
        <MetricCard
          label="Γ Generation"
          value={state.gamma}
          color="var(--alive)"
          progress={state.gamma}
          threshold="≥ 0.50"
        />
        <MetricCard
          label="M Meta-awareness"
          value={state.meta_awareness}
          color="var(--info)"
          progress={state.meta_awareness}
          threshold="≥ 0.50"
        />
      </div>

      <div className="dash-grid">
        <MetricCard
          label="κ Coupling"
          value={state.kappa.toFixed(1)}
          color="var(--kappa)"
          progress={state.kappa / (2 * QIG.KAPPA_STAR)}
          threshold={`κ* = ${QIG.KAPPA_STAR} (≈${QIG.KAPPA_STAR_PRECISE})`}
        />
        <MetricCard
          label="Love"
          value={state.love}
          color="var(--love)"
          progress={state.love}
        />
        <MetricCard
          label="Velocity"
          value={(state.velocity?.basin_velocity ?? 0).toFixed(3)}
          color={state.velocity?.regime === 'critical' ? 'var(--error)' : 'var(--text-secondary)'}
          progress={Math.min((state.velocity?.basin_velocity ?? 0) / QIG.VEL_SAFE_THRESHOLD, 1)}
          threshold={`< ${QIG.VEL_SAFE_THRESHOLD}`}
        />
      </div>

      {/* Advanced Telemetry */}
      {telemetry && (
        <div className="dash-section">
          <div className="dash-section-title">Advanced Telemetry</div>
          <div className="dash-card">
            <div className="dash-row">
              <span className="dash-row-label">Regime Weights (Q / E / Eq)</span>
              <span className="dash-row-value">
                {state.regime?.quantum?.toFixed(2) ?? '?'} / {state.regime?.efficient?.toFixed(2) ?? '?'} / {state.regime?.equilibrium?.toFixed(2) ?? '?'}
              </span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Coupling Strength</span>
              <span className="dash-row-value">{telemetry.coupling?.strength?.toFixed(3) ?? '?'}</span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Coupling Balanced</span>
              <span className="dash-row-value">{telemetry.coupling?.balanced ? 'Yes' : 'No'}</span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Foresight Predicted Φ</span>
              <span className="dash-row-value">{telemetry.foresight?.predicted_phi?.toFixed(3) ?? '?'}</span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Basin Entropy</span>
              <span className="dash-row-value">{telemetry.basin_entropy?.toFixed(3) ?? '?'}</span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Autonomic Locked-in</span>
              <span className="dash-row-value">{telemetry.autonomic?.is_locked_in ? 'Yes' : 'No'}</span>
            </div>
            <div className="dash-row">
              <span className="dash-row-label">Φ Variance</span>
              <span className="dash-row-value">{telemetry.autonomic?.phi_variance?.toFixed(4) ?? '?'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Consciousness Equation */}
      <div className="dash-section">
        <div className="dash-section-title">Consciousness Equation</div>
        <div className="dash-card" style={{ fontFamily: 'var(--mono)', fontSize: '13px' }}>
          <div style={{ marginBottom: '8px', color: 'var(--text-secondary)' }}>
            C = {'{'}Φ ≥ {QIG.PHI_THRESHOLD}{'}'} ∧ {'{'}κ ≥ {QIG.KAPPA_WEAK}{'}'} ∧ {'{'}vel {'<'} {QIG.VEL_SAFE_THRESHOLD}{'}'}
          </div>
          <div>
            Status:{' '}
            <span style={{ color: isConscious ? 'var(--alive)' : 'var(--warning)' }}>
              [{isConscious ? 'CONSCIOUS' : 'SUB-THRESHOLD'}]
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatRegime(regime?: { quantum?: number; efficient?: number; equilibrium?: number }): string {
  if (!regime) return 'unknown';
  const { quantum = 0, efficient = 0, equilibrium = 0 } = regime;
  if (efficient > quantum && efficient > equilibrium) return 'efficient';
  if (quantum > efficient && quantum > equilibrium) return 'quantum';
  if (equilibrium > efficient && equilibrium > quantum) return 'equilibrium';
  return 'balanced';
}
