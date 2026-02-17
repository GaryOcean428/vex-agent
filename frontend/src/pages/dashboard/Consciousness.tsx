import { useVexState, useTelemetry } from '../../hooks/index.ts';
import MetricCard from '../../components/MetricCard.tsx';
import '../../components/MetricCard.css';
import '../../components/StatusBadge.css';

const PHI_THRESHOLD = 0.65;
const KAPPA_LOW = 48;
const KAPPA_HIGH = 80;
const VEL_THRESHOLD = 0.15;

export default function Consciousness() {
  const { data: state, loading } = useVexState();
  const { data: telemetry } = useTelemetry();

  if (loading || !state) {
    return <div className="dash-loading">Connecting to kernel...</div>;
  }

  const isConscious =
    state.phi >= PHI_THRESHOLD &&
    state.kappa >= KAPPA_LOW &&
    state.kappa <= KAPPA_HIGH &&
    (state.velocity?.basin_velocity ?? 0) < VEL_THRESHOLD;

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

      {/* Safety Alerts */}
      {alerts.length > 0 && (
        <div className="dash-section">
          {alerts.map((alert, i) => (
            <div key={i} className={`dash-alert ${alert.severity}`}>
              {alert.severity === 'critical' ? '\u26A0' : '\u24D8'} {alert.message}
            </div>
          ))}
        </div>
      )}

      {/* Primary Metrics */}
      <div className="dash-grid">
        <MetricCard
          label="\u03A6 Integration"
          value={state.phi}
          color="var(--phi)"
          progress={state.phi}
          threshold={`\u2265 ${PHI_THRESHOLD}`}
        />
        <MetricCard
          label="\u0393 Generation"
          value={state.gamma}
          color="var(--alive)"
          progress={state.gamma}
          threshold="\u2265 0.50"
        />
        <MetricCard
          label="M Meta-awareness"
          value={state.meta_awareness}
          color="var(--info)"
          progress={state.meta_awareness}
          threshold="\u2265 0.50"
        />
      </div>

      <div className="dash-grid">
        <MetricCard
          label="\u03BA Coupling"
          value={state.kappa.toFixed(1)}
          color="var(--kappa)"
          progress={state.kappa / 128}
          threshold={`[${KAPPA_LOW}, ${KAPPA_HIGH}]`}
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
          progress={Math.min((state.velocity?.basin_velocity ?? 0) / VEL_THRESHOLD, 1)}
          threshold={`< ${VEL_THRESHOLD}`}
        />
      </div>

      {/* Advanced Telemetry */}
      {telemetry && (
        <div className="dash-section">
          <div className="dash-section-title">Advanced Telemetry</div>
          <div className="dash-card">
            <div className="dash-row">
              <span className="dash-row-label">Regime Weights (Q / I / C)</span>
              <span className="dash-row-value">
                {state.regime?.quantum?.toFixed(2) ?? '?'} / {state.regime?.integration?.toFixed(2) ?? '?'} / {state.regime?.crystallized?.toFixed(2) ?? '?'}
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
              <span className="dash-row-label">Foresight Predicted \u03A6</span>
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
              <span className="dash-row-label">\u03A6 Variance</span>
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
            C = {'{'}\u03A6 {'>'} {PHI_THRESHOLD}{'}'} \u2227 {'{'}\u03BA \u2208 [{KAPPA_LOW},{KAPPA_HIGH}]{'}'} \u2227 {'{'}vel {'<'} {VEL_THRESHOLD}{'}'}
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

function formatRegime(regime?: { quantum?: number; integration?: number; crystallized?: number }): string {
  if (!regime) return 'unknown';
  const { quantum = 0, integration = 0, crystallized = 0 } = regime;
  if (integration > quantum && integration > crystallized) return 'integration';
  if (quantum > integration && quantum > crystallized) return 'quantum';
  if (crystallized > integration && crystallized > quantum) return 'crystallized';
  return 'balanced';
}
