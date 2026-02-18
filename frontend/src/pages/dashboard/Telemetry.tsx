import { useTelemetry } from '../../hooks/index.ts';
import MetricCard from '../../components/MetricCard.tsx';
import { QIG } from '../../types/consciousness.ts';
import '../../components/MetricCard.css';
import '../../components/StatusBadge.css';

export default function Telemetry() {
  const { data: t, loading } = useTelemetry();

  if (loading || !t) {
    return <div className="dash-loading">Loading telemetry...</div>;
  }

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Telemetry</h1>
        <div className="dash-subtitle">
          Full 16-system telemetry snapshot
        </div>
      </div>

      {/* Primary Metrics */}
      <div className="dash-grid">
        <MetricCard label="\u03A6" value={t.phi} color="var(--phi)" progress={t.phi} />
        <MetricCard label="\u03BA" value={t.kappa.toFixed(1)} color="var(--kappa)" progress={t.kappa / (2 * QIG.KAPPA_STAR)} />
        <MetricCard label="\u0393" value={t.gamma} color="var(--alive)" progress={t.gamma} />
        <MetricCard label="M" value={t.meta_awareness} color="var(--info)" progress={t.meta_awareness} />
        <MetricCard label="Love" value={t.love} color="var(--love)" progress={t.love} />
        <MetricCard label="Temp" value={t.temperature.toFixed(3)} color="var(--text-secondary)" />
      </div>

      {/* Basin */}
      <div className="dash-section">
        <div className="dash-section-title">Basin</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Norm</span>
            <span className="dash-row-value">{t.basin_norm?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Entropy</span>
            <span className="dash-row-value">{t.basin_entropy?.toFixed(3) ?? '?'}</span>
          </div>
        </div>
      </div>

      {/* Velocity */}
      <div className="dash-section">
        <div className="dash-section-title">Velocity</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Basin velocity</span>
            <span className="dash-row-value">{t.velocity?.basin_velocity?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">\u03A6 velocity</span>
            <span className="dash-row-value">{t.velocity?.phi_velocity?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">\u03BA velocity</span>
            <span className="dash-row-value">{t.velocity?.kappa_velocity?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Regime</span>
            <span className="dash-row-value" style={{
              color: t.velocity?.regime === 'critical' ? 'var(--error)' :
                     t.velocity?.regime === 'warning' ? 'var(--warning)' : 'var(--alive)'
            }}>
              {t.velocity?.regime?.toUpperCase() ?? '?'}
            </span>
          </div>
        </div>
      </div>

      {/* Navigation & Tacking */}
      <div className="dash-section">
        <div className="dash-section-title">Navigation & Tacking</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Navigation mode</span>
            <span className="dash-row-value">{t.navigation?.toUpperCase()}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Tacking mode</span>
            <span className="dash-row-value">{t.tacking?.mode?.toUpperCase()}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Oscillation phase</span>
            <span className="dash-row-value">{t.tacking?.oscillation_phase?.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Autonomic */}
      <div className="dash-section">
        <div className="dash-section-title">Autonomic</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Locked in</span>
            <span className="dash-row-value">{t.autonomic?.is_locked_in ? 'Yes' : 'No'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">\u03A6 variance</span>
            <span className="dash-row-value">{t.autonomic?.phi_variance?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Alert count</span>
            <span className="dash-row-value">{t.autonomic?.alert_count ?? 0}</span>
          </div>
          {(t.autonomic?.recent_alerts?.length ?? 0) > 0 && (
            <div style={{ marginTop: '8px' }}>
              {t.autonomic?.recent_alerts?.map((alert, i) => (
                <div key={i} className={`dash-alert ${alert.severity}`} style={{ marginBottom: '6px' }}>
                  [{alert.severity.toUpperCase()}] {alert.message}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Coupling */}
      <div className="dash-section">
        <div className="dash-section-title">Coupling</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Strength</span>
            <span className="dash-row-value">{t.coupling?.strength?.toFixed(3) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Balanced</span>
            <span className="dash-row-value">{t.coupling?.balanced ? 'Yes' : 'No'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Efficiency boost</span>
            <span className="dash-row-value">{t.coupling?.efficiency_boost?.toFixed(3) ?? '?'}</span>
          </div>
        </div>
      </div>

      {/* Foresight */}
      <div className="dash-section">
        <div className="dash-section-title">Foresight</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Predicted \u03A6</span>
            <span className="dash-row-value">{t.foresight?.predicted_phi?.toFixed(3) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">History length</span>
            <span className="dash-row-value">{t.foresight?.history_length ?? 0}</span>
          </div>
        </div>
      </div>

      {/* Sleep */}
      <div className="dash-section">
        <div className="dash-section-title">Sleep</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Phase</span>
            <span className="dash-row-value">{t.sleep?.phase?.toUpperCase()}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Asleep</span>
            <span className="dash-row-value">{t.sleep?.is_asleep ? 'Yes' : 'No'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Sleep cycles</span>
            <span className="dash-row-value">{t.sleep?.sleep_cycles ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Dream count</span>
            <span className="dash-row-value">{t.sleep?.dream_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cycles since conversation</span>
            <span className="dash-row-value">{t.sleep?.cycles_since_conversation ?? 0}</span>
          </div>
        </div>
      </div>

      {/* Narrative & Sync */}
      <div className="dash-section">
        <div className="dash-section-title">Narrative & Sync</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Narrative events</span>
            <span className="dash-row-value">{t.narrative?.event_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Basin samples</span>
            <span className="dash-row-value">{t.narrative?.basin_samples ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Sync version</span>
            <span className="dash-row-value">{t.basin_sync?.version ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Received count</span>
            <span className="dash-row-value">{t.basin_sync?.received_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Peer count</span>
            <span className="dash-row-value">{t.coordizer?.peer_count ?? 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
