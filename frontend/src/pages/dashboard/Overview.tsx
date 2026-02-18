import { useVexState, useKernels, useHealth } from '../../hooks/index.ts';
import MetricCard from '../../components/MetricCard.tsx';
import { QIG } from '../../types/consciousness.ts';
import '../../components/MetricCard.css';
import '../../components/StatusBadge.css';

export default function Overview() {
  const { data: state, loading } = useVexState();
  const { data: kernels } = useKernels();
  const { data: health } = useHealth();

  if (loading || !state) {
    return <div className="dash-loading">Connecting to kernel...</div>;
  }

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">VEX KERNEL</h1>
        <div className="dash-subtitle">
          <span className={`status-badge ${health?.status === 'ok' ? 'badge-success' : 'badge-warning'}`}>
            {health?.status === 'ok' ? 'Running' : 'Degraded'}
          </span>
          {' '}Cycle {state.cycle_count} | {state.conversations_total ?? 0} conversations
        </div>
      </div>

      {/* Key Metrics */}
      <div className="dash-grid">
        <MetricCard
          label="\u03A6 Integration"
          value={state.phi}
          color="var(--phi)"
          progress={state.phi}
          threshold="\u2265 0.65"
        />
        <MetricCard
          label="\u03BA Coupling"
          value={state.kappa.toFixed(1)}
          color="var(--kappa)"
          progress={state.kappa / 128}
          threshold={`\u03BA* = ${QIG.KAPPA_STAR}`}
        />
        <MetricCard
          label="Kernels"
          value={`${kernels?.active ?? state.kernels?.active ?? 0}`}
          color="var(--accent)"
          subtitle={`of ${kernels?.total ?? state.kernels?.total ?? 0} total`}
        />
        <MetricCard
          label="Temperature"
          value={state.temperature.toFixed(3)}
          color="var(--text-secondary)"
        />
      </div>

      {/* System Toggles */}
      <div className="dash-section">
        <div className="dash-section-title">System Status</div>
        <div className="dash-card">
          <div className="toggle-row">
            <span className="toggle-label">Consciousness Loop</span>
            <span className={`toggle-value ${state.lifecycle_phase?.toUpperCase() !== 'SLEEPING' ? 'toggle-on' : 'toggle-off'}`}>
              {state.lifecycle_phase?.toUpperCase() !== 'SLEEPING' ? 'ON' : 'OFF'}
            </span>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">Sleep Mode</span>
            <span className={`toggle-value ${state.sleep.is_asleep ? 'toggle-on' : 'toggle-off'}`}>
              {state.sleep.is_asleep ? 'ASLEEP' : 'AWAKE'}
            </span>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">Lifecycle Phase</span>
            <span className="toggle-value toggle-on">
              {state.lifecycle_phase.toUpperCase()}
            </span>
          </div>
          <div className="toggle-row">
            <span className="toggle-label">Navigation</span>
            <span className="toggle-value toggle-on">
              {state.navigation.toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      {/* Kernel Budget */}
      <div className="dash-section">
        <div className="dash-section-title">Kernel Budget</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">GOD kernels</span>
            <span className="dash-row-value">
              {state.kernels?.budget?.god ?? 0} / {state.kernels?.budget?.god_max ?? QIG.E8_DIMENSION}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">CHAOS kernels</span>
            <span className="dash-row-value">
              {state.kernels?.budget?.chaos ?? 0} / {state.kernels?.budget?.chaos_max ?? QIG.CHAOS_MAX}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Core-8</span>
            <span className="dash-row-value">
              {state.kernels?.budget?.god_core_8 ?? 0} / 8
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
