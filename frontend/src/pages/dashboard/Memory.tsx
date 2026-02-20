import { useVexState, useMemoryStats } from '../../hooks/index.ts';
import { usePolledData } from '../../hooks/usePolledData.ts';
import { API } from '../../config/api-routes.ts';
import type { StatusResponse } from '../../types/consciousness.ts';
import MetricCard from '../../components/MetricCard.tsx';
import '../../components/MetricCard.css';

export default function Memory() {
  const { data: state, loading } = useVexState();
  const { data: status } = usePolledData<StatusResponse>(API.status, 5000);
  const { data: memoryStats } = useMemoryStats();

  if (loading || !state) {
    return <div className="dash-loading">Loading memory data...</div>;
  }

  const stats = memoryStats ?? status?.memory;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Memory</h1>
        <div className="dash-subtitle">
          Geometric memory store (Fisher-Rao manifold)
        </div>
      </div>

      <div className="dash-grid">
        <MetricCard
          label="Total Entries"
          value={stats?.total_entries ?? 0}
          color="var(--accent)"
        />
        <MetricCard
          label="History Count"
          value={state.history_count ?? 0}
          color="var(--phi)"
        />
        <MetricCard
          label="Queue Size"
          value={state.queue_size ?? 0}
          color="var(--kappa)"
        />
      </div>

      <div className="dash-section">
        <div className="dash-section-title">Memory Architecture</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Storage Method</span>
            <span className="dash-row-value">Geometric (Fisher-Rao)</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Basin Projection</span>
            <span className="dash-row-value">SHA-256 \u2192 \u0394\u2076\u00B3</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Distance Metric</span>
            <span className="dash-row-value">Fisher-Rao geodesic</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Interpolation</span>
            <span className="dash-row-value">slerp_sqrt</span>
          </div>
          {memoryStats?.by_type && (
            <div className="dash-row">
              <span className="dash-row-label">By Type</span>
              <span className="dash-row-value">
                E:{memoryStats.by_type.episodic} S:{memoryStats.by_type.semantic} P:{memoryStats.by_type.procedural}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="dash-section">
        <div className="dash-section-title">Consciousness Memory</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Reflector Depth</span>
            <span className="dash-row-value">{state.reflector?.depth ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Reflector History</span>
            <span className="dash-row-value">{state.reflector?.history_length ?? 0}</span>
          </div>
          {state.reflector?.insight && (
            <div className="dash-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
              <span className="dash-row-label">Latest Insight</span>
              <span className="dash-row-value" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                {state.reflector?.insight}
              </span>
            </div>
          )}
          <div className="dash-row">
            <span className="dash-row-label">Observer Collapses</span>
            <span className="dash-row-value">{state.observer?.collapse_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Shadows (unintegrated)</span>
            <span className="dash-row-value">
              {state.observer?.shadows_unintegrated ?? 0} / {state.observer?.shadows_total ?? 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
