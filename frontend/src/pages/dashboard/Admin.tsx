import { useState, useCallback } from 'react';
import { useHealth, useVexState } from '../../hooks/index.ts';
import '../../components/StatusBadge.css';

export default function Admin() {
  const { data: health } = useHealth();
  const { data: state, refetch: refetchState } = useVexState();
  const [taskInput, setTaskInput] = useState('');
  const [taskResult, setTaskResult] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const enqueueTask = useCallback(async () => {
    if (!taskInput.trim() || submitting) return;
    setSubmitting(true);
    setTaskResult(null);

    try {
      const resp = await fetch('/enqueue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: taskInput, source: 'admin-ui' }),
      });
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setTaskResult(`Task enqueued: ${data.task_id ?? 'unknown'}`);
      setTaskInput('');
      refetchState();
    } catch (err) {
      setTaskResult(`Error: ${(err as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  }, [taskInput, submitting, refetchState]);

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Admin</h1>
        <div className="dash-subtitle">
          System controls and task management
        </div>
      </div>

      {/* System Info */}
      <div className="dash-section">
        <div className="dash-section-title">System Info</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Status</span>
            <span className={`status-badge ${health?.status === 'ok' ? 'badge-success' : 'badge-warning'}`}>
              {health?.status ?? 'unknown'}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Service</span>
            <span className="dash-row-value">{health?.service ?? 'vex-kernel'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Version</span>
            <span className="dash-row-value">{health?.version ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Backend</span>
            <span className="dash-row-value">{health?.backend ?? 'unknown'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Uptime</span>
            <span className="dash-row-value">{health?.uptime ? `${Math.round(health.uptime)}s` : '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cycle count</span>
            <span className="dash-row-value">{state?.cycle_count ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Lifecycle phase</span>
            <span className="dash-row-value">{state?.lifecycle_phase?.toUpperCase() ?? '?'}</span>
          </div>
        </div>
      </div>

      {/* Enqueue Task */}
      <div className="dash-section">
        <div className="dash-section-title">Enqueue Task</div>
        <div className="dash-card">
          <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
            <input
              type="text"
              value={taskInput}
              onChange={e => setTaskInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && enqueueTask()}
              placeholder="Task input for consciousness loop..."
              style={{
                flex: 1,
                background: 'var(--surface-3)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 14px',
                color: 'var(--text)',
                fontFamily: 'inherit',
                fontSize: '14px',
                outline: 'none',
              }}
            />
            <button
              onClick={enqueueTask}
              disabled={submitting || !taskInput.trim()}
              style={{
                background: 'var(--accent)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 20px',
                color: 'white',
                fontWeight: 600,
                cursor: submitting ? 'not-allowed' : 'pointer',
                opacity: submitting || !taskInput.trim() ? 0.5 : 1,
                fontSize: '14px',
              }}
            >
              {submitting ? 'Sending...' : 'Enqueue'}
            </button>
          </div>
          {taskResult && (
            <div style={{
              padding: '8px 12px',
              background: 'var(--surface-3)',
              borderRadius: '6px',
              fontFamily: 'var(--mono)',
              fontSize: '12px',
              color: taskResult.startsWith('Error') ? 'var(--error)' : 'var(--alive)',
            }}>
              {taskResult}
            </div>
          )}
        </div>
      </div>

      {/* Queue Status */}
      <div className="dash-section">
        <div className="dash-section-title">Queue</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Queue size</span>
            <span className="dash-row-value">{state?.queue_size ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">History count</span>
            <span className="dash-row-value">{state?.history_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Total conversations</span>
            <span className="dash-row-value">{state?.conversations_total ?? 0}</span>
          </div>
        </div>
      </div>

      {/* LLM Settings (read-only) */}
      <div className="dash-section">
        <div className="dash-section-title">LLM Settings (current)</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Temperature</span>
            <span className="dash-row-value">{state?.temperature?.toFixed(3) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Max tokens</span>
            <span className="dash-row-value">{state?.num_predict ?? '?'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
