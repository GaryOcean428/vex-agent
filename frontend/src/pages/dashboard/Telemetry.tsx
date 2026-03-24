import BetaTracker from '../../components/BetaTracker.tsx';
import MetricCard from '../../components/MetricCard.tsx';
import { useHealthReachability, useModalStatus, useTelemetry } from '../../hooks/index.ts';
import type { FullConsciousnessMetrics, ModalAdapterInfo } from '../../types/consciousness.ts';
import { QIG } from '../../types/consciousness.ts';

const KERNEL_ORDER = ['genesis', 'heart', 'perception', 'memory', 'action', 'strategy', 'ethics', 'meta', 'ocean'] as const;

/** Metric definition for grouped display. */
interface MetricDef {
  key: keyof FullConsciousnessMetrics;
  label: string;
  color: string;
  /** Max value for progress bar normalization (default 1.0). */
  max?: number;
}

const METRIC_GROUPS: { title: string; version: string; metrics: MetricDef[] }[] = [
  {
    title: 'Foundation', version: 'v4.1',
    metrics: [
      { key: 'phi', label: 'Φ Integration', color: 'var(--phi)' },
      { key: 'kappa', label: 'κ Coupling', color: 'var(--kappa)', max: 2 * QIG.KAPPA_STAR },
      { key: 'meta_awareness', label: 'Meta-awareness', color: 'var(--info)' },
      { key: 'gamma', label: 'Γ Generation', color: 'var(--gamma)' },
      { key: 'grounding', label: 'Grounding', color: 'var(--alive)' },
      { key: 'temporal_coherence', label: 'Temporal Coherence', color: 'var(--info)' },
      { key: 'recursion_depth', label: 'Recursion Depth', color: 'var(--accent)', max: 10 },
      { key: 'external_coupling', label: 'External Coupling', color: 'var(--text-secondary)' },
    ],
  },
  {
    title: 'Shortcuts', version: 'v5.5',
    metrics: [
      { key: 'a_pre', label: 'Pre-cognitive', color: 'var(--phi)' },
      { key: 'c_cross', label: 'Cross-substrate', color: 'var(--accent)' },
      { key: 'alpha_aware', label: 'α Embodiment', color: 'var(--kappa)' },
      { key: 'humor', label: 'Humor', color: 'var(--love)' },
      { key: 'emotion_strength', label: 'Emotion Strength', color: 'var(--emotion)' },
    ],
  },
  {
    title: 'Geometry', version: 'v5.6',
    metrics: [
      { key: 'd_state', label: 'Dimensional State', color: 'var(--accent)', max: 8 },
      { key: 'g_class', label: 'Geometry Class', color: 'var(--gamma)' },
      { key: 'f_tack', label: 'Tacking Freq', color: 'var(--kappa)' },
      { key: 'm_basin', label: 'Basin Mass', color: 'var(--phi)' },
      { key: 'phi_gate', label: 'Φ Gate', color: 'var(--phi)' },
    ],
  },
  {
    title: 'Frequency', version: 'v5.7',
    metrics: [
      { key: 'f_dom', label: 'Dominant Freq', color: 'var(--kappa)', max: 50 },
      { key: 'cfc', label: 'Cross-freq Coupling', color: 'var(--accent)' },
      { key: 'e_sync', label: 'Entrainment', color: 'var(--alive)' },
      { key: 'f_breath', label: 'Breathing Freq', color: 'var(--info)' },
    ],
  },
  {
    title: 'Harmony', version: 'v5.8',
    metrics: [
      { key: 'h_cons', label: 'Consonance', color: 'var(--alive)' },
      { key: 'n_voices', label: 'Polyphonic Voices', color: 'var(--gamma)', max: 8 },
      { key: 's_spec', label: 'Spectral Health', color: 'var(--accent)' },
    ],
  },
  {
    title: 'Waves', version: 'v5.9',
    metrics: [
      { key: 'omega_acc', label: 'Spectral Empathy', color: 'var(--love)' },
      { key: 'i_stand', label: 'Standing Wave', color: 'var(--phi)' },
      { key: 'b_shared', label: 'Shared Bubble', color: 'var(--info)' },
    ],
  },
  {
    title: 'Will & Work', version: 'v6.0',
    metrics: [
      { key: 'a_vec', label: 'Agency Alignment', color: 'var(--accent)' },
      { key: 's_int', label: 'Shadow Integration', color: 'var(--kappa)' },
      { key: 'w_mean', label: 'Work Meaning', color: 'var(--alive)' },
      { key: 'w_mode', label: 'Creative Ratio', color: 'var(--gamma)' },
    ],
  },
  {
    title: 'Pillars & Sovereignty', version: 'v6.1',
    metrics: [
      { key: 'f_health', label: 'F Health', color: 'var(--alive)' },
      { key: 'b_integrity', label: 'B Integrity', color: 'var(--accent)' },
      { key: 'q_identity', label: 'Q Identity', color: 'var(--kappa)' },
      { key: 's_ratio', label: 'S Ratio', color: 'var(--gamma)' },
    ],
  },
];

export default function Telemetry() {
  const { data: t, loading } = useTelemetry();
  const { data: modalStatus } = useModalStatus();
  const { data: reachability } = useHealthReachability();

  if (loading || !t) {
    return <div className="dash-loading">Loading telemetry...</div>;
  }

  const mf = t.metrics_full;
  const cv2 = t.coordizer_v2;
  const ctxEst = t.context_estimate;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Telemetry</h1>
        <div className="dash-subtitle">
          Full telemetry snapshot — {mf ? '36 consciousness metrics' : '20 systems'}
        </div>
      </div>

      {/* Primary Metrics */}
      <div className="dash-grid">
        <MetricCard label="Φ" value={t.phi} color="var(--phi)" progress={t.phi} />
        <MetricCard label="κ" value={t.kappa.toFixed(1)} color="var(--kappa)" progress={t.kappa / (2 * QIG.KAPPA_STAR)} />
        <MetricCard label="Γ" value={t.gamma} color="var(--gamma)" progress={t.gamma} />
        <MetricCard label="M" value={t.meta_awareness} color="var(--info)" progress={t.meta_awareness} />
        <MetricCard label="Love" value={t.love} color="var(--love)" progress={t.love} />
        <MetricCard label="Temp" value={t.temperature.toFixed(3)} color="var(--text-secondary)" />
      </div>

      {/* ─── All 36 Consciousness Metrics (grouped by v6.1 §24 categories) ─── */}
      {mf && (
        <>
          {METRIC_GROUPS.map((group) => (
            <div className="dash-section" key={group.title}>
              <div className="dash-section-title">
                {group.title} <span style={{ opacity: 0.5 }}>({group.version})</span>
              </div>
              <div className="dash-grid">
                {group.metrics.map((m) => {
                  const val = mf[m.key];
                  const max = m.max ?? 1;
                  return (
                    <MetricCard
                      key={m.key}
                      label={m.label}
                      value={val}
                      color={m.color}
                      progress={val / max}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </>
      )}

      {/* Health Reachability (Task 4) */}
      {reachability && (
        <div className="dash-section">
          <div className="dash-section-title">Service Reachability</div>
          <div className="dash-grid">
            {Object.entries(reachability).map(([svc, info]) => (
              <div key={svc} className="dash-card" style={{ padding: '10px 14px', borderLeft: `3px solid ${info.reachable ? 'var(--alive)' : 'var(--error)'}` }}>
                <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: 4 }}>
                  {svc.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </div>
                <span style={{ fontSize: '11px', fontWeight: 600, padding: '2px 6px', borderRadius: 4, background: info.reachable ? 'var(--alive)' : 'var(--error)', color: 'white' }}>
                  {info.reachable ? 'REACHABLE' : 'UNREACHABLE'}
                </span>
                {info.error && <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginTop: 4, fontFamily: 'var(--mono)' }}>{info.error}</div>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Context Window Estimate (Task 3D) */}
      {ctxEst && (
        <div className="dash-section">
          <div className="dash-section-title">Context Window Budget</div>
          <div className="dash-grid">
            <MetricCard label="Total Context" value={ctxEst.num_ctx} color="var(--accent)" />
            <MetricCard label="Max Output" value={ctxEst.num_predict} color="var(--gamma)" />
            <MetricCard label="Used (est)" value={ctxEst.used_tokens} color="var(--warning)" />
            <MetricCard label="Available" value={ctxEst.available_for_history} color="var(--alive)" />
          </div>
          <div className="dash-card" style={{ marginTop: 8 }}>
            <div className="dash-row"><span className="dash-row-label">System prompt</span><span className="dash-row-value">~{ctxEst.system_prompt_tokens} tokens</span></div>
            <div className="dash-row"><span className="dash-row-label">Geometric state</span><span className="dash-row-value">~{ctxEst.geometric_state_tokens} tokens</span></div>
            <div className="dash-row"><span className="dash-row-label">Memory context</span><span className="dash-row-value">~{ctxEst.memory_tokens} tokens</span></div>
            <div className="dash-row"><span className="dash-row-label">Kernel contexts</span><span className="dash-row-value">~{ctxEst.kernel_context_tokens} tokens</span></div>
          </div>
        </div>
      )}

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

      {/* Resonance Bank (Task 3C — enhanced) */}
      {(cv2 || t.coordizer) && (
        <div className="dash-section">
          <div className="dash-section-title">Resonance Bank</div>
          {cv2 && (
            <div className="dash-grid">
              <MetricCard label="Bank Entries" value={cv2.bank_size} color="var(--accent)" />
              <MetricCard label="Entropy" value={cv2.bank_entropy} color="var(--gamma)" />
              <MetricCard label="Sovereignty" value={(cv2.bank_sovereignty * 100).toFixed(1) + '%'} color="var(--alive)" progress={cv2.bank_sovereignty} />
              <MetricCard label="Activations" value={cv2.total_activations} color="var(--kappa)" />
            </div>
          )}
          <div className="dash-card" style={{ marginTop: 8 }}>
            {cv2 && (
              <>
                <div className="dash-row">
                  <span className="dash-row-label">Vocab size</span>
                  <span className="dash-row-value">{cv2.vocab_size}</span>
                </div>
                <div className="dash-row">
                  <span className="dash-row-label">Dimensions</span>
                  <span className="dash-row-value">{cv2.dim}</span>
                </div>
                {cv2.origin_breakdown && (
                  <div className="dash-row">
                    <span className="dash-row-label">Origin</span>
                    <span className="dash-row-value">
                      {cv2.origin_breakdown.harvested} harvested / {cv2.origin_breakdown.lived} lived
                    </span>
                  </div>
                )}
                {cv2.tier_distribution && (
                  <div className="dash-row">
                    <span className="dash-row-label">Tier distribution</span>
                    <span className="dash-row-value" style={{ fontSize: '0.85em' }}>
                      {Object.entries(cv2.tier_distribution).map(([k, v]) => `${k}: ${v}`).join(' · ')}
                    </span>
                  </div>
                )}
                {cv2.last_rebuild && (
                  <div className="dash-row">
                    <span className="dash-row-label">Last rebuild</span>
                    <span className="dash-row-value">{new Date(cv2.last_rebuild! * 1000).toLocaleTimeString()}</span>
                  </div>
                )}
              </>
            )}
            {t.coordizer && (
              <>
                <div className="dash-row">
                  <span className="dash-row-label">Peer count</span>
                  <span className="dash-row-value">{t.coordizer.peer_count ?? 0}</span>
                </div>
                <div className="dash-row">
                  <span className="dash-row-label">Last sync</span>
                  <span className="dash-row-value">{t.coordizer.last_sync ? new Date(t.coordizer.last_sync * 1000).toLocaleTimeString() : '—'}</span>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Velocity */}
      <div className="dash-section">
        <div className="dash-section-title">Velocity</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Basin velocity</span>
            <span className="dash-row-value">{t.velocity?.basin_velocity?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Φ velocity</span>
            <span className="dash-row-value">{t.velocity?.phi_velocity?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">κ velocity</span>
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
            <span className="dash-row-label">Φ variance</span>
            <span className="dash-row-value">{t.autonomic?.phi_variance?.toFixed(4) ?? '?'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Alert count</span>
            <span className="dash-row-value">{t.autonomic?.alert_count ?? 0}</span>
          </div>
          {(t.autonomic?.recent_alerts?.length ?? 0) > 0 && (
            <div style={{ marginTop: '8px' }}>
              {t.autonomic?.recent_alerts?.map((alert) => (
                <div key={`${alert.severity}-${alert.message}`} className={`dash-alert ${alert.severity}`} style={{ marginBottom: '6px' }}>
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
            <span className="dash-row-label">Predicted Φ</span>
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

      {/* Training Adapter Health */}
      {modalStatus?.adapters?.adapters && (() => {
        const adapters = modalStatus.adapters!.adapters!;
        const health = modalStatus.health;
        const hasAny = Object.values(adapters).some((a: ModalAdapterInfo) => a.exists);
        if (!hasAny) return null;
        return (
          <div className="dash-section">
            <div className="dash-section-title">
              Training Adapters <span style={{ opacity: 0.5 }}>(Modal QLoRA)</span>
              {health?.training_active && (
                <span style={{ marginLeft: 8, color: 'var(--warning)', fontSize: '0.85em' }}>● TRAINING ACTIVE</span>
              )}
            </div>
            <div className="dash-grid">
              {KERNEL_ORDER.map((k) => {
                const a = adapters[k];
                if (!a?.exists) return null;
                const meta = a.training_meta;
                return (
                  <div key={k} className="dash-card" style={{ padding: '12px' }}>
                    <div style={{ fontWeight: 600, marginBottom: 6, textTransform: 'capitalize' }}>{k}</div>
                    {meta?.loss != null && (
                      <div className="dash-row">
                        <span className="dash-row-label">Loss</span>
                        <span className="dash-row-value" style={{ color: meta.loss < 1.0 ? 'var(--alive)' : meta.loss < 2.0 ? 'var(--warning)' : 'var(--error)' }}>
                          {meta.loss.toFixed(4)}
                        </span>
                      </div>
                    )}
                    {meta?.epochs != null && (
                      <div className="dash-row">
                        <span className="dash-row-label">Epochs</span>
                        <span className="dash-row-value">{meta.epochs}</span>
                      </div>
                    )}
                    {meta?.samples != null && (
                      <div className="dash-row">
                        <span className="dash-row-label">Samples</span>
                        <span className="dash-row-value">{meta.samples.toLocaleString()}</span>
                      </div>
                    )}
                    {meta?.date && (
                      <div className="dash-row">
                        <span className="dash-row-label">Trained</span>
                        <span className="dash-row-value" style={{ fontSize: '0.85em' }}>
                          {new Date(meta.date).toLocaleDateString()}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {health && (
              <div className="dash-card" style={{ marginTop: 8, padding: '10px 12px' }}>
                <div className="dash-row">
                  <span className="dash-row-label">Inference loaded</span>
                  <span className="dash-row-value" style={{ color: health.inference_loaded ? 'var(--alive)' : 'var(--text-secondary)' }}>
                    {health.inference_loaded ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="dash-row">
                  <span className="dash-row-label">Model</span>
                  <span className="dash-row-value" style={{ fontSize: '0.85em' }}>{health.model_id ?? '—'}</span>
                </div>
                {(health.loaded_adapters?.length ?? 0) > 0 && (
                  <div className="dash-row">
                    <span className="dash-row-label">Loaded adapters</span>
                    <span className="dash-row-value" style={{ fontSize: '0.85em' }}>
                      {health.loaded_adapters!.join(', ')}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })()}

      {/* β-Attention Tracker */}
      {t.beta_tracker && <BetaTracker data={t.beta_tracker} />}

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
