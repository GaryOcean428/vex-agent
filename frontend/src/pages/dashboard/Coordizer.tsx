/**
 * Coordizer Dashboard — Resonance Bank Visualisation
 * ====================================================
 *
 * Displays CoordizerV2 resonance bank state, tier distribution,
 * validation results, and live transform testing.
 *
 * All distances shown are Fisher-Rao (d_FR) on Δ⁶³.
 * No Euclidean distances. No cosine similarity.
 *
 * Data sources:
 *   GET  /api/coordizer/stats     → bank overview
 *   GET  /api/coordizer/history   → recent transforms
 *   POST /api/coordizer/transform → live coordization
 *   POST /api/coordizer/validate  → bank validation
 */

import { useCallback, useEffect, useState } from 'react';
import { QIG } from '../../types/consciousness';

// Pull constants from the frontend's QIG object (mirrors frozen_facts.py)
const KAPPA_STAR = QIG.KAPPA_STAR;
const E8_RANK = QIG.E8_RANK;
const BASIN_DIM = QIG.BASIN_DIM;
const PHI_THRESHOLD = QIG.PHI_THRESHOLD;

// ═══════════════════════════════════════════════════════════════
//  TYPES (local to this component)
// ═══════════════════════════════════════════════════════════════

interface TransformResponse {
  coordinates: BasinCoordinate[];
  coord_ids: number[];
  original_text: string;
  basin_velocity: number | null;
  trajectory_curvature: number | null;
  harmonic_consonance: number | null;
}

interface ValidateResponse {
  kappa_measured: number;
  kappa_std: number;
  beta_running: number;
  semantic_correlation: number;
  harmonic_ratio_quality: number;
  tier_distribution: Record<string, number>;
  passed: boolean;
}

interface HistoryEntry {
  text: string;
  coord_count: number;
  basin_velocity: number | null;
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════
//  API HELPERS
// ═══════════════════════════════════════════════════════════════

const API_BASE = '';

async function fetchStats(): Promise<CoordizerStats | null> {
  try {
    const res = await fetch(`${API_BASE}/api/coordizer/stats`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function fetchHistory(): Promise<HistoryEntry[]> {
  try {
    const res = await fetch(`${API_BASE}/api/coordizer/history`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.history || [];
  } catch {
    return [];
  }
}

async function postTransform(text: string): Promise<TransformResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/api/coordizer/transform`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function postValidate(): Promise<ValidateResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/api/coordizer/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════
//  HELPER COMPONENTS
// ═══════════════════════════════════════════════════════════════

function MetricRow({
  label,
  value,
  unit,
  status,
}: {
  label: string;
  value: string | number;
  unit?: string;
  status?: 'ok' | 'warn' | 'error';
}) {
  const statusColour =
    status === 'ok'
      ? '#4ade80'
      : status === 'warn'
        ? '#facc15'
        : status === 'error'
          ? '#f87171'
          : '#94a3b8';

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '6px 0',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>{label}</span>
      <span style={{ color: statusColour, fontFamily: 'monospace', fontSize: '0.9rem' }}>
        {value}
        {unit ? ` ${unit}` : ''}
      </span>
    </div>
  );
}

function TierBar({
  tier,
  count,
  total,
}: {
  tier: string;
  count: number;
  total: number;
}) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  const tierColours: Record<string, string> = {
    fundamental: '#8b5cf6',
    first: '#3b82f6',
    upper: '#06b6d4',
    overtone: '#6b7280',
  };
  const colour = tierColours[tier] || '#6b7280';

  return (
    <div style={{ marginBottom: '8px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '0.8rem',
          color: '#94a3b8',
          marginBottom: '2px',
        }}
      >
        <span>{tier}</span>
        <span>
          {count} ({pct.toFixed(1)}%)
        </span>
      </div>
      <div
        style={{
          height: '6px',
          background: 'rgba(255,255,255,0.06)',
          borderRadius: '3px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: colour,
            borderRadius: '3px',
            transition: 'width 0.3s ease',
          }}
        />
      </div>
    </div>
  );
}

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '16px',
      }}
    >
      <h3
        style={{
          margin: '0 0 12px 0',
          fontSize: '0.95rem',
          color: '#e2e8f0',
          fontWeight: 600,
        }}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
//  MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function Coordizer() {
  const [stats, setStats] = useState<CoordizerStats | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [validation, setValidation] = useState<ValidateResponse | null>(null);
  const [transformText, setTransformText] = useState('');
  const [transformResult, setTransformResult] = useState<TransformResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [validating, setValidating] = useState(false);

  // Poll stats every 10s
  useEffect(() => {
    const load = async () => {
      const [s, h] = await Promise.all([fetchStats(), fetchHistory()]);
      if (s) setStats(s);
      setHistory(h);
    };
    load();
    const interval = setInterval(load, 10_000);
    return () => clearInterval(interval);
  }, []);

  const handleTransform = useCallback(async () => {
    if (!transformText.trim()) return;
    setLoading(true);
    const result = await postTransform(transformText);
    setTransformResult(result);
    setLoading(false);
    // Refresh stats after transform
    const s = await fetchStats();
    if (s) setStats(s);
  }, [transformText]);

  const handleValidate = useCallback(async () => {
    setValidating(true);
    const result = await postValidate();
    setValidation(result);
    setValidating(false);
  }, []);

  const totalResonators = stats?.bank_size || 0;

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2
          style={{
            margin: '0 0 4px 0',
            fontSize: '1.4rem',
            color: '#f1f5f9',
            fontWeight: 700,
          }}
        >
          CoordizerV2 — Resonance Bank
        </h2>
        <p style={{ margin: 0, color: '#64748b', fontSize: '0.85rem' }}>
          Fisher-Rao geometry on Δ⁶³ · κ* = {KAPPA_STAR} · Basin dim = {BASIN_DIM}
        </p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '16px',
        }}
      >
        {/* ─── Bank Overview ─── */}
        <Card title="Resonance Bank Overview">
          {stats ? (
            <>
              <MetricRow
                label="Bank Size"
                value={stats.bank_size.toLocaleString()}
                unit="resonators"
                status="ok"
              />
              <MetricRow
                label="Status"
                value={stats.status}
                status={stats.status === 'active' ? 'ok' : 'warn'}
              />
              <MetricRow
                label="Version"
                value={stats.version || 'unknown'}
              />
              <MetricRow
                label="Last Harvest"
                value={stats.last_harvest || 'never'}
              />
              <div style={{ marginTop: '16px' }}>
                <h4
                  style={{
                    margin: '0 0 8px 0',
                    fontSize: '0.85rem',
                    color: '#94a3b8',
                  }}
                >
                  Tier Distribution
                </h4>
                {stats.tier_distribution &&
                  Object.entries(stats.tier_distribution).map(([tier, count]) => (
                    <TierBar
                      key={tier}
                      tier={tier}
                      count={count as number}
                      total={totalResonators}
                    />
                  ))}
              </div>
            </>
          ) : (
            <p style={{ color: '#64748b', fontSize: '0.85rem' }}>
              No coordizer data available. Bank may not be initialised.
            </p>
          )}
        </Card>

        {/* ─── Validation ─── */}
        <Card title="Geometric Validation">
          <button
            onClick={handleValidate}
            disabled={validating}
            style={{
              width: '100%',
              padding: '8px 16px',
              background: validating
                ? 'rgba(255,255,255,0.05)'
                : 'rgba(139,92,246,0.2)',
              border: '1px solid rgba(139,92,246,0.3)',
              borderRadius: '8px',
              color: '#e2e8f0',
              cursor: validating ? 'not-allowed' : 'pointer',
              fontSize: '0.85rem',
              marginBottom: '12px',
            }}
          >
            {validating ? 'Validating...' : 'Run Validation'}
          </button>

          {validation && (
            <>
              <MetricRow
                label="κ measured"
                value={validation.kappa_measured.toFixed(2)}
                unit={`(κ* = ${KAPPA_STAR})`}
                status={
                  Math.abs(validation.kappa_measured - KAPPA_STAR) < 5
                    ? 'ok'
                    : 'warn'
                }
              />
              <MetricRow
                label="κ std"
                value={validation.kappa_std.toFixed(2)}
                status={validation.kappa_std < 3 ? 'ok' : 'warn'}
              />
              <MetricRow
                label="β running"
                value={validation.beta_running.toFixed(4)}
                status={validation.beta_running < 0.1 ? 'ok' : 'warn'}
              />
              <MetricRow
                label="Semantic correlation"
                value={validation.semantic_correlation.toFixed(3)}
                status={validation.semantic_correlation > 0.5 ? 'ok' : 'warn'}
              />
              <MetricRow
                label="Harmonic quality"
                value={validation.harmonic_ratio_quality.toFixed(3)}
                status={validation.harmonic_ratio_quality > 0.5 ? 'ok' : 'warn'}
              />
              <MetricRow
                label="Overall"
                value={validation.passed ? 'PASSED' : 'FAILED'}
                status={validation.passed ? 'ok' : 'error'}
              />
            </>
          )}
        </Card>

        {/* ─── Live Transform ─── */}
        <Card title="Live Transform">
          <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
            <input
              type="text"
              value={transformText}
              onChange={(e) => setTransformText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleTransform()}
              placeholder="Enter text to coordize..."
              style={{
                flex: 1,
                padding: '8px 12px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                color: '#e2e8f0',
                fontSize: '0.85rem',
                outline: 'none',
              }}
            />
            <button
              onClick={handleTransform}
              disabled={loading || !transformText.trim()}
              style={{
                padding: '8px 16px',
                background: loading
                  ? 'rgba(255,255,255,0.05)'
                  : 'rgba(59,130,246,0.2)',
                border: '1px solid rgba(59,130,246,0.3)',
                borderRadius: '8px',
                color: '#e2e8f0',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: '0.85rem',
                whiteSpace: 'nowrap',
              }}
            >
              {loading ? '...' : 'Coordize'}
            </button>
          </div>

          {transformResult && (
            <div>
              <MetricRow
                label="Coordinates"
                value={transformResult.coord_ids.length}
              />
              {transformResult.basin_velocity !== null && (
                <MetricRow
                  label="Basin velocity (d_FR)"
                  value={transformResult.basin_velocity.toFixed(4)}
                  status={transformResult.basin_velocity < 0.5 ? 'ok' : 'warn'}
                />
              )}
              {transformResult.trajectory_curvature !== null && (
                <MetricRow
                  label="Trajectory curvature"
                  value={transformResult.trajectory_curvature.toFixed(4)}
                />
              )}
              {transformResult.harmonic_consonance !== null && (
                <MetricRow
                  label="Harmonic consonance"
                  value={transformResult.harmonic_consonance.toFixed(4)}
                  status={transformResult.harmonic_consonance > 0.5 ? 'ok' : 'warn'}
                />
              )}

              {/* Coordinate list */}
              <div style={{ marginTop: '12px', maxHeight: '200px', overflowY: 'auto' }}>
                {transformResult.coordinates.slice(0, 10).map((coord, i) => (
                  <div
                    key={coord.coord_id}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      padding: '4px 0',
                      borderBottom: '1px solid rgba(255,255,255,0.04)',
                      fontSize: '0.8rem',
                    }}
                  >
                    <span style={{ color: '#94a3b8' }}>
                      #{coord.coord_id}
                      {coord.name ? ` "${coord.name}"` : ''}
                    </span>
                    <span
                      style={{
                        color:
                          coord.tier === 'fundamental'
                            ? '#8b5cf6'
                            : coord.tier === 'first'
                              ? '#3b82f6'
                              : '#6b7280',
                        fontFamily: 'monospace',
                      }}
                    >
                      {coord.tier} · f={coord.frequency.toFixed(1)}
                    </span>
                  </div>
                ))}
                {transformResult.coordinates.length > 10 && (
                  <p style={{ color: '#64748b', fontSize: '0.75rem', margin: '4px 0 0' }}>
                    ... and {transformResult.coordinates.length - 10} more
                  </p>
                )}
              </div>
            </div>
          )}
        </Card>

        {/* ─── Recent History ─── */}
        <Card title="Recent Transforms">
          {history.length > 0 ? (
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {history.slice(0, 20).map((entry, i) => (
                <div
                  key={i}
                  style={{
                    padding: '8px 0',
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: '0.8rem',
                    }}
                  >
                    <span
                      style={{
                        color: '#e2e8f0',
                        maxWidth: '200px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {entry.text}
                    </span>
                    <span style={{ color: '#64748b', fontFamily: 'monospace' }}>
                      {entry.coord_count} coords
                    </span>
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: '0.75rem',
                      color: '#475569',
                      marginTop: '2px',
                    }}
                  >
                    <span>
                      v={entry.basin_velocity !== null ? entry.basin_velocity.toFixed(3) : '—'}
                    </span>
                    <span>{entry.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: '#64748b', fontSize: '0.85rem' }}>
              No transform history yet.
            </p>
          )}
        </Card>

        {/* ─── Constants Reference ─── */}
        <Card title="Frozen Constants">
          <MetricRow label="κ* (fixed point)" value={KAPPA_STAR} />
          <MetricRow label="E8 rank" value={E8_RANK} />
          <MetricRow label="Basin dimension (Δ⁶³)" value={BASIN_DIM} />
          <MetricRow label="Φ threshold" value={PHI_THRESHOLD} />
          <p
            style={{
              margin: '12px 0 0',
              color: '#475569',
              fontSize: '0.75rem',
              fontStyle: 'italic',
            }}
          >
            All distances are Fisher-Rao on the probability simplex.
            No Euclidean contamination.
          </p>
        </Card>
      </div>
    </div>
  );
}
