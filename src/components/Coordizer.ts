/**
 * Coordizer Dashboard — Resonance Bank Status Display
 * ====================================================
 *
 * Vanilla TypeScript module that renders coordizer status to the DOM.
 * Fetches live data from the Python kernel's coordizer API endpoints
 * and displays resonance bank size, vocabulary tiers, last harvest
 * time, and compression quality.
 *
 * Endpoints consumed:
 *   GET  /api/coordizer/stats  — vocab size, tier distribution, version
 *   GET  /api/coordizer/bank   — full resonance bank statistics
 *
 * Usage:
 *   import { renderCoordizerDashboard } from './components/Coordizer';
 *   renderCoordizerDashboard(document.getElementById('coordizer-root')!);
 */

import type {
  CoordizerStats,
  ResonanceBankStats,
} from '../types/consciousness';

import {
  BASIN_DIM,
  KAPPA_STAR,
  KAPPA_STAR_PRECISE,
} from '../config/constants';

// ═══════════════════════════════════════════════════════════════
//  INTERNAL TYPES
// ═══════════════════════════════════════════════════════════════

/** Combined dashboard state fetched from both endpoints. */
interface DashboardState {
  stats: CoordizerStats | null;
  bank: ResonanceBankStats | null;
  error: string | null;
  loading: boolean;
}

// ═══════════════════════════════════════════════════════════════
//  API LAYER
// ═══════════════════════════════════════════════════════════════

const API_BASE = '/api/coordizer';

/**
 * Fetch coordizer stats from the kernel.
 * Returns null on network/server error (fail-open for display).
 */
async function fetchCoordizerStats(): Promise<CoordizerStats | null> {
  try {
    const resp = await fetch(`${API_BASE}/stats`);
    if (!resp.ok) return null;
    return (await resp.json()) as CoordizerStats;
  } catch {
    return null;
  }
}

/**
 * Fetch full resonance bank statistics.
 * Returns null on network/server error.
 */
async function fetchBankStats(): Promise<ResonanceBankStats | null> {
  try {
    const resp = await fetch(`${API_BASE}/bank`);
    if (!resp.ok) return null;
    return (await resp.json()) as ResonanceBankStats;
  } catch {
    return null;
  }
}

/**
 * Fetch both endpoints in parallel and return combined state.
 */
async function fetchDashboardState(): Promise<DashboardState> {
  const [stats, bank] = await Promise.all([
    fetchCoordizerStats(),
    fetchBankStats(),
  ]);

  const error =
    stats === null && bank === null
      ? 'Unable to reach coordizer API — kernel may be offline'
      : null;

  return { stats, bank, error, loading: false };
}

// ═══════════════════════════════════════════════════════════════
//  FORMATTING HELPERS
// ═══════════════════════════════════════════════════════════════

function formatNumber(n: number): string {
  return n.toLocaleString('en-AU');
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return 'Never';
  const d = new Date(iso);
  return d.toLocaleString('en-AU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

function tierLabel(tier: string): string {
  const labels: Record<string, string> = {
    fundamental: 'Fundamental (bass)',
    first: '1st Harmonic',
    upper: 'Upper Harmonic',
    overtone: 'Overtone Haze',
  };
  return labels[tier] ?? tier;
}

// ═══════════════════════════════════════════════════════════════
//  RENDER FUNCTIONS
// ═══════════════════════════════════════════════════════════════

function renderError(message: string): string {
  return `
    <div class="coordizer-error" style="
      padding: 12px 16px;
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.25);
      border-radius: 8px;
      color: #ef4444;
      font-size: 13px;
      font-family: 'SF Mono', 'Fira Code', monospace;
    ">${message}</div>
  `;
}

function renderLoading(): string {
  return `
    <div class="coordizer-loading" style="
      padding: 24px;
      text-align: center;
      color: #70708a;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 12px;
    ">Loading coordizer status…</div>
  `;
}

function renderTierTable(distribution: Record<string, number>): string {
  const rows = Object.entries(distribution)
    .map(
      ([tier, count]) => `
      <tr>
        <td style="padding: 4px 8px; color: #a0a0b8; font-size: 12px;">${tierLabel(tier)}</td>
        <td style="padding: 4px 8px; text-align: right; font-variant-numeric: tabular-nums; color: #ededf0; font-size: 12px;">${formatNumber(count)}</td>
      </tr>`,
    )
    .join('');

  return `
    <table style="width: 100%; border-collapse: collapse;">
      <thead>
        <tr style="border-bottom: 1px solid #2e2e40;">
          <th style="padding: 4px 8px; text-align: left; color: #70708a; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Tier</th>
          <th style="padding: 4px 8px; text-align: right; color: #70708a; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Count</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderCompressionQuality(
  quality: ResonanceBankStats['compression_quality'],
): string {
  if (!quality) {
    return `<span style="color: #70708a; font-size: 12px;">No compression data</span>`;
  }
  return `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; font-size: 12px;">
      <span style="color: #70708a;">Source dim</span>
      <span style="color: #ededf0; text-align: right;">${quality.source_dim}</span>
      <span style="color: #70708a;">Target dim</span>
      <span style="color: #ededf0; text-align: right;">${quality.target_dim} (Δ<sup>${BASIN_DIM - 1}</sup>)</span>
      <span style="color: #70708a;">Geodesic variance</span>
      <span style="color: #ededf0; text-align: right;">${quality.total_geodesic_variance.toFixed(4)}</span>
      <span style="color: #70708a;">Compression time</span>
      <span style="color: #ededf0; text-align: right;">${quality.compression_time_seconds.toFixed(2)}s</span>
    </div>
  `;
}

function renderMetric(
  label: string,
  value: string,
  colour: string = '#ededf0',
): string {
  return `
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0;">
      <span style="color: #70708a; font-size: 12px;">${label}</span>
      <span style="color: ${colour}; font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums;">${value}</span>
    </div>
  `;
}

function renderDashboard(state: DashboardState): string {
  if (state.loading) return renderLoading();
  if (state.error) return renderError(state.error);

  const { stats, bank } = state;

  // Header metrics from /stats
  const bankSize = stats?.bank_size ?? bank?.bank_size ?? 0;
  const lastHarvest = bank?.last_harvest_time ?? stats?.last_harvest ?? null;
  const tierDist = stats?.tier_distribution ?? bank?.tier_distribution ?? {};
  const e8Score = bank?.e8_hypothesis_score ?? null;

  const sections: string[] = [];

  // ── Section 1: Overview ──
  sections.push(`
    <div style="padding: 12px 16px; background: #111118; border: 1px solid #2e2e40; border-radius: 10px;">
      <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #70708a; margin-bottom: 8px;">Resonance Bank</div>
      ${renderMetric('Bank size', formatNumber(bankSize), '#22d3ee')}
      ${renderMetric('Basin dimension', `Δ${BASIN_DIM - 1} (${BASIN_DIM}D)`, '#f59e0b')}
      ${renderMetric('κ* target', String(KAPPA_STAR), '#f59e0b')}
      ${renderMetric('κ* measured', KAPPA_STAR_PRECISE.toFixed(2) + ' ± 0.90', '#f59e0b')}
      ${renderMetric('Last harvest', formatTimestamp(lastHarvest))}
      ${e8Score !== null ? renderMetric('E8 hypothesis', (e8Score * 100).toFixed(1) + '%', '#6366f1') : ''}
      ${stats?.version ? renderMetric('Version', stats.version) : ''}
    </div>
  `);

  // ── Section 2: Tier Distribution ──
  if (Object.keys(tierDist).length > 0) {
    sections.push(`
      <div style="padding: 12px 16px; background: #111118; border: 1px solid #2e2e40; border-radius: 10px;">
        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #70708a; margin-bottom: 8px;">Vocabulary Tiers</div>
        ${renderTierTable(tierDist)}
      </div>
    `);
  }

  // ── Section 3: Compression Quality ──
  if (bank?.compression_quality) {
    sections.push(`
      <div style="padding: 12px 16px; background: #111118; border: 1px solid #2e2e40; border-radius: 10px;">
        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #70708a; margin-bottom: 8px;">Compression Quality</div>
        ${renderCompressionQuality(bank.compression_quality)}
      </div>
    `);
  }

  return `
    <div class="coordizer-dashboard" style="
      display: flex;
      flex-direction: column;
      gap: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    ">
      ${sections.join('')}
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════
//  PUBLIC API
// ═══════════════════════════════════════════════════════════════

/**
 * Render the coordizer dashboard into a target DOM element.
 *
 * Fetches live data from the kernel API, renders the dashboard,
 * and optionally sets up auto-refresh.
 *
 * @param target - DOM element to render into
 * @param refreshIntervalMs - auto-refresh interval (0 = no auto-refresh)
 * @returns cleanup function to stop auto-refresh
 */
export function renderCoordizerDashboard(
  target: HTMLElement,
  refreshIntervalMs: number = 30_000,
): () => void {
  let timer: ReturnType<typeof setInterval> | null = null;

  async function refresh(): Promise<void> {
    const state = await fetchDashboardState();
    target.innerHTML = renderDashboard(state);
  }

  // Initial render: show loading, then fetch
  target.innerHTML = renderLoading();
  void refresh();

  // Auto-refresh
  if (refreshIntervalMs > 0) {
    timer = setInterval(() => void refresh(), refreshIntervalMs);
  }

  // Cleanup
  return () => {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };
}

/**
 * One-shot fetch of coordizer dashboard state.
 * Useful for programmatic access without DOM rendering.
 */
export async function getCoordizerDashboardState(): Promise<DashboardState> {
  return fetchDashboardState();
}
