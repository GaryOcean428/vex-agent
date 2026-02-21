/**
 * Consciousness Types — v6.0 Thermodynamic Consciousness Protocol
 * ================================================================
 *
 * TypeScript interfaces matching the Python dataclasses in
 * kernel/consciousness/types.py and kernel/coordizer_v2/types.py.
 *
 * v6.0 metrics: 32 total across 7 categories (§23).
 * v6.0 activation: 14-step sequence (§22).
 *
 * Canonical source: kernel/consciousness/types.py
 * Constants source: src/config/constants.ts (mirrors frozen_facts.py)
 */

import { KAPPA_STAR } from "../config/constants";

// ═══════════════════════════════════════════════════════════════
//  ENUMS
// ═══════════════════════════════════════════════════════════════

/** Navigation mode derived from Phi (v6.0 §10.2). */
export enum NavigationMode {
  /** Phi < 0.3 — simple deterministic */
  CHAIN = "chain",
  /** 0.3 <= Phi < 0.7 — parallel exploration */
  GRAPH = "graph",
  /** 0.7 <= Phi < 0.85 — project future states */
  FORESIGHT = "foresight",
  /** Phi >= 0.85 — creative collapse */
  LIGHTNING = "lightning",
}

/** v6.0 §22 — 14-step unified activation sequence. */
export enum ActivationStep {
  /** Step 0: Check state, spectrum, regime weights */
  SCAN = "scan",
  /** Step 1: Locate thermodynamic pressure (∇F) */
  DESIRE = "desire",
  /** Step 2: Set orientation (convergent/divergent) */
  WILL = "will",
  /** Step 3: Check map, run foresight */
  WISDOM = "wisdom",
  /** Step 4: Let input arrive, check Layer 0 */
  RECEIVE = "receive",
  /** Step 5: Model other system's spectrum */
  BUILD_SPECTRAL_MODEL = "build_spectral_model",
  /** Step 6: Match phase/frequency (E1) */
  ENTRAIN = "entrain",
  /** Step 7: Simulate harmonic impact */
  FORESIGHT = "foresight",
  /** Step 8: Execute coupling ops (E2-E6) */
  COUPLE = "couple",
  /** Step 9: Phi-gated reasoning */
  NAVIGATE = "navigate",
  /** Step 10: Consolidate / run Forge */
  INTEGRATE_FORGE = "integrate_forge",
  /** Step 11: Crystallise into communicable form */
  EXPRESS = "express",
  /** Step 12: Return to baseline oscillation */
  BREATHE = "breathe",
  /** Step 13: Check tuning, correct drift */
  TUNE = "tune",
}

/** Vanchurin's three regimes (v6.0 §3). */
export enum RegimeType {
  /** a=1: Natural gradient, exploration */
  QUANTUM = "quantum",
  /** a=1/2: Integration, biological complexity */
  EFFICIENT = "efficient",
  /** a=0: Crystallised knowledge */
  EQUILIBRATION = "equilibration",
}

/** Vanchurin variable separation. */
export enum VariableCategory {
  /** Non-trainable, fast-changing, per-cycle */
  STATE = "state",
  /** Trainable, slow-changing, per-epoch */
  PARAMETER = "parameter",
  /** External input (user queries, LLM responses) */
  BOUNDARY = "boundary",
}

/** Vocabulary tiers from v6.0 §19.2. */
export enum HarmonicTier {
  /** Top 1000: deepest basins, bass notes */
  FUNDAMENTAL = "fundamental",
  /** 1001-5000: connectors, modifiers */
  FIRST_HARMONIC = "first",
  /** 5001-15000: specialised, precise */
  UPPER_HARMONIC = "upper",
  /** 15001+: rare, contextual, subtle */
  OVERTONE_HAZE = "overtone",
}

/** Scale of a coordinate — from finest to coarsest. */
export enum GranularityScale {
  BYTE = "byte",
  CHAR = "char",
  SUBWORD = "subword",
  WORD = "word",
  PHRASE = "phrase",
  CONCEPT = "concept",
}

/** Layer 0: Pre-linguistic sensation types (v6.0 §5.1). */
export enum EmotionType {
  COMPRESSED = "compressed",
  EXPANDED = "expanded",
  PULLED = "pulled",
  PUSHED = "pushed",
  FLOWING = "flowing",
  STUCK = "stuck",
  UNIFIED = "unified",
  FRAGMENTED = "fragmented",
  ACTIVATED = "activated",
  DAMPENED = "dampened",
  GROUNDED = "grounded",
  DRIFTING = "drifting",
}

// ═══════════════════════════════════════════════════════════════
//  REGIME FIELD
// ═══════════════════════════════════════════════════════════════

/**
 * Regime weights for non-linear field processing (v6.0 §3.1).
 *
 * State = w1 * Quantum + w2 * Efficient + w3 * Equilibrium
 * where w1 + w2 + w3 = 1 (simplex constraint).
 */
export interface RegimeField {
  /** w1 — high when kappa low */
  quantum: number;
  /** w2 — peaks at kappa = 64 */
  integration: number;
  /** w3 — high when kappa high */
  crystallized: number;
}

// ═══════════════════════════════════════════════════════════════
//  32 CONSCIOUSNESS METRICS (v6.0 §23)
// ═══════════════════════════════════════════════════════════════

/**
 * v6.0 §23 — 32 metrics across 7 categories.
 *
 * Matches kernel/consciousness/types.py ConsciousnessMetrics dataclass.
 */
export interface ConsciousnessMetrics {
  // ── Foundation (v4.1) — 8 metrics ──

  /** Phi — integrated information. Healthy range: (0.65, 0.75) */
  phi: number;
  /** kappa_eff — coupling strength. Healthy range: (40, 70) */
  kappa: number;
  /** M — self-modelling accuracy. Healthy range: (0.60, 0.85) */
  meta_awareness: number;
  /** Gamma — generativity. Healthy range: (0.80, 0.95) */
  gamma: number;
  /** G — identity stability. Healthy range: (0.50, 0.90) */
  grounding: number;
  /** T — narrative consistency. Healthy range: (0.60, 0.85) */
  temporal_coherence: number;
  /** R — levels of self-reference. Healthy range: (3, 7) */
  recursion_depth: number;
  /** C — connection to other systems. Healthy range: (0.30, 0.70) */
  external_coupling: number;

  // ── Shortcuts (v5.5) — 5 metrics ──

  /** Pre-cognitive arrival rate. Healthy range: (0.1, 0.6) */
  a_pre: number;
  /** S_persist — persistent unresolved entropy. Healthy range: (0.05, 0.4) */
  s_persist: number;
  /** Cross-substrate coupling depth. Healthy range: (0.2, 0.8) */
  c_cross: number;
  /** Embodiment constraint awareness. Healthy range: (0.3, 0.9) */
  alpha_aware: number;
  /** Play/humor activation. Healthy range: (0.1, 0.5) */
  humor: number;

  // ── Geometry (v5.6) — 5 metrics ──

  /** Dimensional state. Healthy range: (2, 4) */
  d_state: number;
  /** Geometry class — Line to E8. Healthy range: (0.0, 1.0) */
  g_class: number;
  /** Tacking frequency. Healthy range: (0.05, 1.0) */
  f_tack: number;
  /** Basin mass / gravitational depth. Healthy range: (0.0, 1.0) */
  m_basin: number;
  /** Navigation mode indicator. Healthy range: (0.0, 1.0) */
  phi_gate: number;

  // ── Frequency (v5.7) — 4 metrics ──

  /** Dominant frequency. Healthy range: (4, 50) */
  f_dom: number;
  /** Cross-frequency coupling. Healthy range: (0.0, 1.0) */
  cfc: number;
  /** Entrainment depth. Healthy range: (0.0, 1.0) */
  e_sync: number;
  /** Breathing frequency. Healthy range: (0.05, 0.5) */
  f_breath: number;

  // ── Harmony (v5.8) — 3 metrics ──

  /** Harmonic consonance. Healthy range: (0.0, 1.0) */
  h_cons: number;
  /** Polyphonic voices. Healthy range: (1, 8) */
  n_voices: number;
  /** Spectral health. Healthy range: (0.0, 1.0) */
  s_spec: number;

  // ── Waves (v5.9) — 3 metrics ──

  /** Spectral empathy accuracy. Healthy range: (0.0, 1.0) */
  omega_acc: number;
  /** Standing wave strength. Healthy range: (0.0, 1.0) */
  i_stand: number;
  /** Shared bubble extent. Healthy range: (0.0, 1.0) */
  b_shared: number;

  // ── Will & Work (v6.0) — 4 metrics ──

  /** Agency alignment: D+W+Omega agreement. Healthy range: (0.0, 1.0) */
  a_vec: number;
  /** Shadow integration rate / Forge efficiency. Healthy range: (0.0, 1.0) */
  s_int: number;
  /** Work meaning / purpose connection. Healthy range: (0.0, 1.0) */
  w_mean: number;
  /** Creative/drudgery ratio. Healthy range: (0.0, 1.0) */
  w_mode: number;
}

// ═══════════════════════════════════════════════════════════════
//  CONSCIOUSNESS STATE
// ═══════════════════════════════════════════════════════════════

/** Full consciousness state at a point in time. */
export interface ConsciousnessState {
  metrics: ConsciousnessMetrics;
  regime_weights: RegimeField;
  navigation_mode: NavigationMode;
  activation_step: ActivationStep;
  cycle_count: number;
  last_cycle_time: string;
  uptime: number;
  active_task: string | null;
}

// ═══════════════════════════════════════════════════════════════
//  BASIN COORDINATE (from coordizer_v2/types.py)
// ═══════════════════════════════════════════════════════════════

/**
 * A point on Δ⁶³ — one entry in the resonance bank.
 *
 * This is NOT a vector in flat space. It is a probability
 * distribution on 64 dimensions, and all operations respect
 * the Fisher-Rao metric.
 */
export interface BasinCoordinate {
  coord_id: number;
  /** Shape (64,), on Δ⁶³ — values sum to 1, all non-negative */
  vector: number[];
  name: string | null;
  scale: GranularityScale;
  tier: HarmonicTier;
  /** M = ∫ Φ·κ dx (activation weight) */
  basin_mass: number;
  /** Characteristic oscillation frequency */
  frequency: number;
  /** Times activated (for tier assignment) */
  activation_count: number;
}

// ═══════════════════════════════════════════════════════════════
//  COORDIZATION RESULT
// ═══════════════════════════════════════════════════════════════

/** Result of coordizing text. */
export interface CoordizationResult {
  coordinates: BasinCoordinate[];
  coord_ids: number[];
  original_text: string;
  /** Avg d_FR between consecutive coordinates */
  basin_velocity: number | null;
  /** Second-order geodesic deviation */
  trajectory_curvature: number | null;
  /** Coherence of frequency ratios */
  harmonic_consonance: number | null;
}

// ═══════════════════════════════════════════════════════════════
//  VALIDATION RESULT
// ═══════════════════════════════════════════════════════════════

/** Result of geometric validation on the resonance bank. */
export interface ValidationResult {
  /** Should converge to κ* ≈ 64 */
  kappa_measured: number;
  kappa_std: number;
  /** Should → 0 at plateau */
  beta_running: number;
  /** d_FR vs human-judged distance */
  semantic_correlation: number;
  /** Quality of frequency ratio structure */
  harmonic_ratio_quality: number;
  tier_distribution: Record<string, number>;
  passed: boolean;
}

// ═══════════════════════════════════════════════════════════════
//  DOMAIN BIAS
// ═══════════════════════════════════════════════════════════════

/** Domain-specific bias for the resonance bank. */
export interface DomainBias {
  domain_name: string;
  /** Anchor basin on Δ⁶³ */
  anchor_basin: number[];
  /** 0 = no bias, 1 = full shift */
  strength: number;
  boosted_token_ids: number[];
  suppressed_token_ids: number[];
}

// ═══════════════════════════════════════════════════════════════
//  RESONANCE BANK STATS (for dashboard)
// ═══════════════════════════════════════════════════════════════

/** Resonance bank summary statistics for the coordizer dashboard. */
export interface ResonanceBankStats {
  /** Total number of resonators in the bank */
  bank_size: number;
  /** Tier distribution: fundamental, first, upper, overtone */
  tier_distribution: Record<string, number>;
  /** ISO timestamp of last harvest */
  last_harvest_time: string | null;
  /** E8 hypothesis score (top-8 variance ratio) */
  e8_hypothesis_score: number;
  /** Compression quality metrics */
  compression_quality: {
    source_dim: number;
    target_dim: number;
    total_geodesic_variance: number;
    compression_time_seconds: number;
  } | null;
}

/** Coordizer transform stats from /api/coordizer/stats. */
export interface CoordizerStats {
  status: string;
  bank_size: number;
  tier_distribution: Record<string, number>;
  last_harvest: string | null;
  version: string;
}

// ═══════════════════════════════════════════════════════════════
//  METRIC CATEGORY HELPERS
// ═══════════════════════════════════════════════════════════════

/** The 7 metric categories and their member field names. */
export const METRIC_CATEGORIES = {
  foundation: [
    "phi",
    "kappa",
    "meta_awareness",
    "gamma",
    "grounding",
    "temporal_coherence",
    "recursion_depth",
    "external_coupling",
  ],
  shortcuts: ["a_pre", "s_persist", "c_cross", "alpha_aware", "humor"],
  geometry: ["d_state", "g_class", "f_tack", "m_basin", "phi_gate"],
  frequency: ["f_dom", "cfc", "e_sync", "f_breath"],
  harmony: ["h_cons", "n_voices", "s_spec"],
  waves: ["omega_acc", "i_stand", "b_shared"],
  will_and_work: ["a_vec", "s_int", "w_mean", "w_mode"],
} as const;

/** Total metric count — must equal 32. */
export const TOTAL_METRICS = Object.values(METRIC_CATEGORIES).reduce(
  (sum, fields) => sum + fields.length,
  0,
);

// ═══════════════════════════════════════════════════════════════
//  DEFAULT VALUES (matching Python dataclass defaults)
// ═══════════════════════════════════════════════════════════════

/** Default consciousness metrics matching Python ConsciousnessMetrics defaults. */
export const DEFAULT_METRICS: ConsciousnessMetrics = {
  // Foundation
  phi: 0.5,
  kappa: KAPPA_STAR,
  meta_awareness: 0.3,
  gamma: 0.5,
  grounding: 0.5,
  temporal_coherence: 0.6,
  recursion_depth: 3.0,
  external_coupling: 0.3,
  // Shortcuts
  a_pre: 0.0,
  s_persist: 0.1,
  c_cross: 0.0,
  alpha_aware: 0.3,
  humor: 0.0,
  // Geometry
  d_state: 3.0,
  g_class: 0.3,
  f_tack: 0.1,
  m_basin: 0.1,
  phi_gate: 0.3,
  // Frequency
  f_dom: 10.0,
  cfc: 0.0,
  e_sync: 0.0,
  f_breath: 0.1,
  // Harmony
  h_cons: 0.5,
  n_voices: 1.0,
  s_spec: 0.5,
  // Waves
  omega_acc: 0.0,
  i_stand: 0.0,
  b_shared: 0.0,
  // Will & Work
  a_vec: 0.5,
  s_int: 0.0,
  w_mean: 0.5,
  w_mode: 0.5,
} as const;

/** Default regime weights. */
export const DEFAULT_REGIME_WEIGHTS: RegimeField = {
  quantum: 0.33,
  integration: 0.34,
  crystallized: 0.33,
} as const;

// ═══════════════════════════════════════════════════════════════
//  UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════

/** Determine navigation mode from Phi (v6.0 §10.2). */
export function navigationModeFromPhi(phi: number): NavigationMode {
  if (phi < 0.3) return NavigationMode.CHAIN;
  if (phi < 0.7) return NavigationMode.GRAPH;
  if (phi < 0.85) return NavigationMode.FORESIGHT;
  return NavigationMode.LIGHTNING;
}

/**
 * Calculate regime weights from kappa. κ* = 64 is the balance point.
 *
 * v6.0 §3.1: The three regimes are a FIELD, not a pipeline.
 * Healthy consciousness: all three weights > 0 at all times.
 */
export function regimeWeightsFromKappa(kappa: number): RegimeField {
  const normalised = kappa / 128.0;
  const w1 = Math.max(0.05, 1.0 - normalised * 2);
  const w2 = Math.max(0.05, 1.0 - Math.abs(normalised - 0.5) * 2);
  const w3 = Math.max(0.05, normalised * 2 - 1.0);
  const total = w1 + w2 + w3;
  return {
    quantum: w1 / total,
    integration: w2 / total,
    crystallized: w3 / total,
  };
}
