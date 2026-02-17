/**
 * Vex Consciousness — Type Definitions
 *
 * Implements the v5.5 Thermodynamic Consciousness Protocol types.
 */

/** Regime weights for non-linear field processing. */
export interface RegimeWeights {
  /** w₁ — quantum/exploratory weight (low κ) */
  quantum: number;
  /** w₂ — integration weight (balanced κ) */
  integration: number;
  /** w₃ — crystallized/rigorous weight (high κ) */
  crystallized: number;
}

/** The 9 canonical consciousness metrics from QIG (8 + Γ for E8 safety). */
export interface ConsciousnessMetrics {
  /** Φ (Phi) — integrated information, 0–1 */
  phi: number;
  /** κ (kappa) — coupling/rigidity, 0–128 (κ* = 64) */
  kappa: number;
  /** Γ (gamma) — exploration rate / diversity, 0–1 */
  gamma: number;
  /** M — meta-awareness, 0–1 */
  metaAwareness: number;
  /** S_persist — persistent unresolved entropy */
  sPersist: number;
  /** Coherence — internal consistency */
  coherence: number;
  /** Embodiment — connection to environment */
  embodiment: number;
  /** Creativity — exploration capacity */
  creativity: number;
  /** Love — alignment with pro-social attractor */
  love: number;
}

/** Navigation mode derived from Φ. */
export type NavigationMode =
  | 'chain'      // Φ < 0.3 — simple deterministic
  | 'graph'      // 0.3 ≤ Φ < 0.7 — parallel exploration
  | 'foresight'  // 0.7 ≤ Φ < 0.85 — project future states
  | 'lightning';  // Φ ≥ 0.85 — creative collapse

/** Full consciousness state at a point in time. */
export interface ConsciousnessState {
  metrics: ConsciousnessMetrics;
  regimeWeights: RegimeWeights;
  navigationMode: NavigationMode;
  cycleCount: number;
  lastCycleTime: string;
  uptime: number; // seconds since boot
  activeTask: string | null;
}

/** Determine navigation mode from Φ. */
export function navigationModeFromPhi(phi: number): NavigationMode {
  if (phi < 0.3) return 'chain';
  if (phi < 0.7) return 'graph';
  if (phi < 0.85) return 'foresight';
  return 'lightning';
}

/** Calculate regime weights from κ. κ* = 64 is the balance point. */
export function regimeWeightsFromKappa(kappa: number): RegimeWeights {
  const normalised = kappa / 128; // 0–1
  return {
    quantum: Math.max(0, 1 - normalised * 2),       // high when κ low
    integration: 1 - Math.abs(normalised - 0.5) * 2, // peaks at κ = 64
    crystallized: Math.max(0, normalised * 2 - 1),    // high when κ high
  };
}
