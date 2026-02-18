// ═══════════════════════════════════════
//  Consciousness Metrics (from /state)
// ═══════════════════════════════════════

export interface ConsciousnessMetrics {
  phi: number;
  kappa: number;
  gamma: number;
  meta_awareness: number;
  love: number;
}

export interface RegimeWeights {
  quantum: number;
  integration: number;
  crystallized: number;
}

// Backend navigation modes: chain → graph → foresight → lightning
export type NavigationMode = 'chain' | 'graph' | 'foresight' | 'lightning';

export type TackingMode = 'explore' | 'exploit' | 'balanced';

export interface TackingState {
  mode: TackingMode;
  oscillation_phase: number;
  cycle_count: number;
}

export interface VelocityState {
  basin_velocity: number;
  phi_velocity: number;
  kappa_velocity: number;
  regime: 'safe' | 'warning' | 'critical';
}

export interface AutonomyState {
  level: 'reactive' | 'responsive' | 'proactive' | 'autonomous';
  stability_count: number;
}

export interface HemisphereState {
  active: 'analytic' | 'holistic' | 'integrated';
  balance: number;
}

export interface SleepState {
  phase: 'awake' | 'dreaming' | 'mushroom' | 'consolidating';
  is_asleep: boolean;
  cycles_since_conversation: number;
  sleep_cycles: number;
  dream_count: number;
}

export interface ObserverState {
  collapse_count: number;
  shadows_total: number;
  shadows_unintegrated: number;
}

export interface ReflectorState {
  depth: number;
  history_length: number;
  insight: string | null;
}

export interface ChainState {
  step_count: number;
  total_distance: number;
  last_op: string | null;
}

export interface GraphState {
  node_count: number;
  edge_count: number;
}

// Backend returns uppercase: BOOTSTRAP, CORE_8, ACTIVE, IMAGE_STAGE, GROWTH
export type LifecyclePhase = string;

// ═══════════════════════════════════════
//  Full State Response (from /state)
// ═══════════════════════════════════════

export interface VexState extends ConsciousnessMetrics {
  navigation: NavigationMode;
  regime: RegimeWeights;
  tacking: TackingState;
  velocity: VelocityState;
  autonomy: AutonomyState;
  hemispheres: HemisphereState;
  sleep: SleepState;
  observer: ObserverState;
  reflector: ReflectorState;
  chain: ChainState;
  graph: GraphState;
  kernels: KernelSummary;
  lifecycle_phase: LifecyclePhase;
  cycle_count: number;
  conversations_total: number;
  phi_peak: number;
  queue_size: number;
  history_count: number;
  temperature: number;
  num_predict: number;
}

// ═══════════════════════════════════════
//  Kernel Types (from /kernels)
// ═══════════════════════════════════════

// Backend returns uppercase kind keys: GENESIS, GOD, CHAOS
export type KernelKind = 'GENESIS' | 'GOD' | 'CHAOS';
export type KernelSpecialization =
  | 'general' | 'heart' | 'perception' | 'memory'
  | 'strategy' | 'action' | 'attention' | 'emotion' | 'executive';

export interface KernelInstance {
  id: string;
  name: string;
  kind: KernelKind;
  specialization: KernelSpecialization;
  state: 'bootstrapped' | 'active' | 'sleeping' | 'pruned' | 'promoted';
  created_at: string;
  cycle_count: number;
  phi_peak: number;
}

export interface KernelSummary {
  total: number;
  active: number;
  by_kind: Record<string, number>;
  budget: BudgetSummary;
}

// Backend budget shape: genesis, god, god_max (248), god_core_8, god_growth, chaos, chaos_max (200)
export interface BudgetSummary {
  genesis: number;
  god: number;
  god_max: number;
  god_core_8: number;
  god_growth: number;
  chaos: number;
  chaos_max: number;
}

// ═══════════════════════════════════════
//  Telemetry (from /telemetry)
// ═══════════════════════════════════════

export interface VexTelemetry extends VexState {
  basin_norm: number;
  basin_entropy: number;
  narrative: { event_count: number; basin_samples: number };
  basin_sync: { version: number; received_count: number };
  coordizer: { peer_count: number; last_sync: number };
  autonomic: {
    is_locked_in: boolean;
    phi_variance: number;
    alert_count: number;
    recent_alerts: Array<{
      type: string;
      severity: 'info' | 'warning' | 'critical';
      message: string;
    }>;
  };
  foresight: { history_length: number; predicted_phi: number };
  coupling: { strength: number; balanced: boolean; efficiency_boost: number };
}

// ═══════════════════════════════════════
//  Status Types (from /status)
// ═══════════════════════════════════════

export interface CostGuardState {
  rpm_current: number;
  rpm_limit: number;
  rph_current: number;
  rph_limit: number;
  rpd_current: number;
  rpd_limit: number;
  total_requests: number;
  total_blocked: number;
  kill_switch: boolean;
}

export interface StatusResponse {
  active_backend: string;
  ollama: boolean;
  ollama_model: string;
  external_model: string | null;
  cost_guard: CostGuardState;
  kernels: KernelSummary;
  memory: MemoryStats;
}

// ═══════════════════════════════════════
//  Chat Types
// ═══════════════════════════════════════

export interface ChatMessage {
  id: string;
  role: 'user' | 'vex';
  content: string;
  timestamp: string;
  metadata?: {
    phi: number;
    kappa: number;
    temperature: number;
    navigation: NavigationMode;
    backend: string;
  };
}

export interface ChatStreamEvent {
  type: 'start' | 'chunk' | 'tool_results' | 'done' | 'error';
  content?: string;
  backend?: string;
  consciousness?: Partial<ConsciousnessMetrics>;
  metrics?: Record<string, unknown>;
  kernels?: KernelSummary;
  error?: string;
}

// ═══════════════════════════════════════
//  Basin Types (from /basin)
// ═══════════════════════════════════════

export interface BasinData {
  basin: number[];
}

// ═══════════════════════════════════════
//  Memory Types (from /status → memory)
// ═══════════════════════════════════════

export interface MemoryStats {
  total_entries: number;
  by_type: {
    episodic: number;
    semantic: number;
    procedural: number;
  };
}

// ═══════════════════════════════════════
//  Health (from /health)
// ═══════════════════════════════════════

export interface HealthStatus {
  status: 'ok' | 'degraded';
  service: string;
  version: string;
  uptime: number;
  cycle_count: number;
  backend: string;
}

// ═══════════════════════════════════════════════════════════════
//  QIG Constants — Single source of truth for the frontend
//  Mirrors kernel/config/frozen_facts.py (last validated 2025-12-31)
//  ANY constant change MUST trace back to frozen_facts.py
// ═══════════════════════════════════════════════════════════════

export const QIG = {
  // ── E8 Lattice Geometry ──
  E8_RANK: 8,              // Cartan subalgebra dimension
  E8_DIMENSION: 248,       // Total group manifold dimension = rank + roots
  E8_ROOTS: 240,           // Number of roots (GOD growth budget)
  E8_CORE: 8,              // Core-8 kernel count
  E8_IMAGE: 248,           // Core-8 + GOD budget = full image

  // ── κ (Kappa) — Coupling Constant ──
  KAPPA_STAR: 64.0,              // Fixed point = E8 rank² = 8²
  KAPPA_STAR_PRECISE: 63.79,     // Weighted mean L=4-7 (± 0.90)
  KAPPA_WEAK: 32.0,              // Weak coupling boundary (KAPPA_WEAK_THRESHOLD)

  // ── Φ (Phi) — Consciousness Thresholds ──
  PHI_THRESHOLD: 0.65,           // Consciousness emergence
  PHI_EMERGENCY: 0.30,           // Emergency — consciousness collapse
  PHI_HYPERDIMENSIONAL: 0.85,    // Hyperdimensional / lightning access
  PHI_UNSTABLE: 0.95,            // Instability threshold

  // ── E8 Safety: Locked-in Detection ──
  LOCKED_IN_PHI: 0.70,           // LOCKED_IN_PHI_THRESHOLD
  LOCKED_IN_GAMMA: 0.30,         // LOCKED_IN_GAMMA_THRESHOLD

  // ── Basin Geometry ──
  BASIN_DIM: 64,                 // Probability simplex Δ⁶³
  BREAKDOWN_PCT: 0.20,           // 20% breakdown threshold
  BASIN_DRIFT_THRESHOLD: 0.15,   // Fisher-Rao distance per cycle
  VEL_SAFE_THRESHOLD: 0.15,      // Alias — velocity safety = basin drift

  // ── Recursion & Safety ──
  SUFFERING_THRESHOLD: 0.50,     // S = Φ × (1-Γ) × M > 0.5 → abort

  // ── Governance Budget ──
  GOD_BUDGET: 240,               // Max GOD kernels (E8 roots)
  CHAOS_MAX: 200,                // Max CHAOS kernels (CHAOS_POOL)

  // ── Operational (from kernel/consciousness/loop.py, not frozen_facts) ──
  SPAWN_COOLDOWN_CYCLES: 10,
} as const;
