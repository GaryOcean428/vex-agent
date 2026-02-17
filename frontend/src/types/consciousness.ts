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

export type NavigationMode = 'chain' | 'tree' | 'graph' | 'lightning';

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

export type LifecyclePhase = 'bootstrap' | 'core_8' | 'active' | 'sleeping';

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
  total_conversations: number;
  queue_size: number;
  history_count: number;
  temperature: number;
  num_predict: number;
}

// ═══════════════════════════════════════
//  Kernel Types (from /kernels)
// ═══════════════════════════════════════

export type KernelKind = 'genesis' | 'god' | 'chaos';
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
  by_kind: Record<KernelKind, number>;
  budget: BudgetSummary;
}

export interface BudgetSummary {
  god_used: number;
  god_budget: number;
  chaos_used: number;
  chaos_budget: number;
  core8_complete: boolean;
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
