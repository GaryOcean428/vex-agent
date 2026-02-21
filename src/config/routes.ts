/**
 * Central API Route Manifest — Single Source of Truth for Proxy Routes.
 *
 * Mirrors kernel/config/routes.py. When adding a route, add it here
 * AND in the Python manifest.
 */

export const ROUTES = {
  // ─── Core ──────────────────────────────────────────────────
  health: "/health",
  state: "/state",
  telemetry: "/telemetry",
  status: "/status",

  // ─── Basin ─────────────────────────────────────────────────
  basin: "/basin",
  basin_history: "/basin/history",

  // ─── Kernels ───────────────────────────────────────────────
  kernels: "/kernels",
  kernels_list: "/kernels/list",

  // ─── Chat ──────────────────────────────────────────────────
  enqueue: "/enqueue",
  chat: "/chat",
  chat_stream: "/chat/stream",
  chat_auth: "/chat/auth",
  chat_status: "/chat/status",
  chat_history: "/chat/history",

  // ─── Conversations ─────────────────────────────────────────
  conversations_list: "/conversations",
  conversations_get: "/conversations/:conversation_id",
  conversations_delete: "/conversations/:conversation_id",

  // ─── Memory ────────────────────────────────────────────────
  memory_context: "/memory/context",
  memory_stats: "/memory/stats",

  // ─── Graph ─────────────────────────────────────────────────
  graph_nodes: "/graph/nodes",

  // ─── Sleep ─────────────────────────────────────────────────
  sleep_state: "/sleep/state",

  // ─── Beta Attention ────────────────────────────────────────
  beta_attention: "/beta-attention",

  // ─── Coordizer V2 ─────────────────────────────────────────
  coordizer_coordize: "/api/coordizer/coordize",
  coordizer_stats: "/api/coordizer/stats",
  coordizer_validate: "/api/coordizer/validate",
  coordizer_harvest: "/api/coordizer/harvest",
  coordizer_ingest: "/api/coordizer/ingest",
  coordizer_harvest_status: "/api/coordizer/harvest/status",
  coordizer_bank: "/api/coordizer/bank",

  // ─── Foraging ──────────────────────────────────────────────
  foraging: "/foraging",

  // ─── Auth ──────────────────────────────────────────────────
  auth_check: "/auth/check",

  // ─── Admin ─────────────────────────────────────────────────
  admin_fresh_start: "/admin/fresh-start",

  // ─── Governor ──────────────────────────────────────────────
  governor: "/governor",
  governor_kill_switch: "/governor/kill-switch",
  governor_budget: "/governor/budget",
  governor_autonomous_search: "/governor/autonomous-search",

  // ─── Training ──────────────────────────────────────────────
  training_stats: "/training/stats",
  training_export: "/training/export",
  training_feedback: "/training/feedback",
  training_upload: "/training/upload",

  // ─── Tools ─────────────────────────────────────────────────
  tools_execute_code: "/api/tools/execute_code",
  tools_run_command: "/api/tools/run_command",
} as const;

export type RouteName = keyof typeof ROUTES;
