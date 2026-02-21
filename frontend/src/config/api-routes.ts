/**
 * Central API Route Manifest for Frontend.
 *
 * All fetch() calls should reference these constants instead of
 * hardcoding path strings. Mirrors kernel/config/routes.py.
 */

export const API = {
	// ─── Core ──────────────────────────────────────────────────
	health: '/health',
	state: '/state',
	telemetry: '/telemetry',
	status: '/status',

	// ─── Basin ─────────────────────────────────────────────────
	basin: '/basin',
	basinHistory: '/basin/history',

	// ─── Kernels ───────────────────────────────────────────────
	kernels: '/kernels',
	kernelList: '/kernels/list',

	// ─── Chat ──────────────────────────────────────────────────
	enqueue: '/enqueue',
	chatStream: '/chat/stream',
	chatAuth: '/chat/auth',
	chatStatus: '/chat/status',
	chatHistory: '/chat/history',

	// ─── Conversations ─────────────────────────────────────────
	conversations: '/conversations',
	conversationGet: (id: string) => `/conversations/${id}`,
	conversationDelete: (id: string) => `/conversations/${id}`,

	// ─── Memory ────────────────────────────────────────────────
	memoryContext: '/memory/context',
	memoryStats: '/memory/stats',

	// ─── Graph ─────────────────────────────────────────────────
	graphNodes: '/graph/nodes',

	// ─── Sleep ─────────────────────────────────────────────────
	sleepState: '/sleep/state',

	// ─── Beta Attention ────────────────────────────────────────
	betaAttention: '/beta-attention',

	// ─── Auth ──────────────────────────────────────────────────
	authCheck: '/auth/check',

	// ─── Admin ─────────────────────────────────────────────────
	adminFreshStart: '/admin/fresh-start',

	// ─── Governor ──────────────────────────────────────────────
	governor: '/governor',
	governorKillSwitch: '/governor/kill-switch',
	governorBudget: '/governor/budget',
	governorAutonomousSearch: '/governor/autonomous-search',

	// ─── Training ──────────────────────────────────────────────
	trainingStats: '/training/stats',
	trainingExport: '/training/export',
	trainingUpload: '/training/upload',
} as const;

export type ApiRoute = (typeof API)[keyof typeof API];
