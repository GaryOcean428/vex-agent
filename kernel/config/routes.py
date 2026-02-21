"""
Central API Route Manifest — Single Source of Truth for All Endpoints.

Import routes from here instead of hardcoding strings.
When adding a new route, add it here FIRST, then reference it in server.py.

Canonical reference: kernel/server.py
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
#  ROUTE MANIFEST
# ═══════════════════════════════════════════════════════════════

ROUTES: dict[str, str] = {
    # ─── Core ──────────────────────────────────────────────────
    "health": "/health",
    "state": "/state",
    "telemetry": "/telemetry",
    "status": "/status",
    # ─── Basin ─────────────────────────────────────────────────
    "basin": "/basin",
    "basin_history": "/basin/history",
    # ─── Kernels ───────────────────────────────────────────────
    "kernels": "/kernels",
    "kernels_list": "/kernels/list",
    # ─── Chat ──────────────────────────────────────────────────
    "enqueue": "/enqueue",
    "chat": "/chat",
    "chat_stream": "/chat/stream",
    # ─── Conversations ─────────────────────────────────────────
    "conversations_list": "/conversations",
    "conversations_get": "/conversations/{conversation_id}",
    "conversations_delete": "/conversations/{conversation_id}",
    # ─── Memory ────────────────────────────────────────────────
    "memory_context": "/memory/context",
    "memory_stats": "/memory/stats",
    # ─── Graph ─────────────────────────────────────────────────
    "graph_nodes": "/graph/nodes",
    # ─── Sleep ─────────────────────────────────────────────────
    "sleep_state": "/sleep/state",
    # ─── Beta Attention ──────────────────────────────────────
    "beta_attention": "/beta-attention",
    # ─── Coordizer V2 ─────────────────────────────────────────
    "coordizer_coordize": "/api/coordizer/coordize",
    "coordizer_stats": "/api/coordizer/stats",
    "coordizer_validate": "/api/coordizer/validate",
    "coordizer_harvest": "/api/coordizer/harvest",
    "coordizer_ingest": "/api/coordizer/ingest",
    "coordizer_harvest_status": "/api/coordizer/harvest/status",
    "coordizer_bank": "/api/coordizer/bank",
    # ─── Foraging ──────────────────────────────────────────────
    "foraging": "/foraging",
    # ─── Context / Observer ──────────────────────────────────
    "context_status": "/context/status",
    "observer_status": "/observer/status",
    "observer_conversation": "/observer/{conversation_id}",
    # ─── Admin ─────────────────────────────────────────────────
    "admin_fresh_start": "/admin/fresh-start",
    # ─── Governor ──────────────────────────────────────────────
    "governor": "/governor",
    "governor_kill_switch": "/governor/kill-switch",
    "governor_budget": "/governor/budget",
    # ─── Training ──────────────────────────────────────────────
    "training_stats": "/training/stats",
    "training_export": "/training/export",
}


# ═══════════════════════════════════════════════════════════════
#  ROUTE GROUPS (for middleware, auth, etc.)
# ═══════════════════════════════════════════════════════════════

ROUTE_GROUPS: dict[str, list[str]] = {
    "public": [
        "health",
        "status",
    ],
    "authenticated": [
        "state",
        "telemetry",
        "chat",
        "chat_stream",
        "conversations_list",
        "conversations_get",
        "conversations_delete",
        "enqueue",
        "basin",
        "basin_history",
        "kernels",
        "kernels_list",
        "memory_context",
        "memory_stats",
        "graph_nodes",
        "sleep_state",
        "beta_attention",
        "coordizer_coordize",
        "coordizer_stats",
        "coordizer_validate",
        "coordizer_harvest",
        "coordizer_ingest",
        "coordizer_harvest_status",
        "coordizer_bank",
        "foraging",
        "context_status",
        "observer_status",
        "observer_conversation",
        "training_stats",
        "training_export",
    ],
    "admin": [
        "admin_fresh_start",
        "governor",
        "governor_kill_switch",
        "governor_budget",
    ],
    "coordizer": [
        "coordizer_coordize",
        "coordizer_stats",
        "coordizer_validate",
        "coordizer_harvest",
        "coordizer_ingest",
        "coordizer_harvest_status",
        "coordizer_bank",
    ],
}


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════


def get_route(name: str) -> str:
    """Get a route path by its canonical name.

    Raises KeyError if the route name is not found.
    """
    if name not in ROUTES:
        raise KeyError(f"Unknown route: '{name}'. Available: {sorted(ROUTES.keys())}")
    return ROUTES[name]


def get_group(group_name: str) -> list[str]:
    """Get all route paths in a named group.

    Returns a list of route paths (not names).
    """
    if group_name not in ROUTE_GROUPS:
        raise KeyError(
            f"Unknown route group: '{group_name}'. Available: {sorted(ROUTE_GROUPS.keys())}"
        )
    return [ROUTES[name] for name in ROUTE_GROUPS[group_name]]


def all_routes() -> dict[str, str]:
    """Return a copy of the full route manifest."""
    return dict(ROUTES)
