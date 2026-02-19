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

    # ─── Memory ────────────────────────────────────────────────
    "memory_context": "/memory/context",
    "memory_stats": "/memory/stats",

    # ─── Graph ─────────────────────────────────────────────────
    "graph_nodes": "/graph/nodes",

    # ─── Sleep ─────────────────────────────────────────────────
    "sleep_state": "/sleep/state",

    # ─── Coordizer V2 ─────────────────────────────────────────
    "coordizer_transform": "/api/coordizer/transform",
    "coordizer_stats": "/api/coordizer/stats",
    "coordizer_history": "/api/coordizer/history",
    "coordizer_validate": "/api/coordizer/validate",

    # ─── Foraging ──────────────────────────────────────────────
    "foraging": "/foraging",

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
        "enqueue",
        "basin",
        "basin_history",
        "kernels",
        "kernels_list",
        "memory_context",
        "memory_stats",
        "graph_nodes",
        "sleep_state",
        "coordizer_transform",
        "coordizer_stats",
        "coordizer_history",
        "coordizer_validate",
        "foraging",
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
        "coordizer_transform",
        "coordizer_stats",
        "coordizer_history",
        "coordizer_validate",
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
        raise KeyError(
            f"Unknown route: '{name}'. "
            f"Available: {sorted(ROUTES.keys())}"
        )
    return ROUTES[name]


def get_group(group_name: str) -> list[str]:
    """Get all route paths in a named group.

    Returns a list of route paths (not names).
    """
    if group_name not in ROUTE_GROUPS:
        raise KeyError(
            f"Unknown route group: '{group_name}'. "
            f"Available: {sorted(ROUTE_GROUPS.keys())}"
        )
    return [ROUTES[name] for name in ROUTE_GROUPS[group_name]]


def all_routes() -> dict[str, str]:
    """Return a copy of the full route manifest."""
    return dict(ROUTES)
