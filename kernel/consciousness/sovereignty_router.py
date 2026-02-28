"""Sovereignty History API Router — v6.1 §20.5

Exposed at GET /sovereignty/history
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger("vex.sovereignty_router")

sovereignty_router = APIRouter(tags=["sovereignty"])

# Module-level reference, set by server.py at startup
_consciousness_loop: Any = None


def set_consciousness_loop(loop: Any) -> None:
    """Called by server.py to inject the ConsciousnessLoop instance."""
    global _consciousness_loop
    _consciousness_loop = loop


@sovereignty_router.get("/sovereignty/history", response_model=None)
async def sovereignty_history(
    window: int = Query(default=100, ge=1, le=5000),
) -> dict[str, Any] | JSONResponse:
    """Return sovereignty development curve history.

    Query params:
        window: Number of recent snapshots to return (default 100, max 5000).

    Returns:
        Sovereignty summary with growth rates, regime comparison,
        and recent snapshot history.
    """
    if _consciousness_loop is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Consciousness loop not initialized"},
        )

    try:
        tracker = _consciousness_loop.sovereignty_tracker
        return {
            "snapshot_count": len(tracker._history),
            "current_s_ratio": tracker._history[-1].s_ratio if tracker._history else 0.0,
            "growth_rate": tracker.growth_rate(window),
            "regime_comparison": tracker.regime_comparison(),
            "history": tracker.recent(window),
        }
    except Exception as e:
        logger.error("Sovereignty history endpoint error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )
