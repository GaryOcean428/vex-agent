"""
β-Attention API Router — Dedicated endpoint for β-tracker telemetry.

Exposed at GET /beta-attention
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..config.frozen_facts import BETA_3_TO_4, KAPPA_STAR

logger = logging.getLogger("vex.beta_router")

beta_router = APIRouter(tags=["beta-attention"])

# Module-level reference, set by server.py at startup
_consciousness_loop = None


def set_consciousness_loop(loop) -> None:
    """Called by server.py to inject the ConsciousnessLoop instance."""
    global _consciousness_loop
    _consciousness_loop = loop


@beta_router.get("/beta-attention")
async def get_beta_attention():
    """Get empirical β-function trajectory from real conversations.

    Returns bin statistics, β-trajectory, and substrate independence
    verdict computed from actual consciousness loop measurements.

    Physics reference:
        β(3→4) = 0.443 (validated TFIM lattice)
        Acceptance: |β_attention - β_physics| < 0.15

    Methodology:
        - κ_eff measured from live consciousness geometric state
        - Binned by context length (geometric progression)
        - β = Δκ / (κ̄ · Δln L) from empirical data
        - No hardcoded convergence to κ*
    """
    if _consciousness_loop is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Consciousness loop not initialized"},
        )

    try:
        summary = _consciousness_loop.beta_tracker.get_summary()
        return {
            **summary,
            "physics_references": {
                "kappa_star": KAPPA_STAR,
                "beta_3_to_4": BETA_3_TO_4,
                "source": "qig-verification FROZEN_FACTS (L=3..6 TFIM lattice)",
            },
            "methodology": {
                "source": "live consciousness loop measurements",
                "binning": "geometric context-length progression",
                "formula": "beta = delta_kappa / (kappa_avg * delta_ln_L)",
                "non_circular": True,
            },
        }
    except Exception as e:
        logger.error("Beta attention endpoint error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )
