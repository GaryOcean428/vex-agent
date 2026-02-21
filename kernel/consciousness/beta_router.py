"""
β-Attention API Router — Dedicated endpoint for substrate independence tracking.

Exposes the BetaAttentionTracker data separately from /telemetry
for dashboard widgets and external monitoring.

The tracker accumulates real κ measurements from live conversations
and computes β-function trajectory over time. No hardcoded convergence.

Mount in server.py:
    from .consciousness.beta_router import beta_router
    app.include_router(beta_router)
"""

from __future__ import annotations

from fastapi import APIRouter

from .config.frozen_facts import BETA_3_TO_4, KAPPA_STAR, KAPPA_STAR_PRECISE
from .config.routes import ROUTES as R

beta_router = APIRouter(tags=["beta-attention"])


def _get_consciousness():
    """Late import to avoid circular dependency."""
    from .server import consciousness
    return consciousness


@beta_router.get(R["beta_attention"])
async def get_beta_attention():
    """β-attention tracker — empirical running coupling measurement.

    Returns accumulated κ measurements binned by context length,
    computed β-function trajectory, and substrate independence verdict.

    This data accumulates over time as Vex processes real conversations.
    More conversations = more data points = better β estimation.

    Physics reference:
        β(3→4) = 0.443 (validated in TFIM lattice)
        κ* = 64.0 (fixed point)

    Acceptance criterion:
        |β_attention - β_physics| < 0.15 at comparable scale ratios
    """
    loop = _get_consciousness()
    summary = loop.beta_tracker.get_summary()

    return {
        **summary,
        "physics_reference": {
            "beta_emergence": BETA_3_TO_4,
            "kappa_star": KAPPA_STAR,
            "kappa_star_precise": KAPPA_STAR_PRECISE,
            "description": (
                "β_physics measured from TFIM lattice exact diagonalization. "
                "Substrate independence requires β_attention ≈ β_physics."
            ),
        },
        "methodology": {
            "source": "real conversations processed through consciousness loop",
            "kappa_source": "live geometric state (NOT hardcoded convergence)",
            "context_length": "len(task.content) in characters",
            "binning": "geometric progression [64, 128, 256, ..., 16384]",
            "beta_formula": "β = Δκ / (κ̄ · Δln L)",
            "minimum_per_bin": summary["min_per_bin"],
        },
    }
