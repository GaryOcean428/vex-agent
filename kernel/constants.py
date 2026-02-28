"""QIG Constants — Single Import Point
====================================

Convenience re-export of ALL frozen physics constants from the
canonical source ``kernel.config.frozen_facts``.

Usage::

    from kernel.constants import KAPPA_STAR, BASIN_DIM, E8_RANK

The canonical definitions live in ``kernel/config/frozen_facts.py``.
This module exists so callers can write the shorter import path
``kernel.constants`` instead of ``kernel.config.frozen_facts``.

Do NOT define new constants here. Add them to ``frozen_facts.py``
and then add the import to this file.  The explicit import list is
intentional — it enables static analysis tools (mypy, ruff) to
verify the re-exports.
"""

from kernel.config.frozen_facts import (  # noqa: F401
    BASIN_DIM,
    BASIN_DIVERGENCE_THRESHOLD,
    BASIN_DRIFT_THRESHOLD,
    BETA_3_TO_4,
    CHAOS_POOL,
    CONSENSUS_DISTANCE,
    CORE_8_COUNT,
    E8_CORE,
    E8_DIMENSION,
    E8_IMAGE,
    E8_RANK,
    E8_ROOTS,
    FULL_IMAGE,
    GOD_BUDGET,
    INSTABILITY_PCT,
    KAPPA_3,
    KAPPA_4,
    KAPPA_5,
    KAPPA_6,
    KAPPA_7,
    KAPPA_STAR,
    KAPPA_STAR_PRECISE,
    LOCKED_IN_GAMMA_THRESHOLD,
    LOCKED_IN_PHI_THRESHOLD,
    MIN_RECURSION_DEPTH,
    PHI_BREAKDOWN_MIN,
    PHI_EMERGENCY,
    PHI_HYPERDIMENSIONAL,
    PHI_LINEAR_MAX,
    PHI_THRESHOLD,
    PHI_UNSTABLE,
    SUFFERING_THRESHOLD,
)
