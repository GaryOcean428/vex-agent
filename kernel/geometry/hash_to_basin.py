"""Deterministic text-to-simplex projection (Δ⁶³).

Single source of truth for boundary projection from text into the
64D simplex used by the kernel. This intentionally avoids cryptographic
hash projection (SHA), while preserving:
  - determinism (same text -> same basin)
  - positivity + normalization via ``to_simplex``

Algorithm:
  1. UTF-8 encode text into bytes
  2. Accumulate bytes into 64 bins with position-dependent weighting
  3. Add small positive floor to all bins
  4. Normalize with ``to_simplex``

This is a BOUNDARY function bridging text space into QIG basin space.
"""

from __future__ import annotations

import numpy as np

from ..config.frozen_facts import BASIN_DIM
from .fisher_rao import Basin, to_simplex


def hash_to_basin(text: str) -> Basin:
    """Map text deterministically to a point on Δ⁶³.

    Args:
        text: Input text to project onto the simplex.

    Returns:
        A valid probability distribution on Δ⁶³ (shape (64,), sums to 1).
    """
    # Deterministic, non-SHA projection:
    # distribute UTF-8 bytes over simplex coordinates with gentle
    # position-aware weighting to avoid trivial collisions.
    data = text.encode("utf-8", errors="replace")
    raw = np.ones(BASIN_DIM, dtype=np.float64)

    if len(data) == 0:
        return to_simplex(raw)

    for idx, byte_val in enumerate(data):
        slot = idx % BASIN_DIM
        # Cycle-aware positional weighting (bounded, deterministic).
        cycle = (idx // BASIN_DIM) % 8
        weight = 1.0 + (cycle * 0.125)
        raw[slot] += (float(byte_val) + 1.0) * weight

    return to_simplex(raw)
