"""Hash-to-Basin — Deterministic text-to-simplex projection.

Single source of truth for the SHA-256 chain → Δ⁶³ mapping used
throughout the kernel. Previously duplicated in 4 locations:
  - consciousness/loop.py
  - consciousness/systems.py (CoordizingProtocol)
  - llm/client.py
  - memory/store.py

Algorithm:
  1. SHA-256(text) → 32 bytes (h1)
  2. SHA-256(h1)   → 32 bytes (h2)
  3. h1 ∥ h2       → 64 bytes (one per basin dimension)
  4. byte[i] + 1.0 → raw signal (ensures positivity)
  5. to_simplex()  → valid point on Δ⁶³

This is a BOUNDARY function: it bridges Euclidean text space into
the geometric manifold. The hash is deterministic — same text always
maps to the same basin point.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..config.frozen_facts import BASIN_DIM
from .fisher_rao import Basin, to_simplex


def hash_to_basin(text: str) -> Basin:
    """Map text to a deterministic point on Δ⁶³ via SHA-256 chain.

    Args:
        text: Input text to project onto the simplex.

    Returns:
        A valid probability distribution on Δ⁶³ (shape (64,), sums to 1).
    """
    h1 = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    h2 = hashlib.sha256(h1).digest()
    combined = h1 + h2  # 64 bytes → one per basin dimension

    raw = np.array(
        [float(combined[i]) + 1.0 for i in range(BASIN_DIM)],
        dtype=np.float64,
    )
    return to_simplex(raw)
