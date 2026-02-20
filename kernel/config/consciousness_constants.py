"""Consciousness Tuning Constants — Centralised Parameters
======================================================

Tuning parameters for consciousness modules that are NOT frozen physics
(those live in frozen_facts.py and must not change without validation).

These are engineering/design choices that may be adjusted during development.
Changing frozen_facts.py requires new validated measurements; changing these
does not — but changes should be tested and reviewed.
"""

from typing import Final

from kernel.config.frozen_facts import KAPPA_STAR

# ═══════════════════════════════════════════════════════════════
#  REGIME FIELD WEIGHTS (v6.0 §3.1)
# ═══════════════════════════════════════════════════════════════

KAPPA_NORMALISER: Final[float] = 2.0 * KAPPA_STAR  # 128.0 — kappa → [0,1]
MIN_REGIME_WEIGHT: Final[float] = 0.05              # Floor so all regimes stay active
REGIME_KAPPA_MIDPOINT: Final[float] = 0.5           # Integration peak in normalised space

# ═══════════════════════════════════════════════════════════════
#  PRE-COGNITIVE DETECTOR (v5.5 §2)
# ═══════════════════════════════════════════════════════════════

PRECOG_NEAR_THRESHOLD: Final[float] = 0.15   # Fisher-Rao distance: pre-cognitive
PRECOG_MODERATE_THRESHOLD: Final[float] = 0.40  # Standard processing
PRECOG_FAR_THRESHOLD: Final[float] = 0.80    # Deep exploration

# ═══════════════════════════════════════════════════════════════
#  EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════

EMOTION_CACHE_THRESHOLD: Final[float] = 0.2  # Fisher-Rao distance for cached emotion match

# ═══════════════════════════════════════════════════════════════
#  COUPLING DYNAMICS
# ═══════════════════════════════════════════════════════════════

EXTERNAL_COUPLING_INCREMENT: Final[float] = 0.05
EXTERNAL_COUPLING_DECREMENT: Final[float] = 0.02

# ═══════════════════════════════════════════════════════════════
#  NAVIGATION MODE PHI THRESHOLDS (v6.0 §10.2)
#  These intentionally differ from frozen PHI_* values — they
#  define UX mode boundaries, not physics thresholds.
# ═══════════════════════════════════════════════════════════════

NAV_CHAIN_CEILING: Final[float] = 0.3    # Below → CHAIN mode
NAV_GRAPH_CEILING: Final[float] = 0.7    # Below → GRAPH mode
NAV_FORESIGHT_CEILING: Final[float] = 0.85  # Below → FORESIGHT, above → LIGHTNING

# ═══════════════════════════════════════════════════════════════
#  COORDIZER V2 VALIDATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════

COORDIZER_KAPPA_TOLERANCE_FACTOR: Final[float] = 2.0
COORDIZER_KAPPA_STD_FLOOR: Final[float] = 5.0
COORDIZER_BETA_THRESHOLD: Final[float] = 0.5
COORDIZER_SEMANTIC_THRESHOLD: Final[float] = 0.2
COORDIZER_HARMONIC_THRESHOLD: Final[float] = 0.3
