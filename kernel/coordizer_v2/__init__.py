"""
CoordizerV2 — Resonance-Based Geometric Coordizer on Δ⁶³

A complete replacement for BPE-style coordization with a
resonance bank seeded from LLM output distribution harvesting.

All operations live on the probability simplex using the
Fisher-Rao metric. No Euclidean distances. No cosine similarity.
No dot product attention. No Adam. No vector reps.

Quick start:

    from kernel.coordizer_v2 import CoordizerV2

    # Build from LLM harvest
    c = CoordizerV2.from_harvest(
        model_id="LiquidAI/LFM2.5-1.2B-Thinking",
        corpus_path="corpus.txt",
    )

    # Coordize text
    result = c.coordize("consciousness emerges from geometry")

    # Generate
    token_id, basin = c.generate_next(trajectory)

    # Validate
    v = c.validate()
    print(v.summary())

Architecture:
    geometry.py         Simplex primitives (Δ⁶³, Fisher-Rao, SLERP)
    types.py            BasinCoordinate, tiers, domain bias, results
    harvest.py          LLM output distribution harvesting (Method 1)
    compress.py         Fisher-Rao PGA (Δ^(V-1) → Δ⁶³)
    resonance_bank.py   Standing-wave vocabulary with activation
    coordizer.py        Main CoordizerV2 class
    validate.py         κ, β, semantic, harmonic validation
    modal_harvest.py    Modal GPU integration (Railway-side client)
"""

from .coordizer import CoordizerV2
from .compress import CompressionResult, compress
from .geometry import (
    BASIN_DIM,
    KAPPA_STAR,
    E8_RANK,
    Basin,
    bhattacharyya_coefficient,
    exp_map,
    fisher_rao_distance,
    fisher_rao_distance_batch,
    frechet_mean,
    geodesic_midpoint,
    log_map,
    natural_gradient,
    random_basin,
    slerp,
    softmax_to_simplex,
    to_simplex,
)
from .harvest import (
    HarvestConfig,
    HarvestResult,
    Harvester,
    harvest_model,
    harvest_model_auto,
)
from .resonance_bank import ResonanceBank
from .types import (
    BasinCoordinate,
    CoordizationResult,
    DomainBias,
    GranularityScale,
    HarmonicTier,
    ValidationResult,
)
from .validate import validate_resonance_bank

# modal_harvest is imported lazily (requires httpx + settings)
# Use: from kernel.coordizer_v2.modal_harvest import modal_harvest

__all__ = [
    # Main class
    "CoordizerV2",
    # Resonance bank
    "ResonanceBank",
    # Harvesting
    "Harvester",
    "HarvestConfig",
    "HarvestResult",
    "harvest_model",
    "harvest_model_auto",
    # Compression
    "compress",
    "CompressionResult",
    # Geometry
    "BASIN_DIM",
    "KAPPA_STAR",
    "E8_RANK",
    "Basin",
    "to_simplex",
    "softmax_to_simplex",
    "random_basin",
    "fisher_rao_distance",
    "fisher_rao_distance_batch",
    "bhattacharyya_coefficient",
    "slerp",
    "geodesic_midpoint",
    "frechet_mean",
    "log_map",
    "exp_map",
    "natural_gradient",
    # Types
    "BasinCoordinate",
    "CoordizationResult",
    "DomainBias",
    "GranularityScale",
    "HarmonicTier",
    "ValidationResult",
    # Validation
    "validate_resonance_bank",
]

# Import canonical version — do not hardcode.
try:
    from kernel.config.version import VERSION as __version__
except ImportError:
    __version__ = "2.4.0"  # fallback for isolated imports
