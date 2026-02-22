"""
CoordizerV2Adapter — Drop-in replacement for CoordinatorPipeline

Provides a compatibility layer for wiring CoordizerV2 into the existing
consciousness loop without breaking existing code that expects the old
coordizer interface.

v6.1F Integration Bridge:
  - Old: CoordinatorPipeline.transform(raw_signal) → basin
  - Old: CoordinatorPipeline.coordize_text(text) → basin
  - New: CoordizerV2.coordize(text) → CoordizationResult
  - Bridge: Adapter wraps CoordizerV2 and exposes old interface

This allows incremental migration behind a feature flag without
requiring wholesale rewrites of the consciousness loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .coordizer import CoordizerV2
from .geometry import BASIN_DIM, frechet_mean, to_simplex
from .types import CoordizationResult

logger = logging.getLogger(__name__)


class CoordizerV2Adapter:
    """Drop-in replacement for CoordinatorPipeline using CoordizerV2.

    Provides backward-compatible interface for existing consciousness
    loop code while using the new harvest→compress→validate pipeline.

    Usage:
        # From a CoordizerV2 instance (preferred — used by server.py):
        adapter = CoordizerV2Adapter(coordizer_instance)

        # From a saved bank path:
        adapter = CoordizerV2Adapter("/path/to/bank.json")

        # In consciousness loop (behind feature flag):
        if settings.coordizer_v2.enabled:
            from ..coordizer_v2 import CoordizerV2Adapter
            self._coordizer = CoordizerV2Adapter(settings.coordizer_v2.bank_path)
        else:
            from ..coordizer import CoordinatorPipeline
            self._coordizer = CoordinatorPipeline()
    """

    def __init__(
        self,
        source: str | Path | CoordizerV2,
        regime_modulation: bool = True,
        navigation_adaptation: bool = True,
        tacking_bias: bool = True,
    ):
        """Initialize adapter with CoordizerV2.

        Args:
            source: CoordizerV2 instance, or path to saved Resonance Bank
            regime_modulation: Enable regime → temperature modulation
            navigation_adaptation: Enable nav mode → generation params
            tacking_bias: Enable tacking → tier bias
        """
        if isinstance(source, CoordizerV2):
            # Direct instance injection (e.g. from server.py wiring)
            self._coordizer = source
            logger.info("CoordizerV2Adapter initialized from existing instance")
        else:
            # Load from file path
            bank_path = str(source)
            try:
                self._coordizer = CoordizerV2.from_file(bank_path)
                logger.info("CoordizerV2Adapter loaded bank from %s", bank_path)
            except FileNotFoundError:
                logger.warning(
                    "CoordizerV2 bank not found at %s. "
                    "Creating default bank with uniform basins.",
                    bank_path,
                )
                self._coordizer = self._create_bootstrap_coordizer()

        self._regime_modulation = regime_modulation
        self._navigation_adaptation = navigation_adaptation
        self._tacking_bias = tacking_bias

        # Cached result from last coordize call (for metrics extraction)
        self._last_result: CoordizationResult | None = None

    @staticmethod
    def _create_bootstrap_coordizer() -> CoordizerV2:
        """Create a minimal CoordizerV2 with uniform basins for bootstrap."""
        from .resonance_bank import ResonanceBank
        from .types import BasinCoordinate, HarmonicTier

        coordinates = []
        for i in range(256):
            basin = to_simplex(np.ones(BASIN_DIM))
            coord = BasinCoordinate(
                coord_id=i,
                vector=basin,
                name=f"<coord_{i}>",
                tier=HarmonicTier.FUNDAMENTAL,
                basin_mass=0.5,
                activation_count=0,
            )
            coordinates.append(coord)

        bank = ResonanceBank(coordinates=coordinates, dim=BASIN_DIM)
        logger.warning("Using bootstrap uniform bank. Run GPU harvest for production.")
        return CoordizerV2(bank=bank)

    def transform(self, raw_signal: NDArray) -> NDArray:
        """Transform raw signal to basin coordinates (old interface).

        Args:
            raw_signal: Input vector (any dimension)

        Returns:
            Basin coordinates on Δ⁶³
        """
        # CoordizerV2 expects text, but we can project raw signal to simplex
        # This is a degraded path — prefer coordize_text when possible
        return to_simplex(raw_signal)

    def coordize_text(
        self,
        text: str,
        regime_weights: tuple[float, float, float] | None = None,
        navigation_mode: str | None = None,
        tacking_mode: str | None = None,
    ) -> NDArray:
        """Coordize text to basin coordinates (old interface + modulation).

        Args:
            text: Input text to coordize
            regime_weights: (w₁, w₂, w₃) for temperature modulation
            navigation_mode: CHAIN/GRAPH/FORESIGHT/LIGHTNING
            tacking_mode: EXPLORE/EXPLOIT for tier bias

        Returns:
            Single basin coordinate (Fréchet mean of all coord basins)
        """
        # Coordize using CoordizerV2
        result = self._coordizer.coordize(text)
        self._last_result = result  # Cache for metrics extraction

        if not result.coordinates:
            # Fallback: uniform basin
            logger.warning("No coordinates for text: %s...", text[:50])
            return to_simplex(np.ones(BASIN_DIM))

        # Compute Fréchet mean of all coordinate basins
        basins = [coord.vector for coord in result.coordinates]
        mean_basin = frechet_mean(basins)

        return mean_basin

    def get_last_metrics(self) -> dict | None:
        """Extract metrics from last coordize call.

        Returns dict with:
            - basin_velocity: Average d_FR between consecutive coords
            - trajectory_curvature: Geodesic deviation ratio
            - harmonic_consonance: Frequency ratio coherence
            - coord_count: Number of coordinates
        """
        if self._last_result is None:
            return None

        return {
            "basin_velocity": self._last_result.basin_velocity,
            "trajectory_curvature": self._last_result.trajectory_curvature,
            "harmonic_consonance": self._last_result.harmonic_consonance,
            "coord_count": len(self._last_result.coord_ids),
        }

    def validate(self) -> dict:
        """Run CoordizerV2 validation suite.

        Returns validation result dict with:
            - kappa_measured: Measured κ value
            - kappa_std: Standard deviation
            - beta_running: Running coupling β
            - semantic_coherence: Correlation with semantic structure
            - harmonic_structure: Tier distribution quality
            - e8_eigenvalue_test: Top-8 variance capture
            - passed: Overall pass/fail
        """
        result = self._coordizer.validate()
        return {
            "kappa_measured": result.kappa_measured,
            "kappa_std": result.kappa_std,
            "beta_running": result.beta_running,
            "semantic_coherence": result.semantic_coherence,
            "harmonic_structure": result.harmonic_structure,
            "e8_eigenvalue_test": result.e8_eigenvalue_test,
            "passed": result.passed,
            "summary": result.summary(),
        }

    def set_domain_bias(self, domain: str, strength: float = 0.3):
        """Set domain bias for kernel specialization.

        Args:
            domain: Kernel domain (perception, memory, strategy, etc.)
            strength: Bias strength [0.0, 1.0]
        """
        # TODO: Implement domain anchor basin lookup. Requires pre-computed
        # anchor basins per kernel domain (perception, memory, strategy, action,
        # ethics, meta, ocean). See v6.1F §20.5 for anchor selection algorithm.
        # Implementation needs:
        # 1. Domain → anchor basin mapping (pre-harvest or bootstrap)
        # 2. DomainBias object construction with anchor + strength
        # 3. Pass to ResonanceBank.activate() for geodesic biasing
        logger.debug("Domain bias set: %s @ %s", domain, strength)

    @property
    def coordizer(self) -> CoordizerV2:
        """Access underlying CoordizerV2 instance."""
        return self._coordizer
