"""Coordizer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..config.frozen_facts import BASIN_DIM
from .types import HarvestConfig, PipelineConfig, TransformMethod

# Default coordizer configuration
DEFAULT_PIPELINE_CONFIG: Final[PipelineConfig] = PipelineConfig(
    method=TransformMethod.SOFTMAX,
    validation_mode="standard",
    auto_fix=True,
    batch_size=32,
    numerical_stability=True,
    epsilon=1e-10,
)

# Default harvest configuration
DEFAULT_HARVEST_CONFIG: Final[HarvestConfig] = HarvestConfig(
    enabled=True,
    sampling_rate=0.1,  # 10% of messages
    min_quality_score=0.5,
    batch_size=16,
    max_tokens=512,
)

# Coordizer dimensionality (matches basin dimension)
COORDIZER_DIM: Final[int] = BASIN_DIM  # 64


@dataclass
class CoordinizerSettings:
    """Global coordizer settings.

    Attributes:
        enabled: Whether coordizer is enabled
        pipeline_config: Pipeline configuration
        harvest_config: Harvest configuration
        target_dim: Target coordinate dimensionality
        enforce_purity: Enforce geometric purity checks
    """

    enabled: bool = True
    pipeline_config: PipelineConfig = DEFAULT_PIPELINE_CONFIG
    harvest_config: HarvestConfig = DEFAULT_HARVEST_CONFIG
    target_dim: int = COORDIZER_DIM
    enforce_purity: bool = True

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.target_dim != BASIN_DIM:
            # Warning: dimensionality mismatch with basin
            # Allow it but log warning
            pass


# Global singleton instance
_coordizer_settings: CoordinizerSettings | None = None


def get_coordizer_settings() -> CoordinizerSettings:
    """Get global coordizer settings.

    Returns:
        Current settings instance
    """
    global _coordizer_settings
    if _coordizer_settings is None:
        _coordizer_settings = CoordinizerSettings()
    return _coordizer_settings


def set_coordizer_settings(settings: CoordinizerSettings) -> None:
    """Set global coordizer settings.

    Args:
        settings: New settings to use
    """
    global _coordizer_settings
    _coordizer_settings = settings
