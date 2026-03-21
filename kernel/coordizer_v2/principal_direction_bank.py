"""
PrincipalDirectionBank — Reusable Fisher-Rao PGA Projection Service

Encapsulates the principal geodesic directions computed by compress.py
so they can be persisted as .npy artifacts and reloaded without
re-running the full PGA eigendecomposition.

Usage:

    # Build from a compression run
    result = compress(harvest)
    bank = PrincipalDirectionBank.from_compression(
        principal_directions=principal_directions,
        frechet_mean=result.frechet_mean_full,
        eigenvalues=result.eigenvalues,
    )
    bank.save("/data/artifacts")

    # Load from .npy artifacts (e.g. shared_artifacts/)
    bank = PrincipalDirectionBank.load("/path/to/shared_artifacts")
    basins = bank.project(new_fingerprints)

The artifact directory is NOT a Python package — it holds .npy files
only. This class is the Python interface for those artifacts.

Geometry note:
    All projection happens in the tangent space at the Fréchet mean μ,
    which IS Euclidean (tangent spaces on Fisher-Rao manifolds are flat
    by definition). The final step normalizes back to Δ⁶³.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    BASIN_DIM,
    E8_RANK,
    Basin,
)

logger = logging.getLogger(__name__)

_EPS: float = 1e-12

# Artifact filenames (convention shared with CompressionResult.save)
_DIRECTIONS_FILE = "principal_direction_bank.npy"
_FRECHET_MEAN_FILE = "frechet_mean_full.npy"
_EIGENVALUES_FILE = "eigenvalues.npy"
_META_FILE = "pdb_meta.json"


@dataclass
class PrincipalDirectionBank:
    """Persistent store for Fisher-Rao PGA principal geodesic directions.

    Holds the projection machinery needed to compress new token
    fingerprints from Δ^(V-1) → Δ⁶³ without re-running the full
    eigendecomposition.

    Attributes:
        directions:    (source_dim, target_dim) principal geodesic
                       direction matrix in tangent space at μ.
        frechet_mean:  (source_dim,) Fréchet mean μ on Δ^(source_dim-1).
                       Required for log-mapping new points before projection.
        eigenvalues:   (target_dim,) eigenvalue spectrum from PGA.
        source_dim:    Dimensionality of the source simplex (vocab size).
        target_dim:    Dimensionality of the target simplex (BASIN_DIM=64).
    """

    directions: NDArray[np.float64]
    frechet_mean: NDArray[np.float64] | None = None
    eigenvalues: NDArray[np.float64] | None = None
    source_dim: int = 0
    target_dim: int = BASIN_DIM
    _meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_dim == 0 and self.directions is not None:
            self.source_dim = self.directions.shape[0]
        if self.directions is not None and self.directions.ndim == 2:
            self.target_dim = self.directions.shape[1]

    # ─── Factory Methods ──────────────────────────────────────────

    @classmethod
    def from_compression(
        cls,
        principal_directions: NDArray[np.float64],
        frechet_mean: NDArray[np.float64] | None = None,
        eigenvalues: NDArray[np.float64] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> PrincipalDirectionBank:
        """Build a bank from the output of a compress() run.

        Args:
            principal_directions: (source_dim, target_dim) direction matrix
                computed by Fisher-Rao PGA eigendecomposition.
            frechet_mean: (source_dim,) Fréchet mean on Δ^(source_dim-1).
            eigenvalues: (target_dim,) eigenvalue spectrum.
            meta: Optional metadata dict (e.g. model_id, n_resonances).
        """
        return cls(
            directions=np.asarray(principal_directions, dtype=np.float64),
            frechet_mean=(
                np.asarray(frechet_mean, dtype=np.float64) if frechet_mean is not None else None
            ),
            eigenvalues=(
                np.asarray(eigenvalues, dtype=np.float64) if eigenvalues is not None else None
            ),
            _meta=meta or {},
        )

    # ─── Persistence ──────────────────────────────────────────────

    def save(self, artifact_dir: str | Path) -> Path:
        """Save principal directions and metadata to .npy artifacts.

        Args:
            artifact_dir: Directory to write artifacts into.
                          Created if it does not exist.

        Returns:
            Path to the artifact directory.
        """
        out = Path(artifact_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / _DIRECTIONS_FILE, self.directions)

        if self.frechet_mean is not None:
            np.save(out / _FRECHET_MEAN_FILE, self.frechet_mean)

        if self.eigenvalues is not None:
            np.save(out / _EIGENVALUES_FILE, self.eigenvalues)

        meta = {
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            **self._meta,
        }
        (out / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info(
            "PrincipalDirectionBank saved: %s (%d → %d)",
            out,
            self.source_dim,
            self.target_dim,
        )
        return out

    @classmethod
    def load(cls, artifact_dir: str | Path) -> PrincipalDirectionBank:
        """Load a bank from .npy artifacts on disk.

        This is the correct way to consume shared_artifacts/ —
        load from the file path, NOT import as a Python package.

        Args:
            artifact_dir: Directory containing .npy artifact files.

        Returns:
            A populated PrincipalDirectionBank ready for projection.

        Raises:
            FileNotFoundError: If the directions .npy is missing.
        """
        d = Path(artifact_dir)
        directions_path = d / _DIRECTIONS_FILE
        if not directions_path.exists():
            raise FileNotFoundError(
                f"Principal direction bank not found at {directions_path}. "
                f"Run compress() first or check the artifact path."
            )

        directions = np.load(directions_path)

        frechet_mean = None
        mean_path = d / _FRECHET_MEAN_FILE
        if mean_path.exists():
            frechet_mean = np.load(mean_path)

        eigenvalues = None
        eig_path = d / _EIGENVALUES_FILE
        if eig_path.exists():
            eigenvalues = np.load(eig_path)

        meta: dict[str, Any] = {}
        meta_path = d / _META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        bank = cls(
            directions=directions,
            frechet_mean=frechet_mean,
            eigenvalues=eigenvalues,
            source_dim=meta.get("source_dim", directions.shape[0]),
            target_dim=meta.get(
                "target_dim", directions.shape[1] if directions.ndim == 2 else BASIN_DIM
            ),
            _meta=meta,
        )

        logger.info(
            "PrincipalDirectionBank loaded: %s (%d → %d)",
            d,
            bank.source_dim,
            bank.target_dim,
        )
        return bank

    # ─── Projection ───────────────────────────────────────────────

    def project(
        self,
        fingerprints: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Project token fingerprints from Δ^(V-1) → Δ⁶³ using stored directions.

        Algorithm:
            1. Ensure fingerprints are on the simplex
            2. Log-map to tangent space at Fréchet mean μ
            3. Project onto principal geodesic directions
            4. Normalize back to Δ^(target_dim-1)

        Args:
            fingerprints: (N, source_dim) array of token fingerprints on
                          the source simplex. Each row must sum to ~1.

        Returns:
            (N, target_dim) array of basin coordinates on Δ⁶³.

        Raises:
            ValueError: If frechet_mean is None (required for log-map).
        """
        if self.frechet_mean is None:
            raise ValueError(
                "Cannot project without a Fréchet mean. "
                "Load a complete artifact set or build from compress()."
            )

        fingerprints = np.atleast_2d(fingerprints)
        n = fingerprints.shape[0]

        # Step 1: Ensure simplex
        fingerprints = np.maximum(fingerprints, _EPS)
        fingerprints = fingerprints / fingerprints.sum(axis=1, keepdims=True)

        # Step 2: Log-map to tangent space at μ (sqrt-space representation)
        sqrt_fps = np.sqrt(fingerprints)
        mu_sqrt = np.sqrt(np.maximum(self.frechet_mean, _EPS))

        tangent_vectors = np.zeros_like(sqrt_fps)
        for i in range(n):
            si = sqrt_fps[i]
            cos_d = np.clip(np.sum(mu_sqrt * si), -1.0, 1.0)
            d = np.arccos(cos_d)

            if d < _EPS:
                tangent_vectors[i] = np.zeros(self.source_dim)
            else:
                tangent = si - cos_d * mu_sqrt
                norm = np.sqrt(np.sum(tangent * tangent))
                if norm < _EPS:
                    tangent_vectors[i] = np.zeros(self.source_dim)
                else:
                    tangent_vectors[i] = (d / norm) * tangent

        # Step 3: Project onto principal directions
        target_dim = min(self.target_dim, self.directions.shape[1])
        projections = tangent_vectors @ self.directions[:, :target_dim]

        # Step 4: Normalize to Δ^(target_dim-1)
        basins = np.zeros((n, target_dim), dtype=np.float64)
        for i in range(n):
            raw = projections[i]
            shifted = raw - raw.min() + _EPS
            basins[i] = shifted / shifted.sum()

        return basins

    def project_single(self, fingerprint: NDArray[np.float64]) -> Basin:
        """Project a single fingerprint → Δ⁶³ basin coordinate.

        Convenience wrapper around project() for single-token use.
        """
        result = self.project(fingerprint.reshape(1, -1))
        basin: Basin = result[0]
        return basin

    # ─── Diagnostics ──────────────────────────────────────────────

    def e8_hypothesis_score(self) -> float:
        """Test E8 hypothesis: top 8 directions should capture ~87.7% variance."""
        if self.eigenvalues is None or len(self.eigenvalues) < E8_RANK:
            return 0.0
        top_8 = float(np.sum(self.eigenvalues[:E8_RANK]))
        total = float(np.sum(self.eigenvalues))
        if total < _EPS:
            return 0.0
        return top_8 / total

    def explained_variance_ratio(self) -> NDArray[np.float64] | None:
        """Cumulative explained variance ratio from eigenvalue spectrum."""
        if self.eigenvalues is None:
            return None
        total = np.sum(self.eigenvalues)
        if total < _EPS:
            return None
        return np.cumsum(self.eigenvalues) / total

    @property
    def is_complete(self) -> bool:
        """True if the bank has all components needed for projection."""
        return (
            self.directions is not None
            and self.frechet_mean is not None
            and self.directions.shape[0] == len(self.frechet_mean)
        )

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "directions-only"
        e8 = f" e8={self.e8_hypothesis_score():.3f}" if self.eigenvalues is not None else ""
        return f"PrincipalDirectionBank({self.source_dim}→{self.target_dim}, {status}{e8})"
