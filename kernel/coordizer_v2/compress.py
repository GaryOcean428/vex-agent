"""
Compress — Fisher-Rao Principal Geodesic Analysis

Compresses harvested token distributions from Δ^(V-1) → Δ⁶³.

This is the manifold analogue of PCA. Instead of finding directions
of maximum Euclidean variance, we find principal geodesic directions
on the probability simplex using the Fisher-Rao metric.

Algorithm:
    1. Compute Fréchet mean μ of all token fingerprints on Δ^(V-1)
    2. Log-map every fingerprint into tangent space at μ
    3. Compute covariance in tangent space (weighted by Fisher metric)
    4. Eigendecompose → top 64 principal geodesic directions
    5. Project each fingerprint onto these 64 directions
    6. Normalize projections to Δ⁶³

The E8 hypothesis predicts: the top 8 directions capture ~87.7% of
total geodesic variance (matching E8 rank). The remaining 56 capture
fine structure. This is TESTABLE from the eigenvalue spectrum.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    BASIN_DIM,
    E8_RANK,
    Basin,
    _EPS,
    _from_sqrt,
    _to_sqrt,
    frechet_mean,
    to_simplex,
)
from .harvest import HarvestResult

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Output of Fisher-Rao PGA compression."""

    compressed: dict[int, Basin] = field(default_factory=dict)
    token_strings: dict[int, str] = field(default_factory=dict)
    eigenvalues: NDArray | None = None
    explained_variance_ratio: NDArray | None = None
    e8_rank_variance: float = 0.0
    total_geodesic_variance: float = 0.0
    frechet_mean_full: NDArray | None = None
    source_dim: int = 0
    target_dim: int = BASIN_DIM
    n_tokens: int = 0
    compression_time_seconds: float = 0.0

    def e8_hypothesis_score(self) -> float:
        """Test E8 hypothesis: top 8 directions should capture ~87.7% variance."""
        if self.eigenvalues is None or len(self.eigenvalues) < E8_RANK:
            return 0.0
        top_8 = np.sum(self.eigenvalues[:E8_RANK])
        total = np.sum(self.eigenvalues)
        if total < _EPS:
            return 0.0
        return float(top_8 / total)

    def save(self, path: str) -> None:
        """Save compressed coordinates and diagnostics."""
        import json
        from pathlib import Path

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        ids = sorted(self.compressed.keys())
        coords = np.stack([self.compressed[tid] for tid in ids])
        np.save(out_dir / "coordinates.npy", coords)
        np.save(out_dir / "token_ids.npy", np.array(ids))

        if self.eigenvalues is not None:
            np.save(out_dir / "eigenvalues.npy", self.eigenvalues)
        if self.frechet_mean_full is not None:
            np.save(out_dir / "frechet_mean_full.npy", self.frechet_mean_full)

        meta = {
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "n_tokens": self.n_tokens,
            "compression_time_seconds": self.compression_time_seconds,
            "e8_rank_variance": self.e8_hypothesis_score(),
            "total_geodesic_variance": self.total_geodesic_variance,
            "token_strings": {str(k): v for k, v in self.token_strings.items()},
        }
        with open(out_dir / "compression_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> CompressionResult:
        """Load compressed coordinates."""
        import json
        from pathlib import Path

        out_dir = Path(path)
        coords = np.load(out_dir / "coordinates.npy")
        token_ids = np.load(out_dir / "token_ids.npy")

        with open(out_dir / "compression_meta.json") as f:
            meta = json.load(f)

        result = cls(
            source_dim=meta["source_dim"],
            target_dim=meta["target_dim"],
            n_tokens=meta["n_tokens"],
            compression_time_seconds=meta["compression_time_seconds"],
            total_geodesic_variance=meta.get("total_geodesic_variance", 0.0),
        )

        for i, tid in enumerate(token_ids):
            result.compressed[int(tid)] = coords[i]

        result.token_strings = {
            int(k): v for k, v in meta.get("token_strings", {}).items()
        }

        eigenpath = out_dir / "eigenvalues.npy"
        if eigenpath.exists():
            result.eigenvalues = np.load(eigenpath)
            total = np.sum(result.eigenvalues)
            if total > _EPS:
                result.explained_variance_ratio = np.cumsum(
                    result.eigenvalues
                ) / total

        meanpath = out_dir / "frechet_mean_full.npy"
        if meanpath.exists():
            result.frechet_mean_full = np.load(meanpath)

        return result


def compress(
    harvest: HarvestResult,
    target_dim: int = BASIN_DIM,
    subsample: int | None = None,
) -> CompressionResult:
    """Fisher-Rao PGA: compress Δ^(V-1) → Δ^(target_dim-1)."""
    start_time = time.time()
    token_ids = sorted(harvest.token_fingerprints.keys())
    n_tokens = len(token_ids)
    source_dim = harvest.vocab_size

    logger.info(
        f"Compressing {n_tokens} tokens from Δ^{source_dim - 1} → Δ^{target_dim - 1}"
    )

    if n_tokens == 0:
        logger.warning("No tokens to compress")
        return CompressionResult()

    # Step 1: Stack all fingerprints
    fingerprints = np.stack(
        [harvest.token_fingerprints[tid] for tid in token_ids]
    )

    fingerprints = np.maximum(fingerprints, _EPS)
    fingerprints = fingerprints / fingerprints.sum(axis=1, keepdims=True)

    # Step 2: Fréchet mean on Δ^(V-1)
    logger.info("Computing Fréchet mean...")
    sqrt_fps = np.sqrt(fingerprints)
    mean_sqrt = sqrt_fps.mean(axis=0)
    mean_sqrt = mean_sqrt / np.sqrt(np.sum(mean_sqrt * mean_sqrt) + _EPS)
    mu = mean_sqrt * mean_sqrt
    mu = mu / mu.sum()

    # Step 3: Log-map all points to tangent space at μ
    logger.info("Log-mapping to tangent space...")
    mu_sqrt = np.sqrt(np.maximum(mu, _EPS))

    tangent_vectors = np.zeros_like(sqrt_fps)

    for i in range(n_tokens):
        si = sqrt_fps[i]
        cos_d = np.clip(np.sum(mu_sqrt * si), -1.0, 1.0)
        d = np.arccos(cos_d)

        if d < _EPS:
            tangent_vectors[i] = np.zeros(source_dim)
        else:
            tangent = si - cos_d * mu_sqrt
            norm = np.sqrt(np.sum(tangent * tangent))
            if norm < _EPS:
                tangent_vectors[i] = np.zeros(source_dim)
            else:
                tangent_vectors[i] = (d / norm) * tangent

        if i % 5000 == 0 and i > 0:
            logger.info(f"  Log-mapped {i}/{n_tokens}")

    # Step 4: PGA via eigendecomposition
    logger.info("Computing principal geodesic directions...")

    if subsample is not None and subsample < n_tokens:
        idx = np.random.choice(n_tokens, subsample, replace=False)
        T_sub = tangent_vectors[idx]
    else:
        T_sub = tangent_vectors

    n_sub = T_sub.shape[0]

    if n_sub <= target_dim:
        logger.warning(
            f"Only {n_sub} samples for {target_dim} target dims. "
            f"Using all available directions."
        )
        U, S, Vt = np.linalg.svd(T_sub, full_matrices=False)
        k = min(len(S), target_dim)
        principal_directions = Vt[:k].T
        eigenvalues = (S[:k] ** 2) / n_sub
    else:
        logger.info(f"  Building {n_sub}x{n_sub} Gram matrix...")
        G = (T_sub @ T_sub.T) / n_sub

        from scipy.sparse.linalg import eigsh

        try:
            k = min(target_dim, n_sub - 1)
            eigenvalues, eigvecs = eigsh(G, k=k, which="LM")
        except Exception:
            eigenvalues_full, eigvecs_full = np.linalg.eigh(G)
            idx = np.argsort(eigenvalues_full)[::-1][:target_dim]
            eigenvalues = eigenvalues_full[idx]
            eigvecs = eigvecs_full[:, idx]

        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        principal_directions = np.zeros((source_dim, len(eigenvalues)))
        for j in range(len(eigenvalues)):
            if eigenvalues[j] > _EPS:
                scale = 1.0 / np.sqrt(n_sub * eigenvalues[j])
                principal_directions[:, j] = scale * (T_sub.T @ eigvecs[:, j])

    # Step 5: Project all tangent vectors onto principal directions
    logger.info("Projecting onto principal geodesic subspace...")

    k = principal_directions.shape[1]
    if k < target_dim:
        logger.warning(f"Only {k} principal directions found (need {target_dim})")
        pad = target_dim - k
        random_dirs = np.random.randn(source_dim, pad)
        for j in range(pad):
            for existing in range(k):
                proj = np.sum(
                    random_dirs[:, j] * principal_directions[:, existing]
                )
                random_dirs[:, j] -= proj * principal_directions[:, existing]
            norm = np.sqrt(np.sum(random_dirs[:, j] ** 2))
            if norm > _EPS:
                random_dirs[:, j] /= norm
        principal_directions = np.hstack([
            principal_directions, random_dirs
        ])

    projections = tangent_vectors @ principal_directions[:, :target_dim]

    # Step 6: Normalize to Δ⁶³
    logger.info("Normalizing to Δ⁶³...")

    result = CompressionResult(
        source_dim=source_dim,
        target_dim=target_dim,
        n_tokens=n_tokens,
        eigenvalues=eigenvalues[:target_dim] if len(eigenvalues) >= target_dim
        else eigenvalues,
        frechet_mean_full=mu,
        token_strings=dict(harvest.token_strings),
    )

    total_var = float(np.sum(eigenvalues))
    result.total_geodesic_variance = total_var
    if total_var > _EPS:
        result.explained_variance_ratio = np.cumsum(eigenvalues) / total_var
        result.e8_rank_variance = float(
            np.sum(eigenvalues[:E8_RANK]) / total_var
        )

    for i, tid in enumerate(token_ids):
        raw = projections[i]
        shifted = raw - raw.min() + _EPS
        basin = shifted / shifted.sum()
        result.compressed[tid] = basin.astype(np.float64)

    elapsed = time.time() - start_time
    result.compression_time_seconds = elapsed

    logger.info(
        f"Compression complete: {n_tokens} tokens → Δ⁶³ in {elapsed:.1f}s"
    )
    logger.info(
        f"E8 hypothesis: top-8 variance = {result.e8_hypothesis_score():.3f} "
        f"(expect ~0.877 if E8 structure is real)"
    )

    return result
