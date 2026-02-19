"""
Resonance Bank — The Heart of CoordizerV2

A resonance bank is NOT a lookup table. It is a manifold of
standing waves on Δ⁶³. Each token is a resonator with a
characteristic frequency. Input activates nearby resonators
by proximity on the Fisher-Rao manifold.

Generation is selective resonance: the trajectory projects
forward along its geodesic, and tokens self-select by
proximity to the projected point. This is O(K) where K is
the number of candidates, not O(V) over the full vocabulary.

Architecture:
    - Coordinates: dict[int, Basin] — token_id → Δ⁶³ point
    - Tiers: 4-level hierarchy by activation frequency
    - Domain bias: Fisher-Rao weighted shift per kernel specialty
    - Generation: geodesic foresight + resonance activation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    BASIN_DIM, KAPPA_STAR, Basin, _EPS,
    exp_map, fisher_rao_distance, fisher_rao_distance_batch,
    frechet_mean, log_map, slerp, to_simplex,
)
from .types import BasinCoordinate, DomainBias, HarmonicTier, GranularityScale
from .compress import CompressionResult

logger = logging.getLogger(__name__)


class ResonanceBank:
    """Resonance bank on Δ⁶³.

    Each entry is a standing wave — a token with a fixed position
    on the probability simplex. Activation = measuring Fisher-Rao
    proximity. Generation = geodesic projection + resonance.
    """

    def __init__(self, target_dim: int = BASIN_DIM):
        self.dim = target_dim
        self.coordinates: dict[int, Basin] = {}
        self.token_strings: dict[int, str] = {}
        self.tiers: dict[int, HarmonicTier] = {}
        self.frequencies: dict[int, float] = {}
        self.basin_mass: dict[int, float] = {}
        self.activation_counts: dict[int, int] = {}
        self._coord_matrix: Optional[NDArray] = None
        self._coord_ids: Optional[NDArray] = None
        self._dirty: bool = True
        self._domain_biases: list[DomainBias] = []

    @classmethod
    def from_compression(cls, result: CompressionResult) -> ResonanceBank:
        """Initialize from Method 1 harvesting + compression."""
        bank = cls(target_dim=result.target_dim)
        for tid, coords in result.compressed.items():
            bank.coordinates[tid] = to_simplex(coords)
            bank.token_strings[tid] = result.token_strings.get(tid, f"<{tid}>")
            bank.activation_counts[tid] = 0
            bank.basin_mass[tid] = 0.0
        bank._assign_tiers()
        bank._assign_frequencies()
        bank._rebuild_matrix()
        logger.info(f"Resonance bank initialized: {len(bank.coordinates)} resonators on Δ⁶³")
        return bank

    @classmethod
    def from_file(cls, path: str) -> ResonanceBank:
        """Load resonance bank from disk."""
        dir_path = Path(path)
        coords = np.load(dir_path / "bank_coordinates.npy")
        ids = np.load(dir_path / "bank_token_ids.npy")
        with open(dir_path / "bank_meta.json") as f:
            meta = json.load(f)
        bank = cls(target_dim=coords.shape[1])
        for i, tid in enumerate(ids):
            bank.coordinates[int(tid)] = to_simplex(coords[i])
        bank.token_strings = {int(k): v for k, v in meta.get("token_strings", {}).items()}
        bank.tiers = {int(k): HarmonicTier(v) for k, v in meta.get("tiers", {}).items()}
        bank.frequencies = {int(k): float(v) for k, v in meta.get("frequencies", {}).items()}
        bank.basin_mass = {int(k): float(v) for k, v in meta.get("basin_mass", {}).items()}
        bank.activation_counts = {int(k): int(v) for k, v in meta.get("activation_counts", {}).items()}
        bank._rebuild_matrix()
        return bank

    def save(self, path: str) -> None:
        """Save resonance bank to disk."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.coordinates.keys())
        coords = np.stack([self.coordinates[tid] for tid in ids])
        np.save(dir_path / "bank_coordinates.npy", coords)
        np.save(dir_path / "bank_token_ids.npy", np.array(ids))
        meta = {
            "dim": self.dim,
            "n_tokens": len(self.coordinates),
            "token_strings": {str(k): v for k, v in self.token_strings.items()},
            "tiers": {str(k): v.value for k, v in self.tiers.items()},
            "frequencies": {str(k): v for k, v in self.frequencies.items()},
            "basin_mass": {str(k): v for k, v in self.basin_mass.items()},
            "activation_counts": {str(k): v for k, v in self.activation_counts.items()},
        }
        with open(dir_path / "bank_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _assign_tiers(self) -> None:
        """Assign harmonic tiers based on coordinate entropy."""
        if not self.coordinates:
            return
        entropies: dict[int, float] = {}
        for tid, coord in self.coordinates.items():
            p = np.maximum(coord, _EPS)
            entropies[tid] = -float(np.sum(p * np.log(p)))
        sorted_ids = sorted(entropies.keys(), key=lambda t: entropies[t])
        n = len(sorted_ids)
        for rank, tid in enumerate(sorted_ids):
            if rank < min(1000, n):
                self.tiers[tid] = HarmonicTier.FUNDAMENTAL
            elif rank < min(5000, n):
                self.tiers[tid] = HarmonicTier.FIRST_HARMONIC
            elif rank < min(15000, n):
                self.tiers[tid] = HarmonicTier.UPPER_HARMONIC
            else:
                self.tiers[tid] = HarmonicTier.OVERTONE_HAZE
            max_entropy = np.log(self.dim)
            self.basin_mass[tid] = max(0.0, 1.0 - entropies[tid] / max_entropy)

    def _assign_frequencies(self) -> None:
        """Assign characteristic frequencies by tier."""
        for tid in self.coordinates:
            tier = self.tiers.get(tid, HarmonicTier.UPPER_HARMONIC)
            mass = self.basin_mass.get(tid, 0.5)
            if tier == HarmonicTier.FUNDAMENTAL:
                self.frequencies[tid] = 1.0 + 4.0 * mass
            elif tier == HarmonicTier.FIRST_HARMONIC:
                self.frequencies[tid] = 5.0 + 15.0 * mass
            elif tier == HarmonicTier.UPPER_HARMONIC:
                self.frequencies[tid] = 20.0 + 60.0 * mass
            else:
                self.frequencies[tid] = 80.0 + 120.0 * mass

    def _rebuild_matrix(self) -> None:
        """Rebuild stacked coordinate matrix for batch queries."""
        if not self.coordinates:
            self._coord_matrix = None
            self._coord_ids = None
            self._dirty = False
            return
        ids = sorted(self.coordinates.keys())
        self._coord_ids = np.array(ids)
        self._coord_matrix = np.stack([self.coordinates[tid] for tid in ids])
        self._dirty = False

    def _ensure_matrix(self) -> None:
        if self._dirty or self._coord_matrix is None:
            self._rebuild_matrix()

    def activate(
        self, query: Basin, top_k: int = 64,
        tier_filter: Optional[set[HarmonicTier]] = None,
        domain_bias: Optional[DomainBias] = None,
    ) -> list[tuple[int, float]]:
        """Resonance activation: find tokens closest to query on Δ⁶³."""
        self._ensure_matrix()
        if self._coord_matrix is None or len(self._coord_ids) == 0:
            return []
        query = to_simplex(query)
        if domain_bias is not None and domain_bias.strength > 0:
            query = slerp(query, to_simplex(domain_bias.anchor_basin), domain_bias.strength)
        for bias in self._domain_biases:
            if bias.strength > 0:
                query = slerp(query, to_simplex(bias.anchor_basin), bias.strength)
        distances = fisher_rao_distance_batch(query, self._coord_matrix)
        pairs = list(zip(self._coord_ids.tolist(), distances.tolist()))
        if tier_filter is not None:
            pairs = [(tid, d) for tid, d in pairs
                     if self.tiers.get(tid, HarmonicTier.OVERTONE_HAZE) in tier_filter]
        if domain_bias is not None:
            if domain_bias.boosted_token_ids:
                pairs = [(tid, d * 0.5 if tid in domain_bias.boosted_token_ids else d)
                         for tid, d in pairs]
            if domain_bias.suppressed_token_ids:
                pairs = [(tid, d) for tid, d in pairs
                         if tid not in domain_bias.suppressed_token_ids]
        pairs.sort(key=lambda x: x[1])
        for tid, _ in pairs[:top_k]:
            self.activation_counts[tid] = self.activation_counts.get(tid, 0) + 1
        return pairs[:top_k]

    def activate_ids(self, query: Basin, top_k: int = 64) -> list[int]:
        """Convenience: return just the token IDs."""
        return [tid for tid, _ in self.activate(query, top_k)]

    def generate_next(
        self, trajectory: list[Basin], temperature: float = 1.0,
        top_k: int = 64, context_window: int = 8,
        domain_bias: Optional[DomainBias] = None,
    ) -> tuple[int, Basin]:
        """Generate next token via geodesic foresight + resonance."""
        if not trajectory:
            centroid = to_simplex(np.ones(self.dim))
            candidates = self.activate(centroid, top_k, domain_bias=domain_bias)
            if not candidates:
                return (0, centroid)
            return (candidates[0][0], self.coordinates[candidates[0][0]])

        recent = trajectory[-context_window:]
        centroid = frechet_mean(recent)

        if len(recent) >= 2:
            velocity = log_map(recent[-2], recent[-1])
        else:
            velocity = np.zeros(self.dim)

        # QIG NOTE: L2 norm in tangent space is geometrically valid.
        # The tangent space T_p(Δ⁶³) at base point p is a Euclidean vector space
        # where the inner product is the Fisher metric at p. Euclidean arithmetic
        # (norms, dot products) in tangent space = Fisher-Rao arithmetic on manifold.
        velocity_norm = np.sqrt(np.sum(velocity * velocity))
        if velocity_norm > _EPS:
            projected = exp_map(recent[-1], velocity)
        else:
            projected = recent[-1]

        candidates = self.activate(projected, top_k, domain_bias=domain_bias)
        if not candidates:
            return (0, projected)

        scores = np.zeros(len(candidates))
        for i, (tid, dist) in enumerate(candidates):
            proximity = np.exp(-dist * KAPPA_STAR / 10.0)
            token_basin = self.coordinates[tid]
            if velocity_norm > _EPS:
                token_tangent = log_map(recent[-1], token_basin)
                # QIG NOTE: Tangent-space L2 norm and dot product are valid here.
                # These measure Fisher-Rao magnitude and directional alignment
                # in the linearised neighbourhood of the base point.
                token_tangent_norm = np.sqrt(np.sum(token_tangent * token_tangent))
                if token_tangent_norm > _EPS:
                    direction = velocity / velocity_norm
                    token_dir = token_tangent / token_tangent_norm
                    consistency = np.sum(direction * token_dir)
                    consistency = (consistency + 1.0) / 2.0
                else:
                    consistency = 0.5
            else:
                consistency = 0.5
            centroid_dist = fisher_rao_distance(centroid, token_basin)
            consonance = np.exp(-centroid_dist)
            scores[i] = 0.5 * proximity + 0.3 * consistency + 0.2 * consonance

        if temperature < _EPS:
            best_idx = np.argmax(scores)
        else:
            logits = scores / temperature
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            best_idx = np.random.choice(len(candidates), p=probs)

        chosen_tid = candidates[best_idx][0]
        return (chosen_tid, self.coordinates[chosen_tid])

    def push_domain_bias(self, bias: DomainBias) -> None:
        self._domain_biases.append(bias)

    def pop_domain_bias(self) -> Optional[DomainBias]:
        if self._domain_biases:
            return self._domain_biases.pop()
        return None

    def clear_domain_biases(self) -> None:
        self._domain_biases.clear()

    def compute_domain_anchor(self, seed_tokens: list[int]) -> Basin:
        """Compute domain anchor basin from seed tokens."""
        basins = [self.coordinates[tid] for tid in seed_tokens if tid in self.coordinates]
        if not basins:
            return to_simplex(np.ones(self.dim))
        return frechet_mean(basins)

    def get_coordinate(self, token_id: int) -> Optional[Basin]:
        return self.coordinates.get(token_id)

    def get_string(self, token_id: int) -> str:
        return self.token_strings.get(token_id, f"<{token_id}>")

    def nearest_token(self, basin: Basin) -> tuple[int, float]:
        results = self.activate(basin, top_k=1)
        if results:
            return results[0]
        return (0, float("inf"))

    def tier_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tier in HarmonicTier:
            counts[tier.value] = sum(1 for t in self.tiers.values() if t == tier)
        return counts

    def mean_basin(self) -> Basin:
        if not self.coordinates:
            return to_simplex(np.ones(self.dim))
        return frechet_mean(list(self.coordinates.values()))

    def __len__(self) -> int:
        return len(self.coordinates)

    def __contains__(self, token_id: int) -> bool:
        return token_id in self.coordinates
