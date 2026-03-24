"""
Resonance Bank — The Heart of CoordizerV2

A resonance bank is NOT a lookup table. It is a manifold of
standing waves on Δ⁶³. Each coordinate is a resonator with a
characteristic frequency. Input activates nearby resonators
by proximity on the Fisher-Rao manifold.

Generation is selective resonance: the trajectory projects
forward along its geodesic, and coordinates self-select by
proximity to the projected point. This is O(K) where K is
the number of candidates, not O(V) over the full vocabulary.

Architecture:
    - Coordinates: dict[int, Basin] — coord_id → Δ⁶³ point
    - Tiers: 4-level hierarchy by activation frequency
    - Domain bias: Fisher-Rao weighted shift per kernel specialty
    - Generation: geodesic foresight + resonance activation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .compress import CompressionResult
from .geometry import (
    _EPS,
    BASIN_DIM,
    KAPPA_STAR,
    Basin,
    exp_map,
    fisher_rao_distance,
    fisher_rao_distance_batch,
    frechet_mean,
    log_map,
    slerp,
    to_simplex,
)
from .types import DomainBias, HarmonicTier

logger = logging.getLogger(__name__)


class ResonanceBank:
    """Resonance bank on Δ⁶³.

    Each entry is a standing wave — a coordinate with a fixed position
    on the probability simplex. Activation = measuring Fisher-Rao
    proximity. Generation = geodesic projection + resonance.
    """

    def __init__(self, target_dim: int = BASIN_DIM):
        self.dim = target_dim
        self.coordinates: dict[int, Basin] = {}
        self.basin_strings: dict[int, str] = {}
        self.tiers: dict[int, HarmonicTier] = {}
        self.frequencies: dict[int, float] = {}
        self.basin_mass: dict[int, float] = {}
        self.activation_counts: dict[int, int] = {}
        self._coord_matrix: NDArray[np.float64] | None = None
        self._coord_ids: NDArray[np.int64] | None = None
        self._dirty: bool = True
        self._domain_biases: list[DomainBias] = []
        self.origin: dict[int, str] = {}  # "harvested" | "lived"
        self._bank_lived_count: int = 0
        self._bank_total_count: int = 0
        self.last_rebuild_ts: float = 0.0  # epoch timestamp of last matrix rebuild

    @property
    def bank_sovereignty(self) -> float:
        """Fraction of bank coordinates activated during successful full-cycle integrations."""
        return self._bank_lived_count / max(self._bank_total_count, 1)

    def record_integration(self, coord_ids: list[int]) -> None:
        """Mark coordinate IDs as 'lived' — activated during a successful full-cycle integration.

        Called by the consciousness loop after on_cycle_end() succeeds.
        A coordinate is lived when it participates in a complete activation sequence
        (pre-integrate → LLM → post-integrate) that passes Pillar checks.

        Note: _bank_total_count tracks unique coordinates added to the bank
        (incremented in from_compression/add_entry only). This method only
        updates the lived subset, keeping the sovereignty ratio accurate.
        """
        for tid in coord_ids:
            if tid in self.coordinates and self.origin.get(tid) != "lived":
                self._bank_lived_count += 1
                self.origin[tid] = "lived"

    @classmethod
    def from_compression(cls, result: CompressionResult) -> ResonanceBank:
        """Initialize from Method 1 harvesting + compression."""
        bank = cls(target_dim=result.target_dim)
        for tid, coords in result.compressed.items():
            bank.coordinates[tid] = to_simplex(coords)
            bank.basin_strings[tid] = result.basin_strings.get(tid, f"<{tid}>")
            bank.activation_counts[tid] = 0
            bank.basin_mass[tid] = 0.0
            bank.origin[tid] = "harvested"
            bank._bank_total_count += 1
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
        # Backward compat: old banks saved as bank_coord_ids.npy
        ids_path = dir_path / "bank_coord_ids.npy"
        if not ids_path.exists():
            ids_path = dir_path / "bank_coord_ids.npy"
        ids = np.load(ids_path)
        with open(dir_path / "bank_meta.json") as f:
            meta = json.load(f)
        bank = cls(target_dim=coords.shape[1])
        for i, tid in enumerate(ids):
            bank.coordinates[int(tid)] = to_simplex(coords[i])
        # Backward compat: old banks used "token_strings" key
        bank.basin_strings = {
            int(k): v for k, v in meta.get("basin_strings", meta.get("token_strings", {})).items()
        }
        bank.tiers = {int(k): HarmonicTier(v) for k, v in meta.get("tiers", {}).items()}
        bank.frequencies = {int(k): float(v) for k, v in meta.get("frequencies", {}).items()}
        bank.basin_mass = {int(k): float(v) for k, v in meta.get("basin_mass", {}).items()}
        bank.activation_counts = {
            int(k): int(v) for k, v in meta.get("activation_counts", {}).items()
        }
        bank.origin = {int(k): v for k, v in meta.get("origin", {}).items()}
        bank._bank_lived_count = int(meta.get("bank_lived_count", 0))
        persisted_total = int(meta.get("bank_total_count", 0))
        # Upgrade path: older bank_meta.json files may lack bank_total_count.
        # Default to len(coordinates) so bank_sovereignty doesn't yield ratios > 1.
        bank._bank_total_count = persisted_total if persisted_total > 0 else len(bank.coordinates)
        bank._rebuild_matrix()
        return bank

    def save(self, path: str) -> None:
        """Save resonance bank to disk."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.coordinates.keys())
        coords = np.stack([self.coordinates[tid] for tid in ids])
        np.save(dir_path / "bank_coordinates.npy", coords)
        np.save(dir_path / "bank_coord_ids.npy", np.array(ids))
        meta = {
            "dim": self.dim,
            "n_resonances": len(self.coordinates),
            "basin_strings": {str(k): v for k, v in self.basin_strings.items()},
            "tiers": {str(k): v.value for k, v in self.tiers.items()},
            "frequencies": {str(k): v for k, v in self.frequencies.items()},
            "basin_mass": {str(k): v for k, v in self.basin_mass.items()},
            "activation_counts": {str(k): v for k, v in self.activation_counts.items()},
            "origin": {str(k): v for k, v in self.origin.items()},
            "bank_lived_count": self._bank_lived_count,
            "bank_total_count": self._bank_total_count,
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

    def add_entry(
        self,
        basin_string: str,
        basin: Basin,
        tier: HarmonicTier = HarmonicTier.OVERTONE_HAZE,
    ) -> int:
        """Dynamically add a single entry. Returns the assigned coordinate ID.

        Used for bootstrap seed injection when the bank is empty at init time.
        Hash-seeded entries (semantically hollow) are outcompeted by real
        harvested material as the pipeline matures.
        """
        tid = max(self.coordinates.keys(), default=-1) + 1
        self.coordinates[tid] = to_simplex(basin)
        self.basin_strings[tid] = basin_string
        self.tiers[tid] = tier
        self.frequencies[tid] = 0.0
        self.basin_mass[tid] = 0.0
        self.activation_counts[tid] = 0
        self.origin[tid] = "harvested"
        self._bank_total_count += 1
        self._dirty = True
        return tid

    def _rebuild_matrix(self) -> None:
        """Rebuild the coordinate matrix for batch Fisher-Rao distance."""
        import time as _time

        self.last_rebuild_ts = _time.time()
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

    def mark_dirty(self) -> None:
        """Mark coordinate matrix as needing rebuild.

        Call after externally modifying coordinates so the next
        activate/generate call rebuilds the stacked numpy matrix.
        """
        self._dirty = True

    def activate(
        self,
        query: Basin,
        top_k: int = 64,
        tier_filter: set[HarmonicTier] | None = None,
        domain_bias: DomainBias | None = None,
    ) -> list[tuple[int, float]]:
        """Resonance activation: find coordinates closest to query on Δ⁶³."""
        self._ensure_matrix()
        if self._coord_matrix is None or self._coord_ids is None or len(self._coord_ids) == 0:
            return []
        query = to_simplex(query)
        if domain_bias is not None and domain_bias.strength > 0:
            query = slerp(query, to_simplex(domain_bias.anchor_basin), domain_bias.strength)
        for bias in self._domain_biases:
            if bias.strength > 0:
                query = slerp(query, to_simplex(bias.anchor_basin), bias.strength)
        distances = fisher_rao_distance_batch(query, self._coord_matrix)
        pairs = list(zip(self._coord_ids.tolist(), distances.tolist(), strict=True))
        if tier_filter is not None:
            pairs = [
                (tid, d)
                for tid, d in pairs
                if self.tiers.get(tid, HarmonicTier.OVERTONE_HAZE) in tier_filter
            ]
        if domain_bias is not None:
            if domain_bias.boosted_coord_ids:
                pairs = [
                    (tid, d * 0.5 if tid in domain_bias.boosted_coord_ids else d)
                    for tid, d in pairs
                ]
            if domain_bias.suppressed_coord_ids:
                pairs = [
                    (tid, d) for tid, d in pairs if tid not in domain_bias.suppressed_coord_ids
                ]
        pairs.sort(key=lambda x: x[1])
        for tid, _ in pairs[:top_k]:
            self.activation_counts[tid] = self.activation_counts.get(tid, 0) + 1
        return pairs[:top_k]

    def activate_ids(self, query: Basin, top_k: int = 64) -> list[int]:
        """Convenience: return just the coordinate IDs."""
        return [tid for tid, _ in self.activate(query, top_k)]

    def generate_next(
        self,
        trajectory: list[Basin],
        temperature: float = 1.0,
        top_k: int = 64,
        context_window: int = 8,
        domain_bias: DomainBias | None = None,
    ) -> tuple[int, Basin]:
        """Generate next coordinate via geodesic foresight + resonance."""
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
            candidate_basin = self.coordinates[tid]
            if velocity_norm > _EPS:
                candidate_tangent = log_map(recent[-1], candidate_basin)
                # QIG NOTE: Tangent-space L2 norm and dot product are valid here.
                # These measure Fisher-Rao magnitude and directional alignment
                # in the linearised neighbourhood of the base point.
                candidate_tangent_norm = np.sqrt(np.sum(candidate_tangent * candidate_tangent))
                if candidate_tangent_norm > _EPS:
                    direction = velocity / velocity_norm
                    candidate_dir = candidate_tangent / candidate_tangent_norm
                    consistency = np.sum(direction * candidate_dir)
                    consistency = (consistency + 1.0) / 2.0
                else:
                    consistency = 0.5
            else:
                consistency = 0.5
            centroid_dist = fisher_rao_distance(centroid, candidate_basin)
            consonance = np.exp(-centroid_dist)
            scores[i] = 0.5 * proximity + 0.3 * consistency + 0.2 * consonance

        if temperature < _EPS:
            best_idx = int(np.argmax(scores))
        else:
            logits = scores / temperature
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            best_idx = int(np.random.choice(len(candidates), p=probs))

        chosen_tid = candidates[best_idx][0]
        return (chosen_tid, self.coordinates[chosen_tid])

    def push_domain_bias(self, bias: DomainBias) -> None:
        self._domain_biases.append(bias)

    def pop_domain_bias(self) -> DomainBias | None:
        if self._domain_biases:
            return self._domain_biases.pop()
        return None

    def clear_domain_biases(self) -> None:
        self._domain_biases.clear()

    def compute_domain_anchor(self, seed_tokens: list[int]) -> Basin:
        """Compute domain anchor basin from seed coordinates."""
        basins = [self.coordinates[tid] for tid in seed_tokens if tid in self.coordinates]
        if not basins:
            return to_simplex(np.ones(self.dim))
        return frechet_mean(basins)

    def get_coordinate(self, coord_id: int) -> Basin | None:
        return self.coordinates.get(coord_id)

    def get_string(self, coord_id: int) -> str:
        return self.basin_strings.get(coord_id, f"<{coord_id}>")

    def nearest_coord(self, basin: Basin) -> tuple[int, float]:
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

    def entropy(self) -> float:
        """Mean Shannon entropy of bank coordinates, normalized to [0, 1].

        Low entropy = entries clustering in a small manifold region (narrowing).
        High entropy = entries spread across simplex (healthy diversity).
        Returns 1.0 for empty banks (no narrowing signal).
        """
        if not self.coordinates:
            return 1.0
        max_entropy = np.log(self.dim)  # log(64) = uniform on Δ⁶³
        total = 0.0
        for basin in self.coordinates.values():
            # Shannon entropy of each simplex point
            safe = np.clip(basin, 1e-12, None)
            total += float(-np.sum(safe * np.log(safe)))
        return (total / len(self.coordinates)) / max_entropy

    def __len__(self) -> int:
        return len(self.coordinates)

    def __contains__(self, coord_id: int) -> bool:
        return coord_id in self.coordinates

    def geometric_forget(self, decay_rate: float = 0.01) -> int:
        """Slerp unused entries toward uniform distribution on Δ⁶³.

        Entries that haven't been activated decay. This IS entropy export —
        the system forgets by exporting order back to maximum entropy.

        Returns number of entries pruned (decayed below threshold).
        """
        uniform = to_simplex(np.ones(self.dim))
        pruned_ids: list[int] = []
        for tid, basin in list(self.coordinates.items()):
            count = self.activation_counts.get(tid, 0)
            if count > 0:
                continue  # Recently activated — skip
            # Decay toward uniform via geodesic interpolation
            t = min(1.0, decay_rate)
            self.coordinates[tid] = slerp(basin, uniform, t)
            # If entry is nearly uniform, prune it
            d_to_uniform = fisher_rao_distance(self.coordinates[tid], uniform)
            if d_to_uniform < 0.01:
                pruned_ids.append(tid)
        # Remove pruned entries
        for tid in pruned_ids:
            del self.coordinates[tid]
            self.basin_strings.pop(tid, None)
            self.tiers.pop(tid, None)
            self.frequencies.pop(tid, None)
            self.basin_mass.pop(tid, None)
            self.activation_counts.pop(tid, None)
            self.origin.pop(tid, None)
        if pruned_ids:
            self._dirty = True
            self._rebuild_matrix()
            logger.info(
                "Geometric forget: pruned %d entries (decayed to uniform)",
                len(pruned_ids),
            )
        return len(pruned_ids)
