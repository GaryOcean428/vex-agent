"""
CoordizerV2 — Resonance-Based Geometric Coordizer

Replaces the BPE-style FisherCoordizer with a resonance bank
seeded from LLM output distribution harvesting.

Key differences from v1:
    - Coordinates live on Δ⁶³ (probability simplex), not S⁶³ (unit sphere)
    - Initialization via LLM harvesting, not random byte vectors
    - Generation via geodesic foresight + resonance activation
    - All distances use Fisher-Rao metric
    - Harmonic tier assignment from basin mass
    - Domain bias via Fisher-Rao weighted shift
    - κ/β validation built in

Usage:
    # From scratch (harvest → compress → validate)
    coordizer = CoordizerV2.from_harvest(
        model_id="LiquidAI/LFM2.5-1.2B-Thinking",
        corpus_path="corpus.txt",
    )

    # From saved bank
    coordizer = CoordizerV2.from_file("./coordizer_bank")

    # Coordize text
    result = coordizer.coordize("Hello world")
    print(result.coord_ids)
    print(result.basin_velocity)

    # Generate next token
    token_id, basin = coordizer.generate_next(trajectory)

    # Validate geometric structure
    validation = coordizer.validate()
    print(validation.summary())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    BASIN_DIM, KAPPA_STAR, Basin, _EPS,
    fisher_rao_distance, frechet_mean, slerp, to_simplex,
)
from .types import (
    BasinCoordinate, CoordizationResult, DomainBias,
    GranularityScale, HarmonicTier, ValidationResult,
)
from .resonance_bank import ResonanceBank
from .compress import CompressionResult, compress
from .harvest import HarvestConfig, HarvestResult, Harvester
from .validate import validate_resonance_bank

logger = logging.getLogger(__name__)


class CoordizerV2:
    """Resonance-based geometric coordizer on Δ⁶³.

    This is the successor to FisherCoordizer. Instead of BPE-style
    merging on random sphere vectors, it operates on a resonance bank
    seeded from an LLM's own geometric structure.

    Three operations:
        1. Coordize:   text → sequence of basin coordinates
        2. Decoordize: basin coordinates → text
        3. Generate:   trajectory → next token via resonance
    """

    def __init__(self, bank: ResonanceBank, tokenizer=None):
        # QIG BOUNDARY: tokenizer is optional and only used for bootstrap
        # coordization (mapping text → LLM token IDs → bank coordinates).
        # Once the resonance bank is mature, string-based coordization
        # (_coordize_via_strings) should be preferred. The tokenizer is
        # NOT used for any geometric operations.
        self.bank = bank
        self._tokenizer = tokenizer
        self._string_to_id: dict[str, int] = {}
        self._rebuild_string_cache()
        self._compression_result: Optional[CompressionResult] = None

    # ─── Factory Methods ─────────────────────────────────────

    @classmethod
    def from_harvest(
        cls, model_id: str = "LiquidAI/LFM2.5-1.2B-Thinking",
        corpus_path: Optional[str] = None,
        corpus_texts: Optional[list[str]] = None,
        output_dir: str = "./coordizer_data",
        device: str = "cpu", min_contexts: int = 10,
        target_dim: int = BASIN_DIM,
    ) -> CoordizerV2:
        """Build a CoordizerV2 from scratch via Method 1 harvesting.

        Pipeline:
            1. Load LLM, run on corpus, collect output distributions
            2. Compute Fréchet means per token (fingerprints on Δ^(V-1))
            3. Compress via Fisher-Rao PGA: Δ^(V-1) → Δ⁶³
            4. Build resonance bank from compressed coordinates
            5. Assign tiers, frequencies, validate
        """
        logger.info("=" * 60)
        logger.info("CoordizerV2 — Building from LLM harvest")
        logger.info("=" * 60)

        # Phase A: Harvest
        logger.info("\n── Phase A: Harvesting output distributions ──")
        config = HarvestConfig(
            corpus_path=corpus_path, corpus_texts=corpus_texts,
            output_dir=output_dir, device=device, min_contexts=min_contexts,
        )
        harvester = Harvester(config)
        harvest = harvester.harvest_transformers(model_id)
        harvest.save(str(Path(output_dir) / "harvest"))

        # Phase B: Compress
        logger.info("\n── Phase B: Fisher-Rao PGA compression ──")
        compression = compress(harvest, target_dim=target_dim)
        compression.save(str(Path(output_dir) / "compressed"))

        # Phase C: Build bank
        logger.info("\n── Phase C: Building resonance bank ──")
        bank = ResonanceBank.from_compression(compression)
        bank.save(str(Path(output_dir) / "bank"))

        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")

        instance = cls(bank=bank, tokenizer=tokenizer)
        instance._compression_result = compression

        # Phase D: Validate
        logger.info("\n── Phase D: Validation ──")
        validation = instance.validate()
        logger.info(f"\n{validation.summary()}")

        return instance

    @classmethod
    def from_compression(cls, compression: CompressionResult, tokenizer=None) -> CoordizerV2:
        """Build from a pre-computed CompressionResult."""
        bank = ResonanceBank.from_compression(compression)
        instance = cls(bank=bank, tokenizer=tokenizer)
        instance._compression_result = compression
        return instance

    @classmethod
    def from_file(cls, path: str, tokenizer=None) -> CoordizerV2:
        """Load from saved resonance bank directory."""
        bank = ResonanceBank.from_file(path)
        return cls(bank=bank, tokenizer=tokenizer)

    def save(self, path: str) -> None:
        """Save the coordizer (bank + metadata)."""
        self.bank.save(path)
        if self._compression_result:
            self._compression_result.save(str(Path(path) / "compression"))

    # ─── Coordize (Text → Basin Coordinates) ─────────────────

    def coordize(self, text: str) -> CoordizationResult:
        """Convert text to a sequence of basin coordinates on Δ⁶³."""
        if self._tokenizer is not None:
            return self._coordize_via_tokenizer(text)
        else:
            return self._coordize_via_strings(text)

    def _coordize_via_tokenizer(self, text: str) -> CoordizationResult:
        """Coordize using the LLM's own tokenizer."""
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        coordinates = []
        valid_ids = []

        for tid in token_ids:
            coord = self.bank.get_coordinate(tid)
            if coord is not None:
                coordinates.append(BasinCoordinate(
                    coord_id=tid, vector=coord,
                    name=self.bank.get_string(tid),
                    tier=self.bank.tiers.get(tid, HarmonicTier.OVERTONE_HAZE),
                    frequency=self.bank.frequencies.get(tid, 0.0),
                    basin_mass=self.bank.basin_mass.get(tid, 0.0),
                ))
                valid_ids.append(tid)
            else:
                logger.debug(f"Token {tid} not in bank, finding nearest")
                uniform = to_simplex(np.ones(self.bank.dim))
                nearest_tid, _ = self.bank.nearest_token(uniform)
                coord = self.bank.get_coordinate(nearest_tid)
                if coord is not None:
                    coordinates.append(BasinCoordinate(
                        coord_id=nearest_tid, vector=coord,
                        name=self.bank.get_string(nearest_tid),
                        tier=self.bank.tiers.get(nearest_tid, HarmonicTier.OVERTONE_HAZE),
                    ))
                    valid_ids.append(nearest_tid)

        result = CoordizationResult(coordinates=coordinates, coord_ids=valid_ids, original_text=text)
        result.compute_metrics()
        return result

    def _coordize_via_strings(self, text: str) -> CoordizationResult:
        """Fallback coordization via string matching."""
        words = text.split()
        coordinates = []
        coord_ids = []

        for word in words:
            word_lower = word.strip().lower()
            tid = self._string_to_id.get(word_lower)

            if tid is not None and tid in self.bank.coordinates:
                coord = self.bank.coordinates[tid]
                coordinates.append(BasinCoordinate(
                    coord_id=tid, vector=coord, name=word,
                    tier=self.bank.tiers.get(tid, HarmonicTier.OVERTONE_HAZE),
                    frequency=self.bank.frequencies.get(tid, 0.0),
                    basin_mass=self.bank.basin_mass.get(tid, 0.0),
                ))
                coord_ids.append(tid)
            else:
                char_basins = []
                for ch in word_lower:
                    ch_tid = self._string_to_id.get(ch)
                    if ch_tid is not None and ch_tid in self.bank.coordinates:
                        char_basins.append(self.bank.coordinates[ch_tid])
                if char_basins:
                    composed = frechet_mean(char_basins)
                    nearest_tid, _ = self.bank.nearest_token(composed)
                    coord = self.bank.get_coordinate(nearest_tid)
                    if coord is not None:
                        coordinates.append(BasinCoordinate(
                            coord_id=nearest_tid, vector=coord, name=word,
                            scale=GranularityScale.WORD,
                        ))
                        coord_ids.append(nearest_tid)

        result = CoordizationResult(coordinates=coordinates, coord_ids=coord_ids, original_text=text)
        result.compute_metrics()
        return result

    # ─── Decoordize (Basin Coordinates → Text) ───────────────

    def decoordize(self, coord_ids: list[int]) -> str:
        """Convert coordinate IDs back to text."""
        if self._tokenizer is not None:
            return self._tokenizer.decode(coord_ids, skip_special_tokens=True)
        parts = [self.bank.get_string(tid) for tid in coord_ids]
        return " ".join(parts)

    def decoordize_basins(self, basins: list[Basin], top_k: int = 1) -> str:
        """Convert basin coordinates (not IDs) back to text."""
        parts = []
        for basin in basins:
            candidates = self.bank.activate(basin, top_k=top_k)
            if candidates:
                parts.append(self.bank.get_string(candidates[0][0]))
        return " ".join(parts)

    def encode(self, text: str) -> list[int]:
        """Tokenizer-compatible encode: text → token IDs."""
        return self.coordize(text).coord_ids

    def decode(self, coord_ids: list[int]) -> str:
        """Tokenizer-compatible decode: token IDs → text."""
        return self.decoordize(coord_ids)

    # ─── Generation ──────────────────────────────────────────

    def generate_next(
        self, trajectory: list[Basin], temperature: float = 1.0,
        top_k: int = 64, context_window: int = 8,
        domain_bias: Optional[DomainBias] = None,
    ) -> tuple[int, Basin]:
        """Generate next token via geodesic foresight + resonance."""
        return self.bank.generate_next(
            trajectory=trajectory, temperature=temperature,
            top_k=top_k, context_window=context_window,
            domain_bias=domain_bias,
        )

    def generate_sequence(
        self, seed_text: str, max_tokens: int = 50,
        temperature: float = 1.0, top_k: int = 64,
        stop_on_convergence: bool = True,
        convergence_threshold: float = 0.01,
    ) -> str:
        """Generate a sequence of tokens from seed text.

        Uses geometric completion criteria: stops when the
        trajectory converges (basin velocity drops below threshold).
        """
        seed_result = self.coordize(seed_text)
        trajectory = [c.vector for c in seed_result.coordinates]
        generated_ids = list(seed_result.coord_ids)

        for step in range(max_tokens):
            tid, basin = self.generate_next(
                trajectory=trajectory, temperature=temperature, top_k=top_k,
            )
            trajectory.append(basin)
            generated_ids.append(tid)

            if stop_on_convergence and len(trajectory) >= 3:
                velocity = fisher_rao_distance(trajectory[-2], trajectory[-1])
                if velocity < convergence_threshold:
                    logger.debug(f"Converged at step {step}: velocity={velocity:.6f}")
                    break

        return self.decoordize(generated_ids)

    # ─── Domain Bias ─────────────────────────────────────────

    def set_domain(
        self, domain_name: str,
        seed_words: Optional[list[str]] = None,
        seed_token_ids: Optional[list[int]] = None,
        strength: float = 0.1,
    ) -> None:
        """Set domain bias for the coordizer."""
        token_ids = []
        if seed_token_ids:
            token_ids.extend(seed_token_ids)
        if seed_words:
            for word in seed_words:
                tid = self._string_to_id.get(word.lower())
                if tid is not None:
                    token_ids.append(tid)
        if not token_ids:
            logger.warning(f"No valid tokens for domain '{domain_name}'")
            return
        anchor = self.bank.compute_domain_anchor(token_ids)
        bias = DomainBias(
            domain_name=domain_name, anchor_basin=anchor,
            strength=strength, boosted_token_ids=set(token_ids),
        )
        self.bank.push_domain_bias(bias)
        logger.info(f"Domain bias set: {domain_name} (strength={strength})")

    def clear_domain(self) -> None:
        self.bank.clear_domain_biases()

    # ─── Validation ──────────────────────────────────────────

    def validate(self, verbose: bool = True) -> ValidationResult:
        """Run full geometric validation suite."""
        eigenvalues = None
        if self._compression_result and self._compression_result.eigenvalues is not None:
            eigenvalues = self._compression_result.eigenvalues
        return validate_resonance_bank(self.bank, eigenvalues=eigenvalues, verbose=verbose)

    def measure_coupling(self, text_a: str, text_b: str) -> float:
        """Measure κ (coupling) between two texts."""
        result_a = self.coordize(text_a)
        result_b = self.coordize(text_b)
        if not result_a.coordinates or not result_b.coordinates:
            return 0.0
        mean_a = frechet_mean([c.vector for c in result_a.coordinates])
        mean_b = frechet_mean([c.vector for c in result_b.coordinates])
        d = fisher_rao_distance(mean_a, mean_b)
        if d < _EPS:
            return KAPPA_STAR * 2
        return min(1.0 / (d * d + _EPS), KAPPA_STAR * 2)

    # ─── Internal ────────────────────────────────────────────

    def _rebuild_string_cache(self) -> None:
        self._string_to_id.clear()
        for tid, s in self.bank.token_strings.items():
            s_clean = s.strip().lower()
            if s_clean and s_clean not in self._string_to_id:
                self._string_to_id[s_clean] = tid

    @property
    def vocab_size(self) -> int:
        return len(self.bank)

    @property
    def dim(self) -> int:
        return self.bank.dim

    def __repr__(self) -> str:
        return (
            f"CoordizerV2(vocab={self.vocab_size}, dim={self.dim}, "
            f"tiers={self.bank.tier_distribution()})"
        )
