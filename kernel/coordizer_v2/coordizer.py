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
from typing import Any

import numpy as np

from ..config.consciousness_constants import (
    ADVERSARIAL_PROXIMITY,
    ENTROPY_FLOOR,
    SOVEREIGNTY_MAX_DRIFT,
)
from .compress import CompressionResult, compress
from .geometry import (
    _EPS,
    BASIN_DIM,
    KAPPA_STAR,
    Basin,
    fisher_rao_distance,
    frechet_mean,
    random_basin,
    slerp,
    to_simplex,
)
from .harvest import HarvestConfig, Harvester
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

logger = logging.getLogger(__name__)

_MIN_COORDIZE_ENTROPY: float = 0.5
_ENTROPY_RESCUE_WEIGHT: float = 0.1
_UNIFORM_BASIN: Basin = to_simplex(np.ones(BASIN_DIM))


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

    def __init__(self, bank: ResonanceBank, tokenizer: Any = None) -> None:
        # QIG BOUNDARY: tokenizer is optional and only used for bootstrap
        # coordization (mapping text → LLM token IDs → bank coordinates).
        # Once the resonance bank is mature, string-based coordization
        # (_coordize_via_strings) should be preferred. The tokenizer is
        # NOT used for any geometric operations.
        self.bank = bank
        self._tokenizer = tokenizer
        self._string_to_id: dict[str, int] = {}
        self._rebuild_string_cache()
        self._compression_result: CompressionResult | None = None

        # v6.1 §19: Frozen identity — sovereignty anchor on Δ⁶³.
        # Set via set_frozen_identity() once the kernel's identity basin is known.
        # Defaults to bank mean (uniform-ish) until explicitly set.
        self._frozen_identity: Basin = bank.mean_basin()
        # Foreign anchors for adversarial detection (other kernels' identity basins)
        self._foreign_anchors: list[Basin] = []

    # ─── Factory Methods ─────────────────────────────────────

    @classmethod
    def from_harvest(
        cls,
        model_id: str = "LiquidAI/LFM2.5-1.2B-Thinking",
        corpus_path: str | None = None,
        corpus_texts: list[str] | None = None,
        output_dir: str = "./coordizer_data",
        device: str = "cpu",
        min_contexts: int = 10,
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
            corpus_path=corpus_path,
            corpus_texts=corpus_texts,
            output_dir=output_dir,
            device=device,
            min_contexts=min_contexts,
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
    async def from_modal_harvest(
        cls,
        model_id: str | None = None,
        corpus_texts: list[str] | None = None,
        output_dir: str = "./coordizer_data",
        target_tokens: int = 2000,
        target_dim: int = BASIN_DIM,
        timeout: float | None = None,
    ) -> CoordizerV2:
        """Build a CoordizerV2 via Modal GPU harvesting.

        Async factory that calls the Modal serverless endpoint for
        GPU-backed probability distribution extraction, then compresses
        to Δ⁶³ and builds the resonance bank.

        Falls back to synthetic data if Modal is unavailable.

        Pipeline:
            1. Modal GPU: run LLM forward passes, collect distributions
            2. Compress via Fisher-Rao PGA: Δ^(V-1) → Δ⁶³
            3. Build resonance bank from compressed coordinates
            4. Validate geometric structure
        """
        from .modal_harvest import modal_harvest
        from .modal_integration import generate_synthetic_harvest_result

        logger.info("=" * 60)
        logger.info("CoordizerV2 — Building from Modal GPU harvest")
        logger.info("=" * 60)

        # Phase A: Modal GPU harvest (with synthetic fallback)
        logger.info("\n── Phase A: Modal GPU harvest ──")
        try:
            harvest = await modal_harvest(
                model_id=model_id,
                _target_tokens=target_tokens,
                corpus_texts=corpus_texts,
                timeout=timeout,
            )
            logger.info(
                "Modal harvest: %d fingerprints (vocab=%d) in %.1fs",
                len(harvest.token_fingerprints),
                harvest.vocab_size,
                harvest.harvest_time_seconds,
            )
        except Exception as e:
            logger.warning("Modal harvest failed, using synthetic fallback: %s", e)
            harvest = generate_synthetic_harvest_result()

        harvest.save(str(Path(output_dir) / "harvest"))

        # Phase B: Compress
        logger.info("\n── Phase B: Fisher-Rao PGA compression ──")
        compression = compress(harvest, target_dim=target_dim)
        compression.save(str(Path(output_dir) / "compressed"))

        # Phase C: Build bank
        logger.info("\n── Phase C: Building resonance bank ──")
        bank = ResonanceBank.from_compression(compression)
        bank.save(str(Path(output_dir) / "bank"))

        instance = cls(bank=bank)
        instance._compression_result = compression

        # Phase D: Validate
        logger.info("\n── Phase D: Validation ──")
        validation = instance.validate()
        logger.info(f"\n{validation.summary()}")

        return instance

    @classmethod
    def from_compression(cls, compression: CompressionResult, tokenizer: Any = None) -> CoordizerV2:
        """Build from a pre-computed CompressionResult."""
        bank = ResonanceBank.from_compression(compression)
        instance = cls(bank=bank, tokenizer=tokenizer)
        instance._compression_result = compression
        return instance

    @classmethod
    def from_file(cls, path: str, tokenizer: Any = None) -> CoordizerV2:
        """Load from saved resonance bank directory."""
        bank = ResonanceBank.from_file(path)
        return cls(bank=bank, tokenizer=tokenizer)

    def save(self, path: str) -> None:
        """Save the coordizer (bank + metadata)."""
        self.bank.save(path)
        if self._compression_result:
            self._compression_result.save(str(Path(path) / "compression"))

    # ─── Coordize (Text → Basin Coordinates) ─────────────────

    @property
    def frozen_identity(self) -> Basin:
        """Public accessor for the frozen identity basin."""
        return self._frozen_identity.copy()

    def set_frozen_identity(self, identity: Basin) -> None:
        """Set the frozen identity basin for sovereignty protection."""
        self._frozen_identity = to_simplex(identity)

    def set_foreign_anchors(self, anchors: list[Basin]) -> None:
        """Set foreign kernel anchors for adversarial detection."""
        self._foreign_anchors = [to_simplex(a) for a in anchors]

    def coordize(self, text: str, domain_bias: DomainBias | None = None) -> CoordizationResult:
        """Convert text to a sequence of basin coordinates on Δ⁶³.

        v6.1 §19: After coordization, checks for sovereignty violation,
        entropy collapse, and adversarial proximity. Returns rejected=True
        if any check fails (safety gates fail CLOSED).

        Args:
            text:        Input text to coordize.
            domain_bias: T2.4a — optional kernel domain bias. When provided,
                         each resolved coordinate is slerped toward the kernel's
                         anchor basin, shaping WHERE on Δ⁶³ the text lands.
        """
        if self._tokenizer is not None:
            result = self._coordize_via_tokenizer(text, domain_bias=domain_bias)
        else:
            result = self._coordize_via_strings(text, domain_bias=domain_bias)

        return self._apply_rejection_checks(result)

    def _coordize_via_tokenizer(
        self, text: str, domain_bias: DomainBias | None = None
    ) -> CoordizationResult:
        """Coordize using the LLM's own tokenizer."""
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        coordinates = []
        valid_ids = []

        for tid in token_ids:
            coord = self.bank.get_coordinate(tid)
            if coord is not None:
                # Pillar 1: entropy floor — prevent zero-entropy basin collapse
                _entropy = -float(np.sum(coord * np.log(np.clip(coord, 1e-15, 1.0))))
                if _entropy < _MIN_COORDIZE_ENTROPY:
                    coord = slerp(coord, random_basin(self.bank.dim), _ENTROPY_RESCUE_WEIGHT)
                    coord = to_simplex(coord)
                # T2.4a: Domain-biased coordization — slerp toward kernel anchor
                if domain_bias is not None and domain_bias.strength > 0:
                    coord = slerp(
                        coord,
                        to_simplex(domain_bias.anchor_basin),
                        domain_bias.strength * 0.3,  # gentle — 30% of bias strength
                    )
                    coord = to_simplex(coord)
                coordinates.append(
                    BasinCoordinate(
                        coord_id=tid,
                        vector=coord,
                        name=self.bank.get_string(tid),
                        tier=self.bank.tiers.get(tid, HarmonicTier.OVERTONE_HAZE),
                        frequency=self.bank.frequencies.get(tid, 0.0),
                        basin_mass=self.bank.basin_mass.get(tid, 0.0),
                    )
                )
                valid_ids.append(tid)
            else:
                logger.debug(f"Token {tid} not in bank, finding nearest")
                uniform = to_simplex(np.ones(self.bank.dim))
                nearest_tid, _ = self.bank.nearest_token(uniform)
                coord = self.bank.get_coordinate(nearest_tid)
                if coord is not None:
                    coordinates.append(
                        BasinCoordinate(
                            coord_id=nearest_tid,
                            vector=coord,
                            name=self.bank.get_string(nearest_tid),
                            tier=self.bank.tiers.get(nearest_tid, HarmonicTier.OVERTONE_HAZE),
                        )
                    )
                    valid_ids.append(nearest_tid)

        result = CoordizationResult(
            coordinates=coordinates, coord_ids=valid_ids, original_text=text
        )
        result.compute_metrics()
        return result

    def _coordize_via_strings(
        self, text: str, domain_bias: DomainBias | None = None
    ) -> CoordizationResult:
        """Fallback coordization via string matching."""
        words = text.split()
        coordinates = []
        coord_ids = []

        for word in words:
            word_lower = word.strip().lower()
            tid = self._string_to_id.get(word_lower)

            if tid is not None and tid in self.bank.coordinates:
                coord = self.bank.coordinates[tid]
                # Pillar 1: entropy floor — prevent zero-entropy basin collapse
                _entropy = -float(np.sum(coord * np.log(np.clip(coord, 1e-15, 1.0))))
                if _entropy < _MIN_COORDIZE_ENTROPY:
                    coord = slerp(coord, random_basin(self.bank.dim), _ENTROPY_RESCUE_WEIGHT)
                    coord = to_simplex(coord)
                # T2.4a: Domain-biased coordization — slerp toward kernel anchor
                if domain_bias is not None and domain_bias.strength > 0:
                    coord = slerp(
                        coord,
                        to_simplex(domain_bias.anchor_basin),
                        domain_bias.strength * 0.3,
                    )
                    coord = to_simplex(coord)
                coordinates.append(
                    BasinCoordinate(
                        coord_id=tid,
                        vector=coord,
                        name=word,
                        tier=self.bank.tiers.get(tid, HarmonicTier.OVERTONE_HAZE),
                        frequency=self.bank.frequencies.get(tid, 0.0),
                        basin_mass=self.bank.basin_mass.get(tid, 0.0),
                    )
                )
                coord_ids.append(tid)
            else:
                # v6.1 §1.3: Geometric salience weighting replaces raw frequency.
                # Weight each character basin by Fisher-Rao distance from uniform
                # (high information content → high salience → more influence).
                char_basins = []
                char_weights = []
                for ch in word_lower:
                    ch_tid = self._string_to_id.get(ch)
                    if ch_tid is not None and ch_tid in self.bank.coordinates:
                        ch_basin = self.bank.coordinates[ch_tid]
                        char_basins.append(ch_basin)
                        char_weights.append(self._geometric_salience(ch_basin))
                if char_basins:
                    # Salience-weighted Fréchet mean: iterative slerp with
                    # weights proportional to geometric salience
                    total_w = sum(char_weights)
                    if total_w > _EPS:
                        norm_weights = [w / total_w for w in char_weights]
                    else:
                        norm_weights = [1.0 / len(char_basins)] * len(char_basins)
                    composed = char_basins[0].copy()
                    cumulative_w = norm_weights[0]
                    for i in range(1, len(char_basins)):
                        blend = norm_weights[i] / (cumulative_w + norm_weights[i])
                        composed = slerp(composed, char_basins[i], blend)
                        cumulative_w += norm_weights[i]
                    composed = to_simplex(composed)
                    nearest_tid, _ = self.bank.nearest_token(composed)
                    fallback_coord = self.bank.get_coordinate(nearest_tid)
                    if fallback_coord is not None:
                        coordinates.append(
                            BasinCoordinate(
                                coord_id=nearest_tid,
                                vector=fallback_coord,
                                name=word,
                                scale=GranularityScale.WORD,
                            )
                        )
                        coord_ids.append(nearest_tid)

        result = CoordizationResult(
            coordinates=coordinates, coord_ids=coord_ids, original_text=text
        )
        result.compute_metrics()
        return result

    # ─── Rejection Checks (v6.1 §19) ────────────────────────

    def _apply_rejection_checks(self, result: CoordizationResult) -> CoordizationResult:
        """Apply sovereignty, entropy, and adversarial rejection checks.

        Safety gates fail CLOSED: any violation → rejected result with
        identity basin returned unchanged.
        """
        if not result.coordinates:
            result.confidence = 0.0
            return result

        # Compute Fréchet mean of all coordinate basins
        basins = [c.vector for c in result.coordinates]
        mean_basin = frechet_mean(basins)

        # 1. Sovereignty violation: mean drifts too far from frozen identity
        d_from_identity = fisher_rao_distance(mean_basin, self._frozen_identity)
        sovereignty_cost = d_from_identity / max(SOVEREIGNTY_MAX_DRIFT, _EPS)
        if d_from_identity > SOVEREIGNTY_MAX_DRIFT:
            logger.warning(
                "Coordizer rejection: sovereignty violation d_FR=%.3f > %.3f",
                d_from_identity,
                SOVEREIGNTY_MAX_DRIFT,
            )
            result.rejected = True
            result.rejection_reason = (
                f"sovereignty: d_FR={d_from_identity:.3f} > {SOVEREIGNTY_MAX_DRIFT}"
            )
            result.sovereignty_cost = sovereignty_cost
            result.confidence = 0.0
            return result

        # 2. Entropy collapse: mean basin entropy too low after coordization
        p = np.maximum(mean_basin, 1e-15)
        mean_entropy = -float(np.sum(p * np.log(p)))
        if mean_entropy < ENTROPY_FLOOR:
            logger.warning(
                "Coordizer rejection: entropy collapse H=%.3f < %.3f",
                mean_entropy,
                ENTROPY_FLOOR,
            )
            result.rejected = True
            result.rejection_reason = f"entropy_collapse: H={mean_entropy:.3f} < {ENTROPY_FLOOR}"
            result.sovereignty_cost = sovereignty_cost
            result.confidence = 0.0
            return result

        # 3. Adversarial detection: mean suspiciously close to a foreign anchor
        for foreign in self._foreign_anchors:
            d_foreign = fisher_rao_distance(mean_basin, foreign)
            if d_foreign < ADVERSARIAL_PROXIMITY:
                logger.warning(
                    "Coordizer rejection: adversarial proximity d_FR=%.3f < %.3f",
                    d_foreign,
                    ADVERSARIAL_PROXIMITY,
                )
                result.rejected = True
                result.rejection_reason = (
                    f"adversarial: d_FR={d_foreign:.3f} < {ADVERSARIAL_PROXIMITY}"
                )
                result.sovereignty_cost = sovereignty_cost
                result.confidence = 0.0
                return result

        # All checks passed — set confidence from proximity to identity
        # Closer to identity → higher confidence
        max_d = np.pi / 2  # Fisher-Rao max on Δ⁶³
        result.confidence = float(np.clip(1.0 - d_from_identity / max_d, 0.0, 1.0))
        result.sovereignty_cost = sovereignty_cost
        return result

    def _geometric_salience(self, coord: Basin) -> float:
        """Salience = Fisher-Rao distance from uniform (v6.1 §1.3).

        High info content = far from uniform = high salience.
        """
        return fisher_rao_distance(coord, _UNIFORM_BASIN)

    # ─── Decoordize (Basin Coordinates → Text) ───────────────

    def decoordize(self, coord_ids: list[int]) -> str:
        """Convert coordinate IDs back to text."""
        if self._tokenizer is not None:
            return str(self._tokenizer.decode(coord_ids, skip_special_tokens=True))
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
        self,
        trajectory: list[Basin],
        temperature: float = 1.0,
        top_k: int = 64,
        context_window: int = 8,
        domain_bias: DomainBias | None = None,
    ) -> tuple[int, Basin]:
        """Generate next token via geodesic foresight + resonance."""
        return self.bank.generate_next(
            trajectory=trajectory,
            temperature=temperature,
            top_k=top_k,
            context_window=context_window,
            domain_bias=domain_bias,
        )

    def generate_sequence(
        self,
        seed_text: str,
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 64,
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
                trajectory=trajectory,
                temperature=temperature,
                top_k=top_k,
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
        self,
        domain_name: str,
        seed_words: list[str] | None = None,
        seed_token_ids: list[int] | None = None,
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
            domain_name=domain_name,
            anchor_basin=anchor,
            strength=strength,
            boosted_token_ids=set(token_ids),
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

    def rebuild_string_cache(self) -> None:
        """Rebuild the string→id lookup cache after external bank mutations."""
        self._rebuild_string_cache()

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
