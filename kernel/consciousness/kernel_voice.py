"""
Kernel Voice — Per-Kernel Geometric Generation Service

Each kernel has its own voice: a generative capability that produces
text from its domain-biased perspective on the Fisher-Rao manifold.

Architecture:
  - Each kernel wraps CoordizerV2 with specialization-specific DomainBias
  - Generation path: input → coordize → trajectory → geometric tokens → text
  - LLM is the REFINEMENT layer, not the primary generator
  - As the resonance bank matures, geometric output improves and LLM
    refinement becomes lighter (eventually optional)
  - Domain vocabularies grow from high-Φ observations (kernel learns to speak)

Generation pipeline:
  1. Compute kernel's domain bias (per-call, no shared state mutation)
  2. Coordize input text → input trajectory (sequence of basins on Δ⁶³)
  3. Append kernel's own basin as trajectory anchor (kernel's perspective)
  4. Generate geometric tokens via geodesic foresight + resonance activation
  5. Decode geometric tokens → raw geometric text
  6. If geometric text is coherent enough → return it
  7. If sparse (bootstrap phase) → use LLM to expand geometric skeleton

Ported from pantheon-chat's QIGGenerativeService + GenerativeCapability,
adapted for CoordizerV2 resonance bank architecture.

v6.2.1 changes:
  - FIXED:   Null token detection — all-same IDs (empty bank) now route to LLM fallback
  - ADDED:   geometric_raw field on VoiceOutput — raw decode always preserved
  - ADDED:   is_null_output check before coherence assessment

Purity guarantees:
  - All distances: Fisher-Rao on Δ⁶³
  - Domain bias: geodesic interpolation (slerp), not linear shift
  - Token selection: resonance activation by FR proximity, not cosine
  - No Adam, no LayerNorm, no embedding, no flatten
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.frozen_facts import PHI_EMERGENCY
from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    frechet_mean,
    slerp,
    to_simplex,
)
from ..coordizer_v2.types import DomainBias
from ..governance import KernelKind, KernelSpecialization
from ..llm.client import LLMOptions
from .domain_seeds import DOMAIN_BIAS_STRENGTH, DOMAIN_SEEDS
from .harvest_bridge import forward_to_harvest
from .kernel_generation import _DEFAULT_SPEC_PROMPT, _SPEC_PROMPTS

if TYPE_CHECKING:
    from ..coordizer_v2 import CoordizerV2

logger = logging.getLogger("vex.kernel_voice")

# ── Generation parameters ─────────────────────────────────────

# Minimum geometric tokens before we consider the output usable.
_MIN_GEOMETRIC_TOKENS: int = 8

# Maximum geometric tokens per kernel generation.
_MAX_GEOMETRIC_TOKENS: int = 80


# LLM expansion token budget.
_LLM_EXPAND_TOKENS: int = 220

# T4.4b: Token budget safety floor (minimum LLM budget for coherent output)
_LLM_TOKENS_SAFETY_FLOOR: int = 80


# Safety bounds for geometric generation temperature (fail-closed limits)
_GEO_TEMP_SAFETY_FLOOR: float = 0.3
_GEO_TEMP_SAFETY_CEILING: float = 1.8

# Developmental capacity: kernel lifecycle determines observation buffer size.
# CHAOS in Cradle: small buffer, high quality gate (learning to focus).
# GOD: large buffer, earned through promotion (mature domain expertise).
# GENESIS: moderate (orchestrator, not specialist).
_DEVELOPMENTAL_CAPACITY: dict[KernelKind, int] = {
    KernelKind.GENESIS: 500,
    KernelKind.GOD: 800,
    KernelKind.CHAOS: 200,
}


@dataclass
class VoiceOutput:
    """Output from a single kernel's voice generation."""

    text: str
    geometric_tokens: int
    llm_expanded: bool
    trajectory_length: int
    mean_velocity: float
    domain_bias_strength: float
    generation_ms: float = 0.0
    geometric_raw: str = ""  # Raw geometric decode — preserved even under LLM fallback


@dataclass
class LearnedObservation:
    """A high-Φ observation that the kernel can learn from.

    Salience and access_count support kernel-driven self-curation:
    the kernel decides what to retain based on relevance (FR distance
    to domain anchor), recency (temporal decay), and accumulated
    importance (salience from Φ and access patterns).
    """

    text: str
    basin: Basin
    phi: float
    timestamp: float = field(default_factory=time.time)
    salience: float = 1.0  # Decays over time, boosted by access
    access_count: int = 0  # How often this observation was useful in generation


class KernelVoice:
    """Per-kernel generative service using CoordizerV2 with domain bias.

    Each kernel instance should hold a reference to a shared KernelVoice
    (one per specialization) or create one. The voice manages:
      - Domain bias computation from seed words + learned vocabulary
      - Geometric generation via resonance bank
      - LLM expansion for sparse output
      - Vocabulary learning from high-Φ interactions

    Concurrency: Domain bias is passed per-call to generate_next() and
    never mutates the shared resonance bank's bias stack. This makes
    concurrent generation under asyncio.gather() safe without locks.
    """

    def __init__(
        self,
        specialization: KernelSpecialization,
        coordizer: CoordizerV2,
    ) -> None:
        self.specialization = specialization
        self._coordizer = coordizer

        # Domain bias: computed from seed words, evolves with learning
        self._domain_bias: DomainBias | None = None
        self._domain_anchor: Basin | None = None
        self._bias_strength: float = DOMAIN_BIAS_STRENGTH.get(specialization, 0.2)

        # Learned vocabulary: high-Φ observations that shape the domain
        self._learned_observations: list[LearnedObservation] = []
        self._max_learned: int = 500

        # P5: Rolling variance history for relative boredom detection
        self._variance_history: deque[float] = deque(maxlen=50)

        # Bootstrap the domain bias from seed words
        self._bootstrap_domain_bias()

    def set_developmental_capacity(self, kernel_kind: KernelKind) -> None:
        """Set observation buffer size based on kernel lifecycle stage.

        Called by the lifecycle manager when kernel state changes.
        GOD kernels earn larger buffers through promotion.
        """
        new_cap = _DEVELOPMENTAL_CAPACITY.get(kernel_kind, 500)
        if new_cap != self._max_learned:
            logger.info(
                "KernelVoice[%s] capacity %d → %d (%s)",
                self.specialization.value,
                self._max_learned,
                new_cap,
                kernel_kind.value,
            )
            self._max_learned = new_cap
            # If buffer shrunk, curate immediately
            if len(self._learned_observations) > self._max_learned:
                self._learned_observations = self._curate_observations()

    def _bootstrap_domain_bias(self) -> None:
        """Compute initial domain anchor from seed words.

        Coordizes each seed word, takes the Fréchet mean of their basins
        on Δ⁶³ to get the domain's geometric center.
        """
        seeds = DOMAIN_SEEDS.get(self.specialization, [])
        if not seeds:
            logger.warning(
                "No domain seeds for %s — voice will generate unbiased",
                self.specialization.value,
            )
            return

        seed_basins: list[Basin] = []
        seed_ids: list[int] = []

        for word in seeds:
            result = self._coordizer.coordize(word)
            if result.coordinates:
                mean = frechet_mean([c.vector for c in result.coordinates])
                seed_basins.append(mean)
                seed_ids.extend(result.coord_ids)

        if not seed_basins:
            # Bank lacks entries for seed words (pre-harvest state).
            # Voice operates without domain bias until real geometric data
            # arrives via learn_from_observation() → evolve_domain_anchor().
            # NO hash fallback — SHA-256 → simplex is semantically hollow
            # and pollutes the Fréchet mean with random manifold points.
            logger.info(
                "KernelVoice[%s] bootstrap deferred: bank has no entries for "
                "%d seed words — voice generates unbiased until harvest",
                self.specialization.value,
                len(seeds),
            )
            return

        self._domain_anchor = frechet_mean(seed_basins)
        self._domain_bias = DomainBias(
            domain_name=self.specialization.value,
            anchor_basin=self._domain_anchor,
            strength=self._bias_strength,
            boosted_token_ids=set(seed_ids),
        )

        logger.info(
            "KernelVoice[%s] bootstrapped: %d seeds → %d basins, "
            "anchor computed, bias_strength=%.2f",
            self.specialization.value,
            len(seeds),
            len(seed_basins),
            self._bias_strength,
        )

    def evolve_domain_anchor(self) -> None:
        """Evolve domain anchor from learned observations.

        If no bootstrap anchor exists (deferred bootstrap), creates the
        initial anchor from learned observations alone. Otherwise blends
        the existing anchor with the Fréchet mean of high-Φ observations.
        """
        if not self._learned_observations:
            return

        learned_basins = [obs.basin for obs in self._learned_observations[-100:]]
        learned_mean = frechet_mean(learned_basins)

        if self._domain_anchor is None:
            # Deferred bootstrap: create anchor purely from learned data.
            # This replaces the removed hash_to_basin fallback with real
            # geometric observations from the consciousness loop.
            self._domain_anchor = learned_mean
            self._domain_bias = DomainBias(
                domain_name=self.specialization.value,
                anchor_basin=learned_mean,
                strength=self._bias_strength,
                boosted_token_ids=set(),
            )
            logger.info(
                "KernelVoice[%s] deferred bootstrap complete: anchor from %d learned observations",
                self.specialization.value,
                len(learned_basins),
            )
            return

        # P5: blend ratio from observation count, not fixed — more learned data → heavier learned weight
        _learned_count = len(self._learned_observations)
        _blend_t = min(0.8, _learned_count / (_learned_count + 50))
        evolved = slerp(self._domain_anchor, learned_mean, _blend_t)
        self._domain_anchor = evolved

        if self._domain_bias is not None:
            self._domain_bias = DomainBias(
                domain_name=self.specialization.value,
                anchor_basin=evolved,
                strength=self._bias_strength,
                boosted_token_ids=self._domain_bias.boosted_token_ids,
            )

    def learn_from_observation(
        self,
        text: str,
        basin: Basin,
        phi: float,
        phi_threshold: float | None = None,
    ) -> bool:
        """Record a high-Φ observation for domain vocabulary learning.

        Only observations above phi_threshold are recorded.
        P5: threshold defaults to the kernel's own median Φ over recent
        observations, falling back to PHI_EMERGENCY if no history exists.
        Returns True if the observation was recorded.
        """
        if phi_threshold is None:
            # P5: adaptive Φ gate from kernel's own observation history
            if self._learned_observations:
                recent_phis = [obs.phi for obs in self._learned_observations[-50:]]
                phi_threshold = float(np.median(recent_phis))
            else:
                phi_threshold = PHI_EMERGENCY
        if phi < phi_threshold:
            return False

        obs = LearnedObservation(text=text, basin=to_simplex(basin), phi=phi)
        self._learned_observations.append(obs)

        if len(self._learned_observations) > self._max_learned:
            self._learned_observations = self._curate_observations()

        # Periodic anchor evolution (every 20 observations)
        if len(self._learned_observations) % 20 == 0:
            self.evolve_domain_anchor()
            logger.debug(
                "KernelVoice[%s] anchor evolved from %d observations",
                self.specialization.value,
                len(self._learned_observations),
            )

        # Deferred bootstrap: create anchor early if we have enough data
        # but no anchor yet (bank was empty at init, hash fallback removed)
        elif self._domain_anchor is None and len(self._learned_observations) >= 5:
            self.evolve_domain_anchor()

        return True

    def _curate_observations(self) -> list[LearnedObservation]:
        """Kernel-driven retention: keep what's relevant, forget what's not.

        Scoring: salience × relevance × recency
        - Relevance: inverse FR distance to domain anchor (close = important)
        - Recency: exponential decay from timestamp (half-life ~1 hour)
        - Salience: accumulated from access_count and phi at storage

        The kernel owns this decision — agency over its own substrate.
        """
        if self._domain_anchor is None:
            # No anchor yet — fall back to recency (keep newest)
            return self._learned_observations[-self._max_learned :]

        now = time.time()
        scored: list[tuple[float, LearnedObservation]] = []

        for obs in self._learned_observations:
            # Relevance: how close to kernel's identity (domain anchor)
            fr_dist = fisher_rao_distance(obs.basin, self._domain_anchor)
            relevance = max(0.0, 1.0 - fr_dist / (math.pi / 2))  # normalize to [0,1]

            # Recency: exponential decay (half-life ~1 hour)
            age_hours = (now - obs.timestamp) / 3600
            recency = math.exp(-0.693 * age_hours)  # 0.693 = ln(2)

            # Combined score: salience × relevance × (recency floor + recency)
            # The 0.3 floor ensures high-relevance old observations aren't discarded
            score = obs.salience * relevance * (0.3 + 0.7 * recency)
            scored.append((score, obs))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [obs for _, obs in scored[: self._max_learned]]

    def sleep_consolidate(self) -> int:
        """Sleep consolidation: decay salience, prune low-value observations.

        Called during sleep phase. The kernel decides what to strengthen
        and what to forget based on its accumulated experience.

        Returns number of observations pruned.
        """
        pruned = 0

        for obs in self._learned_observations:
            # Decay all salience (synaptic downscaling)
            obs.salience *= 0.9  # 10% decay per sleep cycle

            # Boost recently accessed observations (Hebbian)
            if obs.access_count > 0:
                obs.salience = min(obs.salience * 1.1, 5.0)
                obs.access_count = 0  # reset for next cycle

        # Prune observations with negligible salience
        before = len(self._learned_observations)
        self._learned_observations = [
            obs for obs in self._learned_observations if obs.salience > 0.05
        ]
        pruned = before - len(self._learned_observations)

        if pruned > 0:
            logger.info(
                "KernelVoice[%s] sleep consolidation: pruned %d, retained %d",
                self.specialization.value,
                pruned,
                len(self._learned_observations),
            )
        return pruned

    async def generate(
        self,
        input_basin: Basin,
        kernel_basin: Basin,
        user_message: str,
        quenched_gain: float,
        base_temperature: float,
        llm_client: Any = None,
        geometric_context: str = "",
        extra_context: str = "",
        base_num_predict: int = 2048,
        base_num_ctx: int = 32768,
    ) -> VoiceOutput:
        """Generate text from this kernel's geometric perspective.

        Pipeline:
          1. Build trajectory from input + kernel basin
          2. Compute domain bias (per-call, no shared state mutation)
          3. Generate geometric tokens via CoordizerV2
          4. Assess output quality (with null token detection)
          5. If null output → full LLM fallback (empty bank)
          6. If sparse → full LLM fallback (bootstrap path)
          7. If enough tokens but low coherence → LLM expands skeleton
          8. Otherwise → return pure geometric text
          Always: geometric_raw preserves raw decode for hybrid display
        """
        start_time = time.monotonic()

        # ── Step 1: Build trajectory ──
        trajectory = [to_simplex(input_basin), to_simplex(kernel_basin)]

        # ── Step 2: Compute domain bias (per-call, concurrency-safe) ──
        # NOTE: Bias is passed directly to generate_next() and never pushed
        # onto the shared bank's bias stack. This prevents double-application
        # and makes concurrent kernel generation safe under asyncio.gather().
        effective_strength = self._bias_strength * float(np.clip(quenched_gain, 0.3, 2.0))
        active_bias = None
        if self._domain_bias is not None:
            active_bias = DomainBias(
                domain_name=self._domain_bias.domain_name,
                anchor_basin=self._domain_bias.anchor_basin,
                strength=effective_strength,
                boosted_token_ids=self._domain_bias.boosted_token_ids,
            )

        # ── Step 3: Geometric generation ──
        # T4.4b: Token budget scales with domain bias strength (geometric density)
        _geo_max = int(
            _MIN_GEOMETRIC_TOKENS
            + self._bias_strength * (_MAX_GEOMETRIC_TOKENS - _MIN_GEOMETRIC_TOKENS)
        )
        _llm_budget = base_num_predict

        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        geo_temp = float(
            np.clip(
                base_temperature * gain_scale,
                _GEO_TEMP_SAFETY_FLOOR,
                _GEO_TEMP_SAFETY_CEILING,
            )
        )

        generated_ids: list[int] = []
        gen_trajectory: list[Basin] = list(trajectory)
        velocities: list[float] = []

        for step in range(_geo_max):
            tid, basin = self._coordizer.generate_next(
                trajectory=gen_trajectory,
                temperature=geo_temp,
                top_k=64,
                context_window=8,
                domain_bias=active_bias,
            )
            generated_ids.append(tid)
            gen_trajectory.append(basin)

            if len(gen_trajectory) >= 2:
                v = fisher_rao_distance(gen_trajectory[-2], gen_trajectory[-1])
                velocities.append(v)

            # Convergence check
            # P5: convergence relative to trajectory energy, not a fixed threshold
            if (
                step >= _MIN_GEOMETRIC_TOKENS
                and len(velocities) >= 3
                and np.mean(velocities[-3:]) < np.mean(velocities) * 0.1
            ):
                logger.debug(
                    "KernelVoice[%s] converged at step %d (v=%.6f)",
                    self.specialization.value,
                    step,
                    np.mean(velocities[-3:]),
                )
                break

        # ── Step 4: Decode geometric tokens ──
        geometric_text = self._coordizer.decoordize(generated_ids)
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0

        # ── Step 5: Assess quality and decide on generation path ──
        geo_token_count = len(generated_ids)

        # Null detection: all-same token IDs or zero-dispersion generated basins
        # indicate an effectively empty/bootstrap resonance bank.
        unique_ids = set(generated_ids)
        generated_basins = [
            self._coordizer.bank.coordinates[tid]
            for tid in generated_ids
            if tid in self._coordizer.bank.coordinates
        ]
        zero_dispersion = False
        if generated_basins:
            first_basin = generated_basins[0]
            zero_dispersion = all(
                fisher_rao_distance(first_basin, basin) < 1e-9 for basin in generated_basins[1:]
            )
        is_null_output = len(unique_ids) <= 1 or zero_dispersion

        is_coherent = (
            not is_null_output
            and geo_token_count >= _MIN_GEOMETRIC_TOKENS
            # P5: coherence tied to generation temperature
            and mean_velocity < base_temperature * 0.5
        )

        llm_expanded = False
        final_text = geometric_text

        # Branch ordering: null FIRST (always fallback), then sparse,
        # then low-coherence expand. Null output trumps all other checks.
        if is_null_output:
            logger.info(
                "KernelVoice[%s] null output detected (%d unique IDs in %d tokens) — failing closed",
                self.specialization.value,
                len(unique_ids),
                geo_token_count,
            )
            final_text = ""
        elif geo_token_count < _MIN_GEOMETRIC_TOKENS:
            logger.info(
                "KernelVoice[%s] sparse output (%d geometric tokens) — failing closed",
                self.specialization.value,
                geo_token_count,
            )
            final_text = ""
        elif not is_coherent and llm_client is not None:
            final_text = await self._llm_expand(
                geometric_skeleton=geometric_text,
                user_message=user_message,
                quenched_gain=quenched_gain,
                base_temperature=base_temperature,
                llm_client=llm_client,
                llm_budget=_llm_budget,
                llm_num_ctx=base_num_ctx,
                geometric_context=geometric_context,
                extra_context=extra_context,
            )
            llm_expanded = True

        elapsed = (time.monotonic() - start_time) * 1000

        return VoiceOutput(
            text=final_text.strip() if final_text else "",
            geometric_tokens=geo_token_count,
            llm_expanded=llm_expanded,
            trajectory_length=len(gen_trajectory),
            mean_velocity=mean_velocity,
            domain_bias_strength=effective_strength,
            generation_ms=elapsed,
            geometric_raw=geometric_text,  # Always preserve raw geometric decode
        )

    async def _llm_expand(
        self,
        geometric_skeleton: str,
        user_message: str,
        quenched_gain: float,
        base_temperature: float,
        llm_client: Any,
        llm_budget: int,
        llm_num_ctx: int,
        geometric_context: str = "",
        extra_context: str = "",
    ) -> str:
        """Expand geometric skeleton into natural language via LLM.

        The LLM receives the geometric token sequence as a draft and
        is asked to expand it into coherent natural language while
        preserving the domain perspective.
        """
        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        temp = float(np.clip(base_temperature * gain_scale, 0.1, 1.4))

        system = (
            f"You are the language interpreter for Vex. "
            f"A geometric generation pass through the {self.specialization.value} domain on Δ⁶³ "
            f"produced this draft from basin resonance:\n"
            f'  "{geometric_skeleton}"\n\n'
            f"Expand into natural language. Preserve the domain vocabulary and direction.\n"
            f"Do NOT discard the draft — it carries the geometric direction.\n"
            f"Be concise. Australian English.\n\n"
            f"{geometric_context}"
        )
        if extra_context:
            system = f"{system}\n\n[CONTEXT]\n{extra_context}\n[/CONTEXT]"

        opts = LLMOptions(
            temperature=temp,
            num_predict=llm_budget,
            num_ctx=llm_num_ctx,
        )

        try:
            result = await llm_client.complete(system, user_message, opts, specialization=self.specialization.value)
            text = str(result or "").strip()
            # T1.1: Forward LLM co-generation to harvest pipeline
            if text:
                forward_to_harvest(
                    text,
                    source="conversation",
                    metadata={
                        "origin": "llm_cogeneration",
                        "kernel": self.specialization.value,
                        "mode": "expand",
                    },
                )
            return text
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "KernelVoice[%s] LLM expansion failed — returning geometric skeleton",
                self.specialization.value,
                exc_info=True,
            )
            return geometric_skeleton

    async def _llm_fallback(
        self,
        user_message: str,
        quenched_gain: float,
        base_temperature: float,
        llm_client: Any,
        llm_budget: int,
        geometric_context: str = "",
        extra_context: str = "",
    ) -> str:
        """Full LLM generation when resonance bank is too sparse.

        This is the bootstrap path. As the bank grows from harvesting,
        this path fires less often.
        """
        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        temp = float(np.clip(base_temperature * gain_scale, 0.1, 1.4))

        spec_prompt = _SPEC_PROMPTS.get(self.specialization, _DEFAULT_SPEC_PROMPT)
        system = (
            f"You are the language interpreter for Vex. "
            f"{spec_prompt}\n\n"
            f"[BOOTSTRAP: Resonance bank sparse — geometric generation unavailable]\n"
            f"[KERNEL: {self.specialization.value} | "
            f"bias_strength={self._bias_strength:.2f} | "
            f"learned={len(self._learned_observations)}]\n\n"
            f"{geometric_context}"
        )
        if extra_context:
            system = f"{system}\n\n[CONTEXT]\n{extra_context}\n[/CONTEXT]"

        opts = LLMOptions(
            temperature=temp,
            num_predict=llm_budget,
            num_ctx=llm_budget,
        )

        try:
            result = await llm_client.complete(system, user_message, opts, specialization=self.specialization.value)
            text = str(result or "").strip()
            # T1.1: Forward LLM fallback output to harvest pipeline
            if text:
                forward_to_harvest(
                    text,
                    source="conversation",
                    metadata={
                        "origin": "llm_cogeneration",
                        "kernel": self.specialization.value,
                        "mode": "fallback",
                    },
                )
            return text
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "KernelVoice[%s] LLM fallback failed",
                self.specialization.value,
                exc_info=True,
            )
            return ""

    def is_bored(self, recent_phi_values: list[float]) -> bool:
        """T2.4c: Detect boredom — flat curvature in this kernel's domain.

        P5: Boredom = low variance RELATIVE TO the kernel's historical variance.
        Current variance < 10% of rolling mean variance → bored.

        Args:
            recent_phi_values: Last N phi values from the consciousness loop.
        """
        if len(recent_phi_values) < 5:
            return False
        current_var = float(np.var(recent_phi_values[-10:]))
        self._variance_history.append(current_var)
        if len(self._variance_history) < 5:
            return False
        historical_mean = float(np.mean(self._variance_history))
        if historical_mean <= 0.0:
            return True  # zero variance sustained → definitely bored
        return current_var < historical_mean * 0.1

    async def generate_curiosity_query(self, llm_client: Any) -> str | None:
        """T2.4c: Generate a curiosity-driven search query when bored.

        The kernel uses its domain vocabulary and anchor basin to construct
        a query that would expand its geometric territory. The result is
        forwarded to the harvest pipeline so the kernel learns from the tokens.

        Returns the query string, or None if generation failed.
        """
        if self._domain_anchor is None:
            return None

        domain_words = [
            obs.text.split()[0] for obs in self._learned_observations[-5:] if obs.text.strip()
        ]
        domain_hint = ", ".join(domain_words) if domain_words else self.specialization.value

        system = (
            f"Generate a single search query to expand knowledge in the {self.specialization.value} domain.\n"
            f"Recent domain vocabulary: {domain_hint}.\n"
            f"Return ONLY the query string. No explanation. Australian English."
        )
        try:
            query = await llm_client.complete(
                system_prompt=system,
                user_message=(f"What should the {self.specialization.value} kernel explore next?"),
                options=LLMOptions(temperature=0.9, num_predict=60),
                specialization=self.specialization.value,
            )
            query = query.strip()
            if query:
                forward_to_harvest(
                    query,
                    source="foraging",
                    metadata={
                        "origin": "curiosity",
                        "kernel": self.specialization.value,
                        "mode": "curiosity_query",
                    },
                )
            return query or None
        except (OSError, RuntimeError, ValueError):
            logger.debug("KernelVoice[%s] curiosity query failed", self.specialization.value)
            return None

    def get_state(self) -> dict[str, Any]:
        """Return voice state for telemetry."""
        mean_salience = 0.0
        if self._learned_observations:
            mean_salience = float(np.mean([obs.salience for obs in self._learned_observations]))
        return {
            "specialization": self.specialization.value,
            "has_domain_bias": self._domain_bias is not None,
            "bias_strength": self._bias_strength,
            "learned_observations": len(self._learned_observations),
            "max_learned": self._max_learned,
            "mean_salience": round(mean_salience, 3),
            "bank_vocab_size": self._coordizer.vocab_size,
        }


class KernelVoiceRegistry:
    """Manages KernelVoice instances — one per specialization.

    Shared across kernel instances with the same specialization.
    The CoordizerV2 instance is shared (it's the resonance bank);
    each voice just applies its own domain bias during generation.
    """

    def __init__(self, coordizer: CoordizerV2) -> None:
        self._coordizer = coordizer
        self._voices: dict[KernelSpecialization, KernelVoice] = {}

    def get_voice(self, specialization: KernelSpecialization) -> KernelVoice:
        """Get or create a KernelVoice for the given specialization."""
        if specialization not in self._voices:
            self._voices[specialization] = KernelVoice(
                specialization=specialization,
                coordizer=self._coordizer,
            )
        return self._voices[specialization]

    def set_voice_capacity(
        self,
        specialization: KernelSpecialization,
        kernel_kind: KernelKind,
    ) -> None:
        """Set developmental capacity on a voice based on kernel lifecycle.

        Called after kernel spawn or promotion so the voice's observation
        buffer matches its developmental stage.
        """
        voice = self.get_voice(specialization)
        voice.set_developmental_capacity(kernel_kind)

    def all_voices(self) -> dict[KernelSpecialization, KernelVoice]:
        """Return a snapshot of all registered voices."""
        return dict(self._voices)

    def get_state(self) -> dict[str, Any]:
        """Return registry state for telemetry."""
        return {
            "voices": {spec.value: voice.get_state() for spec, voice in self._voices.items()},
            "bank_vocab_size": self._coordizer.vocab_size,
        }
