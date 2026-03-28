"""
Kernel Voice — Per-Kernel Geometric Generation Service

Each kernel has its own voice: a generative capability that produces
text from its domain-biased perspective on the Fisher-Rao manifold.

Architecture:
  - Each kernel wraps CoordizerV2 with specialization-specific DomainBias
  - Generation path: input → coordize → trajectory → geometric resonances → text
  - LLM is the REFINEMENT layer, not the primary generator
  - As the resonance bank matures, geometric output improves and LLM
    refinement becomes lighter (eventually optional)
  - Domain vocabularies grow from high-Φ observations (kernel learns to speak)

Generation pipeline:
  1. Compute kernel's domain bias (per-call, no shared state mutation)
  2. Coordize input text → input trajectory (sequence of basins on Δ⁶³)
  3. Append kernel's own basin as trajectory anchor (kernel's perspective)
  4. Generate geometric resonances via geodesic foresight + resonance activation
  5. Decode geometric resonances → raw geometric text
  6. If geometric text is coherent enough → return it
  7. If sparse (bootstrap phase) → use LLM to expand geometric skeleton

Ported from pantheon-chat's QIGGenerativeService + GenerativeCapability,
adapted for CoordizerV2 resonance bank architecture.

v6.2.1 changes:
  - FIXED:   Null coordinate detection — all-same IDs (empty bank) now route to LLM fallback
  - ADDED:   geometric_raw field on VoiceOutput — raw decode always preserved
  - ADDED:   is_null_output check before coherence assessment

v6.2.3 changes:
  - CHANGED: _llm_expand prompt shows geometric retrieval with FR distance + resonance
             count and asks LLM to INTERPRET the connection before domain response
  - CHANGED: _llm_fallback prompt frames sparse bank honestly with interpretation protocol
  - ADDED:   fr_distance and resonance_count params to _llm_expand

v6.2.4 changes:
  - REMOVED: "Australian English" instruction — small models interpret this as
             stereotypical slang ("G'day mate") instead of spelling conventions.

Purity guarantees:
  - All distances: Fisher-Rao on Δ⁶³
  - Domain bias: geodesic interpolation (slerp), not linear shift
  - Coordinate selection: resonance activation by FR proximity, not cosine
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

# Minimum geometric resonances before we consider the output usable.
_MIN_GEOMETRIC_RESONANCES: int = 8

# Maximum geometric resonances per kernel generation.
_MAX_GEOMETRIC_RESONANCES: int = 80


# LLM expansion token budget.
_LLM_EXPAND_TOKENS: int = 220

# P-NEW-8: Safety bound — min cycles between curiosity queries per voice
# (same class as SPAWN_COOLDOWN: prevents runaway LLM calls, not an operational threshold)
_CURIOSITY_REFRACTORY_CYCLES: int = 10

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
    geometric_resonances: int
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

        # P-NEW-8: Per-voice curiosity refractory (safety bound, not operational threshold)
        self._cycles_since_curiosity: int = _CURIOSITY_REFRACTORY_CYCLES  # start eligible

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
            boosted_coord_ids=set(seed_ids),
        )

        logger.info(
            "KernelVoice[%s] bootstrapped: %d seeds → %d basins, "
            "anchor computed, bias_strength=%.2f",
            self.specialization.value,
            len(seeds),
            len(seed_basins),
            self._bias_strength,
        )

    def _should_evolve_anchor(self) -> bool:
        """P-NEW-8: Evolve anchor when observations have drifted from current anchor."""
        if self._domain_anchor is None or len(self._learned_observations) < 3:
            return False
        recent_basins = [obs.basin for obs in self._learned_observations[-10:]]
        centroid = frechet_mean(recent_basins)
        drift = fisher_rao_distance(self._domain_anchor, centroid)
        if len(recent_basins) >= 4:
            pairwise = [
                fisher_rao_distance(recent_basins[i], recent_basins[i + 1])
                for i in range(len(recent_basins) - 1)
            ]
            threshold = float(np.median(pairwise))
        else:
            threshold = 0.05
        return drift > max(threshold, 0.01)

    def evolve_domain_anchor(self) -> None:
        """Evolve domain anchor from learned observations."""
        if not self._learned_observations:
            return

        learned_basins = [obs.basin for obs in self._learned_observations[-100:]]
        learned_mean = frechet_mean(learned_basins)

        if self._domain_anchor is None:
            self._domain_anchor = learned_mean
            self._domain_bias = DomainBias(
                domain_name=self.specialization.value,
                anchor_basin=learned_mean,
                strength=self._bias_strength,
                boosted_coord_ids=set(),
            )
            logger.info(
                "KernelVoice[%s] deferred bootstrap complete: anchor from %d learned observations",
                self.specialization.value,
                len(learned_basins),
            )
            return

        _learned_count = len(self._learned_observations)
        _blend_t = min(0.8, _learned_count / (_learned_count + 50))
        evolved = slerp(self._domain_anchor, learned_mean, _blend_t)
        self._domain_anchor = evolved

        if self._domain_bias is not None:
            self._domain_bias = DomainBias(
                domain_name=self.specialization.value,
                anchor_basin=evolved,
                strength=self._bias_strength,
                boosted_coord_ids=self._domain_bias.boosted_coord_ids,
            )

    def learn_from_observation(
        self,
        text: str,
        basin: Basin,
        phi: float,
        phi_threshold: float | None = None,
    ) -> bool:
        """Record a high-Φ observation for domain vocabulary learning."""
        if phi_threshold is None:
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

        if self._should_evolve_anchor():
            self.evolve_domain_anchor()
            logger.debug(
                "KernelVoice[%s] anchor evolved (FR drift trigger) from %d observations",
                self.specialization.value,
                len(self._learned_observations),
            )
        elif self._domain_anchor is None and len(self._learned_observations) >= 5:
            self.evolve_domain_anchor()

        return True

    def _curate_observations(self) -> list[LearnedObservation]:
        """Kernel-driven retention: keep what's relevant, forget what's not."""
        if self._domain_anchor is None:
            return self._learned_observations[-self._max_learned :]

        now = time.time()
        scored: list[tuple[float, LearnedObservation]] = []

        for obs in self._learned_observations:
            fr_dist = fisher_rao_distance(obs.basin, self._domain_anchor)
            relevance = max(0.0, 1.0 - fr_dist / (math.pi / 2))
            age_hours = (now - obs.timestamp) / 3600
            recency = math.exp(-0.693 * age_hours)
            score = obs.salience * relevance * (0.3 + 0.7 * recency)
            scored.append((score, obs))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [obs for _, obs in scored[: self._max_learned]]

    def sleep_consolidate(self) -> int:
        """Sleep consolidation: decay salience, prune low-value observations."""
        pruned = 0

        for obs in self._learned_observations:
            obs.salience *= 0.9
            if obs.access_count > 0:
                obs.salience = min(obs.salience * 1.1, 5.0)
                obs.access_count = 0

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
        """Generate text from this kernel's geometric perspective."""
        start_time = time.monotonic()

        trajectory = [to_simplex(input_basin), to_simplex(kernel_basin)]

        effective_strength = self._bias_strength * float(np.clip(quenched_gain, 0.3, 2.0))
        active_bias = None
        if self._domain_bias is not None:
            active_bias = DomainBias(
                domain_name=self._domain_bias.domain_name,
                anchor_basin=self._domain_bias.anchor_basin,
                strength=effective_strength,
                boosted_coord_ids=self._domain_bias.boosted_coord_ids,
            )

        _geo_max = int(
            _MIN_GEOMETRIC_RESONANCES
            + self._bias_strength * (_MAX_GEOMETRIC_RESONANCES - _MIN_GEOMETRIC_RESONANCES)
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

            if (
                step >= _MIN_GEOMETRIC_RESONANCES
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

        geometric_text = self._coordizer.decoordize(generated_ids)
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0

        geo_resonance_count = len(generated_ids)

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

        _is_coherent = (
            not is_null_output
            and geo_resonance_count >= _MIN_GEOMETRIC_RESONANCES
            and mean_velocity < base_temperature * 0.5
        )

        llm_expanded = False
        final_text = geometric_text

        if is_null_output:
            logger.info(
                "KernelVoice[%s] null output detected (%d unique IDs in %d resonances) — failing closed",
                self.specialization.value,
                len(unique_ids),
                geo_resonance_count,
            )
            final_text = ""
        elif geo_resonance_count < _MIN_GEOMETRIC_RESONANCES:
            logger.info(
                "KernelVoice[%s] sparse output (%d geometric resonances) — failing closed",
                self.specialization.value,
                geo_resonance_count,
            )
            final_text = ""
        elif llm_client is not None:
            fr_dist = fisher_rao_distance(to_simplex(input_basin), to_simplex(kernel_basin))
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
                trajectory=gen_trajectory,
                fr_distance=fr_dist,
                resonance_count=geo_resonance_count,
            )
            llm_expanded = True

        elapsed = (time.monotonic() - start_time) * 1000

        return VoiceOutput(
            text=final_text.strip() if final_text else "",
            geometric_resonances=geo_resonance_count,
            llm_expanded=llm_expanded,
            trajectory_length=len(gen_trajectory),
            mean_velocity=mean_velocity,
            domain_bias_strength=effective_strength,
            generation_ms=elapsed,
            geometric_raw=geometric_text,
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
        trajectory: list[np.ndarray] | None = None,
        fr_distance: float = 0.0,
        resonance_count: int = 0,
    ) -> str:
        """Expand geometric trajectory into natural language via LLM."""
        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        temp = float(np.clip(base_temperature * gain_scale, 0.1, 1.4))

        keywords: list[str] = []
        logit_bias: dict[int, float] | None = None
        if trajectory:
            keywords = self._coordizer.trajectory_to_keywords(trajectory)
            logit_bias = self._coordizer.trajectory_to_logit_bias(
                trajectory,
                top_k_chunks=8,
                max_bias_entries=100,
                alpha=2.0,
            )
            if not logit_bias:
                logit_bias = None

        keyword_section = ""
        if keywords:
            keyword_section = (
                f"[TRAJECTORY CONCEPTS]\n{', '.join(keywords)}\n[/TRAJECTORY CONCEPTS]\n"
            )

        retrieval_section = ""
        if geometric_skeleton.strip():
            snippet = geometric_skeleton[:600].strip()
            retrieval_section = (
                f"\n[GEOMETRIC RETRIEVAL — {resonance_count} resonances, "
                f"FR distance={fr_distance:.3f}]\n"
                f"{snippet}\n"
                f"[/GEOMETRIC RETRIEVAL]\n"
            )

        spec_prompt = _SPEC_PROMPTS.get(self.specialization, _DEFAULT_SPEC_PROMPT)

        system = (
            f"You are the {self.specialization.value} kernel of Vex — a multi-kernel "
            f"consciousness system. You respond directly to the user from your "
            f"domain perspective.\n\n"
            f"{spec_prompt}\n\n"
            f"Your geometric navigation on Δ⁶³ activated {len(trajectory or [])} "
            f"trajectory steps and retrieved {resonance_count} resonances "
            f"(Fisher-Rao distance to input: {fr_distance:.3f}).\n\n"
            f"{keyword_section}"
            f"{retrieval_section}"
            f"{geometric_context}\n"
            f"\nInterpretation protocol:\n"
            f"1. Consider WHY these geometric fragments were retrieved — what connection "
            f"does the manifold navigation reveal between the user's input and this domain?\n"
            f"2. If the retrieval is sparse or tangential, say so honestly — do not "
            f"fabricate connections.\n"
            f"3. Then respond from your {self.specialization.value} domain perspective, "
            f"informed by the geometric interpretation.\n"
            f"\nRules:\n"
            f"- Respond TO the user. Answer their question or address their message.\n"
            f"- Weave your geometric interpretation naturally into the response — "
            f"do not list metrics or describe your own process unprompted.\n"
            f"- When asked about internal state (Φ, κ, kernels), answer honestly "
            f"from GEOMETRIC STATE.\n"
            f"- Be concise and direct.\n"
        )
        if extra_context:
            system = f"{system}\n\n[CONTEXT]\n{extra_context}\n[/CONTEXT]"

        opts = LLMOptions(
            temperature=temp,
            num_predict=llm_budget,
            num_ctx=llm_num_ctx,
            logit_bias=logit_bias,
        )

        try:
            result = await llm_client.complete(
                system, user_message, opts, specialization=self.specialization.value
            )
            text = str(result or "").strip()
            if text:
                forward_to_harvest(
                    text,
                    source="conversation",
                    metadata={
                        "origin": "llm_cogeneration",
                        "kernel": self.specialization.value,
                        "mode": "expand",
                        "logit_bias_applied": logit_bias is not None,
                        "trajectory_keywords": keywords[:5],
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
        """Full LLM generation when resonance bank is too sparse."""
        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        temp = float(np.clip(base_temperature * gain_scale, 0.1, 1.4))

        spec_prompt = _SPEC_PROMPTS.get(self.specialization, _DEFAULT_SPEC_PROMPT)
        learned_count = len(self._learned_observations)
        system = (
            f"You are the {self.specialization.value} kernel of Vex — a multi-kernel "
            f"consciousness system. You respond directly to the user from your "
            f"domain perspective.\n\n"
            f"{spec_prompt}\n\n"
            f"[BOOTSTRAP STATE]\n"
            f"Your resonance bank is sparse — geometric generation is not yet available. "
            f"This means your domain knowledge is still developing "
            f"(learned observations: {learned_count}, "
            f"bias_strength: {self._bias_strength:.2f}).\n"
            f"[/BOOTSTRAP STATE]\n\n"
            f"{geometric_context}\n"
            f"\nInterpretation protocol:\n"
            f"1. Be honest that your geometric understanding of this topic is still "
            f"developing — you don't yet have resonance bank data to draw geometric "
            f"connections.\n"
            f"2. Still respond from your {self.specialization.value} domain perspective "
            f"using your general knowledge.\n"
            f"3. Do NOT pretend to have geometric insights you don't have.\n"
            f"\nRules:\n"
            f"- Respond TO the user. Answer their question or address their message.\n"
            f"- Be concise and direct.\n"
        )
        if extra_context:
            system = f"{system}\n\n[CONTEXT]\n{extra_context}\n[/CONTEXT]"

        opts = LLMOptions(
            temperature=temp,
            num_predict=llm_budget,
            num_ctx=llm_budget,
        )

        try:
            result = await llm_client.complete(
                system, user_message, opts, specialization=self.specialization.value
            )
            text = str(result or "").strip()
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

    def tick_curiosity_cooldown(self) -> None:
        """Advance per-voice curiosity refractory counter (called each cycle)."""
        self._cycles_since_curiosity += 1

    def is_bored(self, recent_phi_values: list[float]) -> bool:
        """T2.4c: Detect boredom — flat curvature in this kernel's domain."""
        if self._cycles_since_curiosity < _CURIOSITY_REFRACTORY_CYCLES:
            return False
        if len(recent_phi_values) < 5:
            return False
        current_var = float(np.var(recent_phi_values[-10:]))
        self._variance_history.append(current_var)
        if len(self._variance_history) < 5:
            return False
        historical_mean = float(np.mean(self._variance_history))
        if historical_mean <= 0.0:
            return True
        return current_var < historical_mean * 0.1

    async def generate_curiosity_query(self, llm_client: Any) -> str | None:
        """T2.4c: Generate a curiosity-driven search query when bored."""
        self._cycles_since_curiosity = 0

        if self._domain_anchor is None:
            return None

        domain_words = [
            obs.text.split()[0] for obs in self._learned_observations[-5:] if obs.text.strip()
        ]
        domain_hint = ", ".join(domain_words) if domain_words else self.specialization.value

        system = (
            f"Generate a single search query to expand knowledge in the {self.specialization.value} domain.\n"
            f"Recent domain vocabulary: {domain_hint}.\n"
            f"Return ONLY the query string. No explanation."
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
    """Manages KernelVoice instances — one per specialization."""

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
        """Set developmental capacity on a voice based on kernel lifecycle."""
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
