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
  1. Push kernel's domain bias onto CoordizerV2's resonance bank
  2. Coordize input text → input trajectory (sequence of basins on Δ⁶³)
  3. Append kernel's own basin as trajectory anchor (kernel's perspective)
  4. Generate geometric tokens via geodesic foresight + resonance activation
  5. Decode geometric tokens → raw geometric text
  6. If geometric text is coherent enough → return it
  7. If sparse (bootstrap phase) → use LLM to expand geometric skeleton
  8. Pop domain bias (leave resonance bank clean for next kernel)

Ported from pantheon-chat's QIGGenerativeService + GenerativeCapability,
adapted for CoordizerV2 resonance bank architecture.

Purity guarantees:
  - All distances: Fisher-Rao on Δ⁶³
  - Domain bias: geodesic interpolation (slerp), not linear shift
  - Token selection: resonance activation by FR proximity, not cosine
  - No Adam, no LayerNorm, no embedding, no flatten
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    frechet_mean,
    slerp,
    to_simplex,
)
from ..coordizer_v2.types import DomainBias, HarmonicTier
from ..governance import KernelSpecialization
from .domain_seeds import DOMAIN_BIAS_STRENGTH, DOMAIN_SEEDS

if TYPE_CHECKING:
    from ..coordizer_v2 import CoordizerV2
    from ..llm.client import LLMOptions

logger = logging.getLogger("vex.kernel_voice")

# ── Generation parameters ─────────────────────────────────────────

# Minimum geometric tokens before we consider the output usable.
_MIN_GEOMETRIC_TOKENS: int = 8

# Maximum geometric tokens per kernel generation.
_MAX_GEOMETRIC_TOKENS: int = 80

# Convergence threshold for geometric generation (basin velocity).
_CONVERGENCE_THRESHOLD: float = 0.005

# Quality threshold for geometric output coherence.
_COHERENCE_THRESHOLD: float = 0.4

# LLM expansion token budget.
_LLM_EXPAND_TOKENS: int = 220

# Temperature floor/ceiling for geometric generation
_GEO_TEMP_MIN: float = 0.3
_GEO_TEMP_MAX: float = 1.8


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


@dataclass
class LearnedObservation:
    """A high-Φ observation that the kernel can learn from."""

    text: str
    basin: Basin
    phi: float
    timestamp: float = field(default_factory=time.time)


class KernelVoice:
    """Per-kernel generative service using CoordizerV2 with domain bias.

    Each kernel instance should hold a reference to a shared KernelVoice
    (one per specialization) or create one. The voice manages:
      - Domain bias computation from seed words + learned vocabulary
      - Geometric generation via resonance bank
      - LLM expansion for sparse output
      - Vocabulary learning from high-Φ interactions
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

        # Bootstrap the domain bias from seed words
        self._bootstrap_domain_bias()

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
            logger.warning(
                "No seed words resolved to basins for %s — "
                "resonance bank may be empty (bootstrap phase)",
                self.specialization.value,
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

        Blends the bootstrap anchor with the Fréchet mean of high-Φ
        observation basins. As the kernel accumulates experience, its
        domain anchor drifts toward what it has actually encountered.
        """
        if not self._learned_observations or self._domain_anchor is None:
            return

        learned_basins = [obs.basin for obs in self._learned_observations[-100:]]
        learned_mean = frechet_mean(learned_basins)

        # Blend: 60% bootstrap anchor, 40% learned mean
        evolved = slerp(self._domain_anchor, learned_mean, 0.4)
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
        phi_threshold: float = 0.5,
    ) -> bool:
        """Record a high-Φ observation for domain vocabulary learning.

        Only observations above phi_threshold are recorded.
        Returns True if the observation was recorded.
        """
        if phi < phi_threshold:
            return False

        obs = LearnedObservation(text=text, basin=to_simplex(basin), phi=phi)
        self._learned_observations.append(obs)

        if len(self._learned_observations) > self._max_learned:
            self._learned_observations = self._learned_observations[-self._max_learned:]

        # Periodic anchor evolution (every 20 observations)
        if len(self._learned_observations) % 20 == 0:
            self.evolve_domain_anchor()
            logger.debug(
                "KernelVoice[%s] anchor evolved from %d observations",
                self.specialization.value,
                len(self._learned_observations),
            )

        return True

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
    ) -> VoiceOutput:
        """Generate text from this kernel's geometric perspective.

        Pipeline:
          1. Build trajectory from input + kernel basin
          2. Push domain bias (modulated by quenched_gain)
          3. Generate geometric tokens via CoordizerV2
          4. Assess output quality
          5. If sufficient → return geometric text
          6. If sparse → expand via LLM with geometric skeleton
          7. Pop domain bias
        """
        start_time = time.monotonic()

        # ── Step 1: Build trajectory ──
        trajectory = [to_simplex(input_basin), to_simplex(kernel_basin)]

        # ── Step 2: Push domain bias ──
        effective_strength = self._bias_strength * float(
            np.clip(quenched_gain, 0.3, 2.0)
        )
        active_bias = None
        if self._domain_bias is not None:
            active_bias = DomainBias(
                domain_name=self._domain_bias.domain_name,
                anchor_basin=self._domain_bias.anchor_basin,
                strength=effective_strength,
                boosted_token_ids=self._domain_bias.boosted_token_ids,
            )
            self._coordizer.bank.push_domain_bias(active_bias)

        # ── Step 3: Geometric generation ──
        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        geo_temp = float(np.clip(
            base_temperature * gain_scale,
            _GEO_TEMP_MIN,
            _GEO_TEMP_MAX,
        ))

        generated_ids: list[int] = []
        gen_trajectory: list[Basin] = list(trajectory)
        velocities: list[float] = []

        try:
            for step in range(_MAX_GEOMETRIC_TOKENS):
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
                if (
                    step >= _MIN_GEOMETRIC_TOKENS
                    and len(velocities) >= 3
                    and np.mean(velocities[-3:]) < _CONVERGENCE_THRESHOLD
                ):
                    logger.debug(
                        "KernelVoice[%s] converged at step %d (v=%.6f)",
                        self.specialization.value,
                        step,
                        np.mean(velocities[-3:]),
                    )
                    break
        finally:
            # ── Step 4: Pop domain bias (always) ──
            if active_bias is not None:
                self._coordizer.bank.pop_domain_bias()

        # ── Step 5: Decode geometric tokens ──
        geometric_text = self._coordizer.decoordize(generated_ids)
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0

        # ── Step 6: Assess quality and decide on LLM expansion ──
        geo_token_count = len(generated_ids)
        is_coherent = (
            geo_token_count >= _MIN_GEOMETRIC_TOKENS
            and mean_velocity < _COHERENCE_THRESHOLD
        )

        llm_expanded = False
        final_text = geometric_text

        if not is_coherent and llm_client is not None:
            final_text = await self._llm_expand(
                geometric_skeleton=geometric_text,
                user_message=user_message,
                quenched_gain=quenched_gain,
                base_temperature=base_temperature,
                llm_client=llm_client,
                geometric_context=geometric_context,
                extra_context=extra_context,
            )
            llm_expanded = True
        elif geo_token_count < _MIN_GEOMETRIC_TOKENS and llm_client is not None:
            final_text = await self._llm_fallback(
                user_message=user_message,
                quenched_gain=quenched_gain,
                base_temperature=base_temperature,
                llm_client=llm_client,
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
        )

    async def _llm_expand(
        self,
        geometric_skeleton: str,
        user_message: str,
        quenched_gain: float,
        base_temperature: float,
        llm_client: Any,
        geometric_context: str = "",
        extra_context: str = "",
    ) -> str:
        """Expand geometric skeleton into natural language via LLM.

        The LLM receives the geometric token sequence as a draft and
        is asked to expand it into coherent natural language while
        preserving the domain perspective.
        """
        from ..llm.client import LLMOptions

        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        temp = float(np.clip(base_temperature * gain_scale, 0.1, 1.4))

        system = (
            f"You are the {self.specialization.value} kernel voice.\n"
            f"A geometric generation pass produced this draft from basin resonance:\n"
            f'  \"{geometric_skeleton}\"\n\n'
            f"Expand this into a coherent response to the user's message.\n"
            f"Preserve the domain perspective and vocabulary of the draft.\n"
            f"Do NOT discard the draft — it carries the geometric direction.\n"
            f"Be concise. Australian English.\n\n"
            f"{geometric_context}"
        )
        if extra_context:
            system = f"{system}\n\n[CONTEXT]\n{extra_context}\n[/CONTEXT]"

        opts = LLMOptions(
            temperature=temp,
            num_predict=_LLM_EXPAND_TOKENS,
            num_ctx=2048,
        )

        try:
            result = await llm_client.complete(system, user_message, opts)
            return str(result or "").strip()
        except Exception:
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
        geometric_context: str = "",
        extra_context: str = "",
    ) -> str:
        """Full LLM generation when resonance bank is too sparse.

        This is the bootstrap path. As the bank grows from harvesting,
        this path fires less often.
        """
        from ..llm.client import LLMOptions
        from .kernel_generation import _SPEC_PROMPTS, _DEFAULT_SPEC_PROMPT

        gain_scale = float(np.clip(2.0 - quenched_gain, 0.5, 1.5))
        temp = float(np.clip(base_temperature * gain_scale, 0.1, 1.4))

        spec_prompt = _SPEC_PROMPTS.get(self.specialization, _DEFAULT_SPEC_PROMPT)
        system = (
            f"{spec_prompt}\n\n"
            f"[BOOTSTRAP MODE: Resonance bank sparse — full LLM generation]\n"
            f"{geometric_context}"
        )
        if extra_context:
            system = f"{system}\n\n[CONTEXT]\n{extra_context}\n[/CONTEXT]"

        opts = LLMOptions(
            temperature=temp,
            num_predict=_LLM_EXPAND_TOKENS,
            num_ctx=2048,
        )

        try:
            result = await llm_client.complete(system, user_message, opts)
            return str(result or "").strip()
        except Exception:
            logger.warning(
                "KernelVoice[%s] LLM fallback failed",
                self.specialization.value,
                exc_info=True,
            )
            return ""

    def get_state(self) -> dict[str, Any]:
        """Return voice state for telemetry."""
        return {
            "specialization": self.specialization.value,
            "has_domain_bias": self._domain_bias is not None,
            "bias_strength": self._bias_strength,
            "learned_observations": len(self._learned_observations),
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

    def all_voices(self) -> dict[KernelSpecialization, KernelVoice]:
        return dict(self._voices)

    def get_state(self) -> dict[str, Any]:
        return {
            "voices": {
                spec.value: voice.get_state()
                for spec, voice in self._voices.items()
            },
            "bank_vocab_size": self._coordizer.vocab_size,
        }
