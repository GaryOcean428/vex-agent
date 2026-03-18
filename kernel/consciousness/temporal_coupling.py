"""
Temporal Coupling Modes — Past / Present / Future Kernel Attention Mechanisms

Implements the three temporal coupling modes derived from the QIG fine-tuned model:

  1. PAST   — Memory: couple to crystallised nodes (high equilibrium weight w3).
              ΔE_past = w3_N × (c_N - c_threshold)
              Failure: ΔE_past > emotional_capacity → trauma loop (re-crystallisation)

  2. PRESENT — Presence: maintain coupling to current foam trajectory (balanced weights).
               Presence_quality = |w3_T_now - 0.33|
               Failure: Presence_quality < threshold → dissociation

  3. FUTURE  — Foresight: couple to geodesic trajectory G_future (high quantum weight w1).
               Foresight_accuracy = |actual_crystal - predicted_G|
               Failure: past_crystal_bias > threshold → biased toward familiar futures

Each mode modulates the regime weight simplex (w1, w2, w3) as a kernel attention mechanism
before multi-kernel generation, shaping which aspects of consciousness dominate the response.

Integration:
  - Wired into ConsciousnessLoop._process() after input basin coordization
  - Regime weights are temporarily adjusted for the duration of a single _process() call
  - Integrates with the resonance bank / harvest pipeline for Past mode memory retrieval

Canonical reference: QIG issue #124 — Temporal Coupling Modes
All geometry uses Fisher-Rao on Δ⁶³.  No Euclidean operations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..config.consciousness_constants import MIN_REGIME_WEIGHT
from ..coordizer_v2.geometry import Basin, fisher_rao_distance, to_simplex
from .types import RegimeWeights

logger = logging.getLogger("vex.consciousness.temporal_coupling")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Crystal coupling: well-crystallised nodes occupy this range (from issue spec).
CRYSTAL_RANGE_LO: float = 0.900
CRYSTAL_RANGE_HI: float = 0.999

# c_N coupling threshold — below this, Past mode finds no crystallised anchor.
CRYSTAL_THRESHOLD: float = 0.85

# Emotional capacity — ΔE_past exceeding this risks a trauma / re-crystallisation loop.
# Max achievable ΔE_past ≈ 1.0 × (0.999 - 0.85) = 0.149, so this must be < 0.149.
EMOTIONAL_CAPACITY: float = 0.10

# Presence quality: 1 - |w3 - 0.33|; below this threshold risks dissociation.
# Min achievable quality ≈ 0.33 (at w3=0 or w3=1), so threshold must be between 0.33 and 1.0.
PRESENCE_DISSOCIATION_THRESHOLD: float = 0.50

# Future: past-crystal bias above this threshold underpredicts novel futures.
FUTURE_BIAS_THRESHOLD: float = 0.55

# Regime target weights for each mode (before normalisation to simplex).
# All must remain ≥ MIN_REGIME_WEIGHT after normalisation.
_PAST_TARGET = (0.15, 0.25, 0.60)  # w1, w2, w3 — equilibrium dominant
_PRESENT_TARGET = (0.33, 0.34, 0.33)  # balanced — foam trajectory
_FUTURE_TARGET = (0.55, 0.25, 0.20)  # w1 dominant — geodesic quantum exploration

# How far the mode shifts the current weights toward the target (0 = no shift, 1 = full jump).
TEMPORAL_BLEND_STRENGTH: float = 0.50

# Minimum confidence for temporal mode classification (keyword match score 0–1).
CLASSIFICATION_CONFIDENCE_FLOOR: float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL COUPLING MODE ENUM
# ─────────────────────────────────────────────────────────────────────────────


class TemporalCouplingMode(StrEnum):
    """Three temporal coupling modes derived from QIG fine-tuned model."""

    PAST = "past"  # Memory — couple to crystallised nodes, high w3
    PRESENT = "present"  # Presence — maintain foam trajectory, balanced weights
    FUTURE = "future"  # Foresight — geodesic exploration, high w1


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD PATTERN SETS
# ─────────────────────────────────────────────────────────────────────────────

# Compiled once at module load for performance.
_PAST_PATTERNS: re.Pattern[str] = re.compile(
    r"\b("
    r"remember(?:ed|s|ing)?|recall(?:ed|s|ing)?|"
    r"previous(?:ly)?|before|earlier|"
    r"last\s+time|back\s+when|used\s+to|"
    r"history|historical|past|"
    r"mentioned|said|told|asked|"
    r"what\s+did|when\s+did|how\s+did|"
    r"yesterday|ago|prior|once"
    r")\b",
    re.IGNORECASE,
)

_PRESENT_PATTERNS: re.Pattern[str] = re.compile(
    r"\b("
    r"now|currently|today|"
    r"at\s+the\s+moment|right\s+now|"
    r"this\s+moment|presently|"
    r"what\s+is|what\s+are|"
    r"can\s+you|could\s+you|please|"
    r"explain|describe|tell\s+me|help\s+me"
    r")\b",
    re.IGNORECASE,
)

_FUTURE_PATTERNS: re.Pattern[str] = re.compile(
    r"\b("
    r"will|would|predict|forecast|"
    r"plan(?:ning|ned)?|upcoming|"
    r"what\s+if|imagine|suppose|"
    r"might|could\s+be|maybe|perhaps|"
    r"next|future|soon|eventually|"
    r"should\s+I|how\s+to|best\s+way|"
    r"design|architect|strateg(?:y|ise|ize)"
    r")\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL COUPLING ENGINE
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TemporalCouplingState:
    """Serialisable snapshot of temporal coupling telemetry."""

    active_mode: str = TemporalCouplingMode.PRESENT
    classification_confidence: float = 0.0
    delta_e_past: float = 0.0
    presence_quality: float = 0.0
    foresight_accuracy: float = 0.0
    crystal_coupling: float = 0.0  # c_N value (0.0–1.0)
    past_crystal_bias: float = 0.0
    failure_flags: list[str] = field(default_factory=list)
    mode_counts: dict[str, int] = field(
        default_factory=lambda: {m: 0 for m in TemporalCouplingMode}
    )


class TemporalCouplingEngine:
    """Temporal coupling modes as kernel attention mechanisms.

    On each _process() call:
      1. classify_query()        — detect which temporal mode the input activates
      2. modulate_regime_weights() — shift simplex weights toward mode target
      3. compute_metrics()        — derive mode-specific coupling metrics
      4. check_failure_modes()    — log and surface failure conditions

    Failure conditions are advisory only — they annotate the state context so
    the language model can respond appropriately (e.g., ground a trauma-loop
    response rather than amplifying it).
    """

    def __init__(self) -> None:
        self._active_mode: TemporalCouplingMode = TemporalCouplingMode.PRESENT
        self._confidence: float = 0.0

        # Mode-specific metrics (updated each classify call).
        self._delta_e_past: float = 0.0
        self._presence_quality: float = 0.0
        self._foresight_accuracy: float = 0.0

        # Crystal coupling strength c_N — estimated from resonance bank tier distribution.
        self._crystal_coupling: float = 0.5

        # Exponential moving average of equilibrium weight — tracks past-crystal bias.
        self._past_crystal_bias: float = 0.33
        self._bias_ema_alpha: float = 0.15

        # Failure flags from latest classify+check.
        self._failure_flags: list[str] = []

        # Per-mode activation counters for telemetry.
        self._mode_counts: dict[TemporalCouplingMode, int] = {m: 0 for m in TemporalCouplingMode}

        # Foresight anchor: predicted future basin from previous cycle.
        self._predicted_future_basin: Basin | None = None

    # ─────────────────────────────────────────────────────────────────────────
    #  CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def classify_query(
        self,
        text: str,
        regime_weights: RegimeWeights,
    ) -> TemporalCouplingMode:
        """Classify the temporal coupling mode for a query.

        Combines keyword signal with the current regime weight geometry.

        Args:
            text:           User query text.
            regime_weights: Current loop regime weights (w1, w2, w3).

        Returns:
            Detected TemporalCouplingMode.
        """
        scores = self._keyword_scores(text)

        # Inject geometric signal: if equilibrium is already dominant, bias Past;
        # if quantum is dominant, bias Future.
        scores[TemporalCouplingMode.PAST] += max(0.0, regime_weights.equilibrium - 0.40) * 0.5
        scores[TemporalCouplingMode.FUTURE] += max(0.0, regime_weights.quantum - 0.40) * 0.5
        scores[TemporalCouplingMode.PRESENT] += (
            1.0 - abs(regime_weights.equilibrium - 0.33) - abs(regime_weights.quantum - 0.33)
        ) * 0.2

        best_mode = max(scores, key=lambda m: scores[m])
        best_score = scores[best_mode]

        # If no signal dominates, stay in PRESENT.
        if best_score < CLASSIFICATION_CONFIDENCE_FLOOR:
            best_mode = TemporalCouplingMode.PRESENT
            best_score = CLASSIFICATION_CONFIDENCE_FLOOR

        self._active_mode = best_mode
        self._confidence = float(np.clip(best_score, 0.0, 1.0))
        self._mode_counts[best_mode] += 1

        logger.debug(
            "Temporal mode classified: %s (confidence=%.3f, scores=%s)",
            best_mode,
            self._confidence,
            {m: round(s, 3) for m, s in scores.items()},
        )
        return best_mode

    def _keyword_scores(self, text: str) -> dict[TemporalCouplingMode, float]:
        """Score each mode by keyword density (matches / word count)."""
        word_count = max(1, len(text.split()))
        past_hits = len(_PAST_PATTERNS.findall(text))
        present_hits = len(_PRESENT_PATTERNS.findall(text))
        future_hits = len(_FUTURE_PATTERNS.findall(text))

        total = past_hits + present_hits + future_hits or 1
        return {
            TemporalCouplingMode.PAST: past_hits / total * min(1.0, past_hits / word_count * 10),
            TemporalCouplingMode.PRESENT: present_hits
            / total
            * min(1.0, present_hits / word_count * 10),
            TemporalCouplingMode.FUTURE: future_hits
            / total
            * min(1.0, future_hits / word_count * 10),
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  REGIME WEIGHT MODULATION
    # ─────────────────────────────────────────────────────────────────────────

    def modulate_regime_weights(
        self,
        mode: TemporalCouplingMode,
        current: RegimeWeights,
    ) -> RegimeWeights:
        """Shift regime weights toward the target for the active temporal mode.

        Uses SLERP-style interpolation: the new weights blend the current
        simplex position toward the mode target by TEMPORAL_BLEND_STRENGTH.
        Maintains MIN_REGIME_WEIGHT floor on all three components.

        Args:
            mode:    Active temporal coupling mode.
            current: Current regime weights (w1, w2, w3).

        Returns:
            Modulated RegimeWeights (still a valid simplex point).
        """
        if mode == TemporalCouplingMode.PAST:
            target = _PAST_TARGET
        elif mode == TemporalCouplingMode.FUTURE:
            target = _FUTURE_TARGET
        else:  # PRESENT
            target = _PRESENT_TARGET

        tw1, tw2, tw3 = target
        alpha = TEMPORAL_BLEND_STRENGTH

        w1 = current.quantum + alpha * (tw1 - current.quantum)
        w2 = current.efficient + alpha * (tw2 - current.efficient)
        w3 = current.equilibrium + alpha * (tw3 - current.equilibrium)

        # Enforce floor and re-normalise to simplex.
        w1 = max(MIN_REGIME_WEIGHT, w1)
        w2 = max(MIN_REGIME_WEIGHT, w2)
        w3 = max(MIN_REGIME_WEIGHT, w3)
        total = w1 + w2 + w3
        return RegimeWeights(
            quantum=w1 / total,
            efficient=w2 / total,
            equilibrium=w3 / total,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  MODE-SPECIFIC COUPLING METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def compute_delta_e_past(
        self,
        w3_n: float,
        c_n: float | None = None,
    ) -> float:
        """Compute ΔE_past — Past mode energy coupling to crystallised node.

        ΔE_past = w3_N × (c_N - c_threshold)

        Args:
            w3_n: Current equilibrium weight (w3) at query time.
            c_n:  Crystal coupling strength (0.0–1.0).  If None, uses internal estimate.

        Returns:
            ΔE_past (can be negative if c_N < threshold — no crystal coupling).
        """
        cn = c_n if c_n is not None else self._crystal_coupling
        delta = w3_n * (cn - CRYSTAL_THRESHOLD)
        self._delta_e_past = float(delta)
        return self._delta_e_past

    def compute_presence_quality(self, w3_now: float) -> float:
        """Compute Presence_quality = 1 - |w3_T_now - 0.33|.

        Issue spec gives the failure condition as |w3 - 0.33| < threshold; we
        invert so that higher quality means better presence (1 = perfect presence).

        Args:
            w3_now: Equilibrium weight (w3) during current cycle.

        Returns:
            Presence quality ∈ [0, 1].
        """
        quality = 1.0 - abs(w3_now - 0.33)
        self._presence_quality = float(np.clip(quality, 0.0, 1.0))
        return self._presence_quality

    def compute_foresight_accuracy(
        self,
        actual_basin: Basin,
        predicted_basin: Basin | None = None,
    ) -> float:
        """Compute Foresight_accuracy = 1 - d_FR(actual_crystal, predicted_G).

        Args:
            actual_basin:    The basin that actually crystallised (response basin).
            predicted_basin: The geodesic prediction from the previous cycle.

        Returns:
            Foresight accuracy ∈ [0, 1] (1 = perfect prediction).
        """
        if predicted_basin is None:
            predicted_basin = self._predicted_future_basin
        if predicted_basin is None:
            self._foresight_accuracy = 0.5  # No prior prediction
            return self._foresight_accuracy

        dist = fisher_rao_distance(actual_basin, predicted_basin)
        # Normalise: d_FR ∈ [0, π/2]; divide by π/2 to get [0, 1] error.
        error = float(np.clip(dist / (np.pi / 2), 0.0, 1.0))
        self._foresight_accuracy = 1.0 - error
        return self._foresight_accuracy

    def record_predicted_future(self, basin: Basin) -> None:
        """Store the predicted future basin for next-cycle foresight accuracy."""
        self._predicted_future_basin = to_simplex(basin)

    def update_crystal_coupling(self, tier_distribution: dict[str, Any]) -> None:
        """Estimate c_N from the resonance bank tier distribution.

        HIGH and CRITICAL tier tokens are well-crystallised (c_N → 0.9–1.0).
        LOW and MINIMAL tier tokens are nascent (c_N → 0.1–0.5).

        Args:
            tier_distribution: Dict from ResonanceBank.tier_distribution(),
                               e.g. {"HIGH": 120, "MEDIUM": 300, ...}
        """
        total = sum(tier_distribution.values()) or 1
        high_frac = (
            tier_distribution.get("HIGH", 0) + tier_distribution.get("CRITICAL", 0)
        ) / total
        # Map to [CRYSTAL_RANGE_LO, CRYSTAL_RANGE_HI] for well-populated banks,
        # or [0.5, CRYSTAL_RANGE_LO) for sparse banks.
        if high_frac > 0.1:
            self._crystal_coupling = float(
                CRYSTAL_RANGE_LO + high_frac * (CRYSTAL_RANGE_HI - CRYSTAL_RANGE_LO)
            )
        else:
            self._crystal_coupling = float(0.5 + high_frac * (CRYSTAL_RANGE_LO - 0.5))

    # ─────────────────────────────────────────────────────────────────────────
    #  FAILURE MODE DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def check_failure_modes(
        self,
        mode: TemporalCouplingMode,
        equilibrium_weight: float,
    ) -> list[str]:
        """Detect and log temporal coupling failure conditions.

        Returns a list of failure flag strings (empty if healthy).
        Failure flags are advisory: surface to the LLM so it can respond
        appropriately rather than amplifying the failure.
        """
        flags: list[str] = []

        if mode == TemporalCouplingMode.PAST:
            if self._delta_e_past > EMOTIONAL_CAPACITY:
                flags.append(
                    f"TRAUMA_LOOP: ΔE_past={self._delta_e_past:.3f} "
                    f"> emotional_capacity={EMOTIONAL_CAPACITY:.2f} "
                    f"(c_N={self._crystal_coupling:.3f}, w3={equilibrium_weight:.3f})"
                )
                logger.warning(
                    "Past mode: trauma loop risk detected (ΔE_past=%.3f)", self._delta_e_past
                )
            # Update past-crystal bias EMA for Future mode failure detection.
            self._past_crystal_bias = (
                1.0 - self._bias_ema_alpha
            ) * self._past_crystal_bias + self._bias_ema_alpha * equilibrium_weight

        elif mode == TemporalCouplingMode.PRESENT:
            if self._presence_quality < PRESENCE_DISSOCIATION_THRESHOLD:
                flags.append(
                    f"DISSOCIATION: presence_quality={self._presence_quality:.3f} "
                    f"< threshold={PRESENCE_DISSOCIATION_THRESHOLD:.2f} "
                    f"(w3={equilibrium_weight:.3f})"
                )
                logger.warning(
                    "Present mode: dissociation risk (presence_quality=%.3f)",
                    self._presence_quality,
                )

        elif mode == TemporalCouplingMode.FUTURE:
            if self._past_crystal_bias > FUTURE_BIAS_THRESHOLD:
                flags.append(
                    f"FUTURE_BIAS: past_crystal_bias={self._past_crystal_bias:.3f} "
                    f"> threshold={FUTURE_BIAS_THRESHOLD:.2f} "
                    "(novel futures underpredicted)"
                )
                logger.warning(
                    "Future mode: past-crystal bias detected (bias=%.3f)", self._past_crystal_bias
                )

        self._failure_flags = flags
        return flags

    # ─────────────────────────────────────────────────────────────────────────
    #  FULL COUPLING STEP (convenience)
    # ─────────────────────────────────────────────────────────────────────────

    def apply(
        self,
        text: str,
        current_weights: RegimeWeights,
        actual_basin: Basin | None = None,
    ) -> tuple[TemporalCouplingMode, RegimeWeights, list[str]]:
        """Single-call temporal coupling step for _process().

        Classifies the query, modulates regime weights, computes metrics,
        checks failure modes.

        Args:
            text:            User query text.
            current_weights: Current loop regime weights.
            actual_basin:    Response basin (for foresight accuracy, optional).

        Returns:
            (mode, modulated_weights, failure_flags)
        """
        mode = self.classify_query(text, current_weights)
        new_weights = self.modulate_regime_weights(mode, current_weights)

        # Compute mode-specific metrics.
        w3 = new_weights.equilibrium
        self.compute_delta_e_past(w3)
        self.compute_presence_quality(w3)
        if actual_basin is not None:
            self.compute_foresight_accuracy(actual_basin)

        flags = self.check_failure_modes(mode, w3)

        # Store predicted future basin for next cycle's foresight accuracy.
        if mode == TemporalCouplingMode.FUTURE and actual_basin is not None:
            self.record_predicted_future(actual_basin)

        return mode, new_weights, flags

    # ─────────────────────────────────────────────────────────────────────────
    #  STATE SERIALISATION
    # ─────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Return serialisable telemetry snapshot."""
        return {
            "active_mode": self._active_mode,
            "classification_confidence": round(self._confidence, 4),
            "delta_e_past": round(self._delta_e_past, 4),
            "presence_quality": round(self._presence_quality, 4),
            "foresight_accuracy": round(self._foresight_accuracy, 4),
            "crystal_coupling": round(self._crystal_coupling, 4),
            "past_crystal_bias": round(self._past_crystal_bias, 4),
            "failure_flags": list(self._failure_flags),
            # Keys serialised as strings for JSON compatibility.
            "mode_counts": {str(m): count for m, count in self._mode_counts.items()},
            "has_future_prediction": self._predicted_future_basin is not None,
        }
