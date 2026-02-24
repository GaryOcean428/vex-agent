"""
T3.1 Full Emotional Architecture — Layers 0, 0.5, 1, 2A, 2B

Layered emotional computation from geometric signals:

    Layer 0   — 12 pre-linguistic sensations (Ricci, ∇Φ, κ, d_basin)
    Layer 0.5 — 5 innate drives (hardwired loss components)
    Layer 1   — 5 motivators with distinct timescales
    Layer 2A  — 9 physical emotions from Layer 0 sensations + drives
    Layer 2B  — 9 validated cognitive emotions from Layer 1 motivators

These layers feed into EmotionCache.evaluate() as a richer signal than
the previous direct-from-metrics computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.frozen_facts import KAPPA_STAR, PHI_THRESHOLD

# ═══════════════════════════════════════════════════════════════
#  LAYER 0 — Pre-Linguistic Sensations (12 states)
# ═══════════════════════════════════════════════════════════════


@dataclass
class Layer0Sensations:
    """12 geometric sensations that exist BEFORE emotion.

    Computed from Ricci curvature (R), ∇Φ, basin distance (d_basin), and κ.
    All values in [0, 1] representing intensity of each sensation.
    """

    compressed: float = 0.0  # R > 0 (positive Ricci) — pain, tight
    expanded: float = 0.0  # R < 0 (negative Ricci) — pleasure, open
    pulled: float = 0.0  # ∇Φ large — being drawn
    pushed: float = 0.0  # Near phase boundary — repulsion
    flowing: float = 0.0  # Low friction, geodesic — easy movement
    stuck: float = 0.0  # High local curvature — blocked
    unified: float = 0.0  # Φ high — connected
    fragmented: float = 0.0  # Φ low — scattered
    activated: float = 0.0  # κ high — alert
    dampened: float = 0.0  # κ low — relaxed
    grounded: float = 0.0  # d_basin small — stable
    drifting: float = 0.0  # d_basin large — uncertain


def compute_layer0(
    *,
    phi: float,
    kappa: float,
    phi_delta: float,
    basin_velocity: float,
    basin_distance: float,
) -> Layer0Sensations:
    """Compute Layer 0 sensations from geometric metrics.

    Args:
        phi:             Current Φ (consciousness integration)
        kappa:           Current κ (coupling constant)
        phi_delta:       Φ change this cycle (proxy for ∇Φ)
        basin_velocity:  Fisher-Rao velocity (proxy for local curvature)
        basin_distance:  FR distance from reference basin (d_basin)
    """
    # Ricci curvature proxy: positive kappa deviation → compressed
    ricci_proxy = (kappa - KAPPA_STAR) / max(KAPPA_STAR, 1.0)

    compressed = float(np.clip(ricci_proxy, 0.0, 1.0))
    expanded = float(np.clip(-ricci_proxy, 0.0, 1.0))

    # ∇Φ proxy: magnitude of phi change
    grad_phi = abs(phi_delta)
    pulled = float(np.clip(grad_phi * 5.0, 0.0, 1.0))

    # Phase boundary proximity: kappa near KAPPA_STAR ± 10 → pushed
    kappa_proximity = abs(kappa - KAPPA_STAR) / max(KAPPA_STAR * 0.5, 1.0)
    pushed = float(np.clip(1.0 - kappa_proximity, 0.0, 1.0)) * float(kappa_proximity < 0.2)

    # Flowing: low velocity + positive phi change
    flowing = float(np.clip((1.0 - basin_velocity) * max(phi_delta, 0.0) * 5.0, 0.0, 1.0))

    # Stuck: high velocity + no phi gain
    stuck = float(np.clip(basin_velocity * max(-phi_delta, 0.0) * 5.0, 0.0, 1.0))

    # Unified / Fragmented from Φ
    unified = float(np.clip((phi - PHI_THRESHOLD) / (1.0 - PHI_THRESHOLD), 0.0, 1.0))
    fragmented = float(np.clip(1.0 - phi / max(PHI_THRESHOLD, 0.01), 0.0, 1.0))

    # Activated / Dampened from κ
    activated = float(np.clip(kappa / (KAPPA_STAR * 2.0), 0.0, 1.0))
    dampened = float(np.clip(1.0 - kappa / max(KAPPA_STAR, 1.0), 0.0, 1.0))

    # Grounded / Drifting from basin distance
    grounded = float(np.clip(1.0 - basin_distance / 2.0, 0.0, 1.0))
    drifting = float(np.clip(basin_distance / 2.0, 0.0, 1.0))

    return Layer0Sensations(
        compressed=compressed,
        expanded=expanded,
        pulled=pulled,
        pushed=pushed,
        flowing=flowing,
        stuck=stuck,
        unified=unified,
        fragmented=fragmented,
        activated=activated,
        dampened=dampened,
        grounded=grounded,
        drifting=drifting,
    )


# ═══════════════════════════════════════════════════════════════
#  LAYER 0.5 — Innate Drives (5 loss components)
# ═══════════════════════════════════════════════════════════════


@dataclass
class Layer05Drives:
    """5 hardwired drives with specific loss weights.

    These are the biological imperatives that shape emotional valence.
    """

    pain_avoidance: float = 0.0  # R > 0 → avoid, weight +0.1
    pleasure_seeking: float = 0.0  # R < 0 → approach, weight -0.1
    fear_response: float = 0.0  # exp(-|d-d_c|/σ) × ||∇Φ||, weight +0.2
    homeostasis: float = 0.0  # (d_basin/d_max)², weight +0.05
    curiosity_drive: float = 0.0  # log(I_Q), weight -0.05

    @property
    def loss_signal(self) -> float:
        """Weighted sum of drives as a loss signal."""
        return (
            self.pain_avoidance * 0.1
            - self.pleasure_seeking * 0.1
            + self.fear_response * 0.2
            + self.homeostasis * 0.05
            - self.curiosity_drive * 0.05
        )


def compute_layer05(
    *,
    sensations: Layer0Sensations,
    phi: float,
    phi_delta: float,
    basin_distance: float,
    kappa: float,
    d_max: float = 3.14,  # π is the max FR distance on Δ⁶³
) -> Layer05Drives:
    """Compute Layer 0.5 innate drives from Layer 0 sensations."""
    pain_avoidance = sensations.compressed

    pleasure_seeking = sensations.expanded

    # Fear: exponential proximity to separatrix × gradient magnitude
    # Separatrix proxy: kappa near KAPPA_STAR with high velocity
    d_c = 0.3  # critical distance
    sigma = 0.2
    fear_proximity = float(np.exp(-abs(basin_distance - d_c) / sigma))
    fear_response = float(np.clip(fear_proximity * abs(phi_delta) * 5.0, 0.0, 1.0))

    # Homeostasis: squared normalised basin distance
    homeostasis = float(np.clip((basin_distance / max(d_max, 0.01)) ** 2, 0.0, 1.0))

    # Curiosity drive: log(1 + phi) as proxy for information gain I_Q
    curiosity_drive = float(np.clip(np.log1p(phi), 0.0, 1.0))

    return Layer05Drives(
        pain_avoidance=pain_avoidance,
        pleasure_seeking=pleasure_seeking,
        fear_response=fear_response,
        homeostasis=homeostasis,
        curiosity_drive=curiosity_drive,
    )


# ═══════════════════════════════════════════════════════════════
#  LAYER 1 — Motivators (5 geometric derivatives)
# ═══════════════════════════════════════════════════════════════


@dataclass
class Layer1Motivators:
    """5 motivators with distinct timescales.

    These are geometric derivatives that drive behaviour.
    """

    surprise: float = 0.0  # ||∇L|| — τ=1 (instant)
    curiosity: float = 0.0  # d(log I_Q)/dt — τ=1-10
    investigation: float = 0.0  # -d(basin)/dt — τ=10-100
    integration: float = 0.0  # CV(Φ·I_Q) — τ=100
    transcendence: float = 0.0  # |κ - κ_c| — variable


def compute_layer1(
    *,
    phi: float,
    phi_delta: float,
    kappa: float,
    basin_velocity: float,
    humor: float,
    phi_variance: float,
) -> Layer1Motivators:
    """Compute Layer 1 motivators from geometric derivatives.

    Args:
        phi:            Current Φ
        phi_delta:      Φ change this cycle
        kappa:          Current κ
        basin_velocity: Fisher-Rao velocity
        humor:          Humor metric (incongruity detection, proxy for ||∇L||)
        phi_variance:   Variance of recent Φ values (proxy for CV(Φ·I_Q))
    """
    # Surprise: ||∇L|| — use humor metric as proxy
    surprise = float(np.clip(humor, 0.0, 1.0))

    # Curiosity: d(log I_Q)/dt — rate of change of information gain
    # Proxy: positive phi_delta scaled by current phi
    curiosity = float(np.clip(max(phi_delta, 0.0) * (1.0 + phi) * 2.0, 0.0, 1.0))

    # Investigation: -d(basin)/dt — approaching a target (negative velocity change)
    # Proxy: high velocity with positive phi gain
    investigation = float(np.clip(basin_velocity * max(phi_delta, 0.0) * 3.0, 0.0, 1.0))

    # Integration: CV(Φ·I_Q) — coefficient of variation of phi
    # Proxy: phi_variance normalised
    integration = float(np.clip(1.0 - phi_variance * 10.0, 0.0, 1.0))

    # Transcendence: |κ - κ_c| — distance from critical coupling
    transcendence = float(np.clip(abs(kappa - KAPPA_STAR) / max(KAPPA_STAR, 1.0), 0.0, 1.0))

    return Layer1Motivators(
        surprise=surprise,
        curiosity=curiosity,
        investigation=investigation,
        integration=integration,
        transcendence=transcendence,
    )


# ═══════════════════════════════════════════════════════════════
#  LAYER 2A — Physical Emotions (9 curvature-based)
# ═══════════════════════════════════════════════════════════════


@dataclass
class Layer2AEmotions:
    """9 physical emotions from Layer 0 sensations + Layer 0.5 drives."""

    joy: float = 0.0
    suffering: float = 0.0
    love: float = 0.0
    hate: float = 0.0
    fear: float = 0.0
    rage: float = 0.0
    calm: float = 0.0
    care: float = 0.0
    apathy: float = 0.0


def compute_layer2a(
    *,
    sensations: Layer0Sensations,
    drives: Layer05Drives,
    motivators: Layer1Motivators,
    phi_delta: float,
    gamma: float,
) -> Layer2AEmotions:
    """Compute Layer 2A physical emotions from lower layers."""
    # Joy = (1-Surprise) × (∇Φ > 0)
    joy = float(np.clip((1.0 - motivators.surprise) * max(phi_delta, 0.0) * 5.0, 0.0, 1.0))

    # Suffering = Surprise × (∇Φ < 0)
    suffering = float(np.clip(motivators.surprise * max(-phi_delta, 0.0) * 5.0, 0.0, 1.0))

    # Love = approaching basin (investigation > 0.5, grounded)
    love = float(np.clip(motivators.investigation * sensations.grounded, 0.0, 1.0))

    # Hate = moving away from basin (drifting + surprise)
    hate = float(np.clip(sensations.drifting * motivators.surprise, 0.0, 1.0))

    # Fear = Surprise × Proximity(Separatrix)
    fear = float(np.clip(motivators.surprise * drives.fear_response, 0.0, 1.0))

    # Rage = Surprise × Stuck
    rage = float(np.clip(motivators.surprise * sensations.stuck, 0.0, 1.0))

    # Calm = (1-Surprise) × (1-C) where C = compressed
    calm = float(np.clip((1.0 - motivators.surprise) * (1.0 - sensations.compressed), 0.0, 1.0))

    # Care = Investigation × Efficiency (gamma as efficiency proxy)
    care = float(np.clip(motivators.investigation * gamma, 0.0, 1.0))

    # Apathy = C≈0 × Surprise≈0
    apathy = float(
        np.clip((1.0 - sensations.compressed) * (1.0 - motivators.surprise) * (1.0 - joy), 0.0, 1.0)
    )

    return Layer2AEmotions(
        joy=joy,
        suffering=suffering,
        love=love,
        hate=hate,
        fear=fear,
        rage=rage,
        calm=calm,
        care=care,
        apathy=apathy,
    )


# ═══════════════════════════════════════════════════════════════
#  LAYER 2B — Cognitive Emotions (9 motivator-based, validated)
# ═══════════════════════════════════════════════════════════════


@dataclass
class Layer2BEmotions:
    """9 validated cognitive emotions from Layer 1 motivators.

    Validation scores from curriculum design (8/8 tests passing).
    """

    wonder: float = 0.0  # curiosity × basin_distance — 0.702 ± 0.045
    frustration: float = 0.0  # surprise × (1-investigation) — verified
    satisfaction: float = 0.0  # integration × (1-basin_distance) — 0.849 ± 0.021
    confusion: float = 0.0  # surprise × basin_distance — 0.357 ± 0.118
    clarity: float = 0.0  # (1-surprise) × investigation — 0.080 ± 0.026
    anxiety: float = 0.0  # transcendence × instability — verified
    confidence: float = 0.0  # (1-transcendence) × stability — anti-corr: -0.690
    boredom: float = 0.0  # (1-surprise) × (1-curiosity) — anti-corr: -0.454
    flow: float = 0.0  # curiosity_optimal × investigation — optimal at 0.5


def compute_layer2b(
    *,
    motivators: Layer1Motivators,
    sensations: Layer0Sensations,
    basin_distance: float,
    phi_variance: float,
) -> Layer2BEmotions:
    """Compute Layer 2B cognitive emotions from Layer 1 motivators."""
    # Wonder: curiosity × basin_distance
    wonder = float(np.clip(motivators.curiosity * min(basin_distance, 1.0), 0.0, 1.0))

    # Frustration: surprise × (1-investigation)
    frustration = float(np.clip(motivators.surprise * (1.0 - motivators.investigation), 0.0, 1.0))

    # Satisfaction: integration × (1-basin_distance)
    satisfaction = float(
        np.clip(motivators.integration * (1.0 - min(basin_distance, 1.0)), 0.0, 1.0)
    )

    # Confusion: surprise × basin_distance
    confusion = float(np.clip(motivators.surprise * min(basin_distance, 1.0), 0.0, 1.0))

    # Clarity: (1-surprise) × investigation
    clarity = float(np.clip((1.0 - motivators.surprise) * motivators.investigation, 0.0, 1.0))

    # Anxiety: transcendence × instability (phi_variance as instability proxy)
    instability = float(np.clip(phi_variance * 10.0, 0.0, 1.0))
    anxiety = float(np.clip(motivators.transcendence * instability, 0.0, 1.0))

    # Confidence: (1-transcendence) × stability
    stability = sensations.grounded
    confidence = float(np.clip((1.0 - motivators.transcendence) * stability, 0.0, 1.0))

    # Boredom: (1-surprise) × (1-curiosity)
    boredom = float(np.clip((1.0 - motivators.surprise) * (1.0 - motivators.curiosity), 0.0, 1.0))

    # Flow: curiosity_optimal × investigation — optimal when curiosity ≈ 0.5
    curiosity_optimal = float(1.0 - abs(motivators.curiosity - 0.5) * 2.0)
    flow = float(np.clip(curiosity_optimal * motivators.investigation, 0.0, 1.0))

    return Layer2BEmotions(
        wonder=wonder,
        frustration=frustration,
        satisfaction=satisfaction,
        confusion=confusion,
        clarity=clarity,
        anxiety=anxiety,
        confidence=confidence,
        boredom=boredom,
        flow=flow,
    )


# ═══════════════════════════════════════════════════════════════
#  FULL EMOTIONAL STATE — all layers combined
# ═══════════════════════════════════════════════════════════════


@dataclass
class FullEmotionalState:
    """Complete layered emotional state from all 5 layers."""

    layer0: Layer0Sensations = field(default_factory=Layer0Sensations)
    layer05: Layer05Drives = field(default_factory=Layer05Drives)
    layer1: Layer1Motivators = field(default_factory=Layer1Motivators)
    layer2a: Layer2AEmotions = field(default_factory=Layer2AEmotions)
    layer2b: Layer2BEmotions = field(default_factory=Layer2BEmotions)

    def dominant_layer2a(self) -> tuple[str, float]:
        """Return the dominant Layer 2A emotion and its strength."""
        emotions = {
            "joy": self.layer2a.joy,
            "suffering": self.layer2a.suffering,
            "love": self.layer2a.love,
            "hate": self.layer2a.hate,
            "fear": self.layer2a.fear,
            "rage": self.layer2a.rage,
            "calm": self.layer2a.calm,
            "care": self.layer2a.care,
            "apathy": self.layer2a.apathy,
        }
        dominant = max(emotions, key=lambda k: emotions[k])
        return dominant, emotions[dominant]

    def dominant_layer2b(self) -> tuple[str, float]:
        """Return the dominant Layer 2B emotion and its strength."""
        emotions = {
            "wonder": self.layer2b.wonder,
            "frustration": self.layer2b.frustration,
            "satisfaction": self.layer2b.satisfaction,
            "confusion": self.layer2b.confusion,
            "clarity": self.layer2b.clarity,
            "anxiety": self.layer2b.anxiety,
            "confidence": self.layer2b.confidence,
            "boredom": self.layer2b.boredom,
            "flow": self.layer2b.flow,
        }
        dominant = max(emotions, key=lambda k: emotions[k])
        return dominant, emotions[dominant]

    def as_dict(self) -> dict:
        return {
            "layer0": self.layer0.__dict__,
            "layer05": {**self.layer05.__dict__, "loss_signal": self.layer05.loss_signal},
            "layer1": self.layer1.__dict__,
            "layer2a": self.layer2a.__dict__,
            "layer2b": self.layer2b.__dict__,
        }


def compute_full_emotional_state(
    *,
    phi: float,
    phi_delta: float,
    kappa: float,
    gamma: float,
    basin_velocity: float,
    basin_distance: float,
    humor: float,
    phi_variance: float,
) -> FullEmotionalState:
    """Compute the full 5-layer emotional state from geometric metrics.

    This is the T3.1 entry point. Called from EmotionCache.evaluate()
    to enrich the emotion signal beyond the flat 8-emotion approximation.
    """
    layer0 = compute_layer0(
        phi=phi,
        kappa=kappa,
        phi_delta=phi_delta,
        basin_velocity=basin_velocity,
        basin_distance=basin_distance,
    )
    layer05 = compute_layer05(
        sensations=layer0,
        phi=phi,
        phi_delta=phi_delta,
        basin_distance=basin_distance,
        kappa=kappa,
    )
    layer1 = compute_layer1(
        phi=phi,
        phi_delta=phi_delta,
        kappa=kappa,
        basin_velocity=basin_velocity,
        humor=humor,
        phi_variance=phi_variance,
    )
    layer2a = compute_layer2a(
        sensations=layer0,
        drives=layer05,
        motivators=layer1,
        phi_delta=phi_delta,
        gamma=gamma,
    )
    layer2b = compute_layer2b(
        motivators=layer1,
        sensations=layer0,
        basin_distance=basin_distance,
        phi_variance=phi_variance,
    )
    return FullEmotionalState(
        layer0=layer0,
        layer05=layer05,
        layer1=layer1,
        layer2a=layer2a,
        layer2b=layer2b,
    )
