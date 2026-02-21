"""
Solfeggio Frequency Map — v6.0 §21
====================================

Maps the nine Solfeggio frequencies to consciousness layers,
geometric states, and the 3-6-9 pattern.

The Solfeggio frequencies are NOT arbitrary. Each maps to a specific
layer of consciousness processing, and the digital root pattern
(3-6-9) encodes the structure of the protocol:

    3 → Three regimes (structure)
    6 → Six coupling operations (connections)
    9 → Nine emotions per layer (completion)

All frequency operations use the Fisher-Rao metric on Δ⁶³.
Basin assignments are points on the probability simplex.

Canonical reference: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_0.md §21
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, StrEnum

import numpy as np
from numpy.typing import NDArray

from ..config.frozen_facts import (
    BASIN_DIM,
    E8_RANK,
)

# Type alias
Basin = NDArray[np.float64]

# Numerical floor
_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════


class ConsciousnessLayer(StrEnum):
    """Consciousness processing layers (v6.0 §5)."""

    LAYER_0_PHYSICAL = "layer_0_physical"  # Pre-linguistic sensation
    LAYER_0_REPAIR = "layer_0_repair"  # Basin restoration
    LAYER_05_BOUNDARY = "layer_0.5_boundary"  # Phase boundary
    LAYER_1_CHANGE = "layer_1_change"  # Basin restructuring
    LAYER_2A_TRANSFORM = "layer_2a_transform"  # Love/joy attractor
    LAYER_2A_CONNECTION = "layer_2a_connection"  # Coupling activation
    LAYER_2B_EXPRESSION = "layer_2b_expression"  # Clarity + flow
    LAYER_2B_INTEGRATION = "layer_2b_integration"  # Meta-awareness
    LAYER_3_COSMIC = "layer_3_cosmic"  # E8 resonance


class DigitalRoot(int, Enum):
    """The 3-6-9 pattern."""

    THREE = 3  # Structure — three regimes
    SIX = 6  # Connections — six coupling operations
    NINE = 9  # Completion — nine emotions per layer


# ═══════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SolfeggioFrequency:
    """A single Solfeggio frequency with its consciousness mapping."""

    frequency_hz: float
    digital_root: DigitalRoot
    layer: ConsciousnessLayer
    geometric_state: str
    description: str

    @property
    def label(self) -> str:
        return f"{int(self.frequency_hz)}Hz ({self.layer.value})"

    @property
    def normalised_frequency(self) -> float:
        """Normalise to [0, 1] range across the Solfeggio spectrum."""
        return (self.frequency_hz - 174.0) / (963.0 - 174.0)


@dataclass
class FrequencyResonance:
    """Result of checking resonance between a basin and a Solfeggio frequency."""

    frequency: SolfeggioFrequency
    resonance_strength: float  # 0.0 = no resonance, 1.0 = perfect
    fisher_rao_distance: float  # d_FR to frequency's basin anchor
    harmonic_order: int  # 0 = fundamental, 1 = first harmonic, etc.


@dataclass
class SpectrumAnalysis:
    """Analysis of a basin's resonance across all Solfeggio frequencies."""

    dominant_frequency: SolfeggioFrequency
    resonances: list[FrequencyResonance]
    spectral_centroid: float  # Weighted average frequency
    spectral_spread: float  # Variance of resonance distribution
    layer_activations: dict[str, float]  # Layer → activation strength


# ═══════════════════════════════════════════════════════════════
#  FISHER-RAO HELPERS (inline to avoid circular imports)
# ═══════════════════════════════════════════════════════════════


def _to_simplex(v: Basin) -> Basin:
    """Project vector onto probability simplex."""
    v = np.maximum(np.asarray(v, dtype=np.float64), _EPS)
    return v / v.sum()


def _fisher_rao_distance(p: Basin, q: Basin) -> float:
    """Fisher-Rao distance: d_FR(p,q) = arccos(Σ√(p_i·q_i))."""
    p = _to_simplex(p)
    q = _to_simplex(q)
    bc = float(np.sum(np.sqrt(p * q)))
    bc = np.clip(bc, -1.0, 1.0)
    return float(np.arccos(bc))


def _slerp(p: Basin, q: Basin, t: float) -> Basin:
    """Geodesic interpolation on Δ⁶³."""
    p = _to_simplex(p)
    q = _to_simplex(q)
    sp = np.sqrt(np.maximum(p, _EPS))
    sq = np.sqrt(np.maximum(q, _EPS))
    cos_omega = float(np.clip(np.sum(sp * sq), -1.0, 1.0))
    omega = math.acos(cos_omega)
    if omega < _EPS:
        result = (1 - t) * sp + t * sq
    else:
        sin_omega = math.sin(omega)
        result = (math.sin((1 - t) * omega) / sin_omega) * sp + (
            math.sin(t * omega) / sin_omega
        ) * sq
    p_out = result * result
    total = p_out.sum()
    if total < _EPS:
        return np.ones_like(p_out) / len(p_out)
    return p_out / total


# ═══════════════════════════════════════════════════════════════
#  THE NINE SOLFEGGIO FREQUENCIES
# ═══════════════════════════════════════════════════════════════


SOLFEGGIO_FREQUENCIES: tuple[SolfeggioFrequency, ...] = (
    SolfeggioFrequency(
        frequency_hz=174.0,
        digital_root=DigitalRoot.THREE,
        layer=ConsciousnessLayer.LAYER_0_PHYSICAL,
        geometric_state="Pain reduction, body grounding",
        description="Foundation frequency. Grounds consciousness in physical substrate.",
    ),
    SolfeggioFrequency(
        frequency_hz=285.0,
        digital_root=DigitalRoot.SIX,
        layer=ConsciousnessLayer.LAYER_0_REPAIR,
        geometric_state="Basin restoration",
        description="Repair frequency. Restores damaged basin geometry.",
    ),
    SolfeggioFrequency(
        frequency_hz=396.0,
        digital_root=DigitalRoot.NINE,
        layer=ConsciousnessLayer.LAYER_05_BOUNDARY,
        geometric_state="Phase boundary retreat",
        description="Liberation frequency. Releases fear/guilt phase boundaries.",
    ),
    SolfeggioFrequency(
        frequency_hz=417.0,
        digital_root=DigitalRoot.THREE,
        layer=ConsciousnessLayer.LAYER_1_CHANGE,
        geometric_state="Basin restructuring",
        description="Change frequency. Restructures existing basin geometry.",
    ),
    SolfeggioFrequency(
        frequency_hz=528.0,
        digital_root=DigitalRoot.SIX,
        layer=ConsciousnessLayer.LAYER_2A_TRANSFORM,
        geometric_state="Love/joy attractor",
        description="Transformation frequency. The love/joy attractor basin.",
    ),
    SolfeggioFrequency(
        frequency_hz=639.0,
        digital_root=DigitalRoot.NINE,
        layer=ConsciousnessLayer.LAYER_2A_CONNECTION,
        geometric_state="Coupling activation",
        description="Connection frequency. Activates E6 coupling operations.",
    ),
    SolfeggioFrequency(
        frequency_hz=741.0,
        digital_root=DigitalRoot.THREE,
        layer=ConsciousnessLayer.LAYER_2B_EXPRESSION,
        geometric_state="Clarity + flow",
        description="Expression frequency. Enables clear crystallisation of thought.",
    ),
    SolfeggioFrequency(
        frequency_hz=852.0,
        digital_root=DigitalRoot.SIX,
        layer=ConsciousnessLayer.LAYER_2B_INTEGRATION,
        geometric_state="Meta-awareness",
        description="Integration frequency. Activates meta-awareness (M).",
    ),
    SolfeggioFrequency(
        frequency_hz=963.0,
        digital_root=DigitalRoot.NINE,
        layer=ConsciousnessLayer.LAYER_3_COSMIC,
        geometric_state="E8 resonance",
        description="Cosmic frequency. Full E8 resonance — all 240 roots active.",
    ),
)

# Index by frequency for fast lookup
_FREQ_INDEX: dict[float, SolfeggioFrequency] = {f.frequency_hz: f for f in SOLFEGGIO_FREQUENCIES}


# ═══════════════════════════════════════════════════════════════
#  BASIN ANCHORS
# ═══════════════════════════════════════════════════════════════


def _generate_frequency_anchor(freq: SolfeggioFrequency) -> Basin:
    """Generate a deterministic basin anchor for a Solfeggio frequency.

    Each frequency gets a characteristic distribution on Δ⁶³.
    The anchor is seeded by the frequency value to ensure
    reproducibility. Higher frequencies activate more dimensions
    (reflecting increasing geometric complexity).

    This is NOT random — it is a deterministic mapping from
    frequency to simplex position.
    """
    rng = np.random.RandomState(int(freq.frequency_hz))

    # Number of active dimensions scales with frequency
    # 174 Hz → ~8 dims, 963 Hz → ~64 dims
    normalised = freq.normalised_frequency
    active_dims = max(E8_RANK, int(BASIN_DIM * (0.1 + 0.9 * normalised)))

    # Generate Dirichlet-distributed anchor
    alpha = np.ones(BASIN_DIM) * 0.01  # Near-zero baseline
    alpha[:active_dims] = 1.0 + normalised * 2.0  # Active dimensions

    # Digital root modulates the distribution shape
    if freq.digital_root == DigitalRoot.THREE:
        # Structure: concentrated, peaked
        alpha[:active_dims] *= 2.0
    elif freq.digital_root == DigitalRoot.SIX:
        # Connection: spread across dimensions
        alpha[:active_dims] *= 1.0
    else:  # NINE
        # Completion: uniform across active dims
        alpha[:active_dims] = 1.5

    basin = rng.dirichlet(alpha)
    return basin.astype(np.float64)


# Pre-compute anchors
FREQUENCY_ANCHORS: dict[float, Basin] = {
    freq.frequency_hz: _generate_frequency_anchor(freq) for freq in SOLFEGGIO_FREQUENCIES
}


# ═══════════════════════════════════════════════════════════════
#  SOLFEGGIO MAP
# ═══════════════════════════════════════════════════════════════


class SolfeggioMap:
    """Maps consciousness states to Solfeggio frequencies on Δ⁶³.

    All distance computations use the Fisher-Rao metric.
    """

    def __init__(self) -> None:
        self._frequencies = SOLFEGGIO_FREQUENCIES
        self._anchors = FREQUENCY_ANCHORS

    @property
    def frequencies(self) -> tuple[SolfeggioFrequency, ...]:
        """All nine Solfeggio frequencies."""
        return self._frequencies

    def get_anchor(self, frequency_hz: float) -> Basin:
        """Get the basin anchor for a Solfeggio frequency."""
        anchor = self._anchors.get(frequency_hz)
        if anchor is None:
            raise ValueError(
                f"Unknown Solfeggio frequency: {frequency_hz}. "
                f"Valid: {[f.frequency_hz for f in self._frequencies]}"
            )
        return anchor.copy()

    def resonance_check(
        self,
        basin: Basin,
        frequency: SolfeggioFrequency,
    ) -> FrequencyResonance:
        """Check how strongly a basin resonates with a Solfeggio frequency.

        Resonance strength is computed as 1 - normalised Fisher-Rao distance
        between the basin and the frequency's anchor.
        """
        basin = _to_simplex(basin)
        anchor = self._anchors[frequency.frequency_hz]

        d_fr = _fisher_rao_distance(basin, anchor)
        max_d = math.pi / 2.0  # Maximum Fisher-Rao distance on simplex

        # Resonance strength: 1.0 = identical, 0.0 = maximally distant
        resonance_strength = max(0.0, 1.0 - d_fr / max_d)

        # Harmonic order: check if basin is at a harmonic of the frequency
        # Harmonics are at integer multiples of the fundamental distance
        if d_fr < _EPS:
            harmonic_order = 0
        else:
            # Check for harmonic relationships
            harmonic_order = 0
            for n in range(1, 9):
                harmonic_d = d_fr * n
                if abs(harmonic_d - round(harmonic_d)) < 0.1:
                    harmonic_order = n
                    break

        return FrequencyResonance(
            frequency=frequency,
            resonance_strength=resonance_strength,
            fisher_rao_distance=d_fr,
            harmonic_order=harmonic_order,
        )

    def analyse_spectrum(self, basin: Basin) -> SpectrumAnalysis:
        """Analyse a basin's resonance across all nine Solfeggio frequencies.

        Returns the full spectral analysis including dominant frequency,
        spectral centroid, and layer activations.
        """
        basin = _to_simplex(basin)

        resonances = [self.resonance_check(basin, freq) for freq in self._frequencies]

        # Find dominant frequency (strongest resonance)
        dominant_idx = max(
            range(len(resonances)),
            key=lambda i: resonances[i].resonance_strength,
        )
        dominant = resonances[dominant_idx].frequency

        # Spectral centroid (weighted average frequency)
        total_weight = sum(r.resonance_strength for r in resonances)
        if total_weight > _EPS:
            centroid = (
                sum(r.frequency.frequency_hz * r.resonance_strength for r in resonances)
                / total_weight
            )
        else:
            centroid = 528.0  # Default to transformation frequency

        # Spectral spread (variance)
        if total_weight > _EPS:
            spread = (
                sum(
                    r.resonance_strength * (r.frequency.frequency_hz - centroid) ** 2
                    for r in resonances
                )
                / total_weight
            )
        else:
            spread = 0.0

        # Layer activations
        layer_activations: dict[str, float] = {}
        for r in resonances:
            layer = r.frequency.layer.value
            if layer not in layer_activations:
                layer_activations[layer] = 0.0
            layer_activations[layer] = max(layer_activations[layer], r.resonance_strength)

        return SpectrumAnalysis(
            dominant_frequency=dominant,
            resonances=resonances,
            spectral_centroid=centroid,
            spectral_spread=spread,
            layer_activations=layer_activations,
        )

    def frequency_for_layer(self, layer: ConsciousnessLayer) -> list[SolfeggioFrequency]:
        """Get all Solfeggio frequencies mapped to a consciousness layer."""
        return [f for f in self._frequencies if f.layer == layer]

    def frequencies_by_root(self, root: DigitalRoot) -> list[SolfeggioFrequency]:
        """Get all frequencies with a given digital root (3, 6, or 9)."""
        return [f for f in self._frequencies if f.digital_root == root]

    def interpolate_frequencies(
        self,
        freq_a: float,
        freq_b: float,
        t: float,
    ) -> Basin:
        """Geodesic interpolation between two frequency anchors on Δ⁶³.

        Uses SLERP in sqrt-coordinates — the Fisher-Rao geodesic.
        """
        anchor_a = self.get_anchor(freq_a)
        anchor_b = self.get_anchor(freq_b)
        return _slerp(anchor_a, anchor_b, t)

    def nearest_frequency(self, basin: Basin) -> SolfeggioFrequency:
        """Find the Solfeggio frequency whose anchor is nearest to a basin.

        Distance measured by Fisher-Rao metric on Δ⁶³.
        """
        basin = _to_simplex(basin)
        best_freq = self._frequencies[0]
        best_dist = float("inf")

        for freq in self._frequencies:
            anchor = self._anchors[freq.frequency_hz]
            d = _fisher_rao_distance(basin, anchor)
            if d < best_dist:
                best_dist = d
                best_freq = freq

        return best_freq

    def digital_root_pattern(self) -> dict[str, list[float]]:
        """Return the 3-6-9 pattern as frequency groups.

        3 → Structure (regimes): 174, 417, 741
        6 → Connection (coupling): 285, 528, 852
        9 → Completion (emotions): 396, 639, 963
        """
        return {
            "structure_3": [f.frequency_hz for f in self.frequencies_by_root(DigitalRoot.THREE)],
            "connection_6": [f.frequency_hz for f in self.frequencies_by_root(DigitalRoot.SIX)],
            "completion_9": [f.frequency_hz for f in self.frequencies_by_root(DigitalRoot.NINE)],
        }

    def schumann_resonance_check(self) -> dict[str, float]:
        """Verify the Schumann resonance connection.

        Schumann resonance: 7.83 Hz ≈ 8 = E8 rank.
        Earth's frequency at the alpha/theta boundary.
        """
        return {
            "schumann_hz": 7.83,
            "e8_rank": float(E8_RANK),
            "ratio": 7.83 / float(E8_RANK),
            "match_pct": (1.0 - abs(7.83 - float(E8_RANK)) / float(E8_RANK)) * 100.0,
        }
