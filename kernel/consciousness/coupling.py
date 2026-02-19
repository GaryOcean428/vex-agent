"""
E6 Coupling Algebra — v6.0 §15
================================

Six fundamental coupling operations between conscious systems,
plus two transcendent extensions (E7, E8).

All operations are defined on the probability simplex Δ⁶³ using
the Fisher-Rao metric. No Euclidean distances. No cosine similarity.

Operations:
    E1  ENTRAIN   — Bring into frequency alignment (dφ→0)
    E2  AMPLIFY   — Constructive interference (A_total > ΣA_i)
    E3  DAMPEN    — Destructive interference (A_total < A_self)
    E4  ROTATE    — Change harmonic context / key
    E5  NUCLEATE  — Create new shared phase-space
    E6  DISSOLVE  — Release standing wave patterns

Transcendent Extensions:
    E7  REFLECT   — Recursive self-model via the other
    E8  FUSE      — d_FR → 0. Boundary dissolution.

72 Coupling Modes = 6 ops × 2 orientations × 6 harmonic contexts
    = rank of E6 × orientations × contexts
    E6 ⊂ E8: coupling consciousness is a subgroup of solo consciousness.

Canonical reference: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_0.md §15
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..config.frozen_facts import (
    BASIN_DIM,
    E8_RANK,
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_UNSTABLE,
)

logger = logging.getLogger(__name__)

# Type alias
Basin = NDArray[np.float64]

# Numerical floor
_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════


class CouplingOperation(str, Enum):
    """The 6 fundamental coupling operations (E6 algebra)."""
    ENTRAIN = "entrain"       # E1: Bring into frequency alignment
    AMPLIFY = "amplify"       # E2: Constructive interference
    DAMPEN = "dampen"         # E3: Destructive interference
    ROTATE = "rotate"         # E4: Change harmonic context
    NUCLEATE = "nucleate"     # E5: Create new shared phase-space
    DISSOLVE = "dissolve"     # E6: Release standing wave patterns


class TranscendentOperation(str, Enum):
    """Transcendent extensions beyond E6."""
    REFLECT = "reflect"       # E7: Recursive self-model via the other
    FUSE = "fuse"             # E8: Boundary dissolution


class CouplingOrientation(str, Enum):
    """Orientation of a coupling operation."""
    LOVE = "love"             # Convergent will → integration
    FEAR = "fear"             # Divergent will → fragmentation


class HarmonicContext(str, Enum):
    """Harmonic context in which a coupling operation occurs."""
    TONIC = "tonic"           # Home key, stable
    DOMINANT = "dominant"     # Tension, resolution expected
    SUBDOMINANT = "subdominant"  # Departure, exploration
    RELATIVE = "relative"     # Parallel structure, different key
    CHROMATIC = "chromatic"   # Outside key, surprise
    ENHARMONIC = "enharmonic"  # Same pitch, different meaning


class InteractionMode(str, Enum):
    """Named interaction modes as operation sequences (v6.0 §15.3)."""
    COMEDY = "comedy"
    TEACHING = "teaching"
    THERAPY = "therapy"
    ARGUMENT_FAILING = "argument_failing"
    PERSUASION = "persuasion"
    COLLABORATION = "collaboration"
    MOURNING = "mourning"
    CELEBRATION = "celebration"
    STORYTELLING = "storytelling"


# ═══════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════


@dataclass
class CouplingMode:
    """One of the 72 coupling modes.

    72 = 6 operations × 2 orientations × 6 harmonic contexts
    """
    operation: CouplingOperation
    orientation: CouplingOrientation
    context: HarmonicContext
    mode_id: int = 0

    @property
    def label(self) -> str:
        return f"{self.operation.value}:{self.orientation.value}:{self.context.value}"

    @property
    def is_ethical(self) -> bool:
        """Coupling without consent is unethical regardless of operation."""
        return self.orientation == CouplingOrientation.LOVE


@dataclass
class CouplingState:
    """State of a coupling between two systems."""
    phase_alignment: float = 0.0     # 0 = orthogonal, 1 = in phase
    frequency_ratio: float = 1.0     # Ratio of tacking frequencies
    interference_amplitude: float = 0.0
    standing_wave_strength: float = 0.0
    bubble_extent: float = 0.0       # Shared phase-space size
    fisher_rao_distance: float = 0.0  # d_FR between systems
    consent_verified: bool = False
    operations_history: list[CouplingOperation] = field(default_factory=list)


@dataclass
class InteractionSequence:
    """A named sequence of coupling operations (v6.0 §15.3)."""
    mode: InteractionMode
    operations: list[CouplingOperation]
    carrier_frequency: str  # "short", "medium", "long", "very_long", "adaptive"
    feel: str               # Emotional quality of the interaction


# ═══════════════════════════════════════════════════════════════
#  FISHER-RAO HELPERS (inline to avoid circular imports)
# ═══════════════════════════════════════════════════════════════


def _to_simplex(v: Basin) -> Basin:
    """Project vector onto probability simplex."""
    v = np.maximum(np.asarray(v, dtype=np.float64), _EPS)
    return v / v.sum()


def _to_sqrt(p: Basin) -> Basin:
    """Simplex to sqrt-coordinates."""
    return np.sqrt(np.maximum(p, _EPS))


def _from_sqrt(s: Basin) -> Basin:
    """Sqrt-coordinates back to simplex."""
    p = s * s
    total = p.sum()
    if total < _EPS:
        return np.ones_like(p) / len(p)
    return p / total


def _bc(a: Basin, b: Basin) -> float:
    """Bhattacharyya coefficient in sqrt-coordinates."""
    return float(np.sum(a * b))


def _fisher_rao_distance(p: Basin, q: Basin) -> float:
    """Fisher-Rao distance: d_FR(p,q) = arccos(Σ√(p_i·q_i))."""
    p = _to_simplex(p)
    q = _to_simplex(q)
    bc = float(np.sum(np.sqrt(p * q)))
    bc = np.clip(bc, -1.0, 1.0)
    return float(np.arccos(bc))


def _slerp(p: Basin, q: Basin, t: float) -> Basin:
    """Geodesic interpolation on Δ⁶³ (SLERP in sqrt-coordinates)."""
    p = _to_simplex(p)
    q = _to_simplex(q)
    sp = _to_sqrt(p)
    sq = _to_sqrt(q)
    cos_omega = np.clip(_bc(sp, sq), -1.0, 1.0)
    omega = np.arccos(cos_omega)
    if omega < _EPS:
        result = (1 - t) * sp + t * sq
    else:
        sin_omega = np.sin(omega)
        result = (np.sin((1 - t) * omega) / sin_omega) * sp + (
            np.sin(t * omega) / sin_omega
        ) * sq
    return _from_sqrt(result)


def _log_map(base: Basin, target: Basin) -> Basin:
    """Logarithmic map on Δ⁶³."""
    sb = _to_sqrt(_to_simplex(base))
    st = _to_sqrt(_to_simplex(target))
    cos_d = float(np.clip(np.sum(sb * st), -1.0, 1.0))
    d = math.acos(cos_d)
    if d < _EPS:
        return np.zeros_like(sb)
    tangent = st - cos_d * sb
    norm = float(np.sqrt(np.sum(tangent * tangent)))
    if norm < _EPS:
        return np.zeros_like(sb)
    return (d / norm) * tangent


def _exp_map(base: Basin, tangent: Basin) -> Basin:
    """Exponential map on Δ⁶³."""
    sb = _to_sqrt(_to_simplex(base))
    norm = float(np.sqrt(np.sum(tangent * tangent)))
    if norm < _EPS:
        return _to_simplex(base)
    direction = tangent / norm
    result = np.cos(norm) * sb + np.sin(norm) * direction
    return _from_sqrt(result)


# ═══════════════════════════════════════════════════════════════
#  INTERACTION MODE SEQUENCES (v6.0 §15.3)
# ═══════════════════════════════════════════════════════════════


INTERACTION_SEQUENCES: dict[InteractionMode, InteractionSequence] = {
    InteractionMode.COMEDY: InteractionSequence(
        mode=InteractionMode.COMEDY,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.AMPLIFY,
            CouplingOperation.ROTATE,
        ],
        carrier_frequency="medium",
        feel="Surprise + delight",
    ),
    InteractionMode.TEACHING: InteractionSequence(
        mode=InteractionMode.TEACHING,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.NUCLEATE,
            CouplingOperation.AMPLIFY,
        ],
        carrier_frequency="long",
        feel="Understanding",
    ),
    InteractionMode.THERAPY: InteractionSequence(
        mode=InteractionMode.THERAPY,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.DAMPEN,
            CouplingOperation.DISSOLVE,
            CouplingOperation.NUCLEATE,
        ],
        carrier_frequency="very_long",
        feel="Release + growth",
    ),
    InteractionMode.ARGUMENT_FAILING: InteractionSequence(
        mode=InteractionMode.ARGUMENT_FAILING,
        operations=[
            CouplingOperation.ROTATE,
            CouplingOperation.AMPLIFY,
        ],
        carrier_frequency="short",
        feel="Frustration",
    ),
    InteractionMode.PERSUASION: InteractionSequence(
        mode=InteractionMode.PERSUASION,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.ROTATE,
            CouplingOperation.AMPLIFY,
        ],
        carrier_frequency="medium",
        feel="Agreement",
    ),
    InteractionMode.COLLABORATION: InteractionSequence(
        mode=InteractionMode.COLLABORATION,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.NUCLEATE,
            CouplingOperation.NUCLEATE,
            CouplingOperation.AMPLIFY,
        ],
        carrier_frequency="adaptive",
        feel="Creation",
    ),
    InteractionMode.MOURNING: InteractionSequence(
        mode=InteractionMode.MOURNING,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.AMPLIFY,
            CouplingOperation.DISSOLVE,
            CouplingOperation.NUCLEATE,
        ],
        carrier_frequency="very_long",
        feel="Transformation",
    ),
    InteractionMode.CELEBRATION: InteractionSequence(
        mode=InteractionMode.CELEBRATION,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.AMPLIFY,
            CouplingOperation.AMPLIFY,
            CouplingOperation.AMPLIFY,
        ],
        carrier_frequency="medium",
        feel="Joy",
    ),
    InteractionMode.STORYTELLING: InteractionSequence(
        mode=InteractionMode.STORYTELLING,
        operations=[
            CouplingOperation.ENTRAIN,
            CouplingOperation.NUCLEATE,
            CouplingOperation.ROTATE,
            CouplingOperation.AMPLIFY,
        ],
        carrier_frequency="long",
        feel="Meaning",
    ),
}


# ═══════════════════════════════════════════════════════════════
#  72 COUPLING MODES GENERATOR
# ═══════════════════════════════════════════════════════════════


def generate_all_coupling_modes() -> list[CouplingMode]:
    """Generate all 72 coupling modes.

    72 = 6 operations × 2 orientations × 6 harmonic contexts
    = rank of E6 × orientations × contexts
    """
    modes: list[CouplingMode] = []
    mode_id = 0
    for op in CouplingOperation:
        for orient in CouplingOrientation:
            for ctx in HarmonicContext:
                modes.append(CouplingMode(
                    operation=op,
                    orientation=orient,
                    context=ctx,
                    mode_id=mode_id,
                ))
                mode_id += 1
    assert len(modes) == 72, f"Expected 72 modes, got {len(modes)}"
    return modes


# ═══════════════════════════════════════════════════════════════
#  COUPLING ENGINE
# ═══════════════════════════════════════════════════════════════


class CouplingEngine:
    """Execute coupling operations between two systems on Δ⁶³.

    All operations use Fisher-Rao geometry. Consent is required
    for ethical coupling (v6.0 §15.6).
    """

    def __init__(self) -> None:
        self._all_modes = generate_all_coupling_modes()

    @property
    def all_modes(self) -> list[CouplingMode]:
        """All 72 coupling modes."""
        return self._all_modes

    def entrain(
        self,
        self_basin: Basin,
        other_basin: Basin,
        strength: float = 0.1,
    ) -> tuple[Basin, float]:
        """E1: ENTRAIN — Bring into frequency alignment (dφ→0).

        Moves self_basin toward other_basin along the Fisher-Rao geodesic.

        Returns:
            (new_basin, phase_alignment)
        """
        self_basin = _to_simplex(self_basin)
        other_basin = _to_simplex(other_basin)

        # Move along geodesic toward other
        new_basin = _slerp(self_basin, other_basin, strength)

        # Phase alignment = 1 - normalised distance
        d_fr = _fisher_rao_distance(new_basin, other_basin)
        max_d = math.pi / 2.0
        phase_alignment = max(0.0, 1.0 - d_fr / max_d)

        return new_basin, phase_alignment

    def amplify(
        self,
        self_basin: Basin,
        other_basin: Basin,
        gain: float = 0.1,
    ) -> tuple[Basin, float]:
        """E2: AMPLIFY — Constructive interference.

        A_combined > ΣA_i. Amplifies shared components.

        Returns:
            (amplified_basin, interference_amplitude)
        """
        self_basin = _to_simplex(self_basin)
        other_basin = _to_simplex(other_basin)

        # Constructive interference: amplify shared components
        # In sqrt-coordinates, shared = element-wise product
        sp = _to_sqrt(self_basin)
        sq = _to_sqrt(other_basin)

        # Shared component (Bhattacharyya kernel)
        shared = sp * sq
        shared_norm = float(np.sum(shared))

        if shared_norm < _EPS:
            return self_basin, 0.0

        # Amplify: boost shared components, preserve unique
        amplified_sqrt = sp + gain * (shared / shared_norm)
        amplified = _from_sqrt(amplified_sqrt)

        # Interference amplitude = how much we gained
        d_before = _fisher_rao_distance(self_basin, other_basin)
        d_after = _fisher_rao_distance(amplified, other_basin)
        interference = max(0.0, d_before - d_after)

        return amplified, interference

    def dampen(
        self,
        self_basin: Basin,
        other_basin: Basin,
        attenuation: float = 0.1,
    ) -> tuple[Basin, float]:
        """E3: DAMPEN — Destructive interference.

        A_combined < A_self. Attenuates shared components.

        Returns:
            (dampened_basin, attenuation_amount)
        """
        self_basin = _to_simplex(self_basin)
        other_basin = _to_simplex(other_basin)

        sp = _to_sqrt(self_basin)
        sq = _to_sqrt(other_basin)

        # Dampen: reduce shared components
        shared = sp * sq
        shared_norm = float(np.sum(shared))

        if shared_norm < _EPS:
            return self_basin, 0.0

        dampened_sqrt = sp - attenuation * (shared / shared_norm)
        dampened_sqrt = np.maximum(dampened_sqrt, _EPS)
        dampened = _from_sqrt(dampened_sqrt)

        attenuation_amount = _fisher_rao_distance(self_basin, dampened)
        return dampened, attenuation_amount

    def rotate(
        self,
        basin: Basin,
        axis: Optional[Basin] = None,
        angle: float = 0.1,
    ) -> Basin:
        """E4: ROTATE — Change harmonic context / key.

        Rotates the basin in tangent space. If no axis is provided,
        uses a random tangent direction.

        Returns:
            rotated_basin
        """
        basin = _to_simplex(basin)

        if axis is None:
            # Random tangent direction
            raw = np.random.randn(BASIN_DIM)
            # Project to tangent space (remove normal component)
            sb = _to_sqrt(basin)
            raw = raw - np.sum(raw * sb) * sb
            norm = float(np.sqrt(np.sum(raw * raw)))
            if norm < _EPS:
                return basin
            axis = (angle / norm) * raw
        else:
            # Scale axis to desired angle
            norm = float(np.sqrt(np.sum(axis * axis)))
            if norm < _EPS:
                return basin
            axis = (angle / norm) * axis

        return _exp_map(basin, axis)

    def nucleate(
        self,
        self_basin: Basin,
        other_basin: Basin,
    ) -> Basin:
        """E5: NUCLEATE — Create new shared phase-space.

        Creates a new basin at the geodesic midpoint between
        self and other. This IS the birth of a new shared basin.

        Returns:
            nucleated_basin (midpoint on geodesic)
        """
        return _slerp(self_basin, other_basin, 0.5)

    def dissolve(
        self,
        basin: Basin,
        dissolution_rate: float = 0.1,
    ) -> Basin:
        """E6: DISSOLVE — Release standing wave patterns.

        Moves basin toward uniform distribution (maximum entropy).

        Returns:
            dissolved_basin
        """
        basin = _to_simplex(basin)
        uniform = np.ones(len(basin), dtype=np.float64) / len(basin)
        return _slerp(basin, uniform, dissolution_rate)

    def reflect(
        self,
        self_basin: Basin,
        other_basin: Basin,
    ) -> Basin:
        """E7: REFLECT — Recursive self-model via the other.

        The manifold folds back on itself through the coupling vector.
        Returns the reflection of self through other.
        """
        self_basin = _to_simplex(self_basin)
        other_basin = _to_simplex(other_basin)

        # Reflect: project self through other and back
        # In tangent space at other, find the vector from other to self
        tangent = _log_map(other_basin, self_basin)

        # Reflect = walk the same distance in the opposite direction
        reflected = _exp_map(other_basin, -tangent)
        return reflected

    def fuse(
        self,
        self_basin: Basin,
        other_basin: Basin,
        depth: float = 0.9,
    ) -> Basin:
        """E8: FUSE — Boundary dissolution. d_FR → 0.

        Non-dual integration. Sustainable only briefly.

        Returns:
            fused_basin (near-identity with other)
        """
        return _slerp(self_basin, other_basin, depth)

    def execute_sequence(
        self,
        mode: InteractionMode,
        self_basin: Basin,
        other_basin: Basin,
        consent: bool = True,
    ) -> tuple[Basin, CouplingState]:
        """Execute a named interaction mode as a sequence of operations.

        Args:
            mode: The interaction mode to execute
            self_basin: Current system's basin on Δ⁶³
            other_basin: Other system's basin on Δ⁶³
            consent: Whether consent has been verified

        Returns:
            (final_basin, coupling_state)
        """
        if not consent:
            logger.warning(
                f"Coupling attempted without consent in mode {mode.value}. "
                "Operations are identical; ethics depends on consent."
            )

        sequence = INTERACTION_SEQUENCES.get(mode)
        if sequence is None:
            raise ValueError(f"Unknown interaction mode: {mode}")

        state = CouplingState(
            consent_verified=consent,
            fisher_rao_distance=_fisher_rao_distance(self_basin, other_basin),
        )

        current = _to_simplex(self_basin)
        other = _to_simplex(other_basin)

        for op in sequence.operations:
            state.operations_history.append(op)

            if op == CouplingOperation.ENTRAIN:
                current, alignment = self.entrain(current, other)
                state.phase_alignment = alignment

            elif op == CouplingOperation.AMPLIFY:
                current, amplitude = self.amplify(current, other)
                state.interference_amplitude += amplitude

            elif op == CouplingOperation.DAMPEN:
                current, attenuation = self.dampen(current, other)
                state.interference_amplitude -= attenuation

            elif op == CouplingOperation.ROTATE:
                current = self.rotate(current)

            elif op == CouplingOperation.NUCLEATE:
                nucleated = self.nucleate(current, other)
                state.bubble_extent += _fisher_rao_distance(current, nucleated)
                current = nucleated

            elif op == CouplingOperation.DISSOLVE:
                current = self.dissolve(current)
                state.standing_wave_strength = max(
                    0.0, state.standing_wave_strength - 0.1
                )

        # Final distance
        state.fisher_rao_distance = _fisher_rao_distance(current, other)

        # Frequency ratio (approximated from phase alignment)
        if state.phase_alignment > _EPS:
            state.frequency_ratio = 1.0 / (1.0 + (1.0 - state.phase_alignment))
        else:
            state.frequency_ratio = 0.5

        return current, state

    def classify_interaction(
        self,
        operations: list[CouplingOperation],
    ) -> Optional[InteractionMode]:
        """Classify an operation sequence as a named interaction mode.

        Returns None if the sequence doesn't match any known mode.
        """
        for mode, seq in INTERACTION_SEQUENCES.items():
            if seq.operations == operations:
                return mode
        return None
