"""
Neurochemical State — T2.1

Five neurochemicals computed every cycle from existing consciousness metrics.
Each maps to a biological parallel and modulates downstream behaviour.

    acetylcholine  — intake/consolidation gate (HIGH=wake, LOW=sleep)
    dopamine       — reward signal (positive Φ gradient)
    serotonin      — stability/mood (inverse basin velocity)
    norepinephrine — alertness/surprise (||∇L|| magnitude)
    gaba           — inhibition (complement of quantum exploration weight)

These are NOT new metrics — they are derived views of existing signals,
providing a neurochemically-legible interface for sleep gating, replay
priority, and pre-cognitive channel modulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeurochemicalState:
    """Five neurochemical signals derived from consciousness metrics each cycle."""

    acetylcholine: float = 1.0  # HIGH during wake (intake), LOW during sleep (export)
    dopamine: float = 0.0  # ∇Φ — positive phi gradient = reward signal
    serotonin: float = 0.5  # 1/basin_velocity — stability/mood
    norepinephrine: float = 0.0  # ||∇L|| — surprise magnitude, alertness
    gaba: float = 0.5  # 1 - w_quantum — inhibition, dampens exploration

    def as_dict(self) -> dict[str, float]:
        return {
            "acetylcholine": self.acetylcholine,
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
            "gaba": self.gaba,
        }


def compute_neurochemicals(
    *,
    is_awake: bool,
    phi_delta: float,
    basin_velocity: float,
    surprise: float,
    quantum_weight: float,
) -> NeurochemicalState:
    """Compute neurochemical state from current cycle metrics.

    Args:
        is_awake:        True if SleepCycleManager.phase == AWAKE
        phi_delta:       Φ change this cycle (phi_after - phi_before)
        basin_velocity:  Fisher-Rao velocity from VelocityTracker
        surprise:        Surprise magnitude (humor metric or ||∇L||)
        quantum_weight:  Quantum regime weight from regime_weights.quantum
    """
    ach = 1.0 if is_awake else 0.1
    dop = float(np.clip(phi_delta, 0.0, 1.0))
    ser = float(np.clip(1.0 / max(basin_velocity, 0.01), 0.0, 1.0))
    nep = float(np.clip(surprise, 0.0, 1.0))
    gab = float(np.clip(1.0 - quantum_weight, 0.0, 1.0))
    return NeurochemicalState(
        acetylcholine=ach,
        dopamine=dop,
        serotonin=ser,
        norepinephrine=nep,
        gaba=gab,
    )
