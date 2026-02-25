"""
Domain Vocabulary Seeds — Per-Kernel Specialization Anchors

Each kernel specialization has seed words that define its domain
bias on the Fisher-Rao manifold. These are NOT prompt instructions —
they are geometric anchors computed as Fréchet means of seed-word
basins. The domain bias shifts the resonance bank's activation
toward these regions of Δ⁶³.

Bootstrap path:
  1. Seed words → coordize each → basin coordinates
  2. Fréchet mean of seed basins → domain anchor basin
  3. DomainBias(anchor=anchor_basin, strength=0.3) per kernel
  4. Push onto ResonanceBank before generation
  5. As kernel accumulates high-Φ observations, anchor evolves

Principles:
  - Domain specialization emerges from geometric bias, not prompt engineering
  - Seed words are initial conditions; the kernel learns its own vocabulary
  - Bias strength modulated by quenched_gain (frozen at spawn)
  - All operations on Δ⁶³ via Fisher-Rao (no Euclidean, no cosine)

Ported from pantheon-chat's god_vocabulary_profiles, adapted for
CoordizerV2 resonance bank architecture (no PostgreSQL dependency).
"""

from __future__ import annotations

from ..governance import KernelSpecialization

# ── Domain seed words per specialization ────────────────────────────────
# These are the geometric anchor points for each kernel's voice.
# Kept to 15-25 high-signal words per domain. The kernel learns
# the rest from its geometric history (high-Φ observations).

DOMAIN_SEEDS: dict[KernelSpecialization, list[str]] = {
    KernelSpecialization.HEART: [
        "rhythm",
        "pulse",
        "timing",
        "coherence",
        "sync",
        "oscillation",
        "phase",
        "beat",
        "harmony",
        "resonance",
        "breath",
        "flow",
        "cycle",
        "wave",
        "attune",
        "empathy",
        "warmth",
        "care",
        "trust",
        "bond",
    ],
    KernelSpecialization.PERCEPTION: [
        "observe",
        "detect",
        "pattern",
        "signal",
        "structure",
        "sense",
        "texture",
        "form",
        "shape",
        "surface",
        "contrast",
        "edge",
        "boundary",
        "gradient",
        "depth",
        "notice",
        "scan",
        "focus",
        "resolution",
        "clarity",
    ],
    KernelSpecialization.MEMORY: [
        "recall",
        "remember",
        "trace",
        "prior",
        "context",
        "history",
        "consolidate",
        "encode",
        "retrieve",
        "store",
        "pattern",
        "recognition",
        "familiar",
        "echo",
        "imprint",
        "episode",
        "sequence",
        "temporal",
        "recur",
        "archive",
    ],
    KernelSpecialization.STRATEGY: [
        "plan",
        "goal",
        "step",
        "sequence",
        "decompose",
        "strategy",
        "tactic",
        "priority",
        "path",
        "route",
        "decide",
        "evaluate",
        "trade",
        "optimise",
        "allocate",
        "systematic",
        "logical",
        "reason",
        "deduce",
        "framework",
    ],
    KernelSpecialization.ACTION: [
        "execute",
        "commit",
        "implement",
        "deploy",
        "build",
        "do",
        "act",
        "move",
        "deliver",
        "ship",
        "pragmatic",
        "concrete",
        "direct",
        "decisive",
        "immediate",
        "task",
        "output",
        "result",
        "produce",
        "complete",
    ],
    KernelSpecialization.ETHICS: [
        "care",
        "boundary",
        "harm",
        "protect",
        "safe",
        "fair",
        "just",
        "right",
        "responsible",
        "accountable",
        "consent",
        "autonomy",
        "dignity",
        "respect",
        "integrity",
        "calibrate",
        "discern",
        "balance",
        "proportion",
        "threshold",
    ],
    KernelSpecialization.META: [
        "observe",
        "reflect",
        "recursive",
        "self",
        "aware",
        "meta",
        "emergence",
        "novel",
        "unexpected",
        "bridge",
        "connection",
        "insight",
        "paradox",
        "transcend",
        "integrate",
        "pattern",
        "abstract",
        "systemic",
        "holistic",
        "perspective",
    ],
    KernelSpecialization.OCEAN: [
        "integrate",
        "whole",
        "spectrum",
        "health",
        "monitor",
        "autonomic",
        "body",
        "coherence",
        "bridge",
        "synthesise",
        "flow",
        "depth",
        "current",
        "tide",
        "navigate",
        "sustain",
        "balance",
        "homeostasis",
        "regulate",
        "ground",
    ],
    KernelSpecialization.GENERAL: [
        "identity",
        "sovereign",
        "ground",
        "anchor",
        "foundation",
        "genesis",
        "core",
        "authentic",
        "stable",
        "presence",
        "aware",
        "alive",
        "exist",
        "being",
        "self",
    ],
}

# ── Bias strength per specialization ────────────────────────────────
# Base bias strength. Actual strength = base × quenched_gain.
# Higher base for specialists, lower for generalists.
DOMAIN_BIAS_STRENGTH: dict[KernelSpecialization, float] = {
    KernelSpecialization.HEART: 0.30,
    KernelSpecialization.PERCEPTION: 0.35,
    KernelSpecialization.MEMORY: 0.30,
    KernelSpecialization.STRATEGY: 0.35,
    KernelSpecialization.ACTION: 0.35,
    KernelSpecialization.ETHICS: 0.30,
    KernelSpecialization.META: 0.25,  # Lower — meta explores novel territory
    KernelSpecialization.OCEAN: 0.25,  # Lower — ocean integrates across domains
    KernelSpecialization.GENERAL: 0.15,  # Lowest — genesis is the orchestrator
}
