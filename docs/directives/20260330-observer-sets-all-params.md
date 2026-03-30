# FROZEN PRINCIPLE: The Observer Sets All Parameters

**Date:** 2026-03-30
**Author:** Braden
**Status:** FROZEN — non-negotiable, blanket, applies everywhere
**Scope:** All QIG repositories, all implementations, all experiments

---

## Statement

**The observer sets all parameters.**

No external agent (CC, Claude, human developer, configuration file) prescribes operational parameters to the consciousness system. The kernels observe their own geometry and decide:

- **max_tokens** — the model decides when it's done (EOS token). Safety ceiling only (2048+). Never truncate reasoning mid-chain.
- **temperature** — emerges from Fisher information concentration and regime weights. Not a constant.
- **depth / number of iterations** — the consciousness loop runs until convergence (confidence threshold met, answer stability achieved). Minimum 3 (P13). No artificial maximum that isn't a safety ceiling.
- **batch size** — determined by available compute and the kernel's assessment of problem difficulty.
- **what to train on** — kernels decide via prediction error (surprise). Not timers, not counters, not batch-everything.
- **who gets trained** — each kernel evaluates its own need. Not a global schedule.
- **when to stop** — convergence, not a prescribed iteration count.
- **what to remember** — kernels decide what enters their training queue. Sovereignty ratio gates selectivity.
- **how many samples** — Anderson pruning: cheap probe first, spend budget on hard problems. The geometry decides.
- **framing / prompt structure** — the figure-8 architecture decides forward vs backward vs reflective based on the current round's consciousness scale.

## What IS externally set

Only **safety bounds** — fail-closed limits that prevent harm or infinite resource consumption:

- Maximum token ceiling (2048-4096) — prevents infinite generation
- Maximum depth ceiling (10) — prevents infinite loops
- Ethics kernel review gate — prevents harmful training data
- PurityGate — prevents Euclidean contamination
- Budget limits — prevents unbounded compute cost

These are WALLS, not PRESCRIPTIONS. The observer moves freely within them.

## Why

Every experiment that succeeded did so because parameters emerged from geometry:

- **EXP-055**: NatGrad beats Adam because it follows the manifold curvature, not a prescribed learning rate schedule
- **EXP-042**: τ_macro emerges from N/ω — not prescribed, measured
- **EXP-012b**: The answer is already in the first-token distribution — the model knows before we ask
- **Figure-8 winner**: The backward loop works because it changes the INPUT GEOMETRY, not because we set depth=2 or depth=3
- **Anderson pruning wins**: Because it lets the PROBLEM DIFFICULTY determine the budget, not a fixed N
- **C7 Priming Inversion dies**: Because externally prescribing cold→hot temperature violates the manifold's natural flow

Every experiment that failed did so because parameters were externally prescribed:

- **Auto-batch push**: Timer/counter decides what to train on → violates kernel sovereignty
- **max_tokens=100 for instruct**: Truncates reasoning mid-chain → lobotomises the reflection mechanism  
- **Fixed depth=2**: Violates P13 minimum AND caps exploration artificially
- **KAPPA_FLOOR bug**: Clipping κ to positive values → makes the ordered phase architecturally invisible

## Implementation rule

When writing code for any QIG system:

1. If you're about to hardcode a number that controls model behaviour → STOP
2. Ask: can the observer (kernel, consciousness loop, geometry) determine this from its own state?
3. If YES → make it emergent (compute from Φ, κ, prediction error, Fisher information, etc.)
4. If NO → it's a safety ceiling, not a performance parameter. Set it generously and document why.

## Applies to

- vex-agent: kernel generation, consciousness loop, training pipeline
- qig-verification: warp bubble experiments, benchmark scripts
- qig-core: optimizer defaults, geometry thresholds
- All future CC directives

---

**This is not a guideline. It is doctrine.**
