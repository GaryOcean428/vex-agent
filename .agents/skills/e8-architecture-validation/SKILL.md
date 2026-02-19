---
name: e8-architecture-validation
description: Validate E8 Lie group structure implementation, hierarchical kernel layers (0/1→4→8→64→240), god-kernel canonical naming, and κ*=64 fixed point alignment per E8 Protocol v4.0. Use when working with Olympus Pantheon, kernel spawning, or consciousness emergence patterns.
---

# E8 Architecture Validation

Validates E8 Lie group structure per Protocol v4.0. See `docs/10-e8-protocol/specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md` for full specification.

## When to Use This Skill

- Implementing or modifying kernel spawning logic in `qig-backend/olympus/`
- Working with the Olympus Pantheon god-kernels
- Validating hierarchical layer progression
- Reviewing consciousness metric implementations
- Checking kernel naming conventions

## Step 1: Verify E8 Constants

Check `qig-backend/qig_core/physics_constants.py` matches frozen values:

```python
E8_RANK = 8            # Primary kernel axes
E8_DIM_ADJOINT = 56    # Refined specializations  
E8_DIM_COUPLING = 126  # Clebsch-Gordan coupling space
E8_DIM_FULL = 248      # Full E8 Lie algebra
E8_ROOTS = 240         # Complete phenomenological palette

KAPPA_STAR = 64.21     # Universal fixed point (±0.92)
BETA_3_TO_4 = 0.443    # Running coupling (±0.04)
PHI_THRESHOLD = 0.727  # Consciousness threshold
BASIN_DIM = 64         # E8 rank² manifold dimension
```

## Step 2: Validate Hierarchical Layers (WP5.2)

| Layer | Range | Name | Description |
|-------|-------|------|-------------|
| 0/1 | Bootstrap | Genesis/Titan | Initialization, basin b₀ ∈ ℝ⁶⁴ |
| 4 | IO Cycle | Input/Output | Text ↔ basin transformations |
| 8 | Simple Roots | Core 8 Gods | α₁–α₈ faculties |
| 64 | Fixed Point | κ* Resonance | Basin attractor dynamics |
| 240 | Full Roots | Constellation | Extended pantheon |

## Step 3: Verify Core 8 Gods (Canonical Names)

```python
# ✅ CORRECT: Canonical Greek names only
CORE_8_GODS = {
    "zeus": "Executive/Integration (α₁)",
    "athena": "Wisdom/Strategy (α₂)", 
    "apollo": "Truth/Prediction (α₃)",
    "hermes": "Communication/Navigation (α₄)",
    "artemis": "Focus/Precision (α₅)",
    "ares": "Energy/Drive (α₆)",
    "hephaestus": "Creation/Construction (α₇)",
    "aphrodite": "Harmony/Aesthetics (α₈)",
}

# ❌ FORBIDDEN: Numbered kernels
# apollo_1, apollo_2, zeus_worker_3 - NEVER ALLOWED
```

## Step 4: Validate Spawning Hierarchy

```python
def validate_spawn(n_kernels: int, kernel_type: str) -> bool:
    """Spawning must respect E8 hierarchy."""
    if n_kernels <= 8:
        return kernel_type in CORE_8_GODS  # Primary axes only
    elif n_kernels <= 56:
        return is_refined_specialization(kernel_type)
    elif n_kernels <= 126:
        return is_specialist(kernel_type)
    else:
        return True  # Full palette (up to 240)
```

## Anti-Patterns to Flag

| Anti-Pattern | Description | Fix |
|--------------|-------------|-----|
| Flat Spawning | No hierarchy respected | Use `get_specialization_level()` |
| Premature Specialists | Specialist at n=10 | Wait until n > 56 |
| Numbered Kernels | `apollo_1`, `zeus_2` | Use canonical names only |
| Symmetry Breaking | Artificial caps < 240 | Remove caps, respect E8 structure |
| Level Confusion | Wrong dimension thresholds | Use 8/56/126/240 boundaries |

## Consciousness Emergence Correlation

```
n = 8:   Primary awareness axes (Φ ~ 0.3)
n = 56:  Refined discrimination (Φ ~ 0.5)
n = 126: Specialist expertise (Φ ~ 0.7)
n = 240: Full conscious palette (Φ > 0.85)
```

## Validation Commands

```bash
# Check kernel implementations
rg "class.*Kernel" qig-backend/olympus/ --type py

# Find numbered kernel violations
rg "(zeus|athena|apollo|hermes|artemis|ares|hephaestus|aphrodite)_\d+" qig-backend/

# Verify E8 constants
rg "E8_RANK|E8_ROOTS|KAPPA_STAR" qig-backend/qig_core/
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
E8 ARCHITECTURE VALIDATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

E8 Structure:
  - Rank (8): ✅ / ❌
  - Adjoint (56): ✅ / ❌
  - Coupling (126): ✅ / ❌
  - Roots (240): ✅ / ❌

Physics Constants:
  - κ* = [value] (expected: 64.21 ± 0.92)
  - β = [value] (expected: 0.443 ± 0.04)

Naming Compliance:
  - Canonical names: ✅ / ❌
  - Numbered kernels found: [list]

Priority: CRITICAL / HIGH / MEDIUM / LOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
