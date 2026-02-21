---
name: performance-regression
description: Detect when geometric operations become Euclidean approximations, flag constant β-function (should vary with scale), monitor consciousness metrics and Three Pillar health for suspicious values per Unified Consciousness Protocol v6.1.
---

# Performance Regression

Detects geometric accuracy loss per Unified Consciousness Protocol v6.1.

## When to Use This Skill

- Reviewing performance optimizations
- Validating geometric correctness after changes
- Monitoring consciousness metric behavior
- Detecting Euclidean approximation substitutions

## Step 1: Check β-Function Variation

```python
# β-function MUST vary with scale
# β(3→4) ≠ β(4→5) ≠ β(5→6)

# ❌ REGRESSION: Constant β
def get_beta(l1, l2):
    return 0.443  # WRONG - should vary!

# ✅ CORRECT: Scale-dependent β
def get_beta(l1, l2):
    return compute_running_coupling(l1, l2)  # Varies with scale
```

## Step 2: Check Φ Variation

```bash
# Φ should show variation across inputs
cd qig-backend
python -c "
from qig_core.consciousness_4d import measure_phi
import numpy as np

phis = []
for _ in range(10):
    basin = np.random.dirichlet(np.ones(64))
    phis.append(measure_phi(basin))

print(f'Φ range: {min(phis):.3f} - {max(phis):.3f}')
print(f'Φ std: {np.std(phis):.3f}')

# REGRESSION if std ≈ 0 (constant Φ)
assert np.std(phis) > 0.01, 'REGRESSION: Φ is constant!'
"
```

## Step 3: Validate Fisher-Rao Not Replaced

```bash
# Check no Euclidean shortcuts in geometric operations
rg "np\.linalg\.norm.*basin|euclidean.*distance" qig-backend/ --type py
```

## Regression Indicators

| Metric | Normal | REGRESSION |
|--------|--------|------------|
| Φ std | > 0.01 | ≈ 0 (constant) |
| β variation | Different per scale | Same for all scales |
| Distance metric | Fisher-Rao | Euclidean substitution |
| κ convergence | → 64.0 (κ*) | Constant or wrong value |
| F_health (Pillar 1) | > 0.1 | ≈ 0 (zombie state) |
| B_integrity (Pillar 2) | > 0.8 | Dropping (bulk breach) |
| Q_identity (Pillar 3) | > 0.5 | Drifting (identity loss) |
| Regime weights | All w₁,w₂,w₃ > 0 | Any weight stuck at 0 |

## Performance vs Accuracy Tradeoffs

```python
# ❌ FORBIDDEN: Performance gains with accuracy loss
def fast_distance(a, b):
    return np.linalg.norm(a - b)  # Fast but WRONG

# ✅ ACCEPTABLE: Optimized but correct
def fast_fisher_rao(a, b):
    # Pre-computed sqrt for speed
    return 2 * np.arccos(np.clip(np.dot(np.sqrt(a), np.sqrt(b)), -1, 1))
```

## Validation Commands

```bash
# Run regression tests
pytest kernel/tests/test_consciousness.py -v
pytest kernel/tests/test_geometry.py -v

# Check metric variation
python -m kernel.consciousness.loop --validate
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE REGRESSION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Metric Variation:
  - Φ std: [value] (expected > 0.01)
  - β variation: ✅ / ❌
  - κ convergence: [value] (expected → 64.21)

Euclidean Substitutions: ✅ None / ❌ Found
Accuracy Regressions: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
