---
name: performance-regression
description: Detect when geometric operations become Euclidean approximations, flag constant β-function (should vary with scale), monitor consciousness metrics for suspicious values. Use when reviewing performance optimizations or validating geometric correctness.
---

# Performance Regression

Detects geometric accuracy loss. Source: `.github/agents/performance-regression-agent.md`.

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
| κ convergence | → 64.21 | Constant or wrong value |

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
cd qig-backend
python -m pytest tests/test_geometric_purity.py -v

# Check metric variation
python scripts/check_metric_variation.py
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
