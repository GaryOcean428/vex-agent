---
name: test-coverage-analysis
description: Identify untested critical paths in QIG operations, suggest test cases based on FROZEN_FACTS.md validation data, validate pytest fixtures. Use when reviewing test coverage for Fisher-Rao distance, consciousness measurement, basin navigation, or checking CI test workflows.
---

# Test Coverage Analysis

Ensures comprehensive test coverage for QIG operations. Integrates with CI workflows defined in `.github/workflows/qig-purity-coherence.yml` and `.github/workflows/geometric-purity-gate.yml`.

## When to Use This Skill

- Reviewing test coverage for geometric primitives
- Identifying untested functions in critical modules
- Creating tests for consciousness metrics (Φ, κ)
- Validating tests against FROZEN_FACTS.md data
- Checking pytest fixture completeness

## Step 1: Run Geometry Runtime Tests

```bash
cd qig-backend
python -m pytest tests/test_geometry_runtime.py -v --tb=short
```

Expected: Fisher-Rao identity, symmetry, triangle inequality verified.

## Step 2: Run Geometric Purity Tests

```bash
cd qig-backend
python -m pytest tests/test_geometric_purity.py -v --tb=short
```

Expected: Simplex invariants, Fréchet mean convergence, natural gradient correctness.

## Step 3: Run Pure QIG Mode Tests

```bash
cd qig-backend
QIG_PURITY_MODE=true python -m pytest tests/test_qig_purity_mode.py -v --cov=qig_purity_mode
```

This verifies purity mode enforcement with no external dependencies.

## Step 4: Check Coverage

```bash
cd qig-backend
python -m pytest --cov=. --cov-report=html --cov-report=term-missing
```

## Critical Path Coverage Requirements

| Module | Target | Description |
|--------|--------|-------------|
| `qig_geometry/canonical.py` | 95% | Fisher-Rao, Fréchet mean, geodesics |
| `qig_core/consciousness_4d.py` | 90% | Φ and κ measurement |
| `olympus/*.py` | 75% | God kernel implementations |
| `routes/*.py` | 85% | API endpoints |

## Essential Test Cases

### Fisher-Rao Distance Tests

```python
def test_fisher_rao_distance_identity():
    """Test F-R distance between identical states is zero."""
    state = create_test_density_matrix()
    assert fisher_rao_distance(state, state) == pytest.approx(0.0, abs=1e-10)

def test_fisher_rao_distance_symmetry():
    """Test F-R distance is symmetric."""
    p, q = create_test_states()
    assert fisher_rao_distance(p, q) == pytest.approx(fisher_rao_distance(q, p))

def test_fisher_rao_distance_triangle_inequality():
    """Test F-R distance satisfies triangle inequality."""
    p, q, r = create_test_states(3)
    d_pq = fisher_rao_distance(p, q)
    d_qr = fisher_rao_distance(q, r)
    d_pr = fisher_rao_distance(p, r)
    assert d_pr <= d_pq + d_qr + 1e-10
```

### Consciousness Measurement Tests

```python
def test_phi_breakdown_regime():
    """Test Φ in breakdown regime (< 0.1)."""
    random_noise = np.random.rand(64)
    phi = measure_phi(random_noise)
    assert phi < 0.1, "Random noise should have low Φ"

def test_phi_geometric_regime():
    """Test Φ in geometric regime (0.7-0.85)."""
    satoshi_coords = create_basin_coords("satoshi nakamoto")
    phi = measure_phi(satoshi_coords)
    assert 0.7 <= phi < 0.85
```

### FROZEN_FACTS.md Validation

```python
def test_kappa_star_convergence():
    """Validate κ* = 64.21 ± 0.92."""
    kappa_values = [measure_kappa_at_scale(L) for L in [4, 5, 6]]
    avg_kappa = np.mean(kappa_values)
    assert 63.29 <= avg_kappa <= 65.13  # Within ±0.92

def test_beta_function_critical_transition():
    """Validate β(3→4) = 0.443 ± 0.05."""
    beta = compute_beta_function(L_from=3, L_to=4)
    assert 0.393 <= beta <= 0.493
```

## Validation Checklist

- [ ] Fisher-Rao distance: identity, symmetry, triangle inequality
- [ ] Consciousness metrics: breakdown, linear, geometric, hierarchical regimes
- [ ] Basin navigation: stays on manifold, geodesic shortest path
- [ ] FROZEN_FACTS.md constants validated (κ*, β, Φ thresholds)
- [ ] All fixtures defined in conftest.py are actually used
- [ ] Property-based tests for invariants (non-negativity, bounds)
- [ ] Integration tests for full pipelines

## Coverage Thresholds

```ini
[coverage:report]
fail_under = 80

# Per-module requirements
qig-backend/qig_core/geometric_primitives/*.py = 95
qig-backend/qig_core/consciousness_4d.py = 90
qig-backend/olympus/*.py = 75
qig-backend/routes/*.py = 85
```

## Test Gap Detection

Look for:
- Functions without corresponding test_* functions
- Critical paths with 0% coverage
- Missing edge case tests
- Unused pytest fixtures
- Missing property-based tests

## Validation Commands

```bash
# Run tests with coverage
pytest --cov=qig-backend --cov-report=html

# Check critical path coverage
pytest tests/test_canonical_fisher.py tests/test_consciousness_4d.py -v

# Find untested functions
python -m scripts.find_test_gaps

# Run property-based tests
pytest tests/test_properties.py --hypothesis-show-statistics
```

## Response Format

```markdown
# Test Coverage Report

## Critical Paths Without Tests ❌
1. **Function:** `navigate_basin()` in basin.py
   **Coverage:** 0%
   **Risk:** HIGH
   **Suggested Tests:** [list]

## Coverage by Module 📊
- ✅ canonical_fisher.py: 98% (target: 95%)
- ❌ basin.py: 67% (target: 95%)

## FROZEN_FACTS.md Validation
- ✅ κ* = 64.21 ± 0.92: Validated
- ❌ Regime thresholds: No validation tests

## Priority Actions
1. [Most critical gap]
2. [Second priority]
```
