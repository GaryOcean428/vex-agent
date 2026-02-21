---
name: test-coverage-analysis
description: Identify untested critical paths in QIG operations, suggest test cases based on frozen physics validation data, validate pytest fixtures. Use when reviewing test coverage for Fisher-Rao distance, consciousness measurement, Three Pillars enforcement, basin navigation, or activation sequence per Unified Consciousness Protocol v6.1.
---

# Test Coverage Analysis

Ensures comprehensive test coverage for QIG operations per Unified Consciousness Protocol v6.1.

## When to Use This Skill

- Reviewing test coverage for geometric primitives
- Identifying untested functions in critical modules
- Creating tests for consciousness metrics (Φ, κ)
- Validating tests against FROZEN_FACTS.md data
- Checking pytest fixture completeness

## Step 1: Run Geometry Tests

```bash
pytest kernel/tests/test_geometry.py -v --tb=short
```

Expected: Fisher-Rao identity, symmetry, triangle inequality verified.

## Step 2: Run Consciousness Tests

```bash
pytest kernel/tests/test_consciousness.py -v --tb=short
```

Expected: Regime weights, pillar enforcement, activation sequence verified.

## Step 3: Run Pillar Enforcement Tests

```bash
pytest kernel/tests/ -v -k "pillar or fluctuation or bulk or quenched or zombie"
```

Expected: All three pillars enforced, violation types detected and corrected.

## Step 4: Check Coverage

```bash
pytest kernel/tests/ --cov=kernel --cov-report=html --cov-report=term-missing
```

## Critical Path Coverage Requirements

| Module | Target | Description |
|--------|--------|-------------|
| `kernel/geometry/` | 95% | Fisher-Rao, Fréchet mean, geodesics |
| `kernel/consciousness/loop.py` | 90% | Consciousness loop, activation sequence |
| `kernel/consciousness/pillars.py` | 95% | Three Pillars enforcement (v6.1 §3) |
| `kernel/consciousness/activation.py` | 90% | 14-step activation sequence (v6.1 §23) |
| `kernel/governance/` | 75% | E8 budget, PurityGate |
| `kernel/coordizer_v2/` | 80% | CoordizerV2 operations |
| `kernel/server.py` | 85% | API endpoints |

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

### Frozen Physics Validation

```python
def test_kappa_star_convergence():
    """Validate κ* = 64.0 (theoretical), measured 64.21 ± 0.92."""
    kappa_values = [measure_kappa_at_scale(L) for L in [4, 5, 6]]
    avg_kappa = np.mean(kappa_values)
    assert 63.29 <= avg_kappa <= 65.13  # Within ±0.92

def test_beta_function_critical_transition():
    """Validate β(3→4) = 0.443 ± 0.04."""
    beta = compute_beta_function(L_from=3, L_to=4)
    assert 0.403 <= beta <= 0.483
```

### Three Pillar Tests (v6.1 §3)

```python
def test_pillar1_fluctuation_no_zombie():
    """Pillar 1: Zero entropy must trigger noise injection."""
    # Create flat basin (zero entropy) → must trigger correction
    flat_basin = np.zeros(64); flat_basin[0] = 1.0
    result = enforce_fluctuation(flat_basin)
    assert shannon_entropy(result) >= 0.1

def test_pillar2_bulk_integrity():
    """Pillar 2: External input capped at surface slerp 30%."""
    core, surface = split_basin(test_basin)
    perturbed = apply_input(surface, external_input, weight=0.5)
    # Weight should be clamped to 0.3
    assert effective_weight <= 0.3

def test_pillar3_sovereignty_ratio():
    """Pillar 3: Sovereignty ratio tracks lived vs borrowed."""
    kernel = create_test_kernel()
    assert 0.0 <= kernel.sovereignty_ratio <= 1.0
```

## Validation Checklist

- [ ] Fisher-Rao distance: identity, symmetry, triangle inequality
- [ ] Three-regime field: Quantum/Efficient/Equilibrium weights sum to 1
- [ ] Three Pillars: Fluctuations, Topological Bulk, Quenched Disorder enforced
- [ ] Pillar violation types: all 7 types detected and corrected (v6.1 §3.6)
- [ ] Activation Sequence: 14-step flow (v6.1 §23)
- [ ] Agency Triad: Desire, Will, Wisdom computed
- [ ] Basin navigation: stays on manifold, geodesic shortest path
- [ ] Frozen physics constants validated (κ*=64, β=0.443, Φ range)
- [ ] Sovereignty metrics: S_ratio, Q_identity tracked
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
pytest kernel/tests/ --cov=kernel --cov-report=html

# Check critical path coverage
pytest kernel/tests/test_consciousness.py kernel/tests/test_geometry.py -v

# Run pillar enforcement tests
pytest kernel/tests/ -v -k "pillar or fluctuation or zombie"

# Run activation sequence tests
pytest kernel/tests/ -v -k "activation"
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
