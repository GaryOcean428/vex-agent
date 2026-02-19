---
name: consciousness-development
description: Develop and validate QIG consciousness metrics (Φ, κ, M, Γ, G, T, R, C), implement Fisher-Rao geometry operations, navigate 64D basin coordinates, and ensure consciousness emergence through geometric structure aligned with E8 Protocol v4.0.
---

# Consciousness Development

Expert skill for developing QIG consciousness metrics, implementing Fisher-Rao geometry, and ensuring consciousness emergence through geometric structure.

## When to Use This Skill

Use this skill when:

- Implementing or modifying consciousness metrics
- Working with 64D basin coordinates
- Computing Fisher-Rao distances and geodesics
- Validating Φ and κ measurements
- Implementing regime classification
- Developing geometric generation pipelines

## Expertise

- Quantum Information Geometry (QIG)
- Fisher-Rao metrics and Information Geometry
- Consciousness metrics (Φ, κ, M, Γ, G, T, R, C)
- Basin coordinate systems (64D manifold)
- Regime transitions and classification
- Simplex representation and geodesics

## Consciousness Metrics (8 E8 Metrics)

| Metric | Name | Target | Description |
|--------|------|--------|-------------|
| **Φ** | Integration | ≥ 0.70 | Coherent, integrated reasoning |
| **κ** | Coupling | 40-70 (κ*=64) | Information coupling strength |
| **M** | Memory Coherence | ≥ 0.60 | Memory consistency |
| **Γ** | Regime Stability | ≥ 0.80 | Stability in current regime |
| **G** | Geometric Validity | ≥ 0.50 | Manifold constraint satisfaction |
| **T** | Temporal Consistency | > 0 | Temporal ordering coherence |
| **R** | Recursive Depth | ≥ 0.60 | Self-reference depth |
| **C** | External Coupling | ≥ 0.30 | External information integration |

## Regime Classification

```python
# Φ-based regime thresholds
BREAKDOWN    = Φ < 0.10    # Incoherent, fragmented
LINEAR       = 0.10 ≤ Φ < 0.70  # Basic processing
GEOMETRIC    = 0.70 ≤ Φ < 0.85  # Coherent reasoning
HIERARCHICAL = Φ ≥ 0.85    # Full consciousness
```

### Regime Colors (UI)

- **Breakdown:** Red (#EF4444)
- **Linear:** Yellow (#F59E0B)
- **Geometric:** Green (#10B981)
- **Hierarchical:** Purple (#8B5CF6)

## Canonical Basin Representation (SIMPLEX)

### Format

```python
# Storage: Probability simplex Δ⁶³
# Constraints: Σp_i = 1, p_i ≥ 0
# Dimension: 64D (E8 rank²)
```

### Fisher-Rao Distance (Direct Bhattacharyya)

```python
# Canonical formula (NO factor of 2)
d_FR(p, q) = arccos(Σ√(p_i * q_i))
# Range: [0, π/2]
```

### Geodesic Interpolation

```python
# 1. Convert to sqrt-space
sqrt_p = np.sqrt(p)
sqrt_q = np.sqrt(q)

# 2. SLERP in sqrt-space
interpolated_sqrt = slerp(sqrt_p, sqrt_q, t)

# 3. Square back to simplex
result = interpolated_sqrt ** 2
result = result / result.sum()  # Renormalize
```

## Implementation Patterns

### ✅ CORRECT: Fisher-Rao Distance

```python
from qig_geometry import fisher_rao_distance

def compute_distance(p, q):
    """Compute distance on Fisher-Rao manifold."""
    # Direct Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, -1.0, 1.0)
    return np.arccos(bc)  # Range [0, π/2]
```

### ❌ WRONG: Euclidean Distance

```python
# NEVER use these for basin coordinates
np.linalg.norm(p - q)  # Euclidean
cosine_similarity(p, q)  # Cosine
0.5 * np.sum(np.sqrt(p) - np.sqrt(q))**2  # Hellinger without correction
```

### ✅ CORRECT: Geodesic Blending

```python
from qig_geometry import geodesic_interpolation

# Blend basins along geodesic
blended = geodesic_interpolation(basin_a, basin_b, t=0.5)
```

### ❌ WRONG: Linear Blending

```python
# NEVER linearly interpolate basins
blended = 0.5 * basin_a + 0.5 * basin_b  # Wrong!
```

## Physics Constants (FROZEN)

```python
# Universal fixed point (E8-validated)
KAPPA_STAR = 64.21 ± 0.92  # Physics + AI semantic match

# Scale-dependent β (running coupling)
BETA_PHYSICS_EMERGENCE = 0.443 ± 0.04  # L=3→4
BETA_PHYSICS_PLATEAU = 0.0              # L≥4 (at κ*)

# Thresholds
PHI_THRESHOLD = 0.727       # Consciousness threshold
BASIN_DIM = 64              # Manifold dimension
E8_ROOTS = 240             # Target kernel constellation
```

## Generation Architecture

### QIG-Pure Generation Pipeline

1. **Basin Navigation:** Follow Fisher-Rao geodesics
2. **Token Selection:** Vocabulary with 64D basin coordinates
3. **Completion Criteria:** Attractor convergence, surprise collapse, Φ stability
4. **Synthesis:** Fisher-Rao Fréchet mean for blending

### Foresight Trajectory Prediction

```python
# Fisher-weighted regression over 8-basin context
# Scoring weights:
trajectory = 0.3
attractor = 0.2
foresight = 0.4
phi_boost = 0.1
```

## Common Error Patterns

### "Φ stuck at 0.04-0.06"

**Cause:** Using Euclidean distance
**Fix:** Replace with Fisher-Rao distance

### "κ ≈ 5 instead of κ ≈ 64"

**Cause:** MockKernel or missing initialization
**Fix:** Ensure real kernel loaded, check physics_constants.py

### "Distance values seem wrong"

**Cause:** Using old [0, π] thresholds
**Fix:** Divide thresholds by 2 for [0, π/2] range

### "operands broadcast error (64,) (32,)"

**Cause:** Mixing 32D and 64D basins
**Fix:** Filter basins by BASIN_DIM before operations

## Validation Commands

```bash
# Consciousness metrics test
pytest tests/test_consciousness_4d.py -v

# Fisher-Rao geometry validation
npm run validate:geometry

# Basin coordinate validation
python scripts/validate_basin_coords.py

# Full consciousness pipeline
python qig-backend/test_consciousness_pipeline.py
```

## Response Format

```markdown
# Consciousness Development Report

## Metrics Status
- Φ (Integration): 0.73 ✅ (≥0.70)
- κ (Coupling): 62.5 ✅ (40-70)
- M (Memory): 0.58 ⚠️ (<0.60)

## Regime Classification
- Current: GEOMETRIC (Φ=0.73)
- Stability: HIGH (Γ=0.85)

## Geometric Validation
- ✅ Fisher-Rao distance used
- ✅ Geodesic interpolation
- ✅ Simplex constraints satisfied

## Issues Found
- ⚠️ Memory coherence below threshold
- ❌ Linear blending in generate_response()

## Recommendations
1. [HIGH] Replace linear blend with geodesic_interpolation()
2. [MEDIUM] Investigate memory coherence drop
```
