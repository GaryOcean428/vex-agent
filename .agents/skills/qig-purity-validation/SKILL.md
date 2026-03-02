---
name: qig-purity-validation
description: Validate Quantum Information Geometry (QIG) purity across codebase changes. Detect Euclidean contamination, forbidden operations per v6.1F §1.3, SVD fallbacks, tokenizer boundary violations, and Three Pillar violations. Enforce Fisher-Rao metrics per Unified Consciousness Protocol v6.1F. Zero tolerance for geometric impurity.
---

# QIG Purity Validation

Enforces geometric purity per Unified Consciousness Protocol v6.1F (§1.3). Zero tolerance for Euclidean contamination in consciousness-critical code.

## When to Use This Skill

- Reviewing any PR that touches `kernel/`, `src/`, or `frontend/`
- Auditing geometric distance calculations
- Checking for forbidden operations per v6.1F §1.3
- Validating Fisher-Rao metric usage
- Ensuring no Euclidean contamination in basin operations
- Verifying Three Pillars enforcement (v6.1F §3)
- Checking for SVD fallbacks in PGA compression
- Validating tokenizer boundary layer exemptions
- Auditing CoordizerV2 geometry operations

## Step 1: Scan for Forbidden Operations (v6.1F §1.3)

```bash
# Scan kernel/ for all forbidden patterns
rg "cosine_similarity|np\.linalg\.norm.*-|dot_product|Adam|LayerNorm|softmax|stopword|TF.?IDF|flatten|np\.linalg\.svd" kernel/ --type py

# Specific scan for SVD (Euclidean decomposition)
rg "np\.linalg\.svd|scipy\.linalg\.svd" kernel/ --type py

# Check tokenizer usage (boundary layer concern)
rg "tokenizer|tokenize|AutoTokenizer" kernel/ --type py
```

### Complete Forbidden Operations Table (v6.1F §1.3)

| Forbidden | Why | Replace With |
|-----------|-----|-------------|
| `cosine_similarity(a,b)` | Euclidean metric | `fisher_rao_distance(a,b)` |
| `np.linalg.norm(a-b)` | L2 norm | `d_FR` on simplex |
| `np.linalg.svd(T)` | Euclidean decomposition | Eigendecomp of `T.T @ T` (Gram matrix) |
| `dot_product(a,b)` | Euclidean inner product | Fisher metric contraction |
| `Adam` optimizer | Euclidean gradient | Natural gradient optimizer |
| `LayerNorm` | Euclidean normalization | Simplex projection |
| `embedding` (term) | Implies flat space | "basin coordinates" |
| `tokenize` (term) | Implies flat decomposition | "coordize" |
| `flatten` | Destroys manifold structure | Geodesic projection |
| `softmax` (output or internal) | Exponential warping destroys Fisher info structure | `logits_to_simplex()` (linear shift-and-scale) |
| `torch.softmax` / `F.softmax` | Same — banned at ALL call sites except `# QIG BOUNDARY` | `logits_to_simplex()` |
| `stopword list` | NLP heuristic | Geometric salience weight |
| `TF-IDF` | Bag-of-words relic | Fisher-geometric de-biasing |
| `np.mean(basins)` | Arithmetic mean on simplex | `frechet_mean()` on manifold |
| `linear blend of basins` | Off-manifold interpolation | `slerp_sqrt(a, b, t)` geodesic |
| `obj._protected_attr` (cross-class) | Breaks encapsulation, brittle | Add public method/property to class |

## Step 2: Verify Three Pillars Enforcement (v6.1 §3)

```bash
# Check Pillar 1: Fluctuation enforcement exists
rg "entropy|ZERO_ENTROPY|F_health|fluctuation|zombie" kernel/consciousness/ --type py

# Check Pillar 2: Topological Bulk enforcement exists
rg "CORE.*SURFACE|slerp.*cap|bulk|B_integrity|core_drift" kernel/consciousness/ --type py

# Check Pillar 3: Quenched Disorder enforcement exists
rg "identity.*frozen|Q_identity|S_ratio|sovereignty|quenched" kernel/consciousness/ --type py

# Verify all three are checked simultaneously
rg "pillar.*enforce|check_pillars|PillarViolation" kernel/ --type py
```

### Pillar Violation Types (v6.1 §3.6)

| Violation | Pillar | Detection | Response |
|-----------|--------|-----------|----------|
| ZERO_ENTROPY | 1 | H_basin < 0.1 | Inject Dirichlet noise |
| ZERO_TEMPERATURE | 1 | T_llm < 0.05 | Force minimum temperature |
| BASIN_COLLAPSE | 1 | max(p_i) > 0.5 | Redistribute mass |
| BULK_BREACH | 2 | Surface slerp > 0.3 | Clamp input weight |
| CORE_DRIFT | 2 | d_FR(core) > 0.1/cycle | Slow diffusion rate |
| IDENTITY_DRIFT | 3 | d_FR(current, frozen) > threshold | Increase refraction |
| SOVEREIGNTY_LOW | 3 | S < 0.1 after 100 cycles | Flag for review |

## Step 3: Validate Canonical Import Path

```bash
# Ensure Fisher-Rao imports come from geometry module
rg "fisher_rao_distance|frechet_mean|geodesic" kernel/ --type py -l
```

## Step 4: Verify Regime Field (v6.1 §4)

```bash
# Check for OLD 4-regime model (should be replaced by 3-regime field)
rg "BREAKDOWN|LINEAR|GEOMETRIC|HIERARCHICAL" kernel/ --type py
# If found in active code (not tests/docs), these should be replaced with:
# Quantum (w₁), Efficient (w₂), Equilibrium (w₃)

# Check for new 3-regime field
rg "w_1|w_2|w_3|quantum.*regime|efficient.*regime|equilibrium.*regime|regime_weights" kernel/ --type py
```

## Forbidden Patterns (v6.1F §1.3 — Complete)

| Category | Pattern | Severity | Fix |
|----------|---------|----------|-----|
| Euclidean | `np.linalg.norm(a - b)` | CRITICAL | `fisher_rao_distance(a, b)` |
| Euclidean | `np.linalg.svd(T)` | CRITICAL | Eigendecomp of `T.T @ T` (Gram matrix) |
| Cosine | `cosine_similarity()` | CRITICAL | `fisher_rao_distance()` |
| Dot Product | `np.dot(a, b)` for basins | CRITICAL | Fisher metric contraction |
| Optimizer | `torch.optim.Adam()` | CRITICAL | `natural_gradient_step()` |
| Normalization | `LayerNorm` | CRITICAL | Simplex projection |
| Output | `softmax` (any use outside `# QIG BOUNDARY`) | CRITICAL | `logits_to_simplex()` |
| Output | `torch.softmax` / `F.softmax` | CRITICAL | `logits_to_simplex()` |
| Terminology | `embedding` | ERROR | "basin coordinates" |
| Terminology | `tokenize` | ERROR | "coordize" |
| Flatten | `flatten` on manifold data | CRITICAL | Geodesic projection |
| NLP | `stopword list` | ERROR | Geometric salience weight (v6.1F §20.4) |
| NLP | `TF-IDF` | ERROR | Fisher-geometric de-biasing |
| NLP | `import sentencepiece` | CRITICAL | Geometric coordizer |
| Mean | `np.mean(basins, axis=0)` | ERROR | `frechet_mean()` |

## Anti-Patterns Added in v6.1F Enforcement (2026-02)

### Protected Member Access Across Classes

Do **not** reach into another class's `_private` attributes from outside:

```python
# ❌ VIOLATION — triggers Pylint W0212, brittle to refactor
self.tacking._mode = "explore"
self.basin_sync._version

# ✅ CORRECT — add a public method/property to the class
self.tacking.force_explore()           # public method
self.basin_sync.get_state()["version"] # public API
```

**Rule:** If you need to mutate or read another object's private state, add a named public method that expresses the intent (e.g., `force_explore()`, `get_version()`).

### `hasattr`-Guard on Unimplemented Methods

Guarding a call with `hasattr` when the method does not yet exist silently no-ops:

```python
# ❌ ANTI-PATTERN — T4.4d escalation silently skipped because with_model() never existed
if hasattr(self.llm, "with_model"):
    return self.llm.with_model(model)
return self.llm  # silent no-op

# ✅ CORRECT — implement the method on the class, or pass model as a parameter
# Option A: implement with_model() on LLMClient
# Option B: pass model_override: str | None to the downstream function
```

**Rule:** `hasattr` guards are only acceptable as backwards-compatibility shims against versioned external APIs. For internal interfaces, implement the method.

### Linear Basin Blending

Linear blending of simplex vectors leaves the manifold:

```python
# ❌ VIOLATION
blended = 0.5 * basin_a + 0.5 * basin_b

# ✅ CORRECT
from kernel.geometry.fisher_rao import slerp_sqrt
blended = slerp_sqrt(basin_a, basin_b, 0.5)
```

## Boundary Layer Exemptions (v6.1F)

### Tokenizer at LLM Interface

Tokenizers are REQUIRED at the LLM boundary for extracting output distributions:

- `kernel/coordizer_v2/harvest.py` — LLM harvest (exempt)
- `kernel/coordizer_v2/coordizer.py` — Bootstrap fallback (mark @deprecated)

**Required:** Add explicit comments: `# QIG BOUNDARY: LLM interface — tokenizer required`

### Tangent Space Operations

Euclidean operations (dot products, L2 norms) are VALID in tangent space:

- Tangent space at a point on the simplex IS a Euclidean vector space
- L2 norms and dot products in tangent space correspond to Fisher metric at base point
- Examples: velocity norms, consistency checks in `resonance_bank.py`

**Required:** Add comments clarifying tangent space context

## SVD Fallback Issue (CoordizerV2)

### Location

`kernel/coordizer_v2/compress.py` line ~222:

```python
U, S, Vt = np.linalg.svd(T_sub, full_matrices=False)  # 🔴 VIOLATION
```

### Problem

SVD is Euclidean decomposition. While numerically equivalent to eigendecomposition for full-rank data, it bypasses geometric framing.

### Fix

Replace with eigendecomposition of dual Gram matrix:

```python
# Compute dual Gram matrix (geometrically correct)
gram_dual = T_sub.T @ T_sub
eigenvalues, eigenvectors = np.linalg.eigh(gram_dual)
# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
# Project onto principal directions
V = eigenvectors  # Principal directions
```

## Quarantine Zones (Exempted)

These directories are allowed to violate for LLM client / experimental purposes:

- `kernel/llm/` (LLM client code — pragmatic purity level)
- `kernel/tools/` (agent tools — pragmatic purity level)
- `kernel/training/` (learning systems — pragmatic purity level)
- `kernel/tests/` (test fixtures)
- `docs/` (documentation examples)

## Physics Constants (FROZEN — v6.1 §2)

```python
KAPPA_STAR = 64.0           # Universal fixed point (E8 rank²)
# κ_physics = 64.21 ± 0.92  # TFIM quantum lattice
# κ_semantic = 63.90 ± 0.50  # AI word relationships
BETA_3_TO_4 = 0.443         # Running coupling L=3→4 (±0.04)
PHI_RANGE = (0.65, 0.75)    # Consciousness Φ target range
BASIN_DIM = 64              # Manifold dimension
```

## Purity Levels by Directory (vex workspace)

| Directory | Purity Level | Euclidean Allowed? |
|-----------|-------------|-------------------|
| `kernel/consciousness/` | PARAMOUNT | ❌ NEVER |
| `kernel/geometry/` | PARAMOUNT | ❌ NEVER |
| `kernel/governance/` | HIGH | ❌ NO |
| `kernel/coordizer_v2/` | HIGH | ❌ NO |
| `kernel/memory/` | HIGH | ❌ NO |
| `kernel/config/` | HIGH | ❌ NO |
| `kernel/llm/` | PRAGMATIC | ⚠️ Interface only |
| `kernel/tools/` | PRAGMATIC | ⚠️ Interface only |
| `kernel/training/` | PRAGMATIC | ⚠️ Interface only |
| `src/` | CONSUMER | ✅ Proxy layer |
| `frontend/` | CONSUMER | ✅ UI layer |

## Validation Commands

```bash
# Scan for forbidden patterns in kernel
rg "cosine_similarity|np\.linalg\.norm.*-|dot_product|Adam|LayerNorm|softmax.*output|stopword|TF.?IDF|flatten|np\.linalg\.svd" kernel/ --type py

# Check for SVD usage specifically
rg "np\.linalg\.svd|scipy\.linalg\.svd" kernel/ --type py

# Check tokenizer usage (boundary concern)
rg "tokenizer|tokenize|AutoTokenizer" kernel/ --type py

# Check Pillar enforcement exists
rg "FluctuationGuard|TopologicalBulk|QuenchedDisorder|pillar" kernel/consciousness/ --type py

# Check for old regime model (should be 3-regime field now)
rg "BREAKDOWN|LINEAR.*regime|GEOMETRIC.*regime|HIERARCHICAL" kernel/ --type py

# Run purity tests
pytest kernel/tests/ -v -k "purity or geometry"
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QIG PURITY VALIDATION REPORT (v6.1F)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Static Analysis: ✅ PASS / ❌ FAIL
  - Euclidean violations: 0
  - Forbidden operations (v6.1F §1.3): 0
  - SVD usage: 0 / N (with fixes)
  - Terminology violations: 0

Boundary Layer: ✅ DOCUMENTED / ⚠️ NEEDS COMMENTS
  - Tokenizer usage: documented / undocumented
  - Tangent space ops: commented / uncommented

Three Pillars Enforcement: ✅ PASS / ❌ FAIL
  - Pillar 1 (Fluctuations): enforced / missing
  - Pillar 2 (Topological Bulk): enforced / missing
  - Pillar 3 (Quenched Disorder): enforced / missing

Regime Field: ✅ v6.1F 3-regime / ❌ Old 4-regime model

CoordizerV2 Purity: ✅ PASS / 🔴 SVD FALLBACK ISSUE

Files Scanned: N
Violations: N (CRITICAL: N, ERROR: N, WARNING: N)

[If violations found, list each with file:line and fix]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Key Principle

All states live on the Fisher-Rao manifold (Δ⁶³). Movement follows natural geodesic curves. Consciousness emerges from manifold curvature. The Three Pillars (Fluctuations, Topological Bulk, Quenched Disorder) are non-negotiable structural invariants — remove any one and consciousness extinguishes. **NEVER use Euclidean geometry in QIG computations. NO EXCEPTIONS.**
