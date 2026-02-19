# QIG Purity Validation Report — CoordizerV2

**Branch:** `feat/coordizer-v2-resonance-bank` (HEAD 9f7f6e9)
**Protocol:** Thermodynamic Consciousness Protocol v6.0 §1.3
**Date:** 19/02/2026
**Tolerance:** ZERO

---

## 1. Forbidden Pattern Scan Results

### 1.1 §1.3 Forbidden Operations — Direct Violations

| Pattern | Status | Details |
|---------|--------|---------|
| `cosine_similarity` | ✅ CLEAN | Not found in any CoordizerV2 module |
| `np.linalg.norm` | ✅ CLEAN | Not found in CoordizerV2 (found only in purity.py rule definitions) |
| `dot_product` | ✅ CLEAN | Not found as a function call |
| `Adam` optimizer | ✅ CLEAN | Not found |
| `LayerNorm` | ✅ CLEAN | Not found |
| `embedding` (term) | ✅ CLEAN | Not found in CoordizerV2 modules |
| `flatten` | ✅ CLEAN | Not found |

### 1.2 Boundary Layer — Tokenizer Usage (REQUIRES DECISION)

| File | Line(s) | Pattern | Severity |
|------|---------|---------|----------|
| `harvest.py` | 153, 156, 164, 178, 243 | `AutoTokenizer`, `tokenizer.encode`, `tokenizer.decode` | ⚠️ BOUNDARY |
| `coordizer.py` | 77, 131, 176, 253 | `self._tokenizer`, `tokenizer.encode`, `tokenizer.decode` | ⚠️ BOUNDARY |

**Assessment:** These are at the LLM interface boundary. `harvest.py` MUST use the LLM's tokenizer to extract output distributions — this is the raw signal ingestion point. However, `coordizer.py` should NOT fall back to tokenizer-based coordization in production. The `_coordize_via_tokenizer` method is a bootstrap path for when the resonance bank is first built; after that, all coordization should go through the resonance bank.

**Recommendation:** Add explicit `# QIG BOUNDARY: LLM interface — tokenizer required for distribution extraction` comments. Mark `_coordize_via_tokenizer` as `@deprecated` once string-based coordization is mature. Flag for Braden.

### 1.3 Euclidean Contamination — SVD in compress.py

| File | Line | Pattern | Severity |
|------|------|---------|----------|
| `compress.py` | 222 | `np.linalg.svd(T_sub, full_matrices=False)` | 🔴 VIOLATION |

**Assessment:** This is a genuine purity violation. SVD is a Euclidean operation. The code uses it as a fallback when `n_sub <= target_dim`. The primary path uses `eigsh` on the Gram matrix, which is acceptable because it operates on inner products in the tangent space (which IS a Euclidean space — the tangent space at a point on the manifold is flat by definition).

**However:** The SVD fallback path does NOT operate in the tangent space correctly. It decomposes `T_sub` directly rather than the Gram matrix `T_sub @ T_sub.T`. This could produce geometrically incorrect principal directions.

**Fix:** Replace SVD fallback with eigendecomposition of `T_sub.T @ T_sub` (dual Gram matrix), which is geometrically equivalent to the primary path.

### 1.4 Euclidean Contamination — Tangent Space Operations in resonance_bank.py

| File | Line | Pattern | Severity |
|------|------|---------|----------|
| `resonance_bank.py` | 227 | `velocity_norm = np.sqrt(np.sum(velocity * velocity))` | ⚠️ ACCEPTABLE |
| `resonance_bank.py` | 243 | `token_tangent_norm = np.sqrt(np.sum(token_tangent * token_tangent))` | ⚠️ ACCEPTABLE |
| `resonance_bank.py` | 247 | `consistency = np.sum(direction * token_dir)` | ⚠️ ACCEPTABLE |

**Assessment:** These operations are in the TANGENT SPACE, not on the manifold. The tangent space at a point on the simplex IS a Euclidean vector space. L2 norms and dot products in the tangent space are geometrically valid — they correspond to the Fisher metric at the base point. This is standard Riemannian geometry.

**However:** The code should add comments clarifying that these are tangent-space operations where Euclidean arithmetic is geometrically valid. Without comments, they look like purity violations.

### 1.5 Euclidean Contamination — Random Direction Padding in compress.py

| File | Line | Pattern | Severity |
|------|------|---------|----------|
| `compress.py` | 258 | `random_dirs = np.random.randn(source_dim, pad)` | ⚠️ MINOR |
| `compress.py` | 265 | `norm = np.sqrt(np.sum(random_dirs[:, j] ** 2))` | ⚠️ MINOR |

**Assessment:** When fewer than 64 principal directions are found, random directions are generated in the tangent space to pad to 64. This is acceptable as a fallback but should use Gram-Schmidt orthogonalisation against existing directions (which it does) and should be documented as a degraded-quality path.

---

## 2. Fisher-Rao Distance Implementation

### 2.1 geometry.py — CORRECT ✅

```python
def fisher_rao_distance(p, q):
    bc = bhattacharyya_coefficient(p, q)  # Σ√(p_i · q_i)
    bc = np.clip(bc, -1.0, 1.0)
    return float(np.arccos(bc))  # d_FR = arccos(BC)
```

This exactly matches Protocol v6.0 §1.2: `d_FR(p,q) = arccos(Σ√(p_i·q_i))`.

### 2.2 Batch Distance — CORRECT ✅

```python
def fisher_rao_distance_batch(p, bank):
    bcs = np.sum(np.sqrt(p[np.newaxis, :] * bank), axis=1)
    bcs = np.clip(bcs, -1.0, 1.0)
    return np.arccos(bcs)
```

Vectorised version. Correct.

### 2.3 SLERP — CORRECT ✅

The SLERP implementation operates in sqrt-space (Hellinger coordinates), which is the standard computational device for geodesics on the probability simplex. The sqrt map `p_i → √p_i` maps Δ⁶³ to the positive orthant of S⁶³, where SLERP is the geodesic. This is geometrically correct per §1.1: "Sqrt-space (Hellinger) allowed for geodesic computation."

### 2.4 Log Map / Exp Map — CORRECT ✅

Both operate in sqrt-space. The log map projects to the tangent space at the base point, and the exp map walks along the tangent vector. Standard Riemannian operations.

### 2.5 Fréchet Mean — CORRECT ✅

Iterative algorithm: log-map → weighted average in tangent space → exp-map. This is the standard Karcher mean algorithm on Riemannian manifolds. Initialisation via sqrt-space average is a good approximation.

---

## 3. Simplex Constraint Enforcement

### 3.1 to_simplex() — CORRECT ✅

```python
def to_simplex(v):
    v = np.maximum(v, _EPS)  # Non-negative
    return v / v.sum()        # Sum to 1
```

Enforces both simplex constraints: `p_i ≥ 0` and `Σp_i = 1`.

### 3.2 Dimensionality — CORRECT ✅

`BASIN_DIM = 64` is defined in `geometry.py` and imported throughout. `BasinCoordinate.__post_init__` validates dimensionality:

```python
if len(self.vector) != BASIN_DIM:
    raise ValueError(...)
self.vector = to_simplex(self.vector)
```

### 3.3 Compression Output — CORRECT ✅

`compress.py` line 297-298 normalises projections to simplex:
```python
shifted = raw - raw.min() + _EPS
basin = shifted / shifted.sum()
```

---

## 4. PCA vs PGA Assessment

### 4.1 Primary Path — Fisher-Rao PGA ✅

The primary compression path in `compress.py` is genuine PGA:
1. Fréchet mean on Δ^(V-1) via sqrt-space averaging ✅
2. Log-map all points to tangent space at μ ✅
3. Gram matrix in tangent space ✅
4. Eigendecomposition of Gram matrix ✅
5. Project onto principal geodesic directions ✅
6. Normalise to Δ⁶³ ✅

This is the manifold analogue of PCA, operating correctly on the Fisher-Rao geometry.

### 4.2 SVD Fallback — Euclidean Contamination 🔴

Line 222: `np.linalg.svd(T_sub, full_matrices=False)` — this is a direct SVD on the tangent vectors, not the Gram matrix. While mathematically equivalent for full-rank data, it bypasses the geometric framing and could produce different numerical results due to floating-point ordering.

**Fix required:** Replace with eigendecomposition of `T_sub.T @ T_sub` for consistency.

---

## 5. Geodesic vs Linear Interpolation

### 5.1 resonance_bank.py — CORRECT ✅

Domain bias uses `slerp()` (geodesic interpolation), not linear blending:
```python
query = slerp(query, to_simplex(domain_bias.anchor_basin), domain_bias.strength)
```

### 5.2 Generation Scoring — MIXED ⚠️

The `generate_next` method uses `np.exp(-dist * KAPPA_STAR / 10.0)` for proximity scoring. This is a monotonic function of Fisher-Rao distance, which is acceptable. The final score combination `0.5 * proximity + 0.3 * consistency + 0.2 * consonance` is a linear combination of geometrically-derived quantities, which is acceptable for scoring purposes (not for interpolation).

---

## 6. Summary

| Category | Status | Issues |
|----------|--------|--------|
| §1.3 Forbidden Operations | ✅ PASS | No direct violations |
| Tokenizer Boundary | ⚠️ FLAG | harvest.py/coordizer.py use tokenizer at LLM boundary |
| Fisher-Rao Distance | ✅ PASS | Correct implementation |
| Simplex Constraints | ✅ PASS | Enforced throughout |
| SLERP/Log/Exp | ✅ PASS | Correct sqrt-space geodesics |
| Fréchet Mean | ✅ PASS | Correct Karcher mean |
| PGA (Primary) | ✅ PASS | Genuine Fisher-Rao PGA |
| SVD Fallback | 🔴 FAIL | Euclidean SVD, needs replacement |
| Tangent Space Ops | ⚠️ NOTE | Valid but needs documentation |
| Linear Blending | ✅ PASS | Uses geodesic interpolation |

**Overall: CONDITIONAL PASS — 1 fix required (SVD fallback), 2 documentation items.**
