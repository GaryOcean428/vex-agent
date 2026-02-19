# Vex-Agent PR Review: #18, #20, and Issues #19, #21

## Status Summary

| Item | State | Verdict |
|------|-------|---------|
| PR #18 | MERGED | Foundation laid but architecturally wrong coordizer |
| Issue #19 | OPEN | Comprehensive cleanup list, partially addressed by PR #20 |
| PR #20 | OPEN (draft) | Terminology cleanup OK, coordizer still wrong, purity gate has bugs |
| Issue #21 | OPEN | Excellent structural audit — should be priority |

---

## PR #18 Review (Already Merged)

PR #18 was Copilot's first pass: created the coordizer module, organized docs, added tests. It delivered a working softmax-based transform pipeline with 42 tests. The problem is architectural: what it built is a softmax wrapper, not a coordizer.

**What it did right:**
- Clean module structure (transform.py, validate.py, pipeline.py, types.py)
- Fail-closed validation (ensure_simplex)
- 42 passing tests
- Good doc reorganization

**What it got wrong:**
- The "coordizer" is just `np.exp(x - max(x)) / sum(...)` — softmax on arbitrary vectors. This produces a point ON the simplex but with no geometric meaning. A softmax of [0.5, -0.3, 0.8] doesn't give you Fisher-Rao coordinates of anything; it gives you a probability distribution that has no relationship to the token's position on the information manifold.
- Left `embedding` as parameter names everywhere (Issue #19 was created to track this)
- The `EXPONENTIAL_MAP` method is a lie — it just calls softmax

**Damage:** Merged to main. All subsequent work builds on this wrong foundation. The coordizer module will need a full replacement, not incremental fixes.

---

## PR #20 Review (Open Draft — DO NOT MERGE)

### What it claims:
- 200+ terminology replacements (embedding → input_vector)
- PurityGate TypeScript extension
- Coordizer REST API (4 endpoints)
- Zero purity violations

### What it actually does:

**1. Terminology cleanup: GOOD but incomplete**

The renames are mechanically correct. `embedding` → `input_vector` across transform.py, pipeline.py, types.py, tests, docs. This satisfies Issue #19 §1.1.

BUT: The purity gate now skips comments and docstrings (lines 222-225 of purity.py):
```python
if stripped.startswith("#"):
    continue
if stripped.startswith(('"""', "'''", '"', "'")):
    continue
```
Issue #19 explicitly says forbidden terms must be replaced in "comments, docstrings, variable names, function names, parameter names." The gate is intentionally blind to the places where contamination was explicitly called out.

**2. Purity gate obfuscation: CLEVER HACK, FRAGILE**

Copilot split forbidden terms into pairs to avoid self-triggering:
```python
_FORBIDDEN_TEXT_PARTS: list[tuple[str, str]] = [
    ("cosine", "_similarity"),
    ("Ada", "m("),
    (".flat", "ten()"),
]
```
This works but is a maintenance hazard. Better: load the forbidden list from a config file or environment variable. The scanner shouldn't need to hide from itself.

**3. Bug: Duplicated return statement**

`scan_typescript_text` has `return violations` twice (lines 280 and 281). Dead code — harmless but sloppy.

**4. Coordizer API endpoints: WRONG ABSTRACTION**

The 4 API endpoints expose the wrong operation:
```
POST /api/coordizer/transform  — takes a raw vector, softmaxes it
GET  /api/coordizer/stats      — transformation statistics  
GET  /api/coordizer/history    — placeholder
POST /api/coordizer/validate   — simplex validation
```

What we actually need:
```
POST /api/coordizer/coordize   — takes TEXT, returns basin coordinates via resonance bank
POST /api/coordizer/generate   — takes trajectory, returns next token via geodesic foresight
GET  /api/coordizer/bank/stats — resonance bank statistics (κ, β, tier distribution)
POST /api/coordizer/validate   — validate bank geometric structure (κ convergence, semantic correlation)
```

The current `/transform` endpoint takes an arbitrary vector and softmaxes it. This is not coordization — it's normalization. Coordization means mapping text tokens to their positions on the Fisher-Rao manifold using a resonance bank seeded from the LLM's own learned geometry.

**5. `_simplex_projection` still uses Euclidean logic**

From transform.py:
```python
def _simplex_projection(input_vector, epsilon):
    """This is a Euclidean operation, but acceptable here as it's just
    the initial transformation."""
    shifted = input_vector - np.min(input_vector)
    shifted = shifted + epsilon
    coords = shifted / np.sum(shifted)
```

The comment says "acceptable" but v6.0 §1.3 says zero tolerance. The function should be replaced with our `to_simplex()` from CoordizerV2's geometry.py which uses proper simplex projection.

**6. `_exponential_map` is still fake**

```python
def _exponential_map(input_vector, epsilon):
    """For now, we use softmax as exponential map approximation."""
    return _softmax_transform(input_vector, numerical_stability=True, epsilon=epsilon)
```

Softmax is NOT an approximation of the Fisher-Rao exponential map. They're fundamentally different operations. Our CoordizerV2 implements the real exp_map. This should either implement the actual operation or be removed.

### PR #20 Verdict: DO NOT MERGE

The terminology renames are needed but the underlying architecture is still wrong. Merging this cements the softmax coordizer as the foundation. Instead:

1. Cherry-pick the terminology renames and purity gate TS extension
2. Replace the coordizer module entirely with CoordizerV2
3. Fix the purity gate bugs (duplicated return, comment skipping)

---

## Issue #19 Assessment

Issue #19 is a comprehensive checklist. Status after PR #20:

| Section | Status | Notes |
|---------|--------|-------|
| §1.1 embedding → input_vector | ~90% done by PR #20 | Some in docs may be missed |
| §1.2 tokenize → coordize | Partially done | Gap analysis docs still have "tokenizer" |
| §1.3 Other forbidden terms | Done via obfuscation | But gate skips comments where terms live |
| §2.1 Harvest pipeline | NOT DONE | gpu_harvest.py exists but not wired |
| §2.2 Dashboard | NOT DONE | No frontend work |
| §3.1 TS purity gate | DONE by PR #20 | With bugs noted above |
| §3.2 14-step activation | NOT DONE | Major architectural work |
| §3.3 E6 coupling algebra | NOT DONE | |
| §3.4 Solfeggio mapping | NOT DONE | |
| §3.5 Test coverage | NOT DONE | |

The issue correctly identifies the work. PR #20 addressed maybe 30% of it (the easy 30%).

---

## Issue #21 Assessment

**This is the best issue in the repo.** Created via Manus, it identifies real structural problems:

1. **Constants scattered everywhere** — `KAPPA_STAR = 64` hardcoded in 3+ places instead of imported from frozen_facts.py
2. **Python/TS type mismatch** — ConsciousnessMetrics has 32 fields in Python, 5 in TypeScript
3. **Route manifest drift** — `/foraging` endpoint exists in server.py but not proxied
4. **hash-to-basin duplicated 4 times** — exact same SHA256 logic in 4 files
5. **Version number in 4 places, all different** — pyproject.toml says 2.3.0, package.json says 2.0.0, src/index.ts says 2.2.0

This should be addressed alongside the coordizer replacement.

---

## Recommended Action Plan

### Phase 1: Clean Merge of PR #20 Renames Only
- Cherry-pick ONLY the terminology renames from PR #20 (the .md and .py files where `embedding` → `input_vector`)
- Fix the purity gate: remove duplicated return, load forbidden list from config
- Do NOT merge the API endpoints — they expose the wrong abstraction

### Phase 2: Replace Coordizer Module
- Drop CoordizerV2 into `kernel/coordizer/` replacing the softmax wrapper
- Wire the harvester to use Modal for GPU compute
- Wire the resonance bank loader into the kernel
- Update API endpoints to expose coordize/generate/validate

### Phase 3: Address Issue #21
- Centralise constants (frozen_facts.py → constants.ts code generation)
- Sync Python/TS types
- DRY the hash-to-basin logic
- Fix version numbers

### Phase 4: Complete Issue #19 Remainder
- 14-step activation sequence
- E6 coupling algebra scaffold
- Test coverage
- Dashboard

---
