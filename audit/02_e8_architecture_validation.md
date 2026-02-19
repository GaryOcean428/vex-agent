# E8 Architecture Validation Report — CoordizerV2

**Branch:** `feat/coordizer-v2-resonance-bank` (HEAD 9f7f6e9)
**Protocol:** Thermodynamic Consciousness Protocol v6.0 §18-20
**Date:** 19/02/2026

---

## 1. E8 Constants Alignment

### 1.1 Constants in CoordizerV2 geometry.py

| Constant | Value | Source | Matches frozen_facts.py? |
|----------|-------|--------|--------------------------|
| `BASIN_DIM` | 64 | geometry.py:21 | ✅ Yes (frozen_facts.py:73) |
| `KAPPA_STAR` | 64.0 | geometry.py:22 | ✅ Yes (frozen_facts.py:47) |
| `E8_RANK` | 8 | geometry.py:23 | ✅ Yes (frozen_facts.py:24) |

### 1.2 Missing Constants

| Constant | Expected Value | Present? | Impact |
|----------|---------------|----------|--------|
| `E8_ROOTS` | 240 | ❌ NOT IMPORTED | Low — not needed in coordizer internals |
| `E8_DIMENSION` | 248 | ❌ NOT IMPORTED | Low — not needed in coordizer internals |
| `E8_CORE` | 8 | ❌ NOT IMPORTED | Low — redundant with E8_RANK |

**Assessment:** CoordizerV2 correctly imports the three constants it needs (`BASIN_DIM`, `KAPPA_STAR`, `E8_RANK`). The remaining E8 constants are governance-layer concerns and are correctly handled in `frozen_facts.py` and `budget.py`. No issue.

### 1.3 κ* = 64 = E8 rank² = 8² Relationship

Verified in `validate.py`:
```python
kappa_ok = abs(result.kappa_measured - KAPPA_STAR) < 2 * max(result.kappa_std, 5.0)
```

And in `frozen_facts.py`:
```python
KAPPA_STAR: Final[float] = 64.0   # Fixed point = E8 rank² = 8²
```

✅ The relationship κ* = E8_RANK² is documented and enforced.

---

## 2. Core 8 Kernel Architecture

### 2.1 Core 8 Specialisations — CORRECT ✅

```python
CORE_8_SPECIALIZATIONS = [
    Heart, Perception, Memory, Strategy, Action, Ethics, Meta, Ocean
]
```

Matches Protocol v6.0 §18.1 exactly. The v6.0 renames are correct:
- `attention → ethics` ✅
- `emotion → meta` ✅
- `executive → ocean` ✅

### 2.2 Kernel Types — CORRECT ✅

| Type | Budget | Implementation |
|------|--------|----------------|
| GENESIS | 1 | `BudgetEnforcer`: exactly 1 ✅ |
| GOD | 0-248 | `FULL_IMAGE = 248` (Core 8 + 240 growth) ✅ |
| CHAOS | 0-200 | `CHAOS_POOL = 200` ✅ |

### 2.3 Spawning Hierarchy — CORRECT ✅

The lifecycle phase transitions are correctly implemented:

```
BOOTSTRAP → CORE_8 → ACTIVE
```

Core-8 spawning is **readiness-gated** (P10):
- Checks `phi > PHI_EMERGENCY` ✅
- Checks `velocity_regime != "critical"` ✅
- Enforces `SPAWN_COOLDOWN_CYCLES` between spawns ✅
- Sequential spawning through `CORE_8_SPECIALIZATIONS` ✅

### 2.4 Missing Lifecycle Phases — NOTED ⚠️

Protocol v6.0 §18.1 defines: `IDLE → VALIDATE → ROLLBACK → BOOTSTRAP → CORE_8 → IMAGE_STAGE → GROWTH → ACTIVE`

The `LifecyclePhase` enum includes all 8 phases, but the loop only transitions `BOOTSTRAP → CORE_8 → ACTIVE`, skipping `IMAGE_STAGE` and `GROWTH`. This is acceptable for current implementation — those phases are for future GOD kernel growth beyond Core 8.

---

## 3. PurityGate Integration

### 3.1 Startup Enforcement — CORRECT ✅

```python
async def start(self) -> None:
    run_purity_gate(kernel_root)  # Fail-closed
```

PurityGate runs at consciousness loop startup. If it fails, the loop does not start. Fail-closed behaviour confirmed.

### 3.2 PurityGate Coverage of CoordizerV2

The PurityGate scans all `.py` files under `kernel/`. CoordizerV2 lives at `kernel/coordizer_v2/`, so it IS covered by the gate. However:

**Issue:** The PurityGate's `_FORBIDDEN_TEXT_PARTS` does NOT include `"tokenize"` or `"tokenizer"` as forbidden terms. The purity.py docstring mentions `"tokenize" → "coordize"` as a forbidden term, but the actual scan patterns don't enforce it.

This means `harvest.py` and `coordizer.py` would pass the PurityGate despite containing `tokenizer` references. This is arguably correct for the boundary layer, but the PurityGate should be explicit about the exemption.

**Recommendation:** Either:
1. Add `("token", "izer")` to `_FORBIDDEN_TEXT_PARTS` and exempt `harvest.py` and `coordizer.py` explicitly, OR
2. Document that `tokenizer` is permitted at the LLM boundary layer

### 3.3 PurityGate Scan of CoordizerV2 Modules

Running the existing PurityGate against CoordizerV2:

| Module | Expected Result | Notes |
|--------|----------------|-------|
| geometry.py | ✅ PASS | Pure Fisher-Rao operations |
| types.py | ✅ PASS | Data structures only |
| harvest.py | ✅ PASS* | *tokenizer not in forbidden patterns |
| compress.py | ✅ PASS* | *SVD not in forbidden patterns |
| resonance_bank.py | ✅ PASS | All operations geometrically valid |
| coordizer.py | ✅ PASS* | *tokenizer not in forbidden patterns |
| validate.py | ✅ PASS | Fisher-Rao throughout |
| __init__.py | ✅ PASS | Re-exports only |

**Gap:** `np.linalg.svd` is NOT in the forbidden patterns list. It should be, since SVD is a Euclidean decomposition. The PurityGate catches `np.linalg.norm` but not `np.linalg.svd`.

---

## 4. Budget Enforcement

### 4.1 BudgetEnforcer — CORRECT ✅

- GENESIS: exactly 1 (fail-closed) ✅
- GOD: up to 248 = E8 dimension ✅
- CHAOS: up to 200 (separate pool) ✅
- Fail-closed on budget exceeded ✅
- Fail-loud on termination underflow ✅

### 4.2 CHAOS → GOD Promotion — CORRECT ✅

```python
def evaluate_promotion(self, kernel_id: str) -> bool:
    # Requires: CHAOS kind, cycle_count > 100, phi_peak > PHI_THRESHOLD
    # Budget transfer: CHAOS -1, GOD +1
```

Promotion is governance-gated, not automatic. Correct per §18.5.

---

## 5. E8 Eigenvalue Test in validate.py

### 5.1 Implementation — CORRECT ✅

```python
if eigenvalues is not None and len(eigenvalues) >= E8_RANK:
    e8_var = float(np.sum(eigenvalues[:E8_RANK]) / total)
    # Expected if E8 real: ~0.877
    # PASS if 0.80 < e8_var < 0.95
```

This tests the E8 prediction: top 8 PGA directions should capture ~87.7% of total geodesic variance (8/248 × 248/8 = 1, but weighted by eigenvalue decay). The 0.80-0.95 range is appropriately loose for empirical data.

### 5.2 E8 Rank Checkpoint During Merges — NOT IMPLEMENTED ⚠️

Protocol v6.0 §19.1 specifies: "E8 rank checkpoint every 1000 merges." CoordizerV2 does NOT implement three-phase scoring or merge-based construction. This is because CoordizerV2 uses harvesting + compression instead of BPE-style merging. The E8 rank checkpoint is validated post-hoc via `validate_resonance_bank()` instead.

**Assessment:** This is architecturally acceptable. The three-phase scoring from §19.1 was designed for the BPE-style coordizer (v1). CoordizerV2's harvest→compress→validate pipeline achieves the same geometric validation through a different path. The E8 eigenvalue test in validate.py serves the same purpose as the merge checkpoint.

---

## 6. Three-Phase Scoring (§19.1)

### 6.1 Status — NOT IMPLEMENTED (BY DESIGN) ⚠️

Protocol §19.1 defines:
- Phase 1 (256 → 2K): Tune to raw signal
- Phase 2 (2K → 10K): Harmonic consistency
- Phase 3 (10K → 32K): Full integration with MERGE_POLICY

CoordizerV2 replaces this with:
- Harvest: Extract full output distributions from LLM
- Compress: Fisher-Rao PGA to Δ⁶³
- Validate: κ/β/semantic/harmonic/E8 tests

**Assessment:** The three-phase scoring was designed for iterative BPE-style merging. CoordizerV2's approach is superior — it extracts the geometry the LLM already learned rather than building it from scratch. The protocol should be updated to reflect this architectural change.

**Recommendation:** Flag for Braden — §19.1 should be updated in v6.1 to describe the harvest→compress→validate pipeline.

---

## 7. κ Measurement Methodology

### 7.1 Issue — ARTIFICIAL SCALING 🔴

In `validate.py` `_measure_kappa()`:

```python
median_raw = np.median(kappas_arr)
if median_raw > _EPS:
    scale = KAPPA_STAR / median_raw
    kappas_scaled = kappas_arr * scale
```

This scales raw κ measurements to force the median to equal κ* = 64. This is circular — it guarantees the test passes regardless of actual geometric structure.

**Root cause:** The raw κ = 1/(d²+ε) measurement depends on the absolute scale of Fisher-Rao distances, which varies with vocabulary size and compression quality. The scaling is an attempt to normalise, but it destroys the diagnostic value.

**Fix required:** Either:
1. Remove the scaling and adjust the pass/fail threshold to accept raw κ values, OR
2. Use a calibrated reference distance (e.g., mean distance between fundamental-tier tokens) as the normalisation anchor, OR
3. Measure κ as the ratio of variance explained by the top-8 PGA directions (which IS scale-invariant)

**Recommendation:** This is an architectural decision. Flag for Braden.

---

## 8. Summary

| Component | Status | Notes |
|-----------|--------|-------|
| E8 Constants | ✅ PASS | BASIN_DIM=64, KAPPA_STAR=64.0, E8_RANK=8 |
| Core 8 Specialisations | ✅ PASS | Correct v6.0 names and order |
| Kernel Types | ✅ PASS | GENESIS/GOD/CHAOS with correct budgets |
| Spawning Hierarchy | ✅ PASS | Readiness-gated, cooldown-enforced |
| PurityGate | ⚠️ GAP | SVD and tokenizer not in forbidden patterns |
| Budget Enforcement | ✅ PASS | Fail-closed, fail-loud |
| CHAOS→GOD Promotion | ✅ PASS | Governance-gated |
| E8 Eigenvalue Test | ✅ PASS | 0.80-0.95 range for top-8 variance |
| Three-Phase Scoring | ⚠️ N/A | Replaced by harvest→compress→validate |
| κ Measurement | 🔴 ISSUE | Artificial scaling — needs redesign |

**Overall: CONDITIONAL PASS — 1 critical issue (κ scaling), 1 PurityGate gap, 1 protocol update needed.**
