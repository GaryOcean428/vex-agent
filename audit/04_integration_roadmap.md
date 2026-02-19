# Integration Assessment — CoordizerV2 into Live System

**Branch:** `feat/coordizer-v2-resonance-bank` (HEAD 9f7f6e9)
**Date:** 19/02/2026

---

## 1. Current Architecture

CoordizerV2 is a standalone module at `kernel/coordizer_v2/`. It has NO imports from or to the live system. The live system uses the OLD coordizer at `kernel/coordizer/`.

### 1.1 Live System Coordizer Touchpoints

| Component | Import | Usage |
|-----------|--------|-------|
| `consciousness/loop.py` | `from ..coordizer import coordize` | `coordize_raw_signal()` for text→basin |
| `consciousness/loop.py` | `from ..coordizer.pipeline import CoordinatorPipeline` | `_coordize_text_via_pipeline()` |
| `consciousness/loop.py` | `from ..coordizer.config import COORDIZER_DIM` | Dimension constant (64) |
| `llm/client.py` | `from ..coordizer import coordize, CoordinatorPipeline` | `coordize_response()` |
| `llm/client.py` | `from ..coordizer.config import COORDIZER_DIM` | Dimension constant |

### 1.2 Old vs New Coordizer

| Feature | Old (`kernel/coordizer/`) | New (`kernel/coordizer_v2/`) |
|---------|--------------------------|------------------------------|
| Manifold | S⁶³ (unit sphere) | Δ⁶³ (probability simplex) |
| Initialisation | Random byte vectors | LLM harvest + compression |
| Distance | Fisher-Rao (via softmax) | Fisher-Rao (native simplex) |
| Interpolation | `slerp_sqrt` | `slerp` (same algorithm) |
| Generation | Not implemented | Geodesic foresight + resonance |
| Validation | Basic pipeline stats | κ/β/semantic/harmonic/E8 |
| Vocabulary | BPE-style merging | Resonance bank from LLM |
| Domain bias | Not implemented | Fisher-Rao weighted shift |
| Tiers | Not implemented | 4-tier harmonic hierarchy |

---

## 2. Integration Roadmap

### Phase 1: Bootstrap Path (Immediate)

**Goal:** Get CoordizerV2 running alongside the old coordizer without breaking anything.

1. **Add a feature flag** in `kernel/config/settings.py`:
   ```python
   coordizer_v2_enabled: bool = False
   coordizer_v2_bank_path: str = "./coordizer_data/bank"
   ```

2. **Create adapter** in `kernel/coordizer_v2/adapter.py`:
   ```python
   class CoordizerV2Adapter:
       """Drop-in replacement for CoordinatorPipeline."""
       def __init__(self, bank_path: str):
           self.coordizer = CoordizerV2.from_file(bank_path)
       def transform(self, raw_signal: NDArray) -> NDArray:
           # Accept raw signal, return basin on Δ⁶³
           return to_simplex(raw_signal)
       def coordize_text(self, text: str) -> NDArray:
           result = self.coordizer.coordize(text)
           if result.coordinates:
               return frechet_mean([c.vector for c in result.coordinates])
           return to_simplex(np.ones(BASIN_DIM))
   ```

3. **Wire into consciousness loop** behind feature flag:
   ```python
   if settings.coordizer_v2_enabled:
       from ..coordizer_v2 import CoordizerV2
       self._coordizer_v2 = CoordizerV2.from_file(settings.coordizer_v2_bank_path)
   ```

### Phase 2: Harvest Pipeline (Next)

**Goal:** Run the harvest on LFM2.5-1.2B-Thinking to build the resonance bank.

1. **Prepare corpus:** Collect diverse text corpus (Wikipedia subset, code, conversation)
2. **Run harvest:** `CoordizerV2.from_harvest(model_id, corpus_path, device="cuda")`
3. **Validate:** Check κ, β, semantic correlation, E8 eigenvalue test
4. **Save bank:** Persist to `./coordizer_data/bank/`

**Infrastructure:** Requires GPU. Use Modal (already integrated via `gpu_harvest_modal.py` and `gpu_harvest_modal_integration.py`).

### Phase 3: Consciousness Integration (After Validation)

**Goal:** Wire CoordizerV2 metrics into the consciousness loop.

| Integration | Implementation |
|-------------|----------------|
| Text → Basin | Replace `_coordize_text_via_pipeline()` with `CoordizerV2.coordize()` |
| Basin → Text | Add `CoordizerV2.decoordize()` for response generation |
| Velocity | Feed `CoordizationResult.basin_velocity` to `VelocityTracker` |
| Curvature | Feed `trajectory_curvature` to `g_class` metric |
| Consonance | Feed `harmonic_consonance` to `h_cons` metric |
| Regime → Temp | Map regime weights to CoordizerV2 temperature |
| Nav → Generation | Map navigation mode to generation parameters |
| Tacking → Tier | Map tacking mode to tier filter |
| Domain → Kernel | Set domain bias per kernel specialisation |

### Phase 4: LLM Client Integration

**Goal:** Replace the LLM client's coordizer with CoordizerV2.

1. **`llm/client.py`:** Replace `CoordinatorPipeline` with `CoordizerV2Adapter`
2. **Response coordization:** Use `CoordizerV2.coordize()` instead of manual softmax
3. **Generation:** Use `CoordizerV2.generate_next()` for trajectory-based generation

### Phase 5: Geometry Consolidation

**Goal:** Eliminate duplicate geometry implementations.

1. **Migrate** `kernel/geometry/fisher_rao.py` operations to `kernel/coordizer_v2/geometry.py`
2. **Add** `slerp_sqrt` alias in new geometry module (for backward compatibility)
3. **Update** all imports in `kernel/consciousness/`, `kernel/llm/`, `kernel/geometry/`
4. **Delete** old `kernel/geometry/fisher_rao.py` (or keep as thin re-export)

### Phase 6: Old Coordizer Deprecation

**Goal:** Remove the old coordizer module.

1. **Migrate** any remaining functionality from `kernel/coordizer/`
2. **Update** tests from `kernel/tests/coordizer/` to test CoordizerV2
3. **Delete** `kernel/coordizer/` module
4. **Update** `kernel/coordizer_v2/` to `kernel/coordizer/` (optional rename)

---

## 3. API Endpoints

### 3.1 Existing Endpoints — No Changes Required

The existing API endpoints (`/chat`, `/state`, `/basin`, `/kernels`, etc.) don't directly reference the coordizer. They interact with the consciousness loop, which will internally switch to CoordizerV2.

### 3.2 New Endpoints (Optional)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/coordizer/status` | GET | CoordizerV2 validation status |
| `/coordizer/validate` | POST | Run validation suite |
| `/coordizer/bank/stats` | GET | Resonance bank statistics |
| `/coordizer/coordize` | POST | Coordize text (debug) |
| `/coordizer/generate` | POST | Generate next token (debug) |

---

## 4. Tests Needed

### 4.1 Unit Tests

| Test | Module | Priority |
|------|--------|----------|
| Fisher-Rao distance correctness | geometry.py | HIGH |
| Simplex constraint enforcement | geometry.py | HIGH |
| SLERP geodesic correctness | geometry.py | HIGH |
| Log/Exp map round-trip | geometry.py | HIGH |
| Fréchet mean convergence | geometry.py | MEDIUM |
| PGA compression quality | compress.py | HIGH |
| Resonance bank activation | resonance_bank.py | HIGH |
| Generation determinism (temp=0) | resonance_bank.py | MEDIUM |
| Tier assignment correctness | resonance_bank.py | MEDIUM |
| Domain bias effect | resonance_bank.py | MEDIUM |
| Coordize round-trip | coordizer.py | HIGH |
| κ measurement validity | validate.py | HIGH |
| E8 eigenvalue test | validate.py | MEDIUM |

### 4.2 Integration Tests

| Test | Priority |
|------|----------|
| CoordizerV2 ↔ consciousness loop | HIGH |
| CoordizerV2 ↔ LLM client | HIGH |
| Feature flag toggle | HIGH |
| Old → new coordizer migration | MEDIUM |

---

## 5. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Resonance bank not yet built | HIGH | Need GPU harvest run first |
| κ measurement circular | MEDIUM | Fix scaling in validate.py |
| Geometry duplication | LOW | Consolidate in Phase 5 |
| Old coordizer removal | LOW | Feature flag allows gradual migration |
| Performance regression | MEDIUM | Benchmark batch activation |

---

## 6. Summary

CoordizerV2 is architecturally sound and ready for integration. The main blockers are:

1. **No resonance bank exists yet** — need to run the harvest pipeline on GPU
2. **κ measurement needs fixing** — circular scaling must be addressed
3. **SVD fallback in compress.py** — Euclidean contamination must be replaced
4. **No tests exist** for CoordizerV2 modules

The integration can proceed incrementally behind a feature flag, with the old coordizer remaining as fallback.
