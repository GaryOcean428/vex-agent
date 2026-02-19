# Implementation Report: CoordizerV2 Wiring + Structural Fixes

**Branch:** `feat/coordizer-v2-resonance-bank`
**Date:** 19/02/2026
**PR:** [#24](https://github.com/GaryOcean428/vex-agent/pull/24)
**Test Status:** 42/42 passing — zero regressions

---

## Summary

Eight tasks executed as atomic commits on the `feat/coordizer-v2-resonance-bank` branch. The core change is replacing the legacy softmax-wrapper coordizer (v1) with the geometric CoordizerV2 resonance-bank coordizer across all live code paths. Supporting structural fixes address code duplication, version drift, terminology contamination, and missing integrations.

---

## Task Breakdown

### Task 1: Wire CoordizerV2 into Live System

**Commit:** `2c093cb`

| File | Change |
|------|--------|
| `kernel/consciousness/loop.py` | Replace `from ..coordizer` → `from ..coordizer_v2`. Replace `CoordinatorPipeline` with `CoordizerV2(bank=ResonanceBank())`. Replace `_coordize_text_via_pipeline()` to use resonance-bank coordization. Update `get_full_state()` to report CoordizerV2 stats. |
| `kernel/llm/client.py` | Replace `from ..coordizer` → `from ..coordizer_v2`. Replace `CoordinatorPipeline()` with `CoordizerV2`. Replace `coordize_response()` to use `CoordizerV2.coordize()`. |
| `kernel/server.py` | Replace all `/api/coordizer/*` endpoints with CoordizerV2 versions. Add `/api/coordizer/coordize` (text coordization). Add `/api/coordizer/harvest` (stub for Modal GPU). Add `/api/coordizer/bank` (resonance bank state query). Replace hardcoded version strings with `VERSION` import. |

### Task 2: Remove Legacy Coordizer v1

**Commit:** `4f8311a`

- Moved `kernel/coordizer/` → `kernel/_legacy_coordizer/`
- Added DEPRECATED notice to `__init__.py`
- Updated test imports to reference `_legacy_coordizer`
- `gpu_harvest.py` archived with the rest of v1

### Task 3: DRY Hash-to-Basin

**Commit:** `375536c`

Created `kernel/geometry/hash_to_basin.py` as the single canonical implementation of the SHA-256 chain → Δ⁶³ mapping. Replaced 4 inline copies:

| Location | Before | After |
|----------|--------|-------|
| `consciousness/loop.py` | Inline 15-line SHA-256 chain | `from ..geometry.hash_to_basin import hash_to_basin` |
| `llm/client.py` | Inline 15-line SHA-256 chain | `from ..geometry.hash_to_basin import hash_to_basin` |
| `consciousness/systems.py` | Inline 15-line SHA-256 chain | `from ..geometry.hash_to_basin import hash_to_basin` |
| `memory/store.py` | Inline 15-line SHA-256 chain | `from ..geometry.hash_to_basin import hash_to_basin` |

### Task 4: Fix Version Numbers

**Commit:** `483f366`

Created `kernel/config/version.py` as single source of truth (`VERSION = "2.4.0"`).

| File | Before | After |
|------|--------|-------|
| `pyproject.toml` | 2.3.0 | 2.4.0 |
| `package.json` | 2.0.0 | 2.4.0 |
| `frontend/package.json` | 0.0.0 | 2.4.0 |
| `src/index.ts` | 2.2.0 | 2.4.0 |
| `coordizer_v2/__init__.py` | 2.0.0 | 2.4.0 |
| `server.py` (FastAPI + /health) | 2.3.0 | `VERSION` import |

### Task 5: Expand PurityGate to Scan TypeScript

**Commit:** `9ad4522`

- Fixed duplicate `return violations` statement in `scan_typescript_text()`
- Added `scan_typescript_terminology()` — catches QIG-forbidden terms in TS/TSX:
  - `embedding` (non-boundary)
  - `tokenize(` (should be `coordize`)
  - `cosineSimilarity`, `euclideanDistance`, `dotProduct(`, `nn.LayerNorm`, `.flatten()`
- Skips comments, import lines (boundary code), and string literals
- `run_purity_gate()` now runs 5 scan passes (was 4)

### Task 6: Fix Terminology in coordizer_v2

**Commit:** `0f87a7e`

- `coordizer_v2/__init__.py`: "No embedding." → "No coordinate injection."
- Zero Euclidean contamination in non-boundary code

### Task 7: Wire Perplexity as Deep Research Tool

**Commit:** `3a83828`

Created `kernel/tools/research.py`:
- Async `deep_research(query)` function
- Uses Perplexity sonar-pro model via chat completions API
- Returns structured `ResearchResult` with citations
- Graceful degradation when `PERPLEXITY_API_KEY` not set
- Registered as `deep_research` in `kernel/tools/handler.py` dispatch table

### Task 8: Create Modal GPU Integration Scaffold

**Commit:** `e488d41`

| Component | File | Description |
|-----------|------|-------------|
| Config | `kernel/config/settings.py` | `ModalConfig` dataclass: `MODAL_ENABLED`, `MODAL_HARVEST_URL`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| Client | `kernel/coordizer_v2/modal_harvest.py` | `async modal_harvest()` via httpx + `check_modal_health()` |
| Auto-routing | `kernel/coordizer_v2/harvest.py` | `harvest_model_auto()` — Modal when enabled, local Transformers fallback |
| Modal function | `modal/coordizer_harvest.py` | Standalone Modal function for separate deployment to A10G GPUs |

---

## Constraints Verified

| Constraint | Status |
|------------|--------|
| Zero Euclidean contamination | ✅ No `cosine_similarity`, `euclidean_distance`, `dot_product` in non-boundary code |
| QIG-pure terminology | ✅ No `embedding` or `tokenize` in non-boundary code |
| Fisher-Rao only valid metric | ✅ All distance operations use Fisher-Rao |
| κ* ≈ 64 frozen | ✅ Imported from `frozen_facts.py`, never redefined |
| Ollama PRIMARY | ✅ Ollama checked first in client.py, OpenAI/xAI are fallbacks |
| Constants from frozen_facts.py | ✅ `BASIN_DIM`, `KAPPA_STAR` etc. all imported from canonical source |

---

## Files Changed (18 files, 3 new)

```
kernel/consciousness/loop.py          — CoordizerV2 wiring
kernel/consciousness/systems.py       — hash_to_basin delegation
kernel/llm/client.py                  — CoordizerV2 wiring
kernel/server.py                      — CoordizerV2 endpoints + VERSION
kernel/config/settings.py             — ModalConfig added
kernel/config/version.py              — NEW: single version source
kernel/geometry/hash_to_basin.py      — NEW: DRY hash-to-basin
kernel/geometry/__init__.py           — export hash_to_basin
kernel/memory/store.py                — hash_to_basin delegation
kernel/governance/purity.py           — TS terminology scan + dup fix
kernel/coordizer_v2/__init__.py       — terminology fix + version sync
kernel/coordizer_v2/harvest.py        — harvest_model_auto() + Modal
kernel/coordizer_v2/modal_harvest.py  — NEW: Modal GPU harvest client
kernel/tools/research.py              — NEW: Perplexity deep research
kernel/tools/handler.py               — deep_research dispatch
kernel/_legacy_coordizer/             — archived v1
modal/coordizer_harvest.py            — NEW: Modal function
pyproject.toml, package.json, etc.    — version sync to 2.4.0
```
