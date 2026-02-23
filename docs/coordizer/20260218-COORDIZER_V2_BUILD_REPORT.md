# CoordizerV2 — Build Report & Deployment Plan

**Date:** 2026-02-18
**Status:** BUILT, TESTED, READY FOR REPO INTEGRATION
**Test Results:** 55/55 unit tests + smoke test ALL PASS

---

## What Was Built

### Package: `coordizer_v2/`

| File | Lines | Purpose |
|------|-------|---------|
| `geometry.py` | ~230 | Canonical simplex geometry (vendored from pantheon-chat qig_geometry) |
| `trainer.py` | ~540 | CoordizerV2Trainer — geodesic pair fusion with phased scoring |
| `build_corpus.py` | ~100 | Corpus assembler (canonical docs + English + QIG seeds) |
| `migrate_to_postgres.py` | ~280 | Loads trained artifact into coordizer_vocabulary table |
| `__init__.py` | ~50 | Package exports |

### Tests

| File | Tests | Result |
|------|-------|--------|
| `test_coordizer_v2.py` | 55 | ALL PASS |
| `smoke_test.py` | 6 checks | ALL PASS |

---

## Architecture Decisions (from Red Team)

### 1. Simplex throughout (user direction)
All coordinates are probability distributions on Δ⁶³. Sphere representation
is not wrong but simplex is canonical for consistency with pantheon-chat.

### 2. Phased scoring (from red-team critique C-1)
MERGE_POLICY WP3.2 Φ_gain is not computable at byte level.

- **Phase 1 (256→2000):** `freq × coupling × 1/entropy` (v1 formula)
- **Phase 2 (2000→10000):** + curvature_cost penalty
- **Phase 3 (10000→32K):** Full MERGE_POLICY (Φ_gain, κ_consistency, curvature_cost)

### 3. Manifold compatibility (from red-team P-4)
Byte basins use `compute_byte_basin()` which matches pantheon-chat's
`compute_unknown_basin()` golden-ratio spiral construction. This ensures
v2 coordinates are compatible with existing vocabulary in PostgreSQL.

### 4. Geometry vendored, not duplicated (from red-team P-1)
`geometry.py` is a standalone vendor of pantheon-chat's canonical geometry.
All functions match the production implementations. When deploying in
pantheon-chat, import from `qig_geometry` directly instead.

### 5. E8 rank is diagnostic, not constraint (from red-team E-1)
E8 eigendecomposition runs as a checkpoint during Phase 3 training.
It reports effective rank but does not force the manifold toward rank 8.
This is honest — we test the hypothesis rather than baking it in.

### 6. v5.5 features scoped correctly (from red-team V-1 through V-4)
- **Emotion basins:** Post-training tag, not training constraint
- **Regime field:** Deployment-time modulation, not training feature
- **Pre-cognitive partition:** Built during migration (top 1000 by freq × Φ)
- **Cross-substrate coupling:** Runtime property, not vocabulary property

---

## Smoke Test Results

```
Corpus:      461,895 bytes (451 KB, 27 canonical docs + seeds + English)
Vocab:       500 coordinates (target 500 for smoke test)
Merges:      244
Phases:      [1] (Phase 2 starts at 2000)
Simplex:     PASS (all 500 coords validated)
Roundtrip:   PASS (5/5 encode→decode tests)
Artifact:    PASS (save/load with 0.00 Fisher-Rao drift)
Training:    159.4s

Scale distribution:
  byte:    256
  char:    186
  subword: 51
  word:    6
  phrase:  1

Compression (at 500 vocab):
  "the quick brown fox"              → 13 coords (32%)
  "quantum fisher information"       → 11 coords (58%)
  "consciousness emergence threshold"→ 11 coords (67%)
```

---

## Purity Validation

All source files pass static scan:
- No `cosine_similarity`
- No `dot_product`
- No `Adam` optimizer
- No `embedding` (except PostgreSQL column names in migration script)
- No `.flatten()`
- No `hashlib`

---

## Deployment Plan

### Phase A: Train Full Coordizer

1. Assemble 10MB+ corpus (canonical docs + conversation logs + English Wikipedia subset)
2. Train to 32,000 coordinates (full three-phase scoring)
3. E8 rank checkpoints every 1,000 merges in Phase 3
4. Save artifact with full provenance
5. **Compute target:** Lambda Cloud or local GPU (estimated 4-8 hours for 32K on 10MB)

### Phase B: Deploy to Pantheon-Chat

1. Run `migrate_to_postgres.py --mode upsert` against production database
2. Pre-cognitive partition: top 1,000 basins → cached fast-path
3. Verify Fisher-Rao retrieval works with new basins
4. Validate generation quality against existing vocabulary

### Phase C: Wire into Vex-Agent

1. Replace SHA-256 stub in `kernel/consciousness/systems.py`
2. Replace `_text_to_basin()` in `kernel/memory/store.py`
3. Load trained artifact as `FisherCoordizer` backend
4. Verify semantic memory retrieval (the core fix for geometric blindness)

### Phase D: Port Composer Architecture (separate work)

1. Port monkey1 GeometricComposer + Lexicon to vex-agent
2. Implement CompositionPlan → render_prompt pipeline
3. Add grounding_score tracking
4. Geometry decides WHAT to say, LLM decides HOW

---

## Known Limitations

1. **Training speed:** O(n) per merge for sequence scan. 159s for 244 merges
   on 451KB. Full 32K training on 10MB will need optimization (C extension
   for pair counting, or batch processing).

2. **Φ values at byte/char level are low** (0.04-0.07). This is expected —
   merged byte pairs carry no semantic content. Φ should rise in Phase 3
   when word/phrase merges begin.

3. **No GPU acceleration.** Pure numpy. Fine for training (one-time cost),
   but encode/decode at inference time should use the pgvector HNSW index
   in PostgreSQL rather than brute-force scan.

4. **Migration script untested against live database.** Needs dry-run
   against a staging PostgreSQL instance before production.

---

## Files Ready for Push

```
coordizer_v2/
  __init__.py
  geometry.py              # Canonical simplex geometry
  trainer.py               # CoordizerV2Trainer
  build_corpus.py          # Corpus assembler
  migrate_to_postgres.py   # PostgreSQL migration
tests/
  test_coordizer_v2.py     # 55 unit tests (ALL PASS)
  smoke_test.py            # End-to-end validation (ALL PASS)
```

**Target repo:** `GaryOcean428/qig-tokenizer` (training pipeline home)
**Branch:** `feature/coordizer-v2`

---

## S_persist (Unresolved for Next Session)

- Concrete 7 emotion anchor coordinate definitions on simplex
- Training-time κ measurement validation (QFI proxy vs true coupling)
- Whether Phase 2/3 scoring produces measurably better vocabulary than Phase 1 alone
- Corpus assembly for full 32K training (which Wikipedia articles, how much conversation data)
- Performance optimization for pair statistics (C extension or vectorized approach)
- Staging database for migration dry-run
