# Design: CoordizerV2 Lens — 32D Intermediate Dimension

**Status:** ADOPTED (supersedes speculative 8D design)
**Date:** 2026-03-16
**Empirical basis:** E8 hypothesis experiment — score=0.452, 277 tokens from GLM-4.7-Flash

---

## Summary

The CoordizerV2 lens compresses full vocabulary-dimensional probability
distributions (Δ^(V-1), V ≈ 65K for GLM-4.7-Flash) to the 64D basin simplex
(Δ⁶³) via Fisher-Rao Principal Geodesic Analysis (PGA).

The **lens intermediate dimension** `n` determines how many top PGA directions
are retained during this compression. **n=32 is the adopted default.**

---

## Decision Gate (from Eigenvalue Analysis)

The eigenvalue analysis (`kernel/coordizer_v2/eigenvalue_analysis.py`) uses the
following decision rule based on the E8 hypothesis score (top-8 PGA variance
ratio):

| E8 Score | n | Interpretation |
|----------|---|----------------|
| > 0.85   | 8 | E8 hypothesis SUPPORTED — 8 directions sufficient |
| 0.60–0.85 | 16 | E8 hypothesis PARTIAL — fine structure beyond rank-8 |
| < 0.60   | 32 | E8 hypothesis NOT SUPPORTED — variance broadly distributed |

**Measured result:** score = **0.452** → **n = 32**

---

## Empirical Result (2026-03-16)

| Metric | Value |
|--------|-------|
| E8 hypothesis score (top-8 variance) | **0.452** |
| E8 expected value (speculative, rank²) | ~0.877 |
| Effective dim (90% variance) | 50 |
| Effective dim (95% variance) | 57 |
| Effective dim (99% variance) | 64 |
| Spectral gap λ₁/λ₈ | 4.29 |
| Tokens compressed | 277 |
| Source dim (vocab) | ~65K (GLM-4.7-Flash) |

### Interpretation

- The **spectral gap of 4.29** shows that the spectrum is NOT flat — there IS
  geometric structure. Synthetic Dirichlet data scores ~0.161, so real LLM
  distributions are structured.
- However, the structure does **NOT** concentrate into 8 dimensions. 90%
  variance requires 50 dimensions, 95% requires 57.
- The E8 hypothesis as stated (8 dims capture 87.7%) is **not supported** by
  this data.

### Status of the 8D Claim

The 8D claim was always **SPECULATIVE** — derived 2 hops from κ*=64=8², not
measured. This experiment provides the measurement. The E8 connection to
κ*=64 remains valid in the physics (TFIM lattice) but does **NOT** extend to
the semantic eigenvalue spectrum of LLM output distributions.

---

## Architecture

### Lens Compression Pipeline (GPU-side, Modal)

```
Text → LLM → V-dim output distributions → PGA(n=32) → 32D lens coords
                                                      → 64D basin coords (zero-padded + normalised)
```

Implementation: `modal/vex_coordizer_harvest.py`, `CoordizerHarvester._pga_compress()`

```python
LENS_DIM = 32  # from eigenvalue analysis: cumulative variance 0.7661 at dim 32
BASIN_DIM = 64  # frozen

def _pga_compress(self, fingerprints_dict, lens_dim=LENS_DIM, basin_dim=BASIN_DIM):
    ...
    target_dim = min(lens_dim, N, basin_dim)  # typically 32
    ...
    basin_coords[:target_dim] = lens_coords[:target_dim]
    # Remaining basin_coords[32:64] = 0, then unit-normalised
```

### Key Properties

- **`LENS_DIM = 32`**: Top 32 PGA directions capture ~76.6% of total geodesic
  variance (cumulative variance at dim 32 from the empirical spectrum).
- **`BASIN_DIM = 64`**: Frozen — the probability simplex Δ⁶³ is fixed by κ*=64.
- Basin coords are **zero-padded** from dim 32 to dim 64, then unit-normalised.
  This is valid because the PGA basis is orthogonal — unused dimensions contribute
  zero geodesic variance.

---

## Caveats and Future Work

- 277 tokens is a **modest** sample. The eigenvalue analysis has been updated to
  default `target_tokens=5000` for more accurate measurements.
- A second harvest run with target_tokens=5000 and a larger/more-diverse corpus
  would strengthen confidence in n=32.
- If a larger corpus yields a score > 0.60, the lens dim should be revisited.
- Results will be recorded in FROZEN_FACTS once confirmed with ≥5000 tokens.

---

## Related Files

| File | Role |
|------|------|
| `modal/vex_coordizer_harvest.py` | GPU-side PGA compress, `LENS_DIM = 32` |
| `kernel/coordizer_v2/eigenvalue_analysis.py` | Decision gate, reporting |
| `kernel/coordizer_v2/compress.py` | CPU-side compression, `e8_hypothesis_score()` |
| `kernel/coordizer_v2/validate.py` | E8 eigenvalue test (informational) |
| `docs/coordizer/DESIGN_identity_seeded_lens.md` | Identity-seeded lens at n=32 |
