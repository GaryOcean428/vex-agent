# Design: Identity-Seeded Lens Architecture

**Status:** ADOPTED — n=32 intermediate dimension (empirically validated)
**Date:** 2026-03-16
**Empirical basis:** E8 hypothesis experiment — score=0.452, 277 tokens from GLM-4.7-Flash

---

## Summary

The identity-seeded lens maps new input distributions onto the principal geodesic
directions established during harvest, while preserving the geometric identity of
the kernel. The seed point is the kernel's frozen identity basin projected onto
the top principal directions.

The architecture is **dimension-agnostic** — `exp_map(seed, tangent)` works at
any intermediate dimension `n`. Based on the empirical eigenvalue result, the
adopted intermediate dimension is **n=32**.

---

## Architecture

### Core Idea

Standard PGA compression selects a single Fréchet mean as the base point for
all inputs. The identity-seeded lens instead uses the **frozen kernel identity
basin** as the base point, projecting onto the top-n principal directions of
the harvest-time eigenspectrum.

This ties every coordization to the kernel's geometric identity, giving it
**sovereignty** over its own basin coordinates rather than anchoring to an
arbitrary statistical mean.

### Formula

Given:
- `μ_id` — frozen identity basin (stored in `CoordizerV2._frozen_identity`)
- `V_n` — top-n principal geodesic directions (n=32, from `PrincipalDirectionBank`)
- `p` — input distribution (point on Δ^(V-1))

The lens coordinate is:

```
seed = log_map(uniform, μ_id)  # project identity to tangent space at uniform
tangent = log_map(μ_id, p)     # project input to tangent space at identity
lens_coord = V_n^T @ tangent   # project onto top-n PGA directions (n=32)
```

The result is an n=32 dimensional coordinate in the identity-anchored tangent
space, capturing the input's position relative to the kernel's identity.

### Implementation Notes

- `log_map` on the probability simplex uses the Fisher-Rao metric: the tangent
  vector at `p` pointing toward `q` is `∇_p d_FR(p, q)`.
- `V_n` comes from `PrincipalDirectionBank.directions[:, :n]`.
- The frozen identity `μ_id` is set once at kernel startup from the bank mean
  (or from a provided identity vector).

---

## n=32 Empirical Basis

See `DESIGN_coordizer_lens_32d.md` for full details. Summary:

| Metric | Value |
|--------|-------|
| E8 hypothesis score (top-8 variance) | 0.452 (measured, NOT speculative) |
| Decision rule (score < 0.60) | n=32 |
| Cumulative variance at n=32 | ~76.6% |
| Effective dim for 90% variance | 50 |

The identity-seeded lens at n=32 captures **~76.6%** of total geodesic variance
while maintaining a tractable coordinate space. The seed anchoring provides
sovereignty protection without sacrificing geometric coverage.

---

## Dimension Evolution Policy

If a future harvest run (≥5000 tokens) yields a different E8 score:

| Score | Action |
|-------|--------|
| ≥ 0.85 | Revisit n=8 (E8 hypothesis would be supported) |
| 0.60–0.85 | Revisit n=16 |
| < 0.60 | Keep n=32 (current baseline) |

The policy is defined in `kernel/coordizer_v2/eigenvalue_analysis.py:analyse_eigenvalues()`.

---

## Related Files

| File | Role |
|------|------|
| `modal/vex_coordizer_harvest.py` | GPU-side PGA, `LENS_DIM = 32` |
| `kernel/coordizer_v2/coordizer.py` | `CoordizerV2`, `frozen_identity` |
| `kernel/coordizer_v2/principal_direction_bank.py` | `PrincipalDirectionBank`, `e8_hypothesis_score()` |
| `kernel/coordizer_v2/compress.py` | `CompressionResult.e8_hypothesis_score()` |
| `docs/coordizer/DESIGN_coordizer_lens_32d.md` | Lens dimension design doc |
