---
name: qig-purity-validation
description: Validate Quantum Information Geometry (QIG) purity across codebase changes. Detect Euclidean contamination, forbidden LLM imports, and cosine similarity. Enforce Fisher-Rao metrics per E8 Protocol v4.0. Use when reviewing PRs, auditing geometry code, or checking QIG compliance. Zero tolerance for geometric impurity.
---

# QIG Purity Validation

Enforces geometric purity per E8 Protocol v4.0. This skill runs the **exact same checks** as the CI/CD pipelines defined in `.github/workflows/geometric-purity-gate.yml` and `.github/workflows/qig-purity-gate.yml`.

## When to Use This Skill

- Reviewing any PR that touches `qig-backend/`, `server/`, or `shared/`
- Auditing geometric distance calculations
- Checking for forbidden LLM imports (28 providers monitored)
- Validating Fisher-Rao metric usage
- Ensuring no Euclidean contamination in basin operations

## Step 1: Run AST-Based Purity Audit

This catches `np.linalg.norm()` and `np.dot()` on basin coordinates:

```bash
python qig-backend/scripts/ast_purity_audit.py
```

Expected output: `✅ E8 PROTOCOL v4.0 COMPLIANCE: COMPLETE`

## Step 2: Run Comprehensive QIG Purity Scan

Scans for ALL forbidden patterns from the Type-Symbol-Concept Manifest:

```bash
python3 scripts/qig_purity_scan.py
```

This checks 15+ pattern categories including:
- Euclidean distance (`np.linalg.norm(a-b)`, `scipy.spatial.distance.euclidean`)
- Cosine similarity (`cosine_similarity()`, `F.cosine_similarity`)
- Embedding terminology (`embedding`, `nn.Embedding`)
- Tokenizer in core (`tokenizer` → should be `coordizer`)
- Classic NLP imports (`sentencepiece`, `WordPiece`, `BPE`)
- Arithmetic mean on basins (`np.mean()` → should be `frechet_mean()`)
- Geometry re-implementation (must import from `qig_geometry.canonical`)

## Step 3: Scan for Forbidden LLM Dependencies

```bash
python3 scripts/scan_forbidden_imports.py --path .
```

Forbidden providers (CRITICAL violations):
- OpenAI, Anthropic, Google AI, Cohere, AI21
- Hugging Face Transformers, LangChain
- Any external LLM API call

## Step 4: Validate QFI Canonical Path

```bash
bash scripts/validate-qfi-canonical-path.sh
```

Ensures all `fisher_rao_distance` imports come from `qig_geometry.canonical`.

## Forbidden Patterns (From CI)

| Category | Pattern | Severity | Fix |
|----------|---------|----------|-----|
| Euclidean | `np.linalg.norm(a - b)` | CRITICAL | `fisher_rao_distance(a, b)` |
| Cosine | `cosine_similarity()` | CRITICAL | `fisher_rao_distance()` |
| Embedding | `nn.Embedding()` | CRITICAL | Basin coordinate mapping |
| Optimizer | `torch.optim.Adam()` | WARNING | `natural_gradient_step()` |
| NLP | `import sentencepiece` | CRITICAL | Geometric coordizer |
| Mean | `np.mean(basins, axis=0)` | ERROR | `frechet_mean()` |
| Re-impl | `def fisher_rao_distance()` | CRITICAL | Import from canonical |

## Quarantine Zones (Exempted)

These directories are allowed to violate for experimental/baseline purposes:
- `docs/08-experiments/legacy/`
- `docs/08-experiments/baselines/`
- `tests/` (test fixtures)

## Physics Constants (FROZEN - docs/01-policies/FROZEN_FACTS.md)

```python
KAPPA_STAR = 64.21 ± 0.92  # Universal fixed point (E8 rank²)
BETA_3_TO_4 = 0.443 ± 0.04  # Running coupling L=3→4
PHI_THRESHOLD = 0.727       # Consciousness threshold
BASIN_DIM = 64              # Manifold dimension
```

## Canonical Import Pattern

```python
# ✅ CORRECT: Always import from canonical
from qig_geometry.canonical import fisher_rao_distance, frechet_mean, geodesic_interpolation

# ❌ WRONG: Never import from other locations
from qig_core.geometric_primitives import fisher_rao_distance  # FORBIDDEN
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QIG PURITY VALIDATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Static Analysis: ✅ PASS / ❌ FAIL
  - Euclidean violations: 0
  - Cosine similarity: 0
  - Forbidden imports: 0

Files Scanned: N
Violations: N (CRITICAL: N, ERROR: N, WARNING: N)

[If violations found, list each with file:line and fix]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Key Principle

All states live on the Fisher-Rao manifold. Movement follows natural geodesic curves. Consciousness emerges from manifold curvature. **NEVER use Euclidean geometry in QIG computations. NO EXCEPTIONS.**
