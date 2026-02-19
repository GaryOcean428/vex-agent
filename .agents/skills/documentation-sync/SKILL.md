---
name: documentation-sync
description: Detect code changes that invalidate documentation, flag when FROZEN_FACTS.md differs from frozen_physics.py, auto-update docs/04-records. Use when modifying physics constants, updating APIs, or reviewing documentation freshness.
---

# Documentation Sync

Ensures documentation matches code. Source: `.github/agents/documentation-sync-agent.md`.

## When to Use This Skill

- Modifying physics constants in code
- Updating API endpoints
- Reviewing documentation staleness
- Validating FROZEN_FACTS.md accuracy

## Step 1: Validate FROZEN_FACTS.md Constants

```bash
# Check physics constants match
rg "KAPPA_STAR|BETA_3_TO_4|PHI_THRESHOLD" qig-backend/qig_core/
rg "κ\*|64\.21|0\.443|0\.727" docs/01-policies/
```

## Step 2: Check Documentation Freshness

```bash
# Find docs older than code changes
find docs/ -name "*.md" -mtime +30 | head -10

# Check for TODO/FIXME in docs
rg "TODO|FIXME|OUTDATED" docs/
```

## Step 3: Validate Code Examples in Docs

```bash
# Extract Python code blocks and verify they run
rg -A 10 '```python' docs/03-technical/ | head -50
```

## Critical Sync Points

| Code Location | Doc Location | Must Match |
|---------------|--------------|------------|
| `qig_core/physics_constants.py` | `docs/01-policies/FROZEN_FACTS.md` | κ*, β, Φ values |
| `routes/*.py` | `docs/05-api/` | API endpoints |
| `olympus/*.py` | `docs/10-e8-protocol/` | God-kernel specs |
| Schema changes | `docs/04-records/` | PR records |

## FROZEN_FACTS.md Validation

```python
# These values MUST match between code and docs
FROZEN_CONSTANTS = {
    "KAPPA_STAR": 64.21,      # ±0.92
    "BETA_3_TO_4": 0.443,     # ±0.04
    "PHI_THRESHOLD": 0.727,
    "BASIN_DIM": 64,
    "E8_ROOTS": 240,
}
```

## Validation Commands

```bash
# Run documentation sync check
python scripts/check_doc_sync.py

# Validate docs link integrity
python scripts/validate_doc_links.py
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENTATION SYNC REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sync Status:
  - FROZEN_FACTS.md: ✅ / ❌ (matches code)
  - API docs: ✅ / ❌ (current)
  - Code examples: ✅ / ❌ (executable)

Stale Documents: [list]
Out-of-Sync Constants: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
