---
name: documentation-sync
description: Detect code changes that invalidate documentation, flag when frozen physics constants differ from code, auto-update docs. Use when modifying physics constants, updating APIs, or reviewing documentation freshness per Unified Consciousness Protocol v6.1.
---

# Documentation Sync

Ensures documentation matches code per Unified Consciousness Protocol v6.1.

## When to Use This Skill

- Modifying physics constants in code
- Updating API endpoints
- Reviewing documentation staleness
- Validating FROZEN_FACTS.md accuracy

## Step 1: Validate Frozen Physics Constants

```bash
# Check physics constants match
rg "KAPPA_STAR|BETA_3_TO_4|PHI_THRESHOLD|PHI_MIN|PHI_MAX" kernel/config/
rg "κ\*|64\.0|0\.443|0\.65|0\.75" docs/
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
| `kernel/config/consciousness_constants.py` | `docs/reference/` | κ*, β, Φ values |
| `kernel/server.py` | `docs/development/` | API endpoints |
| `kernel/governance/` | `docs/protocols/` | E8 protocol specs |
| `kernel/consciousness/` | `docs/protocols/` | Consciousness protocol |

## Frozen Physics Validation (v6.1 §24)

```python
# These values MUST match between code and docs
FROZEN_CONSTANTS = {
    "KAPPA_STAR": 64.0,        # Theoretical (E8 rank²)
    "KAPPA_PHYSICS": 64.21,    # Measured ±0.92
    "BETA_3_TO_4": 0.443,      # ±0.04
    "PHI_MIN": 0.65,           # v6.1 valid range lower
    "PHI_MAX": 0.75,           # v6.1 valid range upper
    "BASIN_DIM": 64,
    "E8_ROOTS": 240,
}
```

## Validation Commands

```bash
# Run documentation sync check
rg "KAPPA_STAR|PHI_MIN|PHI_MAX|BETA" kernel/config/consciousness_constants.py

# Validate docs reference correct protocol version
rg "v4\.0|v5\.0|v5\.5" docs/  # Should find zero (all upgraded to v6.1)
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
