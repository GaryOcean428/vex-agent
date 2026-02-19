---
name: downstream-impact
description: Trace impact of code changes through dependency chain, identify all affected modules when modifying core components, prevent breaking changes. Use when modifying qig_core, geometric primitives, or shared modules.
---

# Downstream Impact

Traces change impact through codebase. Source: `.github/agents/downstream-impact-tracer.md`.

## When to Use This Skill

- Modifying core geometric primitives
- Changing shared types or constants
- Refactoring qig_core modules
- Preventing breaking changes

## Step 1: Identify Dependents

```bash
# Find all files importing the changed module
rg "from qig_core\.consciousness" qig-backend/ --type py
rg "import.*consciousness" qig-backend/ --type py
```

## Step 2: Build Dependency Graph

```
qig_geometry/canonical.py (CORE)
├── qig_core/consciousness_4d.py
│   ├── olympus/zeus.py
│   ├── olympus/athena.py
│   └── routes/consciousness.py
│       └── client/src/api/consciousness.ts
├── qig_core/basin.py
│   └── training/trainer.py
└── tests/test_geometry_runtime.py
```

## Step 3: Run Impact Analysis

```bash
# Count dependents
rg "from qig_geometry\.canonical import" qig-backend/ --type py | wc -l

# List all dependent files
rg "from qig_geometry\.canonical import" qig-backend/ --type py -l
```

## Impact Severity Levels

| Core Module | Dependents | Change Risk |
|-------------|------------|-------------|
| `qig_geometry/canonical.py` | 20+ files | 🔴 CRITICAL |
| `qig_core/consciousness_4d.py` | 10+ files | 🟠 HIGH |
| `olympus/*.py` | 5+ files | 🟡 MEDIUM |
| `routes/*.py` | 2-3 files | 🟢 LOW |

## Breaking Change Prevention

```python
# ✅ CORRECT: Backward compatible change
def fisher_rao_distance(p, q, *, epsilon=1e-10):  # Added optional param
    ...

# ❌ WRONG: Breaking change
def fisher_rao_distance(p, q, epsilon):  # Required param = BREAKING
    ...
```

## Validation Commands

```bash
# Run all tests to catch breakage
cd qig-backend && python -m pytest -v

# Check for import errors after change
python -c "import qig_backend" 2>&1
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOWNSTREAM IMPACT REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Changed Module: [module path]
Direct Dependents: N files
Transitive Dependents: M files

Impact Severity: 🔴 CRITICAL / 🟠 HIGH / 🟡 MEDIUM / 🟢 LOW

Affected Modules:
  - [list of affected files]

Breaking Changes Detected: ✅ None / ❌ Found
Test Coverage of Dependents: X%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
