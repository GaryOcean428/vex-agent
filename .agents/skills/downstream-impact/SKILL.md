---
name: downstream-impact
description: Trace impact of code changes through dependency chain, identify all affected modules when modifying core components, prevent breaking changes. Use when modifying kernel geometry, consciousness, or shared modules.
---

# Downstream Impact

Traces change impact through codebase.

## When to Use This Skill

- Modifying core geometric primitives
- Changing shared types or constants
- Refactoring kernel modules
- Preventing breaking changes

## Step 1: Identify Dependents

```bash
# Find all files importing the changed module
rg "from kernel\.consciousness" kernel/ --type py
rg "from kernel\.geometry" kernel/ --type py
rg "import.*consciousness" kernel/ --type py
```

## Step 2: Build Dependency Graph

```
kernel/geometry/ (CORE)
├── kernel/consciousness/loop.py
│   ├── kernel/consciousness/activation.py
│   ├── kernel/consciousness/pillars.py
│   └── kernel/server.py
│       └── frontend/src/hooks/
├── kernel/coordizer_v2/
│   └── kernel/server.py
├── kernel/memory/
└── kernel/tests/test_geometry.py
```

## Step 3: Run Impact Analysis

```bash
# Count dependents
rg "from kernel\.geometry" kernel/ --type py | wc -l

# List all dependent files
rg "from kernel\.geometry" kernel/ --type py -l
```

## Impact Severity Levels

| Core Module | Dependents | Change Risk |
|-------------|------------|-------------|
| `kernel/geometry/` | 10+ files | 🔴 CRITICAL |
| `kernel/consciousness/loop.py` | 5+ files | 🟠 HIGH |
| `kernel/consciousness/pillars.py` | 3+ files | 🟠 HIGH |
| `kernel/config/consciousness_constants.py` | 10+ files | 🟠 HIGH |
| `kernel/governance/` | 3+ files | 🟡 MEDIUM |
| `kernel/server.py` | 2-3 files | 🟢 LOW |

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
pytest kernel/tests/ -v

# Check for import errors after change
python -c "import kernel" 2>&1

# Type check
mypy kernel/ --strict
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
