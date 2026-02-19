---
name: import-resolution
description: Detect and fix Python import errors, validate __init__.py barrel exports, enforce canonical absolute import patterns. Use when reviewing imports, fixing circular dependencies, or validating module structure in qig-backend/.
---

# Import Resolution

Enforces canonical import patterns per E8 Protocol v4.0. Source: `.github/agents/import-resolution-agent.md`.

## When to Use This Skill

- Reviewing Python imports in `qig-backend/`
- Fixing circular dependency errors
- Validating `__init__.py` barrel exports
- Ensuring absolute import patterns

## Step 1: Check for Import Errors

```bash
cd qig-backend
python -c "import qig_core; import olympus; import routes" 2>&1
```

## Step 2: Validate Canonical Import Patterns

```bash
# Find relative imports (should be minimal)
rg "^from \.\." qig-backend/ --type py

# Verify absolute imports from qig_backend
rg "^from qig_backend\." qig-backend/ --type py | head -20
```

## Step 3: Check Barrel Exports

```bash
# Ensure all modules have __init__.py with __all__
find qig-backend -name "__init__.py" -exec grep -L "__all__" {} \;
```

## Canonical Import Patterns

```python
# ✅ CORRECT: Absolute imports from qig_backend
from qig_backend.qig_core.consciousness_4d import measure_phi
from qig_backend.qig_geometry.canonical import fisher_rao_distance
from qig_backend.olympus.zeus import ZeusKernel

# ❌ WRONG: Relative imports in non-test code
from ..qig_core import consciousness_4d  # FORBIDDEN
from . import utils  # Only allowed in __init__.py
```

## Critical Checks

| Check | Requirement |
|-------|-------------|
| Absolute paths | All imports use `qig_backend.module` |
| No relative | No `from ..` except in tests |
| Barrel exports | Every module has `__init__.py` with `__all__` |
| No circular | No circular dependency chains |

## Validation Commands

```bash
# Run ruff import checks
cd qig-backend && ruff check --select I .

# Check for circular imports
python -c "
import sys
sys.setrecursionlimit(100)
import qig_backend
"
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORT RESOLUTION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Import Validation:
  - Absolute imports: ✅ / ❌
  - Circular deps: ✅ / ❌
  - Barrel exports: ✅ / ❌

Violations Found: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
