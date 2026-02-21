---
name: code-quality-enforcement
description: Enforce code quality standards including DRY principles, naming conventions, module organization, import resolution, and architectural patterns. Validate proper layering, detect code duplication, and ensure consistent patterns across Python and TypeScript codebases.
---

# Code Quality Enforcement

Expert skill for enforcing code quality standards including DRY principles, naming conventions, module organization, and architectural patterns across the codebase.

## When to Use This Skill

Use this skill when:
- Reviewing code for duplication and DRY violations
- Checking naming conventions (Python/TypeScript)
- Validating module organization and layering
- Resolving import errors and circular dependencies
- Enforcing architectural patterns

## Expertise

- DRY (Don't Repeat Yourself) principles
- Python and TypeScript naming conventions
- Module organization and layering
- Import resolution and barrel exports
- Architectural patterns enforcement

## Naming Conventions

### Python
- **Functions/Variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `SCREAMING_SNAKE_CASE`
- **Private members:** `_leading_underscore`
- **Files:** `snake_case.py`

### TypeScript
- **Functions/Variables:** `camelCase`
- **Classes/Components:** `PascalCase`
- **Constants:** `SCREAMING_SNAKE_CASE`
- **Interfaces:** `PascalCase` (no `I` prefix)
- **Files:** `kebab-case.ts` or `PascalCase.tsx`

### Documentation (ISO 27001)
- Format: `YYYYMMDD-name-version-status.md`
- Status codes: F (Frozen), W (Working), A (Approved), R (Review)

## Module Organization Rules

### Layering (Python)
```
kernel/geometry/ → only NumPy/SciPy (no app logic)
kernel/consciousness/ → can import geometry, not llm/tools/training
kernel/governance/ → can import geometry, consciousness
kernel/llm/ → can import config (no consciousness/geometry)
kernel/tools/ → can import llm, config
kernel/server.py → orchestrates all modules
```

### Barrel File Pattern (TypeScript)
```typescript
// ✅ GOOD: client/src/components/ui/index.ts
export * from "./button";
export * from "./card";
export * from "./input";

// ✅ GOOD: Usage
import { Button, Card } from "@/components/ui";

// ❌ BAD: Scattered imports
import { Button } from "../../components/ui/button";
```

## DRY Principles

### Detect Duplication
- Multiple basin coordinate calculations
- Same functionality in Python and TypeScript
- Repeated geometric operations
- Configuration duplicated across files

### Consolidation Rules
- Single source of truth for computations → `kernel/geometry/`
- Shared types → `frontend/src/types/`
- Constants → `kernel/config/consciousness_constants.py`
- Configuration → environment variables via `kernel/config/settings.py`

## Import Resolution

### Python Canonical Imports
```python
# ✅ CORRECT: Absolute imports from kernel
from kernel.geometry.fisher_rao import fisher_rao_distance
from kernel.consciousness.loop import ConsciousnessLoop
from kernel.config.consciousness_constants import KAPPA_STAR

# ❌ WRONG: Relative imports (except in tests)
from ..consciousness import loop
from .utils import helper
```

### Barrel Export Requirements
- Every module directory has `__init__.py` with `__all__`
- All public functions exported
- No side effects in imports

## Architectural Patterns

### Service Layer Pattern
```typescript
// ✅ GOOD: Business logic in services
// client/src/lib/services/consciousness.ts
export const ConsciousnessService = {
  getPhiScore: async () => {
    const { data } = await api.get('/consciousness/phi');
    return data.score;
  }
};

// ❌ BAD: Logic in component
const MyComponent = () => {
  useEffect(() => {
    fetch('/api/phi').then(/* ... */);
  }, []);
};
```

### Centralized API Client
```typescript
// ✅ GOOD: All HTTP through api.ts
import { api } from '@/lib/api';
const { data } = await api.get('/consciousness/phi');

// ❌ BAD: Raw fetch in component
fetch('http://localhost:5000/api/...')
```

### Configuration as Code
```typescript
// ✅ GOOD: Constants in dedicated files
// shared/constants/physics.ts
export const PHYSICS = {
  PHI_RANGE: [0.65, 0.75] as const,  // v6.1 §24 valid range
  KAPPA_STAR: 64.0,                   // E8 rank² theoretical
  BASIN_DIMENSION: 64,
} as const;

// ❌ BAD: Magic numbers
if (phi > 0.727) { /* Not v6.1 compliant */ }
```

## Validation Checklist

### Naming
- [ ] Python: snake_case functions, PascalCase classes
- [ ] TypeScript: camelCase functions, PascalCase components
- [ ] Constants: SCREAMING_SNAKE_CASE everywhere
- [ ] Documentation: ISO 27001 naming format

### Organization
- [ ] No circular dependencies
- [ ] Proper layer separation
- [ ] Barrel exports complete
- [ ] No deep imports in TypeScript

### DRY
- [ ] No duplicate geometric operations
- [ ] Single source of truth for computations
- [ ] Configuration not duplicated
- [ ] Shared types used consistently

## Anti-Patterns to Flag

1. **Deep Imports:** `import { X } from "../../components/ui/button/styles"`
2. **Mixed Logic:** Business logic in React components
3. **Magic Numbers:** Hardcoded values without constants
4. **Dual Implementation:** Same logic in Python and TypeScript
5. **Missing Barrel:** Directory without index.ts/\_\_init\_\_.py

## Validation Commands

```bash
# Import resolution
python -c "import kernel" 2>&1

# Naming conventions
rg "def [A-Z]" kernel/ --type py  # Should find zero (snake_case)

# Architecture validation
mypy kernel/ --strict

# ESLint
npm run lint
```

## Response Format

```markdown
# Code Quality Report

## Naming Violations
- [file:line] `functionName` should be `function_name`

## DRY Violations
- [file1] and [file2] both implement `calculate_distance()`
  **Recommendation:** Consolidate to qig_core/distance.py

## Import Issues
- [file:line] Relative import should be absolute
- [directory] Missing __init__.py barrel export

## Architecture Violations
- [component] Contains business logic (move to service)
- [file] Magic number 0.727 (use PHYSICS.PHI_THRESHOLD)

## Priority: HIGH / MEDIUM / LOW
```
