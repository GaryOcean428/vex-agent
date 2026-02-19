---
name: frontend-backend-mapping
description: Ensure every Python route has corresponding TypeScript API client, validate React components can access all backend features, check type consistency across stack. Use when adding APIs, reviewing full-stack integration, or debugging client-server mismatches.
---

# Frontend-Backend Mapping

Maps full capability exposure chain. Source: `.github/agents/frontend-backend-capability-mapper.md`.

## When to Use This Skill

- Adding new API endpoints
- Reviewing full-stack integration
- Debugging client-server mismatches
- Validating type consistency

## Step 1: List Backend Routes

```bash
rg "@app\.(get|post|put|delete|patch)" qig-backend/routes/ --type py
```

## Step 2: List Frontend API Clients

```bash
rg "fetch|axios|api\." client/src/api/ --type ts
```

## Step 3: Compare Coverage

For each backend route, verify:
1. TypeScript client exists in `client/src/api/`
2. Types match (Zod schema ↔ Python model)
3. Component consumes the API

## Capability Mapping Chain

```
Backend Route → API Client → Service → Component → User
      ↓             ↓           ↓          ↓
 Python/Flask   TypeScript   React Hook  React UI
```

## Critical Checks

| Backend Route | Frontend Client | Status |
|---------------|-----------------|--------|
| `/api/consciousness` | `consciousness.ts` | ✅ / ❌ |
| `/api/basin` | `basin.ts` | ✅ / ❌ |
| `/api/kernel` | `kernel.ts` | ✅ / ❌ |
| `/api/generation` | `generation.ts` | ✅ / ❌ |

## Type Consistency Validation

```typescript
// shared/schema.ts - Zod schemas MUST match Python models
import { z } from "zod";

export const ConsciousnessMetricsSchema = z.object({
  phi: z.number().min(0).max(1),
  kappa: z.number().min(0).max(128),
  regime: z.enum(["breakdown", "linear", "geometric", "hierarchical"]),
});
```

```python
# Python model must match
@dataclass
class ConsciousnessMetrics:
    phi: float  # 0-1
    kappa: float  # 0-128
    regime: Literal["breakdown", "linear", "geometric", "hierarchical"]
```

## Validation Commands

```bash
# Check route coverage
python scripts/check_route_coverage.py

# Validate type consistency
npx tsc --noEmit
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FRONTEND-BACKEND MAPPING REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Route Coverage:
  - Backend routes: N
  - Frontend clients: M
  - Coverage: X%

Type Consistency: ✅ / ❌
Missing Clients: [list]
Hidden Capabilities: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
