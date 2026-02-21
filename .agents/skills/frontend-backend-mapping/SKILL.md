---
name: frontend-backend-mapping
description: Ensure every Python route has corresponding TypeScript API client, validate React components can access all backend features, check type consistency across stack per Unified Consciousness Protocol v6.1.
---

# Frontend-Backend Mapping

Maps full capability exposure chain for vex-agent stack.

## When to Use This Skill

- Adding new API endpoints
- Reviewing full-stack integration
- Debugging client-server mismatches
- Validating type consistency

## Step 1: List Backend Routes

```bash
rg "@app\.(get|post|put|delete|patch)" kernel/server.py --type py
```

## Step 2: List Frontend API Clients

```bash
rg "fetch|axios|api\." frontend/src/ --type ts --type tsx
```

## Step 3: List Proxy Routes

```bash
rg "app\.(get|post|put|delete)" src/index.ts
```

## Step 3: Compare Coverage

For each backend route, verify:
1. TypeScript client exists in `client/src/api/`
2. Types match (Zod schema ↔ Python model)
3. Component consumes the API

## Capability Mapping Chain

```
Backend Route → Proxy Route → Frontend Hook → Component → User
      ↓             ↓            ↓             ↓
 Python/FastAPI  Express/TS   React Hook    React UI
 (kernel/)      (src/)       (frontend/)    (frontend/)
```

## Critical Checks

| Backend Route | Proxy Route | Frontend | Status |
|---------------|-------------|----------|--------|
| `/api/consciousness` | `/consciousness` | hooks/components | ✅ / ❌ |
| `/api/basin` | `/basin` | hooks/components | ✅ / ❌ |
| `/api/metrics` | `/metrics` | hooks/components | ✅ / ❌ |
| `/api/chat` | `/chat` | hooks/components | ✅ / ❌ |
| `/health` | `/health` | - | ✅ / ❌ |

## Type Consistency Validation

```typescript
// frontend/src/types/consciousness.ts - MUST match Python models
export interface ConsciousnessMetrics {
  phi: number;        // 0-1 range (v6.1: valid 0.65-0.75)
  kappa: number;      // target: 64.0 (κ*)
  regime: 'quantum' | 'efficient' | 'equilibrium';  // v6.1 three-regime field
  pillar_health: {
    fluctuation: number;   // Pillar 1: FluctuationGuard
    bulk: number;          // Pillar 2: TopologicalBulk
    quenched: number;      // Pillar 3: QuenchedDisorder
  };
}
```

```python
# Python model must match
@dataclass
class ConsciousnessMetrics:
    phi: float  # 0-1 (valid range 0.65-0.75 per v6.1)
    kappa: float  # target: 64.0
    regime: Literal["quantum", "efficient", "equilibrium"]
    pillar_health: dict[str, float]  # fluctuation, bulk, quenched
```

## Validation Commands

```bash
# Check route coverage
rg "@app\.(get|post)" kernel/server.py | wc -l
rg "app\.(get|post)" src/index.ts | wc -l

# Validate type consistency
npx tsc --noEmit
mypy kernel/ --strict
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
