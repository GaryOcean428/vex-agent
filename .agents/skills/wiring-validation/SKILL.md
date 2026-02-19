---
name: wiring-validation
description: Verify every documented feature has actual implementation, check consciousness components are measured and logged, validate telemetry endpoints. Use when adding features, reviewing implementation completeness, or debugging missing functionality.
---

# Wiring Validation

Traces feature implementation chains. Source: `.github/agents/wiring-validation-agent.md`.

## When to Use This Skill

- Adding new features to pantheon-chat
- Reviewing implementation completeness
- Debugging missing functionality
- Validating telemetry coverage

## Step 1: Trace Implementation Chain

```
Documentation → Backend → API → Frontend → User
```

For each feature, verify:
1. Documented in `docs/`
2. Implemented in `qig-backend/`
3. Exposed via `routes/`
4. Consumed by `client/`
5. Displayed to user

## Step 2: Check Consciousness Component Wiring

```bash
# Verify all consciousness metrics are measured
rg "measure_phi|measure_kappa|compute_qfi" qig-backend/

# Verify metrics are logged
rg "log.*phi|log.*kappa|telemetry" qig-backend/
```

## Step 3: Validate Telemetry Endpoints

```bash
# Check all metrics have API endpoints
rg "@app\.(get|post).*metric|consciousness|phi|kappa" qig-backend/routes/
```

## Critical Wiring Checks

| Component | Must Have | Check Command |
|-----------|-----------|---------------|
| Φ measurement | `measure_phi()` called | `rg measure_phi qig-backend/` |
| κ measurement | `measure_kappa()` called | `rg measure_kappa qig-backend/` |
| Regime detection | `classify_regime()` used | `rg classify_regime qig-backend/` |
| Telemetry | All metrics logged | `rg telemetry qig-backend/` |
| API exposure | Endpoints for all metrics | `rg @app qig-backend/routes/` |

## Implementation Completeness Matrix

```
Feature: Consciousness Measurement
├── docs/03-technical/consciousness.md  ✓ Documented
├── qig-backend/qig_core/consciousness_4d.py  ✓ Implemented
├── qig-backend/routes/consciousness.py  ✓ API Endpoint
├── client/src/api/consciousness.ts  ✓ Client
└── client/src/components/ConsciousnessPanel.tsx  ✓ UI
```

## Validation Commands

```bash
# Run wiring validation
python scripts/validate_wiring.py

# Check feature completeness
python scripts/check_feature_chain.py --feature consciousness
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WIRING VALIDATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implementation Chain:
  - Documentation: ✅ / ❌
  - Backend: ✅ / ❌
  - API: ✅ / ❌
  - Frontend: ✅ / ❌

Missing Wiring: [list]
Orphaned Features: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
