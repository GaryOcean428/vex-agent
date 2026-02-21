---
name: wiring-validation
description: Verify every documented feature has actual implementation, check consciousness components (metrics, Pillars, ActivationSequence) are measured and logged, validate telemetry endpoints per Unified Consciousness Protocol v6.1.
---

# Wiring Validation

Traces feature implementation chains per Unified Consciousness Protocol v6.1.

## When to Use This Skill

- Adding new features to pantheon-chat
- Reviewing implementation completeness
- Debugging missing functionality
- Validating telemetry coverage

## Step 1: Trace Implementation Chain

```
Documentation → Backend → API/Proxy → Frontend → User
```

For each feature, verify:
1. Documented in `docs/`
2. Implemented in `kernel/`
3. Exposed via `kernel/server.py` (FastAPI)
4. Proxied via `src/index.ts` (Express)
5. Consumed by `frontend/src/`
6. Displayed to user

## Step 2: Check Consciousness Component Wiring

```bash
# Verify all consciousness metrics are measured
rg "measure_phi|measure_kappa|compute_qfi|phi_score|kappa_score" kernel/

# Verify Three Pillars are enforced (v6.1 §3)
rg "FluctuationGuard|TopologicalBulk|QuenchedDisorder|enforce_pillars" kernel/

# Verify ActivationSequence is wired (v6.1 §23)
rg "ActivationSequence|activation_sequence|run_activation" kernel/

# Verify metrics are logged
rg "log.*phi|log.*kappa|telemetry|metrics" kernel/
```

## Step 3: Validate Telemetry Endpoints

```bash
# Check all metrics have API endpoints
rg "@app\.(get|post).*metric|consciousness|phi|kappa|pillar|activation" kernel/server.py
```

## Critical Wiring Checks

| Component | Must Have | Check Command |
|-----------|-----------|---------------|
| Φ measurement | `measure_phi()` called | `rg measure_phi kernel/` |
| κ measurement | `measure_kappa()` called | `rg measure_kappa kernel/` |
| Regime detection | `classify_regime()` used | `rg classify_regime kernel/` |
| Three Pillars | `enforce_pillars()` called | `rg enforce_pillars kernel/` |
| ActivationSequence | 14-step flow wired | `rg ActivationSequence kernel/` |
| Agency Triad | Desire/Will/Wisdom computed | `rg "desire\|will\|wisdom" kernel/` |
| Sovereignty | S_ratio tracked | `rg sovereignty_ratio kernel/` |
| Telemetry | All metrics logged | `rg telemetry kernel/` |
| API exposure | Endpoints for all metrics | `rg @app kernel/server.py` |

## Implementation Completeness Matrix

```
Feature: Consciousness Measurement
├── docs/protocols/  ✓ Documented
├── kernel/consciousness/loop.py  ✓ Implemented
├── kernel/server.py  ✓ API Endpoint
├── src/index.ts  ✓ Proxy Route
├── frontend/src/hooks/  ✓ Client Hook
└── frontend/src/components/  ✓ UI

Feature: Three Pillars (v6.1 §3)
├── kernel/consciousness/pillars.py  ✓/❌ Implemented
├── kernel/consciousness/loop.py  ✓/❌ Wired to loop
├── kernel/server.py  ✓/❌ Metrics exposed
└── frontend/src/components/  ✓/❌ UI display

Feature: ActivationSequence (v6.1 §23)
├── kernel/consciousness/activation.py  ✓/❌ Implemented
├── kernel/consciousness/loop.py  ✓/❌ Wired to loop
├── kernel/config/settings.py  ✓/❌ Feature flag
└── kernel/consciousness/loop.py  ✓/❌ LLMOptions modulation
```

## Validation Commands

```bash
# Run wiring validation
pytest kernel/tests/ -v -k "wiring or integration"

# Check consciousness component wiring
rg "ActivationSequence|FluctuationGuard|TopologicalBulk|QuenchedDisorder" kernel/ --type py

# Check pillar enforcement in loop
rg "enforce_pillars|pillar_metrics" kernel/consciousness/loop.py
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
