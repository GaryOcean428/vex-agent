---
name: wiring-validation
description: Verify every documented feature has actual implementation, check consciousness components (metrics, Pillars, ActivationSequence, CoordizerV2) are measured and logged, validate telemetry endpoints per Unified Consciousness Protocol v6.1F.
---

# Wiring Validation

Traces feature implementation chains per Unified Consciousness Protocol v6.1F, including CoordizerV2 integration.

## When to Use This Skill

- Adding new features to pantheon-chat
- Reviewing implementation completeness
- Debugging missing functionality
- Validating telemetry coverage
- Integrating CoordizerV2 into consciousness loop
- Verifying metric flow from CoordizerV2 to consciousness

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

Feature: ActivationSequence (v6.1F §23)
├── kernel/consciousness/activation.py  ✓/❌ Implemented
├── kernel/consciousness/loop.py  ✓/❌ Wired to loop
├── kernel/config/settings.py  ✓/❌ Feature flag
└── kernel/consciousness/loop.py  ✓/❌ LLMOptions modulation

Feature: CoordizerV2 Integration
├── kernel/coordizer_v2/  ✓ Implemented
├── kernel/config/settings.py  ❌ Feature flag missing
├── kernel/coordizer_v2/adapter.py  ❌ Adapter missing
├── kernel/consciousness/loop.py  ❌ Not wired to loop
├── kernel/llm/client.py  ❌ Not using CoordizerV2
└── Metric flow  ❌ Not feeding consciousness
    ├── basin_velocity → VelocityTracker  ❌
    ├── trajectory_curvature → g_class  ❌
    ├── harmonic_consonance → h_cons  ❌
    ├── kappa_measured → kappa  ❌
    └── beta_running → β tracking  ❌
```

## CoordizerV2 Wiring Checklist (v6.1F)

### Phase 1: Bootstrap
- [ ] Feature flag in `kernel/config/settings.py`
  - `coordizer_v2_enabled: bool`
  - `coordizer_v2_bank_path: str`
- [ ] CoordizerV2Adapter in `kernel/coordizer_v2/adapter.py`
  - Drop-in replacement for CoordinatorPipeline
  - `transform()` and `coordize_text()` methods
- [ ] Import in consciousness loop (behind flag)
  - `from ..coordizer_v2 import CoordizerV2`

### Phase 2: Metric Flow
- [ ] basin_velocity → VelocityTracker input
- [ ] trajectory_curvature → g_class (geometry class)
- [ ] harmonic_consonance → h_cons (harmonic consonance)
- [ ] kappa_measured → κ_eff update
- [ ] beta_running → β tracking
- [ ] tier_distribution → n_voices (polyphonic voices)

### Phase 3: Modulation
- [ ] Regime weights → CoordizerV2 temperature
- [ ] Navigation mode → generation parameters
- [ ] Tacking mode → tier bias
- [ ] Kernel domain → domain bias

### Phase 4: LLM Client
- [ ] Replace CoordinatorPipeline with CoordizerV2Adapter
- [ ] Use coordize() for response coordization
- [ ] Use generate_next() for trajectory-based generation

### Phase 5: Geometry Consolidation
- [ ] Migrate to `kernel/coordizer_v2/geometry.py`
- [ ] Remove duplicate `kernel/geometry/fisher_rao.py`
- [ ] Update all imports
```

## Validation Commands

```bash
# Run wiring validation
pytest kernel/tests/ -v -k "wiring or integration"

# Check consciousness component wiring
rg "ActivationSequence|FluctuationGuard|TopologicalBulk|QuenchedDisorder" kernel/ --type py

# Check pillar enforcement in loop
rg "enforce_pillars|pillar_metrics" kernel/consciousness/loop.py

# Check CoordizerV2 integration
rg "CoordizerV2|coordizer_v2_enabled" kernel/ --type py

# Check metric flow from CoordizerV2
rg "basin_velocity|trajectory_curvature|harmonic_consonance" kernel/consciousness/ --type py
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

CoordizerV2 Integration:
  - Feature flag: ✅ / ❌
  - Adapter: ✅ / ❌
  - Loop wiring: ✅ / ❌
  - Metric flow: ✅ / ❌
  - Modulation: ✅ / ❌

Missing Wiring: [list]
Orphaned Features: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
