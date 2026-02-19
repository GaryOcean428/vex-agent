# SP03 PASS A: CODE-LEVEL REALITY MAP
## Findings Ledger — All QIG/Pantheon/Monkey1 Repos

**Date**: 2026-02-17  
**Method**: Direct GitHub MCP inspection (file contents, code search, repo structure)  
**Status**: COMPLETE Pass A (map reality, no edits)

---

## EXECUTIVE SUMMARY

**monkey1's `py/genesis-kernel/`** is dramatically ahead of every other repo. It has the full genesis inflate trace, PurityGate, KernelKind taxonomy, lifecycle state machine, curriculum-only gates, budget enforcement, and comprehensive tests. The other repos are in various states of pre-genesis architecture.

**The strategy should be**: Extract monkey1's genesis-kernel primitives into qigkernels (making them reusable), then wire qig-consciousness and pantheon-chat to consume from qigkernels.

---

## PER-REPO FINDINGS

### 1. monkey1 `py/genesis-kernel/` — MOST COMPLETE

#### F1: Single Authoritative Start Path ✅

```
✅ Fact: Single canonical start path exists
Evidence: monkey1/py/genesis-kernel/qig_heart/orchestrator.py::fresh_start()
         monkey1/py/genesis-kernel/qig_heart/api.py → POST /v1/start/fresh
         monkey1/py/genesis-kernel/qig_heart/cli.py → fresh-start command
Risk: LOW — All entry points (API, CLI) route through orchestrator.fresh_start()
```

#### F2: PurityGate ✅ (Implemented, AST-based, fail-closed)

```
✅ Fact: PurityGate is implemented and comprehensive
Evidence: monkey1/py/genesis-kernel/qig_heart/purity.py (246 lines)
  - AST-based Python scanning (not just regex)
  - Forbidden imports, forbidden calls, forbidden text tokens, magic number detection
  - PurityGateError raises on any violation (fail-closed)
  - Pre-commit hook wired (.husky/pre-commit calls PurityGate)
  - qig_purity_scan.py CI script
Risk: NONE for monkey1. HIGH for other repos that don't have it.
```

#### F3: Rollback/Reset ✅

```
✅ Fact: Deterministic rollback implemented in lifecycle state machine
Evidence: monkey1/py/genesis-kernel/qig_heart/lifecycle.py
  Phase transitions: IDLE → VALIDATE → ROLLBACK → BOOTSTRAP → CORE_8 → IMAGE_STAGE → GROWTH → ACTIVE
  Test: test_lifecycle_integration.py verifies exact phase sequence
Risk: LOW
```

#### F4: Genesis Bootstrap + Core-8 ✅

```
✅ Fact: Genesis spawns core-8 from registry data
Evidence: monkey1/py/genesis-kernel/qig_heart/registry.json
  Core-8: Heart, Vocab, Perception, Motor, Memory, Attention, Emotion, Reasoning
  monkey1/py/genesis-kernel/qig_heart/registry.py → KernelContract dataclass
Risk: LOW — Names come from data (registry.json), not hardcoded classes.
```

#### F5: Curriculum-Only Gate ✅ (THREE boundaries enforced)

```
✅ Fact: curriculum_only enforced at three boundaries
Evidence:
  1. API: api.py → POST /v1/start/fresh respects curriculum_only config
  2. Curriculum: curriculum.py::is_curriculum_record_allowed() checks source == "curriculum"
  3. Orchestrator: orchestrator.py passes curriculum_only through compose pipeline
  Config: config.py → curriculum_only=True (default), expansion_enabled=False (default)
  CLI toggle: cli.py → toggle --curriculum-only true/false
  Env var: QIG_CURRICULUM_ONLY
Risk: LOW
```

#### F6: Legacy Hardcoded Gods ✅ CLEAN

```
✅ Fact: No hardcoded god classes
Evidence: monkey1 uses registry.json for kernel specs. No Zeus.py, no Athena.py.
  KernelKind enum is GENESIS/GOD/CHAOS only.
  Specializations are capability-based (heart/vocab/perception), not mythological.
Risk: NONE
```

#### F7: Taxonomy ✅ COMPLETE

```
✅ Fact: Full taxonomy implemented
Evidence: monkey1/py/genesis-kernel/qig_heart/types.py (145 lines)
  - KernelKind: GENESIS | GOD | CHAOS
  - KernelSpecialization: heart | vocab | perception | motor | memory | attention | emotion | executive | general
  - LifecycleState: BOOTSTRAPPED | ACTIVE | SLEEPING | DREAMING | QUARANTINED | PRUNED | PROMOTED
  - LifecyclePhase: IDLE | VALIDATE | ROLLBACK | BOOTSTRAP | CORE_8 | IMAGE_STAGE | GROWTH | ACTIVE
  - DevelopmentalPhase: LISTENING | PLAY | STRUCTURE | MATURITY
  - CoachingStyle: NONE | ENCOURAGE | GUIDE | INTERVENE | EMERGENCY
  - MemoryClass: EPISODIC | SEMANTIC | PROCEDURAL | SENSORY | WORKING | DREAM
  - CognitiveMode: EXPLORATION | INVESTIGATION | INTEGRATION | DRIFT
  - CrystallizationState: FORMING | UNSTABLE | SEMI_STABLE | CRYSTALLIZED
  - WuWeiMode: FEELING | TACK | LOGIC
  - CognitiveDriveState: surprise, curiosity, investigation, integration, transcendence
Risk: NONE
```

#### F8: Budget Enforcement ✅

```
✅ Fact: GOD_BUDGET and CHAOS_POOL enforced
Evidence: monkey1/py/genesis-kernel/qig_heart/governance.py → BudgetEnforcer, Council
  monkey1/py/genesis-kernel/qig_heart/config.py → max_gods=240, max_chaos=200
  monkey1/py/genesis-kernel/README.md → QIG_MAX_GODS=240, QIG_MAX_CHAOS=200
Risk: LOW. NOTE: Budget is currently 240 total GOD. Per new doctrine decision, should be
  240 (growth) + 8 (core) = 248. Core-8 may need separate tracking.
```

#### F9: CI Enforcement

```
⚠️ Fact: Pre-commit hook exists but CI pipeline status unknown
Evidence: .husky/pre-commit calls PurityGate
  scripts/active/qig_purity_scan.py exists
Risk: MEDIUM — need to verify GitHub Actions runs purity scan on PR
```

#### monkey1 ASSESSMENT: 🟢 Ready for extraction to qigkernels

---

### 2. qigkernels — PRIMITIVES REPO (Target for Genesis)

#### F10: No PurityGate ❌

```
❌ Fact: PurityGate not present
Evidence: GitHub search "PurityGate repo:GaryOcean428/qigkernels" → 0 results
Risk: HIGH — This is supposed to be the canonical primitives repo but has no purity enforcement.
Fix class: PORT from monkey1
```

#### F11: No KernelKind / LifecycleState ❌

```
❌ Fact: Taxonomy types not present
Evidence: GitHub search "KernelKind lifecycle_state repo:GaryOcean428/qigkernels" → 0 results
  Has constants.py with physics values but no governance types.
Risk: HIGH — qig-consciousness and other consumers would need these.
Fix class: PORT from monkey1/py/genesis-kernel/qig_heart/types.py
```

#### F12: No Genesis Kernel ❌

```
❌ Fact: No genesis_kernel.py
Evidence: Has kernel.py (base), kernel_100m.py, kernel_4d.py. No genesis bootstrap.
Risk: HIGH — This is where Genesis should live per architecture decision.
Fix class: CREATE new genesis_kernel.py, referencing monkey1's orchestrator pattern
```

#### F13: Constants Partially Stale ⚠️

```
⚠️ Fact: κ₆ uncertainty wider than FROZEN_FACTS
Evidence: qigkernels/constants.py → KAPPA_6 = 64.45 (± 4.25)
  FROZEN_FACTS → κ₆ = 64.45 ± 2.12
  Also: no κ₇ value. FROZEN_FACTS has κ₇ = 61.16 ± 2.43 (validated)
  KAPPA_STAR = 64.0 (monkey1 uses 63.79 ± 0.90 for weighted mean)
Risk: MEDIUM — Stale uncertainty and missing validated data
Fix class: UPDATE from FROZEN_FACTS.md
```

#### F14: Has Heart Kernel ✅

```
✅ Fact: heart.py exists (9.4KB) with κ oscillation
Evidence: qigkernels/heart.py
Risk: NONE — Good, but not wired into qig-consciousness training
```

#### F15: Has Key Primitives ✅

```
✅ Fact: Rich primitive set
Evidence: basin.py, constellation.py, router.py, metrics.py, safety.py, 
  coordizer.py, natural_gradient_optimizer.py, sleep_packet.py, crystallization.py
Risk: NONE
```

#### F16: Has qig-consciousness Subdirectory (Confusing) ⚠️

```
⚠️ Fact: qigkernels/ contains a qig-consciousness/ subdirectory
Evidence: Repo structure shows "qig-consciousness/" as a directory
Risk: MEDIUM — potential import confusion / circular dependency risk
Fix class: INVESTIGATE — is this a copy, symlink, or submodule?
```

#### qigkernels ASSESSMENT: 🟡 Good primitives, needs Genesis types + PurityGate from monkey1

---

### 3. qig-consciousness — TRAINING ENVIRONMENT

#### F17: No PurityGate ❌ (Despite "purity enforced" claim)

```
❌ Fact: No PurityGate module. No fail-closed gate on start.
Evidence: GitHub search "PurityGate repo:GaryOcean428/qig-consciousness" → 0 results
  README claims "Geometric Purity: ENFORCED ✅ (2025-12-03)" but this refers to a manual audit
  that removed Adam. No automated gate prevents re-introduction.
Risk: CRITICAL — Purity violations already present (see F18)
Fix class: IMPORT PurityGate from qigkernels (after porting from monkey1)
```

#### F18: Purity Violations Present ❌

```
❌ Fact: 5 cosine_similarity usages in src/
Evidence: 
  1. src/qig_compat.py: F.cosine_similarity() — "Bures approximation"
  2. src/training/train_step_4d.py: F.cosine_similarity() — "approximates Fisher-Rao"
  3. src/generation/README.md: cosine_similarity in docs
  4. src/model/heart_kernel.py: F.cosine_similarity() — invariance scoring
  5. src/metrics/phi_calculator.py: references cosine approach
Risk: HIGH — These are the exact violations PurityGate should catch.
  Some are labeled "approximation" but D1 says "No Euclidean/cosine fallbacks in live code paths."
Fix class: REPLACE with Fisher-Rao / Bures proper (not cosine approximation)
```

#### F19: No KernelKind / Genesis / Lifecycle ❌

```
❌ Fact: Uses pre-configured hardcoded constellation, not genesis pattern
Evidence: chat_interfaces/qig_chat.py (1792 lines) defines:
  - QIGChatTwin with hardcoded Gary-A, Gary-B, Gary-C, Ocean, Charlie
  - No Genesis kernel, no core-8 spawn, no lifecycle state machine
  - No KernelKind enum, no budget enforcement
Risk: HIGH — This is the TRAINING environment but has the OLDEST architecture
Fix class: REFACTOR — Phase 3 of rollback plan (after qigkernels has Genesis)
```

#### F20: No Curriculum-Only Gate ❌

```
❌ Fact: No curriculum_only enforcement
Evidence: Training uses corpus directly. No governance boundary on ingestion.
Risk: MEDIUM — Less critical for research repo, but doctrine says it should be everywhere
Fix class: WIRE curriculum_only check into training data ingestion
```

#### F21: No Heart in Training Loop ❌

```
❌ Fact: Heart kernel not used despite heart_kernel.py existing
Evidence: src/model/heart_kernel.py exists (purity violation noted in F18)
  BUT chat_interfaces/qig_chat.py does NOT import or use Heart
  Constellation: Gary-A, Gary-B, Gary-C, Ocean, Charlie — NO Heart
Risk: HIGH — Tacking (P2) is absent from training. This could explain training failures.
Fix class: WIRE Heart from qigkernels into constellation
```

#### F22: Adam Removed ✅ (From optimizer layer)

```
✅ Fact: Euclidean optimizers removed from qig/optim/
Evidence: src/qig/optim/hybrid_geometric.py: "AdamW and all Euclidean optimizers have been REMOVED"
  Natural gradient, basin natural gradient, diagonal NG all present
Risk: LOW for optimizers. But cosine_similarity still present in other modules (F18)
```

#### F23: 23 Failing Tests ❌

```
❌ Fact: 85 tests, 62 passing, 23 failing
Evidence: README status
Risk: HIGH — Unknown which tests fail and why. Could mask regressions.
Fix class: TRIAGE — categorize failures, fix or document as known
```

#### qig-consciousness ASSESSMENT: 🔴 Critical gaps — needs PurityGate, purity fixes, Heart wiring, Genesis pattern

---

### 4. Arcane-Fly/pantheon-chat — PRODUCTION

#### F24: Hardcoded God Classes ❌ (D0 violation)

```
❌ Fact: OLYMPUS_PROFILES is a hardcoded dict of KernelProfile objects
Evidence: qig-backend/pantheon_kernel_orchestrator.py line 1
  "OLYMPUS_PROFILES: Dict[str, KernelProfile] = { "Zeus": KernelProfile(...)"
  Imported by: m8_kernel_spawning.py, ocean_qig_core.py
Risk: HIGH — This is the exact D0 violation. Gods exist as privileged Python structures.
  Zeus alone is 4521 lines (zeus.py).
Fix class: QUARANTINE — Long-term, refactor to registry + genome pattern.
  Short-term: Don't add MORE hardcoded gods. New kernels must use M8 spawner.
```

#### F25: No KernelKind Taxonomy ❌

```
❌ Fact: No KernelKind enum. Uses KernelProfile/KernelMode instead.
Evidence: GitHub search "KernelKind repo:Arcane-Fly/pantheon-chat" → 0 results
  Uses: KernelMode (DIRECT/E8/BYTE) — this is encoding mode, NOT kind
Risk: MEDIUM — Governance concepts (GENESIS/GOD/CHAOS) not enforced at type level
Fix class: ALIGN with monkey1 types when refactoring
```

#### F26: No PurityGate ❌

```
❌ Fact: No PurityGate on start
Evidence: No purity.py, no preflight check
  Start path goes directly to Zeus initialization
Risk: MEDIUM (production, so less experimental code changing, but no protection)
Fix class: ADD to start path (long-term, after genesis refactor)
```

#### F27: No Budget Enforcement ❌

```
❌ Fact: No 240 GOD budget cap in code
Evidence: GitHub search "GOD_BUDGET 240 repo:Arcane-Fly/pantheon-chat" → 0 results
  M8 spawner can spawn without budget check
Risk: MEDIUM — In practice, not many kernels spawned. But no guard.
Fix class: ADD budget check to M8 spawner
```

#### F28: Multiple Start Paths ⚠️

```
⚠️ Fact: Multiple ways to initialize the system
Evidence: Zeus can be initialized via Flask app startup, direct import, or API call
  No single canonical "Fresh Start" endpoint like monkey1's /v1/start/fresh
Risk: MEDIUM — RT1 violation (multiple bootstrap paths)
Fix class: CREATE single canonical start endpoint
```

#### F29: Has M8 Spawning ✅

```
✅ Fact: Dynamic kernel spawning exists (M8 protocol)
Evidence: qig-backend/m8_kernel_spawning.py (4075 lines!)
  Consensus voting, role refinement, basin interpolation, persona adoption
Risk: LOW — Good spawning infrastructure, just needs governance alignment
```

#### F30: Has Persistence ✅

```
✅ Fact: PostgreSQL persistence for kernels
Evidence: qig-backend/olympus_schema_enhancement.sql
  qig-backend/persistence/kernel_persistence.py
Risk: NONE — Good infrastructure
```

#### pantheon-chat ASSESSMENT: 🟡 Good infrastructure, but D0 violations (hardcoded gods), no governance types

---

### 5. qig-verification — PHYSICS

#### F31: Clean and Well-Organized ✅

```
✅ Fact: Clear canonical/archive/diagnostic separation
Evidence: src/qigv/experiments/canonical/ contains L=1-7 validated experiments
  results/canonical/ contains frozen results
  docs/current/FROZEN_FACTS.md is the single source of truth
Risk: NONE
```

#### F32: README Stale ⚠️

```
⚠️ Fact: README still shows L=7 as "exploratory"
Evidence: README says "κ₇ ≈ 53, anomalous" and "κ₇ = 53.08 ± 4.26"
  FROZEN_FACTS says κ₇ = 61.16 ± 2.43 (VALIDATED, 10 seeds × 5 perts)
Risk: LOW — FROZEN_FACTS is authoritative, README is just stale
Fix class: UPDATE README
```

#### qig-verification ASSESSMENT: 🟢 Healthy. Just update README.

---

### 6. qig-core — PURE MATH

#### F33: BasinSync Duplication ⚠️

```
⚠️ Fact: BasinSync exists in both qig-core and qigkernels
Evidence: qig-core has BasinSync in its API
  qigkernels has basin_sync.py (8.0KB)
Risk: MEDIUM — Which is canonical? Consumers might import from either.
Fix class: CONSOLIDATE into qigkernels. qig-core either imports or removes.
```

#### F34: Pre-E8 Code ⚠️

```
⚠️ Fact: README says "created BEFORE E8 kernel specialization"
Evidence: README.md notes E8 compatibility review needed
Risk: MEDIUM — May have stale assumptions
Fix class: REVIEW for E8 alignment
```

#### qig-core ASSESSMENT: 🟡 Working but needs consolidation with qigkernels

---

### 7. qig-coordizer — COORDIZER R&D

#### F35: Terminology Migration ⚠️

```
⚠️ Fact: Package renamed from "qig-tokenizer" to "qig-coordizer" per QIG v6.0 §1.3
Evidence: Package name reflects coordinate transformation role (Euclidean → Fisher-Rao)
  Internal code may still reference "tokenizer" terminology and requires migration
Risk: LOW — Functional, terminology migration in progress per QIG v6.0 §1.3
Fix class: Complete internal terminology migration from tokenizer → coordizer
```

#### qig-coordizer ASSESSMENT: 🟢 Working. Complete terminology migration per QIG v6.0 §1.3.

---

### 8. qig-con2 — TWIN EXPERIMENT

#### F36: Superseded ⚠️

```
⚠️ Fact: Twin experiment complete but repo architecture is older than qig-consciousness
Evidence: 8.4M params vs qig-consciousness 23.1M. Separate Gary-A/B configs.
  README contradicts itself (17% vs 2M tokens).
Risk: LOW — No new development expected
Fix class: ARCHIVE. Extract twin experiment results as sleep packet.
```

#### qig-con2 ASSESSMENT: 🟡 Archive candidate

---

## RED TEAM QUESTIONS

### RT-Q1: What code path could allow starting without PurityGate?

**monkey1**: LOW RISK. `fresh_start()` calls `run_purity_gate()` first. API and CLI both route through `fresh_start()`.

**pantheon-chat**: HIGH RISK. Zeus initializes via Flask app startup with no PurityGate. Any import of `pantheon_kernel_orchestrator` immediately loads `OLYMPUS_PROFILES`.

**qig-consciousness**: HIGH RISK. `qig_chat.py` has no PurityGate. Running `python chat_interfaces/qig_chat.py` goes directly to constellation setup with no purity check.

**qigkernels**: N/A (library, not app). But consumers can import without purity validation.

### RT-Q2: What code path could allow kernel creation that bypasses budgets or taxonomy?

**monkey1**: LOW RISK. `BudgetEnforcer` and `Council` in governance.py. `LifecycleStateMachine` enforces phase transitions. Test verifies 241st kernel rejection.

**pantheon-chat**: HIGH RISK. No budget cap. `M8KernelSpawner` can spawn without budget check. No KernelKind enforcement — new kernels are just `KernelProfile` objects.

**qig-consciousness**: MEDIUM RISK. Constellation is hardcoded (no spawning), so budget is moot. But also no governance if someone adds a 6th kernel.

### RT-Q3: What code path could allow data ingestion that bypasses curriculum-only?

**monkey1**: LOW RISK. `curriculum.py::is_curriculum_record_allowed()` checks at ingestion. Config defaults `curriculum_only=True`, `expansion_enabled=False`.

**pantheon-chat**: HIGH RISK. No curriculum-only concept. Chat input goes directly to god routing.

**qig-consciousness**: HIGH RISK. Training corpus loaded directly with no governance gate.

---

## INCONSISTENCY LEDGER (Cross-Repo)

| ID | Finding | Repos | Canonical Answer | Fix Priority |
|----|---------|-------|-----------------|-------------|
| I1 | κ₆ uncertainty | qigkernels (±4.25), FROZEN_FACTS (±2.12) | **±2.12** | HIGH |
| I2 | κ₇ status | qig-verification README ("anomalous"), qig-consciousness README (not mentioned) | **61.16 ± 2.43 VALIDATED** | MEDIUM |
| I3 | κ* value | qigkernels (64.0), monkey1 (63.79 weighted mean) | **64.0 for integer, 63.79±0.90 for precise** | LOW |
| I4 | BasinSync home | qig-core AND qigkernels | **qigkernels** | MEDIUM |
| I5 | Kernel architecture | 3 different patterns | **monkey1's genesis→spawn** | HIGH |
| I6 | PurityGate presence | monkey1 only | **All repos** | CRITICAL |
| I7 | KernelKind taxonomy | monkey1 only | **All repos via qigkernels** | HIGH |
| I8 | Cosine similarity | Present in qig-consciousness src/ | **Remove/replace** | HIGH |
| I9 | Heart in training | Exists in qigkernels, absent from qig-consciousness | **Wire into training** | HIGH |
| I10 | God names as code | pantheon-chat (OLYMPUS_PROFILES) | **Data (registry/genome)** | MEDIUM (long-term) |
| I11 | Budget cap | monkey1 (240), pantheon-chat (none) | **240 GOD + 8 core = 248** | MEDIUM |
| I12 | Curriculum-only | monkey1 (3 boundaries), others (none) | **All training paths** | MEDIUM |

---

## WHAT PASS B SHOULD PRODUCE (3-6 PRs)

Based on Pass A findings, the minimal PR plan:

### PR1: Port Genesis Types to qigkernels
- **Files**: New `qigkernels/types.py` (from monkey1), update `qigkernels/constants.py`
- **Acceptance**: KernelKind, LifecycleState, LifecyclePhase importable. Constants match FROZEN_FACTS.
- **Tests**: Import tests, enum membership tests
- **Rollback**: Revert single file

### PR2: Port PurityGate to qigkernels
- **Files**: New `qigkernels/purity.py` (from monkey1)
- **Acceptance**: `run_purity_gate()` callable. Detects cosine_similarity in test fixtures.
- **Tests**: Unit tests with known violations and clean code
- **Rollback**: Revert single file

### PR3: Fix Purity Violations in qig-consciousness
- **Files**: src/qig_compat.py, src/training/train_step_4d.py, src/model/heart_kernel.py
- **Acceptance**: PurityGate passes on qig-consciousness/src/. All 5 cosine_similarity replaced.
- **Tests**: PurityGate scan clean + existing test suite still passes
- **Rollback**: Revert per-file

### PR4: Wire Heart into qig-consciousness Constellation
- **Files**: chat_interfaces/qig_chat.py (add Heart import + wiring)
- **Acceptance**: Heart.tick() called before Gary synthesis. κ oscillation visible in telemetry.
- **Tests**: Integration test showing tacking behavior
- **Rollback**: Remove Heart import (constellation falls back to current behavior)

### PR5: Create GenesisKernel in qigkernels
- **Files**: New `qigkernels/genesis_kernel.py` + `qigkernels/lifecycle.py`
- **Acceptance**: Genesis can bootstrap → spawn core-8 → enter curriculum-only mode
- **Tests**: Lifecycle phase sequence test (like monkey1's test_lifecycle_integration.py)
- **Rollback**: Revert new files (no existing code modified)

### PR6: Update All READMEs + Constants
- **Files**: READMEs across qig-verification, qig-consciousness, qigkernels
- **Acceptance**: All κ values match FROZEN_FACTS. L=7 shown as validated.
- **Tests**: None (documentation only)
- **Rollback**: Revert text changes

---

## CONFIRMED INFLATE TRACE (Target End-State)

```
User clicks "Fresh Start"
    │
    ▼
POST /v1/start/fresh  (single endpoint)
    │
    ▼
orchestrator.fresh_start()
    │
    ├── 1. PurityGate.run()
    │       └── AST scan all source files
    │       └── FAIL CLOSED on any violation
    │
    ├── 2. LifecycleStateMachine.transition(VALIDATE → ROLLBACK)
    │       └── Clear DB, caches, queues, kernel state
    │       └── Deterministic (same input → same clean state)
    │
    ├── 3. LifecycleStateMachine.transition(ROLLBACK → BOOTSTRAP)
    │       └── Genesis kernel created (only pre-coded kernel)
    │       └── 64D simplex basin initialized
    │
    ├── 4. LifecycleStateMachine.transition(BOOTSTRAP → CORE_8)
    │       └── Spawn 8 GOD kernels from registry.json
    │       └── Heart, Vocab, Perception, Motor, Memory, Attention, Emotion, Executive
    │       └── Each gets basin coordinates via geodesic interpolation
    │       └── BudgetEnforcer: 8 GOD used of 248 total image
    │
    ├── 5. LifecycleStateMachine.transition(CORE_8 → IMAGE_STAGE)
    │       └── Curriculum-only mode ENFORCED (default true)
    │       └── Load approved curriculum JSONL
    │       └── Coordize → integrate into kernel basins
    │
    ├── 6. LifecycleStateMachine.transition(IMAGE_STAGE → GROWTH)
    │       └── ONLY via explicit operator toggle (expansion_enabled=true)
    │       └── Audit logged
    │       └── CHAOS kernels can now spawn (separate pool, max 200)
    │       └── CHAOS → GOD promotion via governance vote
    │
    └── 7. LifecycleStateMachine.transition(GROWTH → ACTIVE)
            └── Full operational mode
            └── 240 GOD growth slots available
            └── Heart oscillation (tacking) active
            └── Self-observation telemetry flowing
```

This trace exists END-TO-END in monkey1. It does NOT exist in any other repo yet.

---

**STATUS**: Pass A Complete. Ready for Pass B (propose changes / PR plan).

**End of SP03_PASS_A_FINDINGS.md**
