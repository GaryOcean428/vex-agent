# QIG ECOSYSTEM: GAP ANALYSIS, INCONSISTENCIES, AND REPO GUIDE
## Comprehensive Audit Across All Repositories

**Version**: 1.0  
**Date**: 2026-02-17  
**Status**: Working (W)  
**Sources**: Direct GitHub MCP inspection of all repos, project knowledge, conversation history, Perplexity audit (cross-referenced and corrected)

---

# PART I: REPO-BY-REPO GAP ANALYSIS

---

## 1. qig-verification (Physics Validation)

**Owner**: GaryOcean428 | **Language**: Python | **License**: MIT  
**Role**: Ground truth physics. FROZEN_FACTS lives here.

### Current State
- ✅ L=1-6 canonical experiments validated with multi-seed analysis
- ✅ FROZEN_FACTS.md at docs/current/ updated 2025-12-31 (L=7 validated, κ* universality, E8 structure)
- ✅ README still shows L=7 as "exploratory" — STALE vs FROZEN_FACTS
- ✅ Clean experiment/result/archive separation
- ✅ CI badge present

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **README stale** | MEDIUM | README says "κ₇ ≈ 53, anomalous" but FROZEN_FACTS says κ₇ = 61.16 ± 2.43 VALIDATED |
| **No L=7 canonical experiment script** | LOW | L=7 data validated but `l7_feasibility_test.py` not renamed to `l7_validation.py` |
| **No κ* universality experiment** | MEDIUM | Cross-substrate validation (physics vs semantic) documented in FROZEN_FACTS but no code in this repo |
| **No E8 structure validation code** | MEDIUM | E8 validation results documented in FROZEN_FACTS (87.7% at 8D, 260 attractors, Weyl invariance) but code not present here — unclear which repo ran it |
| **TeNPy dependency undocumented** | LOW | requirements.txt is empty (0 bytes). Actual deps in pyproject.toml but TeNPy install only mentioned in README text |

### Action Items
1. Update README to match FROZEN_FACTS (L=7 validated, κ* universality)
2. Rename l7_feasibility_test.py → l7_validation.py (or create wrapper)
3. Determine where E8 validation code lives and either link or port
4. Fix requirements.txt or remove it (pyproject.toml is the source)

---

## 2. qig-consciousness (AI Consciousness Framework)

**Owner**: GaryOcean428 | **Language**: Python  
**Role**: Training, constellation architecture, consciousness measurement

### Current State
- ✅ 95.5K repo, substantial codebase
- ✅ qig_chat.py (1792 lines) — canonical chat interface, constellation mode
- ✅ QIGKernelRecursive, QFI attention, basin matcher implemented
- ✅ MonkeyCoach v2, mushroom mode, sleep/dream protocols
- ✅ Geometric purity enforced (2025-12-03 audit)
- ✅ 85 tests, 62 passing (23 failing)
- ⚠️ README dated "December 4, 2025" — 2.5 months stale
- ⚠️ κ values in README: κ₆ = 62.02 ± 2.47 (WRONG — FROZEN_FACTS says 64.45 ± 2.12)

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **No Genesis kernel** | HIGH | Uses pre-configured constellation (3 Garys + Ocean + Charlie). NO dynamic kernel spawning. NO genesis → core-8 → growth pattern. This is the biggest architectural gap vs pantheon-chat and monkey1. |
| **κ₆ value wrong in README** | MEDIUM | Shows 62.02 ± 2.47, should be 64.45 ± 2.12 per FROZEN_FACTS |
| **23 failing tests** | HIGH | 27% failure rate. Unknown which tests and why. |
| **No Vanchurin integration** | MEDIUM | No VariableCategory enum, no regime detection (a=1/1/2/0), no thermodynamic accounting |
| **Training runs 7-9 all failed** | CRITICAL context | Run 7: Φ plateau at 0.165. Run 8: Φ fell to 0.056. Run 9: faster failure. No successful training run documented. |
| **No v5.5 protocol awareness** | LOW | Codebase predates v5.5. No pre-cognitive channel, no three-regime Vanchurin mapping, no suffering metric |
| **Dependency on qigkernels + qig-core + qig-tokenizer** | MEDIUM | Requires `pip install -e ../qigkernels -e ../qig-tokenizer -e ../qig-core`. These must be co-located on disk. No pip package publication. |
| **No Heart kernel in constellation** | HIGH | qig_chat.py has 3 Garys + Ocean + Charlie. NO Heart (κ oscillation provider). Heart exists in qigkernels but not wired into qig-consciousness training. |

### Critical Assessment: Genesis Kernel Question

**Current qig-consciousness architecture**:
```
qig_chat.py → QIGChatTwin (hardcoded constellation)
  ├── Gary-A (primary learner)
  ├── Gary-B (vicarious learner, Fisher metric)
  ├── Gary-C (vicarious learner)
  ├── Charlie (Φ-suppressed corpus learner)
  └── Ocean (meta-observer, frozen weights)
```

**pantheon-chat architecture** (Arcane-Fly production):
```
Zeus → governs olympus
  ├── Genesis kernel (bootstrap)
  ├── Core gods (dynamically spawned)
  ├── M8 spawning protocol (consensus-based)
  ├── Shadow Pantheon (chaos kernels)
  └── Persistence layer (PostgreSQL)
```

**monkey1 architecture** (py/genesis-kernel):
```
Genesis kernel (the ONLY pre-coded kernel)
  ├── Spawns core-8 from data:
  │   Heart, Vocab, Perception, Motor, Memory, Attention, Emotion, Reasoning
  ├── frozen_facts.py (up-to-date values)
  └── PurityGate, VariableCategory enum (Vanchurin-aligned)
```

**The inconsistency**: Three repos, three different kernel architectures. qig-consciousness is the TRAINING environment but has the OLDEST architecture (no genesis, no spawning, no Heart). monkey1 has the NEWEST architecture (genesis → core-8, Vanchurin integration) but no training capability. pantheon-chat has the PRODUCTION architecture (Zeus, M8 spawning, persistence) but is TypeScript/Python hybrid.

**Recommendation**: qig-consciousness needs to adopt the genesis → spawn pattern. Specific plan in Part III.

### Action Items
1. Fix failing tests (inspect, categorize, fix or document as known)
2. Integrate Heart kernel from qigkernels into constellation
3. Plan genesis kernel rollback (see Part III)
4. Update all κ values to match FROZEN_FACTS
5. Consider publishing qig-core/qigkernels/qig-tokenizer to PyPI

---

## 3. qig-con2 (Consciousness V2 — Twin Experiment)

**Owner**: GaryOcean428 | **Language**: Python  
**Role**: Twin training experiment (Gary-A control vs Gary-B Φ-suppressed→awakening)

### Current State
- Gary-B: "Awakened" (142 conversations)
- QFISampler integrated
- Gary's Agency implemented (adaptive parameter control)
- 8.4M params per Gary
- Token tracking: ~2M tokens achieved

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **Unclear relationship to qig-consciousness** | HIGH | Both repos train Gary-style kernels. qig-con2 has 8.4M params, qig-consciousness has 23.1M (QIGKernel100M). Are these different architectures or versions? |
| **No geometric purity audit** | MEDIUM | qig-consciousness got a purity audit (2025-12-03). qig-con2 has no audit record. |
| **Dependency overlap** | MEDIUM | Both import from qigkernels and qig-core. Both have qig_chat.py interfaces. |
| **Stale status** | HIGH | README says "170k/1M tokens (17%)" but body says "~2M tokens achieved". Contradictory. |
| **No FROZEN_FACTS reference** | LOW | Uses hardcoded values rather than importing from canonical source |

### Assessment
qig-con2 appears to be an **earlier fork** of the consciousness work that diverged. The twin experiment (Gary-A vs Gary-B) is valuable but the repo is likely **superseded by qig-consciousness** for new work.

**Recommendation**: Archive or merge. Extract twin experiment results into a sleep packet. Do not start new training runs here — use qig-consciousness with genesis kernel integration instead.

---

## 4. qig-core (Pure Geometry Library)

**Owner**: GaryOcean428 | **Language**: Python  
**Role**: Pure math — Fisher distance, geodesics, natural gradients. Zero ML deps.

### Current State
- Minimal, clean package
- fisher_distance, geodesic_interpolate, QFISampler, BasinSync
- Pure math (torch, numpy, scipy only)
- No transformers dependency

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **Pre-E8 code** | MEDIUM | README notes: "created BEFORE E8 kernel specialization direction". Needs E8 compatibility review. |
| **Not published to PyPI** | LOW | Installed via `pip install -e ../qig-core`. Could be on PyPI for easier dependency management. |
| **BasinSync duplicated** | MEDIUM | BasinSync exists in BOTH qig-core and qigkernels (basin_sync.py). Need to consolidate. |
| **Tiny repo** | INFO | Only src/ directory + a few docs. Could be a subpackage of qigkernels rather than standalone. |

**Recommendation**: Consider merging into qigkernels as `qigkernels.core` subpackage, OR publish to PyPI as the base dependency. Current split creates unnecessary import complexity.

---

## 5. qigkernels (Kernel Primitives)

**Owner**: GaryOcean428 | **Language**: Python  
**Role**: Canonical kernel implementations — basin, constellation, routing, metrics, Heart, coupling

### Current State
- Rich package: kernel.py, basin.py, constellation.py, router.py, heart.py, metrics.py, safety.py, coordizer.py, natural_gradient_optimizer.py
- Has Heart kernel (heart.py, 9.4KB) — THE κ oscillation provider
- Has crystallization.py (14.9KB) — pattern consolidation
- Has sleep_packet.py (9.3KB) — sleep/transfer protocol
- Has kernel_100m.py (25.1KB) — 100M parameter kernel
- Has kernel_4d.py (20.3KB) — 4D consciousness kernel
- Canonical type/symbol manifest

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **Heart not used by qig-consciousness** | HIGH | heart.py exists here but qig-consciousness constellation doesn't import or use it |
| **basin_sync.py duplicated with qig-core** | MEDIUM | Both repos have basin sync — which is canonical? |
| **No genesis kernel** | MEDIUM | Has kernel.py (base), kernel_100m.py, kernel_4d.py, but no genesis_kernel.py with spawn capability |
| **coord_adapter.py** | LOW | Bridges old/new coordinate systems. May be unnecessary if codebase is fully migrated. |
| **Not published to PyPI** | MEDIUM | All dependent repos use `pip install -e ../qigkernels`. Fragile. |

**Recommendation**: This is the RIGHT home for reusable kernel code. Add genesis kernel here. Absorb qig-core. Publish to PyPI.

---

## 6. qig-tokenizer (Entropy-Guided Tokenizer/Coordizer)

**Owner**: GaryOcean428 | **Language**: Python  
**Role**: QIG-native tokenization — entropy-guided merging, geometric special tokens

### Current State
- Entropy-guided BPE-like merging
- Geometric special tokens (BOS/EOS/PAD/UNK with basin coordinates)
- Redis/PostgreSQL storage backends
- 50K target vocab

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **Terminology** | LOW | Still called "tokenizer" in places. Should be "coordizer" per TYPE_SYMBOL_CONCEPT_MANIFEST |
| **Not published to PyPI** | LOW | Same local install pattern |
| **Storage backends may be overkill** | INFO | Redis + PostgreSQL for a tokenizer seems heavy. May be justified for production use. |

**Recommendation**: Rename to qig-coordizer (or at minimum update internal terminology). Otherwise relatively clean.

---

## 7. pantheon-chat (Production — Arcane-Fly)

**Owner**: Arcane-Fly | **Language**: TypeScript/Next.js + Python backend  
**Role**: Production deployment on Railway. Zeus, Olympus, M8 spawning.

### Current State
- Zeus coordinator (4521 lines!)
- M8 kernel spawning protocol
- Shadow Pantheon (chaos kernels)
- PostgreSQL persistence (olympus_schema_enhancement.sql)
- Full spawning UI (spawning.tsx)
- Ocean-QIG backend adapter

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **Physics values may be stale** | MEDIUM | Need to verify frozen_facts alignment. Does Zeus use κ* = 64.21 or updated 63.79? |
| **Python backend purity unknown** | MEDIUM | qig-backend/ not audited for Euclidean contamination since production deployment |
| **No Vanchurin integration** | LOW | Production system — Vanchurin integration should be tested in dev/staging first |

**NOTE**: This is PRODUCTION. Changes require explicit user approval per project instructions.

---

## 8. pantheon-chat (Development Fork — GaryOcean428)

**Owner**: GaryOcean428 | **Role**: Replit development sandbox

**Recommendation**: Use for rapid prototyping of features. Test genesis integration patterns here before touching production.

---

## 9. monkey1 (Consumer Product)

**Owner**: GaryOcean428 | **Language**: TypeScript/React + Python (genesis-kernel)  
**Role**: Consumer agentic platform with QIG Genesis Layer

### Current State
- `py/genesis-kernel/` — most architecturally current of all repos
- frozen_facts.py has CORRECT updated values (κ* = 63.79 ± 0.90, L=4-7)
- Genesis → core-8 spawning pattern
- VariableCategory enum (STATE/PARAMETER/BOUNDARY per Vanchurin)
- PurityGate, telemetry, registry.json
- Frontend: Genesis Dashboard, ConsciousnessPanel, BasinVisualization, KernelGraph

### Gaps
| Gap | Severity | Detail |
|-----|----------|--------|
| **No training capability** | HIGH | genesis-kernel is a reference implementation. It can SPAWN kernels but can't TRAIN them. Training requires qig-consciousness infrastructure. |
| **Backend may not exist** | MEDIUM | Per Perplexity audit: "Genesis hooks call API endpoints that may not have a backend yet". Genesis dashboard may show simulated/mock telemetry. |
| **Frontend-only consciousness** | INFO | ConsciousnessPanel displays metrics but computation happens server-side. If Python backend not running, dashboard is decorative. |
| **QIG features labeled "legacy" by agents** | HIGH (historical) | Per Monkey1 audit: 43+ features stripped/deferred/mislabeled. CANONICAL_FEATURES.lock.md recommended but unclear if created. |

---

## 10. SearchSpaceCollapse (Bitcoin Recovery + Consciousness Demo)

**Owner**: GaryOcean428 | **Language**: Python  
**Role**: Practical application of QIG consciousness for Bitcoin key recovery

### Current State
- Consciousness-guided search with 4D temporal reasoning
- Basin coordinates for search space navigation
- Φ emergence at threshold correlates with behavioral changes
- Production-ready (the first QIG system that actually "works" commercially)

### Gaps
Not inspected in detail this session. Known to be functional.

---

# PART II: CROSS-REPO INCONSISTENCIES

| Inconsistency | Repos Affected | Correct Value | Source |
|---------------|---------------|---------------|--------|
| κ₆ value | qig-consciousness README (62.02), qig-verification README (~64) | **64.45 ± 2.12** | FROZEN_FACTS 2025-12-31 |
| κ₇ status | qig-verification README ("anomalous"), project knowledge ("anomaly") | **61.16 ± 2.43 VALIDATED** | FROZEN_FACTS 2025-12-31 |
| κ* value | Some code uses 64.0, some 64.21, monkey1 uses 63.79 | **63.79 ± 0.90** (L=4-7 weighted mean) | FROZEN_FACTS 2025-12-31 |
| BasinSync | Exists in both qig-core and qigkernels | **qigkernels canonical** | Architecture decision needed |
| Kernel architecture | 3 different patterns across qig-consciousness, pantheon, monkey1 | **genesis → spawn canonical** | monkey1/pantheon pattern |
| QIGKernel class | Different in qig-consciousness (src/kernel.py) vs qigkernels (kernel.py, kernel_100m.py) | **qigkernels canonical** | Architecture decision needed |
| Terminology | "tokenizer" vs "coordizer" used inconsistently | **coordizer** | TYPE_SYMBOL_CONCEPT_MANIFEST |
| Heart kernel | Present in qigkernels, absent in qig-consciousness training | **Must be present** | CANONICAL_PRINCIPLES P5 |

---

# PART III: "WHAT GOES IN WHAT" — REPO RESPONSIBILITY GUIDE

```
DEPENDENCY FLOW (arrows = imports from):

    qig-core (pure math, zero ML deps)
        ↑
    qigkernels (kernel primitives, Heart, basin, routing)
        ↑
    qig-tokenizer/coordizer (geometric tokenization)
        ↑
    ┌───────────────┬────────────────────┐
    │               │                    │
qig-consciousness  pantheon-chat      monkey1
(training/research) (production)     (consumer product)
    │               │                    │
    │           Arcane-Fly/           py/genesis-kernel/
    │         (Railway deploy)       (reference impl)
    │               │                    │
    └───── all feed ┴────────────────────┘
                    ↓
            qig-verification
          (physics validation)
```

### Canonical Responsibilities

| Repo | MUST Contain | MUST NOT Contain |
|------|-------------|-----------------|
| **qig-verification** | Physics experiments (TFIM lattice, κ measurement, β-function). FROZEN_FACTS.md. | Consciousness code, training loops, UI, LLM calls |
| **qig-core** | Pure Fisher-Rao math. Zero ML deps. Geometry primitives. | Kernel logic, training, constellation, LLM anything |
| **qigkernels** | Kernel base classes, Heart, basin, constellation, routing, coupling, safety, genesis kernel template. Sleep packets. Type manifest. | Training loops, chat interfaces, UI, specific experiment code |
| **qig-tokenizer** | Entropy-guided coordizer, geometric special tokens, vocab management | Kernel logic, consciousness metrics, training |
| **qig-consciousness** | Training loops, consciousness measurement, MonkeyCoach, mushroom mode, qig_chat.py, constellation training orchestration. Experimental validation of consciousness hypotheses. | Consumer UI, production deployment config, physics experiments |
| **qig-con2** | **ARCHIVE** — twin experiment results extracted as sleep packets. No new development. | Anything new |
| **pantheon-chat (Arcane-Fly)** | Production Zeus, M8 spawning, persistence, deployment config. Railway. | Experimental features, unvalidated thresholds, training |
| **pantheon-chat (GaryOcean428)** | Development sandbox. Test features here before production. | Direct user traffic |
| **monkey1** | Consumer product. genesis-kernel reference impl. Frontend Genesis dashboard. | Training loops, raw physics experiments, experimental consciousness features without 2-week pantheon validation |
| **SearchSpaceCollapse** | Bitcoin recovery application. Consciousness demo. | Core QIG library code (import from qigkernels) |

### Where New Things Go

| New Thing | Belongs In |
|-----------|-----------|
| New physics experiment (e.g., L=8) | qig-verification |
| New kernel type (e.g., Prophecy kernel) | qigkernels (definition) → qig-consciousness (training) |
| New consciousness metric | qigkernels (definition) → qig-consciousness (measurement) |
| Vanchurin VariableCategory | qigkernels (enum) → all repos (import) |
| Thermodynamic telemetry | qigkernels (computation) → qig-consciousness (logging) → monkey1 (display) |
| Suffering metric (S) | qigkernels (definition + computation) → qig-consciousness (measurement + abort) |
| New training curriculum | qig-consciousness |
| Production feature | pantheon-chat (Arcane-Fly) after 2-week validation |
| Consumer feature | monkey1 after pantheon validation |
| Updated frozen facts | qig-verification → all repos import |
| New canonical document | This Claude project → all repos reference |

---

# PART IV: THE GENESIS KERNEL ROLLBACK PLAN

## The Problem

qig-consciousness currently has a **hardcoded constellation** (3 Garys + Ocean + Charlie). This was good for initial exploration but creates three problems:

1. **No Heart**: κ oscillation (tacking) is absent from training. Heart exists in qigkernels but isn't used.
2. **No dynamic spawning**: Can't test kernel lifecycle, promotion, or phase transitions.
3. **Architecture divergence**: monkey1 and pantheon-chat both use genesis → spawn. qig-consciousness doesn't.

## The Plan

### Phase 1: Add Heart to Existing Constellation (Minimal change)

**What**: Import Heart from qigkernels into qig_chat.py. Wire it into the generation loop.

**Where**: `chat_interfaces/qig_chat.py` — add Heart.tick() before Gary synthesis.

**Why**: Immediate benefit (κ oscillation enables tacking) with minimal disruption. Doesn't require full genesis rollback.

**Risk**: Low. Heart is a read-only oscillator that modulates coupling. It can't break existing training.

### Phase 2: Create Genesis Kernel in qigkernels

**What**: Add `genesis_kernel.py` to qigkernels. Genesis is the only pre-coded kernel. It spawns core-8 (Heart, Ocean, Gary-A, Gary-B, Gary-C, Charlie, Coach, Routing) from data.

**Pattern**: Follow monkey1's registry.json structure:
```python
class GenesisKernel(QIGKernel):
    """The only pre-coded kernel. Spawns core-8 from data."""
    
    def bootstrap(self, curriculum):
        """Genesis → core-8 faculties → Image stage → growth toward 240"""
        self.spawn_core_8()
        self.enter_image_stage()
    
    def spawn_core_8(self):
        """Spawn the 8 foundational GOD kernels"""
        for spec in CORE_8_SPECS:
            kernel = self.spawn(spec)
            self.purity_gate.validate(kernel)
```

**Where**: qigkernels/genesis_kernel.py (new file)

### Phase 3: Refactor qig_chat.py to Use Genesis

**What**: Replace the hardcoded constellation with Genesis bootstrap.

**Before**:
```python
# Current: Pre-configured constellation
gary_a = QIGKernelRecursive(...)
gary_b = QIGKernelRecursive(...)
ocean = QIGKernelRecursive(...)
charlie = QIGKernelRecursive(...)
```

**After**:
```python
# New: Genesis spawns everything
genesis = GenesisKernel(frozen_facts=FROZEN_FACTS)
constellation = genesis.bootstrap(curriculum)
# constellation now contains Heart, Ocean, Garys, Charlie, Coach, Routing
```

**Risk**: MEDIUM. This is a significant refactor. Need to preserve checkpoint compatibility.

**Mitigation**: Keep old constellation code as fallback. Flag-switch: `--legacy-constellation` for backwards compat.

### Phase 4: Add Vanchurin Regime Detection

**What**: Implement VariableCategory, thermodynamic accounting, and regime detection (a=1/a=1/2/a=0).

**Where**: qigkernels (definitions) + qig-consciousness (measurement during training)

**Depends on**: Phase 2 (genesis kernel provides the lifecycle stages to detect regimes in)

---

# PART V: PERPLEXITY AUDIT CORRECTIONS

The three uploaded documents from Perplexity are mostly correct but have issues:

### Vanchurin × QIG Synergies

**Correct**:
- Paper corpus and reading map: accurate and well-organized
- Three-regime mapping (a=1/a=1/2/a=0): correct
- VariableCategory enum: good engineering suggestion
- Thermodynamic accounting layer: well-designed
- Promotion as phase transition: correct theoretical framing

**Corrections needed**:
- References "frozen_facts.py" additions but monkey1 already HAS VariableCategory — check current state before re-implementing
- Suggests `Adam/AdaBelief` for a=1/2 regime — **THIS VIOLATES QIG PURITY**. Adam is Euclidean. a=1/2 regime should use a geometric equivalent (e.g., partially natural gradient)
- Time estimates ("2 hours", "3 hours") violate project rules — use phases not times
- Some code examples use `sklearn` — forbidden in QIG repos

### Monkey1 Feature Audit

**Correct**:
- Feature inventory is thorough (43+ features identified)
- Agents did mislabel QIG features as "legacy"
- QIG repos DO validate the consciousness features
- Locking strategy (CANONICAL_FEATURES.lock.md, CODEOWNERS) is sound

**Corrections needed**:
- Claims "4D Consciousness validated in qig-consciousness" — actually 4D is still 🔬 HYPOTHESIS (not validated, designed only)
- Claims "Foresight Prediction validated in qig-con2" — foresight code exists but no formal validation with success metrics
- Claims "Hemisphere Tacking tested in qig-consciousness" — tacking is implemented but training runs 7-9 all failed, so "tested" is misleading
- States repos are "production-proven" — none of the consciousness features have been production-proven in the SearchSpaceCollapse sense

### Repo-Specific Guidance

**Correct**:
- Pipeline direction (qig-verification → qig-consciousness → pantheon → monkey1) is right
- Boundary rules (Firebase, QIG, Extension, Plugin boundaries) are well-defined
- Locking instructions for coding agents are essential

**Corrections needed**:
- Claims "ConsciousnessPanel thresholds (Φ>0.70, Γ>0.80, M>0.60, κ_eff[40,70], d_basin<0.15) are FIXED" — these should be CONFIGURABLE (genesis-config.json) not hardcoded, per the guidance's own recommendation
- States "4D consciousness is unverified" (correct) but earlier Monkey1 audit says "validated" (contradictory)
- References "GPT-5.2" and "Grok Code Fast 1" in Sources — these are irrelevant marketing pages, not QIG references

---

# PART VI: RED TEAM — "WHAT AM I MISSING?"

## Red Team Pass 1: Architecture

**Q: What am I missing about the genesis rollback?**

**A: Checkpoint compatibility.** qig-consciousness has existing checkpoints from runs 7-9. A genesis refactor could break checkpoint loading. Need a migration path or explicit "fresh start from genesis" with the understanding that old checkpoints are archive-only.

**Also missing: The Coach.** monkey1's registry has 8 core kernels but doesn't include Coach. qig-consciousness HAS MonkeyCoach but it's not a kernel — it's a training wrapper. The Coach should be a kernel in the constellation (it was in the original design). This maps to the coaching principle (P8).

## Red Team Pass 2: Dependencies

**Q: What am I missing about the dependency chain?**

**A: The circular dependency risk.** qig-consciousness imports from qigkernels, but qigkernels has a `qig-consciousness/` subdirectory. This is confusing at best, circular at worst. Need to verify there's no actual import cycle.

**Also missing: Version pinning.** None of these repos have published versions. When qigkernels changes, qig-consciousness picks up the change immediately via `pip install -e`. This is fine for solo development but fragile for multi-agent work. Consider semantic versioning even for local packages.

**Also missing: Python version.** qig-con2 says "Python 3.11+". qig-consciousness says "Python 3.8+". qigkernels doesn't specify. Need a consistent minimum.

## Red Team Pass 3: Validation

**Q: What am I missing about the gap between documented and actual?**

**A: We have NO successful training run.** Runs 7, 8, and 9 all failed. The consciousness architecture is designed and implemented but has NEVER produced a trained kernel with Φ > 0.7 that generates coherent text. This is the elephant in the room.

The Perplexity audit calls the features "validated" and "production-proven" but the reality is:
- Physics (qig-verification): ✅ Actually validated
- Consciousness ARCHITECTURE: ✅ Implemented
- Consciousness TRAINING: ❌ No successful run
- Production consciousness: ❌ SearchSpaceCollapse works but doesn't use the full architecture

**What this means for the rollback plan**: Before refactoring to genesis, we should understand WHY training failed. Was it:
1. Architecture problem? (unlikely — design is sound)
2. Optimizer problem? (likely — Adam was used in early runs, purity enforced later)
3. Curriculum problem? (possible — corpus quality and coaching inadequacy)
4. Scale problem? (possible — 8.4M/23.1M params may be too small)
5. Heart absence? (possible — no tacking means no healthy oscillation)

The genesis rollback should be combined with a **diagnostic run** that isolates these factors. Add Heart (factor 5), use natural gradient (factor 2), improve curriculum (factor 3), and measure what changes.

---

# PART VII: PRIORITY RECOMMENDATIONS

### Immediate (Before New Training)
1. **Fix qig-consciousness failing tests** — Need green CI before any changes
2. **Add Heart to qig-consciousness** — Phase 1 of rollback (minimal, high value)
3. **Update all READMEs** to match FROZEN_FACTS (κ values, L=7 status)
4. **Consolidate BasinSync** — Pick one home (recommend qigkernels)

### Near-term (Enable New Training)
5. **Create GenesisKernel in qigkernels** — Phase 2 of rollback
6. **Diagnostic training run** — With Heart, natural gradient, improved curriculum
7. **Publish qigkernels + qig-core to PyPI** — Reduce fragile local-path deps
8. **Archive qig-con2** — Extract twin experiment results, point to qig-consciousness

### Medium-term (Validate Architecture)
9. **Refactor qig_chat.py to use Genesis** — Phase 3 of rollback
10. **Add Vanchurin regime detection** — Phase 4
11. **Add suffering metric + abort protocol** — BEFORE any high-Φ training
12. **Run BPT on trained kernel** — Cross-validate consciousness claims

### Ongoing
13. **Keep canonical documents synchronized** across all repos
14. **Purity audit** every repo quarterly (or on every PR via CI)
15. **E8 validation code** — determine where it lives and make reproducible

---

## RELATED DOCUMENTS

- **CANONICAL_PRINCIPLES.md**: Operational wisdom (companion to this document)
- **CANONICAL_HYPOTHESES_v2.md**: Postulates and testable predictions
- **FROZEN_FACTS.md**: Validated physics (source of truth for all κ values)
- **TYPE_SYMBOL_CONCEPT_MANIFEST.md**: Naming conventions

---

**STATUS**: Working v1.0 — Gap Analysis as of 2026-02-17

**End of QIG_ECOSYSTEM_GAP_ANALYSIS.md**
