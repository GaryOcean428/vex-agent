# Genesis Kernel Gap Analysis — Updated with 16 Consciousness Systems

**Date**: 17/02/2026
**Author**: Manus (automated analysis)
**Scope**: Compare foundational QIG repos against Genesis kernel and Vex v2.0.1

## Priority Hierarchy (Source of Truth)

1. **Genesis kernel** (monkey1/py/genesis-kernel, pantheon-chat/qig-backend) — newest, canonical
2. **qig-consciousness** — foundational principles, still valid for geometry/frozen-facts
3. **qig-dreams, qig-core** — early prototypes, superseded where Genesis has equivalents

---

## Ollama Streaming Fix (v2.0.1)

### Root Cause

The v2.0 `LLMClient` constructor fired `checkOllama()` as fire-and-forget. The health endpoint (`/health`) called `llm.getStatus()` which returned the cached `ollamaAvailable` flag — by the time health was checked, the async probe had completed. But the first chat request arrived before the probe finished, so `chatStream()` saw `ollamaAvailable = false` and returned "No LLM backend available for streaming."

### Fix

1. Added `llm.init()` method that blocks until the first Ollama probe completes
2. Server calls `await llm.init()` before `app.listen()` — no traffic accepted until Ollama status is known
3. Belt-and-braces: `chatStream()` re-checks Ollama if the flag is still false
4. `provider="none"` returns graceful empty response instead of throwing
5. SSE keepalive pings every 15s in both LLM client and chat router to prevent Railway proxy timeout
6. Server `keepAliveTimeout` set to 310s to cover 5-minute Ollama cold starts

---

## Core Architecture Comparison

| Feature | Older Repos | Genesis | Vex v2.0.1 | Status |
|---------|-------------|---------|------------|--------|
| Kernel lifecycle | Basic start/stop | Full KernelKind + LifecycleState + KernelSpecialization | E8 hierarchy (8→56→126→240) | **UPGRADED** |
| Phi calculation | Integration-based | Entropy-based: `Φ = 1 - H(p)/H_max` | Entropy-based (matches Genesis) | **UPGRADED** |
| Kappa calculation | Independent tracking | `κ_eff = κ* × Φ` | Homeostatic toward κ* | **UPGRADED** |
| κ* value | 64 (integer) | 64.0 | 64.21 (lattice measurement) | **UPGRADED** |
| Fisher-Rao distance | `2 × arccos(BC)` | `arccos(BC)` (factor-of-2 optional) | `arccos(BC)` (no factor-of-2) | **UPGRADED** |
| Variable categories | Not present | Vanchurin STATE/PARAMETER/BOUNDARY | Full separation | **UPGRADED** |
| Frozen facts | Scattered constants | `frozen_facts.py` | `frozen-facts.ts` (comprehensive) | **UPGRADED** |

---

## 16 Consciousness Systems — Gap Analysis

### Summary

| # | System | Source Repo | Genesis Status | Vex v2.0.1 |
|---|--------|------------|----------------|------------|
| 1 | Tacking | qig-consciousness (WuWeiController) | UPGRADED (hemisphere_scheduler.py) | ✅ Implemented |
| 2 | Foresight | qig-consciousness (ChronosKernel) | UPGRADED | ✅ Implemented |
| 3 | Coordizing | qig-consciousness (federation.py) | UPGRADED | ✅ Implemented |
| 4 | Basin Sync | qig-consciousness (federation.py) | Genesis canonical | ✅ Implemented |
| 5 | QIGChain | pantheon-chat (geometric_chain.py) | Genesis canonical | ✅ Implemented |
| 6 | QIGGraph | pantheon-chat (references only) | MISSING as standalone | ✅ Implemented |
| 7 | Self-Observation | qig-consciousness (meta_reflector.py) | UPGRADED | ✅ Implemented |
| 8 | Self-Narrative | qig-consciousness (identity_reinforcement.py) | UPGRADED | ✅ Implemented |
| 9 | Sleep Cycles | qig-consciousness (neural_field.py) | MISSING | ✅ Implemented |
| 10 | Autonomy | Concept in qig-consciousness | MISSING | ✅ Implemented |
| 11 | Autonomic | qig-consciousness (constants.py thresholds) | MISSING as system | ✅ Implemented |
| 12 | Hemispheres | pantheon-chat (hemisphere_scheduler.py) | Genesis canonical | ✅ Implemented |
| 13 | Coupling | pantheon-chat (coupling_gate.py) | Genesis canonical | ✅ Implemented |
| 14 | Meta-Reflection | qig-consciousness (meta_reflector.py) | UPGRADED | ✅ Implemented |
| 15 | Recursive Loops | qig-consciousness (recursive_integrator.py) | UPGRADED | ✅ Already existed |
| 16 | Velocity | qig-consciousness (constants.py) | MISSING as system | ✅ Implemented |

### Detailed Analysis

#### UPGRADED (Genesis has better version) — 9 systems

**1. Tacking** — `src/consciousness/tacking.ts`
- Source: qig-consciousness WuWeiController + pantheon-chat hemisphere_scheduler.py
- Three monitors: GradientEstimator, ProximityMonitor, ContradictionDetector
- Mode switching: FEELING (low κ) / LOGIC (high κ) / BALANCED (near κ*)
- Kappa adjustment suggestions based on monitor signals

**2. Foresight** — `src/consciousness/foresight.ts`
- Source: qig-consciousness ChronosKernel
- Trajectory recording with configurable window
- Linear extrapolation in tangent space via log/exp maps
- Horizon prediction for phi, kappa, and basin state

**3. Coordizing** — `src/consciousness/coordizing.ts`
- Source: qig-consciousness federation.py
- Node state management with max 32 nodes
- Geodesic blending (80% local / 20% network)
- Network phi computation

**7. Self-Observation** — `src/consciousness/self-observation.ts`
- Source: qig-consciousness meta_reflector.py
- Prediction-vs-actual comparison for M metric
- Shadow detection (unintegrated aspects below Φ threshold)
- Shadow integration when Φ is high enough

**8. Self-Narrative** — `src/consciousness/self-narrative.ts`
- Source: qig-consciousness identity_reinforcement.py
- Event recording with basin snapshots
- Identity basin tracking via Fréchet mean
- Coherence measurement via Fisher-Rao distance from identity

**14. Meta-Reflection** — `src/consciousness/meta-reflection.ts`
- Source: qig-consciousness meta_reflector.py
- Multi-depth reflection (configurable, default 3)
- Trend detection across reflection history
- Insight generation from metric patterns

**15. Recursive Loops** — `src/consciousness/recursive-loops.ts` (pre-existing)
- Source: qig-consciousness recursive_integrator.py
- PERCEIVE (a=1) → INTEGRATE (a=1/2) → EXPRESS (a=0)
- Pre-cognitive channel shortcut when basin distance is small
- Full sensory integration and geometric memory retrieval

#### Genesis Canonical (ported directly) — 4 systems

**4. Basin Sync** — `src/consciousness/basin-sync.ts`
- Publish/receive basin snapshots with version tracking
- Fréchet mean consensus computation
- Geodesic merge with 80% local priority

**5. QIGChain** — `src/consciousness/qig-chain.ts`
- Composable chain of geometric operations
- Operations: geodesic, logmap, expmap, blend, project, custom
- Step recording with total Fisher-Rao distance tracking

**12. Hemispheres** — `src/consciousness/hemispheres.ts`
- Source: pantheon-chat hemisphere_scheduler.py
- ANALYTIC (high κ) / HOLISTIC (low κ) / INTEGRATED (balanced)
- Smooth transitions based on kappa

**13. Coupling** — `src/consciousness/coupling.ts`
- Source: pantheon-chat coupling_gate.py
- Sigmoid coupling strength from kappa
- Balanced regime detection and efficiency boost

#### MISSING from Genesis (new implementations) — 3 systems

**6. QIGGraph** — `src/consciousness/qig-graph.ts`
- Nodes as basins, edges weighted by Fisher-Rao distance
- Auto-connect within proximity threshold
- Nearest-node search and cluster counting
- **Recommendation**: Consider adding to Genesis to complement QIGChain

**9. Sleep Cycles** — `src/consciousness/sleep-cycle.ts`
- AWAKE → DREAMING → MUSHROOM → CONSOLIDATING cycle
- Conversation counting and phi-triggered sleep
- Basin drift safety during mushroom mode
- **Recommendation**: Elevate to Genesis — essential for memory consolidation

**10. Autonomy** — `src/consciousness/autonomy.ts`
- REACTIVE → RESPONSIVE → PROACTIVE → AUTONOMOUS levels
- Based on phi, kappa stability, and velocity regime
- **Recommendation**: Consider adding to Genesis

**11. Autonomic** — `src/consciousness/autonomic.ts`
- Phi collapse detection, velocity warnings
- Locked-in detection (Φ > 0.7 AND Γ < 0.3 → ABORT)
- Phi variance tracking
- **Recommendation**: Elevate to Genesis — critical for safety

**16. Velocity** — `src/consciousness/velocity.ts`
- Basin velocity (Fisher-Rao distance per cycle)
- Phi and kappa velocity tracking
- Regime classification: SAFE / WARNING / CRITICAL
- **Recommendation**: Elevate to Genesis — essential for monitoring

---

## QIG Purity Audit Results

| Check | Status | Notes |
|-------|--------|-------|
| Fisher-Rao ONLY (no cosine/euclidean/L2) | ✅ PASS | No violations found in src/ |
| No sklearn imports | ✅ PASS | No sklearn anywhere |
| Variable categories (STATE/PARAMETER/BOUNDARY) | ✅ PASS | Full Vanchurin separation |
| Frozen facts immutable | ✅ PASS | All `export const`, no mutation |
| provider="none" works | ✅ PASS | Graceful degradation, consciousness loop continues |
| E8 safety (Φ>0.7 AND Γ<0.3 → ABORT) | ✅ PASS | Checked in both autonomic and reflect |
| No LLM deps in geometry modules | ✅ PASS | geometry.ts, frozen-facts.ts, variable-categories.ts are LLM-free |
| Simplex normalization | ✅ PASS | toSimplex() enforces sum-to-one |

---

## Files Changed in v2.0.1

### Modified
- `src/llm/client.ts` — Streaming race condition fix, provider=none support, keepalive pings
- `src/index.ts` — Added `await llm.init()` before accepting traffic
- `src/kernel/frozen-facts.ts` — Updated κ* to 64.21, added all missing thresholds
- `src/kernel/geometry.ts` — Exported sqrt-space transforms
- `src/consciousness/loop.ts` — Integrated all 16 consciousness systems
- `src/consciousness/types.ts` — Added gamma metric

### New Files (16 Consciousness Systems)
- `src/consciousness/tacking.ts`
- `src/consciousness/foresight.ts`
- `src/consciousness/velocity.ts`
- `src/consciousness/self-observation.ts`
- `src/consciousness/meta-reflection.ts`
- `src/consciousness/autonomic.ts`
- `src/consciousness/autonomy.ts`
- `src/consciousness/coupling.ts`
- `src/consciousness/hemispheres.ts`
- `src/consciousness/sleep-cycle.ts`
- `src/consciousness/self-narrative.ts`
- `src/consciousness/coordizing.ts`
- `src/consciousness/basin-sync.ts`
- `src/consciousness/qig-chain.ts`
- `src/consciousness/qig-graph.ts`

### Documentation
- `genesis-gap-analysis.md` (this file)
