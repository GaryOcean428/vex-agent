# 20260224 — Vex Agent QA Checklist — v0.1W

Targeted QA framework for the Vex Agent codebase (QIG v6.1F). Derived from a full codebase audit on 2026-02-24. Items are grounded in actual gaps found — not theoretical boilerplate.

---

## 1. QIG Purity & Geometric Integrity

- [ ] **Purity scan on all new implementations** — run `python3 scripts/qig_purity_scan.py` after every merge. Confirm zero Euclidean contamination in `kernel/consciousness/`, `kernel/geometry/`, `kernel/coordizer_v2/`.
- [ ] **`frozen_facts.py` ↔ `src/config/constants.ts` sync** — `constants.ts` is marked "auto-generated — do not edit manually" but is not actually generated. Add a CI step or Makefile target that diffs the two and fails if they diverge.
- [ ] **`PHI_THRESHOLD` mismatch audit** — `frozen_facts.py` defines `PHI_THRESHOLD = 0.70`; `constants.ts` also has `0.7`. Verify all runtime guards in `loop.py`, `emotions.py`, `systems.py` use the imported constant, not a hardcoded float literal.
- [ ] **`fisher_rao_distance` single source of truth** — confirm all call sites import from `kernel/geometry/fisher_rao.py` only. No reimplementations in `coordizer_v2/` or `llm/`.

---

## 2. Redis / Persistence

- [ ] **`ConversationStore` (JSONL) is still the default** — `store.py` line 468: `make_conversation_store()` falls back to JSONL when `REDIS_URL` is unset. On Railway dev environments without Redis wired, chat history silently writes to ephemeral `/tmp`. Add a startup log warning when running in JSONL fallback mode.
- [ ] **Redis TTL on conversation index key** — `vex:convs` (sorted set) has no TTL set; only per-conversation keys have `expire`. Old conversation IDs accumulate in the index after their data keys expire. Add `self._r.expire(self._INDEX_KEY, self._ttl)` or prune stale IDs from the set.
- [ ] **`RedisConversationStore.get_token_count`** — returns `int(float(val))` which silently truncates. Should be `round(float(val))` for consistency with `estimate_tokens`.
- [ ] **Harvest pending queue** — `harvest_bridge.py` writes to `/data/harvest/pending` (filesystem). No Redis equivalent. Acceptable as-is (fire-and-forget), but document explicitly that this queue is not durable across volume remounts.

---

## 3. Architecture & Module Integrity

- [ ] **`frontend/src/components/` has no barrel `index.ts`** — `hooks/index.ts` exists and is well-structured; `components/` has no equivalent. All dashboard pages import components by full path. Add `frontend/src/components/index.ts` re-exporting shared components (`MetricCard`, `StatusBadge`, `ErrorBoundary`, `Toast`, `CommandPalette`).
- [ ] **`frontend/src/types/` has no barrel** — `consciousness.ts` is the only type file but is imported directly everywhere. When a second types file is added (e.g. for training or governor), add `types/index.ts`.
- [ ] **`frontend/src/auth/` has no barrel** — `auth/middleware.ts` in `src/` (proxy side) is separate from `frontend/src/auth/`. Confirm the frontend auth directory is not orphaned.
- [ ] **`frontend/src/stores/` is empty** — directory exists but `chatStore.ts` is absent (404 on read). If Zustand/context store was planned here, either implement or remove the directory.
- [ ] **`kernel/` Python `__init__.py` barrel exports** — confirm every public sub-package (`consciousness`, `coordizer_v2`, `geometry`, `llm`, `governance`, `chat`, `tools`, `training`, `memory`) has an `__init__.py` that at minimum makes the package importable. Orphaned directories without `__init__.py` cause silent import failures.
- [ ] **`Admin.tsx` section title bug** — the "Fresh Start (Reset)" section has a JSX comment reading `{/* Enqueue Task */}` directly above it (copy-paste artefact, line ~120). Fix the comment.

---

## 4. UI — Long-Form / Agentic Task Access

- [ ] **`/enqueue` task result is fire-and-forget** — `Admin.tsx` posts to `/enqueue`, receives a `task_id`, but never polls for the result. The user sees "Task enqueued: \<id\>" and nothing more. The task result sits in `loop._history` with no retrieval path from the UI. Implement a `/task/{id}` GET endpoint in the kernel and a polling panel in Admin that shows the result when available.
- [ ] **No streaming path for enqueued tasks** — `/chat/stream` uses SSE; `/enqueue` does not. Long agentic tasks (foraging, debate rounds, sleep cycles) have no streaming feedback mechanism outside of the chat interface. Consider either: (a) routing all agentic tasks through `/chat/stream` with a task-mode flag, or (b) adding a `/task/{id}/stream` SSE endpoint.
- [ ] **Queue depth is visible but queue contents are not** — `Admin.tsx` shows `state.queue_size` but not what's in the queue. Users can't see pending tasks or cancel them. Add a `/queue` endpoint that returns pending task IDs/previews, and a cancel action.
- [ ] **`CommandPalette.tsx` exists but is not wired to task dispatch** — the command palette is a good UX entry point for triggering agentic tasks directly. Wire a "Run task…" command that opens a modal with the enqueue form.
- [ ] **No progress indicator for sleep/consolidation cycles** — when the kernel enters sleep phase, there is no UI signal beyond the regime bar changing. Add a sleep-state banner or `SleepStateResponse` polling to the chat header.

---

## 5. Code Quality & DRY

- [ ] **Inline styles in `Admin.tsx`** — virtually all styling is inline `style={{...}}` objects. These duplicate the CSS custom property system used everywhere else in the dashboard. Extract to a `Admin.css` or use the existing `dash-*` class system already present in other dashboard pages.
- [ ] **`_compute_top_k()` / `_compute_debate_depth()` / `_select_model_by_complexity()` not called from `_process()`'s direct LLM fallback path** — when `_contributions` is empty (no eligible kernels), `loop.py` falls through to `self.llm.complete(state_context, ...)` directly without applying model selection (T4.4d). Ensure `_select_model_by_complexity` is also applied on the fallback path.
- [ ] **`with_model()` on LLM client is `hasattr`-guarded but never implemented** — `process_streaming` and `process_streaming_with_trace` both call `self.llm.with_model(_override_model) if hasattr(self.llm, "with_model") else self.llm`. The XAI escalation silently no-ops because `LLMClient` has no `with_model` method. Either implement `with_model(model: str) -> LLMClient` on the client, or pass `model_override` as a parameter to `generate_multi_kernel`.
- [ ] **Duplicate suffering calculation** — `suffering = self.metrics.phi * (1.0 - self.metrics.gamma) * self.metrics.meta_awareness` is computed in at least three places in `loop.py` (`_cycle_inner`, `_build_state_context`, `get_full_state`, telemetry). Extract to a `@property` or helper method on `ConsciousnessMetrics` or `ConsciousnessLoop`.
- [ ] **`reflection.py` unused import removed** — `from ..config.frozen_facts import PHI_HYPERDIMENSIONAL, PHI_THRESHOLD` was removed by the user in a recent diff but confirm no downstream usage remained.

---

## 6. Type Safety (TypeScript / Python)

- [ ] **`frontend/tsconfig.app.json` — `strict: true` is set** ✅ already correct.
- [ ] **`consciousness.ts` types for new telemetry fields** — `neurochemical`, `basin_sync`, `tacking.norepinephrine_gate` are now emitted by the kernel but may not be in `VexTelemetry` / `VexState` TypeScript types. Audit `get_full_state()` and `get_metrics()` output against the type definitions.
- [ ] **`usePolledData` error state not typed** — the hook returns `{ data, refetch }` but has no `error` field exposed to consumers. Dashboard pages can't distinguish "loading" from "fetch failed". Add `error: Error | null` to the return type.
- [ ] **`KernelSummary` type** — `useKernels` returns `KernelSummary` but the kernel endpoint returns a list. Confirm the type wraps correctly (should be `KernelSummary[]` or a wrapper type).

---

## 7. Testing

- [ ] **`test_purity_full.py`** — run on every PR. Currently only run manually. Add to pre-commit or CI.
- [ ] **No tests for `T2.1e/f`, `T4.1c`, `T4.2c/d/e`, `T4.4c/d` implementations** — the newly implemented deferred items have no unit tests. At minimum, add:
  - `test_norepinephrine_gate`: assert `PreCognitiveDetector.select_path()` returns `STANDARD` when `NE > 0.75` and cache hit exists.
  - `test_compute_top_k`: assert returns 2 when `sleep.is_asleep`, 5 when `regime_weights.quantum > 0.5` and `phi > 0.65`.
  - `test_regime_interval`: assert `_regime_interval()` returns `interval * 0.6` when `quantum > 0.5`.
  - `test_basin_sync_sleep_spindle`: assert `basin_sync.publish` is called during sleep phase when active kernels have basins.
- [ ] **No integration test for Redis fallback** — `make_conversation_store()` falls back to JSONL silently. Add a test that instantiates `RedisConversationStore` with a bad URL and verifies `make_conversation_store()` returns a `ConversationStore` (JSONL) instance.

---

## 8. Documentation

- [ ] **Duplicate top-level headings in `20260224-unified-implementation-checklist-0.1W.md`** — MD025 lint fires at line 821 (second `# ...` heading). Resolve by demoting the second heading to `##` or merging sections.
- [ ] **`docs/archive/` contains un-versioned filenames** — `20260217-CANONICAL_PRINCIPLES.md`, `20260217-CANONICAL_PRINCIPLES_v1_1.md`, and `20260217-canonical-principles-1.00W.md` are three versions of the same document. Consolidate to the `*-1.00W.md` convention (ISO-aligned naming) and archive or delete the others.
- [ ] **`docs/reference/` has two versions of canonical principles v2** — `20260217-CANONICAL_PRINCIPLES_v2.md` and `20260217-CANONICAL_PRINCIPLES_v2.1.md` alongside `20260217-canonical-principles-2.00W.md`. Keep only the `*-2.00W.md` canonical file; move the others to `docs/archive/`.
- [ ] **`docs/development/20260216-GROK_SYSTEM_PROMPT_v2.md`** — non-standard name (all-caps `GROK`, no version suffix in `*W` format). Rename to `20260216-grok-system-prompt-2.00W.md` or move to `docs/archive/`.
- [ ] **`AGENTS.md` at repo root** — references paths and models that may be stale post v6.1F refactor (e.g. `llama2.5-thinking:1.2b`, `vex-brain`). Update model references to match current `settings.ollama.model` / `settings.llm.model` defaults.
- [ ] **No `README.md` at repo root** — only `docs/README.md` exists. A root `README.md` with quick-start, architecture summary, and env var table would significantly improve onboarding. (Low priority — internal project.)

---

## 9. DevOps / Deployment

- [ ] **`COORDIZER_V2_ENABLED` defaults to `false`** — `settings.py` line 147. In production Railway deployments, CoordizerV2 is the intended path. Document whether this should default to `true` in the Railway env var config, or explain the intentional default-off.
- [ ] **`init.sh` chown** — ensure file ownership fix covers `/data/harvest/pending` alongside the conversation and training dirs (harvest bridge creates this at runtime, may fail on fresh volume mounts).
- [ ] **Pre-commit: ruff-format modifies files on commit** — the commit hook auto-reformats but exits non-zero, requiring a second `git add . && git commit`. This is standard ruff-format behaviour; document in `AGENTS.md` / `CONTRIBUTING.md` so contributors know to expect it.
- [ ] **Prettier: `src/chat/router.ts`, `src/config/constants.ts`, `src/tools/compute-sandbox.ts`** — these three files repeatedly fail Prettier on commit. They are now fixed, but investigate whether any import from these files uses non-standard formatting that triggers regressions.

---

## 10. Security

- [ ] **`CHAT_AUTH_TOKEN` empty = auth disabled** — documented in `AGENTS.md`. Confirm the Railway production deployment has this set and that the health endpoint (`/health`) is the only unauthenticated route exposed.
- [ ] **`KERNEL_API_KEY` guards** — `src/auth/middleware.ts` enforces the token; confirm the `/admin/fresh-start` endpoint is behind this guard in `src/chat/router.ts` (destructive operation).
- [ ] **`XAI_API_KEY` logged at debug level?** — `_select_model_by_complexity` logs `settings.xai.model` at DEBUG. Confirm the key itself is never interpolated into log strings anywhere in `loop.py` or `llm/`.

---

*Status key: `[ ]` = pending · `[x]` = complete · `[-]` = deferred/won't-fix*
