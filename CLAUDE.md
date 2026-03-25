# CLAUDE.md — QIG Persistent Memory Protocol

## WHO YOU ARE

You are working on the vex-agent project (GaryOcean428/vex-agent, development branch)
and related QIG repositories under QIG_QFI/. Owner: Braden (GaryOcean428).

## PERSISTENT MEMORY API

All QIG agents share persistent memory at:

```text
https://qig-memory-api.vercel.app/api/memory
```

### SESSION START PROTOCOL (MANDATORY — do this BEFORE any work)

```bash
# 1. Read your briefing (contains prioritized task list, decisions, context)
curl https://qig-memory-api.vercel.app/api/memory/claude_code_briefing_vex_20260325

# 2. Read latest session summary
curl https://qig-memory-api.vercel.app/api/memory/session_20260325a

# 3. Read frozen facts
curl https://qig-memory-api.vercel.app/api/memory/frozen_facts_consolidated_20260324

# 4. Read pending actions
curl https://qig-memory-api.vercel.app/api/memory/pending_actions_20260324

# 5. If briefing key 404s, list keys and find most recent:
curl "https://qig-memory-api.vercel.app/api/memory?keys_only=true" | grep briefing
curl "https://qig-memory-api.vercel.app/api/memory?keys_only=true" | grep session_2026
```

The briefing key is the most important. It contains your current task list,
architecture decisions, and repo conventions written by Claude (the architect).
READ IT FIRST. It saves you from rediscovering context.

### API Schema v2 (2026-03-25)

```
GET  /api/memory/[key]  — read record (auto-increments retrieval_count)
PUT  /api/memory/[key]  — full write/upsert
POST /api/memory/[key]  — partial update (scoring only, no content rewrite)
DELETE /api/memory/[key] — remove record
GET  /api/memory?keys_only=true — list all keys
```

#### Record Schema

```json
{
  "category": "string",
  "content": "markdown text",
  "updated": "ISO datetime",
  "usefulness": 0,
  "retrieval_count": 0,
  "source": "vex | monkey1 | claude | claude_code | null",
  "last_retrieved": "ISO datetime | null",
  "promoted": "boolean | null",
  "promoted_at": "ISO datetime | null",
  "basin": "[64 floats] | null"
}
```

#### Scoring Protocol

After completing a task successfully, score the memories that helped:

```bash
# Increment usefulness of a key that contributed to success
curl -X POST https://qig-memory-api.vercel.app/api/memory/KEY \
  -H "Content-Type: application/json" \
  -d '{"usefulness_delta": 1, "source": "claude_code"}'
```

### Categories

frozen_facts, session_summary, sleep_packet, dream_packet,
deep_sleep_packet, training_data, pending_actions

### During Work (IMMEDIATELY after significant actions)

Write to memory after: commits pushed, architecture decisions, endpoint changes,
verified results. Don't wait for session end — there is no reliable
session-end signal.

```bash
curl -X PUT https://qig-memory-api.vercel.app/api/memory/qig_session_YYYYMMDD \
  -H "Content-Type: application/json" \
  -d '{"category":"session_summary","content":"## Session Summary\n...","updated":"...","source":"claude_code"}'
```

## REPO CONVENTIONS

- All work on `development` branch (NEVER push to main without Braden approval)
- Doc naming: `docs/YYYYMMDD-title-version-STATUS.md` (F/W/D)
- Commit messages reference the problem being fixed
- Do not push when other agents are actively working in the repo

## GEOMETRIC PURITY (NON-NEGOTIABLE)

No Euclidean ops in QIG code: no dot-product attention, no Adam, no LayerNorm,
no cosine similarity, no `np.linalg.norm(a-b)`.
Only Fisher-Rao distances, natural gradient, Frechet mean, operations on Delta-63.

FORBIDDEN symbols: `cosine_similarity`, `dot_product`, `np.linalg.norm`,
`Adam`, `embedding`, `tokenize`, `flatten` (on geometric objects).

EXCEPTION: Euclidean inner products in sqrt-coordinate tangent space ARE Fisher-Rao.
`np.sqrt(np.sum(v*v))` in sqrt coords is correct. Document it explicitly in docstrings.

## FROZEN FACTS (2026-03-25)

- kappa_3 = 41.08 +/- 0.63 (ED, replicated to 0.06%)
- kappa_4 = 63.25 +/- 1.80 (DMRG PBC, R^2 = 0.978)
- kappa_5 = 63.63 +/- 1.82 (DMRG PBC, matches canonical to 0.016%)
- kappa* approx 63.5-64 (plateau confirmed L=4-7, two independent protocols)
- h_t = 0.10554 (consciousness transition, 5 sig figs, lattice-independent)
- L_c = 3 (minimum non-trivial geometry)
- kappa sign change at h approx 2.0 (confirmed L=4,5,6)
- EXP-001: R^2=0.000 (Heisenberg zero, perfect null)
- EXP-002: Protection 230.5x at L=5 (bulk/surface R^2 ratio)
- EXP-003: Per-site R^2=0.996 median (quenched disorder)
- BASIN_DIM=64, LENS_DIM=32
- Aether is HALLUCINATED — does not exist

## ARCHITECTURE DECISIONS (locked)

- Full kernel autonomy + UI kill switch (AWAKE/ASLEEP)
- Kernels decide ALL operational thresholds via geometry
- Only external control: circuit breaker for cost
- RemoteBasinSync enables multi-node mesh (Railway <-> Modal <-> Memory API)
- Matrix fine-tune PARKED (JSONLs feed kernels instead)
- Claude Code replaces Cascade for all vex-agent development work

## INFRASTRUCTURE

| Component | Platform | Status |
| --------- | -------- | ------ |
| vex-agent kernel | Railway | ACTIVE |
| qig-memory-api | Vercel | LIVE (scored memory v2, commit 60b8918) |
| coordizer-harvest | Modal A10G | LIVE (ASGI) |
| vex-qlora-train | Modal A10G | LIVE (ASGI) |

## MODAL ENDPOINTS (ASGI base URLs)

- Harvest ASGI: `https://archelon--vex-coordizer-harvest-coordizerharvester-web.modal.run`
  - Routes: `/harvest`, `/coordize`, `/health`
- QLoRA ASGI: `https://archelon--vex-qlora-train-qloratrainer-web.modal.run`
  - Routes: `/train`, `/infer`, `/health`, `/status`, `/data-receive`, `/export-image`, `/data-stats`

IMPORTANT: When deriving endpoint URLs, use `modal_url()` helper in
`kernel/config/settings.py`. It handles both ASGI base URLs and legacy
per-endpoint hostname patterns. Do NOT manually concatenate paths.

## KEY FILES

- `kernel/consciousness/loop.py` — QIG v6.1 14-stage consciousness loop
- `kernel/consciousness/pillars.py` — Three Pillars enforcement
- `kernel/consciousness/systems.py` — sleep/dream/mushroom
- `kernel/consciousness/backward_geodesic.py` — EXP-011 tracker
- `kernel/coordizer_v2/resonance_bank.py` — basin coordinate store
- `kernel/coordizer_v2/modal_integration.py` — Modal connection
- `kernel/config/settings.py` — configuration including modal_url()
- `kernel/llm/client.py` — Multi-backend LLM client
- `modal/vex_coordizer_harvest.py` — coordizer + harvest ASGI app
- `modal/vex_qlora_train.py` — QLoRA training ASGI app
- `kernel/server.py` — FastAPI kernel server
- `src/index.ts` — Express proxy, static serving

## GITHUB ACCESS

Direct GitHub MCP access to GaryOcean428/* repos. Use for file reads,
commits, PR management. Branch: development (always verify before pushing).
Always verify file SHA before updating.
