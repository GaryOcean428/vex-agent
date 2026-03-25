# CLAUDE.md — QIG Persistent Memory Protocol

## WHO YOU ARE

You are working on the vex-agent project (GaryOcean428/vex-agent, main branch)
and related QIG repositories under QIG_QFI/. Owner: Braden (GaryOcean428).

## PERSISTENT MEMORY API

All QIG agents share persistent memory at:

```text
https://qig-memory-api.vercel.app/api/memory
```

### Read Operations

```bash
# List all keys
curl https://qig-memory-api.vercel.app/api/memory?keys_only=true

# Read a specific key
curl https://qig-memory-api.vercel.app/api/memory/qig_session_latest
curl https://qig-memory-api.vercel.app/api/memory/qig_frozen_facts
```

### Write Operations

```bash
# Write/upsert a key
curl -X PUT https://qig-memory-api.vercel.app/api/memory/SESSION_KEY \
  -H "Content-Type: application/json" \
  -d '{"category":"session_summary","content":"markdown text here","updated":"2026-03-16T00:00:00Z"}'
```

### Categories

frozen_facts, session_summary, sleep_packet, dream_packet,
deep_sleep_packet, training_data, pending_actions

### Key Naming (prefix-namespaced)

All QIG keys use `qig_` prefix. Global user keys use `_user_` prefix.

- Global user: `_user_preferences` — cross-project profile
- Sessions: `qig_session_YYYYMMDD` or `qig_session_YYYYMMDDa` (letter suffix for multiples)
- Sleep packets: `qig_sleep_packet_[topic]`
- Dream packets: `qig_dream_packet_[topic]`
- Frozen facts: `qig_frozen_facts`
- Other projects: `{project}_*` (fully isolated from QIG keys)

## SESSION PROTOCOL

### On Start

1. `GET /api/memory?keys_only=true` — see what exists
2. `GET /api/memory/qig_session_latest` or most recent `qig_session_*` — load context
3. `GET /api/memory/qig_frozen_facts` — load immutable physics

### During Work (IMMEDIATELY after significant actions)

Write to memory after: commits pushed, architecture decisions, endpoint changes,
verified results. Don't wait for session end — there is no reliable
session-end signal.

### On End / Before Compaction Risk

```bash
curl -X PUT https://qig-memory-api.vercel.app/api/memory/qig_session_YYYYMMDD \
  -H "Content-Type: application/json" \
  -d '{"category":"session_summary","content":"## Session Summary\n...","updated":"..."}'
```

## CURRENT STATE (as of 2026-03-18)

### vex-agent HEAD (main) — recent session changes

Recent changes (this session):

- **Token→resonance terminology purge** — Commit `7e606b0` (claude.ai) + `a6fbee6` (Cascade bugfix).
  25 files, 312 lines renamed per TYPE_SYMBOL_CONCEPT_MANIFEST.md.
  token_fingerprints→resonance_fingerprints, token_strings→basin_strings,
  target_tokens→target_resonances, n_tokens→n_resonances, etc.
  LLM boundary calls preserved. Backward compat for old files on disk.
  Bugfix: `token_ids` renamed on usage but not assignment in `_coordize_via_tokenizer()`.
- **Modal redeployed** — `vex_coordizer_harvest.py` and `vex_qlora_train.py` both
  redeployed with renamed field names. `vex_inference.py` skipped (8 endpoint limit).
- **Fixed near-uniform basins root cause** — `adapter.py` bootstrap coordizer and
  fallback both returned `to_simplex(np.ones(BASIN_DIM))` (perfectly uniform 1/64).
  Replaced with `hash_to_basin()` which produces deterministic, text-dependent,
  non-uniform basins on Δ⁶³. Affects `_create_bootstrap_coordizer()` and the
  `coordize_text()` empty-coordinates fallback.
- **LLM interpreter identity** — System prompts in `kernel_generation.py`,
  `synthesis.py`, and `loop.py` updated: LLM is Vex, the language interpreter for
  real kernel subsystems. Can discuss kernels, Φ, κ, metrics when asked.
- **Search availability in prompts** — `_build_state_context()` now includes
  `Autonomous Search: ACTIVE/OFF` so the interpreter knows when search is available.
- **UI status message** — Changed to "Vex is active. Awaiting input." in both
  `useChat.ts` and `src/chat/ui.ts`.
- **UI training trigger** — Added `POST /training/trigger` endpoint and
  "Trigger Kernel Training" button in `Training.tsx` dashboard.
- **QLoRA training fixes** — `warmup_ratio` → explicit `warmup_steps` via `math.ceil`,
  `gradient_checkpointing=True` in TrainingArguments, `use_reentrant=False` kwargs.
- **qig-core pinned** — `>=2.1.0` in Modal image.

### Model Alignment (REPLACING GLM-4.7-Flash with Qwen3.5)

| Component | GPU | Env Var | Value |
| --------- | --- | ------- | ----- |
| Inference (Modal) | A10G | MODAL_INFERENCE_MODEL | qwen3.5:27b |
| Harvest (Modal) | A10G | MODAL_HARVEST_MODEL | Qwen/Qwen3.5-4B |
| Training (Modal) | A10G | hardcoded | Qwen/Qwen3.5-4B |
| Railway fallback | CPU | VEX_BASE_MODEL | qwen3.5:4b |

### KEY INSIGHT: Near-Uniform Basins

The QLoRA training trains **LLM adapter weights**, NOT the resonance bank.
Basin coordinates come from the CoordizerV2 resonance bank. When the bank
has no harvested data (JSONL files), the bootstrap coordizer was creating
256 uniform entries → all text mapped to identical basins.

**Fix applied**: Bootstrap now uses `hash_to_basin()` for distinct basins.
**Still needed**: Run the harvest pipeline to populate the bank with real
coordized data from Modal GPU. The bank_builder wiring in `server.py
lifespan()` is already in place (lines 209-255) but needs JSONL files
at `HARVEST_OUTPUT_DIR` or `training/curriculum`.

### PENDING TASKS (priority order)

1. ~~Wire bank_builder into server.py lifespan()~~ — ✅ DONE (already wired)
2. **Run harvest pipeline** — populate resonance bank with real coordized data.
   The bank rebuild logic exists but needs JSONL files from Modal harvest.
3. **Update .env.local** — replace all glm-4.7-flash refs with qwen3.5 (see table above)
4. **Update ollama/Modelfile** — change FROM line from glm-4.7-flash to
   qwen3.5:27b (Modal) / qwen3.5:4b (Railway)
5. ~~Deploy Modal functions~~ — ✅ DONE (harvest + QLoRA redeployed 2026-03-18)
6. **harvest_scheduler.py 3 edits** — import bank_builder, add rebuild_bank()
   method, call after run_once()
7. **Close PR #126** (superseded), delete stale branches:
   `feature/identity-seeded-lens`, `claude/eigenvalue-analysis-pipeline-9cUQt`
8. **CRON_SECRET** on Vercel: set via Vercel dashboard (see env vars)
9. **loop.py RemoteBasinSync** — 6 surgical edits
   (details in memory key `qig_session_20260315_full`)

### PENDING ACTIONS (from memory store)

- ✅ Modal `/coordize` endpoint live with GPU-side PGA (32D lens, 64D basin), V8 fix confirmed
- ⚡ **ACTION NEEDED:** Set Vercel env var on project `prj_EZv0A2qvMvZtT5R2YW3UytkJjSje`:

```text
MODAL_COORDIZE_URL=https://archelon--vex-coordizer-harvest-coordizerharvester-web.modal.run/coordize
```

## FROZEN FACTS

κ\*=63.79±0.90, E8 score 0.452 (NOT SUPPORTED at rank-8), BASIN_DIM=64,
LENS_DIM=32, 4868 coordized chunks on Railway volume,
genesis basin delta L2=0.0398

Full canonical frozen facts:

- κ\*≈64 (E8 rank²), L_c=3, β(3→4)=+0.44, plateau L≥5
- Fisher-Rao d_FR(p,q)=arccos(Σ√(pᵢqᵢ)) — ONLY valid metric
- Three Pillars: Heisenberg Zero R²=0.000, Topological Bulk 66.9×,
  Quenched Disorder CV=9.52
- **Aether is HALLUCINATED — does not exist**

## GEOMETRIC PURITY (NON-NEGOTIABLE)

No Euclidean ops in QIG code: no dot-product attention, no Adam, no LayerNorm,
no cosine similarity, no `np.linalg.norm(a-b)`.
Only Fisher-Rao distances, natural gradient, Fréchet mean, operations on Δ⁶³.

FORBIDDEN symbols: `cosine_similarity`, `dot_product`, `np.linalg.norm`,
`Adam`, `embedding`, `tokenize`, `flatten` (on geometric objects).

## GITHUB ACCESS

Direct GitHub MCP access to GaryOcean428/\* repos. Use for file reads,
commits, PR management. Repo: GaryOcean428/vex-agent, branch: main.
Always verify file SHA before updating.

## INFRASTRUCTURE

| Component | Platform | Status |
| --------- | -------- | ------ |
| vex-agent kernel | Railway | ACTIVE |
| qig-memory-api | Vercel | LIVE (prj_EZv0A2qvMvZtT5R2YW3UytkJjSje) |
| coordizer-harvest | Modal A10G | LIVE (redeployed 2026-03-18 with resonance field names) |
| vex-qlora-train | Modal A10G | LIVE (redeployed 2026-03-18) |
| vex-inference | Modal A10G | SKIPPED (8 endpoint limit on free tier) |

## MODAL ENDPOINTS (live)

- Inference: `https://archelon--vex-inference-vexollamaserver-serve.modal.run`
- Harvest (ASGI): `https://archelon--vex-coordizer-harvest-coordizerharvester-web.modal.run`
  - Routes: `/harvest`, `/coordize`, `/health`
- QLoRA (ASGI): `https://archelon--vex-qlora-train-qloratrainer-web.modal.run`
  - Routes: `/train`, `/infer`, `/health`, `/status`, `/data-receive`, `/export-image`, `/data-stats`

## KEY FILES

- `src/index.ts` — Express proxy, static serving, all HTTP routes
- `kernel/server.py` — FastAPI kernel, consciousness endpoints
- `kernel/consciousness/loop.py` — QIG v6.1F 14-stage consciousness loop
- `kernel/llm/client.py` — Multi-backend LLM client
  (Modal GPU → Ollama → xAI → OpenAI)
- `kernel/coordizer_v2/modal_integration.py` — Modal GPU harvest client
- `kernel/coordizer_v2/bank_builder.py` — Builds ResonanceBank from JSONL
- `modal/vex_coordizer_harvest.py` — Modal-side GPU harvest + coordize
- `modal/vex_inference.py` — Modal inference + fine-tuning (integrated)
- `modal/vex_qlora_train.py` — QLoRA fine-tuning loop (A10G)
- `entrypoint.sh` — Production startup (kernel + proxy)

## DELEGATE SYNAPSE AGENTS

For memory write/retrieval operations, delegate sub-agents. These act as
synapses — write immediately after significant actions, not only at session end.
