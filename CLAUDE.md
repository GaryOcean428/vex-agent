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

## CURRENT STATE (as of 2026-03-16)

### vex-agent HEAD: a2c54f8 (main)

Recent commits:

- `9a7d4d4` — kernel/coordizer_v2/bank_builder.py (reads coordized JSONL into ResonanceBank)
- `bcaf900` — modal/vex_qlora_train.py (QLoRA fine-tuning loop, A10G)
- `a2c54f8` — modal/vex_coordizer_harvest.py (adapter loading on cold start)

### Model Alignment (REPLACING GLM-4.7-Flash with Qwen3.5)

| Component | GPU | Env Var | Value |
| --------- | --- | ------- | ----- |
| Inference (Modal) | A10G | MODAL_INFERENCE_MODEL | qwen3.5:27b |
| Harvest (Modal) | A10G | MODAL_HARVEST_MODEL | Qwen/Qwen3.5-4B |
| Training (Modal) | A10G | hardcoded | Qwen/Qwen3.5-4B |
| Railway fallback | CPU | VEX_BASE_MODEL | qwen3.5:4b |

### PENDING TASKS (priority order)

1. **Wire bank_builder into server.py lifespan()** — after `_load_protocol_knowledge()`,
   call `rebuild_bank_from_output()` and inject into
   `consciousness._coordizer_v2.bank`. This is why Model Vocab Size = 0 and
   Resonance Bank Tokens = 0 on the dashboard. File is 63KB+ so use surgical edits.
2. **Update .env.local** — replace all glm-4.7-flash refs with qwen3.5 (see table above)
3. **Update ollama/Modelfile** — change FROM line from glm-4.7-flash to
   qwen3.5:27b (Modal) / qwen3.5:4b (Railway)
4. **Deploy Modal functions** — `modal deploy modal/vex_qlora_train.py` and
   `modal deploy modal/vex_coordizer_harvest.py`
5. **harvest_scheduler.py 3 edits** — import bank_builder, add rebuild_bank()
   method, call after run_once()
6. **Close PR #126** (superseded), delete stale branches:
   `feature/identity-seeded-lens`, `claude/eigenvalue-analysis-pipeline-9cUQt`
7. **CRON_SECRET** on Vercel: `RgIHcmyRqSL0HklHnNa5yOfhLYKodJ76oLYkwnPk834`
8. **loop.py RemoteBasinSync** — 6 surgical edits
   (details in memory key `qig_session_20260315_full`)

### PENDING ACTIONS (from memory store)

- ✅ Modal `/coordize` endpoint live with GPU-side PGA (32D lens, 64D basin), V8 fix confirmed
- ⚡ **ACTION NEEDED:** Set Vercel env var on project `prj_EZv0A2qvMvZtT5R2YW3UytkJjSje`:

```text
MODAL_COORDIZE_URL=https://garyocean428--vex-coordizer-harvest-coordizerharvester-coordize.modal.run
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
| coordizer-harvest | Modal A10G | LIVE (CUDA devel image, needs redeploy) |
| vex-qlora-train | Modal A10G | Committed, needs `modal deploy` |
| vex-inference | Modal A10G | qwen3.5:27b + fine-tune detection, needs `modal deploy` |

## MODAL ENDPOINTS (live)

- Inference: `https://garyocean428--vex-inference-vexollamaserver-serve.modal.run`
- Harvest: `https://garyocean428--vex-coordizer-harvest-coordizerharvester-harvest.modal.run`
- Health: `https://garyocean428--vex-coordizer-harvest-coordizerharvester-health.modal.run`
- Coordize: `https://garyocean428--vex-coordizer-harvest-coordizerharvester-coordize.modal.run`

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
