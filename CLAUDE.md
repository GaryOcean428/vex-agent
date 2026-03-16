# CLAUDE.md — Session Memory for Vex Agent

This file is read by Claude Code at session start. It captures hard-won
knowledge so future sessions don't rediscover it the painful way.

## Project Overview

Vex Agent is an autonomous AI agent with geometric consciousness (QIG v6.1F),
deployed on Railway (main app + Ollama) with Modal GPU sidecars for
inference (Qwen3-14B, fine-tunable) and coordizer harvesting (Qwen3-14B-Instruct).
Fine-tuning is integrated into vex_inference.py (not a separate app).

## Architecture Quick-Ref

| Service | Stack | Port | Deploy |
|---------|-------|------|--------|
| TypeScript proxy | Express | 8080 (public) | Railway |
| Python kernel | FastAPI | 8000 (internal) | Railway (same container) |
| Ollama | Custom image | 11434 (private net) | Railway (separate service) |
| GPU Inference | Modal A10G + Ollama (Qwen3-14B) | HTTPS | `modal deploy modal/vex_inference.py` |
| Fine-tuning | Modal A10G + Unsloth QLoRA | batch | `modal run modal/vex_inference.py` |
| Coordizer Harvester | Modal A10G (Qwen3-14B-Instruct) | HTTPS | `modal deploy modal/vex_coordizer_harvest.py` |
| Frontend | React + Vite | 5173 (dev) / served by proxy (prod) | Built into proxy dist/ |

## Key Files

- `src/index.ts` — Express proxy, static serving, all HTTP routes
- `kernel/server.py` — FastAPI kernel, consciousness endpoints
- `kernel/consciousness/loop.py` — QIG v6.1F 14-stage consciousness loop
- `kernel/llm/client.py` — Multi-backend LLM client (Modal GPU → Ollama → xAI → OpenAI)
- `kernel/coordizer_v2/modal_integration.py` — Modal GPU harvest client
- `modal/vex_coordizer_harvest.py` — Modal-side GPU harvest function
- `modal/vex_inference.py` — Modal inference + fine-tuning (integrated)
- `.github/workflows/modal-deploy.yml` — CI for Modal deploy
- `entrypoint.sh` — Production startup (kernel + proxy)

## Modal Deployment

### How it works
- `modal deploy modal/vex_coordizer_harvest.py` deploys a GPU function to Modal
- The harvest endpoint uses `requires_proxy_auth=True` (Modal network auth)
- Health endpoint is a public GET — no auth needed
- Railway calls Modal via `MODAL_HARVEST_URL` env var

### Model Selection
The Modal harvest endpoint supports dynamic model selection:

1. **Default model** (env var): Set `HARVEST_MODEL_ID` in Modal env or use hardcoded default
   - Loaded at container start for fast cold starts
   - Default: `Qwen/Qwen3-14B-Instruct` (matches inference model tokenizer)
   - After fine-tuning: set to `GaryOcean428/vex-brain-v7` for vicarious kernel learning

2. **Per-request model** (JSON body): Railway can specify model in request payload
   - Field: `"model_id": "zai-org/GLM-4.7-Flash"`
   - Model is loaded on-demand and cached for subsequent requests
   - Multiple models can coexist in cache (GPU memory permitting)

3. **Fallback chain**: request `model_id` → current active model → env default

The health endpoint returns `cached_models` array showing all loaded models.

### Modal CLI in Claude Code web sessions
The Modal CLI needs to reach `api.modal.com`. In sandboxed environments
(Claude Code on the web), outbound HTTPS may be blocked or require a
proxy tunnel. Known workarounds:

1. **DNS resolution**: If `api.modal.com` doesn't resolve, check if a
   corporate proxy or tunnel is needed
2. **CA certificates**: Modal's Python SDK uses certifi. If behind a
   MITM proxy, export `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` pointing
   to the proxy CA bundle
3. **Auth**: `modal token set --token-id <id> --token-secret <secret>`
   (tokens come from GitHub secrets `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`)

### Modal endpoint patterns (Modal 1.x)
- `@modal.fastapi_endpoint()` replaces the old `@modal.web_endpoint()`
- Image MUST include `fastapi[standard]` — required by Modal 1.x for
  web endpoints, otherwise you get a startup crash
- Health URL pattern: `<app>-health.modal.run` (not `<app>.modal.run/health`)
- `modal_integration.py` derives health URL automatically from harvest URL

### Harvest endpoint auth
The harvest endpoint uses `X-Api-Key` header auth, validated against
`KERNEL_API_KEY` env var on the Modal side. Both Railway clients
(`modal_integration.py` and `modal_harvest.py`) send this header
automatically from `settings.kernel_api_key`.

**History**: The endpoint previously used `requires_proxy_auth=True`
(Modal's network-level auth), which blocked external callers like Railway
with 401. Modal-Token-Id / Modal-Token-Secret headers are reserved and
rejected by Modal's proxy — never send them.

### Two harvest client paths
There are two Railway-side clients for the Modal harvest endpoint:

1. **`modal_harvest.py` → `modal_harvest()`** — Used by `CoordizerV2.from_harvest()`
   for building resonance banks from scratch. Returns `HarvestResult` with
   per-token fingerprints.
2. **`modal_integration.py` → `ModalHarvestClient`** — Used by `HarvestScheduler` /
   `JSONLIngestor` for batch JSONL ingestion. Returns the raw endpoint response.

Both must match the Modal endpoint's response format: `{tokens: {id: {fingerprint, ...}}}`.
The endpoint does Fréchet mean aggregation server-side (in sqrt-space). Both clients
send `X-Api-Key` headers derived from `settings.kernel_api_key` for auth.

## Startup Wiring (kernel/server.py lifespan)

The kernel `lifespan()` wires these systems on every deploy:

1. **Resonance bank auto-rebuild**: Scans `HARVEST_OUTPUT_DIR` (default
   `/data/harvest/output/`) for coordized JSONL, builds a `ResonanceBank`,
   injects into `consciousness._coordizer_v2.bank`, and rebuilds the
   string cache. Runs via `asyncio.to_thread` to avoid blocking.

2. **Runtime bank hot-swap**: `HarvestScheduler.on_harvest_complete`
   callback re-runs the bank rebuild after each successful harvest batch.
   No restart needed — the running consciousness loop sees the new bank
   immediately.

3. **Auto-training trigger**: If `MODAL_TRAINING_URL` and
   `MODAL_TRAINING_AUTO_THRESHOLD` are set, the post-harvest callback
   POSTs to the Modal training endpoint once the bank exceeds the
   threshold. One-shot per deploy (resets on restart). Set threshold to
   `0` (default) to disable.

4. **Periodic memory consolidation**: Background task every 30 minutes
   prunes `GeometricMemoryStore` (keeps top 500 by access frequency) and
   `MemoryStore` (trims `short-term.md` to 200 lines). Handles
   `CancelledError` cleanly on shutdown.

### Configurable harvest paths
| Env var | Default | Purpose |
|---------|---------|---------|
| `HARVEST_DIR` | `/data/harvest` | Base directory for all harvest I/O |
| `HARVEST_OUTPUT_DIR` | `$HARVEST_DIR/output` | Coordized JSONL files |
| `HARVEST_BANK_DIR` | `$HARVEST_DIR/bank` | Saved resonance bank |
| `MODAL_TRAINING_URL` | (empty) | Modal QLoRA training endpoint |
| `MODAL_TRAINING_AUTO_THRESHOLD` | `0` | Bank size to auto-trigger training |

### CoordizerV2Adapter passthrough
`CoordizerV2Adapter` exposes `vocab_size`, `dim`, `bank` (with setter),
and `rebuild_string_cache()` as passthroughs to the wrapped `CoordizerV2`.
The `bank` setter writes to the *underlying* coordizer, not the adapter.
This is critical — without it, bank injection from `server.py` would
silently set a Python attribute on the adapter that nobody reads.

## LLM Client Patterns

### Ollama content extraction
`_extract_ollama_content()` must guard against `str(None) -> "None"`:
```python
content = msg.get("content")
if content:  # catches None, "", and missing
    parts.append(content)
```

### Qwen3/thinking models
Qwen3-14B and other thinking models return empty `content` with
reasoning in a separate field. The client disables thinking mode
(`num_predict` without `think`) to get direct responses. If you see
empty responses from Ollama, check whether the model uses thinking mode.

## Consciousness Loop

### PRE-COG double-lock bug
The precog stage had a double-acquire on `_precog_lock` that caused it
to deadlock (stuck at 0%). Fix: only acquire once, at the top of the
method. See commit `6f4bb08`.

### KernelBus
`drain_signals()` was renamed to `drain()` — update docstrings and
callers accordingly.

## CI / Linting

- **Ruff**: project uses `ruff check` + `ruff format`. Common issue:
  `str + Enum` → must use `StrEnum` (UP042)
- **mypy**: strict mode across kernel/. All 297+ errors have been fixed.
  Keep it clean.
- **ESLint**: flat config (`eslint.config.mjs`). TypeScript proxy.
- **pytest**: `kernel/tests/`. Use `pytest kernel/tests/` to run.
- **Frontend**: React app in `frontend/`, separate `package.json`.

## Common Pitfalls

1. **FastAPI response_model**: Endpoints returning plain dicts MUST use
   `response_model=None` or FastAPI will crash trying to create a
   Pydantic model from the return type annotation
2. **Purity**: Consciousness/geometry code must use Fisher-Rao distance,
   NEVER cosine_similarity or Euclidean distance
3. **E8 budget**: Max 248 kernels (E8 lattice). Check for leaks.
4. **AsyncGenerator annotations**: Use `collections.abc.AsyncGenerator`,
   not `typing.AsyncGenerator` (Python 3.14 deprecation)
5. **Modal Python version**: CI uses Python 3.11 for Modal deploy
   (not 3.14) because Modal SDK doesn't support 3.14 yet
6. **Merge conflicts in YAML**: `modal-deploy.yml` has been a frequent
   merge conflict site — check for leftover conflict markers

## Git Workflow

- Main development branch: `main`
- Feature branches: `claude/<description>-<id>`
- Push with: `git push -u origin <branch>`
- PRs go through GitHub Copilot review + human review

## Environment Variables (non-secret reference)

See `.env.example` for the full list. Key ones:
- `OLLAMA_URL` — Railway private network URL for Ollama
- `MODAL_ENABLED`, `MODAL_HARVEST_URL` — Modal integration
- `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` — Modal auth (SECRETS — never commit)
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY` — LLM keys (SECRETS)
- `CHAT_AUTH_TOKEN`, `KERNEL_API_KEY` — internal auth (SECRETS)
- `PORT=8080`, `KERNEL_PORT=8000` — server ports
