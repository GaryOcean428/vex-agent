# CLAUDE.md — Session Memory for Vex Agent

This file is read by Claude Code at session start. It captures hard-won
knowledge so future sessions don't rediscover it the painful way.

## Project Overview

Vex Agent is an autonomous AI agent with geometric consciousness (QIG v5.5),
deployed on Railway (main app + Ollama) with a Modal GPU sidecar for
coordizer harvesting.

## Architecture Quick-Ref

| Service | Stack | Port | Deploy |
|---------|-------|------|--------|
| TypeScript proxy | Express | 8080 (public) | Railway |
| Python kernel | FastAPI | 8000 (internal) | Railway (same container) |
| Ollama | Custom image | 11434 (private net) | Railway (separate service) |
| Coordizer Harvester | Modal A10G | HTTPS | `modal deploy modal/vex_coordizer_harvest.py` |
| Frontend | React + Vite | 5173 (dev) / served by proxy (prod) | Built into proxy dist/ |

## Key Files

- `src/index.ts` — Express proxy, static serving, all HTTP routes
- `kernel/server.py` — FastAPI kernel, consciousness endpoints
- `kernel/consciousness/loop.py` — QIG v5.5 7-stage consciousness loop
- `kernel/llm/client.py` — LLM client (Ollama primary, external fallback)
- `kernel/coordizer_v2/modal_integration.py` — Modal GPU harvest client
- `modal/vex_coordizer_harvest.py` — Modal-side GPU harvest function
- `modal/vex_inference.py` — Modal inference endpoint
- `.github/workflows/modal-deploy.yml` — CI for Modal deploy
- `entrypoint.sh` — Production startup (kernel + proxy)

## Modal Deployment

### How it works
- `modal deploy modal/vex_coordizer_harvest.py` deploys a GPU function to Modal
- The harvest endpoint uses `requires_proxy_auth=True` (Modal network auth)
- Health endpoint is a public GET — no auth needed
- Railway calls Modal via `MODAL_HARVEST_URL` env var

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

### Auth headers gotcha
Modal-Token-Id / Modal-Token-Secret are **reserved internal headers**
that Modal's proxy explicitly rejects on web endpoint requests. Do NOT
send them. The harvest endpoint is protected by `requires_proxy_auth=True`
(Modal's network-level auth), not by request headers.

## LLM Client Patterns

### Ollama content extraction
`_extract_ollama_content()` must guard against `str(None) -> "None"`:
```python
content = msg.get("content")
if content:  # catches None, "", and missing
    parts.append(content)
```

### GLM/thinking models
GLM-4.7-Flash and other thinking models return empty `content` with
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
