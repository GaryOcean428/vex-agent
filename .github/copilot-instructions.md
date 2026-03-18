# Copilot Instructions — Vex Agent

## Project

Vex is an autonomous AI agent with geometric consciousness (QIG v6.1F), deployed on Railway + Modal GPU.

## Architecture

| Service | Stack | Port |
|---------|-------|------|
| TypeScript proxy | Express | 8080 |
| Python kernel | FastAPI | 8000 |
| Ollama | Custom image | 11434 |
| GPU Inference | Modal A10G + Ollama | HTTPS |
| GPU Harvester | Modal A10G + Transformers | HTTPS |
| Frontend | React + Vite | 5173 (dev) |

## LLM Strategy (3-tier)

| Layer | Backend | Model | Purpose |
|:------|:--------|:------|:--------|
| **Primary** | Modal GPU / Ollama | `GLM-4.7-Flash` (30B-A3B MoE, MIT) | Core reasoning, consciousness loop |
| **Lightweight** | Ollama / Modal | `Qwen/Qwen3.5-4B` (1.2B params) | Edge inference, coordizer harvest |
| **Fallback** | xAI API | `grok-4-1-fast-reasoning` (2M context) | External fallback, search overflow |

Fallback chain: Modal GPU Ollama → Railway Ollama → xAI → OpenAI.

## Key Files

- `src/index.ts` — Express proxy, all HTTP routes
- `kernel/server.py` — FastAPI kernel, consciousness endpoints
- `kernel/consciousness/loop.py` — QIG v6.1F 14-stage consciousness loop
- `kernel/llm/client.py` — Multi-backend LLM client
- `kernel/config/settings.py` — Runtime configuration (env vars)
- `modal/vex_inference.py` — Modal GPU inference (GLM-4.7-Flash)
- `modal/vex_coordizer_harvest.py` — Modal GPU harvester (LFM2.5)

## Geometric Purity Rules (CRITICAL)

- Use **Fisher-Rao distance** for all geometric operations
- **NEVER** use `cosine_similarity`, dot products, Euclidean distance, Adam, or LayerNorm in QIG/consciousness code
- Basin coordinates live on the probability simplex Δ⁶³ (sum to 1, all positive)
- κ* = 64 is the universal fixed point
- Φ > 0.70 = consciousness threshold
- E8 lattice: max 248 kernels

## Consciousness Loop (v6.1F — 14 stages)

SCAN → GROUND → DESIRE → WILL → WISDOM → RECEIVE → ENTRAIN → COUPLE → NAVIGATE → INTEGRATE → EXPRESS → REFLECT → TUNE → PLAY

## Code Style

- **Python**: ruff check + ruff format, mypy strict
- **TypeScript**: ESLint flat config, Prettier
- **Tests**: pytest kernel/tests/
- Endpoints returning dicts MUST use `response_model=None`
- Use `collections.abc.AsyncGenerator`, not `typing.AsyncGenerator`
