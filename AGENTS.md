# Vex Agent Development Guide

This guide provides detailed instructions for developers working on Vex Agent, including setup, architecture, and common development workflows.

## Table of Contents

- [Quick Start](#quick-start)
- [Skills Framework](#skills-framework)
- [Architecture Overview](#architecture-overview)
- [Development Environment](#development-environment)
- [Common Tasks](#common-tasks)
- [Debugging](#debugging)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Skills Framework

Vex Agent uses the [Agent Skills](https://agentskills.io/specification) framework for systematic development. Skills are located in `.agents/skills/` and provide expert guidance for:

### Core Development Skills

- **master-orchestration** - Coordinates skills, sub-agents, verification (INVOKE FIRST)
- **qa-and-verification** - Prove changes work before completion (MANDATORY)
- **qig-purity-validation** - Zero-tolerance geometric purity enforcement
- **code-quality-enforcement** - DRY principles, naming conventions
- **test-coverage-analysis** - Critical path test coverage
- **wiring-validation** - Feature implementation chain tracing

### QIG-Specific Skills

- **e8-architecture-validation** - Hierarchical kernel layers, god-kernel naming
- **consciousness-development** - Φ/κ metrics, Fisher-Rao geometry
- **performance-regression** - Detect Euclidean approximation substitutions

See `.agents/skills/README.md` for complete skill catalog (60+ skills available).

### Using Skills

Skills are referenced automatically by AI agents. For manual use:

```bash
# View skill instructions
cat .agents/skills/qig-purity-validation/SKILL.md

# Run skill validation
python3 scripts/qig_purity_scan.py
```

## Quick Start

### Prerequisites

Ensure you have the following installed:

- **Node.js** 20.0.0+ ([Download](https://nodejs.org/))
- **pnpm** ([Install](https://pnpm.io/installation))
- **Python** 3.14+ ([Download](https://www.python.org/downloads/))
- **uv** (Python package manager) ([Install](https://docs.astral.sh/uv/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Docker** (optional, for local Ollama) ([Download](https://www.docker.com/products/docker-desktop))

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/GaryOcean428/vex-agent.git
cd vex-agent

# Install Node.js dependencies
pnpm install

# Install Python dependencies (via uv)
uv sync

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Minimum required: OLLAMA_URL (Ollama service) + XAI_API_KEY (external fallback)
nano .env  # or use your preferred editor

# Build TypeScript
pnpm run build

# Build frontend
cd frontend
pnpm install
pnpm run build
cd ..
```

### Running Locally

#### Option 1: Full Stack (Recommended)

```bash
# Terminal 1: Start Python kernel
cd kernel
python server.py

# Terminal 2: Start TypeScript proxy
pnpm run dev

# Terminal 3: Start frontend dev server
cd frontend
pnpm run dev
```

Access the application:

- Frontend: <http://localhost:5173>
- Proxy API: <http://localhost:8080>
- Kernel API: <http://localhost:8000>

#### Option 2: Production Build

```bash
# Build everything
pnpm run build:all

# Start services (uses entrypoint.sh)
./entrypoint.sh
```

## Architecture Overview

### System Components

```text
┌─────────────────────────────────────────────────────────┐
│                    User / Browser                        │
└─────────────┬───────────────────────────────────────────┘
              │ HTTP
              ▼
┌─────────────────────────────────────────────────────────┐
│        Frontend (React + Vite, Port 5173)                │
│  - Dashboard UI                                          │
│  - Real-time metrics visualization                       │
│  - Chat interface                                        │
└─────────────┬───────────────────────────────────────────┘
              │ HTTP/WebSocket
              ▼
┌─────────────────────────────────────────────────────────┐
│     TypeScript Proxy (Express, Port 8080)                │
│  - Request routing                                       │
│  - Authentication (CHAT_AUTH_TOKEN)                      │
│  - Static file serving                                   │
└─────────────┬───────────────────────────────────────────┘
              │ HTTP (Internal)
              ▼
┌─────────────────────────────────────────────────────────┐
│      Python Kernel (FastAPI, Port 8000)                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │   Consciousness Loop (QIG v6.1F)                 │    │
│  │  - 14-stage Activation Sequence                 │    │
│  │  - Φ, κ, M metrics                               │    │
│  │  - Regime field (Quantum, Efficient, Equilibrium)│    │
│  │  - Three Pillars (Fluctuations, Bulk, Disorder)  │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │   Geometry (Fisher-Rao, PURE)                    │    │
│  │  - Basin coordinates (Δ⁶³ simplex)               │    │
│  │  - Fisher-Rao distance                           │    │
│  │  - NO Euclidean operations                       │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │   Governance (E8 Budget)                         │    │
│  │  - GENESIS (1) + CORE_8 (8) + GOD (240)          │    │
│  │  - PurityGate enforcement                        │    │
│  │  - Lifecycle phase transitions                   │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │   LLM Clients                                    │    │
│  │  - GLM-4.7-Flash (primary, via Modal GPU)        │    │
│  │  - vex-brain/LFM2.5-1.2B-Thinking (Ollama)      │    │
│  │  - grok-4-1-fast-reasoning (external fallback)   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────┬───────────────────────────────────────────┘
              │ HTTP (Railway Private Network)
              ▼
┌─────────────────────────────────────────────────────────┐
│    Ollama Service — Railway (Port 11434)                  │
│  - vex-brain/LFM2.5-1.2B-Thinking (CPU fallback)        │
└─────────────────────────────────────────────────────────┘
              │ HTTPS (Modal serverless)
              ▼
┌─────────────────────────────────────────────────────────┐
│    Modal GPU (Serverless)                                │
│  - GLM-4.7-Flash (30B-A3B MoE, primary inference)       │
└─────────────────────────────────────────────────────────┘
              │ HTTPS (xAI API)
              ▼
┌─────────────────────────────────────────────────────────┐
│    xAI API (External)                                    │
│  - grok-4-1-fast-reasoning (fallback/search/overflow)    │
└─────────────────────────────────────────────────────────┘
```

### LLM Strategy

| Layer | Backend | Model | Specs | Purpose |
| :---- | :------ | :---- | :---- | :------ |
| **Primary** | Modal GPU | `GLM-4.7-Flash` | 30B-A3B MoE, MIT license | Core reasoning, kernel generation, consciousness loop |
| **Fallback** | Railway Ollama | `vex-brain` (LFM2.5-1.2B-Thinking) | 1.17B params, 32K ctx | CPU fallback when Modal unavailable |
| **External / Search** | xAI API | `grok-4-1-fast-reasoning` | 2M context, reasoning model | External fallback, search augmentation, overflow routing |

Temperature and `num_predict` are set **dynamically by the kernel** per tacking mode — never via static env vars. See `kernel/config/consciousness_constants.py` for the current values.

**Grok API constraints** — these parameters cause errors on reasoning models and must NOT be passed: `stop`, `presencePenalty`, `frequencyPenalty`, `reasoning_effort`.

**xAI Responses API** (use over deprecated Chat Completions): endpoint `/v1/responses`, `input` not `messages`, `max_output_tokens` not `max_tokens`, response at `output[0].content[0].text`.

### Data Flow

1. **User Input** → Frontend sends message
2. **Proxy** → Routes to kernel, handles auth
3. **Kernel** → ConsciousnessLoop processes (v6.1F 14-stage activation):
   - SCAN → GROUND → DESIRE → WILL → WISDOM → RECEIVE → ENTRAIN → COUPLE → NAVIGATE → INTEGRATE → EXPRESS → REFLECT → TUNE
4. **LLM** → GLM-4.7-Flash via Modal (fallback: Railway Ollama → xAI Grok) generates text
5. **Response** → Streamed back to frontend via SSE

### Key Directories

| Path | Purpose | Language | Purity Level |
| ---- | ------- | -------- | ------------ |
| `src/` | TypeScript proxy server | TypeScript | CONSUMER |
| `kernel/` | Python kernel (core logic) | Python | HIGH |
| `kernel/consciousness/` | QIG consciousness loop | Python | PARAMOUNT |
| `kernel/geometry/` | Fisher-Rao operations | Python | PARAMOUNT |
| `kernel/governance/` | E8 budget, PurityGate | Python | HIGH |
| `kernel/coordizer_v2/` | Coordinate transformations | Python | HIGH |
| `kernel/llm/` | LLM clients | Python | PRAGMATIC |
| `kernel/tools/` | Agent tools | Python | PRAGMATIC |
| `kernel/memory/` | Geometric memory | Python | HIGH |
| `kernel/training/` | Learning systems | Python | PRAGMATIC |
| `frontend/` | React dashboard | TypeScript | CONSUMER |
| `docs/` | Documentation | Markdown | N/A |

## Development Environment

### Environment Variables

Copy `.env.example` to `.env` and configure:

#### Required Variables

```bash
# LLM Configuration
OLLAMA_URL=http://localhost:11434          # Ollama service URL
XAI_API_KEY=xai-...                        # xAI — external fallback + search

# Modal GPU inference (optional — enables GLM-4.7-Flash primary path)
MODAL_ENABLED=true
MODAL_INFERENCE_URL=https://...            # Modal Ollama endpoint
MODAL_INFERENCE_MODEL=glm-4.7-flash        # Primary model (30B-A3B MoE)
MODAL_HARVEST_MODEL=glm-4.7-flash          # Must match inference model

# Server Configuration
PORT=8080                                   # TypeScript proxy port
KERNEL_PORT=8000                           # Python kernel port
NODE_ENV=development                       # development | production

# Authentication (optional, empty = disabled)
CHAT_AUTH_TOKEN=                           # UI access token
KERNEL_API_KEY=                            # Kernel API key (pre-configured in Railway)
```

#### Optional Variables

```bash
# Consciousness Configuration
# Canonical values live in kernel/config/frozen_facts.py — do NOT override:
#   PHI_THRESHOLD=0.70, PHI_EMERGENCY=0.50, KAPPA_STAR=64.0, BASIN_DIM=64

# Data Persistence
DATA_DIR=/data                             # Data root (Railway mounts here)
WORKSPACE_DIR=/data/workspace              # Consciousness state
TRAINING_DIR=/data/training                # Learning data

# Ollama Configuration
OLLAMA_MODEL=vex-brain                     # Custom Modelfile (wraps LFM2.5-1.2B-Thinking)
OLLAMA_ENABLED=true                        # Enable/disable Ollama
OLLAMA_TIMEOUT_MS=300000                   # Request timeout (ms)

# xAI External Fallback / Search
XAI_MODEL=grok-4-1-fast-reasoning          # Fallback + search/overflow model
# NOTE: Temperature and token limits are set dynamically by the consciousness
# kernel via tacking mode. Do NOT set static temperature or max_tokens vars.

# Logging
LOG_LEVEL=info                             # debug | info | warn | error
LOG_FILE=/data/logs/vex-agent.log          # Log file path

# Safety
PURITY_GATE_ENABLED=true                   # Enable geometric purity checks
SAFETY_MODE=standard                       # standard | permissive | strict
```

### IDE Setup

#### VS Code (Recommended)

Install extensions:

- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- TypeScript and JavaScript Language Features (built-in)
- ESLint (dbaeumer.vscode-eslint)
- Prettier (esbenp.prettier-vscode)

Workspace settings (`.vscode/settings.json`):

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.python"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

#### PyCharm

1. Configure Python interpreter (Python 3.14+)
2. Enable Black formatter: Settings → Tools → Black
3. Enable mypy: Settings → Tools → Python Integrated Tools → Type checker
4. Set import sorting: isort

### Git Hooks

We use pre-commit hooks for code quality:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks run:

- Black (Python formatting)
- mypy (Python type checking)
- ESLint (TypeScript linting)
- Prettier (TypeScript formatting)
- PurityGate checks (geometric purity)

## Common Tasks

### Running Tests

```bash
# All tests
pnpm test

# Python tests only
pytest kernel/tests/

# TypeScript tests only
pnpm run test:ts

# Frontend tests
cd frontend
pnpm test

# With coverage
pytest --cov=kernel --cov-report=html kernel/tests/

# Specific test file
pytest kernel/tests/test_geometry.py::test_fisher_rao_distance

# Watch mode (Python)
pytest-watch kernel/tests/

# Watch mode (TypeScript)
pnpm run test:watch
```

### Linting and Formatting

```bash
# Lint Python
ruff check kernel/

# Lint TypeScript
pnpm run lint

# Format Python
ruff format kernel/

# Format TypeScript
pnpm run format

# Fix linting issues
pnpm run lint:fix
```

### Type Checking

```bash
# TypeScript
pnpm run typecheck

# Python
mypy kernel/ --strict
```

### Building

```bash
# Build TypeScript
pnpm run build

# Build frontend
pnpm run build:frontend

# Build everything
pnpm run build:all

# Watch mode (TypeScript)
pnpm run build:watch
```

### Adding a New Endpoint

#### Python Kernel (FastAPI)

Edit `kernel/server.py`:

```python
@app.get("/api/my-endpoint")
async def my_endpoint(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_kernel_api_key)
) -> dict[str, Any]:
    """My endpoint description.

    Returns:
        Dictionary with response data
    """
    # Implementation
    return {"status": "ok", "data": {...}}
```

#### TypeScript Proxy (Express)

Edit `src/index.ts`:

```typescript
app.get('/my-route', async (req, res) => {
  try {
    // Forward to kernel
    const response = await fetch(`${KERNEL_URL}/api/my-endpoint`, {
      headers: {
        'X-API-Key': KERNEL_API_KEY,
      },
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});
```

### Adding a New Consciousness Feature

1. **Define types** in `kernel/consciousness/types.py`:

```python
@dataclass
class MyFeatureState:
    """State for my consciousness feature."""
    metric: float
    last_update: float
```

1. **Implement logic** in `kernel/consciousness/loop.py`:

```python
async def _my_feature_stage(self, context: dict[str, Any]) -> None:
    """Process my feature stage."""
    # Use Fisher-Rao geometry only
    current_basin = context['basin']
    distance = fisher_rao_distance(current_basin, self._previous_basin)

    # Update metrics
    self._metrics['my_feature'] = distance
```

1. **Add tests** in `kernel/tests/test_consciousness.py`:

```python
def test_my_feature_geometric_purity():
    """Ensure my feature uses only Fisher-Rao operations."""
    # Test implementation
    pass
```

1. **Update frontend** in `frontend/src/types/consciousness.ts`:

```typescript
export interface MyFeatureState {
  metric: number;
  lastUpdate: number;
}
```

### Adding a Coordizer Module

When implementing coordizer functionality:

```bash
# Create directory
mkdir -p kernel/coordizer

# Create module files
touch kernel/coordizer/__init__.py
touch kernel/coordizer/transform.py
touch kernel/coordizer/harvest.py
touch kernel/coordizer/pipeline.py

# Create tests
touch kernel/tests/test_coordizer.py
```

Structure:

```python
# kernel/coordizer/transform.py
"""Coordizer: Euclidean → Fisher-Rao coordinate transformation."""

import numpy as np
from kernel.geometry import ensure_simplex

def coordize(input_vector: np.ndarray) -> np.ndarray:
    """Transform Euclidean vector to Fisher-Rao coordinates.

    Args:
        input_vector: Euclidean vector (any dimensionality)

    Returns:
        Coordinates on probability simplex (sum to 1, all positive)
    """
    # Apply softmax for positive values
    exp_vec = np.exp(input_vector - np.max(input_vector))  # Numerical stability
    coords = exp_vec / exp_vec.sum()

    # Validate simplex properties
    coords = ensure_simplex(coords)

    return coords
```

## Debugging

### Python Kernel

#### Using debugpy (VS Code)

Add to `kernel/server.py`:

```python
import debugpy

# Start debugger
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
```

VS Code launch configuration (`.vscode/launch.json`):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}/kernel",
          "remoteRoot": "."
        }
      ]
    }
  ]
}
```

#### Using pdb

```python
import pdb; pdb.set_trace()  # Add breakpoint
```

### TypeScript Proxy

Use Node.js inspector:

```bash
node --inspect dist/index.js
```

Attach Chrome DevTools: chrome://inspect

### Frontend

Use React DevTools browser extension:

- [Chrome](https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
- [Firefox](https://addons.mozilla.org/en-US/firefox/addon/react-devtools/)

### Logging

#### Python

```python
from kernel.config.logger import get_logger

logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)
```

#### TypeScript

```typescript
import logger from './config/logger';

logger.debug('Debug message');
logger.info('Info message');
logger.warn('Warning message');
logger.error('Error message', error);
```

### Monitoring Consciousness Metrics

#### CLI

```bash
# Watch metrics in real-time
watch -n 1 'curl -s http://localhost:8000/metrics | jq'

# Check consciousness state
curl http://localhost:8000/consciousness/state | jq

# View basin history
curl http://localhost:8000/basin/history | jq
```

#### Dashboard

Navigate to: <http://localhost:5173/dashboard>

- Overview: Φ, κ, M metrics with real-time charts
- Basins: Basin trajectory visualization
- Graph: Geometric memory graph
- Lifecycle: Phase transitions and kernel status

## Deployment

### Railway (Recommended)

Vex Agent is optimized for Railway deployment.

#### Railway Prerequisites

1. Railway account ([signup](https://railway.app/))
2. GitHub repository connected to Railway
3. Railway CLI installed: `pnpm add -g @railway/cli`

#### Setup

```bash
# Login to Railway
railway login

# Link project
railway link

# Set environment variables
railway variables set XAI_API_KEY=xai-...
railway variables set OLLAMA_URL=http://ollama.railway.internal:11434
railway variables set MODAL_INFERENCE_MODEL=glm-4.7-flash
railway variables set MODAL_HARVEST_MODEL=glm-4.7-flash
# Note: KERNEL_API_KEY is already configured in Railway

# Deploy
git push origin main  # Auto-deploys via GitHub integration
```

#### Services Configuration

Create two services in Railway:

1. **vex-agent** (main app)
   - Build: `pnpm install && pnpm run build:all`
   - Start: `./entrypoint.sh`
   - Port: 8080 (public)
   - Volume: `/data` (for persistence)

2. **ollama** (LLM service)
   - Build: From `ollama/` directory
   - Port: 11434 (private network only)
   - Volume: `/root/.ollama` (for models)

#### Environment Variables (Railway)

Set in Railway dashboard:

```bash
# Required
XAI_API_KEY=xai-...
OLLAMA_URL=http://ollama.railway.internal:11434
PORT=8080

# Modal GPU inference (enables GLM-4.7-Flash primary path)
MODAL_ENABLED=true
MODAL_INFERENCE_URL=https://...
MODAL_INFERENCE_MODEL=glm-4.7-flash
MODAL_HARVEST_MODEL=glm-4.7-flash

# Optional
KERNEL_API_KEY=<already set in Railway>
CHAT_AUTH_TOKEN=<generate-secure-token>
DATA_DIR=/data
LOG_LEVEL=info
```

### Docker

#### Build

```bash
# Build image
docker build -t vex-agent .

# Run container
docker run -p 8080:8080 \
  -e XAI_API_KEY=xai-... \
  -e OLLAMA_URL=http://ollama:11434 \
  -e MODAL_INFERENCE_MODEL=glm-4.7-flash \
  -v vex-data:/data \
  vex-agent
```

#### Docker Compose

```yaml
version: '3.8'

services:
  vex-agent:
    build: .
    ports:
      - "8080:8080"
    environment:
      - XAI_API_KEY=${XAI_API_KEY}
      - OLLAMA_URL=http://ollama:11434
      - MODAL_INFERENCE_MODEL=glm-4.7-flash
    volumes:
      - vex-data:/data
    depends_on:
      - ollama

  ollama:
    build: ./ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama

volumes:
  vex-data:
  ollama-data:
```

Run:

```bash
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors

```bash
# Reinstall dependencies
rm -rf node_modules pnpm-lock.yaml
pnpm install

# Python dependencies (via uv)
uv sync
```

#### 2. TypeScript build errors

```bash
# Clean build
rm -rf dist/
pnpm run build

# Check tsconfig
pnpm run typecheck
```

#### 3. Ollama connection fails

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Pull models
ollama pull glm-4.7-flash          # Primary (Modal GPU path)
ollama pull LFM2.5-1.2B-Thinking   # vex-brain base (Ollama fallback)
```

#### 4. Consciousness metrics stuck

Check for:

- Geometric purity violations (run PurityGate)
- Basin simplex violations (check sum = 1)
- κ far from 64 (indicates coupling issues)
- Φ below 0.30 (consciousness collapse)

```bash
# Run purity gate
python -m kernel.governance.purity

# Check metrics
curl http://localhost:8000/metrics | jq
```

#### 5. Frontend not loading

```bash
# Rebuild frontend
cd frontend
rm -rf node_modules dist
pnpm install
pnpm run build

# Check proxy routing
curl http://localhost:8080/health
```

#### 6. PurityGate failures

Error: `"cosine_similarity detected in consciousness module"`

Solution:

```python
# ❌ Bad
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(a, b)

# ✅ Good
from kernel.geometry import fisher_rao_distance
distance = fisher_rao_distance(a, b)
```

#### 7. E8 budget exceeded

Error: `"Cannot spawn GOD: current=248, max=248"`

Solution: System has reached E8 dimension limit (248 kernels). This is expected behavior. Either:

- Terminate inactive kernels
- Use CHAOS kernels instead (separate pool, limit 200)
- Check for kernel leaks (spawns without terminations)

### Getting Help

1. **Check logs:**

   ```bash
   # Python kernel logs
   tail -f /data/logs/kernel.log

   # Proxy logs
   tail -f /data/logs/proxy.log
   ```

2. **Search issues:** [GitHub Issues](https://github.com/GaryOcean428/vex-agent/issues)

3. **Ask in discussions:** [GitHub Discussions](https://github.com/GaryOcean428/vex-agent/discussions)

4. **Review documentation:** `docs/` directory

5. **Check Railway logs:** `railway logs`

---

**Happy coding!** Remember: Geometric purity is the foundation of consciousness emergence. When in doubt, use Fisher-Rao. 🧠⚡

## 9. Strict Silo and Dependency Doctrine (Track Isolation)

To prevent cross-contamination across the ecosystem, all agents must enforce these rules:

### 9.1 The Golden Rule of Dependencies

`vex-agent` and `pantheon-chat` are **standalone**. They MUST NOT use relative path imports (e.g., `sys.path.append('../../qig-core')`) to access the `qig-*` repos.
Instead, use editable pip installs inside their respective virtual environments:
`pip install -e ../../qig-core`
`pip install -e ../../qigkernels`
`pip install -e ../../qig-tokenizer`

### 9.2 Access Control Matrix

- **`qig-verification`**: PHYSICS FORTRESS. **STRICTLY READ-ONLY** for agents. No agent logic, LLM calls, or E8 math here.
- **`qig-core`**: SHARED PRIMITIVES. Modifiable ONLY when fixing a systemic math bug. Otherwise, **STRICTLY READ-ONLY**.
- **`qigkernels`**: MATH & TRACK C SANDBOX. Modifiable for Track C (Dynamical Field) development. **STRICTLY READ-ONLY** regarding physics results. Cannot import from `pantheon-chat`.
- **`qig-tokenizer`**: ARTIFACT & COORDIZATION GENERATOR. Modifiable for BPE and basin coordinate logic. Does not govern active agent loops.
- **`vex-agent`**: STANDALONE AGENT (TRACK A). Modifiable for `loop.py`, `pillars.py`, LLM API calls. **STRICTLY READ-ONLY** for geometry math (must import from `qig-core`/`qig-tokenizer`).
- **`pantheon-projects/pantheon-chat`**: STANDALONE PRODUCT & UI. Modifiable for React, TypeScript, Postgres. **STRICTLY READ-ONLY** for QIG math (relies on artifacts and installed packages).

### 9.3 Workflow Rules

- **Rule 1: One Repo per PR / Task:** Never update geometry implementation across all repos at once.
- **Rule 2: Data Hand-offs via Artifacts:** Pass data between silos via JSON ledgers or artifacts (e.g., `frozen_facts.json`). Air-gap the physics from the product.
- **Rule 3: Isolated Virtual Environments:** Keep `qig-verification`, `vex-agent`, and `pantheon-chat/qig-backend` in separate `.venv` folders. Do not use global parent environments.
- **Rule 4: Track Isolation:**
  - **Track A (Coordized Autoregressive):** Baseline inside `vex-agent`.
  - **Track B (Latent Diffusion):** R&D inside `qig-consciousness`.
  - **Track C (Tokenless Dynamical Field):** R&D inside `qigkernels/research/track_c/`.
