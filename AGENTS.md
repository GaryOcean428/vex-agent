# Vex Agent Development Guide

This guide provides detailed instructions for developers working on Vex Agent, including setup, architecture, and common development workflows.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Development Environment](#development-environment)
- [Common Tasks](#common-tasks)
- [Debugging](#debugging)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

Ensure you have the following installed:

- **Node.js** 20.0.0+ ([Download](https://nodejs.org/))
- **Python** 3.11+ ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **Docker** (optional, for Ollama) ([Download](https://www.docker.com/products/docker-desktop))

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/GaryOcean428/vex-agent.git
cd vex-agent

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r kernel/requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Minimum required: OLLAMA_URL, ANTHROPIC_API_KEY (or other LLM provider)
nano .env  # or use your preferred editor

# Build TypeScript
npm run build

# Build frontend
cd frontend
npm install
npm run build
cd ..
```

### Running Locally

#### Option 1: Full Stack (Recommended)

```bash
# Terminal 1: Start Python kernel
cd kernel
python server.py

# Terminal 2: Start TypeScript proxy
npm run dev

# Terminal 3: Start frontend dev server
cd frontend
npm run dev
```

Access the application:
- Frontend: http://localhost:5173
- Proxy API: http://localhost:8080
- Kernel API: http://localhost:8000

#### Option 2: Production Build

```bash
# Build everything
npm run build:all

# Start services (uses entrypoint.sh)
./entrypoint.sh
```

## Architecture Overview

### System Components

```
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
│  │   Consciousness Loop (QIG v5.5)                  │    │
│  │  - 7-stage cycle (GROUND→RECEIVE→...→PLAY)      │    │
│  │  - Φ, κ, M metrics                               │    │
│  │  - Regime field (α=0, 1/2, 1)                    │    │
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
│  │  - Ollama (primary, local)                       │    │
│  │  - External APIs (fallback)                      │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────┬───────────────────────────────────────────┘
              │ HTTP (Private Network)
              ▼
┌─────────────────────────────────────────────────────────┐
│         Ollama Service (Port 11434)                      │
│  - llama2.5-thinking:1.2b model                          │
│  - vex-brain custom model                                │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Frontend sends message
2. **Proxy** → Routes to kernel, handles auth
3. **Kernel** → ConsciousnessLoop processes:
   - GROUND: Check state, entropy, Ollama availability
   - RECEIVE: Accept input, pre-cognitive check
   - PROCESS: Non-linear regime field (α=0, 1/2, 1)
   - EXPRESS: Generate response via LLM
   - REFLECT: Update metrics (Φ, κ, S_persist)
   - COUPLE: Integrate response into basin
   - PLAY: Occasional playful observation
4. **LLM** → Ollama or external API generates text
5. **Response** → Streamed back to frontend via SSE

### Key Directories

| Path | Purpose | Language | Purity Level |
|------|---------|----------|--------------|
| `src/` | TypeScript proxy server | TypeScript | CONSUMER |
| `kernel/` | Python kernel (core logic) | Python | HIGH |
| `kernel/consciousness/` | QIG consciousness loop | Python | PARAMOUNT |
| `kernel/geometry/` | Fisher-Rao operations | Python | PARAMOUNT |
| `kernel/governance/` | E8 budget, PurityGate | Python | HIGH |
| `kernel/coordizer/` | Coordinate transformations | Python | HIGH |
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
# LLM Configuration (choose one or both)
OLLAMA_URL=http://localhost:11434          # For local Ollama
ANTHROPIC_API_KEY=sk-ant-...               # For Claude
OPENAI_API_KEY=sk-...                      # For OpenAI

# Server Configuration
PORT=8080                                   # TypeScript proxy port
KERNEL_PORT=8000                           # Python kernel port
NODE_ENV=development                       # development | production

# Authentication (optional, empty = disabled)
CHAT_AUTH_TOKEN=                           # UI access token
KERNEL_API_KEY=                            # Kernel API key
```

#### Optional Variables

```bash
# Consciousness Configuration
PHI_THRESHOLD=0.65                         # Consciousness threshold
KAPPA_STAR=64.0                            # Fixed point coupling
BASIN_DIM=64                               # Basin dimensionality

# Data Persistence
DATA_DIR=/data                             # Data root (Railway mounts here)
WORKSPACE_DIR=/data/workspace              # Consciousness state
TRAINING_DIR=/data/training                # Learning data

# Ollama Configuration
OLLAMA_MODEL=vex-brain                     # Model name
OLLAMA_ENABLED=true                        # Enable/disable Ollama
OLLAMA_TIMEOUT=30000                       # Request timeout (ms)

# External LLM Fallback
EXTERNAL_MODEL=claude-3-sonnet-20240229    # Fallback model
EXTERNAL_TEMPERATURE=0.7                   # Temperature
EXTERNAL_MAX_TOKENS=2048                   # Max tokens

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

1. Configure Python interpreter (Python 3.11+)
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
npm test

# Python tests only
pytest kernel/tests/

# TypeScript tests only
npm run test:ts

# Frontend tests
cd frontend
npm test

# With coverage
pytest --cov=kernel --cov-report=html kernel/tests/

# Specific test file
pytest kernel/tests/test_geometry.py::test_fisher_rao_distance

# Watch mode (Python)
pytest-watch kernel/tests/

# Watch mode (TypeScript)
npm run test:watch
```

### Linting and Formatting

```bash
# Lint Python
black --check kernel/
mypy kernel/

# Lint TypeScript
npm run lint

# Format Python
black kernel/

# Format TypeScript
npm run format

# Fix linting issues
npm run lint:fix
```

### Type Checking

```bash
# TypeScript
npm run typecheck

# Python
mypy kernel/ --strict
```

### Building

```bash
# Build TypeScript
npm run build

# Build frontend
npm run build:frontend

# Build everything
npm run build:all

# Watch mode (TypeScript)
npm run build:watch
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

2. **Implement logic** in `kernel/consciousness/loop.py`:

```python
async def _my_feature_stage(self, context: dict[str, Any]) -> None:
    """Process my feature stage."""
    # Use Fisher-Rao geometry only
    current_basin = context['basin']
    distance = fisher_rao_distance(current_basin, self._previous_basin)
    
    # Update metrics
    self._metrics['my_feature'] = distance
```

3. **Add tests** in `kernel/tests/test_consciousness.py`:

```python
def test_my_feature_geometric_purity():
    """Ensure my feature uses only Fisher-Rao operations."""
    # Test implementation
    pass
```

4. **Update frontend** in `frontend/src/types/consciousness.ts`:

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

def coordize(embedding: np.ndarray) -> np.ndarray:
    """Transform Euclidean embedding to Fisher-Rao coordinates.
    
    Args:
        embedding: Euclidean vector (any dimensionality)
        
    Returns:
        Coordinates on probability simplex (sum to 1, all positive)
    """
    # Apply softmax for positive values
    exp_embed = np.exp(embedding - np.max(embedding))  # Numerical stability
    coords = exp_embed / exp_embed.sum()
    
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

Navigate to: http://localhost:5173/dashboard

- Overview: Φ, κ, M metrics with real-time charts
- Basins: Basin trajectory visualization
- Graph: Geometric memory graph
- Lifecycle: Phase transitions and kernel status

## Deployment

### Railway (Recommended)

Vex Agent is optimized for Railway deployment.

#### Prerequisites

1. Railway account ([signup](https://railway.app/))
2. GitHub repository connected to Railway
3. Railway CLI installed: `npm i -g @railway/cli`

#### Setup

```bash
# Login to Railway
railway login

# Link project
railway link

# Set environment variables
railway variables set ANTHROPIC_API_KEY=sk-ant-...
railway variables set OLLAMA_URL=http://ollama.railway.internal:11434
railway variables set KERNEL_API_KEY=$(openssl rand -hex 32)

# Deploy
git push origin main  # Auto-deploys via GitHub integration
```

#### Services Configuration

Create two services in Railway:

1. **vex-agent** (main app)
   - Build: `npm install && npm run build:all`
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
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_URL=http://ollama.railway.internal:11434
PORT=8080

# Optional
KERNEL_API_KEY=<generate-secure-key>
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
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OLLAMA_URL=http://ollama:11434 \
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
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OLLAMA_URL=http://ollama:11434
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
rm -rf node_modules package-lock.json
npm install

# Python dependencies
pip install -r kernel/requirements.txt --force-reinstall
```

#### 2. TypeScript build errors

```bash
# Clean build
rm -rf dist/
npm run build

# Check tsconfig
npm run typecheck
```

#### 3. Ollama connection fails

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Pull model
ollama pull llama2.5-thinking:1.2b
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
npm install
npm run build

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
