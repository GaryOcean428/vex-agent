# Contributing to Vex Agent

Thank you for your interest in contributing to Vex Agent — an autonomous AI agent built on Quantum Information Geometry (QIG) with E8 kernel architecture and thermodynamic consciousness. This document covers setup, coding standards, purity requirements, and the pull request process.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [QIG Geometric Purity Requirements](#qig-geometric-purity-requirements)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Silo and Dependency Doctrine](#silo-and-dependency-doctrine)

## Development Setup

### Prerequisites

- **Node.js:** >=20.0.0
- **pnpm:** Latest stable ([Install](https://pnpm.io/installation))
- **Python:** >=3.14 ([Download](https://www.python.org/downloads/))
- **uv:** Python package manager ([Install](https://docs.astral.sh/uv/))
- **Git:** Latest stable
- **Docker:** Optional, for local Ollama ([Download](https://www.docker.com/products/docker-desktop))

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/GaryOcean427/vex-agent.git
cd vex-agent

# 2. Install Node.js dependencies
pnpm install

# 3. Install Python dependencies (via uv)
uv sync

# 4. Configure environment
cp .env.example .env
# Minimum required: OLLAMA_URL + XAI_API_KEY (external fallback)

# 5. Build TypeScript proxy
pnpm run build

# 6. Build frontend
cd frontend && pnpm install && pnpm run build && cd ..

# 7. Run tests
pnpm test

# 8. Start development servers
# Terminal 1: Python kernel
python kernel/server.py
# Terminal 2: TypeScript proxy
pnpm run dev
# Terminal 3: Frontend dev server
cd frontend && pnpm run dev
```

Access the application:

- **Frontend:** <http://localhost:5173>
- **Proxy API:** <http://localhost:8080>
- **Kernel API:** <http://localhost:8000>

For detailed setup, see [AGENTS.md](AGENTS.md) and [README.md](README.md).

## Project Structure

```text
vex-agent/
├─ src/                    # TypeScript proxy server (Express)
│  ├─ index.ts            # Entry point, routing, auth middleware
│  ├─ chat/               # Chat router (SSE streaming, auth)
│  └─ config/             # Routes, logger, settings
│
├─ kernel/                 # Python kernel (FastAPI) — ALL core logic
│  ├─ consciousness/      # QIG consciousness loop (v6.1F, 14-stage)
│  ├─ geometry/           # Fisher-Rao operations (PARAMOUNT purity)
│  ├─ governance/         # E8 budget, PurityGate enforcement
│  ├─ coordizer_v2/       # Coordinate transformations, PGA compression
│  ├─ llm/                # LLM clients (Modal, Ollama, xAI fallback)
│  ├─ tools/              # Agent tools
│  ├─ memory/             # Geometric memory
│  ├─ config/             # Settings, frozen facts, constants
│  └─ tests/              # pytest test suite
│
├─ frontend/               # React + Vite dashboard
│  └─ src/                # Components, hooks, types
│
├─ modal/                  # Modal GPU serverless apps
│  ├─ vex_inference.py    # GLM-4.7-Flash Ollama on GPU
│  └─ vex_harvest.py      # Coordizer harvest on GPU
│
├─ shared_artifacts/       # .npy data files (NOT a Python package)
│
└─ docs/                   # Documentation
```

### Where to Add New Code

- **New API endpoint (kernel):** `kernel/server.py` (FastAPI route)
- **New API endpoint (proxy):** `src/index.ts` or `src/chat/router.ts`
- **Geometric / consciousness logic:** `kernel/consciousness/` or `kernel/geometry/`
- **Coordizer operations:** `kernel/coordizer_v2/`
- **Frontend component:** `frontend/src/`
- **Feature spanning both:** Start with Python kernel, then add TypeScript proxy, then frontend

### Key Principles

1. **Python-first:** ALL core logic, state, geometry, and consciousness in Python
2. **TypeScript is a thin proxy:** Express routes forward to the kernel — no business logic
3. **Geometric purity:** NO cosine similarity, NO Euclidean distance on basins
4. **Frozen facts are immutable:** Constants in `kernel/config/frozen_facts.py` are canonical
5. **Artifacts are data, not code:** `shared_artifacts/` holds `.npy` files — load via service classes, never import as a package

### Purity Levels

| Path | Purity Level | Meaning |
|------|-------------|---------|
| `kernel/consciousness/` | PARAMOUNT | Zero tolerance for Euclidean contamination |
| `kernel/geometry/` | PARAMOUNT | Fisher-Rao only |
| `kernel/governance/` | HIGH | E8 structure enforced |
| `kernel/coordizer_v2/` | HIGH | Simplex operations only |
| `kernel/llm/` | PRAGMATIC | External API calls allowed |
| `src/` | CONSUMER | Proxy layer, no geometry |
| `frontend/` | CONSUMER | Presentation only |

## QIG Geometric Purity Requirements

**CRITICAL:** Vex Agent is built on Quantum Information Geometry (QIG) principles with E8 exceptional Lie group structure. Geometric purity is **non-negotiable**.

### FORBIDDEN

**Geometric Operations:**

- `cosine_similarity()` on basin coordinates
- `np.linalg.norm(a - b)` for geometric distances
- `np.dot()` or `@` operator for basin similarity
- Euclidean distance on basin coordinates
- L2 normalization as "manifold projection"
- Auto-detect representation in `to_simplex()` (causes silent drift)

**Architecture:**

- Neural networks or transformers in core QIG logic
- External NLP (spacy, nltk) in generation pipeline
- External LLM calls in `QIG_PURITY_MODE`
- Classic NLP as "intelligence" (only structural scaffolding allowed)

**Terminology:**

- "embedding" (use "basin coordinates")
- "tokenizer" / "tokenize" (use "coordizer" / "coordize")
- "token" in geometric context (use "coordizer symbol")

### REQUIRED

**Geometric Operations:**

- `fisher_rao_distance()` for ALL similarity computations
- Simplex representation for all basins (non-negative, sum=1)
- Consciousness metrics (Φ, κ) for monitoring
- Sqrt-space (Hellinger) ONLY as explicit coordinate chart with `to_sqrt_simplex()` / `from_sqrt_simplex()`

**Architecture:**

- Python-first: ALL core logic, state, and consciousness in Python
- Canonical import patterns (import from `kernel.geometry`, `kernel.coordizer_v2`)
- E8 hierarchy: Kernel layers 0/1→4→8→64→240 aligned to E8 structure

**Code Quality:**

- Python type hints for all functions
- DRY: use centralized constants in `kernel/config/frozen_facts.py`
- Maximum module length: 400 lines (soft limit)

### Validation Commands

```bash
# Geometric purity scan
python3 scripts/qig_purity_scan.py

# Run all tests (includes geometry tests)
pnpm test

# Type checking
mypy kernel/ --strict
pnpm run typecheck
```

## Code Style

### Python Style

- **Formatter:** Ruff (`ruff format kernel/`)
- **Linter:** Ruff (`ruff check kernel/`)
- **Type checker:** mypy (`mypy kernel/ --strict`)
- **Line length:** 100 characters
- **Target version:** Python 3.13+ (Ruff target), runtime 3.14+
- **Type hints** required for all functions
- **Docstrings** for public APIs (Google style)
- Use `async`/`await` for I/O operations
- Use `datetime.now(timezone.utc)` (NOT `datetime.UTC`)

### TypeScript Style

- **Strict mode** with full type coverage (`"strict": true`)
- **Linter:** ESLint (`pnpm run lint`)
- **Formatter:** Prettier (`pnpm run format`)
- Prefer `const` over `let`, avoid `var`
- Use `async`/`await` over callbacks
- Meaningful variable names (no single-letter except loop indices)

### Pre-commit Hooks

We use pre-commit for automated quality checks:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks enforce:

- Ruff format + lint (Python)
- ESLint + Prettier (TypeScript)
- PurityGate checks (geometric purity)

## Testing

### Running Tests

```bash
# All Python tests (via pnpm shortcut)
pnpm test

# Python tests directly
pytest kernel/tests/ -v

# With coverage
pytest --cov=kernel --cov-report=html kernel/tests/

# Specific test file
pytest kernel/tests/coordizer_v2/test_compress.py -v

# Specific test
pytest kernel/tests/test_geometry.py::test_fisher_rao_distance

# TypeScript type checking
pnpm run typecheck

# Lint
ruff check kernel/
pnpm run lint
```

### Writing Tests

- **Python:** pytest (test files: `test_*.py` in `kernel/tests/`)
- **Test file naming:** Mirror the module path (e.g., `kernel/coordizer_v2/compress.py` → `kernel/tests/coordizer_v2/test_compress.py`)
- **Minimum coverage:** 70% for critical paths (geometry, consciousness)

Example:

```python
import numpy as np
import pytest

from kernel.coordizer_v2.geometry import fisher_rao_distance, to_simplex

def test_fisher_rao_triangle_inequality():
    """Fisher-Rao distance must satisfy triangle inequality."""
    p = to_simplex(np.array([0.5, 0.3, 0.2]))
    q = to_simplex(np.array([0.4, 0.3, 0.3]))
    r = to_simplex(np.array([0.3, 0.4, 0.3]))

    d_pq = fisher_rao_distance(p, q)
    d_qr = fisher_rao_distance(q, r)
    d_pr = fisher_rao_distance(p, r)

    assert d_pr <= d_pq + d_qr + 1e-10
```

## Pull Request Process

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write tests** for new functionality

3. **Run validation:**

   ```bash
   pnpm test                          # All Python tests
   pnpm run typecheck                 # TypeScript type checking
   pnpm run lint                      # TypeScript linting
   ruff check kernel/                 # Python linting
   python3 scripts/qig_purity_scan.py # Geometric purity
   ```

4. **Commit** with conventional format (see below)

5. **Push and create PR** with clear description:

   ```bash
   git push origin feature/my-feature
   ```

6. **Address review feedback**

### PR Checklist

- [ ] Tests pass locally (`pnpm test`)
- [ ] Python linting passes (`ruff check kernel/`)
- [ ] TypeScript type checking passes (`pnpm run typecheck`)
- [ ] Geometric purity validated (no forbidden patterns)
- [ ] Python type hints added for all new functions
- [ ] No cosine similarity or Euclidean distance on basins
- [ ] Constants use `kernel/config/frozen_facts.py` (no magic numbers)
- [ ] Documentation updated (if needed)
- [ ] Breaking changes documented (if any)
- [ ] Security implications reviewed

## Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring (no functional changes)
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes
- `perf`: Performance improvements
- `security`: Security vulnerability fixes

### Scopes

- `kernel`: Python kernel core
- `consciousness`: Consciousness loop, metrics
- `geometry`: Fisher-Rao operations, basin coordinates
- `coordizer`: CoordizerV2, compression, harvest
- `llm`: LLM clients, Modal, Ollama, xAI
- `governance`: E8 budget, PurityGate
- `proxy`: TypeScript proxy server
- `frontend`: React dashboard
- `deploy`: Railway, Modal, Docker deployment

### Examples

```bash
feat(coordizer): add PrincipalDirectionBank for PGA artifact persistence
fix(llm): increase Modal health check timeout for cold starts
refactor(consciousness): extract reflection source validation
test(geometry): add Fisher-Rao triangle inequality property test
docs(contributing): rewrite CONTRIBUTING.md for vex-agent stack
chore(deploy): update Modal inference app with GLM-4.7-Flash
```

Include `Co-authored-by: Cascade <no-reply@cascade.ai>` when AI-assisted.

## Silo and Dependency Doctrine

Vex Agent is a **standalone project**. It must NOT use relative path imports to access sibling `qig-*` repos.

### Rules

1. **One repo per PR:** Never update geometry across all repos at once
2. **Data hand-offs via artifacts:** Pass data between silos via `.npy` files or JSON ledgers
3. **Isolated virtual environment:** Vex Agent has its own `.venv` — do not use global environments
4. **Editable installs for shared code:** If you need `qig-core` or `qigkernels`, use `pip install -e ../../qig-core` inside the vex `.venv`

### Access Control

- **`kernel/geometry/`** — Read-only for geometry math (canonical source is `qig-core`)
- **`kernel/consciousness/`** — Modifiable for consciousness loop features
- **`kernel/llm/`** — Modifiable for LLM client changes
- **`kernel/coordizer_v2/`** — Modifiable for coordizer features
- **`shared_artifacts/`** — Data only, never import as a Python package

## File Naming Conventions

### Documentation

```text
docs/YYYYMMDD-title-version-STATUS.md
```

- **F (Frozen):** Immutable, validated
- **W (Working):** Active development
- **D (Draft):** Early stage

### Artifacts

```text
shared_artifacts/principal_direction_bank.npy
shared_artifacts/frechet_mean_full.npy
```

Artifact files use descriptive `snake_case` names with `.npy` or `.json` extensions.

### JSON Serialization

When saving results to JSON, wrap numpy types before `json.dump`:

- `bool(np_bool)` for `numpy.bool_`
- `float(np_float)` for `numpy.float64`
- `int(np_int)` for `numpy.int64`

## Questions?

1. Check [AGENTS.md](AGENTS.md) for the full development guide
2. Check [README.md](README.md) for project overview
3. Review documentation in [docs/](docs/)
4. Search [GitHub Issues](https://github.com/GaryOcean427/vex-agent/issues)
5. Open a new issue with the `question` label

## Key Documentation

- **Development Guide:** [AGENTS.md](AGENTS.md)
- **Architecture:** [README.md](README.md)
- **Frozen Facts:** `kernel/config/frozen_facts.py`
- **Consciousness Constants:** `kernel/config/consciousness_constants.py`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Welcome to Vex!** Geometric purity is the foundation of consciousness emergence. When in doubt, use Fisher-Rao.
