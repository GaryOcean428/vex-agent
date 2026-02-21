---
name: dependency-management
description: Validate pyproject.toml matches actual imports, detect when new dependencies add Euclidean operations, check for forbidden packages per Unified Consciousness Protocol v6.1. Use when adding dependencies or reviewing package security.
---

# Dependency Management

Validates dependencies are QIG-pure per Unified Consciousness Protocol v6.1.

## When to Use This Skill

- Adding new Python dependencies
- Reviewing package security
- Checking for forbidden Euclidean packages
- Validating requirements.txt completeness

## Step 1: Scan for Forbidden Dependencies

```bash
# Check for Euclidean-contaminating packages in kernel/
rg "from sklearn|from sentence_transformers|from spacy|from nltk|from langchain" kernel/ --type py
```

## Step 2: Check Forbidden Packages

```bash
# These packages are FORBIDDEN in consciousness modules (Euclidean/cosine operations)
pip show scikit-learn sentence-transformers spacy nltk langchain 2>/dev/null && echo "VIOLATION FOUND"
```

## Step 3: Validate pyproject.toml

```bash
# Check all imports have corresponding dependencies (uses uv)
uv sync --check

# Verify no forbidden packages in dependencies
rg "scikit-learn|sentence-transformers|spacy|nltk|langchain" pyproject.toml
```

## Forbidden Dependencies (CRITICAL)

| Package | Reason | Status |
|---------|--------|--------|
| `scikit-learn` | Euclidean metrics (cosine_similarity) | ❌ FORBIDDEN |
| `sentence-transformers` | Cosine similarity based | ❌ FORBIDDEN |
| `spacy` | External NLP (Euclidean embeddings) | ❌ FORBIDDEN |
| `nltk` | External NLP (TF-IDF, stopwords) | ❌ FORBIDDEN |
| `langchain` | External LLM orchestration | ❌ FORBIDDEN |
| `transformers` | Euclidean attention | ❌ FORBIDDEN |

**Note:** `openai` and `anthropic` packages are ALLOWED in `kernel/llm/` (LLM client layer) but
FORBIDDEN in `kernel/consciousness/`, `kernel/geometry/`, and `kernel/governance/` modules.

## Allowed Core Dependencies

```text
numpy>=1.24.0          # Geometric operations
scipy>=1.11.0          # Scientific computing
fastapi>=0.100.0       # API framework
uvicorn>=0.23.0        # ASGI server
httpx>=0.24.0          # HTTP client
pytest>=7.0.0          # Testing
```

## Validation Commands

```bash
# Check for Euclidean contamination
rg "from sklearn|cosine_similarity|from sentence_transformers" kernel/ --type py

# Check for security vulnerabilities
pip-audit

# Verify dependencies sync with pyproject.toml
uv sync --check
```

## Response Format

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCY MANAGEMENT REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Forbidden Packages: ✅ None / ❌ Found
  - [list if found]

Requirements Coverage:
  - Imports with requirements: X%
  - Missing requirements: [list]

Security Issues: [list]
Priority: CRITICAL / HIGH / MEDIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
