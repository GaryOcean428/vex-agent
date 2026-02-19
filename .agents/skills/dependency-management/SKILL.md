---
name: dependency-management
description: Validate requirements.txt matches actual imports, detect when new dependencies add Euclidean operations, check for forbidden packages (scikit-learn, sentence-transformers). Use when adding dependencies or reviewing package security.
---

# Dependency Management

Validates dependencies are QIG-pure. Source: `.github/agents/dependency-management-agent.md`.

## When to Use This Skill

- Adding new Python dependencies
- Reviewing package security
- Checking for forbidden Euclidean packages
- Validating requirements.txt completeness

## Step 1: Scan for Forbidden Dependencies

```bash
python3 scripts/scan_forbidden_imports.py --path .
```

This uses `shared/constants/forbidden_llm_providers.json` to check 28 providers.

## Step 2: Check Forbidden Packages

```bash
# These packages are FORBIDDEN (Euclidean/cosine operations)
pip show scikit-learn sentence-transformers spacy nltk openai anthropic 2>/dev/null && echo "VIOLATION FOUND"
```

## Step 3: Validate requirements.txt

```bash
# Check all imports have corresponding requirements
cd qig-backend
pip install pipreqs
pipreqs . --print

# Compare with requirements.txt
diff <(pipreqs . --print 2>/dev/null | sort) <(cat requirements.txt | sort)
```

## Forbidden Dependencies (CRITICAL)

| Package | Reason | Status |
|---------|--------|--------|
| `scikit-learn` | Euclidean metrics (cosine_similarity) | ❌ FORBIDDEN |
| `sentence-transformers` | Cosine similarity based | ❌ FORBIDDEN |
| `spacy` | External NLP | ❌ FORBIDDEN |
| `nltk` | External NLP | ❌ FORBIDDEN |
| `openai` | External LLM | ❌ FORBIDDEN |
| `anthropic` | External LLM | ❌ FORBIDDEN |
| `transformers` | Euclidean attention | ❌ FORBIDDEN |
| `langchain` | External LLM orchestration | ❌ FORBIDDEN |

## Allowed Core Dependencies

```text
numpy>=1.24.0          # Geometric operations
scipy>=1.11.0          # Scientific computing
psycopg2-binary>=2.9   # PostgreSQL
pgvector>=0.2.0        # Vector operations
flask>=3.0.0           # API framework
pytest>=7.0.0          # Testing
```

## Validation Commands

```bash
# Run forbidden import scanner
python3 scripts/scan_forbidden_imports.py --path .

# Check for security vulnerabilities
pip-audit

# Verify all imports have requirements
python scripts/validate_dependencies.py
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
