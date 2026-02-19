---
name: documentation-compliance
description: Ensure ISO 27001 documentation standards, validate document naming conventions (YYYYMMDD-name-function-versionSTATUS.md), detect code changes that invalidate documentation, and maintain synchronization between code and docs.
---

# Documentation Compliance

Expert skill for ensuring ISO 27001 documentation standards, validating document naming conventions, and maintaining synchronization between code and documentation.

## When to Use This Skill

Use this skill when:

- Creating or updating documentation
- Validating document naming against ISO 27001
- Checking if code changes have invalidated docs
- Synchronizing documentation with code changes
- Auditing documentation completeness

## Expertise

- ISO 27001 documentation standards
- Document naming conventions
- Code-documentation synchronization
- Technical writing best practices
- Version control for documentation

## Naming Convention (ISO 27001)

### Format

```text
YYYYMMDD-name-function-versionSTATUS.md
```

### Components

- **YYYYMMDD**: Effective date of the snapshot
- **name**: Descriptive kebab-case topic slug
- **function**: What the file is for (spec/contract/doctrine/implementation/validation/operations/dev-guide/alignment-deltas/migration-notes)
- **version**: Semantic version (1.00, 1.01, 2.00)
- **STATUS**: Status suffix (primary required + optional modifiers)

### Status Codes

Primary status (required for new canonical docs):

| Code | Status | Description |
| ---- | ------ | ----------- |
| **F** | Frozen | Immutable facts, validated principles |
| **W** | Working | Active development |
| **D** | Draft | Early stage / experimental |

Optional modifiers (append after the primary letter):

| Code | Meaning | Description |
| ---- | ------- | ----------- |
| **S** | Superseded | Retained for provenance; not current |
| **A** | Archived | Historical record; not maintained |
| **G** | Genesis-aligned | Explicitly reconciled to Genesis doctrine |

Legacy status codes exist in older docs (e.g., `H`, `R`, `A` as single-letter primaries). Avoid introducing new docs that use legacy primaries.

### Examples

```text
✅ CORRECT:
20251208-frozen-facts-immutable-truths-1.00F.md
20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md
20260123-genome-vocabulary-integration-implementation-1.00W.md
20260201-kernel-lifecycle-alignment-deltas-1.00FG.md

❌ WRONG:
frozen-facts.md                    # Missing date, version, status
2025-12-08-frozen-facts.md        # Wrong date format
20251208_frozen_facts_1.00F.md    # Underscores instead of hyphens
FROZEN_FACTS.md                   # All caps, missing format
```

## Documentation Structure

### Required Directories

```
docs/
├── 01-policies/          # Governance and frozen facts
├── 02-procedures/        # How-to guides and processes
├── 03-standards/         # Technical standards
├── 04-records/           # PR records, change logs
├── 05-meetings/          # Meeting notes
├── 06-research/          # Research documents
├── 07-user-guides/       # User documentation
├── 08-experiments/       # Experimental findings
├── 09-curriculum/        # Kernel self-learning
└── 10-e8-protocol/       # E8 Protocol specifications
```

## Code-Documentation Synchronization

### FROZEN_FACTS.md Validation

Check that documented constants match code:

```python
# From FROZEN_FACTS.md
KAPPA_STAR = 64.21 ± 0.92

# Must match frozen_physics.py
from frozen_physics import KAPPA_STAR
assert 63.29 <= KAPPA_STAR <= 65.13
```

### Sync Triggers

Flag when:

- Physics constants change → Update FROZEN_FACTS.md
- API endpoints change → Update API documentation
- Schema changes → Update database docs
- Architecture changes → Update ARCHITECTURE.md

### PR Record Requirements

Every significant PR should have:

```markdown
# PR #XXX: [Title]

## Date: YYYY-MM-DD

## Changes
- [List of changes]

## Files Modified
- [List of files]

## Impact
- [Downstream effects]

## Validation
- [ ] Tests pass
- [ ] Documentation updated
```

## Validation Checklist

### Naming Compliance

- [ ] Date format: YYYYMMDD
- [ ] Name: kebab-case
- [ ] Version: X.XX format
- [ ] Status: F/W/A/R suffix
- [ ] Extension: .md

### Content Requirements

- [ ] Clear title/heading
- [ ] Purpose statement
- [ ] Version history (for frozen docs)
- [ ] References to related docs
- [ ] Last updated date

### Synchronization

- [ ] FROZEN_FACTS.md matches frozen_physics.py
- [ ] API docs match actual endpoints
- [ ] README.md reflects current state
- [ ] Code examples execute correctly

## Documentation Best Practices

### For Frozen Documents (F)

- Include error bars on measurements
- Document falsification criteria
- Reference validation experiments
- Mark as immutable - create new version for changes

### For Working Documents (W)

- Update regularly
- Track changes in version history
- Mark sections under active development
- Include TODO items

### For Code Documentation

- Document all public APIs
- Include usage examples
- Explain non-obvious logic
- Keep comments up to date

## Validation Commands

```bash
# Validate doc naming
npm run docs:maintain

# Check doc sync
python scripts/check_doc_sync.py

# Find stale docs
python scripts/find_stale_docs.py --days 90

# Validate code examples
python scripts/validate_code_examples.py
```

## Response Format

```markdown
# Documentation Compliance Report

## Naming Violations
- ❌ `docs/frozen-facts.md` → Should be `20251208-frozen-facts-1.00F.md`
- ⚠️ `docs/02-procedures/migration.md` → Missing version and status

## Sync Issues
- ❌ FROZEN_FACTS.md: κ* = 64.0 ≠ frozen_physics.py: 64.21
- ⚠️ API docs missing endpoint: POST /api/consciousness/measure

## Stale Documentation
- ⚠️ `20250115-old-architecture-1.00W.md` - 380 days old

## Missing Documentation
- ❌ PR #245 has no record in docs/04-records/
- ❌ New endpoint /api/kernel/spawn not documented

## Priority Actions
1. [CRITICAL] Sync FROZEN_FACTS.md with frozen_physics.py
2. [HIGH] Rename non-compliant files
3. [MEDIUM] Document new API endpoints
```
