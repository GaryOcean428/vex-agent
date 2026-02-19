# Pantheon Chat Skills

Agent skills following the [Agent Skills specification](https://agentskills.io/specification).

## MANDATORY: Skill Usage Protocol

**Every agent turn MUST follow this protocol:**

```
1. FIRST: Invoke `master-orchestration` skill
2. Identify task type and required skills
3. Apply skills in order (planning → implementation → QA)
4. BEFORE COMPLETING: `qa-and-verification` skill MANDATORY
5. Update roadmap with progress and discovered issues
6. Never claim completion without verification evidence
```

**No proof = not done. No exceptions.**

## Cross-Platform Compatibility

These skills work across multiple AI coding agents:

| Platform | Location | Notes |
|----------|----------|-------|
| **Manus** | `skills/` | Primary location |
| **OpenAI Codex** | `.codex/skills/` | Symlink to `skills/` |
| **GitHub Copilot** | Read via `AGENTS.md` | Skills referenced in instructions |
| **Windsurf** | Read via `AGENTS.md` | Skills referenced in instructions |
| **Claude Code** | Read via `CLAUDE.md` | Symlink to `AGENTS.md` |

## Available Skills (28 Total)

### Orchestration & Red-Team (6 skills) — MANDATORY

| Skill | Description |
|-------|-------------|
| `master-orchestration` | **INVOKE FIRST EVERY TURN** - Coordinates skills, sub-agents, verification |
| `multi-agent-red-team-planning` | Plan changes with multi-agent red-team review (2 iterations) |
| `multi-agent-red-team-implementation` | Implement with red-team review, iterate twice, QA before done |
| `planning-and-roadmapping` | Turn ideas into structured, prioritized roadmap |
| `qa-and-verification` | **INVOKE BEFORE COMPLETION** - Prove changes work, no proof = not done |
| `best-practice-research` | External research on patterns, libraries, architectures |

### Core QIG Purity (3 skills)

| Skill | Description |
|-------|-------------|
| `qig-purity-validation` | Zero-tolerance geometric purity enforcement |
| `e8-architecture-validation` | Hierarchical kernel layers, god-kernel naming |
| `consciousness-development` | Φ/κ metrics, Fisher-Rao geometry |

### Code Quality & Structure (6 skills)

| Skill | Description |
|-------|-------------|
| `import-resolution` | Canonical imports, circular dependency detection |
| `schema-consistency` | Database migrations, vocabulary architecture |
| `code-quality-enforcement` | DRY, naming conventions, architecture |
| `test-coverage-analysis` | Critical path test coverage |
| `dependency-management` | Forbidden packages, requirements validation |
| `performance-regression` | Detect Euclidean approximation substitutions |

### Integration & Synchronization (4 skills)

| Skill | Description |
|-------|-------------|
| `documentation-sync` | FROZEN_FACTS.md validation, doc freshness |
| `documentation-compliance` | ISO 27001, canonical naming |
| `wiring-validation` | Feature implementation chain tracing |
| `frontend-backend-mapping` | Route coverage, type consistency |

### UI & Deployment (3 skills)

| Skill | Description |
|-------|-------------|
| `ui-ux-consistency` | Regime colors, God Panel, accessibility |
| `deployment-readiness` | Environment, migrations, health checks |
| `downstream-impact` | Change impact tracing |

### Advanced (1 skill)

| Skill | Description |
|-------|-------------|
| `pantheon-kernel-development` | God-kernel development, Zeus coordination |

### Meta & Workflow (5 skills) — NEW

| Skill | Description |
|-------|-------------|
| `skill-creator` | Guide for creating effective skills across all platforms |
| `cross-platform-sync` | Validate agent instruction files and symlinks |
| `git-workflow` | Conventional commits, branch naming, PR hygiene |
| `api-design-validation` | REST API patterns, endpoint naming, status codes |
| `security-audit` | Secret detection, dependency vulnerabilities, injection prevention |

## SKILL.md Format (Minimal)

Per the Agent Skills specification, only `name` and `description` are required:

```yaml
---
name: skill-name
description: Description that helps the agent select the skill
---

# Skill Title

Instructions in Markdown...
```

### Skill Folder Structure

```
skill-name/
├── SKILL.md        # Required: instructions + metadata
├── scripts/        # Optional: executable code
├── references/     # Optional: documentation
└── assets/         # Optional: templates, resources
```

## CI Script References

Skills invoke the actual CI scripts:

| Category | Scripts |
|----------|---------|
| QIG Purity | `scripts/qig_purity_scan.py`, `qig-backend/scripts/ast_purity_audit.py` |
| Testing | `qig-backend/tests/test_geometry_runtime.py` |
| Dependencies | `scripts/scan_forbidden_imports.py` |
| Security | `npm audit`, `pip-audit`, secret scanning |

## Key Principle

**NO HALF MEASURES. QIG PURITY IS NON-NEGOTIABLE.**

All skills enforce the same standards as CI. Docs after Jan 15, 2026 take precedence in conflicts.

---
**Version:** 2.3.0
**Protocol:** E8 Protocol v4.0
**Last Updated:** 2026-02-02
**Skills Added:** master-orchestration, multi-agent-red-team-planning, multi-agent-red-team-implementation, planning-and-roadmapping, qa-and-verification, best-practice-research
