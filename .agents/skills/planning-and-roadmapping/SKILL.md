---
name: planning-and-roadmapping
description: Turn ideas, issues, and prompts into a structured, prioritized roadmap. Maintain living documentation of project progress.
---

# Planning and Roadmapping

## When to Use

Use this skill when:
- New tasks, issues, or prompts arise
- You need to translate them into a clear, prioritized roadmap
- Updating progress after completing work
- Consolidating discovered issues from any phase

## Workflow

### 1. Collect All Inputs

Gather from:
- User prompts and requirements
- Existing issues, TODOs, and backlog items
- Outputs from planning and implementation red-team skills
- Discovered issues during any work phase
- `docs/00-index.md` for project context

### 2. Normalize Into Tasks

For each item, define:

```markdown
### Task: {Title}

**Description:** {What needs to be done}

**Rationale:** {Why this matters / user value}

**Dependencies:** {What must be done first}

**Acceptance Criteria:**
- [ ] {Observable outcome 1}
- [ ] {Observable outcome 2}

**Affected Areas:** {Files, modules, systems}

**Estimated Complexity:** Low / Medium / High

**Status:** Not Started / In Progress / Blocked / Complete
```

### 3. Prioritize

Use a clear scheme:

| Priority | Criteria |
|----------|----------|
| **Now** | Blocking other work, critical bugs, user-facing issues |
| **Next** | Important features, significant improvements |
| **Later** | Nice-to-have, optimizations, technical debt |
| **Backlog** | Ideas, low-priority, needs more definition |

Document reasoning for priorities.

### 4. Update the Canonical Roadmap

Update the master roadmap file:

- `docs/00-roadmap/20260112-master-roadmap-1.00W.md`

Optionally update the entrypoint:

- `docs/00-roadmap/20260202-project-roadmap-entrypoint-1.00W.md`

Structure for roadmap entries:

```markdown
# Project Roadmap

Last Updated: {YYYY-MM-DD}

## In Progress
- [ ] {Task} - {Owner/Branch} - {Status}

## Planned (Next)
- [ ] {Task} - {Priority} - {Dependencies}

## Backlog
- [ ] {Task} - {Origin: planning/implementation/QA}

## Completed
- [x] {Task} - {Date} - {Commit/PR}

## Parking Lot (Ideas/Future)
- {Idea} - {Source} - {Notes}
```

Link tasks to plan files and branches where relevant.

### 5. Cross-Reference

Ensure consistency with:
- `docs/00-roadmap/` for implementation plans and planning artifacts
- Issue tracker (if applicable)
- `docs/00-index.md` for documentation tasks

### 6. Publish Summary

Output a concise roadmap snapshot:

```markdown
## Roadmap Update Summary

### Current Focus
{1-2 sentence description of primary work}

### Completed Since Last Update
- {item 1}
- {item 2}

### Now (Active)
1. {highest priority item}
2. {next priority item}

### Next (Queued)
- {item with dependencies resolved}

### Newly Added
- {item}: {source - planning/implementation/QA/user request}

### Blocked
- {item}: {blocker description}
```

## Issue Tracking Integration

Every issue discovered anywhere must appear in the roadmap:

| Source | Action |
|--------|--------|
| Planning red-team | Add to Planned or Backlog |
| Implementation red-team | Add to Now (if critical) or Next |
| QA findings | Add to Now (if blocking) or Next |
| User feedback | Prioritize and add appropriately |
| Code review | Add with source reference |

## Roadmap File Location

Primary: `docs/00-roadmap/20260112-master-roadmap-1.00W.md`

Entry point (optional): `docs/00-roadmap/20260202-project-roadmap-entrypoint-1.00W.md`

## Required Outputs

1. **Updated master roadmap** file in `docs/00-roadmap/`
2. **Short text summary** of current priorities and next actions
3. **Confirmation** that all discovered issues are represented

## Integration with Other Skills

This skill is invoked by:
- `master-orchestration` - At turn completion
- `multi-agent-red-team-planning` - After planning phase
- `multi-agent-red-team-implementation` - After implementation phase
- `qa-and-verification` - After QA phase
