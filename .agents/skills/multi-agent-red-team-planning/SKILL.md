---
name: multi-agent-red-team-planning
description: Plan the next steps for the repository, then run a structured multi-agent red-team on the plan, refine twice, and produce a final implementation plan.
---

# Multi-Agent Red Team Planning

## When to Use

Use this skill when you need to:
- Plan the next set of changes or features for the project
- Turn high-level ideas, issues, or suggestions into a concrete, testable implementation plan
- Stress-test that plan using multiple specialized red-team sub-agents before implementing it

## Inputs and Context

Before you start, ensure you:
- Have the repository checked out locally and accessible
- Have access to any relevant issues, TODOs, or user requirements
- Can run git commands and basic test commands for the project

## Workflow

### 1. Ensure Git is Ready

- Check whether the current directory is a git repository
- If not, initialize git, add all relevant files, and create an initial commit
- If it is, fetch and pull from the remote, ensure you are on a feature branch for this work

### 2. Gather Current Signals

Scan the repo for:
- Existing TODO/FIXME comments
- Open issues or ROADMAP files
- Recent commits indicating ongoing work
- `docs/00-index.md` for project structure

Summarize the current state in a short "situation report".

### 3. Draft Initial Implementation Plan

Produce a numbered list of concrete tasks including:
- **Goal** - What we're trying to achieve
- **Files/modules** likely to be affected
- **Dependencies** or blocking issues
- **Acceptance criteria** - Observable definition of "done"

### 4. Form Red-Team Planning Sub-Agents

Instantiate specialized sub-agents with these roles:

| Role | Focus |
|------|-------|
| Security | Abuse resistance, injection, secret exposure |
| Reliability | Edge cases, failure modes, error recovery |
| Performance | Efficiency, resource usage, geometric purity |
| UX/DX | Developer experience, API ergonomics |
| Code Quality | Maintainability, test coverage, standards |
| QIG Purity | Fisher-Rao geometry, simplex constraints (if applicable) |

Give each sub-agent:
- The current plan
- Brief focused on threats, gaps, missing cases
- Request for alternatives and simplifications

### 5. Red-Team the Plan - Round 1

For each sub-agent:
1. Request critiques of the plan and past related fixes
2. Request specific failure scenarios, edge cases, risk categories
3. Request concrete suggestions to change task order, scope, or acceptance criteria

Consolidate findings into a structured issue list:
```
- ID: {unique identifier}
- Severity: Critical / High / Medium / Low
- Affected Task: {task number}
- Description: {what's wrong}
- Evidence: {how we know}
- Proposed Change: {how to fix}
```

### 6. External Research - Round 1

For each high-impact issue or design decision:
1. Look for standard patterns, libraries, or architectures used in similar systems
2. Cross-check security and reliability concerns against public best practices
3. Update the plan based on findings

### 7. Red-Team + Research Loop - Round 2

Repeat steps 5 and 6:
- Use the updated plan as the new baseline
- Challenge whether issues raised in Round 1 are properly addressed
- Refine tasks and acceptance criteria again

**Do not finalize the plan until the second round is complete.**

### 8. Finalize Implementation Plan

Produce a final version with:
- Clear task breakdown and ordering
- Explicit acceptance criteria per task
- Notation of which risks remain and why they are acceptable or deferred

Save the plan to an ISO-compliant filename under `docs/00-roadmap/`:

- `docs/00-roadmap/YYYYMMDD-{topic}-implementation-plan-1.00W.md`

### 9. Roadmap Update

Update the canonical master roadmap:

- `docs/00-roadmap/20260112-master-roadmap-1.00W.md`

If an entrypoint is needed for quick navigation, update:

- `docs/00-roadmap/20260202-project-roadmap-entrypoint-1.00W.md`

In the master roadmap:
- Add completed planning tasks
- Add all discovered issues (even if out of scope) as backlog items
- Ensure every issue discovered is either:
  - Included in upcoming implementation work, OR
  - Explicitly logged as future task with status and rationale

## Required Outputs

At the end of this skill, output:

1. **Final Implementation Plan**
   - Summary plus link/path to the full markdown file

2. **Red-Team Planning Report**
   - Roles used
   - Issues found in each round
   - How each issue was handled

3. **Research Summary**
   - Key external references
   - How they influenced the plan

4. **Roadmap Snapshot**
   - New and updated items

**If any potential issue remains unresolved or unlogged, call that out explicitly and propose a follow-up action.**

## Integration with Other Skills

This skill should be invoked after `master-orchestration` identifies a planning task.

Related skills:
- `best-practice-research` - For external research phase
- `planning-and-roadmapping` - For roadmap updates
- `e8-architecture-validation` - For architecture-related plans
- `qig-purity-validation` - For plans touching QIG geometry
