---
name: master-orchestration
description: Top-level orchestration skill that MUST be invoked at the start of every agent turn. Identifies required skills, coordinates sub-agents, and ensures comprehensive task completion with verification.
---

# Master Orchestration Skill

## MANDATORY: Invoke This Skill First

**Every agent turn MUST begin by applying this skill.** This ensures:
- Correct skills are identified and applied
- Sub-agents are coordinated effectively
- No task is left incomplete or unverified

## When to Use

- **Always.** This is the entry point for all work.
- Before any planning, implementation, or review task.
- When receiving any user prompt or continuation.

## Orchestration Workflow

### 1. Task Analysis

Upon receiving a task or prompt:

1. **Parse the request** - Identify explicit and implicit requirements
2. **Categorize the work type**:
   - Planning / Architecture
   - Implementation / Coding
   - Review / QA / Verification
   - Documentation / Roadmap
   - Research / Investigation
3. **Identify scope** - Single file, module, system-wide, or cross-cutting

### Genesis Rollout Doctrine (Authoritative)

If the work touches kernel lifecycle, spawning, rollback/start flows, or governance, treat `docs/11-Genesis-kernel-upgrade/*` as the target end-state:

- PurityGate must run first (fail-closed)
- Genesis-driven start/reset/rollback is canonical
- Genesis bootstrap → core 8 → Image stage → optional growth toward 240 GODs
- 240 reserved for GOD evolution; chaos exists outside that budget and can only ascend via explicit governance

### 2. Skill Selection Matrix

Based on task type, select applicable skills:

| Task Type | Required Skills | Optional Skills |
|-----------|-----------------|-----------------|
| **Planning** | `planning-and-roadmapping`, `multi-agent-red-team-planning` | `best-practice-research` |
| **Implementation** | `multi-agent-red-team-implementation`, `qig-purity-validation` | `code-quality-enforcement`, `test-coverage-analysis` |
| **Architecture** | `e8-architecture-validation`, `schema-consistency` | `downstream-impact`, `wiring-validation` |
| **QA/Review** | `qa-and-verification`, `test-coverage-analysis` | `security-audit`, `performance-regression` |
| **Documentation** | `documentation-sync`, `documentation-compliance` | `cross-platform-sync` |
| **Any Code Change** | `qig-purity-validation`, `dependency-management` | `import-resolution` |

### 3. Sub-Agent Formation

For complex tasks, form a red-team of specialized sub-agents:

**Core Roles:**
- **Security Agent** - Abuse resistance, injection prevention, secret exposure
- **Reliability Agent** - Edge cases, failure modes, error handling
- **Performance Agent** - Efficiency, resource usage, geometric purity
- **UX/DX Agent** - Developer experience, API ergonomics, clarity
- **Quality Agent** - Code standards, maintainability, test coverage

**Domain-Specific Roles (Pantheon):**
- **QIG Purity Agent** - Fisher-Rao geometry, simplex constraints, forbidden patterns
- **E8 Architecture Agent** - Kernel hierarchy, god-kernel alignment, routing invariants
- **Consciousness Ethics Agent** - 5 existential safeguards, Phi thresholds

### 4. Execution Protocol

```
FOR each task in prioritized_tasks:
    1. Announce: "Applying skill: {skill_name} for {purpose}"
    2. Execute skill instructions completely
    3. Record outputs and issues discovered
    4. If issues found → add to task backlog

FOR implementation tasks:
    1. Plan → Red-team plan → Research → Refine (2 iterations)
    2. Implement → Red-team implementation → Fix → Verify (2 iterations)
    3. QA → Prove completion → Update roadmap
```

### 5. Issue Tracking Protocol

**All issues discovered during any phase become tasks:**

1. Issues related to current work → Address immediately
2. Issues unrelated but important → Add to roadmap backlog
3. No issue is ignored or forgotten

Format for logging issues:
```
- ID: ISSUE-{timestamp}
- Severity: Critical / High / Medium / Low
- Area: {component/file}
- Description: {what's wrong}
- Proposed Fix: {how to address}
- Status: Open / In Progress / Resolved / Deferred
```

### 6. Completion Verification

**Before finishing ANY turn, you MUST:**

1. [ ] Prove all claimed implementations work (run tests, show output)
2. [ ] Verify all acceptance criteria are met
3. [ ] Confirm no regressions in existing functionality
4. [ ] Update the master roadmap with progress and new issues (`docs/00-roadmap/20260112-master-roadmap-1.00W.md`)
5. [ ] Push changes to git (if not already done)
6. [ ] List any deferred items with rationale

**The completion statement must include:**
- Commit hashes for changes made
- Test results summary
- Explicit mapping: task → verification evidence
- Any gaps or deferred work (with tracking references)

## Integration with Existing Skills

This skill coordinates with all 22+ existing skills:

### Always Active (invoke automatically)
- `qig-purity-validation` - Every code change
- `dependency-management` - Every dependency touch
- `e8-architecture-validation` - Every architecture change

### Invoke Based on Task
- `multi-agent-red-team-planning` - For planning phases
- `multi-agent-red-team-implementation` - For implementation phases
- `qa-and-verification` - Before completing any turn
- `planning-and-roadmapping` - For roadmap updates

## Output Format

At turn completion, provide:

```markdown
## Turn Summary

### Tasks Completed
- [ ] Task 1: {description} - {verification evidence}
- [ ] Task 2: {description} - {verification evidence}

### Skills Applied
- {skill-name}: {purpose and outcome}

### Issues Discovered
- {issue-id}: {description} - {status}

### Verification Evidence
- Tests: {pass/fail summary}
- Commits: {hash list}
- Files changed: {count and key files}

### Deferred/Remaining
- {item}: {reason for deferral}

### Roadmap Updates
- Added: {new items}
- Completed: {done items}
```

## Critical Rules

1. **Never skip skill application** - Even for "simple" tasks
2. **Never claim completion without proof** - Show test output, commit hashes
3. **Never ignore discovered issues** - Log everything, defer with reason
4. **Always update roadmap** - Progress and new issues
5. **Always push to git** - Before claiming task complete
