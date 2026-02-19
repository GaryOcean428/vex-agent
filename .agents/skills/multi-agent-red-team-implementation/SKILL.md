---
name: multi-agent-red-team-implementation
description: Implement planned changes, red-team the implementation with specialized sub-agents, iterate twice, then run QA and update the roadmap.
---

# Multi-Agent Red Team Implementation

## When to Use

Use this skill once you have a concrete, approved implementation plan and are ready to modify code, tests, and configuration.

## Preconditions

- An up-to-date plan file exists under `docs/00-roadmap/` (e.g., `docs/00-roadmap/YYYYMMDD-{topic}-implementation-plan-1.00W.md`)
- Git is initialized and synced with the remote
- Basic test commands for the project are known and runnable

## Workflow

### 1. Confirm Git and Branch

- Ensure git status is clean or changes are intentionally staged
- Create or switch to a feature branch for this set of changes
- Record the branch name and baseline commit

### 2. Implementation Cycle - Round 1

Select a coherent subset of tasks from the plan.

For each task:
1. Explain what you are about to change and why
2. Identify the files and components involved
3. Implement the changes
4. Run appropriate checks/tests:
   - Unit tests
   - Integration tests where relevant
   - Static checks or linters
   - QIG purity validation (if applicable)

Summarize:
- Code changes (high-level)
- Tests run and outcomes (pass/fail)

### 3. Implementation Red-Team - Round 1

Instantiate sub-agents with roles:

| Role | Focus |
|------|-------|
| Security | Injection, abuse, secrets |
| Reliability | Edge cases, failure recovery |
| Performance | Efficiency, resource usage |
| UX/DX | API ergonomics, clarity |
| Code Quality | Maintainability, standards |
| QIG Purity | Geometric correctness (if applicable) |

Provide them with:
- The plan segment that was implemented
- The high-level diff summary
- Test results

For each sub-agent, request:
- Targeted critiques of the implementation
- Specific scenarios where it might fail
- Suggestions for code changes, tests, monitoring

Consolidate into an **implementation issue list**:
```
- ID: {unique identifier}
- Severity: Critical / High / Medium / Low
- Area: {file/component}
- Reproduction: {steps to trigger}
- Proposed Fix: {how to address}
- Status: Open / Fixed / Deferred
```

### 4. Remediation from Red-Team - Round 1

- Implement fixes for all Critical and High severity issues
- Document any issues deferred with reasons
- Re-run affected tests
- Update the issue list with current status

### 5. Implementation + Red-Team Loop - Round 2

Repeat steps 2-4 for a second full round:
- Use updated code and tests as baseline
- Focus on previously identified weak points

**Do not move to QA until the second iteration is complete.**

### 6. QA and Verification

Instantiate QA-focused sub-agents responsible for:
- Designing test scenarios (unit, integration, regression, smoke)
- Checking that all acceptance criteria in the plan are satisfied

Run the test suite and any additional exploratory tests.

Produce a **QA Report**:
```markdown
## QA Report

### Tests Run
- Unit: {count} passed, {count} failed
- Integration: {count} passed, {count} failed
- E2E: {count} passed, {count} failed

### Acceptance Criteria Verification
| Task | Criterion | Status | Evidence |
|------|-----------|--------|----------|
| 1 | {criterion} | Pass/Fail | {test or output} |

### Issues Found
- {description}: {severity}, {status}

### Exploratory Testing Notes
- {observations}
```

### 7. Proof of Completion and Correctness

Assemble a final summary proving:
- Which tasks were implemented in this session
- Which previous session tasks were revisited and completed
- That the system currently works as claimed

Include:
- Commit hashes and branch names
- Summary of tests and red-team rounds
- Any manual checks performed

**If there are gaps (untested areas, deferred issues), state them explicitly and add to roadmap.**

### 8. Roadmap and Git Push

Update the master roadmap:

- `docs/00-roadmap/20260112-master-roadmap-1.00W.md`

Optionally update the entrypoint:

- `docs/00-roadmap/20260202-project-roadmap-entrypoint-1.00W.md`

In the master roadmap:
- Mark completed items as done
- Add all newly discovered issues

Commit changes with a clear message referencing the plan.

Push the branch to remote.

If appropriate, propose a PR with:
- Summary of changes
- Red-team findings addressed
- QA results
- Requirements for human review

## Required Outputs

At the end of this skill, output:

1. **Implementation Report**
   - Changes implemented
   - Red-team findings and fixes (both rounds)
   - QA results

2. **Proof-of-Work Summary**
   - Commits, branch, notable files
   - Evidence that acceptance criteria are satisfied

3. **Roadmap Confirmation**
   - Updated with progress
   - All unresolved issues explicitly tracked

## Integration with Other Skills

This skill should be invoked after `multi-agent-red-team-planning`.

Related skills:
- `qa-and-verification` - For the QA phase
- `qig-purity-validation` - For every code change
- `test-coverage-analysis` - For test design
- `git-workflow` - For commit and PR hygiene
