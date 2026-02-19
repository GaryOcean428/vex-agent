---
name: qa-and-verification
description: Design and run test suites, prove changes work, verify no regressions. MANDATORY before claiming any task is complete.
---

# QA and Verification

## When to Use

**MANDATORY** - Use this skill:
- Before claiming ANY task is complete
- After implementation red-team rounds
- Before merging or creating PRs
- When verifying past implementations still work

## Core Principle

**Never claim completion without proof.** Every assertion of "done" must be backed by evidence.

## Workflow

### 1. Identify What Needs Verification

Collect:
- Tasks claimed as complete in this session
- Acceptance criteria from the plan
- Past implementations that may be affected
- Areas touched by code changes

### 2. Design Test Strategy

For each item, determine appropriate verification:

| Change Type | Verification Method |
|-------------|---------------------|
| New feature | Unit + integration tests |
| Bug fix | Regression test proving fix |
| Refactor | Existing tests still pass |
| API change | Contract tests + consumer tests |
| UI change | Visual/E2E tests |
| QIG geometry | Purity validation + geometric tests |

### 3. Run Test Suites

Execute in order:

```bash
# 1. Unit tests (fast feedback)
npm test
# or
pytest qig-backend/

# 2. Type checking
npm run check
# or
mypy qig-backend/

# 3. Linting
npm run lint

# 4. Integration tests
npm run test:integration
# or
pytest qig-backend/tests/integration/

# 5. QIG-specific validation (if applicable)
npm run validate:geometry
python3 scripts/qig_purity_scan.py

# 6. E2E tests (if available)
npm run test:e2e
```

### 4. Verify Acceptance Criteria

For each acceptance criterion in the plan:

```markdown
| Criterion | How Verified | Result | Evidence |
|-----------|--------------|--------|----------|
| {criterion 1} | {test/manual/output} | Pass/Fail | {link/output} |
| {criterion 2} | {test/manual/output} | Pass/Fail | {link/output} |
```

### 5. Check for Regressions

- Run full test suite, not just new tests
- Compare test counts before/after (no tests should disappear)
- Check critical paths manually if automated coverage is incomplete

### 6. Document Evidence

Produce a verification report:

```markdown
## Verification Report

**Date:** {YYYY-MM-DD}
**Session/Branch:** {identifier}

### Test Results Summary

| Suite | Total | Passed | Failed | Skipped |
|-------|-------|--------|--------|---------|
| Unit | {n} | {n} | {n} | {n} |
| Integration | {n} | {n} | {n} | {n} |
| E2E | {n} | {n} | {n} | {n} |

### Acceptance Criteria Status

| Task | Criterion | Status | Evidence |
|------|-----------|--------|----------|
| {task} | {criterion} | {status} | {evidence} |

### Regression Check

- [ ] All pre-existing tests pass
- [ ] No tests removed without justification
- [ ] Critical paths verified

### Manual Verification (if any)

| Check | Result | Notes |
|-------|--------|-------|
| {manual check} | Pass/Fail | {notes} |

### Issues Found

| Issue | Severity | Status |
|-------|----------|--------|
| {issue} | {severity} | {open/fixed/deferred} |

### Conclusion

**Verification Status:** PASSED / FAILED / PARTIAL

{Summary statement about readiness}
```

### 7. Handle Failures

If verification fails:

1. **Do not claim completion**
2. Identify root cause
3. Either:
   - Fix immediately and re-verify, OR
   - Document as known issue and add to roadmap
4. Be explicit about what is NOT verified

## Proof Requirements

Before claiming any work is done, you must show:

1. **Test Output** - Actual command output showing tests pass
2. **Commit Hashes** - Git commits for the changes
3. **Criterion Mapping** - Each acceptance criterion linked to verification
4. **No Regressions** - Evidence that existing functionality works

## Integration with Other Skills

This skill is invoked by:
- `master-orchestration` - Before completing any turn
- `multi-agent-red-team-implementation` - In QA phase

Related skills:
- `test-coverage-analysis` - For coverage gaps
- `qig-purity-validation` - For geometric verification
- `performance-regression` - For performance checks

## Critical Rules

1. **No proof = not done** - Period.
2. **Partial verification = partial completion** - Be honest
3. **Failing tests block completion** - Fix or defer explicitly
4. **Evidence must be reproducible** - Commands, not just claims
