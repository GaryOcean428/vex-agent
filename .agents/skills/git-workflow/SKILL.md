---
name: git-workflow
description: Enforce git best practices including conventional commits, branch naming, PR hygiene, and Gitflow. Use when creating commits, branches, PRs, or reviewing git history. Ensures consistent, traceable version control across the team.
---

# Git Workflow

Enforces git best practices for consistent, traceable version control.

## When to Use This Skill

- Creating commits (use conventional commit format)
- Creating branches (use standard naming)
- Opening or reviewing Pull Requests
- Auditing git history for compliance
- Setting up git hooks

## Conventional Commits

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(auth): add OAuth2 login` |
| `fix` | Bug fix | `fix(api): handle null response` |
| `docs` | Documentation only | `docs: update README` |
| `style` | Formatting, no code change | `style: fix indentation` |
| `refactor` | Code change, no feature/fix | `refactor(db): extract query builder` |
| `perf` | Performance improvement | `perf(search): add index` |
| `test` | Adding/fixing tests | `test(auth): add login tests` |
| `chore` | Maintenance tasks | `chore: update dependencies` |
| `ci` | CI/CD changes | `ci: add purity gate` |
| `build` | Build system changes | `build: upgrade webpack` |

### Breaking Changes

Add `!` after type or include `BREAKING CHANGE:` in footer:

```
feat(api)!: change response format

BREAKING CHANGE: Response now returns array instead of object
```

### Scope

Use component or module name:
- `feat(qig-backend): add Fisher-Rao distance`
- `fix(client): handle empty state`
- `docs(skills): add git-workflow`

## Branch Naming

### Format

```
<type>/<ticket>-<description>
```

### Examples

| Type | Example |
|------|---------|
| Feature | `feature/PROJ-123-oauth-login` |
| Bug fix | `fix/PROJ-456-null-pointer` |
| Hotfix | `hotfix/PROJ-789-security-patch` |
| Release | `release/v1.2.0` |
| Docs | `docs/update-readme` |

### Protected Branches

- `main` - Production-ready code
- `develop` - Integration branch
- `release/*` - Release candidates

## Pull Request Hygiene

### PR Title

Follow conventional commit format:
```
feat(auth): implement OAuth2 login flow
```

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Changes
- Added X
- Modified Y
- Removed Z

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Related Issues
Closes #123
```

### PR Checklist

- [ ] Title follows conventional commit format
- [ ] Description explains what and why
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] CI passes
- [ ] Reviewers assigned

## Gitflow Workflow

```
main ─────●─────────────●─────────────●───
          │             ↑             ↑
          │      release/v1.0    release/v1.1
          │             │             │
develop ──●──●──●──●────●──●──●──●────●───
             │     │       │     │
          feature/a  feature/b  feature/c
```

### Branch Lifecycle

1. Create feature branch from `develop`
2. Work on feature, commit frequently
3. Open PR to `develop`
4. After review, merge to `develop`
5. Create `release/*` branch for release prep
6. Merge release to `main` and tag
7. Merge release back to `develop`

## Git Hooks

### Pre-commit

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run linting
npm run lint || exit 1

# Run type check
npm run check || exit 1

# Run purity scan
python3 scripts/qig_purity_scan.py || exit 1
```

### Commit-msg

```bash
#!/bin/bash
# .git/hooks/commit-msg

# Validate conventional commit format
commit_regex='^(feat|fix|docs|style|refactor|perf|test|chore|ci|build)(\(.+\))?(!)?: .{1,72}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "ERROR: Commit message does not follow conventional commit format"
    echo "Format: <type>(<scope>): <description>"
    exit 1
fi
```

## Validation Commands

```bash
# Check recent commits follow convention
git log --oneline -10

# Validate branch name
git branch --show-current | grep -E '^(feature|fix|hotfix|release|docs)/'

# Check for merge conflicts
git diff --check

# Verify clean working tree
git status --porcelain
```

## Common Issues

### Commit Message Too Long

Keep subject line under 72 characters. Use body for details:

```
feat(api): add user authentication endpoint

Implements JWT-based authentication with refresh tokens.
Includes rate limiting and brute force protection.

Closes #123
```

### Merge Conflicts

1. Pull latest from target branch
2. Resolve conflicts locally
3. Test after resolution
4. Commit with clear message

### Accidental Commit to Main

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Create proper branch
git checkout -b feature/proper-branch

# Commit and push
git commit -m "feat: proper commit"
git push -u origin feature/proper-branch
```
