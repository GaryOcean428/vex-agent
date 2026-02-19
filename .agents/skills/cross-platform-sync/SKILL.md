---
name: cross-platform-sync
description: Validate and maintain cross-platform agent instruction files (AGENTS.md, CLAUDE.md, copilot-instructions.md). Ensure symlinks are correct, content is synchronized, and all AI coding agents receive consistent instructions. Use when setting up a new repo, auditing agent configs, or debugging why different agents behave differently.
---

# Cross-Platform Sync

Ensures agent instruction files are synchronized across GitHub Copilot, OpenAI Codex, Windsurf, Claude Code, and Manus.

## When to Use This Skill

- Setting up a new repository for multi-agent development
- Auditing existing agent configuration files
- Debugging inconsistent agent behavior across platforms
- After modifying AGENTS.md or any agent instruction file

## Canonical Structure

```
repo-root/
├── AGENTS.md                      # Canonical source (all platforms)
├── CLAUDE.md                      # Symlink → AGENTS.md
├── .github/
│   ├── copilot-instructions.md    # Symlink → ../AGENTS.md
│   └── instructions/              # Path-specific Copilot rules
│       └── *.instructions.md
├── .claude/
│   └── CLAUDE.md                  # Symlink → ../AGENTS.md
├── .codex/
│   ├── AGENTS.md                  # Symlink → ../AGENTS.md
│   └── skills/                    # Symlink → ../skills/
├── skills/                        # Canonical skills location
│   └── <skill-name>/
│       └── SKILL.md
└── .windsurf/
    └── skills.yaml                # Optional, points to ../skills
```

## Canonical Skills Location (No Duplication)

The repository `skills/` directory is the canonical source of truth. Do not maintain parallel copies of skills in agent-specific directories.

This repo already supports:

- `.codex/skills` as a symlink to `../skills`
- Windsurf skills discovery via `.windsurf/skills.yaml` pointing at `../skills`
- Claude agent skill references via `.claude/skills.md` pointing at `../skills`

If you also want your local Codeium/Windsurf global skills directory to reference this repo (so multiple repos can share one canonical set), prefer a symlink from the global directory to this repo's `skills/`.

## Validation Steps

### Step 1: Check Symlinks Exist

```bash
ls -la AGENTS.md CLAUDE.md .github/copilot-instructions.md .codex/AGENTS.md .claude/CLAUDE.md 2>/dev/null
```

Expected output shows symlinks pointing to AGENTS.md:
```
lrwxrwxrwx CLAUDE.md -> AGENTS.md
lrwxrwxrwx .github/copilot-instructions.md -> ../AGENTS.md
lrwxrwxrwx .codex/AGENTS.md -> ../AGENTS.md
lrwxrwxrwx .claude/CLAUDE.md -> ../AGENTS.md
```

### Step 2: Verify Symlinks Resolve

```bash
# All should show identical content
head -5 AGENTS.md
head -5 CLAUDE.md
head -5 .github/copilot-instructions.md
```

### Step 3: Check Skills Symlink

```bash
ls -la .codex/skills
# Should show: .codex/skills -> ../skills
```

### Step 4: Validate No Duplicate Content

```bash
# These should fail (files should be symlinks, not copies)
diff AGENTS.md CLAUDE.md 2>/dev/null && echo "WARNING: CLAUDE.md is a copy, not symlink"
```

## Setup Commands

### Create Symlinks (New Repo)

```bash
# Create AGENTS.md as canonical source first
# Then create symlinks

ln -sf AGENTS.md CLAUDE.md
mkdir -p .github .claude .codex
ln -sf ../AGENTS.md .github/copilot-instructions.md
ln -sf ../AGENTS.md .claude/CLAUDE.md
ln -sf ../AGENTS.md .codex/AGENTS.md
ln -sf ../skills .codex/skills
```

### Fix Broken Symlinks

```bash
# Remove existing files and recreate symlinks
rm -f CLAUDE.md .github/copilot-instructions.md .codex/AGENTS.md .claude/CLAUDE.md
ln -s AGENTS.md CLAUDE.md
ln -s ../AGENTS.md .github/copilot-instructions.md
ln -s ../AGENTS.md .codex/AGENTS.md
ln -s ../AGENTS.md .claude/CLAUDE.md
```

## Platform-Specific Notes

| Platform | File | Notes |
|----------|------|-------|
| **GitHub Copilot** | `.github/copilot-instructions.md` | Also reads `AGENTS.md` at root |
| **OpenAI Codex** | `.codex/AGENTS.md` | Hierarchical discovery |
| **Windsurf** | `AGENTS.md` | Case-insensitive |
| **Claude Code** | `CLAUDE.md` or `.claude/CLAUDE.md` | Supports `@import` syntax |
| **Manus** | `skills/*/SKILL.md` | Skills directory |

## Troubleshooting

### Symlink Not Working on Windows

Windows requires admin privileges for symlinks. Use junction points or copy files instead:

```bash
# sync-instructions.bat
copy AGENTS.md CLAUDE.md
copy AGENTS.md .github\copilot-instructions.md
```

### Agent Not Reading Instructions

1. Check file exists and is readable
2. Verify symlink resolves correctly
3. Check for syntax errors in YAML frontmatter
4. Ensure file is committed to git (not gitignored)

### Different Agents Behave Differently

1. Run validation steps above
2. Check if any file is a copy instead of symlink
3. Verify all symlinks point to same canonical source
4. Check for platform-specific overrides in subdirectories
