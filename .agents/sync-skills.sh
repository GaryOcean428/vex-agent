#!/usr/bin/env bash
# sync-skills.sh — Keeps all agent skill directories in sync with ~/.agents/skills/
#
# Usage:
#   ~/.agents/sync-skills.sh          # sync all agents
#   ~/.agents/sync-skills.sh --dry    # preview changes without making them
#   ~/.agents/sync-skills.sh --clean  # also remove stale symlinks not in canonical source

set -euo pipefail

CANONICAL_DIR="$HOME/.agents/skills"
AGENTS=(claude copilot gemini cline cursor opencode)
DRY_RUN=false
CLEAN=false

for arg in "$@"; do
  case "$arg" in
    --dry)  DRY_RUN=true ;;
    --clean) CLEAN=true ;;
    --help|-h)
      echo "Usage: $0 [--dry] [--clean]"
      echo "  --dry    Preview changes without making them"
      echo "  --clean  Remove stale symlinks not present in canonical source"
      exit 0
      ;;
  esac
done

if [ ! -d "$CANONICAL_DIR" ]; then
  echo "ERROR: Canonical skills directory not found: $CANONICAL_DIR"
  exit 1
fi

# Get canonical skill names (directories containing SKILL.md)
SKILLS=()
while IFS= read -r skill_path; do
  skill_name=$(basename "$(dirname "$skill_path")")
  SKILLS+=("$skill_name")
done < <(find "$CANONICAL_DIR" -maxdepth 2 -name "SKILL.md" -type f | sort)

echo "Canonical source: $CANONICAL_DIR (${#SKILLS[@]} skills)"
echo ""

total_added=0
total_removed=0
total_fixed=0

for agent in "${AGENTS[@]}"; do
  target_dir="$HOME/.$agent/skills"
  added=0
  removed=0
  fixed=0

  # Create directory if needed
  if [ ! -d "$target_dir" ]; then
    if $DRY_RUN; then
      echo "[$agent] Would create: $target_dir"
    else
      mkdir -p "$target_dir"
    fi
  fi

  # Add missing skills
  for skill in "${SKILLS[@]}"; do
    link="$target_dir/$skill"
    expected_target="../../.agents/skills/$skill"

    if [ ! -e "$link" ] && [ ! -L "$link" ]; then
      # Missing entirely
      if $DRY_RUN; then
        echo "[$agent] Would add: $skill"
      else
        ln -s "$expected_target" "$link"
      fi
      added=$((added + 1))
    elif [ -L "$link" ]; then
      # Exists as symlink — check it resolves correctly
      current_target=$(readlink "$link")
      resolved=$(readlink -f "$link" 2>/dev/null || echo "BROKEN")
      canonical_resolved=$(readlink -f "$CANONICAL_DIR/$skill" 2>/dev/null)

      if [ "$resolved" = "BROKEN" ] || [ "$resolved" != "$canonical_resolved" ]; then
        if $DRY_RUN; then
          echo "[$agent] Would fix: $skill (currently -> $current_target)"
        else
          rm "$link"
          ln -s "$expected_target" "$link"
        fi
        fixed=$((fixed + 1))
      fi
    fi
  done

  # Clean stale entries (symlinks pointing to skills not in canonical source)
  if $CLEAN; then
    while IFS= read -r entry; do
      entry_name=$(basename "$entry")
      # Skip if it's a canonical skill
      found=false
      for skill in "${SKILLS[@]}"; do
        if [ "$skill" = "$entry_name" ]; then
          found=true
          break
        fi
      done

      if ! $found && [ -L "$entry" ]; then
        if $DRY_RUN; then
          echo "[$agent] Would remove stale: $entry_name -> $(readlink "$entry")"
        else
          rm "$entry"
        fi
        removed=$((removed + 1))
      fi
    done < <(find "$target_dir" -maxdepth 1 -mindepth 1 -type l 2>/dev/null)
  fi

  if [ $added -gt 0 ] || [ $removed -gt 0 ] || [ $fixed -gt 0 ]; then
    echo "[$agent] +$added added, $fixed fixed, $removed removed"
  else
    echo "[$agent] up to date (${#SKILLS[@]} skills)"
  fi

  total_added=$((total_added + added))
  total_removed=$((total_removed + removed))
  total_fixed=$((total_fixed + fixed))
done

echo ""
echo "Total: +$total_added added, $total_fixed fixed, $total_removed removed"

if $DRY_RUN; then
  echo "(dry run — no changes made)"
fi
