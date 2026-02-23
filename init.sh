#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Vex Agent — Init Script (runs as root)
#
#  Fixes Railway volume mount permissions, then drops to the
#  non-root vex user for the actual services.
#
#  Railway volumes mount as root-owned, but the Dockerfile creates
#  a vex user for security. This script bridges the gap.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Fix data directory permissions ──────────────────────────────
# Railway volume mounts override Dockerfile chown. Ensure the vex
# user can write to /data (memory, conversations, consciousness state).
if [ -d /data ]; then
    echo "[init] Fixing /data permissions for vex user..."
    # Set ownership on the /data mount point itself without recursing
    # and without following symlinks.
    chown -h --no-dereference vex:vex /data 2>/dev/null || true

    # Create known-safe subdirectories for vex to use.
    mkdir -p /data/workspace /data/training

    # Recursively fix ownership only on known-safe, non-symlink subdirs,
    # and do not follow any symlinks inside them.
    for vex_dir in /data/workspace /data/training; do
        if [ -d "$vex_dir" ] && [ ! -L "$vex_dir" ]; then
            chown -R -h --no-dereference vex:vex "$vex_dir" 2>/dev/null || true
        fi
    done
fi

# ── Drop to vex user and exec entrypoint ────────────────────────
exec su -s /bin/bash vex -c "./entrypoint.sh"
