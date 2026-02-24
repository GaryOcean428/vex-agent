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
#
# Marker-file optimisation: skip expensive recursive chown on subsequent
# starts. Only perform the full chown on initial volume setup, then touch
# a marker so future starts can skip straight to the mkdir calls.
if [ -d /data ]; then
    CHOWN_MARKER="/data/.vex_chown_done"
    if [ ! -f "$CHOWN_MARKER" ]; then
        echo "[init] Initial volume setup — fixing /data permissions for vex user..."
        # Set ownership on the /data mount point itself without following symlinks.
        chown -h --no-dereference vex:vex /data 2>/dev/null || true

        # Create known-safe subdirectories for vex to use.
        mkdir -p /data/workspace \
                 /data/conversations \
                 /data/training \
                 /data/training/curriculum \
                 /data/training/uploads \
                 /data/training/exports \
                 /data/harvest \
                 /data/harvest/pending \
                 /data/harvest/processing \
                 /data/harvest/completed \
                 /data/harvest/failed \
                 /data/harvest/output

        # Recursively fix ownership only on known-safe, non-symlink subdirs,
        # and do not follow any symlinks inside them.
        for vex_dir in /data/workspace /data/conversations /data/training /data/harvest; do
            if [ -d "$vex_dir" ] && [ ! -L "$vex_dir" ]; then
                chown -R -h --no-dereference vex:vex "$vex_dir" 2>/dev/null || true
            fi
        done

        # Also fix files directly in /data (e.g. consciousness_state.json if DATA_DIR=/data)
        find /data -maxdepth 1 -type f -exec chown vex:vex {} + 2>/dev/null || true

        touch "$CHOWN_MARKER"
    else
        echo "[init] /data permissions already set (marker found) — skipping recursive chown"
        # Always re-chown /data itself: Railway re-mounts volumes as root on restart
        chown -h --no-dereference vex:vex /data 2>/dev/null || true

        # Ensure all required dirs exist (may be missing on older volumes)
        mkdir -p /data/workspace \
                 /data/conversations \
                 /data/training \
                 /data/training/curriculum \
                 /data/training/uploads \
                 /data/training/exports \
                 /data/harvest \
                 /data/harvest/pending \
                 /data/harvest/processing \
                 /data/harvest/completed \
                 /data/harvest/failed \
                 /data/harvest/output

        # Re-chown top-level dirs (Railway remounts as root on restart)
        chown -h --no-dereference vex:vex \
            /data/workspace \
            /data/conversations \
            /data/training \
            /data/harvest 2>/dev/null || true

        # Also fix files directly in /data
        find /data -maxdepth 1 -type f -exec chown vex:vex {} + 2>/dev/null || true
    fi
fi

# ── Drop to vex user and exec entrypoint ────────────────────────
exec su -s /bin/bash vex -c "./entrypoint.sh"
