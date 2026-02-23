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
    chown -R vex:vex /data 2>/dev/null || true
    mkdir -p /data/workspace /data/training
    chown -R vex:vex /data 2>/dev/null || true
fi

# ── Drop to vex user and exec entrypoint ────────────────────────
exec su -s /bin/bash vex -c "./entrypoint.sh"
