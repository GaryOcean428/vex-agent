#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Vex Agent — Dual-Service Entrypoint (Resilient)
#
#  Starts:
#    1. Python kernel (FastAPI on port 8000) — background
#    2. Node.js web server (Express on PORT) — background
#
#  Resilience:
#    - Retries each service up to MAX_RETRIES before giving up
#    - Delay before exit prevents Railway rapid-restart loops
#    - Graceful SIGTERM handling for clean shutdown
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

MAX_RETRIES=${MAX_RETRIES:-3}
RETRY_DELAY=${RETRY_DELAY:-5}
SHUTDOWN_DELAY=${SHUTDOWN_DELAY:-10}

KERNEL_PID=""
WEB_PID=""

# ─── Graceful shutdown ────────────────────────────────────────

cleanup() {
    echo "[entrypoint] Received shutdown signal — cleaning up..."
    [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null || true
    [ -n "$KERNEL_PID" ] && kill "$KERNEL_PID" 2>/dev/null || true
    # Give processes time to finish
    sleep 2
    [ -n "$WEB_PID" ] && kill -9 "$WEB_PID" 2>/dev/null || true
    [ -n "$KERNEL_PID" ] && kill -9 "$KERNEL_PID" 2>/dev/null || true
    echo "[entrypoint] Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

# ─── Start Python kernel with retry ──────────────────────────

start_kernel() {
    local attempt=1
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        echo "[entrypoint] Starting Python kernel (attempt $attempt/$MAX_RETRIES)..."
        python3 -m uvicorn kernel.server:app \
            --host 0.0.0.0 \
            --port 8000 \
            --log-level info \
            --no-access-log &
        KERNEL_PID=$!

        # Wait for kernel to be ready (up to 30s)
        local ready=false
        for i in $(seq 1 30); do
            if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
                echo "[entrypoint] Python kernel ready after ${i}s (attempt $attempt)"
                ready=true
                break
            fi
            if ! kill -0 "$KERNEL_PID" 2>/dev/null; then
                echo "[entrypoint] Python kernel process died on attempt $attempt"
                break
            fi
            sleep 1
        done

        if [ "$ready" = true ]; then
            return 0
        fi

        # Kill zombie process if still alive but not responding
        kill "$KERNEL_PID" 2>/dev/null || true
        KERNEL_PID=""
        attempt=$((attempt + 1))

        if [ "$attempt" -le "$MAX_RETRIES" ]; then
            echo "[entrypoint] Retrying kernel in ${RETRY_DELAY}s..."
            sleep "$RETRY_DELAY"
        fi
    done

    echo "[entrypoint] ERROR: Python kernel failed after $MAX_RETRIES attempts"
    return 1
}

# ─── Start Node.js web server with retry ─────────────────────

start_web() {
    local attempt=1
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        echo "[entrypoint] Starting Node.js web server (attempt $attempt/$MAX_RETRIES)..."
        node dist/index.js &
        WEB_PID=$!

        # Brief check that process didn't immediately crash
        sleep 2
        if kill -0 "$WEB_PID" 2>/dev/null; then
            echo "[entrypoint] Node.js web server started (PID $WEB_PID)"
            return 0
        fi

        echo "[entrypoint] Node.js web server died on attempt $attempt"
        WEB_PID=""
        attempt=$((attempt + 1))

        if [ "$attempt" -le "$MAX_RETRIES" ]; then
            echo "[entrypoint] Retrying web server in ${RETRY_DELAY}s..."
            sleep "$RETRY_DELAY"
        fi
    done

    echo "[entrypoint] ERROR: Node.js web server failed after $MAX_RETRIES attempts"
    return 1
}

# ─── Monitor loop ────────────────────────────────────────────

monitor() {
    while true; do
        # Check kernel
        if [ -n "$KERNEL_PID" ] && ! kill -0 "$KERNEL_PID" 2>/dev/null; then
            echo "[entrypoint] Python kernel (PID $KERNEL_PID) exited unexpectedly"
            break
        fi
        # Check web server
        if [ -n "$WEB_PID" ] && ! kill -0 "$WEB_PID" 2>/dev/null; then
            echo "[entrypoint] Node.js web server (PID $WEB_PID) exited unexpectedly"
            break
        fi
        sleep 5
    done
}

# ─── Main ────────────────────────────────────────────────────

echo "═══════════════════════════════════════"
echo "  Vex Agent — Starting dual services"
echo "  Max retries: $MAX_RETRIES"
echo "═══════════════════════════════════════"

if ! start_kernel; then
    echo "[entrypoint] FATAL: Kernel failed to start. Delaying exit by ${SHUTDOWN_DELAY}s..."
    sleep "$SHUTDOWN_DELAY"
    exit 1
fi

if ! start_web; then
    echo "[entrypoint] FATAL: Web server failed to start. Delaying exit by ${SHUTDOWN_DELAY}s..."
    kill "$KERNEL_PID" 2>/dev/null || true
    sleep "$SHUTDOWN_DELAY"
    exit 1
fi

echo "[entrypoint] Both services running — entering monitor loop"
monitor

# If we get here, a process died — clean up and delay before exit
echo "[entrypoint] Service failure detected — shutting down..."
kill "$KERNEL_PID" "$WEB_PID" 2>/dev/null || true

echo "[entrypoint] Delaying exit by ${SHUTDOWN_DELAY}s to prevent rapid-restart..."
sleep "$SHUTDOWN_DELAY"
exit 1
