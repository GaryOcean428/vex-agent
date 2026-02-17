#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Vex Agent — Dual-Service Entrypoint
#
#  Starts:
#    1. Python kernel (FastAPI on port 8000) — background
#    2. Node.js web server (Express on PORT) — background
#
#  If either process dies, the container exits (Railway restarts it).
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════"
echo "  Vex Agent — Starting dual services"
echo "═══════════════════════════════════════"

# Start Python kernel in background
echo "[entrypoint] Starting Python kernel on port 8000..."
python3 -m uvicorn kernel.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  --no-access-log &
KERNEL_PID=$!

# Wait for kernel to be ready (up to 30s)
echo "[entrypoint] Waiting for Python kernel to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[entrypoint] Python kernel ready after ${i}s"
    break
  fi
  if ! kill -0 $KERNEL_PID 2>/dev/null; then
    echo "[entrypoint] ERROR: Python kernel process died"
    exit 1
  fi
  sleep 1
done

# Start Node.js web server in background
echo "[entrypoint] Starting Node.js web server on port ${PORT:-8080}..."
node dist/index.js &
WEB_PID=$!

# Wait for either process to exit
wait -n $KERNEL_PID $WEB_PID

# If we get here, one process died — kill the other and exit
echo "[entrypoint] A process exited, shutting down..."
kill $KERNEL_PID $WEB_PID 2>/dev/null || true
exit 1
