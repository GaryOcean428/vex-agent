#!/bin/bash
set -e

echo "═══════════════════════════════════════"
echo "  Vex Brain — Ollama Service Starting"
echo "═══════════════════════════════════════"

# Ensure Ollama binds to all interfaces and the CLI can reach it
export OLLAMA_HOST="0.0.0.0:11434"

# ═══ CONFIGURABLE MODEL ═══
# VEX_BASE_MODEL: The Ollama model to pull and use as the base for vex-brain.
# Change this env var to switch models without touching code.
# NOTE: This is the FALLBACK model (Tier 3). Primary inference uses
# PEFT adapters on Qwen3.5-4B via Modal (Tier 2).
# Use a small, fast model here — NOT 19GB glm-4.7-flash.
# Examples: qwen3:4b, phi4-mini, llama3.2:3b
BASE_MODEL="${VEX_BASE_MODEL:-qwen3:4b}"

echo "Base model: $BASE_MODEL"

# ═══ MODEL PERSISTENCE ═══
# Use WORKING_DIR (Railway volume mount) for model storage so models
# survive container restarts without re-downloading ~19GB each time.
if [ -n "$WORKING_DIR" ]; then
    export OLLAMA_MODELS="$WORKING_DIR/models"
    mkdir -p "$OLLAMA_MODELS"
    echo "Model storage: $OLLAMA_MODELS (persistent volume)"
else
    echo "WARNING: WORKING_DIR not set — models will NOT persist across restarts"
    echo "Set WORKING_DIR=/root/.ollama and attach a Railway volume at that path"
fi

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready using /dev/tcp (always available in bash)
echo "Waiting for Ollama server to start..."
MAX_RETRIES=30
RETRY=0
while true; do
  if (echo > /dev/tcp/127.0.0.1/11434) 2>/dev/null; then
    # Port is open — now verify API is responding
    sleep 2
    echo "Port 11434 is open, verifying API..."
    if ollama list > /dev/null 2>&1; then
      echo "Ollama server is ready."
      break
    fi
  fi
  RETRY=$((RETRY + 1))
  if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
    echo "ERROR: Ollama failed to start after ${MAX_RETRIES} attempts"
    echo "Dumping process list:"
    ps aux
    exit 1
  fi
  echo "  Attempt ${RETRY}/${MAX_RETRIES}..."
  sleep 2
done

# ═══ PULL BASE MODEL (skip if already cached) ═══
# Use 'ollama show' to check if model exists locally.
# This is more reliable than parsing /api/tags with jq.
if ollama show "$BASE_MODEL" > /dev/null 2>&1; then
  echo "Base model $BASE_MODEL already cached — skipping pull."
else
  echo "Pulling $BASE_MODEL (not cached)..."
  if ollama pull "$BASE_MODEL"; then
    echo "Base model $BASE_MODEL downloaded."
  else
    echo "ERROR: Failed to pull $BASE_MODEL and no cached version. Cannot continue."
    exit 1
  fi
fi

# ═══ CREATE CUSTOM VEX-BRAIN MODEL ═══
# Skip recreation if vex-brain already exists and the base model + Modelfile
# haven't changed. Uses a marker file to track what was used last time.
VEX_BRAIN_MARKER="${OLLAMA_MODELS:-.}/.vex-brain-base"
MODELFILE_HASH="none"
if [ -f /root/Modelfile ]; then
  MODELFILE_HASH=$(sha256sum /root/Modelfile | cut -d' ' -f1)
fi
CURRENT_FINGERPRINT="${BASE_MODEL}:${MODELFILE_HASH}"

NEED_CREATE=true
if ollama show vex-brain > /dev/null 2>&1; then
  if [ -f "$VEX_BRAIN_MARKER" ] && [ "$(cat "$VEX_BRAIN_MARKER" 2>/dev/null)" = "$CURRENT_FINGERPRINT" ]; then
    echo "vex-brain already exists (base: $BASE_MODEL, Modelfile unchanged) — skipping create."
    NEED_CREATE=false
  fi
fi

if [ "$NEED_CREATE" = true ]; then
  if [ ! -f /root/Modelfile ]; then
    echo "WARNING: /root/Modelfile not found — creating minimal vex-brain from $BASE_MODEL"
    echo "FROM $BASE_MODEL" > /tmp/Modelfile.minimal
    ollama create vex-brain -f /tmp/Modelfile.minimal
    rm -f /tmp/Modelfile.minimal
    echo "vex-brain model created (minimal, base: $BASE_MODEL)."
  else
    echo "Creating custom vex-brain model from $BASE_MODEL + Modelfile..."
    awk -v base="$BASE_MODEL" '
      !done && $1 == "FROM" {
        print "FROM " base
        done = 1
        next
      }
      { print }
    ' /root/Modelfile > /tmp/Modelfile.patched
    ollama create vex-brain -f /tmp/Modelfile.patched
    echo "vex-brain model created successfully (base: $BASE_MODEL)."
    rm -f /tmp/Modelfile.patched
  fi
  # Write marker so next restart skips creation
  echo "$CURRENT_FINGERPRINT" > "$VEX_BRAIN_MARKER" 2>/dev/null || true
fi

# List available models
echo ""
echo "Available models:"
ollama list

# ═══ CLEANUP STALE MODELS ═══
# Remove any models that aren't the base or vex-brain to free disk space.
# This catches leftover models from previous configurations (e.g., glm-4.7-flash, lfm2.5).
for model in $(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}'); do
  model_base="${model%%:*}"
  base_base="${BASE_MODEL%%:*}"
  if [ "$model_base" != "$base_base" ] && [ "$model_base" != "vex-brain" ]; then
    echo "Removing stale model: $model"
    ollama rm "$model" 2>/dev/null || true
  fi
done
echo ""

echo "═══════════════════════════════════════"
echo "  Vex Brain — Ready"
echo "  Serving on port 11434"
echo "  Base model: $BASE_MODEL"
echo "  Custom model: vex-brain"
if [ -n "$WORKING_DIR" ]; then
  echo "  Models persisted at: $OLLAMA_MODELS"
fi
echo "═══════════════════════════════════════"

# Keep the server running
wait $OLLAMA_PID
