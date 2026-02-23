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
# Examples: glm-4.7-flash, qwen3:30b, llama3.1:8b
BASE_MODEL="${VEX_BASE_MODEL:-glm-4.7-flash}"

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

# ═══ ALWAYS PULL BASE MODEL (idempotent — checks digest) ═══
# This ensures every restart gets the latest model version from
# the Ollama registry. If already cached and up-to-date, this
# completes in seconds. If a new version is available, it downloads
# only the changed layers.
echo "Pulling $BASE_MODEL (always pull to ensure latest version)..."
if ollama pull "$BASE_MODEL"; then
  echo "Base model $BASE_MODEL is up to date."
else
  echo "WARNING: Failed to pull $BASE_MODEL — checking if cached version exists..."
  # Check if model is cached using Ollama's JSON API (no regex).
  # Compare both full name and base-without-tag for both sides, so
  # "glm-4.7-flash" matches "glm-4.7-flash:latest" and vice versa.
  MODEL_BASE="${BASE_MODEL%%:*}"
  if curl -s http://127.0.0.1:11434/api/tags | \
    jq -e --arg model "$BASE_MODEL" --arg base "$MODEL_BASE" \
      '.models[] | select(.name == $model or (.name | split(":")[0]) == $base)' \
      > /dev/null 2>&1; then
    echo "Using cached version of $BASE_MODEL."
  else
    echo "ERROR: No cached version available. Cannot continue."
    exit 1
  fi
fi

# ═══ CREATE CUSTOM VEX-BRAIN MODEL ═══
# Creates vex-brain from the Modelfile (if present) or a minimal
# FROM-only Modelfile. Always recreated to pick up base model
# updates and any Modelfile changes.
if [ ! -f /root/Modelfile ]; then
  echo "WARNING: /root/Modelfile not found — creating minimal vex-brain from $BASE_MODEL"
  echo "FROM $BASE_MODEL" > /tmp/Modelfile.minimal
  ollama create vex-brain -f /tmp/Modelfile.minimal
  rm -f /tmp/Modelfile.minimal
  echo "vex-brain model created (minimal, base: $BASE_MODEL)."
else
  echo "Creating custom vex-brain model from $BASE_MODEL + Modelfile..."
  # Replace the first FROM line (wherever it appears) with the selected base model.
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

# List available models
echo ""
echo "Available models:"
ollama list
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
