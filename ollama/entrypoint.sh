#!/bin/bash
set -e

echo "═══════════════════════════════════════"
echo "  Vex Brain — Ollama Service Starting"
echo "═══════════════════════════════════════"

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama server to start..."
MAX_RETRIES=30
RETRY=0
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  RETRY=$((RETRY + 1))
  if [ $RETRY -ge $MAX_RETRIES ]; then
    echo "ERROR: Ollama failed to start after ${MAX_RETRIES} attempts"
    exit 1
  fi
  echo "  Attempt ${RETRY}/${MAX_RETRIES}..."
  sleep 2
done
echo "Ollama server is ready."

# Pull the base model if not already present
echo "Checking for base model: lfm2.5-thinking:1.2b"
if ! ollama list | grep -q "lfm2.5-thinking:1.2b"; then
  echo "Pulling lfm2.5-thinking:1.2b (this may take a few minutes on first boot)..."
  ollama pull lfm2.5-thinking:1.2b
  echo "Base model pulled successfully."
else
  echo "Base model already present."
fi

# Create custom Vex model with QIG system prompt
echo "Creating custom vex-brain model from Modelfile..."
ollama create vex-brain -f /root/Modelfile
echo "vex-brain model created successfully."

echo "═══════════════════════════════════════"
echo "  Vex Brain — Ready"
echo "  Models: lfm2.5-thinking:1.2b, vex-brain"
echo "  Serving on port 11434"
echo "═══════════════════════════════════════"

# Keep the server running
wait $OLLAMA_PID
