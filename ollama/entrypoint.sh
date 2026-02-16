#!/bin/bash
set -e

echo "═══════════════════════════════════════"
echo "  Vex Brain — Ollama Service Starting"
echo "═══════════════════════════════════════"

# Ensure Ollama binds to all interfaces and the CLI can reach it
export OLLAMA_HOST="0.0.0.0:11434"

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

# Pull the base model if not already present
echo "Checking for base model: lfm2.5-thinking:1.2b"
if ! ollama list 2>/dev/null | grep -q "lfm2.5-thinking"; then
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

# List available models
echo ""
echo "Available models:"
ollama list
echo ""

echo "═══════════════════════════════════════"
echo "  Vex Brain — Ready"
echo "  Serving on port 11434"
echo "═══════════════════════════════════════"

# Keep the server running
wait $OLLAMA_PID
