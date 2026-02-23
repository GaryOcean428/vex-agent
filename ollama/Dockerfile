# Ollama service for Vex — Railway deployment
#
# Base model is configurable via VEX_BASE_MODEL env var.
# Default: glm-4.7-flash (30B-A3B MoE, 3B active, 19GB Q4_K_M)
#
# entrypoint.sh always pulls the model (idempotent digest check),
# so model updates propagate on next container restart.

# Pin Ollama to a known-good version for supply chain safety.
# To upgrade: check https://hub.docker.com/r/ollama/ollama/tags
# Update the tag, test locally, then push.
FROM ollama/ollama:0.16.3

# jq is used by entrypoint.sh for structured Ollama API queries (no-regex policy)
RUN apt-get update && apt-get install -y --no-install-recommends jq && rm -rf /var/lib/apt/lists/*

# Copy the Modelfile for custom model creation
COPY Modelfile /root/Modelfile

# Create entrypoint script that:
# 1. Starts Ollama server (pinned version)
# 2. Waits for it to be ready
# 3. Pulls VEX_BASE_MODEL (always, to get latest digest)
# 4. Creates custom vex-brain model (patched Modelfile)
# 5. Keeps serving
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

# Ollama default port
EXPOSE 11434

# Persistent model storage via Railway volume mount
# Mount a Railway volume at /root/.ollama in the service settings
# Set WORKING_DIR env var to the volume mount path

ENTRYPOINT ["/root/entrypoint.sh"]
