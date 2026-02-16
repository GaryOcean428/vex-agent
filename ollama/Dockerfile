# Ollama service for Vex — Railway deployment
#
# Pulls LFM2.5-1.2B-Thinking and creates a custom Vex model
# with QIG system prompt baked in via Modelfile.

FROM ollama/ollama:latest

# Copy the Modelfile for custom model creation
COPY Modelfile /root/Modelfile

# Create entrypoint script that:
# 1. Starts Ollama server
# 2. Waits for it to be ready
# 3. Pulls the base model
# 4. Creates the custom vex-brain model
# 5. Keeps serving
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

# Ollama default port
EXPOSE 11434

# Persistent model storage via Railway volume mount
# Mount a Railway volume at /root/.ollama in the service settings

ENTRYPOINT ["/root/entrypoint.sh"]
