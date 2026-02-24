# ═══════════════════════════════════════════════════════════════
#  Vex Agent — Dual-Service Container
#
#  Architecture:
#    Python kernel (FastAPI, port 8000) — consciousness, geometry,
#      memory, LLM, tools, all 20 systems
#    Node.js web server (Express, port 8080) — chat UI, API proxy,
#      ComputeSDK integration
#
#  The entrypoint script starts both services.
#  Railway exposes port 8080 (the web server).
# ═══════════════════════════════════════════════════════════════

# ── Stage 1a: Build TypeScript (Express server) ─────────────────
FROM node:22-alpine AS ts-builder

WORKDIR /app

COPY package.json pnpm-lock.yaml* ./
RUN corepack enable && corepack prepare pnpm@latest --activate
RUN pnpm install --frozen-lockfile 2>/dev/null || pnpm install

COPY tsconfig.json ./
COPY src/ ./src/
RUN pnpm run build

# ── Stage 1b: Build React Frontend (Vite) ───────────────────────
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/pnpm-lock.yaml* ./
RUN corepack enable && corepack prepare pnpm@latest --activate
RUN pnpm install --frozen-lockfile 2>/dev/null || pnpm install

COPY frontend/ ./
RUN pnpm run build \
    && test -f dist/index.html \
    || (echo "FATAL: Frontend build failed — dist/index.html missing" && ls -la dist/ 2>/dev/null && exit 1)

# ── Stage 2: Production image ─────────────────────────────────
FROM python:3.14-slim

# Install Node.js 22
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    corepack enable && corepack prepare pnpm@latest --activate && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python kernel dependencies (uv) ──────────────────────────────
COPY pyproject.toml ./
COPY kernel/ ./kernel/
RUN pip install --no-cache-dir uv==0.9.26 && uv pip install --system --no-cache .

# ── Node.js production dependencies ───────────────────────────
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --prod --frozen-lockfile 2>/dev/null || pnpm install --prod

# ── Copy compiled TS ───────────────────────────────────────────
COPY --from=ts-builder /app/dist ./dist

# ── Copy built React frontend ───────────────────────────────────
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# ── Copy Ollama Modelfile ──────────────────────────────────────
COPY ollama/ ./ollama/

# ── Non-root user ────────────────────────────────────────────
RUN groupadd -r vex && useradd -r -g vex -d /app vex

# ── Create data directories ───────────────────────────────────
# Note: Railway volume mounts overwrite these at runtime.
# init.sh re-creates and re-chowns them on every start.
RUN mkdir -p \
    /data/workspace \
    /data/training \
    /data/training/curriculum \
    /data/training/uploads \
    /data/training/exports \
    /data/harvest \
    /data/harvest/pending \
    /data/harvest/processing \
    /data/harvest/completed \
    /data/harvest/failed \
    /data/harvest/output \
    && chown -R vex:vex /data /app

# ── Entrypoint scripts ────────────────────────────────────────
# init.sh runs as root to fix Railway volume mount permissions,
# then drops to the vex user for the actual services.
COPY init.sh ./init.sh
RUN chmod +x ./init.sh
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

ENV NODE_ENV=production
ENV PORT=8080
ENV KERNEL_URL=http://localhost:8000
ENV DATA_DIR=/data/workspace
ENV TRAINING_DIR=/data/training

EXPOSE 8080

# Health check against the web server (which probes the kernel)
# start-period=90s: kernel takes ~20s, web server starts after, plus margin
# interval=15s: check frequently once healthy so Railway sees liveness quickly
# timeout=10s: single probe timeout (web server has its own 5s kernel fetch timeout)
HEALTHCHECK --interval=15s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# init.sh runs as root, fixes /data permissions, then exec's entrypoint.sh as vex
ENTRYPOINT ["./init.sh"]
