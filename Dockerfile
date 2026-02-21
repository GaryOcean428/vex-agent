# ═══════════════════════════════════════════════════════════════
#  Vex Agent — Dual-Service Container
#
#  Architecture:
#    Python kernel (FastAPI, port 8000) — consciousness, geometry,
#      memory, LLM, tools, all 16 systems
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

# ── Create data directories ───────────────────────────────────
RUN mkdir -p /data/workspace /data/training /data/training/epochs /data/training/exports /data/training/uploads /data/training/curriculum

# ── Entrypoint script ─────────────────────────────────────────
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

ENV NODE_ENV=production
ENV PORT=8080
ENV KERNEL_URL=http://localhost:8000
ENV DATA_DIR=/data/workspace
ENV TRAINING_DIR=/data/training

EXPOSE 8080

# Health check against the web server (which probes the kernel)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./entrypoint.sh"]
