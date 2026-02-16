FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package.json pnpm-lock.yaml* ./
RUN corepack enable && corepack prepare pnpm@latest --activate
RUN pnpm install --frozen-lockfile 2>/dev/null || pnpm install

# Build TypeScript
COPY tsconfig.json ./
COPY src/ ./src/
RUN pnpm run build

# ── Production image ──────────────────────────────────────────
FROM node:22-alpine AS runner

WORKDIR /app

# Install production deps only
COPY package.json pnpm-lock.yaml* ./
RUN corepack enable && corepack prepare pnpm@latest --activate
RUN pnpm install --prod --frozen-lockfile 2>/dev/null || pnpm install --prod

# Copy compiled JS
COPY --from=builder /app/dist ./dist

# Create data directory (will be overridden by Railway volume mount)
RUN mkdir -p /data/workspace

ENV NODE_ENV=production
ENV PORT=8080
ENV DATA_DIR=/data/workspace

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost:8080/health || exit 1

CMD ["node", "dist/index.js"]
