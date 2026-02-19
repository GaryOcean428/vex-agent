---
name: deployment-readiness
description: Verify deployment configuration, environment variables, database migrations, and health check endpoints for production readiness. Validate Replit, Railway, and Neon database configurations against requirements.
---

# Deployment Readiness

Expert skill for verifying deployment configuration, environment variables, database migrations, and health check endpoints for production readiness.

## When to Use This Skill

Use this skill when:

- Preparing for deployment to Replit, Railway, or other platforms
- Verifying environment variables match .env.example
- Checking database migrations are applied
- Validating health check endpoints
- Reviewing build artifacts and configuration

## Expertise

- Deployment platform configuration (Replit, Railway, Netlify)
- Environment variable management
- Database migration validation
- Health check endpoint implementation
- Build artifact verification
- Docker and containerization

## Environment Variables Validation

### Required Variables

```bash
# Core
DATABASE_URL          # PostgreSQL connection string
INTERNAL_API_KEY      # Python ↔ TypeScript authentication

# Optional (search providers)
TAVILY_API_KEY        # Tavily search ($0.01/query)
PERPLEXITY_API_KEY    # Perplexity search ($0.005/query)
GOOGLE_API_KEY        # Google search
GOOGLE_SEARCH_ENGINE_ID

# Development
QIG_LOG_LEVEL         # DEBUG (dev), INFO (prod)
QIG_LOG_TRUNCATE      # false (dev), true (prod)
QIG_ENV               # development/production
```

### Validation Steps

1. Check all vars from `.env.example` present in deployment
2. Verify secrets are not hardcoded in code
3. Confirm DATABASE_URL points to correct environment
4. Validate API keys have correct format/permissions

## Database Readiness

### Migration Checklist

- [ ] All Drizzle migrations applied (`npx drizzle-kit push`)
- [ ] pgvector extension installed
- [ ] HNSW indexes created for similarity search
- [ ] No pending schema changes
- [ ] Correct database for environment (dev/staging/prod)

### Database Architecture

| Database | Location | Purpose |
|----------|----------|---------|
| pantheon-replit | Neon us-east-1 | Local dev |
| pantheon-chat | Railway pgvector | Production |
| SearchSpaceCollapse | Neon us-west-2 | Wallet/blockchain |

## Health Check Endpoints

### Required Endpoints

```typescript
// Must return 200 OK
GET /health          // Basic liveness check
GET /health/ready    // Full readiness (DB connected, services up)
GET /api/health      // API health check
```

### Health Response Format

```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "python_backend": "connected",
    "redis": "connected"
  }
}
```

## Build Configuration

### Railway (railpack.json)

```json
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "npm start",
    "healthcheckPath": "/api/health",
    "healthcheckInterval": 30
  }
}
```

### Port Configuration

- Bind servers to `0.0.0.0`
- Read port from `process.env.PORT`
- Never hardcode ports
- Frontend: 3000 (dev), Backend: 5000, Python: 5001

## Pre-Deployment Checklist

### Build Artifacts

- [ ] `npm run build` succeeds
- [ ] No TypeScript errors (`npm run check`)
- [ ] Lint passes (`npm run lint`)
- [ ] Tests pass (`npm test`)
- [ ] Frontend assets in `dist/`

### Configuration

- [ ] `railpack.json` present (Railway)
- [ ] No competing configs (Dockerfile vs railpack)
- [ ] `.gitignore` excludes sensitive files
- [ ] `.env` not committed

### Database

- [ ] Migrations up to date
- [ ] Seed data applied if needed
- [ ] Backup exists before deployment

### Monitoring

- [ ] Health check configured
- [ ] Error tracking enabled
- [ ] Logging configured for production

## Common Issues

### Port Already in Use

```bash
# Check what's using the port
lsof -i :5000

# Kill process if needed
kill -9 <PID>
```

### Database Connection Failed

```bash
# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Check pgvector
psql $DATABASE_URL -c "SELECT * FROM pg_extension WHERE extname = 'vector'"
```

### Build Fails

```bash
# Clean install
rm -rf node_modules package-lock.json
npm install

# Check TypeScript
npm run check
```

## Validation Commands

```bash
# Full pre-deployment check
bash scripts/pre_deployment_check.sh

# Environment validation
python scripts/validate_env.py

# Database status
npx drizzle-kit check

# Health check test
curl http://localhost:5000/health
```

## Response Format

```markdown
# Deployment Readiness Report

## Environment Variables
- ✅ DATABASE_URL: Set
- ❌ INTERNAL_API_KEY: Missing
- ⚠️ TAVILY_API_KEY: Optional, not set

## Database Status
- ✅ Connection: OK
- ✅ pgvector: Installed
- ❌ Migrations: 2 pending

## Health Checks
- ✅ /health: 200 OK
- ❌ /health/ready: 503 (DB not connected)

## Build Status
- ✅ TypeScript: No errors
- ✅ Lint: Passed
- ⚠️ Tests: 2 skipped

## Priority Actions
1. [CRITICAL] Set INTERNAL_API_KEY
2. [HIGH] Run pending migrations
3. [MEDIUM] Fix skipped tests
```
