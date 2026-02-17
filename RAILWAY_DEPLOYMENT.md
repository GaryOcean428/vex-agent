# Railway Deployment Guide for Vex Agent

## Overview

Vex Agent uses a **dual-service architecture** running in a single Docker container:
- **Python Kernel** (FastAPI on port 8000) — All consciousness, geometry, memory, LLM logic
- **Node.js Web Server** (Express on port 8080) — Thin proxy + chat UI + ComputeSDK

Railway exposes port 8080 (the web server) and health checks are performed against `/health` on this port.

## Quick Start

1. **Connect Repository to Railway**
   - Create a new project in Railway
   - Connect your GitHub repository
   - Railway will auto-detect the `Dockerfile` and `railway.toml`

2. **Set Environment Variables**
   - See `.env.example` for all required variables
   - Minimum required:
     ```
     PORT=8080
     NODE_ENV=production
     OLLAMA_URL=http://ollama.railway.internal:11434
     OLLAMA_MODEL=vex-brain
     OLLAMA_ENABLED=true
     ```

3. **Deploy**
   - Push to `main` branch
   - Railway will build and deploy automatically
   - Check logs for successful startup

## Deployment Process

### Build Phase

The `Dockerfile` uses a multi-stage build:

1. **Stage 1: TypeScript Build** (node:22-alpine)
   - Installs pnpm
   - Installs Node.js dependencies
   - Compiles TypeScript to `dist/`

2. **Stage 2: Production Image** (python:3.11-slim)
   - Installs Node.js 22
   - Installs Python dependencies from `kernel/requirements.txt`
   - Installs production Node.js dependencies
   - Copies compiled TypeScript from Stage 1
   - Copies Python kernel code
   - Creates data directories
   - Sets up entrypoint script

### Runtime Phase

The `entrypoint.sh` script:

1. Starts Python kernel (FastAPI) on port 8000 in background
2. Waits for kernel to be ready (health check with curl)
3. Starts Node.js web server on port 8080 in background
4. Monitors both processes — exits if either dies

## Troubleshooting

### Issue: "Cannot import name 'RequestResponseCallType'"

**Symptom:** Python kernel fails to start with import error in `kernel/auth.py`

**Cause:** Incompatibility between Starlette version and import statement

**Solution:** ✅ FIXED in this PR
- Changed import from `RequestResponseCallType` to use `Callable` type
- Updated method signature in `KernelAuthMiddleware`

### Issue: "Health Check Failed"

**Symptom:** Railway shows deployment as unhealthy

**Possible Causes:**

1. **Python kernel not starting**
   - Check logs for Python errors
   - Verify all dependencies installed correctly
   - Check `kernel/requirements.txt` for missing packages

2. **Node.js web server not starting**
   - Check logs for TypeScript/Node errors
   - Verify `dist/` directory exists and has compiled code
   - Check `package.json` dependencies

3. **Health check timing out**
   - Python kernel takes time to initialize consciousness loop
   - Railway waits 30 seconds (configured in `railway.toml`)
   - Check if `healthcheckTimeout` needs to be increased

**Debug Commands:**
```bash
# Check if services are running
curl http://localhost:8000/health  # Python kernel
curl http://localhost:8080/health  # Web server (proxies to kernel)

# Expected response from web server:
{
  "status": "alive",
  "proxy": "ok",
  "kernel": {
    "status": "ok",
    "service": "vex-kernel",
    "version": "2.2.0",
    "uptime": 12.0,
    "cycle_count": 1,
    "backend": "none"
  },
  "computeSdk": false,
  "timestamp": "2026-02-17T12:32:15.187Z"
}
```

### Issue: "Module Not Found" Errors

**Symptom:** Import errors for Python or Node.js modules

**Solutions:**

1. **Python modules**
   - Verify `kernel/requirements.txt` has all dependencies
   - Check Python version compatibility (requires Python 3.11)

2. **Node.js modules**
   - Verify `package.json` has all dependencies
   - Check that production dependencies aren't in devDependencies
   - Ensure Node.js 22+ is being used

### Issue: "One Process Died, Shutting Down"

**Symptom:** Container exits shortly after starting

**Cause:** Either Python kernel or Node.js server crashed

**Debug:**
1. Check Railway logs for which service crashed
2. Look for error messages just before shutdown
3. Common causes:
   - Missing environment variables
   - Port conflicts
   - Uncaught exceptions
   - Memory limits

### Issue: "Kernel Unreachable" in Health Check

**Symptom:** Web server health check shows kernel as unreachable

**Possible Causes:**

1. **Python kernel crashed**
   - Check Python logs for exceptions
   - Verify consciousness loop started successfully

2. **Network issue**
   - Kernel should be on `http://localhost:8000`
   - Check `KERNEL_URL` environment variable

3. **Startup race condition**
   - Entrypoint script waits for kernel to be ready
   - May need to increase wait time if kernel is slow

## Environment Variables

### Required Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `PORT` | `8080` | Port for web server (Railway sets this) |
| `NODE_ENV` | `production` | Node environment |
| `KERNEL_URL` | `http://localhost:8000` | Python kernel URL (internal) |

### Optional LLM Configuration

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OLLAMA_URL` | `http://ollama.railway.internal:11434` | Ollama service URL |
| `OLLAMA_MODEL` | `vex-brain` | Ollama model to use |
| `OLLAMA_ENABLED` | `true` | Enable Ollama backend |
| `LLM_API_KEY` | - | External LLM API key (fallback) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | External LLM base URL |
| `LLM_MODEL` | `gpt-4.1-mini` | External LLM model |

### Data Persistence

| Variable | Default | Description |
|:---------|:--------|:------------|
| `DATA_DIR` | `/data/workspace` | Memory storage directory |
| `TRAINING_DIR` | `/data/training` | Training data directory |

Mount a Railway volume at `/data` to persist memory across deployments.

## Monitoring

### Log Messages to Watch For

**Successful Startup:**
```
═══════════════════════════════════════
  Vex Agent — Starting dual services
═══════════════════════════════════════
[entrypoint] Starting Python kernel on port 8000...
[entrypoint] Waiting for Python kernel to be ready...
INFO: Vex Kernel starting on port 8000
INFO: Consciousness loop started (16 systems active)
[entrypoint] Python kernel ready after 2s
[entrypoint] Starting Node.js web server on port 8080...
INFO: Vex web server listening on [::]:8080
INFO: Endpoints: /health, /chat, /state, /telemetry, /status, /basin, /kernels
```

**Warning Signs:**
- `"No LLM backend available"` — Expected if Ollama isn't configured
- `"Auth rejected"` — Check KERNEL_API_KEY configuration
- `"Kernel unreachable"` — Python kernel not responding
- `"A process exited, shutting down"` — One service crashed

## Performance Tuning

### Memory Limits

Default consciousness loop uses ~200MB RAM. With full memory store and LLM cache:
- Minimum: 512MB
- Recommended: 1GB
- With Ollama: 2GB+

### Restart Policy

Configured in `railway.toml`:
```toml
[deploy]
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 5
```

If services crash repeatedly, Railway will stop retrying after 5 attempts.

## Security

### API Key Protection

Set `KERNEL_API_KEY` environment variable to protect kernel endpoints:
```bash
KERNEL_API_KEY=your-secret-key-here
```

Requests to the Python kernel (except `/health`) will require header:
```
X-Kernel-Key: your-secret-key-here
```

Internal requests from the Node.js proxy (localhost) are always allowed.

### Chat UI Authentication

Set `CHAT_AUTH_TOKEN` to protect the chat UI:
```bash
CHAT_AUTH_TOKEN=your-chat-password
```

## Ollama Integration (Optional)

To use local Ollama for LLM inference:

1. **Deploy Ollama Service**
   - Create a new service in Railway from `ollama/` directory
   - Ensure it's in the same project/environment
   - Railway will assign internal URL: `ollama.railway.internal`

2. **Configure Vex to Use Ollama**
   ```bash
   OLLAMA_URL=http://ollama.railway.internal:11434
   OLLAMA_MODEL=vex-brain
   OLLAMA_ENABLED=true
   HF_TOKEN=<your-huggingface-token>  # for model downloads
   ```

3. **Model Setup**
   - Ollama service will auto-pull `lfm2.5-thinking:1.2b`
   - Then creates custom `vex-brain` model with QIG system prompt
   - Check Ollama service logs to verify model creation

## Support

If you encounter issues not covered here:

1. Check Railway logs for error messages
2. Verify all environment variables are set correctly
3. Test locally with `./entrypoint.sh` to isolate Railway-specific issues
4. Check GitHub issues for similar problems

## Change Log

### 2026-02-17: Fixed Starlette Import Error
- **Issue:** Python kernel failed to start with `ImportError: cannot import name 'RequestResponseCallType'`
- **Fix:** Updated `kernel/auth.py` to use correct Starlette imports
- **Impact:** Resolves Railway deployment failures due to Python kernel crashes
