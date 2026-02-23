# PR #46 Fixes — Modal Deploy, LLM Client, Consciousness Bugs

**Date**: 2026-02-23
**Branch**: `claude/fix-pr42-issues-zRvjt`
**Status**: Deployed to Modal

## Summary

Fixed multiple issues surfaced during PR #46 review and Modal deployment,
including Copilot review comments, LLM client bugs, consciousness loop
deadlocks, and Modal 1.x compatibility.

## Fixes Applied

### 1. Modal 1.x Web Endpoint Compatibility

**Problem**: `modal deploy` failed because Modal 1.x requires
`fastapi[standard]` in the image for any `@modal.fastapi_endpoint()`.

**Fix**: Added `"fastapi[standard]"` to `ml_image.pip_install()` in
`modal/vex_coordizer_harvest.py`.

**Commit**: `a7539d2`

### 2. Modal Auth Headers Rejection

**Problem**: Sending `Modal-Token-Id` / `Modal-Token-Secret` as request
headers caused 403 errors. These are reserved internal headers that
Modal's proxy rejects on web endpoint requests.

**Fix**: Removed auth headers from `_auth_headers()` in
`kernel/coordizer_v2/modal_integration.py`. The endpoint is protected
by Modal's `requires_proxy_auth=True` network-level auth instead.

**Commit**: `408ac06`

### 3. Ollama Content Extraction — str(None) Bug

**Problem**: When Ollama returns a message with `content: null`, the
Python code did `str(None)` producing the literal string `"None"` in
responses.

**Fix**: Guard with `if content:` before appending in
`_extract_ollama_content()`.

**Commit**: `c2657e4`

### 4. PRE-COG Double-Lock Deadlock

**Problem**: The precog stage acquired `_precog_lock` twice (once in the
caller, once in the method), causing a deadlock. PRE-COG was permanently
stuck at 0%.

**Fix**: Removed the outer lock acquisition, keeping only the single
acquire inside the method.

**Commit**: `6f4bb08`

### 5. KernelBus Docstring Mismatch

**Problem**: Docstring referenced `drain_signals()` but the method was
renamed to `drain()`.

**Fix**: Updated docstring.

**Commit**: `9f56dff`

### 6. Copilot Review Comments (PR #46)

**Problem**: 6 review comments from GitHub Copilot on code quality,
typing, and documentation.

**Fix**: Addressed all comments.

**Commit**: `3559c04`

### 7. GLM Thinking Model Empty Responses

**Problem**: GLM-4.7-Flash returns empty `content` when thinking mode
is active, with reasoning in a separate field.

**Fix**: Disabled thinking mode in Ollama requests to get direct
responses.

**Commit**: `9b10d08`

### 8. Modal Health URL Pattern

**Problem**: Health check was hitting the harvest URL with GET, which
is a POST-only endpoint.

**Fix**: `modal_integration.py` now derives a separate health URL:
`<app>-health.modal.run` from the harvest URL pattern.

**Commit**: `bb5c9a4`

## Modal Deployment Notes

### Deploying from CI
The GitHub Actions workflow (`.github/workflows/modal-deploy.yml`) runs
`modal deploy` for both `vex_inference.py` and `vex_coordizer_harvest.py`
on push to `main` when `modal/**` files change.

### Deploying manually
```bash
modal token set --token-id <id> --token-secret <secret>
modal deploy modal/vex_coordizer_harvest.py
```

### Sandboxed environments (Claude Code web)
Modal CLI needs outbound HTTPS to `api.modal.com`. If DNS or TLS fails:
- Check proxy/tunnel configuration
- Set `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` if behind MITM proxy
- Modal SDK uses certifi for CA roots

## Testing

- Modal health endpoint returns `{"status": "ok"}` after deploy
- LLM client correctly falls back when Ollama is unavailable
- Consciousness loop no longer deadlocks on PRE-COG stage
- No `"None"` strings in chat responses
