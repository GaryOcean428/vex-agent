"""
Auth Middleware — Simple API key authentication for kernel endpoints.

Protects kernel API endpoints from unauthenticated access.
The chat UI auth (CHAT_AUTH_TOKEN) is handled by the TS proxy layer.
This module protects the kernel's own FastAPI endpoints.

Usage:
    Set KERNEL_API_KEY env var. If empty, auth is disabled (dev mode).
    Requests must include header: X-Kernel-Key: <key>
    Internal requests from the TS proxy (localhost) are allowed without auth.
"""

from __future__ import annotations

import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable

logger = logging.getLogger("vex.auth")

KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")

# Paths that don't require auth (health checks, public info)
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
}


class KernelAuthMiddleware(BaseHTTPMiddleware):
    """Simple API key middleware for kernel endpoints.

    Behaviour:
    - If KERNEL_API_KEY is empty: auth disabled (open access, dev mode)
    - If set: requires X-Kernel-Key header matching the key
    - Requests from localhost/127.0.0.1 (TS proxy) are always allowed
    - /health is always public (Railway health checks)
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # No key configured = dev mode (open access)
        if not KERNEL_API_KEY:
            return await call_next(request)

        # Public paths always allowed
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Internal requests from TS proxy (same Railway service) always allowed
        client_host = request.client.host if request.client else ""
        if client_host in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)

        # Check API key header
        provided_key = request.headers.get("X-Kernel-Key", "")
        if provided_key != KERNEL_API_KEY:
            logger.warning(
                "Auth rejected: %s %s from %s",
                request.method, request.url.path, client_host,
            )
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication required. Set X-Kernel-Key header."},
            )

        return await call_next(request)
