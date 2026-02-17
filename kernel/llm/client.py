"""
LLM Client — Ollama primary, external API fallback with cost guard.

Handles:
  - Ollama (local LFM2.5-1.2B-Thinking) as primary backend
  - OpenAI-compatible API as fallback (rate-limited by CostGuard)
  - Streaming support via async generators
  - Health checking and auto-failover

CostGuard prevents runaway costs when Ollama is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional

import httpx

from ..config.settings import settings
from .cost_guard import CostGuard, CostGuardConfig

logger = logging.getLogger("vex.llm")


class LLMClient:
    """Multi-backend LLM client with Ollama primary and API fallback."""

    def __init__(self) -> None:
        self._ollama_available = False
        self._active_backend = "none"
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(
            connect=10.0,
            read=settings.ollama.timeout_ms / 1000.0,
            write=30.0,
            pool=10.0,
        ))

        # Cost guard for external API calls
        self._cost_guard = CostGuard(CostGuardConfig(
            rpm_limit=20,     # 20 requests/minute
            rph_limit=200,    # 200 requests/hour
            rpd_limit=2000,   # 2000 requests/day
            max_tokens_per_request=2048,
        ))

    async def init(self) -> None:
        """Initialise the client — check Ollama availability."""
        self._ollama_available = await self.check_ollama()
        if self._ollama_available:
            self._active_backend = "ollama"
            logger.info("LLM backend: Ollama (%s)", settings.ollama.model)
        elif settings.llm.api_key:
            self._active_backend = "external"
            logger.info("LLM backend: External API (%s)", settings.llm.model)
        else:
            self._active_backend = "none"
            logger.warning("No LLM backend available")

    async def check_ollama(self) -> bool:
        """Check if Ollama is reachable."""
        if not settings.ollama.enabled:
            return False
        try:
            resp = await self._http.get(f"{settings.ollama.url}/api/tags")
            self._ollama_available = resp.status_code == 200
            return self._ollama_available
        except Exception:
            self._ollama_available = False
            return False

    async def complete(self, system_prompt: str, user_message: str) -> str:
        """Non-streaming completion. Returns full response text."""
        if self._active_backend == "ollama":
            return await self._ollama_complete(system_prompt, user_message)
        elif self._active_backend == "external":
            return await self._guarded_external_complete(system_prompt, user_message)
        return "No LLM backend available"

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion. Yields chunks of text."""
        if self._active_backend == "ollama":
            async for chunk in self._ollama_stream(messages):
                yield chunk
        elif self._active_backend == "external":
            async for chunk in self._guarded_external_stream(messages, temperature, max_tokens):
                yield chunk
        else:
            yield "No LLM backend available"

    def get_status(self) -> dict[str, Any]:
        return {
            "active_backend": self._active_backend,
            "ollama": self._ollama_available,
            "ollama_model": settings.ollama.model,
            "external_model": settings.llm.model if settings.llm.api_key else None,
            "cost_guard": self._cost_guard.summary(),
        }

    # ─── Cost-guarded external calls ───────────────────────────

    async def _guarded_external_complete(self, system_prompt: str, user_message: str) -> str:
        """External completion with CostGuard rate limiting."""
        if not self._cost_guard.allow():
            logger.warning("CostGuard blocked external completion — rate limit reached")
            return ("I'm temporarily unable to use the external API due to rate limiting. "
                    "Please try again shortly, or check that Ollama is running.")
        self._cost_guard.record()
        return await self._external_complete(system_prompt, user_message)

    async def _guarded_external_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """External streaming with CostGuard rate limiting."""
        if not self._cost_guard.allow():
            logger.warning("CostGuard blocked external stream — rate limit reached")
            yield ("I'm temporarily unable to use the external API due to rate limiting. "
                   "Please try again shortly, or check that Ollama is running.")
            return
        self._cost_guard.record()
        async for chunk in self._external_stream(messages, temperature, max_tokens):
            yield chunk

    # ─── Ollama ────────────────────────────────────────────────

    async def _ollama_complete(self, system_prompt: str, user_message: str) -> str:
        try:
            resp = await self._http.post(
                f"{settings.ollama.url}/api/chat",
                json={
                    "model": settings.ollama.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 2048},
                },
            )
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error("Ollama completion failed: %s", e)
            if self._active_backend == "ollama" and settings.llm.api_key:
                logger.info("Falling back to external API (cost-guarded)")
                return await self._guarded_external_complete(system_prompt, user_message)
            return f"LLM error: {e}"

    async def _ollama_stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        try:
            async with self._http.stream(
                "POST",
                f"{settings.ollama.url}/api/chat",
                json={
                    "model": settings.ollama.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.7, "num_predict": 2048},
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("Ollama stream failed: %s", e)
            yield f"LLM stream error: {e}"

    # ─── External API (OpenAI-compatible) ──────────────────────

    async def _external_complete(self, system_prompt: str, user_message: str) -> str:
        try:
            resp = await self._http.post(
                f"{settings.llm.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0.7,
                    "max_tokens": self._cost_guard.config.max_tokens_per_request,
                },
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("External API completion failed: %s", e)
            return f"LLM error: {e}"

    async def _external_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        try:
            async with self._http.stream(
                "POST",
                f"{settings.llm.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": min(max_tokens, self._cost_guard.config.max_tokens_per_request),
                    "stream": True,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception as e:
            logger.error("External API stream failed: %s", e)
            yield f"LLM stream error: {e}"

    async def close(self) -> None:
        await self._http.aclose()
