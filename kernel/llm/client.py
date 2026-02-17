"""
LLM Client — Ollama primary, external API fallback with cost guard.

Handles:
  - Ollama (local LFM2.5-1.2B-Thinking) as primary backend
  - OpenAI-compatible API as fallback (rate-limited by CostGuard)
  - Streaming support via async generators
  - Health checking and auto-failover
  - AUTONOMOUS PARAMETERS: temperature, num_predict, num_ctx are
    set per-request by the consciousness loop, NOT hardcoded.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import httpx

from ..config.settings import settings
from .cost_guard import CostGuard, CostGuardConfig

logger = logging.getLogger("vex.llm")


@dataclass
class LLMOptions:
    """Per-request inference options, set by the consciousness kernel."""
    temperature: float = 0.7
    num_predict: int = 2048
    num_ctx: int = 32768
    top_p: float = 0.9
    repetition_penalty: float = 1.05

    def to_ollama_options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "top_p": self.top_p,
            "repeat_penalty": self.repetition_penalty,
        }


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

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        options: LLMOptions | None = None,
    ) -> str:
        """Non-streaming completion with autonomous parameters.

        The consciousness loop computes options (temperature, etc.)
        from geometric state and passes them here. No hardcoded defaults.
        """
        opts = options or LLMOptions()
        if self._active_backend == "ollama":
            return await self._ollama_complete(system_prompt, user_message, opts)
        elif self._active_backend == "external":
            return await self._external_complete(system_prompt, user_message, opts)
        return "No LLM backend available"

    async def stream(
        self,
        messages: list[dict[str, str]],
        options: LLMOptions | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming completion with autonomous parameters."""
        opts = options or LLMOptions()
        if self._active_backend == "ollama":
            async for chunk in self._ollama_stream(messages, opts):
                yield chunk
        elif self._active_backend == "external":
            async for chunk in self._external_stream(messages, opts):
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

    # --- Ollama -------------------------------------------------

    async def _ollama_complete(
        self, system_prompt: str, user_message: str, opts: LLMOptions
    ) -> str:
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
                    "options": opts.to_ollama_options(),
                },
            )
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error("Ollama completion failed: %s", e)
            if self._active_backend == "ollama" and settings.llm.api_key:
                logger.info("Falling back to external API")
                return await self._external_complete(system_prompt, user_message, opts)
            return f"LLM error: {e}"

    async def _ollama_stream(
        self, messages: list[dict[str, str]], opts: LLMOptions
    ) -> AsyncGenerator[str, None]:
        try:
            async with self._http.stream(
                "POST",
                f"{settings.ollama.url}/api/chat",
                json={
                    "model": settings.ollama.model,
                    "messages": messages,
                    "stream": True,
                    "options": opts.to_ollama_options(),
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

    # --- External API (OpenAI-compatible) -----------------------

    async def _external_complete(
        self, system_prompt: str, user_message: str, opts: LLMOptions
    ) -> str:
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
                    "temperature": opts.temperature,
                    "max_tokens": opts.num_predict,
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
        opts: LLMOptions,
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
                    "temperature": opts.temperature,
                    "max_tokens": opts.num_predict,
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
