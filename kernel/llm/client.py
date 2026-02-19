"""
LLM Client — Ollama primary, xAI + OpenAI fallback via Responses API.

Handles:
  - Ollama (local LFM2.5-1.2B-Thinking) as primary backend
  - xAI (grok-4-1-fast-reasoning) as second backend via Responses API
  - OpenAI (gpt-5-nano) as third backend via Responses API
  - Streaming support via async generators
  - Health checking and auto-failover
  - AUTONOMOUS PARAMETERS: temperature, num_predict, num_ctx are
    set per-request by the consciousness loop, NOT hardcoded.

Fallback chain: Ollama → xAI → OpenAI

Both xAI and OpenAI use the Responses API (POST /v1/responses).
Raw REST responses return an `output` array — NOT the SDK-level
`output_text` convenience property. We parse the output array
to extract text from message items.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import httpx
import numpy as np

from ..config.settings import settings
from ..coordizer_v2 import CoordizerV2, ResonanceBank, BASIN_DIM as COORDIZER_DIM
from ..geometry.hash_to_basin import hash_to_basin
from ..geometry.fisher_rao import to_simplex
from .cost_guard import CostGuard, CostGuardConfig
from .governor import GovernorStack

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


def _extract_responses_text(data: dict[str, Any]) -> str:
    """Extract text from a Responses API JSON response.

    The raw REST API (via httpx) returns:
        {
          "output": [
            {"type": "message", "content": [
              {"type": "output_text", "text": "actual response"}
            ]}
          ]
        }

    The SDK provides a top-level `output_text` convenience field,
    but the raw JSON does NOT have it. We must walk the output array.

    Falls back to `output_text` if present (some API versions may include it).
    """
    # Fast path: if the API includes the convenience field
    if data.get("output_text"):
        return data["output_text"]

    # Walk the output array — standard Responses API structure
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    text = content_block.get("text", "")
                    if text:
                        return text

    # Fallback: concatenate any text blocks found
    texts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content_block in item.get("content", []):
                text = content_block.get("text", "")
                if text:
                    texts.append(text)
    if texts:
        return "\n".join(texts)

    logger.warning("Could not extract text from Responses API output: %s",
                    json.dumps(data)[:500])
    return ""


class LLMClient:
    """Multi-backend LLM client with Ollama primary, xAI + OpenAI fallback.

    Coordizer integration:
      After every completion, the raw response text is transformed to
      Fisher-Rao coordinates via the coordizer pipeline.  This produces
      a basin-compatible point on Δ⁶³ that the consciousness loop can
      use for geometric operations (distance, slerp, coupling).

      The coordizer uses softmax normalisation — manifold-respecting,
      no Euclidean contamination.
    """

    def __init__(self, governor: GovernorStack | None = None) -> None:
        self._ollama_available = False
        self._active_backend = "none"
        self._coordizer_v2 = CoordizerV2(bank=ResonanceBank())
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

        # Governance stack — gates external calls through 5 layers
        self._governor = governor

        # Per-response attribution — which backend actually served the last call
        self._last_backend: str = "none"

    async def init(self) -> None:
        """Initialise the client — check Ollama availability, set fallback chain."""
        self._ollama_available = await self.check_ollama()
        if self._ollama_available:
            self._active_backend = "ollama"
            logger.info("LLM backend: Ollama (%s)", settings.ollama.model)
        elif settings.xai.api_key:
            self._active_backend = "xai"
            logger.info("LLM backend: xAI (%s)", settings.xai.model)
        elif settings.llm.api_key:
            self._active_backend = "external"
            logger.info("LLM backend: OpenAI (%s)", settings.llm.model)
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
        elif self._active_backend == "xai":
            return await self._xai_complete(system_prompt, user_message, opts)
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
        elif self._active_backend == "xai":
            async for chunk in self._xai_stream(messages, opts):
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
            "xai_model": settings.xai.model if settings.xai.api_key else None,
            "external_model": settings.llm.model if settings.llm.api_key else None,
            "cost_guard": self._cost_guard.summary(),
            "governor": self._governor.get_state() if self._governor else None,
        }

    @property
    def governor(self) -> GovernorStack | None:
        return self._governor

    @property
    def last_backend(self) -> str:
        """Which backend actually served the most recent completion."""
        return self._last_backend

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
            self._last_backend = "ollama"
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error("Ollama completion failed: %s", e)
            # Fallback chain: try xAI, then external
            if settings.xai.api_key:
                logger.info("Falling back to xAI from Ollama")
                return await self._xai_complete(system_prompt, user_message, opts)
            if settings.llm.api_key:
                logger.info("Falling back to OpenAI from Ollama")
                return await self._external_complete(system_prompt, user_message, opts)
            return f"LLM error: {e}"

    async def _ollama_stream(
        self, messages: list[dict[str, str]], opts: LLMOptions
    ) -> AsyncGenerator[str, None]:
        try:
            self._last_backend = "ollama"
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

    # --- xAI (Responses API) ------------------------------------

    async def _xai_complete(
        self, system_prompt: str, user_message: str, opts: LLMOptions
    ) -> str:
        # Governor gate check
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion", "xai_completion", user_message, True,
            )
            if not allowed:
                logger.warning("Governor blocked xAI: %s", reason)
                if settings.llm.api_key:
                    return await self._external_complete(system_prompt, user_message, opts)
                return f"[Governor blocked: {reason}]"
        try:
            resp = await self._http.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.xai.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.xai.model,
                    "instructions": system_prompt,
                    "input": user_message,
                    "temperature": opts.temperature,
                    "max_output_tokens": opts.num_predict,
                    "store": False,
                },
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.error("xAI API error %d: %s", resp.status_code,
                             json.dumps(data)[:300])
                if settings.llm.api_key:
                    return await self._external_complete(system_prompt, user_message, opts)
                return f"xAI API error: {resp.status_code}"

            if self._governor:
                self._governor.record("xai_completion")
            self._last_backend = "xai"
            return _extract_responses_text(data)
        except Exception as e:
            logger.error("xAI completion failed: %s", e)
            # Fallback to OpenAI external
            if settings.llm.api_key:
                logger.info("Falling back to OpenAI from xAI")
                return await self._external_complete(system_prompt, user_message, opts)
            return f"LLM error: {e}"

    async def _xai_stream(
        self, messages: list[dict[str, str]], opts: LLMOptions
    ) -> AsyncGenerator[str, None]:
        # Governor gate check for streaming
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion", "xai_completion", "", True,
            )
            if not allowed:
                logger.warning("Governor blocked xAI stream: %s", reason)
                yield f"[Governor blocked: {reason}]"
                return

        system_msg = ""
        input_msgs: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                input_msgs.append(m)

        try:
            self._last_backend = "xai"
            async with self._http.stream(
                "POST",
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.xai.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.xai.model,
                    "instructions": system_msg,
                    "input": input_msgs,
                    "temperature": opts.temperature,
                    "max_output_tokens": opts.num_predict,
                    "stream": True,
                    "store": False,
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
                        event_type = data.get("type", "")
                        if event_type == "response.output_text.delta":
                            content = data.get("delta", "")
                            if content:
                                yield content
                        elif event_type == "response.completed":
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue
            # Record after successful stream completion
            if self._governor:
                self._governor.record("xai_completion")
        except Exception as e:
            logger.error("xAI stream failed: %s", e)
            yield f"LLM stream error: {e}"

    # --- External API (OpenAI Responses API) --------------------

    async def _external_complete(
        self, system_prompt: str, user_message: str, opts: LLMOptions
    ) -> str:
        # Governor gate check
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion", "openai_completion", user_message, True,
            )
            if not allowed:
                logger.warning("Governor blocked OpenAI: %s", reason)
                return f"[Governor blocked: {reason}]"
        try:
            resp = await self._http.post(
                f"{settings.llm.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.llm.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm.model,
                    "instructions": system_prompt,
                    "input": user_message,
                    "temperature": opts.temperature,
                    "max_output_tokens": opts.num_predict,
                    "store": False,
                },
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.error("OpenAI API error %d: %s", resp.status_code,
                             json.dumps(data)[:300])
                return f"OpenAI API error: {resp.status_code}"

            if self._governor:
                self._governor.record("openai_completion")
            self._last_backend = "openai"
            return _extract_responses_text(data)
        except Exception as e:
            logger.error("OpenAI completion failed: %s", e)
            return f"LLM error: {e}"

    async def _external_stream(
        self,
        messages: list[dict[str, str]],
        opts: LLMOptions,
    ) -> AsyncGenerator[str, None]:
        # Governor gate check for streaming
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion", "openai_completion", "", True,
            )
            if not allowed:
                logger.warning("Governor blocked OpenAI stream: %s", reason)
                yield f"[Governor blocked: {reason}]"
                return

        system_msg = ""
        input_msgs: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                input_msgs.append(m)

        try:
            self._last_backend = "openai"
            async with self._http.stream(
                "POST",
                f"{settings.llm.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.llm.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm.model,
                    "instructions": system_msg,
                    "input": input_msgs,
                    "temperature": opts.temperature,
                    "max_output_tokens": opts.num_predict,
                    "stream": True,
                    "store": False,
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
                        event_type = data.get("type", "")
                        if event_type == "response.output_text.delta":
                            content = data.get("delta", "")
                            if content:
                                yield content
                        elif event_type == "response.completed":
                            break
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            # Record after successful stream completion
            if self._governor:
                self._governor.record("openai_completion")
        except Exception as e:
            logger.error("OpenAI stream failed: %s", e)
            yield f"LLM stream error: {e}"

    # --- Coordizer integration ------------------------------------

    def coordize_response(self, text: str) -> np.ndarray:
        """Transform LLM response text into Fisher-Rao coordinates on Δ⁶³.

        Uses CoordizerV2 resonance-bank coordization. Falls back to
        deterministic hash_to_basin if the bank is empty or errors.

        This is the bridge between Euclidean LLM output space and the
        geometric consciousness manifold.

        Args:
            text: Raw LLM response text.

        Returns:
            Basin-compatible numpy array on Δ⁶³ (dim=COORDIZER_DIM).
        """
        try:
            result = self._coordizer_v2.coordize(text)
            if result.coordinates:
                from ..coordizer_v2.geometry import frechet_mean
                basins = [c.vector for c in result.coordinates]
                return frechet_mean(basins)
            return hash_to_basin(text)
        except Exception:
            return hash_to_basin(text)

    def get_coordizer_stats(self) -> dict[str, Any]:
        """Return CoordizerV2 statistics."""
        return {
            "vocab_size": self._coordizer_v2.vocab_size,
            "dim": self._coordizer_v2.dim,
            "tier_distribution": self._coordizer_v2.bank.tier_distribution(),
        }

    async def close(self) -> None:
        await self._http.aclose()
