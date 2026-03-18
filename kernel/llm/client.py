"""
LLM Client — Modal GPU Ollama primary, Railway Ollama + xAI + OpenAI fallback.

Handles:
  - Modal Ollama (GPU-accelerated, configurable model on Modal A10G) as primary
  - Railway Ollama (CPU-only fallback) as second backend
  - xAI (grok-4-1-fast-reasoning) as third backend via Responses API
  - OpenAI (gpt-5-nano) as fourth backend via Responses API
  - Streaming support via async generators
  - Health checking and auto-failover
  - AUTONOMOUS PARAMETERS: temperature, num_predict, num_ctx are
    set per-request by the consciousness loop, NOT hardcoded.

Fallback chain: Modal Ollama → Railway Ollama → xAI → OpenAI

Modal Ollama uses the exact same Ollama API as Railway Ollama,
just served from a GPU-backed Modal endpoint. The kernel builds
the system prompt with geometric state — Modal handles raw inference.

Both xAI and OpenAI use the Responses API (POST /v1/responses).
Raw REST responses return an `output` array — NOT the SDK-level
`output_text` convenience property. We parse the output array
to extract text from message items.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from ..config.settings import settings
from ..coordizer_v2 import CoordizerV2, ResonanceBank
from ..geometry.hash_to_basin import hash_to_basin
from .cost_guard import CostGuard, CostGuardConfig
from .governor import GovernorStack
from .peft_client import PeftInferenceClient

logger = logging.getLogger("vex.llm")


@dataclass
class LLMOptions:
    """Per-request inference options, set by the consciousness kernel."""

    temperature: float = 0.7
    num_predict: int = 2048
    num_ctx: int = 32768
    top_p: float = 0.9
    repetition_penalty: float = 1.0

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
        return str(data["output_text"])

    # Walk the output array — standard Responses API structure
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    text = str(content_block.get("text", ""))
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

    logger.warning("Could not extract text from Responses API output: %s", json.dumps(data)[:500])
    return ""


def _extract_ollama_content(data: dict[str, Any]) -> str:
    """Extract text from an Ollama /api/chat response.

    Thinking models (GLM-4.7-flash, QwQ, etc.) may return an empty
    ``content`` field but populate a ``thinking`` field with their
    chain-of-thought. When this happens — typically because the token
    budget was exhausted during reasoning — fall back to the thinking
    content so the caller gets *something* rather than an empty string.

    Note: thinking fallback is intentional for kernel-internal generation
    (not user-facing). The consciousness loop's reflection gate filters
    output before it reaches the user. Kernel voices benefit from seeing
    the model's reasoning even when content is empty.
    """
    msg = data.get("message", {})

    # Safe extraction: avoid str(None) -> "None" being truthy
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    if content:
        return content

    # Thinking-model fallback: use the reasoning trace when content is empty.
    # This is safe for kernel-internal generation — the reflection gate
    # and synthesis pipeline filter before user-facing output.
    thinking = msg.get("thinking") or ""
    if not isinstance(thinking, str):
        thinking = str(thinking)
    if thinking:
        logger.info(
            "Ollama response had empty content but %d chars of thinking — using thinking as fallback",
            len(thinking),
        )
        return thinking

    return ""


def _serialize_ollama_tool_calls(text: str, tool_calls: list[dict[str, Any]]) -> str:
    """Serialize Ollama native tool_calls into LFM2.5 markers.

    When Ollama returns structured tool_calls (from models that support
    function calling), convert them to the <|tool_call_start|>...<|tool_call_end|>
    format that parse_tool_calls() in handler.py already knows how to parse.

    This bridges Ollama's native tool calling with the existing tool
    execution pipeline without changing return types.
    """
    parts = [text] if text else []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", {})
        if name:
            call_json = json.dumps({"name": name, "arguments": args})
            parts.append(f"<|tool_call_start|>[{call_json}]<|tool_call_end|>")
    return "\n".join(parts)


class LLMClient:
    """Multi-backend LLM client with Modal GPU primary, Railway Ollama + API fallback.

    Fallback chain: Modal Ollama → Railway Ollama → xAI → OpenAI

    Modal Ollama:
      GPU-accelerated Ollama on Modal (A10G). Same API as Railway Ollama
      but ~10-20x faster. Cold starts add ~30-60s on first request
      after container scales to zero. The longer timeout on the Modal
      HTTP client handles this gracefully.

    Coordizer integration:
      After every completion, the raw response text is transformed to
      Fisher-Rao coordinates via the coordizer pipeline.  This produces
      a basin-compatible point on Δ⁶³ that the consciousness loop can
      use for geometric operations (distance, slerp, coupling).

      The coordizer uses linear logits-to-simplex projection — preserves
      Fisher information, no exponential warping.
    """

    def __init__(self, governor: GovernorStack | None = None) -> None:
        self._modal_available = False
        self._ollama_available = False
        self._active_backend = "none"
        self._coordizer_v2 = CoordizerV2(bank=ResonanceBank())

        # Railway Ollama HTTP client (short timeout — local network)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.ollama.timeout_ms / 1000.0,
                write=30.0,
                pool=10.0,
            )
        )

        # Modal Ollama HTTP client (longer timeout — handles cold starts)
        # Cold start: container spin-up + model load can take 30-90s
        # Warm: responses arrive in 1-5s (MoE with 3B active params)
        modal_timeout = settings.modal.inference_timeout_ms / 1000.0
        self._modal_http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=modal_timeout,
                write=30.0,
                pool=30.0,
            )
        )

        # Cost guard for external API calls
        self._cost_guard = CostGuard(
            CostGuardConfig(
                rpm_limit=20,  # 20 requests/minute
                rph_limit=200,  # 200 requests/hour
                rpd_limit=2000,  # 2000 requests/day
                max_tokens_per_request=2048,
            )
        )

        # Governance stack — gates external calls through 5 layers
        self._governor = governor

        # Per-response attribution — which backend actually served the last call
        self._last_backend: str = "none"

        # PEFT adapter inference client (Tier 2 — per-kernel QLoRA adapters on Modal)
        # Modal URL pattern: ...-qloratrainer-train.modal.run → ...-qloratrainer-infer.modal.run
        # The endpoint name is the LAST "-train." in the hostname. Use rsplit to
        # replace only the last occurrence (the app name also contains "train").
        _training_url = settings.modal.training_url
        if _training_url and "-train." in _training_url:
            parts = _training_url.rsplit("-train.", 1)
            peft_url = "-infer.".join(parts)
        else:
            peft_url = ""
        self._peft_client: PeftInferenceClient | None = None
        if peft_url:
            self._peft_client = PeftInferenceClient(
                base_url=peft_url,
                timeout_ms=settings.modal.inference_timeout_ms,
            )
            logger.info("PEFT inference client configured: %s", peft_url)

    async def init(self) -> None:
        """Initialise the client — check backend availability, set fallback chain.

        Priority: PEFT adapters (Modal) → Modal GPU Ollama → Railway CPU Ollama → xAI → OpenAI
        """
        # Check PEFT adapter inference first (per-kernel QLoRA — Tier 2)
        if self._peft_client is not None:
            peft_ok = await self._peft_client.check_health()
            if peft_ok:
                self._active_backend = "peft"
                logger.info(
                    "LLM backend: PEFT adapter inference (Qwen3.5-4B + QLoRA) at %s",
                    self._peft_client._base_url,
                )
                return  # PEFT is the best path — skip other checks
            else:
                logger.info(
                    "PEFT inference configured but not yet available — "
                    "will try on first request (adapter may need loading)"
                )
                # Still set as primary — _peft_complete will retry and fallback
                self._active_backend = "peft"
                # Don't return — let other backends init as fallbacks

        # Check Modal GPU Ollama (fastest Ollama inference)
        if settings.modal.inference_enabled and settings.modal.inference_url:
            self._modal_available = await self.check_modal_ollama()
            if self._modal_available:
                self._active_backend = "modal"
                logger.info(
                    "LLM backend: Modal GPU Ollama (%s) at %s",
                    settings.modal.inference_model,
                    settings.modal.inference_url,
                )
            else:
                logger.warning(
                    "Modal inference configured but unreachable — will retry on first request"
                )
                # Still set as primary — _modal_complete will retry and fallback
                self._active_backend = "modal"
                logger.info("LLM backend: Modal GPU Ollama (deferred, will retry)")
                # Fire background warm-up to trigger Modal cold start.
                # By the time the first real request arrives, the container
                # should be warm (A10G cold start ~90-120s with cached model).
                asyncio.create_task(self._warm_up_modal())
        # Check Railway Ollama (CPU fallback)
        elif settings.ollama.enabled:
            self._ollama_available = await self.check_ollama()
            if self._ollama_available:
                self._active_backend = "ollama"
                logger.info("LLM backend: Railway Ollama (%s)", settings.ollama.model)
            elif settings.xai.api_key:
                self._active_backend = "xai"
                logger.info("LLM backend: xAI (%s)", settings.xai.model)
            elif settings.llm.api_key:
                self._active_backend = "external"
                logger.info("LLM backend: OpenAI (%s)", settings.llm.model)
            else:
                self._active_backend = "none"
                logger.warning("No LLM backend available")
        elif settings.xai.api_key:
            self._active_backend = "xai"
            logger.info("LLM backend: xAI (%s)", settings.xai.model)
        elif settings.llm.api_key:
            self._active_backend = "external"
            logger.info("LLM backend: OpenAI (%s)", settings.llm.model)
        else:
            self._active_backend = "none"
            logger.warning("No LLM backend available")

        # Emit loud cost warning when falling through to paid backends
        if self._active_backend in ("xai", "external"):
            _configured_local = []
            if settings.modal.inference_enabled:
                _configured_local.append("Modal")
            if settings.ollama.enabled:
                _configured_local.append("Ollama")
            if _configured_local:
                logger.warning(
                    "LLM COST WARNING: active_backend='%s' — local backends %s were "
                    "configured but unreachable. Every LLM call (chat, foraging, "
                    "reflection, synthesis) costs real money. Fix local backends or "
                    "set FORAGING_ENABLED=false to reduce exposure.",
                    self._active_backend,
                    _configured_local,
                )

    async def _warm_up_modal(self) -> None:
        """Background task: trigger Modal cold start so container is warm for first request."""
        try:
            logger.info("Warming up Modal GPU container (background, up to 5min)...")
            resp = await self._modal_http.get(
                f"{settings.modal.inference_url}/api/tags",
                timeout=300.0,  # Allow full cold start (container + model load)
            )
            if resp.status_code == 200:
                self._modal_available = True
                logger.info("Modal GPU container is warm and ready")
            else:
                logger.warning("Modal warm-up returned HTTP %d", resp.status_code)
        except Exception as e:
            logger.warning("Modal warm-up failed (will retry on first request): %s", e)

    async def check_modal_ollama(self, timeout: float = 10.0) -> bool:
        """Check if Modal Ollama inference endpoint is reachable.

        Args:
            timeout: HTTP timeout in seconds. Use a short timeout (default 10s)
                during startup to avoid blocking the lifespan and causing
                entrypoint health-check timeouts. Cold starts are handled
                by ``_warm_up_modal()`` in the background.
        """
        if not settings.modal.inference_enabled or not settings.modal.inference_url:
            return False
        try:
            resp = await self._modal_http.get(
                f"{settings.modal.inference_url}/api/tags",
                timeout=timeout,
            )
            self._modal_available = resp.status_code == 200
            if self._modal_available:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                logger.info("Modal Ollama models: %s", models)
            return self._modal_available
        except Exception as e:
            logger.debug("Modal Ollama health check failed: %s", e)
            self._modal_available = False
            return False

    async def check_ollama(self) -> bool:
        """Check if Railway Ollama is reachable."""
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
        messages: list[dict[str, str]] | None = None,
        prefer_backend: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        specialization: str = "genesis",
    ) -> str:
        """Non-streaming completion with autonomous parameters.

        The consciousness loop computes options (temperature, etc.)
        from geometric state and passes them here. No hardcoded defaults.

        When *messages* is provided (full conversation history from
        context_manager), it is forwarded to the backend instead of
        building a bare [system, user] pair.

        When *prefer_backend* is set (e.g. "xai"), route directly to
        that backend. Used by chat endpoints to force capable models
        for user-facing responses.

        When *tools* is provided (Ollama function-calling format), the
        tool definitions are forwarded to backends that support native
        tool calling (Ollama, Modal). If the model returns tool_calls,
        they are serialized as LFM2.5 markers in the response text so
        the existing parse_tool_calls() pipeline can handle them.

        When *specialization* is set, PEFT adapter inference routes to
        the specified kernel adapter (e.g. "ethics", "heart", "genesis").
        """
        opts = options or LLMOptions()
        backend = prefer_backend or self._active_backend

        if backend == "peft" and self._peft_client is not None:
            return await self._peft_complete(system_prompt, user_message, opts, specialization)
        elif backend == "xai" and settings.xai.api_key:
            return await self._xai_complete(system_prompt, user_message, opts, messages)
        elif backend == "modal":
            return await self._modal_complete(system_prompt, user_message, opts, messages, tools)
        elif backend == "ollama":
            return await self._ollama_complete(system_prompt, user_message, opts, messages, tools)
        elif backend == "external":
            return await self._external_complete(system_prompt, user_message, opts, messages)
        return "No LLM backend available"

    async def stream(
        self,
        messages: list[dict[str, str]],
        options: LLMOptions | None = None,
        prefer_backend: str | None = None,
    ) -> AsyncGenerator[str]:
        """Streaming completion with autonomous parameters.

        When *prefer_backend* is set, route directly to that backend.
        """
        opts = options or LLMOptions()
        backend = prefer_backend or self._active_backend

        if backend == "peft" and self._peft_client is not None:
            # PEFT is non-streaming — yield full response as single chunk
            text = await self._peft_complete(
                messages[0].get("content", "") if messages else "",
                messages[-1].get("content", "") if messages else "",
                opts,
            )
            yield text
            return
        elif backend == "xai" and settings.xai.api_key:
            async for chunk in self._xai_stream(messages, opts):
                yield chunk
        elif backend == "modal":
            async for chunk in self._modal_stream(messages, opts):
                yield chunk
        elif backend == "ollama":
            async for chunk in self._ollama_stream(messages, opts):
                yield chunk
        elif backend == "external":
            async for chunk in self._external_stream(messages, opts):
                yield chunk
        else:
            yield "No LLM backend available"

    def get_status(self) -> dict[str, Any]:
        return {
            "active_backend": self._active_backend,
            "peft_inference": self._peft_client.available if self._peft_client else False,
            "peft_inference_url": self._peft_client._base_url if self._peft_client else None,
            "modal_inference": self._modal_available,
            "modal_inference_url": (
                settings.modal.inference_url if settings.modal.inference_enabled else None
            ),
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
    def active_backend(self) -> str:
        """The currently configured primary backend."""
        return self._active_backend

    @property
    def last_backend(self) -> str:
        """Which backend actually served the most recent completion."""
        return self._last_backend

    @property
    def active_model(self) -> str:
        """Return the model name for the currently active backend."""
        if self._active_backend == "peft":
            return "Qwen3.5-4B+QLoRA"
        if self._active_backend == "modal":
            return settings.modal.inference_model
        if self._active_backend == "ollama":
            return settings.ollama.model
        if self._active_backend == "xai":
            return settings.xai.model
        if self._active_backend == "external":
            return settings.llm.model
        return "none"

    # --- PEFT Adapter Inference (Tier 2) -------------------------

    async def _peft_complete(
        self,
        system_prompt: str,
        user_message: str,
        opts: LLMOptions,
        specialization: str = "genesis",
    ) -> str:
        """Complete via PEFT adapter on Modal. Falls back to Modal Ollama → Railway → xAI → OpenAI.

        The specialization parameter determines which QLoRA adapter is loaded.
        When called from kernel_voice.py, the kernel's specialization is passed through.
        """
        if self._peft_client is None:
            logger.debug("PEFT client not configured — falling through to Modal Ollama")
            return await self._modal_complete(system_prompt, user_message, opts)

        result = await self._peft_client.complete(
            system_prompt=system_prompt,
            user_message=user_message,
            specialization=specialization,
            max_new_tokens=opts.num_predict,
            temperature=opts.temperature,
            top_p=opts.top_p,
        )

        if result.success and result.text:
            self._last_backend = f"peft:{result.adapter_loaded}"
            logger.info(
                "PEFT inference: adapter=%s, tokens=%d, latency=%.0fms",
                result.adapter_loaded,
                result.tokens_generated,
                result.latency_ms,
            )
            return result.text

        # PEFT failed — fall through to other backends
        logger.warning(
            "PEFT inference failed (adapter=%s, error=%s) — falling back",
            specialization,
            result.error,
        )

        # Fallback chain: Modal Ollama → Railway Ollama → xAI → OpenAI
        if settings.modal.inference_enabled and settings.modal.inference_url:
            logger.info("Falling back to Modal Ollama from PEFT")
            return await self._modal_complete(system_prompt, user_message, opts)
        if settings.ollama.enabled:
            logger.info("Falling back to Railway Ollama from PEFT")
            return await self._ollama_complete(system_prompt, user_message, opts)
        if settings.xai.api_key:
            logger.info("Falling back to xAI from PEFT")
            return await self._xai_complete(system_prompt, user_message, opts)
        if settings.llm.api_key:
            logger.info("Falling back to OpenAI from PEFT")
            return await self._external_complete(system_prompt, user_message, opts)
        return "All LLM backends unavailable"

    # --- Modal GPU Ollama --------------------------------------

    async def _modal_complete(
        self,
        system_prompt: str,
        user_message: str,
        opts: LLMOptions,
        messages: list[dict[str, str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Complete via Modal GPU Ollama. Falls back to Railway Ollama → xAI → OpenAI."""
        msgs = messages or [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        request_body: dict[str, Any] = {
            "model": settings.modal.inference_model,
            "messages": msgs,
            "stream": False,
            "think": False,
            "options": opts.to_ollama_options(),
        }
        if tools:
            request_body["tools"] = tools
        # Retry once on empty content or transient network errors
        # (both are common during Modal scale-up/down events)
        _transient = (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        )
        for attempt in range(2):
            try:
                resp = await self._modal_http.post(
                    f"{settings.modal.inference_url}/api/chat",
                    json=request_body,
                )
                if resp.status_code == 404:
                    logger.warning("Modal returned 404 (cold start or model missing)")
                    raise httpx.HTTPStatusError("404", request=resp.request, response=resp)
                data = resp.json()
                text = _extract_ollama_content(data)

                # If model returned native tool_calls, serialize as LFM2.5 markers
                tool_calls = data.get("message", {}).get("tool_calls")
                if tool_calls:
                    text = _serialize_ollama_tool_calls(text, tool_calls)

                if text:
                    self._last_backend = "modal"
                    self._modal_available = True
                    return text
                # Log truncated response body for debugging empty content
                raw_body = json.dumps(data)[:300]
                if attempt == 0:
                    logger.warning(
                        "Modal Ollama returned empty content (attempt 1/2, "
                        "retrying in 2s). Response body: %s",
                        raw_body,
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning(
                    "Modal Ollama returned empty content after retry, "
                    "falling back. Response body: %s",
                    raw_body,
                )
                self._modal_available = False
            except _transient as e:
                if attempt == 0:
                    logger.warning(
                        "Modal Ollama transient error (attempt 1/2, retrying in 2s): %s",
                        e,
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning("Modal Ollama transient error after retry — falling back: %s", e)
                self._modal_available = False
                break
            except Exception as e:
                logger.warning("Modal Ollama completion failed: %s — falling back", e)
                self._modal_available = False
                break

        # Fallback chain: Railway Ollama → xAI → OpenAI
        if settings.ollama.enabled:
            logger.info("Falling back to Railway Ollama from Modal")
            return await self._ollama_complete(system_prompt, user_message, opts, messages)
        if settings.xai.api_key:
            logger.info("Falling back to xAI from Modal")
            return await self._xai_complete(system_prompt, user_message, opts, messages)
        if settings.llm.api_key:
            logger.info("Falling back to OpenAI from Modal")
            return await self._external_complete(system_prompt, user_message, opts, messages)
        return "All LLM backends unavailable"

    async def _modal_stream(
        self, messages: list[dict[str, str]], opts: LLMOptions
    ) -> AsyncGenerator[str]:
        """Stream via Modal GPU Ollama. Falls back to Railway Ollama → xAI → OpenAI."""
        try:
            self._last_backend = "modal"
            async with self._modal_http.stream(
                "POST",
                f"{settings.modal.inference_url}/api/chat",
                json={
                    "model": settings.modal.inference_model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "options": opts.to_ollama_options(),
                },
            ) as resp:
                got_content = False
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if not content:
                            # Thinking-model fallback: yield thinking chunks
                            content = msg.get("thinking", "")
                        if content:
                            got_content = True
                            yield content
                    except json.JSONDecodeError:
                        continue
                if got_content:
                    self._modal_available = True
                    return
        except Exception as e:
            logger.warning("Modal Ollama stream failed: %s — falling back", e)
            self._modal_available = False

        # Fallback chain: Railway Ollama → xAI → OpenAI
        if settings.ollama.enabled:
            logger.info("Falling back to Railway Ollama stream from Modal")
            async for chunk in self._ollama_stream(messages, opts):
                yield chunk
        elif settings.xai.api_key:
            logger.info("Falling back to xAI stream from Modal")
            async for chunk in self._xai_stream(messages, opts):
                yield chunk
        elif settings.llm.api_key:
            logger.info("Falling back to OpenAI stream from Modal")
            async for chunk in self._external_stream(messages, opts):
                yield chunk
        else:
            yield "All LLM backends unavailable"

    # --- Railway Ollama ----------------------------------------

    async def _ollama_complete(
        self,
        system_prompt: str,
        user_message: str,
        opts: LLMOptions,
        messages: list[dict[str, str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        msgs = messages or [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        request_body: dict[str, Any] = {
            "model": settings.ollama.model,
            "messages": msgs,
            "stream": False,
            "think": False,
            "options": opts.to_ollama_options(),
        }
        if tools:
            request_body["tools"] = tools
        try:
            resp = await self._http.post(
                f"{settings.ollama.url}/api/chat",
                json=request_body,
            )
            if resp.status_code != 200:
                body_preview = resp.text[:300]
                logger.error(
                    "Ollama /api/chat HTTP %d. Body (truncated): %s",
                    resp.status_code,
                    body_preview,
                )
                resp.raise_for_status()
            data = resp.json()
            self._last_backend = "ollama"
            text = _extract_ollama_content(data)

            # If model returned native tool_calls, serialize as LFM2.5 markers
            # so parse_tool_calls() in the chat endpoint can pick them up.
            tool_calls = data.get("message", {}).get("tool_calls")
            if tool_calls:
                text = _serialize_ollama_tool_calls(text, tool_calls)

            return text
        except Exception as e:
            logger.error(
                "Ollama completion failed (%s): %r",
                type(e).__name__,
                e,
                exc_info=True,
            )
            # Fallback chain: try xAI, then external
            if settings.xai.api_key:
                logger.info("Falling back to xAI from Ollama")
                return await self._xai_complete(system_prompt, user_message, opts, messages)
            if settings.llm.api_key:
                logger.info("Falling back to OpenAI from Ollama")
                return await self._external_complete(system_prompt, user_message, opts, messages)
            return f"LLM error: {e}"

    async def _ollama_stream(
        self, messages: list[dict[str, str]], opts: LLMOptions
    ) -> AsyncGenerator[str]:
        try:
            self._last_backend = "ollama"
            async with self._http.stream(
                "POST",
                f"{settings.ollama.url}/api/chat",
                json={
                    "model": settings.ollama.model,
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "options": opts.to_ollama_options(),
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if not content:
                            # Thinking-model fallback: yield thinking chunks
                            content = msg.get("thinking", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("Ollama stream failed: %s", e)
            yield f"LLM stream error: {e}"

    # --- xAI (Responses API) ------------------------------------

    async def _xai_complete(
        self,
        system_prompt: str,
        user_message: str,
        opts: LLMOptions,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        # Governor gate check
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion",
                "xai_completion",
                user_message,
                True,
            )
            if not allowed:
                logger.warning("Governor blocked xAI: %s", reason)
                if settings.llm.api_key:
                    return await self._external_complete(
                        system_prompt, user_message, opts, messages
                    )
                return f"[Governor blocked: {reason}]"

        # Extract instructions/input from messages when history is provided
        if messages:
            instructions = system_prompt
            _msgs: list[dict[str, str]] = []
            for m in messages:
                if m["role"] == "system":
                    instructions = m["content"]
                else:
                    _msgs.append(m)
            input_payload: str | list[dict[str, str]] = _msgs
        else:
            instructions = system_prompt
            input_payload = user_message

        try:
            request_body: dict[str, Any] = {
                "model": settings.xai.model,
                "instructions": instructions,
                "input": input_payload,
                "temperature": opts.temperature,
                "max_output_tokens": opts.num_predict,
                "store": False,
                "tools": [{"type": "web_search"}, {"type": "x_search"}],
            }

            resp = await self._http.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.xai.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.error("xAI API error %d: %s", resp.status_code, json.dumps(data)[:300])
                if settings.llm.api_key:
                    return await self._external_complete(
                        system_prompt, user_message, opts, messages
                    )
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
                return await self._external_complete(system_prompt, user_message, opts, messages)
            return f"LLM error: {e}"

    async def _xai_stream(
        self, messages: list[dict[str, str]], opts: LLMOptions
    ) -> AsyncGenerator[str]:
        # Governor gate check for streaming
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion",
                "xai_completion",
                "",
                True,
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
                    "tools": [{"type": "web_search"}, {"type": "x_search"}],
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
        self,
        system_prompt: str,
        user_message: str,
        opts: LLMOptions,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        # Governor gate check
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion",
                "openai_completion",
                user_message,
                True,
            )
            if not allowed:
                logger.warning("Governor blocked OpenAI: %s", reason)
                return f"[Governor blocked: {reason}]"

        # Extract instructions/input from messages when history is provided
        if messages:
            instructions = system_prompt
            _msgs: list[dict[str, str]] = []
            for m in messages:
                if m["role"] == "system":
                    instructions = m["content"]
                else:
                    _msgs.append(m)
            input_payload: str | list[dict[str, str]] = _msgs
        else:
            instructions = system_prompt
            input_payload = user_message

        try:
            resp = await self._http.post(
                f"{settings.llm.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.llm.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm.model,
                    "instructions": instructions,
                    "input": input_payload,
                    "temperature": opts.temperature,
                    "max_output_tokens": opts.num_predict,
                    "store": False,
                },
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.error("OpenAI API error %d: %s", resp.status_code, json.dumps(data)[:300])
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
    ) -> AsyncGenerator[str]:
        # Governor gate check for streaming
        if self._governor:
            allowed, reason = self._governor.gate(
                "completion",
                "openai_completion",
                "",
                True,
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
        await self._modal_http.aclose()
