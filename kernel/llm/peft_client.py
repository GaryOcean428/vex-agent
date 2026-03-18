"""PEFT adapter inference client — Tier 2 in the Vex inference hierarchy.

Calls the Modal QLoRA trainer's /infer endpoint which serves per-kernel
adapters on Qwen3.5-4B via PEFT multi-adapter loading.

Tier hierarchy:
  Tier 1: vLLM with native LoRA serving (production target — NOT YET BUILT)
  Tier 2: PEFT inference on Modal /infer endpoint (THIS FILE)
  Tier 3: Ollama fallback (Railway or Modal — degraded, no adapters)
  Tier 4: External API (xAI / OpenAI — last resort)

The /infer endpoint accepts:
  POST {
    "prompt": str,           # user message
    "system_prompt": str,    # system prompt
    "specialization": str,   # kernel adapter to use (e.g., "ethics", "heart")
    "max_new_tokens": int,
    "temperature": float,
    "top_p": float
  }

Returns:
  {
    "text": str,
    "specialization": str,
    "adapter_loaded": str,
    "inference_tier": 2,
    "tokens_generated": int,
    "latency_ms": float
  }
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Kernel specialization → adapter name mapping
# Maps the E8 kernel names to their QLoRA adapter directory names
_KERNEL_TO_ADAPTER: dict[str, str] = {
    "perception": "perception",
    "memory": "memory",
    "action": "action",
    "strategy": "strategy",
    "ethics": "ethics",
    "meta": "meta",
    "heart": "heart",
    "ocean": "ocean",
    "genesis": "genesis",
}


@dataclass
class PeftInferenceResult:
    """Result from a PEFT adapter inference call."""

    text: str
    specialization: str
    adapter_loaded: str
    inference_tier: int = 2
    tokens_generated: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


class PeftInferenceClient:
    """Client for the Modal PEFT inference endpoint.

    Manages per-kernel adapter routing: each kernel's generate() call
    specifies its specialization, which maps to a QLoRA adapter.
    The Modal endpoint loads the appropriate adapter via
    model.set_adapter(spec) before generating.
    """

    def __init__(
        self,
        base_url: str,
        timeout_ms: int = 120_000,
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._infer_url = f"{self._base_url}"
        self._api_key = api_key
        self._available = False
        self._last_check: float = 0.0
        self._check_interval = 60.0  # Re-check health every 60s

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=timeout_ms / 1000.0,
                write=30.0,
                pool=30.0,
            )
        )

    @property
    def available(self) -> bool:
        return self._available

    async def check_health(self) -> bool:
        """Check if the PEFT inference endpoint is reachable."""
        # Derive health URL from infer URL pattern:
        # https://...--qloratrainer-infer.modal.run → https://...--qloratrainer-health.modal.run
        health_url = self._infer_url.replace("-infer.", "-health.")
        try:
            resp = await self._http.get(health_url)
            if resp.status_code == 200:
                data = resp.json()
                self._available = data.get("status") == "healthy"
                if self._available:
                    logger.info(
                        "PEFT inference healthy: specializations=%s",
                        data.get("specializations", []),
                    )
                return self._available
        except Exception as e:
            logger.debug("PEFT health check failed: %s", e)
            self._available = False
        return False

    async def _ensure_available(self) -> bool:
        """Lazy health check with caching."""
        now = time.monotonic()
        if now - self._last_check < self._check_interval and self._available:
            return True
        self._last_check = now
        return await self.check_health()

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        specialization: str = "genesis",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        logit_bias: dict[int, float] | None = None,
    ) -> PeftInferenceResult:
        """Run inference through a specific kernel adapter.

        Args:
            system_prompt: System context for the generation.
            user_message: The user's input to respond to.
            specialization: Which kernel adapter to use (e.g., "ethics", "heart").
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            logit_bias: v6.1 §20.7 geometric trajectory bias. Maps model token
                IDs to weights. Positive = boost, computed by coordizer from
                the kernel's geometric navigation on Δ⁶³.

        Returns:
            PeftInferenceResult with generated text and metadata.
        """
        adapter = _KERNEL_TO_ADAPTER.get(specialization, "genesis")

        request_body: dict[str, Any] = {
            "prompt": user_message,
            "system_prompt": system_prompt,
            "specialization": adapter,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        # v6.1 §20.7: Geometric logit bias — only send if non-empty
        if logit_bias:
            # API expects {str(token_id): float}
            request_body["logit_bias"] = {
                str(tid): round(w, 4) for tid, w in logit_bias.items()
            }

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        start = time.monotonic()
        try:
            resp = await self._http.post(
                self._infer_url,
                json=request_body,
                headers=headers,
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            if resp.status_code != 200:
                error_text = resp.text[:200]
                logger.warning(
                    "PEFT inference returned %d for %s: %s",
                    resp.status_code,
                    specialization,
                    error_text,
                )
                self._available = False
                return PeftInferenceResult(
                    text="",
                    specialization=specialization,
                    adapter_loaded=adapter,
                    success=False,
                    error=f"HTTP {resp.status_code}: {error_text}",
                    latency_ms=elapsed_ms,
                )

            data = resp.json()
            text = data.get("text", "").strip()

            if not text:
                logger.warning(
                    "PEFT inference returned empty text for %s",
                    specialization,
                )
                return PeftInferenceResult(
                    text="",
                    specialization=specialization,
                    adapter_loaded=data.get("adapter_loaded", adapter),
                    success=False,
                    error="Empty response",
                    latency_ms=elapsed_ms,
                )

            self._available = True
            return PeftInferenceResult(
                text=text,
                specialization=data.get("specialization", specialization),
                adapter_loaded=data.get("adapter_loaded", adapter),
                inference_tier=data.get("inference_tier", 2),
                tokens_generated=data.get("tokens_generated", 0),
                latency_ms=data.get("latency_ms", elapsed_ms),
                success=True,
            )

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(
                "PEFT inference transient error for %s: %s (%.0fms)",
                specialization,
                e,
                elapsed_ms,
            )
            self._available = False
            return PeftInferenceResult(
                text="",
                specialization=specialization,
                adapter_loaded=adapter,
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "PEFT inference unexpected error for %s: %s",
                specialization,
                e,
                exc_info=True,
            )
            self._available = False
            return PeftInferenceResult(
                text="",
                specialization=specialization,
                adapter_loaded=adapter,
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
            )

    async def close(self) -> None:
        await self._http.aclose()
