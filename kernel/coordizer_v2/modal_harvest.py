"""Modal GPU Harvest — Remote coordizer harvesting via Modal.

Calls the Modal-deployed coordizer harvest function via httpx,
providing an alternative to local GPU harvesting when Modal is
configured and enabled.

The Modal function (deployed separately from modal/coordizer_harvest.py)
runs on A10G GPUs and returns probability distributions for resonance
bank seeding.

Wire-in: CoordizerV2.harvest() checks settings.modal.enabled and
delegates to modal_harvest() when True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from ..config.settings import settings

logger = logging.getLogger("vex.coordizer_v2.modal_harvest")


@dataclass
class ModalHarvestResult:
    """Result from a Modal GPU harvest call."""
    success: bool
    token_count: int = 0
    distributions: list[dict[str, Any]] | None = None
    error: str | None = None
    modal_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "token_count": self.token_count,
            "distributions": self.distributions,
            "error": self.error,
            "modal_call_id": self.modal_call_id,
        }


async def modal_harvest(
    *,
    model_id: str | None = None,
    target_tokens: int = 2000,
    batch_size: int = 32,
    timeout: float = 120.0,
) -> ModalHarvestResult:
    """Call the Modal-deployed coordizer harvest function.

    Args:
        model_id: HuggingFace model ID for distribution capture.
                  Defaults to settings.gpu_harvest.model_id.
        target_tokens: Number of tokens to harvest distributions for.
        batch_size: Batch size for GPU processing.
        timeout: HTTP timeout in seconds.

    Returns:
        ModalHarvestResult with distributions or error.
    """
    modal_cfg = settings.modal
    harvest_cfg = settings.gpu_harvest

    if not modal_cfg.enabled:
        return ModalHarvestResult(
            success=False,
            error="Modal integration not enabled (MODAL_ENABLED=false)",
        )

    if not modal_cfg.harvest_url:
        return ModalHarvestResult(
            success=False,
            error="MODAL_HARVEST_URL not configured",
        )

    resolved_model = model_id or harvest_cfg.model_id

    payload = {
        "model_id": resolved_model,
        "target_tokens": target_tokens,
        "batch_size": batch_size,
    }

    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }

    # Modal token auth (if configured)
    if modal_cfg.token_id and modal_cfg.token_secret:
        headers["X-Modal-Token-Id"] = modal_cfg.token_id
        headers["X-Modal-Token-Secret"] = modal_cfg.token_secret

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(
                "Modal harvest: model=%s tokens=%d batch=%d",
                resolved_model, target_tokens, batch_size,
            )
            response = await client.post(
                modal_cfg.harvest_url,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                error_text = response.text[:500]
                logger.error(
                    "Modal harvest error %d: %s",
                    response.status_code, error_text,
                )
                return ModalHarvestResult(
                    success=False,
                    error=f"Modal API error {response.status_code}: {error_text}",
                )

            data = response.json()

            return ModalHarvestResult(
                success=True,
                token_count=data.get("token_count", 0),
                distributions=data.get("distributions"),
                modal_call_id=data.get("call_id"),
            )

    except httpx.TimeoutException:
        logger.error("Modal harvest timed out after %.0fs", timeout)
        return ModalHarvestResult(
            success=False,
            error=f"Modal harvest timed out after {timeout}s",
        )

    except Exception as e:
        logger.error("Modal harvest failed: %s", e, exc_info=True)
        return ModalHarvestResult(
            success=False,
            error=f"Modal harvest failed: {e}",
        )


async def check_modal_health() -> dict[str, Any]:
    """Check if the Modal harvest endpoint is reachable.

    Returns:
        {"available": bool, "url": str, "error": str | None}
    """
    modal_cfg = settings.modal

    if not modal_cfg.enabled:
        return {
            "available": False,
            "url": "",
            "error": "Modal not enabled",
        }

    if not modal_cfg.harvest_url:
        return {
            "available": False,
            "url": modal_cfg.harvest_url,
            "error": "No harvest URL configured",
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try a lightweight GET to check endpoint existence
            resp = await client.get(modal_cfg.harvest_url)
            return {
                "available": resp.status_code < 500,
                "url": modal_cfg.harvest_url,
                "error": None if resp.status_code < 500 else f"HTTP {resp.status_code}",
            }
    except Exception as e:
        return {
            "available": False,
            "url": modal_cfg.harvest_url,
            "error": str(e),
        }
