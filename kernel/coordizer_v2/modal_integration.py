"""
Modal Integration — Railway-side client for GPU harvesting

Connects the Railway-deployed Vex kernel to the Modal GPU endpoint
for coordizer harvesting. Handles authentication, retry, fallback
to synthetic data, and transformation of raw logits to Δ⁶³ basin
coordinates via CoordizerV2 compression.

Configuration:
    Uses kernel.config.settings.modal (unified env vars):
        MODAL_ENABLED=true
        MODAL_HARVEST_URL=https://<your-modal-app>.modal.run/harvest
        MODAL_TOKEN_ID=wk-...
        MODAL_TOKEN_SECRET=ws-...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config.settings import settings as kernel_settings
from .harvest import HarvestResult

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────


@dataclass
class ModalIntegrationConfig:
    """Runtime configuration for Modal GPU harvesting.

    Derived from kernel.config.settings.ModalConfig to avoid
    env-var duplication. All env vars are read once via settings.modal.

    Note: token_id and token_secret are retained for Modal proxy auth
    on non-public endpoints. The harvest endpoint currently uses Modal's
    network controls, so these are not sent as headers, but they remain
    available if proxy auth is re-enabled.
    """

    enabled: bool = False
    harvest_url: str = ""
    health_url: str = ""
    token_id: str = ""
    token_secret: str = ""
    harvest_auth_token: str = ""
    timeout: int = 120
    max_retries: int = 3
    batch_size: int = 32

    @classmethod
    def from_settings(cls) -> ModalIntegrationConfig:
        """Load configuration from kernel.config.settings.modal."""
        modal = kernel_settings.modal
        harvest_url = modal.harvest_url

        if modal.harvest_health_url:
            health_url = modal.harvest_health_url
        else:
            # Health endpoint: prefer hostname-based pattern when present,
            # otherwise fall back to a conventional /health path to avoid
            # health == harvest (which would send GET to a POST endpoint).
            if "-harvest.modal.run" in harvest_url:
                health_url = harvest_url.replace(
                    "-harvest.modal.run",
                    "-health.modal.run",
                )
            else:
                health_url = harvest_url.rstrip("/") + "/health"

        return cls(
            enabled=modal.enabled,
            harvest_url=harvest_url,
            health_url=health_url,
            token_id=modal.token_id,
            token_secret=modal.token_secret,
            harvest_auth_token=kernel_settings.kernel_api_key,
        )

    def is_configured(self) -> bool:
        """Check if all required fields are present."""
        return bool(self.enabled and self.harvest_url)


# ─── Modal Client ────────────────────────────────────────────────────


class ModalHarvestClient:
    """Async client for the Modal GPU harvest endpoint.

    Handles:
        - Proxy auth headers (wk-*/ws-*)
        - Batch splitting for large corpora
        - Retry on transient failures
        - Fallback to synthetic data
        - Transformation of raw logits to HarvestResult format
    """

    def __init__(self, config: ModalIntegrationConfig | None = None):
        self.config = config or ModalIntegrationConfig.from_settings()
        self._healthy: bool | None = None

    async def check_health(self) -> bool:
        """Check if the Modal endpoint is reachable."""
        if not self.config.is_configured():
            logger.warning("Modal not configured — check environment variables")
            return False

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(self.config.health_url)
                if resp.status_code == 200:
                    data = resp.json()
                    self._healthy = data.get("status") == "ok"
                    logger.info(
                        f"Modal health: {data.get('status')} (vocab={data.get('vocab_size')})"
                    )
                else:
                    self._healthy = False
                    logger.warning(f"Modal health HTTP {resp.status_code}")
                return self._healthy

        except Exception as e:
            logger.warning(f"Modal health check failed: {e}")
            self._healthy = False
            return False

    async def harvest(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> dict[str, Any] | None:
        """Send texts to Modal for GPU harvesting.

        Args:
            texts: List of text strings to harvest
            max_length: Max token length per text

        Returns:
            Raw harvest response dict, or None on failure.
            Response format (matches Modal endpoint):
                {
                    "success": True,
                    "vocab_size": 65536,
                    "total_resonances_processed": 12345,
                    "tokens": {
                        "42": {
                            "string": "hello",
                            "fingerprint": [0.001, 0.002, ...],
                            "context_count": 15
                        },
                        ...
                    },
                    "elapsed_seconds": 45.2
                }
        """
        if not self.config.is_configured():
            logger.warning("Modal not configured, skipping harvest")
            return None

        # Send all texts in a single request — the Modal endpoint
        # aggregates across all texts internally (Fréchet means per coordinate).
        result = await self._harvest_batch(texts, max_length)
        if result is None or not result.get("success"):
            return None

        return result

    async def harvest_to_fingerprints(
        self,
        texts: list[str],
        min_contexts: int = 10,
        max_length: int = 512,
    ) -> dict[str, Any] | None:
        """Harvest and compute Fréchet mean fingerprints per coordinate.

        This is the full pipeline:
            texts → Modal GPU → raw logits → Fréchet means on Δ^(V-1)

        Returns dict of {token_id: fingerprint_array} ready for
        CoordizerV2 compression.
        """
        raw = await self.harvest(texts, max_length)
        if raw is None:
            return None

        vocab_size = raw["vocab_size"]
        _EPS = 1e-12

        # Accumulate in sqrt-space for Fréchet mean
        dist_sums_sqrt: dict[int, NDArray[np.float64]] = {}
        dist_counts: dict[int, int] = {}

        for entry in raw["results"]:
            tokens = entry.get("tokens", [])
            logits = entry.get("logits", [])

            if not tokens or not logits:
                continue

            for pos in range(len(tokens) - 1):
                token_id = tokens[pos]
                output_dist = np.array(logits[pos], dtype=np.float64)

                # Ensure on simplex
                output_dist = np.maximum(output_dist, _EPS)
                output_dist = output_dist / output_dist.sum()

                # Accumulate in sqrt-space
                sqrt_dist = np.sqrt(output_dist)

                if token_id not in dist_sums_sqrt:
                    dist_sums_sqrt[token_id] = sqrt_dist.copy()
                    dist_counts[token_id] = 1
                else:
                    dist_sums_sqrt[token_id] += sqrt_dist
                    dist_counts[token_id] += 1

        # Compute Fréchet means
        fingerprints: dict[int, NDArray[np.float64]] = {}
        for token_id, sqrt_sum in dist_sums_sqrt.items():
            count = dist_counts[token_id]
            if count < min_contexts:
                continue

            mean_sqrt = sqrt_sum / count
            mean_dist = mean_sqrt * mean_sqrt
            mean_dist = mean_dist / mean_dist.sum()
            fingerprints[token_id] = mean_dist

        logger.info(
            f"Computed {len(fingerprints)} token fingerprints from {len(raw['results'])} texts"
        )

        return {
            "fingerprints": fingerprints,
            "context_counts": {tid: dist_counts[tid] for tid in fingerprints},
            "vocab_size": vocab_size,
        }

    async def _harvest_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> dict[str, Any] | None:
        """Send a single batch with retry."""
        import httpx

        last_error = ""
        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.timeout,
                ) as client:
                    resp = await client.post(
                        self.config.harvest_url,
                        headers=self._auth_headers(),
                        json={
                            "texts": texts,
                            "max_length": max_length,
                        },
                    )
                    resp.raise_for_status()
                    result: dict[str, Any] = resp.json()
                    return result

            except httpx.TimeoutException:
                last_error = f"timeout after {self.config.timeout}s"
                logger.warning(
                    "Modal timeout (attempt %d/%d, url=%s)",
                    attempt + 1,
                    self.config.max_retries,
                    self.config.harvest_url,
                )
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                body_preview = e.response.text[:200] if e.response.text else ""
                logger.warning(
                    "Modal HTTP %d (attempt %d/%d, url=%s): %s",
                    e.response.status_code,
                    attempt + 1,
                    self.config.max_retries,
                    self.config.harvest_url,
                    body_preview,
                )
                # Don't retry auth errors — they won't self-heal
                if e.response.status_code in (401, 403):
                    break
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Modal request failed: %s (attempt %d/%d)",
                    e,
                    attempt + 1,
                    self.config.max_retries,
                )

        logger.error(
            "Modal harvest failed after %d attempts: %s (url=%s, texts=%d)",
            self.config.max_retries,
            last_error,
            self.config.harvest_url,
            len(texts),
        )
        return None

    def _auth_headers(self) -> dict[str, str]:
        """Build request headers for the Modal harvest endpoint.

        Auth uses X-Api-Key header checked against KERNEL_API_KEY on the
        Modal side. This replaces the old requires_proxy_auth=True which
        blocked external callers (Railway) with 401.
        """
        headers = {"Content-Type": "application/json"}
        if self.config.harvest_auth_token:
            headers["X-Api-Key"] = self.config.harvest_auth_token
        return headers


# ─── Fallback: Synthetic Harvest ─────────────────────────────────────


def generate_synthetic_harvest(
    vocab_size: int = 32000,
    n_resonances: int = 5000,
    basin_dim: int = 64,
) -> dict[str, Any]:
    """Generate synthetic harvest data for development/testing.

    Used when Modal is unavailable. Produces random distributions
    that pass validation but don't carry real semantic structure.
    """
    logger.warning(
        "Using SYNTHETIC harvest data — no real semantic structure. "
        "Connect Modal for real harvesting."
    )

    rng = np.random.default_rng(42)
    fingerprints: dict[int, NDArray[np.float64]] = {}
    context_counts: dict[int, int] = {}

    for i in range(n_resonances):
        # Dirichlet distribution → point on Δ^(V-1)
        # Use varying concentration to simulate different token types
        alpha = rng.uniform(0.1, 2.0)
        fp = rng.dirichlet(np.ones(vocab_size) * alpha)
        fingerprints[i] = fp.astype(np.float64)
        context_counts[i] = int(rng.integers(10, 500))

    return {
        "fingerprints": fingerprints,
        "context_counts": context_counts,
        "vocab_size": vocab_size,
    }


def generate_synthetic_harvest_result(
    vocab_size: int = 32000,
    n_resonances: int = 500,
) -> HarvestResult:
    """Generate a synthetic HarvestResult for fallback when Modal is unavailable.

    Returns a HarvestResult with Dirichlet-sampled V-dimensional fingerprints
    on Δ^(V-1). These pass geometric validation but carry no real semantic
    structure. Used by CoordizerV2.from_modal_harvest() fallback path.
    """
    logger.warning(
        "Using SYNTHETIC HarvestResult — no real semantic structure. "
        "Connect Modal for real GPU harvesting."
    )

    rng = np.random.default_rng(42)
    result = HarvestResult(
        model_name="synthetic",
        vocab_size=vocab_size,
        corpus_size=0,
        harvest_time_seconds=0.0,
    )

    for i in range(n_resonances):
        alpha = rng.uniform(0.1, 2.0)
        fp = rng.dirichlet(np.ones(vocab_size) * alpha).astype(np.float64)
        result.resonance_fingerprints[i] = fp
        result.context_counts[i] = int(rng.integers(10, 500))
        result.basin_strings[i] = f"<synthetic_{i}>"

    return result
