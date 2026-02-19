"""
Modal Integration — Railway-side client for GPU harvesting

Connects the Railway-deployed Vex kernel to the Modal GPU endpoint
for coordizer harvesting. Handles authentication, retry, fallback
to synthetic data, and transformation of raw logits to Δ⁶³ basin
coordinates via CoordizerV2 compression.

Configuration:
    Set these environment variables in Railway:
        MODAL_HARVEST_ENABLED=true
        MODAL_HARVEST_URL=https://<your-modal-app>.modal.run/harvest
        MODAL_TOKEN_ID=wk-...
        MODAL_TOKEN_SECRET=ws-...
        MODAL_HARVEST_TIMEOUT=120
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────

@dataclass
class ModalConfig:
    """Configuration for Modal GPU harvesting."""
    enabled: bool = False
    harvest_url: str = ""
    health_url: str = ""
    token_id: str = ""
    token_secret: str = ""
    timeout: int = 120
    max_retries: int = 3
    batch_size: int = 32

    @classmethod
    def from_env(cls) -> ModalConfig:
        """Load configuration from environment variables."""
        enabled = os.getenv("MODAL_HARVEST_ENABLED", "false").lower() == "true"
        harvest_url = os.getenv("MODAL_HARVEST_URL", "")

        # Derive health URL from harvest URL
        health_url = ""
        if harvest_url:
            health_url = harvest_url.rsplit("/", 1)[0] + "/health"

        return cls(
            enabled=enabled,
            harvest_url=harvest_url,
            health_url=health_url,
            token_id=os.getenv("MODAL_TOKEN_ID", ""),
            token_secret=os.getenv("MODAL_TOKEN_SECRET", ""),
            timeout=int(os.getenv("MODAL_HARVEST_TIMEOUT", "120")),
        )

    def is_configured(self) -> bool:
        """Check if all required fields are present."""
        return bool(
            self.enabled
            and self.harvest_url
            and self.token_id
            and self.token_secret
        )


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

    def __init__(self, config: Optional[ModalConfig] = None):
        self.config = config or ModalConfig.from_env()
        self._healthy: Optional[bool] = None

    async def check_health(self) -> bool:
        """Check if the Modal endpoint is reachable."""
        if not self.config.is_configured():
            logger.warning("Modal not configured — check environment variables")
            return False

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    self.config.health_url,
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                self._healthy = data.get("status") == "ok"
                logger.info(
                    f"Modal health check: {data.get('status')} "
                    f"(model={data.get('model_id')}, vocab={data.get('vocab_size')})"
                )
                return self._healthy

        except Exception as e:
            logger.warning(f"Modal health check failed: {e}")
            self._healthy = False
            return False

    async def harvest(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> Optional[dict]:
        """Send texts to Modal for GPU harvesting.

        Args:
            texts: List of text strings to harvest
            max_length: Max token length per text

        Returns:
            Raw harvest response dict, or None on failure.
            Response format:
                {
                    "success": True,
                    "vocab_size": 32000,
                    "results": [
                        {"tokens": [...], "logits": [[...], ...], "token_strings": [...]},
                        ...
                    ],
                    "elapsed_seconds": 1.23
                }
        """
        if not self.config.is_configured():
            logger.warning("Modal not configured, skipping harvest")
            return None

        import httpx

        all_results = []
        batch_size = self.config.batch_size
        vocab_size = 0
        total_elapsed = 0.0

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            result = await self._harvest_batch(batch, max_length)

            if result is None:
                logger.warning(
                    f"Batch {batch_start // batch_size} failed, "
                    f"skipping {len(batch)} texts"
                )
                continue

            if result.get("success"):
                all_results.extend(result.get("results", []))
                vocab_size = result.get("vocab_size", vocab_size)
                total_elapsed += result.get("elapsed_seconds", 0)

        if not all_results:
            return None

        return {
            "success": True,
            "vocab_size": vocab_size,
            "results": all_results,
            "elapsed_seconds": total_elapsed,
            "n_texts": len(all_results),
        }

    async def harvest_to_fingerprints(
        self,
        texts: list[str],
        min_contexts: int = 10,
        max_length: int = 512,
    ) -> Optional[dict]:
        """Harvest and compute Fréchet mean fingerprints per token.

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
        dist_sums_sqrt: dict[int, NDArray] = {}
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
        fingerprints: dict[int, NDArray] = {}
        for token_id, sqrt_sum in dist_sums_sqrt.items():
            count = dist_counts[token_id]
            if count < min_contexts:
                continue

            mean_sqrt = sqrt_sum / count
            mean_dist = mean_sqrt * mean_sqrt
            mean_dist = mean_dist / mean_dist.sum()
            fingerprints[token_id] = mean_dist

        logger.info(
            f"Computed {len(fingerprints)} token fingerprints "
            f"from {len(raw['results'])} texts"
        )

        return {
            "fingerprints": fingerprints,
            "context_counts": {
                tid: dist_counts[tid]
                for tid in fingerprints
            },
            "vocab_size": vocab_size,
        }

    async def _harvest_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> Optional[dict]:
        """Send a single batch with retry."""
        import httpx

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
                    return resp.json()

            except httpx.TimeoutException:
                logger.warning(
                    f"Modal timeout (attempt {attempt + 1}/{self.config.max_retries})"
                )
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Modal HTTP {e.response.status_code} "
                    f"(attempt {attempt + 1}/{self.config.max_retries})"
                )
            except Exception as e:
                logger.warning(
                    f"Modal request failed: {e} "
                    f"(attempt {attempt + 1}/{self.config.max_retries})"
                )

        return None

    def _auth_headers(self) -> dict[str, str]:
        """Build Modal proxy auth headers."""
        return {
            "Content-Type": "application/json",
            "Modal-Token-Id": self.config.token_id,
            "Modal-Token-Secret": self.config.token_secret,
        }


# ─── Fallback: Synthetic Harvest ─────────────────────────────────────

def generate_synthetic_harvest(
    vocab_size: int = 32000,
    n_tokens: int = 5000,
    basin_dim: int = 64,
) -> dict:
    """Generate synthetic harvest data for development/testing.

    Used when Modal is unavailable. Produces random distributions
    that pass validation but don't carry real semantic structure.
    """
    logger.warning(
        "Using SYNTHETIC harvest data — no real semantic structure. "
        "Connect Modal for real harvesting."
    )

    rng = np.random.default_rng(42)
    fingerprints: dict[int, NDArray] = {}
    context_counts: dict[int, int] = {}

    for i in range(n_tokens):
        # Dirichlet distribution → point on Δ^(V-1)
        # Use varying concentration to simulate different token types
        alpha = rng.uniform(0.1, 2.0)
        fp = rng.dirichlet(np.ones(vocab_size) * alpha)
        fingerprints[i] = fp.astype(np.float64)
        context_counts[i] = rng.integers(10, 500)

    return {
        "fingerprints": fingerprints,
        "context_counts": context_counts,
        "vocab_size": vocab_size,
    }
