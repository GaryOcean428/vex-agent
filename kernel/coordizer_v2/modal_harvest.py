"""
Modal Harvest Client — Railway-side integration for GPU harvesting.

This module calls the Modal serverless endpoint to run LLM forward
passes on GPU, capturing full probability distributions for the
CoordizerV2 pipeline.

Critical design choice: we request FULL logit distributions, not
top-k approximations. The tail of the distribution carries geometric
information — a token that is specifically suppressed in a context
looks different from one that is uniformly unlikely. Top-k throws
this away.

Flow:
    Railway → HTTP POST → Modal GPU endpoint → full logits
    → Railway transforms logits via logits_to_simplex (linear projection)
    → basin coordinates via compress.py

Auth: X-Api-Key header (KERNEL_API_KEY) validated by the Modal handler.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from ..config.settings import settings
from .geometry import _EPS
from .harvest import HarvestResult

logger = logging.getLogger(__name__)


@dataclass
class ModalHarvestConfig:
    """Configuration for Modal-based harvesting.

    model_id must match the active inference model so resonance bank
    fingerprints use the same vocabulary as the model doing inference.
    Primary: GLM-4.7-Flash (Modal GPU). Ollama fallback: Qwen/Qwen3.5-4B.
    """

    model_id: str = ""
    target_resonances: int = 2000
    batch_size: int = 32
    max_length: int = 512
    min_contexts: int = 10
    timeout: float = 1200.0


async def modal_harvest(
    model_id: str | None = None,
    target_resonances: int = 2000,
    corpus_texts: list[str] | None = None,
    timeout: float | None = None,
    min_contexts: int | None = None,
) -> HarvestResult:
    """Call Modal GPU endpoint to harvest LLM distributions.

    Returns a HarvestResult with full-distribution fingerprints
    ready for compression via compress.py.
    """
    if not settings.modal.enabled:
        raise RuntimeError(
            "Modal not enabled. Set MODAL_ENABLED=true and configure "
            "MODAL_HARVEST_URL, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET."
        )

    if not settings.modal.harvest_url:
        raise RuntimeError("MODAL_HARVEST_URL not configured.")

    # Resolve from Railway env vars if not explicitly passed
    resolved_model = model_id or settings.modal.harvest_model
    if not resolved_model:
        raise RuntimeError(
            "No Modal harvest model configured. Provide model_id to modal_harvest() "
            "or set MODAL_HARVEST_MODEL in the environment/settings."
        )
    # Harvest timeout must be much longer than inference — cold start + model load + processing.
    # Default: ModalHarvestConfig.timeout (600s), NOT inference_timeout_ms (120s).
    resolved_timeout = timeout if timeout is not None else ModalHarvestConfig.timeout
    resolved_min_contexts = (
        min_contexts if min_contexts is not None else ModalHarvestConfig.min_contexts
    )
    config = ModalHarvestConfig(
        model_id=resolved_model,
        target_resonances=target_resonances,
        timeout=resolved_timeout,
        min_contexts=resolved_min_contexts,
    )

    # Default corpus: diverse prompts for distribution harvesting
    if corpus_texts is None:
        corpus_texts = _default_harvest_corpus()

    # Build request.
    # NOTE: model_id is omitted so Modal uses its default loaded model.
    # MODAL_HARVEST_MODEL env var uses Ollama format ("glm-4.7-flash") but
    # Modal needs HF format ("Qwen/Qwen3.5-35B-A3B"). The Modal endpoint
    # already loads the correct model via HARVEST_MODEL_ID env secret.
    payload: dict[str, Any] = {
        "texts": corpus_texts[:200],  # Cap per-request
        "target_resonances": config.target_resonances,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "min_contexts": config.min_contexts,
        "return_full_distribution": True,  # CRITICAL: not top-k
    }

    # Auth: Modal endpoint checks data.get("_api_key"), not headers
    if settings.kernel_api_key:
        payload["_api_key"] = settings.kernel_api_key

    # Auth via X-Api-Key header (KERNEL_API_KEY), checked by the Modal handler.
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if settings.kernel_api_key:
        headers["X-Api-Key"] = settings.kernel_api_key

    # ASGI pattern: base URL + /harvest path.
    # Handle legacy URLs that already end with /harvest
    base = settings.modal.harvest_url.rstrip("/")
    if base.endswith("/harvest"):
        harvest_endpoint = base
    else:
        harvest_endpoint = base + "/harvest"
    logger.info(
        "Sending harvest request to Modal: %s (%d texts, model=%s)",
        harvest_endpoint,
        len(corpus_texts),
        resolved_model,
    )

    start_time = time.time()
    poll_interval = 10.0  # seconds between polls

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        # Initial POST — Modal may return 200 (fast) or 303 (async polling)
        response = await client.post(
            harvest_endpoint,
            json=payload,
            headers=headers,
        )

        # Modal 303 async pattern: poll the redirect URL until 200
        while response.status_code == 303:
            poll_url = response.headers.get("location", "")
            if not poll_url:
                raise RuntimeError("Modal returned 303 without Location header")
            elapsed_so_far = time.time() - start_time
            if elapsed_so_far > config.timeout:
                raise TimeoutError(
                    f"Modal harvest timed out after {elapsed_so_far:.0f}s "
                    f"(limit {config.timeout:.0f}s)"
                )
            logger.info(
                "Modal processing (%.0fs elapsed), polling in %.0fs...",
                elapsed_so_far,
                poll_interval,
            )
            await asyncio.sleep(poll_interval)
            response = await client.get(poll_url, headers=headers)

        if response.status_code >= 400:
            logger.error(
                "Modal harvest HTTP %d: %s",
                response.status_code,
                response.text[:2000],
            )
        response.raise_for_status()
        data = response.json()

    elapsed = time.time() - start_time
    logger.info("Modal harvest completed in %.1fs", elapsed)

    if not data.get("success"):
        error_msg = data.get("error", "Unknown error from Modal endpoint")
        raise RuntimeError(f"Modal harvest failed: {error_msg}")

    # Parse response into HarvestResult
    result = _parse_modal_response(data, resolved_model, elapsed)

    logger.info(
        "Received %d token fingerprints from Modal (vocab_size=%d)",
        len(result.resonance_fingerprints),
        result.vocab_size,
    )

    return result


def _parse_modal_response(
    data: dict[str, Any],
    model_id: str,
    elapsed: float,
) -> HarvestResult:
    """Parse Modal endpoint response into HarvestResult.

    Expected response format:
    {
        "success": true,
        "vocab_size": 65536,
        "tokens": {
            "42": {
                "string": "hello",
                "fingerprint": [0.001, 0.002, ...],  // full V-dim dist
                "context_count": 15
            },
            ...
        },
        "elapsed_seconds": 45.2
    }
    """
    result = HarvestResult(
        model_name=model_id,
        vocab_size=data.get("vocab_size", 0),
        corpus_size=data.get("total_resonances_processed", 0),
        harvest_time_seconds=elapsed,
    )

    tokens = data.get("tokens", {})
    for key, token_data in tokens.items():
        # Modal harvest keys by token string, with token_id inside the entry
        tid = token_data.get("token_id")
        if tid is None:
            # Fallback: key itself is a numeric ID (legacy format)
            try:
                tid = int(key)
            except (ValueError, TypeError):
                logger.warning("Skipping token with no ID: key=%s", key)
                continue

        fp = np.array(token_data["fingerprint"], dtype=np.float64)
        # Ensure valid probability distribution
        fp = np.maximum(fp, _EPS)
        fp = fp / fp.sum()

        result.resonance_fingerprints[tid] = fp
        result.context_counts[tid] = token_data.get("context_count", token_data.get("count", 1))
        result.basin_strings[tid] = token_data.get("string", key)

    return result


def _default_harvest_corpus() -> list[str]:
    """Diverse corpus for distribution harvesting.

    These prompts are chosen to activate different regions of the
    model's vocabulary space — technical, creative, conversational,
    analytical — so that each coordinate's fingerprint captures its
    contextual diversity.
    """
    return [
        # Technical
        "The quantum field equations predict that spacetime curvature",
        "In Python, the most efficient way to sort a dictionary",
        "The Fisher information matrix measures the curvature of",
        "Thermodynamic equilibrium requires that entropy production",
        "The neural network architecture consists of transformer",
        "In category theory, a functor preserves the structure",
        "The eigenvalues of the Hessian determine the stability",
        "Bayesian inference updates prior beliefs given evidence",
        "The compiler optimizes by performing dead code elimination",
        "Differential geometry studies smooth manifolds equipped with",
        # Conversational
        "I was thinking about what you said yesterday, and",
        "The best part about traveling to new places is",
        "Have you ever noticed how people always seem to",
        "My friend told me the most interesting story about",
        "The weather today reminds me of when we used to",
        "I wonder what it would be like if we could",
        "The restaurant on the corner makes the most amazing",
        "Sometimes I think the hardest part about growing up is",
        "When I was a kid, my grandmother always used to",
        "The thing about living in a big city is that",
        # Creative
        "The old lighthouse stood at the edge of the world",
        "She opened the letter with trembling hands, knowing that",
        "In the garden of forgotten memories, a single flower",
        "The astronaut looked out at the vast expanse and felt",
        "Music filled the empty room like water filling a glass",
        "The detective examined the evidence, noting that the blood",
        "Under the ancient tree, two strangers met and discovered",
        "The painting depicted a scene that nobody could explain",
        "As the train pulled away from the station, she realized",
        "The robot paused, considering what it meant to feel",
        # Analytical
        "The economic implications of this policy change suggest that",
        "Comparing the two approaches, we find that the second",
        "The data clearly shows a correlation between income and",
        "From a philosophical perspective, the question of consciousness",
        "The historical evidence suggests that ancient civilizations",
        "Statistical analysis reveals that the sample size was",
        "The ethical considerations surrounding artificial intelligence",
        "Market forces will eventually correct the imbalance between",
        "The scientific consensus on climate change indicates that",
        "Legal precedent established in the landmark case of",
        # Mixed domain
        "The recipe calls for two cups of flour and a",
        "In the third quarter, revenue increased by approximately",
        "The patient presented with symptoms consistent with",
        "The architectural design incorporates sustainable materials",
        "The championship game came down to the final seconds",
        "Teaching children to read requires patience and a",
        "The election results surprised analysts who had predicted",
        "The volcanic eruption created a new island in the",
        "Debugging the memory leak required examining every",
        "The symphony's second movement transitions from allegro to",
    ]
