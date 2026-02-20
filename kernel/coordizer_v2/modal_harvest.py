"""
Modal Harvest Client — Railway-side integration for GPU harvesting.

This module calls the Modal serverless endpoint to run LLM forward
passes on GPU, capturing full probability distributions for the
CoordizerV2 pipeline.

Critical design choice: we request FULL softmax distributions, not
top-k approximations. The tail of the distribution carries geometric
information — a token that is specifically suppressed in a context
looks different from one that is uniformly unlikely. Top-k throws
this away.

Flow:
    Railway → HTTP POST → Modal GPU endpoint → full logits
    → Railway transforms logits → basin coordinates via compress.py

Auth: Modal Proxy Auth Tokens (wk-*/ws-* headers)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import _EPS
from .harvest import HarvestResult

logger = logging.getLogger(__name__)


@dataclass
class ModalHarvestConfig:
    """Configuration for Modal-based harvesting."""
    model_id: str = "LiquidAI/LFM2.5-1.2B-Thinking"
    target_tokens: int = 2000
    batch_size: int = 32
    max_length: int = 512
    min_contexts: int = 10
    timeout: float = 600.0


async def modal_harvest(
    model_id: str = "LiquidAI/LFM2.5-1.2B-Thinking",
    target_tokens: int = 2000,
    corpus_texts: Optional[list[str]] = None,
    timeout: float = 600.0,
) -> HarvestResult:
    """Call Modal GPU endpoint to harvest LLM distributions.

    Returns a HarvestResult with full-distribution fingerprints
    ready for compression via compress.py.
    """
    import httpx

    from ..config.settings import settings

    if not settings.modal.enabled:
        raise RuntimeError(
            "Modal not enabled. Set MODAL_ENABLED=true and configure "
            "MODAL_HARVEST_URL, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET."
        )

    if not settings.modal.harvest_url:
        raise RuntimeError("MODAL_HARVEST_URL not configured.")

    # Default corpus: diverse prompts for distribution harvesting
    if corpus_texts is None:
        corpus_texts = _default_harvest_corpus()

    # Build request
    payload = {
        "model_id": model_id,
        "texts": corpus_texts[:200],  # Cap per-request
        "batch_size": 32,
        "max_length": 512,
        "return_full_distribution": True,  # CRITICAL: not top-k
    }

    # Modal Proxy Auth headers
    headers = {
        "Content-Type": "application/json",
    }
    if settings.modal.token_id and settings.modal.token_secret:
        headers["Modal-Token-Id"] = settings.modal.token_id
        headers["Modal-Token-Secret"] = settings.modal.token_secret

    logger.info(
        f"Sending harvest request to Modal: {settings.modal.harvest_url} "
        f"({len(corpus_texts)} texts, model={model_id})"
    )

    start_time = time.time()

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            settings.modal.harvest_url,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    elapsed = time.time() - start_time
    logger.info(f"Modal harvest completed in {elapsed:.1f}s")

    if not data.get("success"):
        error_msg = data.get("error", "Unknown error from Modal endpoint")
        raise RuntimeError(f"Modal harvest failed: {error_msg}")

    # Parse response into HarvestResult
    result = _parse_modal_response(data, model_id, elapsed)

    logger.info(
        f"Received {len(result.token_fingerprints)} token fingerprints "
        f"from Modal (vocab_size={result.vocab_size})"
    )

    return result


def _parse_modal_response(
    data: dict,
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
        corpus_size=data.get("total_tokens_processed", 0),
        harvest_time_seconds=elapsed,
    )

    tokens = data.get("tokens", {})
    for tid_str, token_data in tokens.items():
        tid = int(tid_str)

        fp = np.array(token_data["fingerprint"], dtype=np.float64)
        # Ensure valid probability distribution
        fp = np.maximum(fp, _EPS)
        fp = fp / fp.sum()

        result.token_fingerprints[tid] = fp
        result.context_counts[tid] = token_data.get("context_count", 1)
        result.token_strings[tid] = token_data.get("string", f"<token_{tid}>")

    return result


def _default_harvest_corpus() -> list[str]:
    """Diverse corpus for distribution harvesting.

    These prompts are chosen to activate different regions of the
    model's vocabulary space — technical, creative, conversational,
    analytical — so that each token's fingerprint captures its
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
