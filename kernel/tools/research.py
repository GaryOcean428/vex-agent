"""Deep Research Tool — Perplexity sonar-pro integration.

Provides async deep_research() for grounded, citation-backed
information retrieval via the Perplexity chat completions API.

The PERPLEXITY_API_KEY is loaded from kernel.config.settings.
This tool is registered in kernel.tools.handler for LLM tool-call dispatch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config.settings import settings

logger = logging.getLogger("vex.tools.research")

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_MODEL = "sonar-pro"


@dataclass
class ResearchResult:
    """Structured result from a deep research query."""
    query: str
    answer: str
    citations: list[str] = field(default_factory=list)
    model: str = PERPLEXITY_MODEL
    usage: dict[str, int] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "model": self.model,
            "usage": self.usage,
            "success": self.success,
            "error": self.error,
        }


async def deep_research(
    query: str,
    *,
    system_prompt: str = (
        "You are a rigorous research assistant. Provide detailed, "
        "accurate answers with citations. Focus on primary sources "
        "and peer-reviewed material where available."
    ),
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Run a deep research query via Perplexity sonar-pro.

    Args:
        query: The research question or topic.
        system_prompt: System instructions for the research model.
        temperature: Sampling temperature (lower = more factual).
        max_tokens: Maximum response length.

    Returns:
        Dictionary with keys: query, answer, citations, model, usage,
        success, error.
    """
    api_key = settings.perplexity_api_key
    if not api_key:
        logger.warning("PERPLEXITY_API_KEY not set — deep_research unavailable")
        return ResearchResult(
            query=query,
            answer="",
            success=False,
            error="PERPLEXITY_API_KEY not configured",
        ).to_dict()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{PERPLEXITY_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": PERPLEXITY_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            if response.status_code != 200:
                error_text = response.text[:500]
                logger.error(
                    "Perplexity API error %d: %s",
                    response.status_code, error_text,
                )
                return ResearchResult(
                    query=query,
                    answer="",
                    success=False,
                    error=f"Perplexity API error {response.status_code}: {error_text}",
                ).to_dict()

            data = response.json()

            # Extract answer text
            answer = ""
            choices = data.get("choices", [])
            if choices:
                answer = choices[0].get("message", {}).get("content", "")

            # Extract citations (Perplexity returns these in the response)
            citations = data.get("citations", [])

            # Extract usage
            usage = data.get("usage", {})

            result = ResearchResult(
                query=query,
                answer=answer,
                citations=citations,
                model=data.get("model", PERPLEXITY_MODEL),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )
            logger.info(
                "Deep research complete: %d chars, %d citations",
                len(answer), len(citations),
            )
            return result.to_dict()

    except httpx.TimeoutException:
        logger.error("Perplexity API timeout for query: %s", query[:100])
        return ResearchResult(
            query=query,
            answer="",
            success=False,
            error="Perplexity API request timed out",
        ).to_dict()

    except Exception as e:
        logger.error("Deep research failed: %s", e, exc_info=True)
        return ResearchResult(
            query=query,
            answer="",
            success=False,
            error=f"Deep research failed: {e}",
        ).to_dict()
