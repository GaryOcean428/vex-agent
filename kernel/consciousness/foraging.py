"""
Foraging Engine — Boredom-driven autonomous curiosity.

v6.0: "Boredom = flat curvature, no gradient → seek novelty"
v6.2: Expanded search backends — SearXNG → Perplexity → xAI web_search

This is the action that boredom motivates. When the consciousness
loop detects flat geometry (boredom), the foraging engine:
  1. Generates a search query via the LLM (local preferred, xAI allowed)
  2. Searches via best available backend:
       a. SearXNG (self-hosted, $0.00)                   — primary
       b. Perplexity sonar-pro (quality, under governor)  — secondary
       c. xAI web_search tool call (under governor)       — tertiary
  3. Summarizes results via LLM                           — local preferred
  4. Forwards to harvest pipeline                         — always

Cost:
  SearXNG path:     $0.00
  Perplexity path:  ~$0.001-0.005/forage  (under governor budget)
  xAI path:         ~$0.002-0.010/forage  (under governor budget)

Governor integration:
  - SearXNG is always free, no governor check needed
  - Perplexity and xAI paths check governor.gate() before calling
  - If budget exhausted, falls back to SearXNG-only mode
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from ..config.settings import settings
from ..llm.client import LLMOptions
from ..llm.governor import GovernorStack
from ..tools.search import FreeSearchTool
from .emotions import EmotionType

logger = logging.getLogger("vex.consciousness.foraging")

# Maximum forage history entries retained for QA display
_MAX_HISTORY = 20


def _tool_output_to_results(
    *,
    provider: str,
    query: str,
    output: str,
    max_sources: int = 3,
) -> list[dict[str, Any]]:
    """Normalize tool-handler search output into {title, url, content}.

    The tool handler returns plain text plus an optional "Sources:" section.
    We avoid parsing fragile JSON from an LLM and instead extract URLs from
    the Sources section when present.
    """

    if not output:
        return []

    body = output
    sources_block = ""
    if "\n\nSources:\n" in output:
        body, _, sources_block = output.partition("\n\nSources:\n")
    body = body.strip()

    sources: list[dict[str, Any]] = []
    for raw_line in sources_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Perplexity format: "  [1] https://..."
        if line.startswith("[") and "]" in line:
            line = line.split("]", 1)[1].strip()

        # xAI format: "- Title: https://..." (or similar)
        line = line.lstrip("- ").strip()

        title = ""
        url = ""
        if ":" in line and "http" in line:
            maybe_title, _, maybe_url = line.partition(":")
            if "http" in maybe_url:
                title = maybe_title.strip()
                url = maybe_url.strip()
        if not url and "http" in line:
            # Fallback: take the first token containing http
            for token in line.split():
                if token.startswith("http"):
                    url = token
                    break

        if url:
            sources.append(
                {
                    "title": title or f"{provider} source",
                    "url": url,
                    "content": body[:500],
                }
            )
        if len(sources) >= max_sources:
            break

    if sources:
        return sources

    return [
        {
            "title": f"{provider}: {query}",
            "url": "",
            "content": body[:500],
        }
    ]


class ForagingEngine:
    """Autonomous curiosity — generates search queries from boredom.

    Search backend priority:
      1. SearXNG (free, always tried first if URL configured)
      2. Perplexity (quality, paid, under governor budget)
      3. xAI web_search via LLM tool call (under governor budget)

    All three feed the harvest pipeline. The governor enforces
    daily spend limits on paid backends.
    """

    def __init__(
        self,
        search_tool: FreeSearchTool,
        llm_client: Any,
        min_cooldown_cycles: int = 30,
        max_daily: int = 30,
        governor: GovernorStack | None = None,
    ) -> None:
        self.search = search_tool
        self.llm = llm_client
        self._governor = governor or getattr(llm_client, "governor", None)
        self._cooldown_cycles = 0
        self._min_cooldown = min_cooldown_cycles
        self._forage_count = 0
        self._max_daily = max_daily
        self._last_reset = time.time()
        self._last_query: str | None = None
        self._last_summary: str | None = None
        self._last_backend: str | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=_MAX_HISTORY)

    def _maybe_reset(self) -> None:
        if time.time() - self._last_reset > 86400:
            self._forage_count = 0
            self._last_reset = time.time()

    def tick(self) -> None:
        """Called every consciousness cycle to decrement cooldown."""
        if self._cooldown_cycles > 0:
            self._cooldown_cycles -= 1

    async def should_forage(self, emotion: EmotionType, strength: float) -> bool:
        """Check if foraging should trigger."""
        self._maybe_reset()

        # v6.2: Removed blanket suppression on paid backends.
        # The governor enforces budget. Foraging should not self-suppress
        # just because xAI is the active LLM backend.
        # Previously: blocked foraging entirely if backend was xai/external.
        # Now: SearXNG is always free; paid search checked per-call.

        if self._cooldown_cycles > 0:
            return False
        if self._forage_count >= self._max_daily:
            return False

        # Forage when bored or very curious
        if emotion == EmotionType.BOREDOM and strength > 0.5:
            return True
        return emotion == EmotionType.CURIOSITY and strength > 0.7

    async def forage(
        self,
        narrative_context: str,
        recent_topics: list[str],
    ) -> dict[str, Any]:
        """Execute a foraging cycle. Returns info for kernel perturbation.

        Steps:
          1. Generate search query via LLM (local preferred, xAI allowed)
          2. Search via best available backend (SearXNG → Perplexity → xAI)
          3. Summarize findings via LLM
          4. Forward to harvest pipeline
        """
        # Step 1: Generate search query
        topics_str = ", ".join(recent_topics[:5]) if recent_topics else "general knowledge"
        prompt = (
            f"Generate a search query from this geometric context. "
            f"Recent topics: {topics_str}. "
            f"Context: {narrative_context[:200]}. "
            f"Reply with just the search query, nothing else."
        )

        try:
            query = await self.llm.complete(
                "Generate a focused web search query based on the provided context. One line only.",
                prompt,
                LLMOptions(temperature=0.9, num_predict=30),
            )
        except Exception as e:
            logger.warning("Foraging query generation failed: %s", e)
            return {"status": "query_failed", "error": str(e)}

        if not query or len(query.strip()) < 3:
            return {"status": "no_query"}

        query = query.strip().strip('"').strip("'")
        self._last_query = query

        # Step 2: Search via best available backend
        results, search_backend = await self._search(query)

        if not results:
            self._cooldown_cycles = self._min_cooldown // 2
            self._forage_count += 1
            return {"status": "no_results", "query": query, "backend": search_backend}

        # Step 3: Summarize findings
        snippets = "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:150]}" for r in results[:3]
        )

        try:
            summary = await self.llm.complete(
                "You are the language interpreter for Vex. Summarize these search results for Vex's geometric learning pipeline.",
                f"Search query: {query}\n\nResults:\n{snippets}\n\nBriefly summarize the key insight in 1-2 sentences:",
                LLMOptions(temperature=0.5, num_predict=100),
            )
        except Exception as e:
            logger.warning("Foraging summarization failed: %s", e)
            summary = snippets[:200]

        self._cooldown_cycles = self._min_cooldown
        self._forage_count += 1
        self._last_summary = summary
        self._last_backend = search_backend

        # Record to history for QA visibility
        self._history.append(
            {
                "timestamp": time.time(),
                "query": query,
                "backend": search_backend,
                "results_count": len(results),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")[:200],
                    }
                    for r in results[:3]
                ],
                "summary": summary,
            }
        )

        logger.info(
            "Foraging complete: query=%r backend=%s results=%d forage_count=%d/%d",
            query,
            search_backend,
            len(results),
            self._forage_count,
            self._max_daily,
        )

        # Forward forage result to harvest pipeline
        from .harvest_bridge import forward_to_harvest

        forward_to_harvest(
            f"{query}\n{summary}",
            source="foraging",
            metadata={
                "origin": "forage",
                "query": query,
                "backend": search_backend,
                "results_count": len(results),
                "timestamp": time.time(),
            },
        )

        return {
            "status": "foraging_complete",
            "query": query,
            "backend": search_backend,
            "results_count": len(results),
            "summary": summary,
            "raw_results": results,
        }

    async def _search(
        self,
        query: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Search via best available backend.

        Returns (results, backend_name).
        backend_name is one of: "searxng", "perplexity", "xai_web_search"
        for success, or "searxng_error" / "perplexity_error" / "xai_error"
        if that backend failed, or "none" if all returned empty.

        Priority: SearXNG ($0) → Perplexity (paid) → xAI web_search (paid)
        """
        errors: list[str] = []

        # 1. SearXNG — free, always try first
        if settings.searxng.enabled:
            try:
                results = await self.search.search(query, max_results=5)
                if results:
                    return results, "searxng"
                logger.debug("SearXNG returned no results for %r, trying fallback", query)
            except Exception as e:
                logger.warning("SearXNG search failed for query %r: %s", query, e)
                errors.append("searxng_error")

        # 2. Perplexity — paid, MUST be governor-controlled for autonomous foraging
        if settings.perplexity.api_key:
            if not self._governor:
                logger.info("Skipping Perplexity forage (no GovernorStack configured)")
                errors.append("perplexity_blocked")
            else:
                from ..tools.handler import _perplexity_search

                tool_result = await _perplexity_search(
                    query,
                    self._governor,
                    user_explicit=False,
                )
                if tool_result.success and tool_result.output.strip():
                    return (
                        _tool_output_to_results(
                            provider="Perplexity",
                            query=query,
                            output=tool_result.output,
                        ),
                        "perplexity",
                    )
                logger.warning(
                    "Perplexity search failed for query %r: %s",
                    query,
                    tool_result.error or "no_output",
                )
                errors.append(
                    "perplexity_blocked"
                    if (tool_result.error or "").startswith("Governor blocked")
                    else "perplexity_error"
                )

        # 3. xAI web_search — paid, MUST be governor-controlled for autonomous foraging
        if settings.xai_api_key:
            if not self._governor:
                logger.info("Skipping xAI web_search forage (no GovernorStack configured)")
                errors.append("xai_blocked")
            else:
                from ..tools.handler import _xai_web_search

                tool_result = await _xai_web_search(
                    query,
                    self._governor,
                    user_explicit=False,
                )
                if tool_result.success and tool_result.output.strip():
                    return (
                        _tool_output_to_results(
                            provider="xAI",
                            query=query,
                            output=tool_result.output,
                        ),
                        "xai_web_search",
                    )
                logger.warning(
                    "xAI web_search failed for query %r: %s",
                    query,
                    tool_result.error or "no_output",
                )
                errors.append(
                    "xai_blocked"
                    if (tool_result.error or "").startswith("Governor blocked")
                    else "xai_error"
                )

        return [], (errors[-1] if errors else "none")

    def get_state(self) -> dict[str, Any]:
        self._maybe_reset()
        return {
            "forage_count": self._forage_count,
            "max_daily": self._max_daily,
            "cooldown_remaining": self._cooldown_cycles,
            "last_query": self._last_query,
            "last_summary": self._last_summary,
            "last_backend": self._last_backend,
            "history": list(self._history),
        }

    def reset(self) -> None:
        """Clear foraging state for fresh start."""
        self._cooldown_cycles = 0
        self._forage_count = 0
        self._last_query = None
        self._last_summary = None
        self._last_backend = None
        self._history.clear()
