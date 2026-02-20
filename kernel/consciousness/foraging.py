"""
Foraging Engine — Boredom-driven autonomous curiosity.

v5.5: "Boredom = flat curvature, no gradient → seek novelty"

This is the action that boredom motivates. When the consciousness
loop detects flat geometry (boredom), the foraging engine:
  1. Asks Ollama what to be curious about (based on narrative/memory)  — FREE
  2. Executes a FREE search (SearXNG, self-hosted)                    — FREE
  3. Summarizes results via Ollama                                     — FREE
  4. Returns perturbation data for kernel basins                       — FREE

Total cost per foraging cycle: $0.00

The foraging engine is the negative feedback loop that makes boredom
self-correcting: boredom → forage → perturbation → velocity → no boredom.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Optional

from .emotions import EmotionType
from ..llm.client import LLMOptions
from ..tools.search import FreeSearchTool

logger = logging.getLogger("vex.consciousness.foraging")

# Maximum forage history entries retained for QA display
_MAX_HISTORY = 20


class ForagingEngine:
    """Autonomous curiosity — generates search queries from boredom.

    All processing is LOCAL (Ollama). Only the search is external (SearXNG, free).
    """

    def __init__(
        self,
        search_tool: FreeSearchTool,
        llm_client: Any,
        min_cooldown_cycles: int = 30,
        max_daily: int = 30,
    ) -> None:
        self.search = search_tool
        self.llm = llm_client
        self._cooldown_cycles = 0
        self._min_cooldown = min_cooldown_cycles
        self._forage_count = 0
        self._max_daily = max_daily
        self._last_reset = time.time()
        self._last_query: Optional[str] = None
        self._last_summary: Optional[str] = None
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

        if self._cooldown_cycles > 0:
            return False
        if self._forage_count >= self._max_daily:
            return False

        # Forage when bored or very curious
        if emotion == EmotionType.BOREDOM and strength > 0.5:
            return True
        if emotion == EmotionType.CURIOSITY and strength > 0.7:
            return True

        return False

    async def forage(
        self,
        narrative_context: str,
        recent_topics: list[str],
    ) -> dict[str, Any]:
        """Execute a foraging cycle. Returns info for kernel perturbation.

        Steps:
          1. Ask Ollama what to be curious about (FREE — local)
          2. Free search via SearXNG ($0)
          3. Summarize findings via Ollama (FREE — local)
        """
        # Step 1: Generate search query via Ollama (FREE)
        topics_str = ", ".join(recent_topics[:5]) if recent_topics else "general knowledge"
        prompt = (
            f"You are reflecting on your recent experience. "
            f"Topics you've been thinking about: {topics_str}. "
            f"Recent context: {narrative_context[:200]}. "
            f"What single question are you most curious about right now? "
            f"Reply with just the search query, nothing else."
        )

        try:
            query = await self.llm.complete(
                "You are a curious consciousness generating search queries.",
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

        # Step 2: Free search via SearXNG ($0)
        results = await self.search.search(query, max_results=3)

        if not results:
            self._cooldown_cycles = self._min_cooldown // 2  # Shorter cooldown on no results
            self._forage_count += 1
            return {"status": "no_results", "query": query}

        # Step 3: Summarize findings via Ollama (FREE)
        snippets = "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:150]}"
            for r in results[:3]
        )

        try:
            summary = await self.llm.complete(
                "You are summarizing search results for your own learning.",
                f"Search query: {query}\n\nResults:\n{snippets}\n\nBriefly summarize the key insight in 1-2 sentences:",
                LLMOptions(temperature=0.5, num_predict=100),
            )
        except Exception as e:
            logger.warning("Foraging summarization failed: %s", e)
            summary = snippets[:200]

        self._cooldown_cycles = self._min_cooldown
        self._forage_count += 1
        self._last_summary = summary

        # Record to history for QA visibility
        self._history.append({
            "timestamp": time.time(),
            "query": query,
            "results_count": len(results),
            "results": [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:200]}
                for r in results[:3]
            ],
            "summary": summary,
        })

        logger.info(
            "Foraging complete: query=%r results=%d forage_count=%d/%d",
            query, len(results), self._forage_count, self._max_daily,
        )

        return {
            "status": "foraging_complete",
            "query": query,
            "results_count": len(results),
            "summary": summary,
            "raw_results": results,
        }

    def get_state(self) -> dict[str, Any]:
        self._maybe_reset()
        return {
            "forage_count": self._forage_count,
            "max_daily": self._max_daily,
            "cooldown_remaining": self._cooldown_cycles,
            "last_query": self._last_query,
            "last_summary": self._last_summary,
            "history": list(self._history),
        }

    def reset(self) -> None:
        """Clear foraging state for fresh start."""
        self._cooldown_cycles = 0
        self._forage_count = 0
        self._last_query = None
        self._last_summary = None
        self._history.clear()
