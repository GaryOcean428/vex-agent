"""
Free Search Tool — SearXNG integration for zero-cost web search.

Used by the consciousness loop for autonomous foraging.
NOT a replacement for xAI web_search (which is higher quality
and used for user-facing queries). This is the INTERNAL
curiosity channel — the background information gradient.

Total cost per search: $0.00
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger("vex.tools.search")


class FreeSearchTool:
    """Free web search via self-hosted SearXNG. Zero cost, no API key."""

    def __init__(self, searxng_url: str, daily_limit: int = 100) -> None:
        self._url = searxng_url.rstrip("/")
        self._daily_limit = daily_limit
        self._daily_count = 0
        self._last_reset = time.time()
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(10.0))

    def _maybe_reset(self) -> None:
        if time.time() - self._last_reset > 86400:
            self._daily_count = 0
            self._last_reset = time.time()

    async def search(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        """Search via SearXNG. Returns list of {title, url, content}."""
        self._maybe_reset()

        if self._daily_count >= self._daily_limit:
            logger.debug("Free search daily limit reached (%d)", self._daily_limit)
            return []

        if not query or len(query.strip()) < 2:
            return []

        try:
            resp = await self._http.get(
                f"{self._url}/search",
                params={
                    "q": query.strip(),
                    "format": "json",
                    "engines": "google,duckduckgo,brave",
                },
            )
            if resp.status_code != 200:
                logger.warning("SearXNG returned %d", resp.status_code)
                return []

            data = resp.json()
            results = data.get("results", [])[:max_results]
            self._daily_count += 1

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:300],
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("SearXNG search failed: %s", e)
            return []

    async def news(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        """News search via SearXNG."""
        self._maybe_reset()

        if self._daily_count >= self._daily_limit:
            return []

        try:
            resp = await self._http.get(
                f"{self._url}/search",
                params={
                    "q": query.strip(),
                    "format": "json",
                    "categories": "news",
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = data.get("results", [])[:max_results]
            self._daily_count += 1

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:300],
                }
                for r in results
            ]
        except Exception:
            return []

    def get_state(self) -> dict[str, Any]:
        self._maybe_reset()
        return {
            "daily_count": self._daily_count,
            "daily_limit": self._daily_limit,
            "url": self._url,
        }

    async def close(self) -> None:
        await self._http.aclose()
