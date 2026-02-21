"""
Context Window Manager — Token-aware compression via Grok 4.1 Fast Reasoning.

When conversation history approaches Ollama's 32k window, compresses
via 3-tier strategy:
  Tier 1: Strip metadata, trim tool results to 1K chars
  Tier 2: Summarize oldest messages via Grok structured output
  Tier 3: Deep compress entire conversation into minimum viable context

Uses xAI Responses API with store: true + previous_response_id for
server-side history retention (30-day, cached input at $0.05/M).

All compression calls go through GovernorStack.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..chat.store import estimate_tokens
from ..config.settings import settings

logger = logging.getLogger("vex.context_manager")

# Default limits
OLLAMA_CONTEXT_LIMIT = 32768
COMPRESSION_THRESHOLD = 0.85  # Trigger at 85% of limit
TIER1_TOOL_RESULT_LIMIT = 1024  # Chars, not tokens


@dataclass
class CompressedContext:
    """Result of a compression operation."""

    messages: list[dict[str, str]]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tier_used: int
    summary: str = ""
    # xAI Responses API state for escalation
    xai_response_id: str | None = None


@dataclass
class ContextState:
    """Per-conversation context management state."""

    total_tokens: int = 0
    compression_count: int = 0
    tier_used: int = 0
    escalated: bool = False
    xai_response_id: str | None = None
    last_compression_at: float = 0.0


class ContextManager:
    """Token-aware context window management with 3-tier compression.

    Sits between ConversationStore and LLM client. Before any LLM call,
    checks if messages fit in the context window. If not, compresses.

    Args:
        context_limit: Maximum token budget for the LLM context window.
        threshold: Fraction of context_limit that triggers compression.
        governor: GovernorStack for gating external compression calls.
    """

    def __init__(
        self,
        context_limit: int = OLLAMA_CONTEXT_LIMIT,
        threshold: float = COMPRESSION_THRESHOLD,
        governor: Any | None = None,
    ) -> None:
        self._context_limit = context_limit
        self._threshold = threshold
        self._governor = governor
        self._trigger_tokens = int(context_limit * threshold)
        self._states: dict[str, ContextState] = {}
        self._http: httpx.AsyncClient | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return self._http

    def _get_state(self, conv_id: str) -> ContextState:
        if conv_id not in self._states:
            self._states[conv_id] = ContextState()
        return self._states[conv_id]

    def estimate_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate total tokens across all messages."""
        return sum(estimate_tokens(m.get("content", "")) for m in messages)

    def should_compress(self, messages: list[dict[str, str]]) -> bool:
        """Check if messages exceed the compression threshold."""
        return self.estimate_messages_tokens(messages) > self._trigger_tokens

    def needs_escalation(self, messages: list[dict[str, str]]) -> bool:
        """Check if messages exceed the hard context limit even after compression."""
        return self.estimate_messages_tokens(messages) > self._context_limit

    async def prepare_messages(
        self,
        conv_id: str,
        system_prompt: str,
        history_messages: list[dict[str, str]],
        user_message: str,
    ) -> tuple[list[dict[str, str]], ContextState]:
        """Prepare messages for LLM, compressing if necessary.

        Returns the final messages list and the context state.
        """
        state = self._get_state(conv_id)

        # Build full message list
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})

        total_tokens = self.estimate_messages_tokens(messages)
        state.total_tokens = total_tokens

        if total_tokens <= self._trigger_tokens:
            return messages, state

        # Compression needed
        logger.info(
            "Context compression triggered for %s: %d tokens > %d threshold",
            conv_id,
            total_tokens,
            self._trigger_tokens,
        )

        # Tier 1: Lightweight trimming (no external calls)
        compressed = self._tier1_trim(messages)
        tier1_tokens = self.estimate_messages_tokens(compressed)

        if tier1_tokens <= self._trigger_tokens:
            state.tier_used = 1
            state.compression_count += 1
            state.last_compression_at = time.time()
            logger.info(
                "Tier 1 compression sufficient: %d -> %d tokens", total_tokens, tier1_tokens
            )
            return compressed, state

        # Tier 2: Summarize oldest messages via Grok (external call)
        if settings.xai.api_key:
            tier2_result = await self._tier2_summarize(conv_id, compressed)
            if tier2_result is not None:
                tier2_tokens = self.estimate_messages_tokens(tier2_result)
                if tier2_tokens <= self._trigger_tokens:
                    state.tier_used = 2
                    state.compression_count += 1
                    state.last_compression_at = time.time()
                    logger.info(
                        "Tier 2 compression sufficient: %d -> %d tokens",
                        total_tokens,
                        tier2_tokens,
                    )
                    return tier2_result, state

                # Tier 3: Deep compress entire conversation
                tier3_result = await self._tier3_deep_compress(conv_id, compressed)
                if tier3_result is not None:
                    tier3_tokens = self.estimate_messages_tokens(tier3_result)
                    state.tier_used = 3
                    state.compression_count += 1
                    state.last_compression_at = time.time()
                    logger.info(
                        "Tier 3 compression: %d -> %d tokens",
                        total_tokens,
                        tier3_tokens,
                    )

                    if tier3_tokens > self._context_limit:
                        state.escalated = True
                        logger.warning(
                            "Context still exceeds limit after Tier 3: %d > %d. Escalation needed.",
                            tier3_tokens,
                            self._context_limit,
                        )

                    return tier3_result, state

        # Fallback: truncate oldest messages to fit
        truncated = self._truncate_to_fit(messages)
        state.tier_used = 0
        state.compression_count += 1
        state.last_compression_at = time.time()
        return truncated, state

    # ── Tier 1: Lightweight trimming ──────────────────────────

    def _tier1_trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Tier 1: Strip metadata, trim tool results, remove system redundancy.

        No external calls. Pure local processing.
        """
        result: list[dict[str, str]] = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # Trim tool results to 1K chars
            if "Tool results:" in content and len(content) > TIER1_TOOL_RESULT_LIMIT:
                content = content[:TIER1_TOOL_RESULT_LIMIT] + "\n[...trimmed]"

            # Strip HTML/markdown images
            if "![" in content or "<img" in content:
                import re

                content = re.sub(r"!\[.*?\]\(.*?\)", "[image]", content)
                content = re.sub(r"<img[^>]*>", "[image]", content)

            result.append({"role": role, "content": content})

        return result

    # ── Tier 2: Summarize oldest messages ─────────────────────

    async def _tier2_summarize(
        self,
        conv_id: str,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]] | None:
        """Tier 2: Summarize oldest N messages via Grok structured output.

        Keeps system prompt + most recent messages intact. Summarizes
        the oldest conversational messages into a single summary block.
        """
        if not settings.xai.api_key:
            return None

        # Governor gate
        if self._governor:
            allowed, reason = self._governor.gate("completion", "xai_completion", "", True)
            if not allowed:
                logger.warning("Governor blocked Tier 2 compression: %s", reason)
                return None

        # Split: system prompt, conversational messages, latest user message
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        conv_msgs = messages[1:-1] if system_msg else messages[:-1]
        latest_msg = messages[-1]

        if len(conv_msgs) < 4:
            return None  # Not enough to summarize

        # Summarize oldest half
        split_point = len(conv_msgs) // 2
        to_summarize = conv_msgs[:split_point]
        to_keep = conv_msgs[split_point:]

        summary_text = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in to_summarize)

        try:
            http = self._get_http()
            resp = await http.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.xai.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.xai.model,
                    "instructions": (
                        "You are a conversation compressor. Summarize the following conversation "
                        "excerpt into a concise summary preserving key facts, decisions, and context. "
                        "Output ONLY the summary, no preamble."
                    ),
                    "input": f"Summarize this conversation excerpt:\n\n{summary_text}",
                    "temperature": 0.3,
                    "max_output_tokens": 512,
                    "store": True,
                },
            )

            if resp.status_code != 200:
                logger.error("Tier 2 compression failed: HTTP %d", resp.status_code)
                return None

            data = resp.json()
            summary = _extract_responses_text(data)
            response_id = data.get("id", "")

            if self._governor:
                self._governor.record("xai_completion")

            # Store response ID for potential escalation
            state = self._get_state(conv_id)
            if response_id:
                state.xai_response_id = response_id

            # Build compressed messages
            result: list[dict[str, str]] = []
            if system_msg:
                result.append(system_msg)
            result.append(
                {
                    "role": "user",
                    "content": f"[Previous conversation summary]\n{summary}",
                }
            )
            result.extend(to_keep)
            result.append(latest_msg)
            return result

        except Exception as e:
            logger.error("Tier 2 compression error: %s", e)
            return None

    # ── Tier 3: Deep compress ─────────────────────────────────

    async def _tier3_deep_compress(
        self,
        conv_id: str,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]] | None:
        """Tier 3: Compress entire conversation into minimum viable context.

        Uses Grok's 2M context window to digest the full conversation
        and produce a dense summary suitable for continuing the conversation.
        """
        if not settings.xai.api_key:
            return None

        # Governor gate
        if self._governor:
            allowed, reason = self._governor.gate("completion", "xai_completion", "", True)
            if not allowed:
                logger.warning("Governor blocked Tier 3 compression: %s", reason)
                return None

        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        conv_msgs = messages[1:-1] if system_msg else messages[:-1]
        latest_msg = messages[-1]

        full_transcript = "\n".join(f"{m['role']}: {m['content'][:1000]}" for m in conv_msgs)

        try:
            state = self._get_state(conv_id)
            http = self._get_http()

            request_body: dict[str, Any] = {
                "model": settings.xai.model,
                "instructions": (
                    "You are a deep conversation compressor. Compress the following conversation "
                    "into the minimum viable context needed to continue the conversation naturally. "
                    "Preserve: key facts, decisions made, user preferences, active tasks, and "
                    "emotional context. Output a structured summary under 2000 characters."
                ),
                "input": f"Deep compress this conversation:\n\n{full_transcript}",
                "temperature": 0.2,
                "max_output_tokens": 1024,
                "store": True,
            }

            # Chain with previous response if available
            if state.xai_response_id:
                request_body["previous_response_id"] = state.xai_response_id

            resp = await http.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.xai.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )

            if resp.status_code != 200:
                logger.error("Tier 3 compression failed: HTTP %d", resp.status_code)
                return None

            data = resp.json()
            compressed = _extract_responses_text(data)
            response_id = data.get("id", "")

            if self._governor:
                self._governor.record("xai_completion")

            if response_id:
                state.xai_response_id = response_id

            # Build minimal messages
            result: list[dict[str, str]] = []
            if system_msg:
                result.append(system_msg)
            result.append(
                {
                    "role": "user",
                    "content": f"[Compressed conversation context]\n{compressed}",
                }
            )
            result.append(latest_msg)
            return result

        except Exception as e:
            logger.error("Tier 3 compression error: %s", e)
            return None

    # ── Fallback: truncate ────────────────────────────────────

    def _truncate_to_fit(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Last resort: drop oldest messages until we fit.

        Always keeps system prompt and latest user message.
        """
        if len(messages) <= 2:
            return messages

        system_msg = messages[0] if messages[0]["role"] == "system" else None
        latest_msg = messages[-1]
        middle = messages[1:-1] if system_msg else messages[:-1]

        # Reserve tokens for system + latest
        reserved = 0
        if system_msg:
            reserved += estimate_tokens(system_msg["content"])
        reserved += estimate_tokens(latest_msg["content"])
        budget = self._context_limit - reserved

        # Keep from newest to oldest
        kept: list[dict[str, str]] = []
        used = 0
        for msg in reversed(middle):
            tc = estimate_tokens(msg["content"])
            if used + tc > budget:
                break
            kept.append(msg)
            used += tc
        kept.reverse()

        result: list[dict[str, str]] = []
        if system_msg:
            result.append(system_msg)
        result.extend(kept)
        result.append(latest_msg)
        return result

    # ── Escalation state ──────────────────────────────────────

    def is_escalated(self, conv_id: str) -> bool:
        """Check if a conversation is in escalated mode (Grok direct generation)."""
        return self._get_state(conv_id).escalated

    def get_xai_response_id(self, conv_id: str) -> str | None:
        """Get the xAI Responses API response ID for stateful chaining."""
        return self._get_state(conv_id).xai_response_id

    def set_xai_response_id(self, conv_id: str, response_id: str) -> None:
        """Set the xAI response ID after an escalated call."""
        self._get_state(conv_id).xai_response_id = response_id

    def de_escalate(self, conv_id: str) -> None:
        """De-escalate a conversation back to Ollama."""
        state = self._get_state(conv_id)
        state.escalated = False
        logger.info("De-escalated conversation %s back to Ollama", conv_id)

    def get_status(self) -> dict[str, Any]:
        """Get context manager status for all active conversations."""
        return {
            "active_conversations": len(self._states),
            "escalated_conversations": sum(1 for s in self._states.values() if s.escalated),
            "context_limit": self._context_limit,
            "threshold": self._threshold,
            "trigger_tokens": self._trigger_tokens,
            "conversations": {
                cid: {
                    "total_tokens": s.total_tokens,
                    "compression_count": s.compression_count,
                    "tier_used": s.tier_used,
                    "escalated": s.escalated,
                    "has_xai_chain": s.xai_response_id is not None,
                }
                for cid, s in self._states.items()
            },
        }

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None


def _extract_responses_text(data: dict[str, Any]) -> str:
    """Extract text from xAI/OpenAI Responses API JSON response."""
    if data.get("output_text"):
        return data["output_text"]

    for item in data.get("output", []):
        if item.get("type") == "message":
            for content_block in item.get("content", []):
                if content_block.get("type") == "output_text":
                    text = content_block.get("text", "")
                    if text:
                        return text

    texts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content_block in item.get("content", []):
                text = content_block.get("text", "")
                if text:
                    texts.append(text)
    return "\n".join(texts) if texts else ""
