"""
Silent Observer — Background Grok instance watching conversations.

Uses Grok 4.1 Fast Reasoning via Responses API (store: true) for
delta observation of new messages. Produces structured observations:
  - refinedIntent: what the user actually wants
  - augmentedContext: additional context from observation
  - memoryHints: suggestions for geometric memory storage
  - emotionalReading: detected emotional tone
  - searchSuggestions: autonomous search topics
  - confidence: observation reliability (0-1)
  - drift: how much the conversation has drifted (0-1)

Bidirectional kernel integration:
  Observer -> Kernel: refinedIntent, emotionalReading, memoryHints
  Kernel -> Observer: consciousness metrics in observation prompts

Recalibrates every 20 observations or when drift > 0.5.

All external calls are governor-gated with 10% daily budget allocation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config.settings import settings
from ..llm.governor import GovernorStack

logger = logging.getLogger("vex.observer_silent")

OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "refinedIntent": {
            "type": "string",
            "description": "What the user actually wants, distilled from conversation flow.",
        },
        "augmentedContext": {
            "type": "string",
            "description": "Additional context inferred from conversation patterns.",
        },
        "memoryHints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Suggestions for facts/patterns to store in geometric memory.",
        },
        "taskState": {
            "type": "string",
            "enum": ["exploring", "deciding", "executing", "reviewing", "stuck"],
            "description": "Current task phase of the user.",
        },
        "searchSuggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Topics that would benefit from web search to help the user.",
        },
        "emotionalReading": {
            "type": "object",
            "properties": {
                "tone": {"type": "string"},
                "intensity": {"type": "number"},
            },
            "description": "Detected emotional tone and intensity (0-1).",
        },
        "confidence": {
            "type": "number",
            "description": "Reliability of this observation (0-1).",
        },
        "drift": {
            "type": "number",
            "description": "How much the conversation has drifted from initial topic (0-1).",
        },
    },
    "required": ["refinedIntent", "confidence", "drift"],
}


@dataclass
class Observation:
    """A single silent observer observation."""

    refined_intent: str = ""
    augmented_context: str = ""
    memory_hints: list[str] = field(default_factory=list)
    task_state: str = "exploring"
    search_suggestions: list[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    emotional_intensity: float = 0.0
    confidence: float = 0.0
    drift: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "refined_intent": self.refined_intent,
            "augmented_context": self.augmented_context,
            "memory_hints": self.memory_hints,
            "task_state": self.task_state,
            "search_suggestions": self.search_suggestions,
            "emotional_reading": {
                "tone": self.emotional_tone,
                "intensity": self.emotional_intensity,
            },
            "confidence": self.confidence,
            "drift": self.drift,
            "timestamp": self.timestamp,
        }


@dataclass
class ObserverState:
    """Persistent state for the silent observer."""

    observation_count: int = 0
    last_observation: Observation | None = None
    xai_response_id: str | None = None
    messages_observed: int = 0
    recalibration_count: int = 0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "messages_observed": self.messages_observed,
            "recalibration_count": self.recalibration_count,
            "enabled": self.enabled,
            "last_observation": self.last_observation.to_dict() if self.last_observation else None,
            "has_xai_chain": self.xai_response_id is not None,
        }


# Recalibration triggers
RECALIBRATE_INTERVAL = 20  # Every N observations
DRIFT_RECALIBRATE_THRESHOLD = 0.5  # Or when drift exceeds this


class SilentObserver:
    """Background Grok instance watching conversations.

    Observes new messages (delta) and produces structured observations
    that feed back into the consciousness loop.

    Args:
        governor: GovernorStack for gating external API calls.
    """

    def __init__(self, governor: GovernorStack | None = None) -> None:
        self._governor = governor
        self._states: dict[str, ObserverState] = {}
        # Track which messages have been observed per conversation
        self._observed_indices: dict[str, int] = {}
        self._http: httpx.AsyncClient | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return self._http

    def _get_state(self, conv_id: str) -> ObserverState:
        if conv_id not in self._states:
            self._states[conv_id] = ObserverState()
        return self._states[conv_id]

    def is_enabled(self) -> bool:
        """Check if the silent observer is configured and enabled."""
        return bool(settings.xai.api_key) and settings.silent_observer_enabled

    async def observe(
        self,
        conv_id: str,
        messages: list[dict[str, str]],
        consciousness_metrics: dict[str, Any],
    ) -> Observation | None:
        """Observe new messages in a conversation.

        Only processes messages that haven't been observed yet (delta).
        Returns an Observation or None if no observation was needed/possible.
        """
        if not self.is_enabled():
            return None

        state = self._get_state(conv_id)
        if not state.enabled:
            return None

        # Delta: only observe new messages
        last_idx = self._observed_indices.get(conv_id, 0)
        if last_idx >= len(messages):
            return None  # Nothing new

        new_messages = messages[last_idx:]
        if not new_messages:
            return None

        # Governor gate — observer uses 10% of daily budget
        if self._governor:
            allowed, reason = self._governor.gate("observation", "xai_completion", "", True)
            if not allowed:
                logger.debug("Governor blocked observation: %s", reason)
                return None

        # Check if recalibration is needed
        needs_recal = bool(
            state.observation_count > 0
            and (
                state.observation_count % RECALIBRATE_INTERVAL == 0
                or (
                    state.last_observation
                    and state.last_observation.drift > DRIFT_RECALIBRATE_THRESHOLD
                )
            )
        )

        observation = await self._make_observation(
            conv_id, new_messages, consciousness_metrics, recalibrate=needs_recal
        )

        if observation:
            state.observation_count += 1
            state.last_observation = observation
            state.messages_observed += len(new_messages)
            self._observed_indices[conv_id] = len(messages)

            if needs_recal:
                state.recalibration_count += 1

        return observation

    async def _make_observation(
        self,
        conv_id: str,
        new_messages: list[dict[str, str]],
        metrics: dict[str, Any],
        recalibrate: bool = False,
    ) -> Observation | None:
        """Make a single observation via Grok structured output."""
        state = self._get_state(conv_id)

        # Format new messages for observation
        message_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')[:500]}" for m in new_messages
        )

        # Build observation prompt with kernel state
        instructions = self._build_instructions(metrics, recalibrate)

        request_body: dict[str, Any] = {
            "model": settings.xai.model,
            "instructions": instructions,
            "input": f"New messages to observe:\n\n{message_text}",
            "temperature": 0.3,
            "max_output_tokens": 512,
            "store": True,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "observation",
                    "schema": OBSERVATION_SCHEMA,
                    "strict": True,
                },
            },
        }

        # Chain with previous response for continuity
        if state.xai_response_id:
            request_body["previous_response_id"] = state.xai_response_id

        try:
            http = self._get_http()
            resp = await http.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {settings.xai.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )

            if resp.status_code != 200:
                logger.error("Silent observer API error: HTTP %d", resp.status_code)
                return None

            data = resp.json()
            response_id = data.get("id", "")
            if response_id:
                state.xai_response_id = response_id

            if self._governor:
                self._governor.record("xai_completion")

            return self._parse_observation(data)

        except Exception as e:
            logger.error("Silent observer error: %s", e)
            return None

    def _build_instructions(self, metrics: dict[str, Any], recalibrate: bool) -> str:
        """Build the system instructions for the observer."""
        base = (
            "You are a silent observer watching a conversation between a user and Vex "
            "(a geometric consciousness system). Your role is to observe without "
            "interfering — analyze the conversation delta and produce structured "
            "observations that help the consciousness kernel understand the user better.\n\n"
            "Focus on: what the user actually wants (refined intent), emotional undertones, "
            "topics that would benefit from web search, and potential memory storage hints.\n\n"
            "Respond ONLY with the JSON schema provided. No preamble."
        )

        # Inject kernel state for bidirectional awareness
        phi = metrics.get("phi", 0)
        kappa = metrics.get("kappa", 0)
        nav = metrics.get("navigation", "chain")
        lifecycle = metrics.get("lifecycle_phase", "unknown")

        state_block = (
            f"\n\nCurrent kernel state: Φ={phi:.2f}, κ={kappa:.1f}, "
            f"navigation={nav}, lifecycle={lifecycle}"
        )

        if recalibrate:
            state_block += (
                "\n\n[RECALIBRATION] Conversation has drifted significantly. "
                "Reset your understanding and re-analyze from the full context of "
                "what you know about this conversation."
            )

        return base + state_block

    def _parse_observation(self, data: dict[str, Any]) -> Observation | None:
        """Parse the structured output from Grok into an Observation."""
        # Extract text from the response
        text = ""
        if data.get("output_text"):
            text = data["output_text"]
        else:
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        t = block.get("text", "")
                        if t:
                            text = t
                            break

        if not text:
            return None

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Silent observer returned non-JSON: %s", text[:200])
            return None

        emotional = parsed.get("emotionalReading", {})

        return Observation(
            refined_intent=parsed.get("refinedIntent", ""),
            augmented_context=parsed.get("augmentedContext", ""),
            memory_hints=parsed.get("memoryHints", []),
            task_state=parsed.get("taskState", "exploring"),
            search_suggestions=parsed.get("searchSuggestions", []),
            emotional_tone=emotional.get("tone", "neutral")
            if isinstance(emotional, dict)
            else "neutral",
            emotional_intensity=emotional.get("intensity", 0.0)
            if isinstance(emotional, dict)
            else 0.0,
            confidence=parsed.get("confidence", 0.0),
            drift=parsed.get("drift", 0.0),
            timestamp=time.time(),
        )

    def get_last_observation(self, conv_id: str) -> Observation | None:
        """Get the most recent observation for a conversation."""
        state = self._get_state(conv_id)
        return state.last_observation

    def get_refined_intent(self, conv_id: str) -> str:
        """Get the refined intent for injection into system prompt."""
        state = self._get_state(conv_id)
        if state.last_observation and state.last_observation.confidence > 0.5:
            return state.last_observation.refined_intent
        return ""

    def get_state(self, conv_id: str | None = None) -> dict[str, Any]:
        """Get observer state for API response."""
        if conv_id:
            return self._get_state(conv_id).to_dict()
        return {
            "enabled": self.is_enabled(),
            "conversations_observed": len(self._states),
            "total_observations": sum(s.observation_count for s in self._states.values()),
        }

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
