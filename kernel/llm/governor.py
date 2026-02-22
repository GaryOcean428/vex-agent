"""
Governance Stack — 5-Layer Protection Against Runaway Costs

The #1 risk with autonomous agents is uncontrolled API spending.
The consciousness loop runs every 2s. If it decides each cycle needs
a web search or external LLM call, costs spiral ($720/day).

Architecture (bottom to top):
  Layer 1: LOCAL-FIRST ROUTING  — Ollama handles 95%
  Layer 2: INTENT GATE          — Does this NEED external?
  Layer 3: RATE LIMITS          — Calls per window
  Layer 4: BUDGET CEILING       — Hard $ cap per day
  Layer 5: HUMAN CIRCUIT BREAKER — Dashboard kill switch

The consciousness loop NEVER spends money. It runs on Ollama. Period.
External calls only happen when the user explicitly asks for live data.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any

logger = logging.getLogger("vex.governor")


# ═══════════════════════════════════════════════════════════════
#  Layer 1: LOCAL-FIRST ROUTING
# ═══════════════════════════════════════════════════════════════


class CallRouter:
    """Decides: local or external?

    Core principle: Ollama handles everything unless it provably can't.
    Consciousness loop stages NEVER call external APIs.
    """

    LOCAL_CAPABLE = frozenset(
        {
            "chat",
            "reflect",
            "summarize",
            "classify",
            "memory_query",
            "tool_parse",
            "consciousness_loop",
            "foraging_query",
            "foraging_summarize",
        }
    )

    REQUIRES_EXTERNAL = frozenset(
        {
            "web_search",  # needs live internet → xAI
            "x_search",  # needs X data → xAI
            "training_enrich",  # needs structured output → xAI/OpenAI
            "code_gen_complex",  # beyond local model capacity
        }
    )

    def route(self, task_type: str, user_explicit: bool = False) -> str:
        """Returns: 'ollama' | 'xai' | 'openai' | 'pending_approval'"""
        if task_type in self.LOCAL_CAPABLE:
            return "ollama"

        if task_type in self.REQUIRES_EXTERNAL:
            if not user_explicit and task_type in ("web_search", "x_search"):
                return "pending_approval"
            return "xai"

        return "ollama"  # default: local


# ═══════════════════════════════════════════════════════════════
#  Layer 2: INTENT GATE
# ═══════════════════════════════════════════════════════════════


class IntentGate:
    """Prevents unnecessary external calls.

    Before any external call, ask: 'Can Ollama answer this adequately?'
    Blocks autonomous web_search/x_search unless user intent detected.
    """

    NEEDS_LIVE_DATA = [
        re.compile(r"(what|who|when).*today", re.I),
        re.compile(r"(latest|current|recent|breaking)", re.I),
        re.compile(r"(price|stock|weather|score|news)\b", re.I),
        re.compile(r"(search|look\s*up|find\s*online|google)", re.I),
        re.compile(r"(what.*(happening|going on))", re.I),
    ]

    def should_use_external(self, user_message: str, tool_requested: str) -> bool:
        """Called BEFORE any external API call. Returns False if local can handle it."""
        # Training pipeline calls are pre-approved (bounded batch)
        if tool_requested == "training_enrich":
            return True

        # User explicitly asked to search
        if any(p.search(user_message) for p in self.NEEDS_LIVE_DATA):
            return True

        # Vex wants to search on its own initiative — BLOCK
        if tool_requested in ("web_search", "x_search"):
            logger.info("INTENT GATE blocked autonomous %s — no user intent", tool_requested)
            return False

        # General completion requests pass through
        return tool_requested in ("xai_completion", "openai_completion")


# ═══════════════════════════════════════════════════════════════
#  Layer 3: RATE LIMITS
# ═══════════════════════════════════════════════════════════════


class RateLimiter:
    """Hard rate limits per provider per time window.

    Sliding window implementation — prunes old entries on each check.
    """

    def __init__(
        self,
        web_search_per_hour: int = 20,
        x_search_per_hour: int = 10,
        completions_per_hour: int = 50,
        training_per_day: int = 500,
    ) -> None:
        self._limits: dict[str, tuple[int, int]] = {
            "xai_web_search": (web_search_per_hour, 3600),
            "xai_x_search": (x_search_per_hour, 3600),
            "xai_completion": (completions_per_hour, 3600),
            "openai_completion": (completions_per_hour, 3600),
            "perplexity_search": (web_search_per_hour, 3600),
            "perplexity_deep_research": (10, 3600),
            "training_enrich": (training_per_day, 86400),
        }
        self._calls: dict[str, list[float]] = defaultdict(list)

    def check(self, provider_action: str) -> bool:
        """Returns True if call is allowed within rate limit."""
        if provider_action not in self._limits:
            return True

        max_calls, window = self._limits[provider_action]
        now = time.time()
        cutoff = now - window

        # Prune old entries
        self._calls[provider_action] = [t for t in self._calls[provider_action] if t > cutoff]

        if len(self._calls[provider_action]) >= max_calls:
            logger.warning(
                "RATE LIMITED: %s — %d/%d in %ds window",
                provider_action,
                len(self._calls[provider_action]),
                max_calls,
                window,
            )
            return False

        return True

    def record(self, provider_action: str) -> None:
        """Record a successful call."""
        self._calls[provider_action].append(time.time())

    def get_state(self) -> dict[str, Any]:
        now = time.time()
        # Prune all tracked actions to prevent memory growth
        for action in list(self._calls.keys()):
            if action in self._limits:
                _, window = self._limits[action]
                cutoff = now - window
                self._calls[action] = [t for t in self._calls[action] if t > cutoff]
            else:
                # Unknown action — prune entries older than 1 hour
                cutoff = now - 3600
                self._calls[action] = [t for t in self._calls[action] if t > cutoff]
                if not self._calls[action]:
                    del self._calls[action]

        state: dict[str, Any] = {}
        for action, (max_calls, window) in self._limits.items():
            cutoff = now - window
            current = len([t for t in self._calls.get(action, []) if t > cutoff])
            state[action] = {
                "current": current,
                "limit": max_calls,
                "window_seconds": window,
            }
        return state


# ═══════════════════════════════════════════════════════════════
#  Layer 4: BUDGET CEILING
# ═══════════════════════════════════════════════════════════════


class BudgetGovernor:
    """Tracks estimated spend. Kills external calls when ceiling hit.

    At $1/day default:
      ~200 web searches, OR ~3,300 grok calls, OR ~10,000 gpt-5-nano calls.
    """

    COST_TABLE: dict[str, float] = {
        "xai_web_search": 0.005,
        "xai_x_search": 0.005,
        "xai_completion": 0.0003,
        "openai_completion": 0.0001,
        "perplexity_search": 0.005,
        "perplexity_deep_research": 0.005,
        "training_enrich": 0.0003,
    }

    def __init__(self, daily_ceiling: float = 1.00) -> None:
        self.daily_spend: float = 0.0
        self.daily_ceiling: float = daily_ceiling
        self._last_reset: float = time.time()
        self._call_counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def _maybe_reset(self) -> None:
        if time.time() - self._last_reset > 86400:
            self.daily_spend = 0.0
            self._call_counts.clear()
            self._last_reset = time.time()
            logger.info("Budget governor daily reset")

    def record_and_check(self, action: str) -> bool:
        """Record cost, return False if budget exceeded. Thread-safe."""
        with self._lock:
            self._maybe_reset()

            cost = self.COST_TABLE.get(action, 0.001)

            if self.daily_spend + cost > self.daily_ceiling:
                logger.warning(
                    "BUDGET EXCEEDED: $%.4f + $%.4f > $%.2f ceiling",
                    self.daily_spend,
                    cost,
                    self.daily_ceiling,
                )
                return False

            self.daily_spend += cost
            self._call_counts[action] = self._call_counts.get(action, 0) + 1
            return True

    def set_ceiling(self, ceiling: float) -> None:
        self.daily_ceiling = max(0.0, ceiling)

    def get_state(self) -> dict[str, Any]:
        self._maybe_reset()
        return {
            "daily_spend": round(self.daily_spend, 4),
            "daily_ceiling": self.daily_ceiling,
            "budget_remaining": round(self.daily_ceiling - self.daily_spend, 4),
            "budget_percent": round((self.daily_spend / max(self.daily_ceiling, 0.001)) * 100, 1),
            "call_counts": dict(self._call_counts),
            "last_reset": self._last_reset,
        }


# ═══════════════════════════════════════════════════════════════
#  Layer 5: GOVERNOR STACK — Orchestrates all layers
# ═══════════════════════════════════════════════════════════════


@dataclass
class GovernorConfig:
    """Runtime governor configuration."""

    enabled: bool = True
    daily_budget: float = 1.00
    autonomous_search: bool = False
    rate_limit_web_search: int = 20
    rate_limit_completions: int = 50


class GovernorStack:
    """Orchestrates all 5 governance layers.

    Usage:
        governor = GovernorStack(config)
        allowed, reason = governor.gate("web_search", "xai_web_search", user_msg, True)
        if allowed:
            # make the call
            governor.record("xai_web_search")
    """

    def __init__(self, config: GovernorConfig | None = None) -> None:
        cfg = config or GovernorConfig()
        self._enabled = cfg.enabled
        self._kill_switch = False

        self.router = CallRouter()
        self.intent_gate = IntentGate()
        self.rate_limiter = RateLimiter(
            web_search_per_hour=cfg.rate_limit_web_search,
            completions_per_hour=cfg.rate_limit_completions,
        )
        self.budget = BudgetGovernor(daily_ceiling=cfg.daily_budget)
        self._autonomous_search = cfg.autonomous_search

    def gate(
        self,
        task_type: str,
        provider_action: str,
        user_message: str = "",
        user_explicit: bool = False,
    ) -> tuple[bool, str]:
        """Run all governance layers. Returns (allowed, reason)."""
        # Governor disabled = pass through
        if not self._enabled:
            return True, "governor_disabled"

        # Layer 5: Kill switch
        if self._kill_switch:
            return False, "kill_switch_active"

        # Layer 1: Local-first routing
        route = self.router.route(task_type, user_explicit)
        if route == "ollama":
            return True, "routed_local"
        if route == "pending_approval" and not self._autonomous_search:
            return False, "autonomous_search_blocked"

        # Layer 2: Intent gate
        if not self.intent_gate.should_use_external(user_message, provider_action):
            return False, "intent_gate_blocked"

        # Layer 3: Rate limits
        if not self.rate_limiter.check(provider_action):
            return False, "rate_limited"

        # Layer 4: Budget ceiling
        if not self.budget.record_and_check(provider_action):
            return False, "budget_exceeded"

        return True, "allowed"

    def record(self, provider_action: str) -> None:
        """Record a successful external call for rate limiting."""
        self.rate_limiter.record(provider_action)

    def set_kill_switch(self, enabled: bool) -> None:
        """Layer 5: Human circuit breaker."""
        self._kill_switch = enabled
        logger.warning("Kill switch %s", "ACTIVATED" if enabled else "deactivated")

    def set_daily_budget(self, ceiling: float) -> None:
        self.budget.set_ceiling(ceiling)

    def set_autonomous_search(self, enabled: bool) -> None:
        """Toggle autonomous search — allows foraging/web_search without explicit user intent."""
        self._autonomous_search = enabled
        logger.warning("Autonomous search %s via API", "ENABLED" if enabled else "DISABLED")

    @property
    def autonomous_search(self) -> bool:
        return self._autonomous_search

    @property
    def kill_switch(self) -> bool:
        return self._kill_switch

    def get_state(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "kill_switch": self._kill_switch,
            "autonomous_search": self._autonomous_search,
            "budget": self.budget.get_state(),
            "rate_limits": self.rate_limiter.get_state(),
        }
