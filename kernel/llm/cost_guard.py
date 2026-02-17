"""
Cost Guard — Rate limiting and budget enforcement for external LLM calls.

Prevents runaway costs when Ollama is unavailable and requests
fall through to paid APIs (OpenAI, Perplexity, etc.).

Fail-safe: if the budget is exceeded, requests are BLOCKED (not queued).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger("vex.llm.cost_guard")


@dataclass
class CostGuardConfig:
    """Cost guard configuration. All limits are PER GUARD INSTANCE."""

    # Requests per minute (RPM) — hard cap
    rpm_limit: int = 20

    # Requests per hour — hard cap
    rph_limit: int = 200

    # Requests per day — hard cap
    rpd_limit: int = 2000

    # Max tokens per request (passed to API)
    max_tokens_per_request: int = 2048

    # Kill switch — if True, all external calls are blocked
    kill_switch: bool = False


@dataclass
class CostGuardState:
    """Tracks request counts for rate limiting."""
    minute_requests: list[float] = field(default_factory=list)
    hour_requests: list[float] = field(default_factory=list)
    day_requests: list[float] = field(default_factory=list)
    total_requests: int = 0
    total_blocked: int = 0
    last_request_at: float = 0.0


class CostGuard:
    """Rate limiter for external LLM API calls.

    Usage:
        guard = CostGuard()
        if guard.allow():
            # make the API call
            guard.record()
        else:
            # return fallback / error
    """

    def __init__(self, config: CostGuardConfig | None = None) -> None:
        self.config = config or CostGuardConfig()
        self._state = CostGuardState()
        self._lock = Lock()

    def allow(self) -> bool:
        """Check if a request is allowed under current limits.

        Thread-safe. Returns False if any limit is exceeded or kill switch is on.
        """
        if self.config.kill_switch:
            return False

        with self._lock:
            now = time.time()
            self._prune(now)

            # Check RPM
            if len(self._state.minute_requests) >= self.config.rpm_limit:
                self._state.total_blocked += 1
                logger.warning(
                    "CostGuard BLOCKED: RPM limit (%d/%d)",
                    len(self._state.minute_requests), self.config.rpm_limit,
                )
                return False

            # Check RPH
            if len(self._state.hour_requests) >= self.config.rph_limit:
                self._state.total_blocked += 1
                logger.warning(
                    "CostGuard BLOCKED: RPH limit (%d/%d)",
                    len(self._state.hour_requests), self.config.rph_limit,
                )
                return False

            # Check RPD
            if len(self._state.day_requests) >= self.config.rpd_limit:
                self._state.total_blocked += 1
                logger.warning(
                    "CostGuard BLOCKED: RPD limit (%d/%d)",
                    len(self._state.day_requests), self.config.rpd_limit,
                )
                return False

            return True

    def record(self) -> None:
        """Record a successful request."""
        with self._lock:
            now = time.time()
            self._state.minute_requests.append(now)
            self._state.hour_requests.append(now)
            self._state.day_requests.append(now)
            self._state.total_requests += 1
            self._state.last_request_at = now

    def summary(self) -> dict[str, int | float | bool]:
        """Get current rate limiting state."""
        with self._lock:
            now = time.time()
            self._prune(now)
            return {
                "rpm_current": len(self._state.minute_requests),
                "rpm_limit": self.config.rpm_limit,
                "rph_current": len(self._state.hour_requests),
                "rph_limit": self.config.rph_limit,
                "rpd_current": len(self._state.day_requests),
                "rpd_limit": self.config.rpd_limit,
                "total_requests": self._state.total_requests,
                "total_blocked": self._state.total_blocked,
                "kill_switch": self.config.kill_switch,
            }

    def _prune(self, now: float) -> None:
        """Remove expired timestamps from sliding windows."""
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400

        self._state.minute_requests = [
            t for t in self._state.minute_requests if t > minute_ago
        ]
        self._state.hour_requests = [
            t for t in self._state.hour_requests if t > hour_ago
        ]
        self._state.day_requests = [
            t for t in self._state.day_requests if t > day_ago
        ]
