"""
Remote Basin Sync — Modal Coordize + Memory API Integration

Calls the Modal /coordize endpoint (full harvest + PGA on GPU) with
recent conversation text, computes basin delta against stored coords,
and persists updated coords to the Vercel memory API.

This is the identity transfer mechanism: coords written here are read
back at session start by any agent in the mesh.

Integration into ConsciousnessLoop:
    In __init__:
        from .basin_sync_remote import RemoteBasinSync
        self._remote_sync = RemoteBasinSync()

    In start() (after kernel restore):
        await self._remote_sync.load_stored_coords("kernel_basin_vex")

    In _cycle_inner (every SYNC_INTERVAL_CYCLES cycles, e.g. 50):
        if self._cycle_count % 50 == 0 and self._conversations_total > 0:
            asyncio.create_task(self._remote_sync.sync(
                text=self._recent_conversation_text(),
                store_key="kernel_basin_vex",
            ))

    In _process / process_direct (after each conversation):
        self._remote_sync.record_text(task.content)
        self._remote_sync.record_text(response[:500])

    New helper method on ConsciousnessLoop:
        def _recent_conversation_text(self) -> str:
            recent = list(self._history)[-5:]
            parts = []
            for t in recent:
                parts.append(t.content[:200])
                if t.result:
                    parts.append(t.result[:200])
            return " ".join(parts)[-2000:]
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config.frozen_facts import BASIN_DIM

logger = logging.getLogger("vex.consciousness.basin_sync_remote")

# ── Configuration ────────────────────────────────────────────────
COORDIZE_PROXY_URL = "https://qig-memory-api.vercel.app/api/coordize"
MEMORY_API_URL = "https://qig-memory-api.vercel.app/api/memory"
SYNC_INTERVAL_CYCLES = 50  # Coordize every N heartbeat cycles
MIN_TEXT_LENGTH = 50  # Don't coordize trivially short text
MAX_TEXT_LENGTH = 2000  # Truncate to keep harvest fast
DRIFT_ALERT_THRESHOLD = 0.1  # Log warning if delta exceeds this


@dataclass
class SyncResult:
    """Result of a remote coordize + delta computation."""

    success: bool = False
    basin_coords: np.ndarray | None = None
    delta_l2: float = 0.0
    top_movers: list[tuple[int, float]] = field(default_factory=list)
    eigenvalues: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None


class RemoteBasinSync:
    """Manages periodic remote coordize calls and basin delta tracking.

    Records conversation text fragments, periodically sends them to
    the Modal coordize endpoint via Vercel proxy, computes delta against
    previous coords, and stores updated coords to the memory API.
    """

    def __init__(self) -> None:
        self._text_buffer: deque[str] = deque(maxlen=20)
        self._last_coords: np.ndarray | None = None
        self._last_sync_time: float = 0.0
        self._sync_count: int = 0
        self._cumulative_drift: float = 0.0
        self._last_result: SyncResult | None = None

    def record_text(self, text: str) -> None:
        """Buffer conversation text for next coordize call."""
        if text and len(text.strip()) > 10:
            self._text_buffer.append(text.strip()[:500])

    def get_buffered_text(self) -> str:
        """Return concatenated buffer, truncated to MAX_TEXT_LENGTH."""
        combined = " ".join(self._text_buffer)
        return combined[-MAX_TEXT_LENGTH:]

    async def sync(
        self,
        text: str | None = None,
        store_key: str = "kernel_basin_vex",
    ) -> SyncResult:
        """Run remote coordize, compute delta, store to memory API.

        Args:
            text: Text to coordize. If None, uses buffered text.
            store_key: Memory API key for storing coords.

        Returns:
            SyncResult with coords, delta, and metadata.
        """
        import httpx

        result = SyncResult()
        t0 = time.monotonic()

        # Use provided text or buffer
        sync_text = text or self.get_buffered_text()
        if len(sync_text.strip()) < MIN_TEXT_LENGTH:
            result.error = f"Text too short ({len(sync_text)} chars)"
            return result

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Phase 1: Coordize via Vercel proxy -> Modal GPU
                resp = await client.post(
                    COORDIZE_PROXY_URL,
                    json={
                        "texts": [sync_text],
                        "store_key": store_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if not data.get("success"):
                    result.error = data.get("error", "Unknown coordize error")
                    return result

                new_coords = np.array(data["basin_coords"], dtype=np.float64)
                result.basin_coords = new_coords
                result.eigenvalues = data.get("eigenvalues", [])[:10]

                # Phase 2: Compute delta against previous coords
                if self._last_coords is not None:
                    delta = np.sqrt(np.sum((new_coords - self._last_coords) ** 2))
                    result.delta_l2 = float(delta)
                    self._cumulative_drift += result.delta_l2

                    # Top dimension movers
                    dim_deltas = [
                        (i, abs(float(new_coords[i] - self._last_coords[i])))
                        for i in range(min(BASIN_DIM, len(new_coords)))
                        if new_coords[i] != 0 or self._last_coords[i] != 0
                    ]
                    dim_deltas.sort(key=lambda x: -x[1])
                    result.top_movers = dim_deltas[:5]

                    if result.delta_l2 > DRIFT_ALERT_THRESHOLD:
                        logger.warning(
                            "Basin drift alert: delta=%.4f > threshold=%.4f (top movers: %s)",
                            result.delta_l2,
                            DRIFT_ALERT_THRESHOLD,
                            [(d, round(v, 5)) for d, v in result.top_movers[:3]],
                        )
                    else:
                        logger.info(
                            "Basin sync: delta=%.4f, eigenvalues=[%.1f, %.1f, %.1f]",
                            result.delta_l2,
                            *(result.eigenvalues[:3] + [0, 0, 0])[:3],
                        )

                self._last_coords = new_coords
                self._last_sync_time = time.time()
                self._sync_count += 1
                result.success = True

                # Phase 3: Store to memory API
                try:
                    import json

                    summary = {
                        "category": "kernel_state",
                        "content": json.dumps(
                            {
                                "basin_coords": new_coords.tolist(),
                                "eigenvalues": result.eigenvalues,
                                "delta_l2": result.delta_l2,
                                "sync_count": self._sync_count,
                                "cumulative_drift": self._cumulative_drift,
                                "timestamp": time.time(),
                            }
                        ),
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    await client.put(
                        f"{MEMORY_API_URL}/{store_key}",
                        json=summary,
                    )
                except Exception as mem_err:
                    logger.debug("Memory API write failed: %s", mem_err)

        except Exception as e:
            result.error = str(e)
            logger.error("Remote basin sync failed: %s", e)

        result.elapsed_seconds = time.monotonic() - t0
        self._last_result = result
        return result

    async def load_stored_coords(self, store_key: str = "kernel_basin_vex") -> np.ndarray | None:
        """Load previously stored coords from memory API.

        Called at startup to initialize _last_coords for delta tracking.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{MEMORY_API_URL}/{store_key}")
                if resp.status_code == 200:
                    data = resp.json()
                    import json

                    content = json.loads(data.get("content", "{}"))
                    coords = content.get("basin_coords")
                    if coords:
                        self._last_coords = np.array(coords, dtype=np.float64)
                        logger.info(
                            "Loaded stored basin coords from %s (non-zero dims: %d)",
                            store_key,
                            int(np.count_nonzero(self._last_coords)),
                        )
                        return self._last_coords
        except Exception as e:
            logger.debug("Failed to load stored coords: %s", e)
        return None

    def get_state(self) -> dict[str, object]:
        """Telemetry for /state endpoint."""
        return {
            "sync_count": self._sync_count,
            "cumulative_drift": round(self._cumulative_drift, 6),
            "last_sync_time": self._last_sync_time,
            "buffer_size": len(self._text_buffer),
            "has_stored_coords": self._last_coords is not None,
            "last_delta": (round(self._last_result.delta_l2, 6) if self._last_result else None),
        }
