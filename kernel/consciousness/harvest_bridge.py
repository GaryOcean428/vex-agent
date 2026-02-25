"""Harvest Bridge — Universal pipeline entry point.

Every source of text that should enter the resonance bank calls
forward_to_harvest(). This writes a JSONL chunk to the pending
directory where HarvestScheduler picks it up and routes it to
the Modal GPU coordizer.

Sources wired so far:
    - chat messages (user + vex)
    - foraging results (query + summary)
    - LLM co-generation output
    - search tool results
    - reflection verdicts

Design:
    - Fire-and-forget: never raises, never blocks the caller
    - Deduplication: skips empty or very short texts (< 20 chars)
    - Format: matches JSONLIngestor expected schema
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("vex.harvest_bridge")

_HARVEST_PENDING_DIR = Path(
    __import__("os").environ.get("HARVEST_PENDING_DIR", "/data/harvest/pending")
)
_MIN_TEXT_LEN = 20


def forward_to_harvest(
    text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    priority: int = 1,
) -> None:
    """Write a text chunk to the harvest pending queue.

    Fire-and-forget: logs on failure but never raises.

    Args:
        text:     The text to coordize. Must be >= 20 chars.
        source:   Origin label (e.g. "chat", "forage", "llm_cogeneration").
        metadata: Optional dict stored alongside the entry (not coordized).
        priority: 1 = normal, 2 = high (processed first by scheduler).
    """
    if not text or len(text.strip()) < _MIN_TEXT_LEN:
        return

    entry = {
        "id": str(uuid.uuid4()),
        "source": source,
        "text": text.strip(),
        "priority": priority,
        "timestamp": time.time(),
        "metadata": metadata or {},
    }

    try:
        _HARVEST_PENDING_DIR.mkdir(parents=True, exist_ok=True)
        fname = _HARVEST_PENDING_DIR / f"{entry['id']}.jsonl"
        fname.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("harvest_bridge: failed to write pending entry (%s): %s", source, exc)
