"""Harvest Bridge — Universal pipeline entry point.

Every source of text that should enter the resonance bank calls
forward_to_harvest(). This writes a JSONL chunk to the pending
directory where HarvestScheduler picks it up and routes it to
the Modal GPU coordizer.

Sources wired so far:
    - chat messages (user + vex)         → source="conversation"
    - foraging results (query + summary) → source="foraging"
    - LLM co-generation output           → source="conversation"
    - search tool results                → source="foraging"
    - reflection verdicts                → source="conversation"
    - debate transcripts                 → source="conversation"
    - curiosity queries                  → source="foraging"

Design:
    - Fire-and-forget: never raises, never blocks the caller
    - Deduplication: skips empty or very short texts (< 20 chars)
    - Format: matches JSONLIngestor expected schema exactly
    - Source mapping: only VALID_SOURCES pass through

Source mapping:
    JSONLIngestor.VALID_SOURCES = {"curriculum", "foraging", "conversation", "document"}
    All call sites MUST use one of these. Original sub-source preserved
    in metadata["origin"] for provenance tracking.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("vex.harvest_bridge")

_HARVEST_PENDING_DIR = Path(
    __import__("os").environ.get("HARVEST_PENDING_DIR", "/data/harvest/pending")
)
_MIN_TEXT_LEN = 20

# Only these sources pass JSONLIngestor validation.
VALID_SOURCES = frozenset({"curriculum", "foraging", "conversation", "document"})

# Sanitize filenames: allow only alphanumeric, dash, underscore
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


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
        source:   Must be one of VALID_SOURCES: "curriculum", "foraging",
                  "conversation", or "document". Use metadata["origin"]
                  for finer provenance (e.g. "llm_cogeneration", "debate").
        metadata: Optional dict stored alongside the entry (not coordized).
        priority: 1 = normal, 2 = high (processed first by scheduler).
    """
    if not text or len(text.strip()) < _MIN_TEXT_LEN:
        return

    # Validate source against JSONLIngestor's allowed set
    if source not in VALID_SOURCES:
        logger.warning(
            "harvest_bridge: invalid source %r — must be one of %s. Dropping entry.",
            source,
            VALID_SOURCES,
        )
        return

    entry_id = str(uuid.uuid4())

    entry = {
        "id": entry_id,
        "source": source,
        "text": text.strip(),
        "priority": priority,
        "timestamp": datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": metadata or {},
    }

    try:
        _HARVEST_PENDING_DIR.mkdir(parents=True, exist_ok=True)
        # Sanitize the filename to prevent path injection
        safe_id = _SAFE_FILENAME_RE.sub("", entry_id)
        fname = _HARVEST_PENDING_DIR / f"{safe_id}.jsonl"
        fname.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("harvest_bridge: failed to write pending entry (%s): %s", source, exc)
