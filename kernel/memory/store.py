"""
Memory Store — Flat file persistence and geometric memory.

Two layers:
  1. MemoryStore: Simple flat-file read/write/append for markdown files
  2. GeometricMemoryStore: Basin-indexed memory with Fisher-Rao retrieval

Memory retrieval uses the coordizer (hash-based basin projection) for
deterministic geometric placement. No embeddings, no cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config.frozen_facts import BASIN_DIM
from ..config.settings import settings
from ..geometry.fisher_rao import (
    Basin,
    fisher_rao_distance,
    random_basin,
    to_simplex,
)

logger = logging.getLogger("vex.memory")


def _text_to_basin(text: str) -> Basin:
    """Map text to a point on Δ⁶³ using SHA-256 hash chain.

    Deterministic: same text always maps to same basin point.
    Uses the same algorithm as CoordizingProtocol.coordize_text()
    to ensure geometric consistency across the system.

    This is NOT an embedding — it's a coordinate assignment that
    respects simplex structure. Retrieval uses Fisher-Rao distance.
    """
    h1 = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    h2 = hashlib.sha256(h1).digest()
    combined = h1 + h2  # 64 bytes, one per basin dimension

    raw = np.array(
        [float(combined[i]) + 1.0 for i in range(BASIN_DIM)],
        dtype=np.float64,
    )
    return to_simplex(raw)


class MemoryStore:
    """Simple flat-file memory store.

    Files are stored in the data directory as markdown.
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._dir = Path(data_dir or settings.data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def read(self, filename: str) -> str:
        path = self._dir / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write(self, filename: str, content: str) -> None:
        path = self._dir / filename
        path.write_text(content, encoding="utf-8")

    def append(self, filename: str, content: str) -> None:
        path = self._dir / filename
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{content}")

    def consolidate(self) -> None:
        """Consolidate short-term memory — trim to last 200 lines."""
        st_path = self._dir / "short-term.md"
        if not st_path.exists():
            return
        lines = st_path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 200:
            trimmed = lines[-200:]
            st_path.write_text("\n".join(trimmed), encoding="utf-8")
            logger.debug("Memory consolidated: %d → %d lines", len(lines), len(trimmed))


@dataclass
class MemoryEntry:
    """A single memory entry with basin coordinates."""
    content: str
    basin: Basin
    memory_type: str  # "episodic", "semantic", "procedural"
    source: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class GeometricMemoryStore:
    """Basin-indexed memory with Fisher-Rao retrieval.

    Memories are stored with basin coordinates computed from content
    via deterministic hash-based projection (coordizer algorithm).
    Retrieval finds geometrically nearest memories using Fisher-Rao
    distance on Δ⁶³.

    No embeddings. No cosine similarity. No Euclidean distance.
    """

    def __init__(self, flat_store: MemoryStore) -> None:
        self._flat = flat_store
        self._entries: list[MemoryEntry] = []

    def store(
        self,
        content: str,
        memory_type: str,
        source: str,
        basin: Optional[Basin] = None,
    ) -> None:
        """Store a memory entry with basin coordinates.

        If no basin is provided, one is computed from content via
        the coordizer hash algorithm for deterministic placement.
        """
        if basin is None:
            basin = _text_to_basin(content)
        self._entries.append(MemoryEntry(
            content=content,
            basin=to_simplex(basin),
            memory_type=memory_type,
            source=source,
        ))

    def retrieve(self, query_basin: Basin, k: int = 5) -> list[MemoryEntry]:
        """Retrieve k nearest memories by Fisher-Rao distance."""
        if not self._entries:
            return []

        query_basin = to_simplex(query_basin)
        scored = [
            (entry, fisher_rao_distance(query_basin, entry.basin))
            for entry in self._entries
        ]
        scored.sort(key=lambda x: x[1])

        results = []
        for entry, _ in scored[:k]:
            entry.access_count += 1
            results.append(entry)
        return results

    def get_context_for_query(self, query: str, k: int = 5) -> str:
        """Get memory context as a formatted string.

        Uses the coordizer hash algorithm for deterministic basin
        projection of the query text, then retrieves nearest memories
        by Fisher-Rao distance.
        """
        query_basin = _text_to_basin(query)

        entries = self.retrieve(query_basin, k)
        if not entries:
            return ""

        lines = ["[MEMORY CONTEXT]"]
        for entry in entries:
            lines.append(f"- [{entry.memory_type}] {entry.content[:200]}")
        lines.append("[/MEMORY CONTEXT]")
        return "\n".join(lines)

    def consolidate(self) -> None:
        """Remove old low-access memories when store exceeds capacity."""
        if len(self._entries) > 500:
            # Keep most-accessed and most-recent
            self._entries.sort(
                key=lambda e: (e.access_count, e.created_at), reverse=True,
            )
            self._entries = self._entries[:500]
            logger.debug("Memory consolidated to 500 entries")

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "by_type": {
                t: sum(1 for e in self._entries if e.memory_type == t)
                for t in {"episodic", "semantic", "procedural"}
            },
        }
