"""
Memory Store — Flat file persistence and geometric memory.

Two layers:
  1. MemoryStore: Simple flat-file read/write/append for markdown files
  2. GeometricMemoryStore: Basin-indexed memory with Fisher-Rao retrieval

Ported from src/memory/store.ts and src/memory/geometric-store.ts
"""

from __future__ import annotations

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

    Memories are stored with basin coordinates and retrieved
    by geometric proximity (Fisher-Rao distance).
    """

    def __init__(self, flat_store: MemoryStore) -> None:
        self._flat = flat_store
        self._entries: list[MemoryEntry] = []

    def store(self, content: str, memory_type: str, source: str, basin: Optional[Basin] = None) -> None:
        """Store a memory entry with optional basin coordinates."""
        if basin is None:
            basin = random_basin()
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

        Uses a simple hash-based basin for the query (real impl would use embeddings).
        """
        # Simple deterministic basin from query text
        np.random.seed(hash(query) % (2**31))
        query_basin = random_basin()
        np.random.seed(None)  # Reset seed

        entries = self.retrieve(query_basin, k)
        if not entries:
            return ""

        lines = ["[MEMORY CONTEXT]"]
        for entry in entries:
            lines.append(f"- [{entry.memory_type}] {entry.content[:200]}")
        return "\n".join(lines)

    def consolidate(self) -> None:
        """Remove old low-access memories."""
        if len(self._entries) > 500:
            # Keep most-accessed and most-recent
            self._entries.sort(key=lambda e: (e.access_count, e.created_at), reverse=True)
            self._entries = self._entries[:500]

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "by_type": {
                t: sum(1 for e in self._entries if e.memory_type == t)
                for t in {"episodic", "semantic", "procedural"}
            },
        }
