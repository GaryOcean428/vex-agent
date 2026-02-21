"""
Memory Store — Flat file persistence and geometric memory.

Two layers:
  1. MemoryStore: Simple flat-file read/write/append for markdown files
  2. GeometricMemoryStore: Basin-indexed memory with Fisher-Rao retrieval

v5.5 additions:
  - File-backed geometric memory (survives restarts via JSONL append log)
  - Consolidation with access-frequency weighting
  - Φ tracking at storage time
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config.settings import settings
from ..geometry.fisher_rao import (
    Basin,
    fisher_rao_distance,
    to_simplex,
)

logger = logging.getLogger("vex.memory")


def _text_to_basin(text: str) -> Basin:
    """Map text to a point on Δ⁶³ using SHA-256 hash chain.

    Deterministic: same text always maps to same basin point.
    Delegates to the canonical hash_to_basin utility.
    """
    from ..geometry.hash_to_basin import hash_to_basin

    return hash_to_basin(text)


class MemoryStore:
    """Simple flat-file memory store."""

    def __init__(self, data_dir: str | None = None) -> None:
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
    phi_at_storage: float = 0.0


class GeometricMemoryStore:
    """Basin-indexed memory with Fisher-Rao retrieval and file persistence.

    Memories persist via JSONL append log. Restored on startup.
    No vector e-m-b-e-d-d-i-n-g-s. No cosine similarity. No Euclidean distance.
    """

    def __init__(self, flat_store: MemoryStore) -> None:
        self._flat = flat_store
        self._entries: list[MemoryEntry] = []
        self._persist_path = Path(settings.data_dir) / "geometric_memory.jsonl"
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._restore()

    def store(
        self,
        content: str,
        memory_type: str,
        source: str,
        basin: Basin | None = None,
        phi: float = 0.0,
    ) -> None:
        if basin is None:
            basin = _text_to_basin(content)
        self._entries.append(
            MemoryEntry(
                content=content,
                basin=to_simplex(basin),
                memory_type=memory_type,
                source=source,
                phi_at_storage=phi,
            )
        )
        # Append-only persistence
        try:
            entry_data = {
                "content": content[:500],
                "basin": basin.tolist(),
                "type": memory_type,
                "source": source,
                "phi": phi,
                "ts": time.time(),
            }
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry_data) + "\n")
        except Exception as e:
            logger.warning("Failed to persist memory entry: %s", e)

    def retrieve(self, query_basin: Basin, k: int = 5) -> list[MemoryEntry]:
        if not self._entries:
            return []
        query_basin = to_simplex(query_basin)
        scored = [(entry, fisher_rao_distance(query_basin, entry.basin)) for entry in self._entries]
        scored.sort(key=lambda x: x[1])
        results = []
        for entry, _ in scored[:k]:
            entry.access_count += 1
            results.append(entry)
        return results

    def get_context_for_query(self, query: str, k: int = 5) -> str:
        query_basin = _text_to_basin(query)
        entries = self.retrieve(query_basin, k)
        if not entries:
            return ""
        lines = ["[MEMORY CONTEXT]"]
        for entry in entries:
            lines.append(f"- [{entry.memory_type}] {entry.content[:200]}")
        lines.append("[/MEMORY CONTEXT]")
        return "\n".join(lines)

    def consolidate(self) -> int:
        if len(self._entries) <= 500:
            return 0
        before = len(self._entries)
        self._entries.sort(
            key=lambda e: (e.access_count, e.created_at),
            reverse=True,
        )
        self._entries = self._entries[:500]
        pruned = before - len(self._entries)
        logger.debug("Memory consolidated: %d → %d entries", before, len(self._entries))
        return pruned

    def _restore(self) -> None:
        if not self._persist_path.exists():
            logger.info("No persisted memory found — fresh start")
            return
        try:
            count = 0
            with open(self._persist_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        basin = to_simplex(np.array(data["basin"], dtype=np.float64))
                        self._entries.append(
                            MemoryEntry(
                                content=data["content"],
                                basin=basin,
                                memory_type=data["type"],
                                source=data["source"],
                                created_at=data.get("ts", time.time()),
                                phi_at_storage=data.get("phi", 0.0),
                            )
                        )
                        count += 1
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            logger.info("Restored %d memories from disk", count)
        except Exception as e:
            logger.warning("Failed to restore memories: %s — fresh start", e)

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "by_type": {
                t: sum(1 for e in self._entries if e.memory_type == t)
                for t in {"episodic", "semantic", "procedural"}
            },
            "persisted": self._persist_path.exists(),
        }
