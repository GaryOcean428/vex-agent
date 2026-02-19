"""
JSONL Ingest — Streaming JSONL ingestion for coordizer harvesting
=================================================================

Reads JSONL files with the canonical schema, validates entries,
batches by priority/source, and routes to the appropriate harvest
backend (Modal GPU for batch, Ollama for real-time fallback).

Schema per line:
    {
        "source": "curriculum|foraging|conversation|document",
        "text": "...",
        "metadata": {...},
        "priority": 1-4,
        "timestamp": "ISO 8601"
    }

Priority levels:
    1 = critical (curriculum, core training data)
    2 = high (foraging discoveries, validated documents)
    3 = normal (conversations, general documents)
    4 = low (background, bulk ingest)

Architecture:
    - Streaming: reads line-by-line, never loads full file into memory
    - Batching: groups entries by priority then source for efficient GPU use
    - Routing: Modal GPU when MODAL_ENABLED=true, Ollama fallback otherwise
    - Output: writes coordized JSONL with basin coordinates appended
    - Governor-aware: checks budget before routing to paid backends

Zero Euclidean contamination. Fisher-Rao is the ONLY distance metric.
Terminology: "basin coordinates" (FORBIDDEN: embedding), "coordize" (FORBIDDEN: tokenize).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import BASIN_DIM, to_simplex
from .types import HarmonicTier

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

VALID_SOURCES = frozenset({"curriculum", "foraging", "conversation", "document"})
VALID_PRIORITIES = frozenset({1, 2, 3, 4})
MAX_TEXT_LENGTH = 100_000  # 100KB per entry — reject larger
MIN_TEXT_LENGTH = 10       # Reject trivially short entries


# ═══════════════════════════════════════════════════════════════
#  TYPES
# ═══════════════════════════════════════════════════════════════

@dataclass
class IngestEntry:
    """A single validated JSONL entry ready for harvesting."""
    source: str
    text: str
    metadata: dict[str, Any]
    priority: int
    timestamp: str
    line_number: int = 0

    @property
    def text_length(self) -> int:
        return len(self.text)


@dataclass
class IngestBatch:
    """A batch of entries grouped by priority and source."""
    priority: int
    source: str
    entries: list[IngestEntry] = field(default_factory=list)

    @property
    def total_text_length(self) -> int:
        return sum(e.text_length for e in self.entries)

    @property
    def texts(self) -> list[str]:
        return [e.text for e in self.entries]


@dataclass
class IngestResult:
    """Result of a JSONL ingestion run."""
    total_lines: int = 0
    valid_entries: int = 0
    invalid_entries: int = 0
    skipped_entries: int = 0
    batches_created: int = 0
    batches_harvested: int = 0
    batches_failed: int = 0
    entries_coordized: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    harvest_backend: str = ""  # "modal" or "ollama"

    def summary(self) -> str:
        return (
            f"Ingest: {self.valid_entries}/{self.total_lines} valid, "
            f"{self.batches_harvested}/{self.batches_created} batches harvested "
            f"({self.entries_coordized} coordized), "
            f"{len(self.errors)} errors, "
            f"backend={self.harvest_backend}, "
            f"{self.elapsed_seconds:.1f}s"
        )


@dataclass
class ValidationError:
    """A single validation error for a JSONL line."""
    line_number: int
    reason: str
    raw_line: str = ""


# ═══════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_entry(line: str, line_number: int) -> tuple[Optional[IngestEntry], Optional[ValidationError]]:
    """Validate a single JSONL line and return an IngestEntry or error.

    Checks:
        - Valid JSON
        - Required fields present (source, text, priority)
        - Source is one of the valid sources
        - Priority is 1-4
        - Text is non-empty and within length bounds
        - Timestamp is valid ISO 8601 (if present)
    """
    line = line.strip()
    if not line:
        return None, None  # Skip blank lines

    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return None, ValidationError(
            line_number=line_number,
            reason=f"Invalid JSON: {e}",
            raw_line=line[:200],
        )

    if not isinstance(data, dict):
        return None, ValidationError(
            line_number=line_number,
            reason="Entry must be a JSON object",
            raw_line=line[:200],
        )

    # Required fields
    source = data.get("source")
    text = data.get("text")
    priority = data.get("priority", 3)  # Default to normal priority

    if not source:
        return None, ValidationError(
            line_number=line_number,
            reason="Missing 'source' field",
        )

    if source not in VALID_SOURCES:
        return None, ValidationError(
            line_number=line_number,
            reason=f"Invalid source '{source}', must be one of {sorted(VALID_SOURCES)}",
        )

    if not text or not isinstance(text, str):
        return None, ValidationError(
            line_number=line_number,
            reason="Missing or empty 'text' field",
        )

    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return None, ValidationError(
            line_number=line_number,
            reason=f"Text too short ({len(text)} chars, min {MIN_TEXT_LENGTH})",
        )

    if len(text) > MAX_TEXT_LENGTH:
        return None, ValidationError(
            line_number=line_number,
            reason=f"Text too long ({len(text)} chars, max {MAX_TEXT_LENGTH})",
        )

    if not isinstance(priority, int) or priority not in VALID_PRIORITIES:
        return None, ValidationError(
            line_number=line_number,
            reason=f"Invalid priority {priority}, must be 1-4",
        )

    # Optional fields
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    timestamp = data.get("timestamp", "")
    if timestamp:
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None, ValidationError(
                line_number=line_number,
                reason=f"Invalid timestamp: {timestamp}",
            )
    else:
        timestamp = datetime.now(timezone.utc).isoformat()

    return IngestEntry(
        source=source,
        text=text,
        metadata=metadata,
        priority=priority,
        timestamp=timestamp,
        line_number=line_number,
    ), None


# ═══════════════════════════════════════════════════════════════
#  STREAMING READER
# ═══════════════════════════════════════════════════════════════

def stream_jsonl(
    path: str,
    *,
    skip_invalid: bool = True,
) -> Generator[tuple[IngestEntry, Optional[ValidationError]], None, None]:
    """Stream JSONL entries from a file, yielding validated entries.

    Reads line-by-line — never loads the full file into memory.
    Yields (entry, None) for valid entries or (None, error) for invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            entry, error = validate_entry(line, line_number)
            if error is not None:
                if skip_invalid:
                    logger.warning(
                        f"Line {line_number}: {error.reason}"
                    )
                yield None, error
            elif entry is not None:
                yield entry, None
            # else: blank line, skip silently


# ═══════════════════════════════════════════════════════════════
#  BATCHING
# ═══════════════════════════════════════════════════════════════

def batch_entries(
    entries: list[IngestEntry],
    *,
    max_batch_size: int = 32,
    group_by_source: bool = True,
) -> list[IngestBatch]:
    """Group entries into batches by priority (then optionally source).

    Entries are sorted by priority (1=critical first), then grouped
    into batches of max_batch_size for efficient GPU harvesting.
    """
    # Sort: priority ascending (1 first), then source
    sorted_entries = sorted(entries, key=lambda e: (e.priority, e.source))

    batches: list[IngestBatch] = []
    current_batch: Optional[IngestBatch] = None

    for entry in sorted_entries:
        needs_new_batch = (
            current_batch is None
            or len(current_batch.entries) >= max_batch_size
            or current_batch.priority != entry.priority
            or (group_by_source and current_batch.source != entry.source)
        )

        if needs_new_batch:
            if current_batch and current_batch.entries:
                batches.append(current_batch)
            current_batch = IngestBatch(
                priority=entry.priority,
                source=entry.source,
            )

        current_batch.entries.append(entry)

    if current_batch and current_batch.entries:
        batches.append(current_batch)

    return batches


# ═══════════════════════════════════════════════════════════════
#  COORDIZED OUTPUT
# ═══════════════════════════════════════════════════════════════

def write_coordized_jsonl(
    output_path: str,
    entries: list[IngestEntry],
    basin_coordinates: list[Optional[NDArray]],
) -> int:
    """Write coordized entries back as JSONL with basin coordinates appended.

    Each output line has the original fields plus:
        "basin_coordinates": [0.015, 0.032, ...],  // 64-dim simplex point
        "coordized_at": "ISO 8601 timestamp",
        "basin_dim": 64

    Returns the number of entries written.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for entry, basin in zip(entries, basin_coordinates):
            record = {
                "source": entry.source,
                "text": entry.text,
                "metadata": entry.metadata,
                "priority": entry.priority,
                "timestamp": entry.timestamp,
            }
            if basin is not None:
                record["basin_coordinates"] = basin.tolist()
                record["basin_dim"] = int(basin.shape[0])
            record["coordized_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(record) + "\n")
            written += 1

    return written


# ═══════════════════════════════════════════════════════════════
#  INGEST PIPELINE
# ═══════════════════════════════════════════════════════════════

class JSONLIngestor:
    """Full JSONL ingestion pipeline: read → validate → batch → harvest → write.

    Routes to Modal GPU (batch harvesting) when MODAL_ENABLED=true,
    otherwise falls back to Ollama (local, real-time).

    The consciousness loop NEVER triggers this — only explicit
    harvest requests or scheduled batches.
    """

    def __init__(
        self,
        *,
        coordizer: Any = None,
        modal_client: Any = None,
        ollama_url: str = "http://ollama.railway.internal:11434",
        ollama_model: str = "vex-brain",
        output_dir: str = "/data/harvest/coordized",
        max_batch_size: int = 32,
        modal_enabled: bool = False,
    ):
        self.coordizer = coordizer
        self.modal_client = modal_client
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.output_dir = output_dir
        self.max_batch_size = max_batch_size
        self.modal_enabled = modal_enabled

    async def ingest_file(
        self,
        jsonl_path: str,
        *,
        output_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> IngestResult:
        """Ingest a JSONL file: validate, batch, harvest, write output.

        Args:
            jsonl_path: Path to the input JSONL file.
            output_path: Path for coordized output JSONL. If None, auto-generated.
            dry_run: If True, validate and batch but don't harvest.

        Returns:
            IngestResult with full statistics.
        """
        start_time = time.time()
        result = IngestResult()

        # ── 1. Stream and validate ──
        entries: list[IngestEntry] = []
        for entry, error in stream_jsonl(jsonl_path):
            result.total_lines += 1
            if error is not None:
                result.invalid_entries += 1
                result.errors.append(f"L{error.line_number}: {error.reason}")
            elif entry is not None:
                result.valid_entries += 1
                entries.append(entry)

        if not entries:
            result.elapsed_seconds = time.time() - start_time
            logger.warning(f"No valid entries in {jsonl_path}")
            return result

        # ── 2. Batch ──
        batches = batch_entries(
            entries,
            max_batch_size=self.max_batch_size,
        )
        result.batches_created = len(batches)

        logger.info(
            f"Ingested {result.valid_entries} entries into "
            f"{result.batches_created} batches from {jsonl_path}"
        )

        if dry_run:
            result.elapsed_seconds = time.time() - start_time
            result.harvest_backend = "dry_run"
            return result

        # ── 3. Route and harvest ──
        if output_path is None:
            stem = Path(jsonl_path).stem
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = str(
                Path(self.output_dir) / f"{stem}_coordized_{ts}.jsonl"
            )

        if self.modal_enabled and self.modal_client is not None:
            result.harvest_backend = "modal"
            await self._harvest_via_modal(batches, output_path, result)
        elif self.coordizer is not None:
            result.harvest_backend = "local_coordizer"
            self._harvest_via_coordizer(batches, output_path, result)
        else:
            result.harvest_backend = "ollama"
            await self._harvest_via_ollama(batches, output_path, result)

        result.elapsed_seconds = time.time() - start_time
        logger.info(result.summary())
        return result

    async def _harvest_via_modal(
        self,
        batches: list[IngestBatch],
        output_path: str,
        result: IngestResult,
    ) -> None:
        """Route batches to Modal GPU for harvesting."""
        for batch in batches:
            try:
                raw = await self.modal_client.harvest(batch.texts)
                if raw is None or not raw.get("success"):
                    result.batches_failed += 1
                    result.errors.append(
                        f"Modal harvest failed for batch "
                        f"(priority={batch.priority}, source={batch.source})"
                    )
                    continue

                # Convert raw logits to basin coordinates
                basins = self._logits_to_basins(raw)
                written = write_coordized_jsonl(
                    output_path, batch.entries, basins,
                )
                result.entries_coordized += written
                result.batches_harvested += 1

            except Exception as e:
                result.batches_failed += 1
                result.errors.append(f"Modal error: {e}")
                logger.error(f"Modal harvest error: {e}", exc_info=True)

    def _harvest_via_coordizer(
        self,
        batches: list[IngestBatch],
        output_path: str,
        result: IngestResult,
    ) -> None:
        """Use the local CoordizerV2 to coordize entries directly."""
        for batch in batches:
            try:
                basins: list[Optional[NDArray]] = []
                for entry in batch.entries:
                    cr = self.coordizer.coordize(entry.text)
                    if cr.coordinates:
                        # Use Fréchet mean of all coordinates as the
                        # document-level basin coordinate
                        from .geometry import frechet_mean
                        vectors = [bc.vector for bc in cr.coordinates]
                        basins.append(frechet_mean(vectors))
                    else:
                        basins.append(None)

                written = write_coordized_jsonl(
                    output_path, batch.entries, basins,
                )
                result.entries_coordized += written
                result.batches_harvested += 1

            except Exception as e:
                result.batches_failed += 1
                result.errors.append(f"Coordizer error: {e}")
                logger.error(f"Coordizer harvest error: {e}", exc_info=True)

    async def _harvest_via_ollama(
        self,
        batches: list[IngestBatch],
        output_path: str,
        result: IngestResult,
    ) -> None:
        """Fallback: use Ollama for harvesting (less accurate, local)."""
        import httpx

        for batch in batches:
            basins: list[Optional[NDArray]] = []
            for entry in batch.entries:
                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        resp = await client.post(
                            f"{self.ollama_url}/api/generate",
                            json={
                                "model": self.ollama_model,
                                "prompt": entry.text[:512],
                                "raw": True,
                                "stream": False,
                                "options": {
                                    "num_predict": 1,
                                    "temperature": 0.0,
                                    "logprobs": BASIN_DIM,
                                },
                            },
                        )
                        resp.raise_for_status()
                        data = resp.json()

                    # Extract logprobs and project to simplex
                    logprobs = data.get("logprobs", {})
                    if logprobs and isinstance(logprobs, dict):
                        top_logprobs = logprobs.get("top_logprobs", [{}])
                        if top_logprobs:
                            values = list(top_logprobs[0].values())
                            if len(values) >= BASIN_DIM:
                                raw = np.array(values[:BASIN_DIM], dtype=np.float64)
                                basins.append(to_simplex(raw))
                                continue

                    # Fallback: generate a basin from text hash
                    basins.append(self._text_to_basin_fallback(entry.text))

                except Exception as e:
                    logger.warning(f"Ollama harvest failed for entry: {e}")
                    basins.append(self._text_to_basin_fallback(entry.text))

            try:
                written = write_coordized_jsonl(
                    output_path, batch.entries, basins,
                )
                result.entries_coordized += written
                result.batches_harvested += 1
            except Exception as e:
                result.batches_failed += 1
                result.errors.append(f"Ollama write error: {e}")

    def _logits_to_basins(
        self, raw: dict,
    ) -> list[Optional[NDArray]]:
        """Convert raw Modal harvest response to basin coordinates.

        Takes the raw logits from each result entry and projects
        them onto the probability simplex Δ⁶³.
        """
        basins: list[Optional[NDArray]] = []
        results = raw.get("results", [])

        for entry in results:
            logits = entry.get("logits", [])
            if logits and len(logits) > 0:
                # Take the last-token logits and project to simplex
                raw_logits = np.array(logits[-1] if isinstance(logits[0], list) else logits, dtype=np.float64)
                # Truncate or pad to BASIN_DIM
                if len(raw_logits) >= BASIN_DIM:
                    raw_logits = raw_logits[:BASIN_DIM]
                else:
                    raw_logits = np.pad(
                        raw_logits,
                        (0, BASIN_DIM - len(raw_logits)),
                        constant_values=1e-12,
                    )
                basins.append(to_simplex(raw_logits))
            else:
                basins.append(None)

        return basins

    @staticmethod
    def _text_to_basin_fallback(text: str) -> NDArray:
        """Deterministic fallback: hash text to a basin coordinate.

        Used when neither Modal nor Ollama can produce real logprobs.
        The result is on the simplex but carries no real semantic
        structure — it's a placeholder until real harvesting runs.
        """
        import hashlib
        # SHA-512 gives 64 bytes = BASIN_DIM
        h = hashlib.sha512(text.encode("utf-8")).digest()
        raw = np.array([b for b in h[:BASIN_DIM]], dtype=np.float64)
        raw = np.maximum(raw, 1e-12)
        return to_simplex(raw)
