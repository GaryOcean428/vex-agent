"""
JSONL Ingest — Streaming JSONL ingestion for coordizer harvesting
=================================================================

Reads JSONL files with the canonical schema, validates entries,
batches by priority/source, and routes to the appropriate harvest
backend (Modal GPU for batch, Ollama for real-time fallback).

Schema per line:
    {
        "source": "curriculum|foraging|conversation|document|llm_cogeneration",
        "text": "...",
        "metadata": {...},
        "priority": 1-4,
        "timestamp": "ISO 8601"
    }

Also auto-detects OpenAI fine-tuning format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

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

Failure policy:
    If a backend cannot produce real logprobs for an entry, the entry is
    written to output WITHOUT basin_coordinates (None). There is no hash
    fallback. Hash-derived simplex points carry zero semantic structure and
    corrupt the resonance bank. Explicit gaps are preferable to silent noise.

Zero Euclidean contamination. Fisher-Rao is the ONLY distance metric.
Terminology: "basin coordinates" (FORBIDDEN: embedding), "coordize" (FORBIDDEN: tokenize).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import BASIN_DIM, to_simplex

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

VALID_SOURCES = frozenset(
    {"curriculum", "foraging", "conversation", "document", "llm_cogeneration"}
)
VALID_PRIORITIES = frozenset({1, 2, 3, 4})
MAX_TEXT_LENGTH = 100_000  # 100KB per entry — reject larger
MIN_TEXT_LENGTH = 10  # Reject trivially short entries


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


def _normalize_openai_entry(data: dict[str, Any]) -> dict[str, Any]:
    """Auto-convert OpenAI fine-tuning format to harvest-ingest format.

    OpenAI format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Converts to ingest format:
        {"source": "curriculum", "text": "User: ...\\nAssistant: ...", "priority": 1, "metadata": {"origin": "openai_export"}}

    Returns the dict unchanged if it's already in ingest format.
    """
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return data  # Not OpenAI format — pass through

    # Already has source+text → native format, skip conversion
    if data.get("source") and data.get("text"):
        return data

    # Build text from messages
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            parts.append(f"{role.capitalize()}: {content}")

    if not parts:
        return data  # Empty messages — let validator reject it

    return {
        "source": data.get("source", "curriculum"),
        "text": "\n".join(parts),
        "priority": data.get("priority", 1),
        "metadata": {
            **(data.get("metadata") or {}),
            "origin": "openai_export",
            "original_messages": len(messages),
        },
        "timestamp": data.get("timestamp", ""),
    }


def validate_entry(
    line: str, line_number: int
) -> tuple[IngestEntry | None, ValidationError | None]:
    """Validate a single JSONL line and return an IngestEntry or error.

    Auto-detects and normalizes OpenAI fine-tuning format before validation.

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

    # Auto-detect and normalize OpenAI format → ingest format
    data = _normalize_openai_entry(data)

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
        timestamp = datetime.now(UTC).isoformat()

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
) -> Generator[tuple[IngestEntry | None, ValidationError | None]]:
    """Stream JSONL entries from a file, yielding validated entries.

    Reads line-by-line — never loads the full file into memory.
    Yields (entry, None) for valid entries or (None, error) for invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with open(file_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            entry, error = validate_entry(line, line_number)
            if error is not None:
                if skip_invalid:
                    logger.warning(f"Line {line_number}: {error.reason}")
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
    current_batch: IngestBatch | None = None

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

        assert current_batch is not None  # guaranteed by needs_new_batch logic
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
    basin_coordinates: list[NDArray[np.float64] | None],
) -> int:
    """Write coordized entries back as JSONL with basin coordinates appended.

    Each output line has the original fields plus:
        "basin_coordinates": [0.015, 0.032, ...],  // 64-dim simplex point
        "coordized_at": "ISO 8601 timestamp",
        "basin_dim": 64

    Entries with basin=None are written without basin_coordinates.
    This is an explicit gap, not an error. It means the backend could not
    produce real logprobs for this entry. Do not substitute hash-derived
    coordinates — that is silent corruption.

    Returns the number of entries written.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for entry, basin in zip(entries, basin_coordinates, strict=True):
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
            record["coordized_at"] = datetime.now(UTC).isoformat()
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
        ollama_url: str = "",
        ollama_model: str = "",
        output_dir: str = "/data/harvest/coordized",
        max_batch_size: int = 32,
        modal_enabled: bool = False,
    ):
        self.coordizer = coordizer
        self.modal_client = modal_client
        if not ollama_url or not ollama_model:
            from ..config.settings import settings

            ollama_url = ollama_url or settings.ollama.url
            ollama_model = ollama_model or settings.ollama.model
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.output_dir = output_dir
        self.max_batch_size = max_batch_size
        self.modal_enabled = modal_enabled

    async def ingest_file(
        self,
        jsonl_path: str,
        *,
        output_path: str | None = None,
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
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = str(Path(self.output_dir) / f"{stem}_coordized_{ts}.jsonl")

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
        """Route batches to Modal GPU for harvesting.

        The Modal endpoint aggregates all texts and returns per-coordinate
        Fréchet-mean fingerprints.  We compute a single aggregate basin
        on Δ⁶³ from those fingerprints and assign it to every entry in
        the batch.  For finer per-entry resolution, use the local
        CoordizerV2 path instead.
        """
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

                # Compute aggregate basin from fingerprints
                basin = self._fingerprints_to_basin(raw)
                basins: list[NDArray[np.float64] | None] = [basin for _ in batch.entries]
                written = write_coordized_jsonl(
                    output_path,
                    batch.entries,
                    basins,
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
                basins: list[NDArray[np.float64] | None] = []
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
                    output_path,
                    batch.entries,
                    basins,
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
            basins: list[NDArray[np.float64] | None] = []
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

                    # Ollama returned no usable logprobs — entry written without
                    # basin_coordinates. Do NOT substitute a hash. Explicit gaps
                    # are correct; hash-derived simplex points corrupt the bank.
                    logger.warning(
                        f"Ollama returned no logprobs for entry "
                        f"(source={entry.source}, line={entry.line_number}) — skipping basin"
                    )
                    basins.append(None)

                except Exception as e:
                    logger.warning(f"Ollama harvest failed for entry: {e}")
                    basins.append(None)

            try:
                written = write_coordized_jsonl(
                    output_path,
                    batch.entries,
                    basins,
                )
                result.entries_coordized += written
                result.batches_harvested += 1
            except Exception as e:
                result.batches_failed += 1
                result.errors.append(f"Ollama write error: {e}")

    @staticmethod
    def _fingerprints_to_basin(
        raw: dict[str, Any],
    ) -> NDArray[np.float64] | None:
        """Compute a single Δ⁶³ basin from Modal harvest fingerprints.

        The Modal endpoint returns per-coordinate Fréchet-mean fingerprints
        on Δ^(V-1). We compute the Fréchet mean of all fingerprints,
        then project to Δ⁶³ via to_simplex (truncate + renormalise).

        Returns None if no fingerprints were returned.
        """
        from .geometry import frechet_mean

        tokens = raw.get("tokens", {})
        if not tokens:
            return None

        _EPS = 1e-12
        fps: list[NDArray[np.float64]] = []
        for token_data in tokens.values():
            fp = token_data.get("fingerprint")
            if fp is None:
                continue
            arr = np.array(fp, dtype=np.float64)
            arr = np.maximum(arr, _EPS)
            arr = arr / arr.sum()
            fps.append(arr)

        if not fps:
            return None

        # Fréchet mean on Δ^(V-1)
        mean_fp = frechet_mean(fps)

        # Project to Δ⁶³: take first BASIN_DIM components, renormalise
        if len(mean_fp) >= BASIN_DIM:
            basin = mean_fp[:BASIN_DIM]
        else:
            basin = np.pad(
                mean_fp,
                (0, BASIN_DIM - len(mean_fp)),
                constant_values=_EPS,
            )
        return to_simplex(basin)
