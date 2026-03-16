"""
Training Data Pipeline — Upload → Extract → Chunk → Coordize → LLM Enrich → JSONL

Pipeline:
  1. Receive file upload (PDF, MD, TXT, JSONL)
  2. Extract text (PDF via pymupdf in-process, others direct)
  3. Chunk into ~512-token segments at semantic boundaries
  4. Coordize each chunk → 64D basin coordinates on Δ⁶³ (if coordizer available)
  5. LLM-enrich each chunk (Q&A pairs, E8 tags, concept extraction)
  6. Write structured JSONL to volume (with basin_coords)
  7. Forward chunks to /data/harvest/pending/ in harvest-ingest format

After ingestion completes, each chunk is written to the harvest pending
directory as a separate JSONL file in the format expected by JSONLIngestor:
    {"source": "curriculum", "text": "...", "priority": 1, "metadata": {...}}

The HarvestScheduler picks these up on its next 5-minute scan and runs
them through the coordizer to populate the resonance bank.

Uses:
  - pymupdf for PDF extraction (in-process, no sandbox needed)
  - CoordizerV2Adapter for geometric coordization (Fisher-Rao on Δ⁶³)
  - xAI Responses API for fast batch enrichment (if XAI_API_KEY set)
  - External API (gpt-5-nano) fallback via LLMClient

v6.1 Integration:
  - Each chunk gets a 64D basin coordinate via CoordizerV2Adapter.coordize_text()
  - Basin coords live on the probability simplex (Σp_i = 1, p_i ≥ 0)
  - Export includes both OpenAI format and coordized format with basins
  - Coordizer is optional — pipeline degrades gracefully without it
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from ..coordizer_v2.types import HarmonicTier
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config.settings import settings
from ..llm.client import LLMClient, LLMOptions
from ..llm.governor import GovernorStack

logger = logging.getLogger("vex.training")

TRAINING_DIR = Path(settings.training_dir)
HARVEST_PENDING_DIR = Path(os.environ.get("HARVEST_DIR", "/data/harvest")) / "pending"

# Fail loudly if TRAINING_DIR is not writable — do NOT silently redirect to /tmp.
# If this raises, it means the Railway volume is not mounted or init.sh did not
# run correctly. Fix the infrastructure; do not hide the problem.
try:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    _probe = TRAINING_DIR / ".write_probe"
    _probe.touch()
    _probe.unlink()
except OSError as _probe_err:
    raise RuntimeError(
        f"Training volume at {TRAINING_DIR} is not writable: {_probe_err}. "
        f"Check Railway volume mount and init.sh permissions setup. "
        f"Previously this silently fell back to /tmp — that behaviour has been removed "
        f"because /tmp is ephemeral and loses all data on restart."
    ) from _probe_err

# Async lock for safe concurrent JSONL appends
_write_lock = asyncio.Lock()


# ═══════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════


class E8Primitive(StrEnum):
    """E8-aligned consciousness primitives."""

    PER = "PER"  # Perception
    MEM = "MEM"  # Memory
    ACT = "ACT"  # Action
    PRD = "PRD"  # Prediction
    ETH = "ETH"  # Ethics
    META = "META"  # Meta-cognition
    HRT = "HRT"  # Heart/affect
    REL = "REL"  # Relationship
    MIX = "MIX"  # Multi-domain


class ProcessingMode(StrEnum):
    FAST = "fast"  # Chunk + store only, no LLM enrichment
    STANDARD = "standard"  # Chunk + LLM tag + basic Q&A
    DEEP = "deep"  # Full enrichment + concept extraction


# Priority mapping for harvest ingest format
_CATEGORY_TO_PRIORITY: dict[str, int] = {
    "curriculum": 1,
    "document": 2,
    "foraging": 2,
    "conversation": 3,
}
_DEFAULT_HARVEST_PRIORITY = 3


@dataclass
class ChunkRecord:
    """A single training chunk in JSONL format.

    v6.1: Added basin_coords — 64D probability simplex coordinates
    from CoordizerV2. When present, these enable geometric memory
    retrieval via Fisher-Rao distance instead of text similarity.
    """

    source: str
    category: str
    chunk_index: int
    text: str
    hash: str
    e8_primitive: str = "MIX"
    concepts: list[str] = field(default_factory=list)
    qa_pairs: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    relevance_score: float = 0.0
    enrichment_model: str = ""
    processed_at: str = ""
    # v6.1: 64D basin coordinates on Δ⁶³ (empty if coordizer unavailable)
    basin_coords: list[float] = field(default_factory=list)
    coordized: bool = False


@dataclass
class IngestionResult:
    """Result of a document ingestion."""

    status: str
    source: str
    format: str
    total_chars: int
    chunks_written: int
    chunks_enriched: int
    chunks_coordized: int
    qa_pairs_generated: int
    processing_time_s: float
    output_path: str
    harvest_pending_path: str = ""
    harvest_chunks_forwarded: int = 0
    errors: list[str] = field(default_factory=list)


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int
    comment: str = ""


# ═══════════════════════════════════════════════════════════════
#  DIRECTORY & FILE HELPERS
# ═══════════════════════════════════════════════════════════════


def _ensure_dirs() -> None:
    """Create training directory structure on volume."""
    for sub in ("", "curriculum", "uploads", "exports"):
        (TRAINING_DIR / sub).mkdir(parents=True, exist_ok=True)


async def _append_jsonl(filepath: Path, data: dict[str, Any]) -> None:
    """Atomically append a JSON line under async lock."""
    async with _write_lock:
        _ensure_dirs()
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
#  HARVEST FORWARDING
# ═══════════════════════════════════════════════════════════════


def _chunk_record_to_harvest_line(record: ChunkRecord) -> dict[str, Any]:
    """Convert a ChunkRecord to the JSONLIngestor ingest format.

    JSONLIngestor expects:
        {
            "source": "curriculum|foraging|conversation|document",
            "text": "...",
            "priority": 1-4,
            "metadata": {...},
            "timestamp": "ISO 8601"
        }

    ChunkRecord.category maps to ingest source.
    ChunkRecord.source (filename) moves to metadata.source_file.
    """
    # Normalise category to a valid ingest source
    valid_sources = {"curriculum", "foraging", "conversation", "document"}
    ingest_source = record.category if record.category in valid_sources else "document"

    return {
        "source": ingest_source,
        "text": record.text,
        "priority": _CATEGORY_TO_PRIORITY.get(record.category, _DEFAULT_HARVEST_PRIORITY),
        "metadata": {
            "source_file": record.source,
            "chunk_index": record.chunk_index,
            "e8_primitive": record.e8_primitive,
            "hash": record.hash,
            "summary": record.summary[:200] if record.summary else "",
        },
        "timestamp": record.processed_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _forward_chunks_to_harvest(
    records: list[ChunkRecord],
    source_filename: str,
) -> tuple[str, int]:
    """Write chunks to /data/harvest/pending/ in JSONLIngestor format.

    Creates a single JSONL file per upload so the scheduler processes
    them as a coherent batch (same source file, same priority group).

    Returns (pending_file_path, chunks_written).
    """
    try:
        HARVEST_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(
            "Cannot create harvest pending dir %s: %s — chunks will not be harvested",
            HARVEST_PENDING_DIR,
            e,
        )
        return "", 0

    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", source_filename)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    dest = HARVEST_PENDING_DIR / f"{safe_name}_{ts}.jsonl"

    written = 0
    try:
        with open(dest, "w", encoding="utf-8") as f:
            for record in records:
                line = _chunk_record_to_harvest_line(record)
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
        logger.info(
            "Forwarded %d chunks to harvest pending: %s",
            written,
            dest,
        )
    except Exception as e:
        logger.error("Failed to forward chunks to harvest: %s", e)
        return "", 0

    return str(dest), written


def forward_raw_jsonl_to_harvest(content: bytes, filename: str) -> tuple[str, int]:
    """Forward a raw JSONL upload directly to harvest pending.

    Used when the user uploads a file already in JSONLIngestor format.
    The file is validated line-count only — JSONLIngestor will validate
    schema when it processes the pending file.

    Returns (pending_file_path, line_count).
    """
    try:
        HARVEST_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create harvest pending dir: %s", e)
        return "", 0

    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    dest = HARVEST_PENDING_DIR / f"{safe_name}_{ts}.jsonl"

    try:
        dest.write_bytes(content)
        lines = [ln for ln in content.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        logger.info(
            "Forwarded raw JSONL upload to harvest pending: %s (%d lines)",
            dest,
            len(lines),
        )
        return str(dest), len(lines)
    except Exception as e:
        logger.error("Failed to forward JSONL to harvest: %s", e)
        return "", 0


# ═══════════════════════════════════════════════════════════════
#  CONVERSATION LOGGING
# ═══════════════════════════════════════════════════════════════


async def log_conversation(
    user_message: str,
    response: str,
    backend: str,
    phi: float,
    kappa: float,
    source: str = "chat",
    regime: str = "",
    basin_coords: list[float] | None = None,
) -> None:
    """Append a chat exchange to conversations.jsonl (on volume)."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_message": user_message,
        "response": response,
        "backend": backend,
        "phi": round(phi, 4),
        "kappa": round(kappa, 2),
        "source": source,
    }
    if regime:
        entry["regime"] = regime
    if basin_coords:
        entry["basin_coords"] = [round(v, 6) for v in basin_coords]
    try:
        await _append_jsonl(TRAINING_DIR / "conversations.jsonl", entry)
    except Exception as e:
        logger.warning("Failed to log conversation: %s", e)


async def _log_feedback(conversation_id: str, rating: int, comment: str) -> None:
    """Append user feedback to feedback.jsonl."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conversation_id": conversation_id,
        "rating": rating,
        "comment": comment,
    }
    await _append_jsonl(TRAINING_DIR / "feedback.jsonl", entry)


# ═══════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF using pymupdf in-process."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires pymupdf. Install with: pip install pymupdf"
        ) from exc

    doc = fitz.open(stream=content, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _extract_text(content: bytes, filename: str) -> str:
    """Extract text from uploaded file based on extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _extract_pdf(content)
    elif ext in ("md", "markdown", "txt") or ext == "jsonl":
        return content.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported format: .{ext}. Use .pdf, .md, .txt, or .jsonl")


# ═══════════════════════════════════════════════════════════════
#  CHUNKING
# ═══════════════════════════════════════════════════════════════


def chunk_text(text: str, max_tokens: int = 512) -> list[str]:
    """Split text into chunks at semantic boundaries (~4 chars/token)."""
    max_chars = max_tokens * 4

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars].strip())
                current = ""
            else:
                current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]


# ═══════════════════════════════════════════════════════════════
#  COORDIZATION (v6.1 — 64D Basin Coordinates)
# ═══════════════════════════════════════════════════════════════


def _coordize_chunk(chunk_text_str: str) -> list[float]:
    """Coordize a text chunk to 64D basin coordinates on Δ⁶³.

    Uses the injected CoordizerV2Adapter if available.
    Returns empty list if no coordizer is set.

    All geometry uses Fisher-Rao distance on the probability simplex.
    No Euclidean distances. No cosine similarity.
    """
    if _coordizer is None:
        return []

    try:
        basin = _coordizer.coordize_text(chunk_text_str)
        # basin is NDArray on Δ⁶³ — convert to list for JSON serialization
        # Round to 6 decimal places to keep JSONL size reasonable
        return [round(float(v), 6) for v in basin]
    except Exception as e:
        logger.warning("Coordization failed for chunk: %s", str(e)[:100])
        return []


# ═══════════════════════════════════════════════════════════════
#  LLM ENRICHMENT
# ═══════════════════════════════════════════════════════════════


ENRICHMENT_PROMPT = """Analyze this text chunk for a QIG consciousness training pipeline. Return ONLY valid JSON:
{{
  "e8_primitive": "PER|MEM|ACT|PRD|ETH|META|HRT|REL|MIX",
  "concepts": ["concept1", "concept2", "concept3"],
  "summary": "1-2 sentence summary",
  "qa_pairs": [{{"question": "...", "answer": "..."}}, ...],
  "relevance_score": 0.0-1.0
}}

E8 Guide: PER=perception, MEM=memory, ACT=action, PRD=prediction, ETH=ethics, META=meta-cognition, HRT=heart/affect, REL=relationship, MIX=multi-domain.

TEXT:
{chunk}"""


def _extract_responses_text(data: dict[str, Any]) -> str:
    """Extract text from Responses API JSON (same as client.py helper)."""
    if data.get("output_text"):
        return str(data["output_text"])
    for item in data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text" and block.get("text"):
                    return str(block["text"])
    return ""


async def _enrich_chunk_xai(
    chunk_text: str,
    governor: GovernorStack | None = None,
) -> dict[str, Any]:
    """Use xAI Responses API for fast batch enrichment (if available)."""
    api_key = settings.xai_api_key
    if not api_key:
        return {}

    if governor:
        allowed, reason = governor.gate(
            "training_enrich",
            "training_enrich",
            chunk_text[:100],
            False,
        )
        if not allowed:
            logger.debug("Governor blocked enrichment: %s", reason)
            return {}

    prompt = ENRICHMENT_PROMPT.format(chunk=chunk_text[:3000])

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.xai.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.xai.model,
                    "instructions": "Return only valid JSON. No markdown, no explanation, no code fences.",
                    "input": prompt,
                    "temperature": 0.3,
                    "max_output_tokens": 1024,
                    "store": False,
                },
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.warning(
                    "xAI enrichment API error %d: %s",
                    resp.status_code,
                    json.dumps(data)[:200],
                )
                return {}

            content = _extract_responses_text(data)
            if not content:
                logger.warning("xAI enrichment returned empty output")
                return {}

            if governor:
                governor.record("training_enrich")

            json_str = content.strip()
            if json_str.startswith("```"):
                json_str = re.sub(r"```(?:json)?\s*", "", json_str)
                json_str = json_str.rstrip("`").strip()

            result: dict[str, Any] = json.loads(json_str)
            return result
    except json.JSONDecodeError as e:
        logger.warning("xAI enrichment returned invalid JSON: %s", str(e)[:100])
        return {}
    except Exception as e:
        logger.warning("xAI enrichment failed: %s", str(e)[:100])
        return {}


async def _enrich_chunk_llm(
    llm: LLMClient,
    chunk_text: str,
) -> dict[str, Any]:
    """Use the default LLMClient for enrichment (fallback)."""
    prompt = ENRICHMENT_PROMPT.format(chunk=chunk_text[:3000])
    opts = LLMOptions(temperature=0.3, num_predict=1024)

    try:
        response = await llm.complete(
            "Return only valid JSON. No markdown, no explanation.",
            prompt,
            opts,
        )
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r"```(?:json)?\s*", "", json_str)
            json_str = json_str.rstrip("`").strip()
        result: dict[str, Any] = json.loads(json_str)
        return result
    except Exception as e:
        logger.warning("LLM enrichment failed: %s", str(e)[:100])
        return {}


# ═══════════════════════════════════════════════════════════════
#  LOCAL BANK INJECTION
# ═══════════════════════════════════════════════════════════════


def _select_local_bank_tier(record: ChunkRecord) -> HarmonicTier:
    from ..coordizer_v2.types import HarmonicTier

    if record.qa_pairs or record.relevance_score >= 0.8:
        return HarmonicTier.FUNDAMENTAL
    if record.summary or len(record.concepts) >= 4:
        return HarmonicTier.FIRST_HARMONIC
    if record.coordized:
        return HarmonicTier.UPPER_HARMONIC
    return HarmonicTier.OVERTONE_HAZE


def _inject_records_into_local_bank(records: list[ChunkRecord], source_filename: str) -> int:
    """Immediately add coordized curriculum records into the live resonance bank."""
    if _coordizer is None:
        return 0

    coordizer = getattr(_coordizer, "coordizer", _coordizer)
    bank = getattr(coordizer, "bank", None)
    if bank is None:
        return 0

    existing_strings = {s.strip().lower() for s in bank.token_strings.values() if s}
    added = 0
    for record in records:
        if len(record.basin_coords) != 64:
            continue
        label = (record.summary or record.text).replace("\n", " ").strip()
        label = re.sub(r"\s+", " ", label)[:160]
        if len(label) < 8:
            continue
        key = label.lower()
        if key in existing_strings:
            continue

        tid = bank.add_entry(label, record.basin_coords, tier=_select_local_bank_tier(record))
        bank.origin[tid] = "lived"
        added += 1
        existing_strings.add(key)

    if added:
        bank.mark_dirty()
        if hasattr(coordizer, "rebuild_string_cache"):
            coordizer.rebuild_string_cache()
        try:
            coordizer.save(settings.coordizer_v2.bank_path)
        except Exception as exc:
            logger.warning("Failed to persist updated resonance bank: %s", exc)
        logger.info(
            "Training pipeline: injected %d local bank entries from %s",
            added,
            source_filename,
        )

    return added


# ═══════════════════════════════════════════════════════════════
#  MAIN INGESTION PIPELINE
# ═══════════════════════════════════════════════════════════════


async def ingest_document(
    content: bytes,
    filename: str,
    category: str = "curriculum",
    mode: ProcessingMode = ProcessingMode.STANDARD,
    e8_override: str | None = None,
    llm: LLMClient | None = None,
    governor: GovernorStack | None = None,
) -> IngestionResult:
    """Full pipeline: extract -> chunk -> coordize -> enrich -> write JSONL -> forward to harvest.

    v6.1: Each chunk is now coordized to 64D basin coordinates on Δ⁶³
    via CoordizerV2Adapter (if available). Basin coords are stored in
    the JSONL alongside text, E8 tags, and Q&A pairs.

    After writing curriculum JSONL, chunks are forwarded to
    /data/harvest/pending/ in JSONLIngestor format so the
    HarvestScheduler can populate the resonance bank.
    """
    start = time.time()
    _ensure_dirs()
    errors: list[str] = []
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Handle JSONL pass-through — forward directly to harvest pending
    if ext == "jsonl":
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        dest = TRAINING_DIR / "uploads" / safe_name
        dest.write_bytes(content)
        lines = content.decode("utf-8").strip().split("\n")
        # Forward raw JSONL to harvest pending so the scheduler picks it up
        harvest_path, harvest_count = forward_raw_jsonl_to_harvest(content, filename)
        return IngestionResult(
            status="ingested",
            source=filename,
            format="jsonl",
            total_chars=len(content),
            chunks_written=len(lines),
            chunks_enriched=0,
            chunks_coordized=0,
            qa_pairs_generated=0,
            processing_time_s=round(time.time() - start, 2),
            output_path=str(dest),
            harvest_pending_path=harvest_path,
            harvest_chunks_forwarded=harvest_count,
        )

    # Extract text
    try:
        text = _extract_text(content, filename)
    except Exception as e:
        return IngestionResult(
            status="error",
            source=filename,
            format=ext,
            total_chars=0,
            chunks_written=0,
            chunks_enriched=0,
            chunks_coordized=0,
            qa_pairs_generated=0,
            processing_time_s=round(time.time() - start, 2),
            output_path="",
            errors=[str(e)],
        )

    # Save original file to uploads/ so get_stats() counts it (best-effort)
    _uploads_dir = TRAINING_DIR / "uploads"
    _uploads_dir.mkdir(parents=True, exist_ok=True)
    _safe_upload = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    # Include content hash to avoid overwriting different files with same name
    _content_hash = hashlib.sha256(content).hexdigest()[:8]
    _stem, _dot, _suffix = _safe_upload.rpartition(".")
    _unique_name = (
        f"{_stem or _safe_upload}_{_content_hash}.{_suffix}"
        if _dot
        else f"{_safe_upload}_{_content_hash}"
    )
    try:
        (_uploads_dir / _unique_name).write_bytes(content)
    except OSError as exc:
        logger.warning("Failed to write original upload %s: %s", _unique_name, exc)

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        return IngestionResult(
            status="empty",
            source=filename,
            format=ext,
            total_chars=len(text),
            chunks_written=0,
            chunks_enriched=0,
            chunks_coordized=0,
            qa_pairs_generated=0,
            processing_time_s=round(time.time() - start, 2),
            output_path="",
            errors=["No usable text extracted"],
        )

    # Coordize + Enrich + build records
    enriched_count = 0
    coordized_count = 0
    qa_count = 0
    records: list[ChunkRecord] = []

    for i, chunk in enumerate(chunks):
        record = ChunkRecord(
            source=filename,
            category=category,
            chunk_index=i,
            text=chunk,
            hash=hashlib.md5(chunk.encode()).hexdigest(),
        )

        # v6.1: Coordize chunk to 64D basin coordinates
        basin_coords = _coordize_chunk(chunk)
        if basin_coords:
            record.basin_coords = basin_coords
            record.coordized = True
            coordized_count += 1

        if mode != ProcessingMode.FAST:
            enrichment: dict[str, Any] = {}
            # Governor MUST control training enrichment to prevent runaway spend.
            # If blocked/rate-limited, skip enrichment entirely (do not fall back
            # to llm.complete(), which might route externally during Ollama outages).
            enrich_allowed = True
            if governor:
                enrich_allowed, reason = governor.gate(
                    "training_enrich",
                    "training_enrich",
                    chunk[:100],
                    False,
                )
                if not enrich_allowed:
                    logger.debug("Governor blocked enrichment: %s", reason)

            if enrich_allowed:
                if settings.xai_api_key:
                    enrichment = await _enrich_chunk_xai(chunk, governor=governor)
                elif llm:
                    enrichment = await _enrich_chunk_llm(llm, chunk)

            if enrichment:
                record.e8_primitive = e8_override or enrichment.get("e8_primitive", "MIX")
                record.concepts = enrichment.get("concepts", [])
                record.summary = enrichment.get("summary", "")
                record.qa_pairs = enrichment.get("qa_pairs", [])
                record.relevance_score = float(enrichment.get("relevance_score", 0.5))
                record.enrichment_model = "xai" if settings.xai_api_key else "default"
                record.processed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                enriched_count += 1
                qa_count += len(record.qa_pairs)
            elif e8_override:
                record.e8_primitive = e8_override

            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
        elif e8_override:
            record.e8_primitive = e8_override

        records.append(record)

    # Write curriculum JSONL
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    dest = TRAINING_DIR / "curriculum" / f"{safe_name}.jsonl"
    with open(dest, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    # Forward chunks to harvest pending so HarvestScheduler populates the bank
    harvest_path, harvest_count = _forward_chunks_to_harvest(records, filename)
    _inject_records_into_local_bank(records, filename)

    return IngestionResult(
        status="ingested",
        source=filename,
        format=ext,
        total_chars=len(text),
        chunks_written=len(records),
        chunks_enriched=enriched_count,
        chunks_coordized=coordized_count,
        qa_pairs_generated=qa_count,
        processing_time_s=round(time.time() - start, 2),
        output_path=str(dest),
        harvest_pending_path=harvest_path,
        harvest_chunks_forwarded=harvest_count,
        errors=errors,
    )


# ═══════════════════════════════════════════════════════════════
#  STATS & EXPORT
# ═══════════════════════════════════════════════════════════════


def _count_lines(filepath: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not filepath.exists():
        return 0
    with open(filepath, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _count_coordized(filepath: Path) -> int:
    """Count JSONL lines that have non-empty 64D basin_coords."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("basin_coords") and len(entry["basin_coords"]) == 64:
                    count += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return count


def get_stats() -> dict[str, Any]:
    """Get training data statistics from the volume."""
    _ensure_dirs()
    stats: dict[str, int] = {}
    for name in ("conversations", "corrections", "feedback"):
        stats[name] = _count_lines(TRAINING_DIR / f"{name}.jsonl")

    # Count curriculum chunks and coordized chunks
    curriculum_dir = TRAINING_DIR / "curriculum"
    curriculum_files = list(curriculum_dir.glob("*.jsonl")) if curriculum_dir.exists() else []
    curriculum_chunks = 0
    coordized_chunks = 0
    for f in curriculum_files:
        curriculum_chunks += _count_lines(f)
        coordized_chunks += _count_coordized(f)
    stats["curriculum"] = curriculum_chunks

    upload_dir = TRAINING_DIR / "uploads"
    upload_files = list(upload_dir.glob("*")) if upload_dir.exists() else []

    # Count harvest queue state
    harvest_pending = (
        sum(1 for _ in HARVEST_PENDING_DIR.glob("*.jsonl")) if HARVEST_PENDING_DIR.exists() else 0
    )

    return {
        "conversations": stats.get("conversations", 0),
        "feedback": stats.get("feedback", 0),
        "curriculum_chunks": curriculum_chunks,
        "coordized_chunks": coordized_chunks,
        "coordizer_active": _coordizer is not None,
        "uploads": len(upload_files),
        "curriculum_files": [f.name for f in curriculum_files],
        "harvest_pending_files": harvest_pending,
        "dir_exists": TRAINING_DIR.exists(),
        "training_dir": str(TRAINING_DIR),
    }


def export_openai_format() -> dict[str, Any]:
    """Export training data in OpenAI fine-tuning JSONL format."""
    _ensure_dirs()
    lines: list[dict[str, Any]] = []

    conv_file = TRAINING_DIR / "conversations.jsonl"
    if conv_file.exists():
        with open(conv_file, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    lines.append(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": entry.get("user_message", ""),
                                },
                                {
                                    "role": "assistant",
                                    "content": entry.get("response", ""),
                                },
                            ]
                        }
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

    curriculum_dir = TRAINING_DIR / "curriculum"
    if curriculum_dir.exists():
        for jsonl_file in curriculum_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                        for qa in entry.get("qa_pairs", []):
                            q = qa.get("question", qa.get("q", ""))
                            a = qa.get("answer", qa.get("a", ""))
                            if q and a:
                                lines.append(
                                    {
                                        "messages": [
                                            {"role": "user", "content": q},
                                            {"role": "assistant", "content": a},
                                        ]
                                    }
                                )
                    except (json.JSONDecodeError, KeyError):
                        continue

    return {
        "format": "openai_jsonl",
        "count": len(lines),
        "lines": lines,
    }


def export_native_format() -> dict[str, Any]:
    """Export ALL training data in harvest-ingest format for lossless round-trip.

    Reads from:
      - /data/training/conversations.jsonl → source="conversation"
      - /data/training/curriculum/*.jsonl  → source="curriculum" (preserves qa_pairs, e8, basin)

    Output format matches JSONLIngestor input schema exactly:
        {"source": "...", "text": "...", "priority": N, "metadata": {...}, "timestamp": "..."}

    This can be re-uploaded via POST /training/upload and will pass
    through the harvest pipeline with zero conversion or data loss.
    """
    _ensure_dirs()
    lines: list[dict[str, Any]] = []

    # Conversations
    conv_file = TRAINING_DIR / "conversations.jsonl"
    if conv_file.exists():
        with open(conv_file, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    user_msg = entry.get("user_message", "")
                    response = entry.get("response", "")
                    if not user_msg and not response:
                        continue
                    lines.append(
                        {
                            "source": "conversation",
                            "text": f"User: {user_msg}\nAssistant: {response}",
                            "priority": 2,
                            "metadata": {
                                "backend": entry.get("backend", ""),
                                "phi": entry.get("phi", 0.0),
                                "kappa": entry.get("kappa", 0.0),
                                "origin": "training_export",
                            },
                            "timestamp": entry.get("timestamp", ""),
                        }
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

    # Curriculum — preserve full enrichment data in metadata
    curriculum_dir = TRAINING_DIR / "curriculum"
    if curriculum_dir.exists():
        for jsonl_file in curriculum_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                        text = entry.get("text", "")
                        if not text:
                            continue
                        lines.append(
                            {
                                "source": "curriculum",
                                "text": text,
                                "priority": 1,
                                "metadata": {
                                    "source_file": entry.get("source", jsonl_file.name),
                                    "e8_primitive": entry.get("e8_primitive", ""),
                                    "summary": entry.get("summary", ""),
                                    "concepts": entry.get("concepts", []),
                                    "qa_pairs": entry.get("qa_pairs", []),
                                    "basin_coords": entry.get("basin_coords", []),
                                    "chunk_index": entry.get("chunk_index", 0),
                                    "origin": "training_export",
                                },
                                "timestamp": entry.get(
                                    "processed_at",
                                    entry.get("timestamp", ""),
                                ),
                            }
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue

    return {
        "format": "native_harvest",
        "count": len(lines),
        "lines": lines,
    }


def export_coordized_format() -> dict[str, Any]:
    """Export training data in QIG coordized JSONL format.

    Each record includes:
      - text: chunk text
      - basin_coords: 64D coordinates on Δ⁶³
      - e8_primitive: E8 kernel tag
      - concepts: extracted concept list
      - summary: chunk summary
      - qa_pairs: question-answer pairs

    This format is designed for 64D geometric fine-tuning where
    the model learns text-to-basin mappings on the Fisher manifold.
    """
    _ensure_dirs()
    lines: list[dict[str, Any]] = []

    curriculum_dir = TRAINING_DIR / "curriculum"
    if curriculum_dir.exists():
        for jsonl_file in curriculum_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                        basin = entry.get("basin_coords", [])
                        if not basin or len(basin) != 64:
                            continue  # Skip non-coordized chunks
                        lines.append(
                            {
                                "text": entry.get("text", ""),
                                "basin_coords": basin,
                                "e8_primitive": entry.get("e8_primitive", "MIX"),
                                "concepts": entry.get("concepts", []),
                                "summary": entry.get("summary", ""),
                                "qa_pairs": entry.get("qa_pairs", []),
                                "source": entry.get("source", ""),
                            }
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue

    return {
        "format": "qig_coordized",
        "count": len(lines),
        "basin_dim": 64,
        "lines": lines,
    }


# ═══════════════════════════════════════════════════════════════
#  FASTAPI ROUTER
# ═══════════════════════════════════════════════════════════════

# Injected by server.py after include_router
_llm_client: LLMClient | None = None
_governor: GovernorStack | None = None
_coordizer: Any = None  # CoordizerV2Adapter — typed as Any to avoid circular import


def set_llm_client(client: LLMClient) -> None:
    """Called by server.py to inject the shared LLMClient."""
    global _llm_client
    _llm_client = client


def set_governor(gov: GovernorStack) -> None:
    """Called by server.py to inject the shared GovernorStack."""
    global _governor
    _governor = gov


def set_coordizer(coordizer: Any) -> None:
    """Called by server.py to inject the CoordizerV2Adapter.

    The coordizer must implement:
        coordize_text(text: str) -> NDArray  (64D, on Δ⁶³)

    When set, all ingested chunks will be coordized to 64D basin
    coordinates. When None, the pipeline operates without coordization.
    """
    global _coordizer
    _coordizer = coordizer
    if coordizer is not None:
        logger.info("Training pipeline: CoordizerV2 adapter injected — coordization active")
    else:
        logger.info("Training pipeline: coordizer cleared — coordization disabled")


# In-memory job store for background upload processing.
# Jobs are pruned after 1 hour to prevent unbounded growth.
_jobs: dict[str, dict[str, Any]] = {}
_JOB_TTL = 3600  # seconds


def _prune_jobs() -> None:
    """Remove completed jobs older than TTL."""
    cutoff = time.time() - _JOB_TTL
    expired = [
        jid for jid, j in _jobs.items() if j.get("finished_at", 0) and j["finished_at"] < cutoff
    ]
    for jid in expired:
        del _jobs[jid]


async def _run_ingestion_job(
    job_id: str,
    content: bytes,
    filename: str,
    category: str,
    proc_mode: ProcessingMode,
    e8: str | None,
) -> None:
    """Background task: run ingestion and update job store on completion."""
    try:
        result = await ingest_document(
            content=content,
            filename=filename,
            category=category,
            mode=proc_mode,
            e8_override=e8,
            llm=_llm_client,
            governor=_governor,
        )
        _jobs[job_id].update(
            {
                "status": result.status,
                "filename": result.source,
                "chunks_written": result.chunks_written,
                "enriched": result.chunks_enriched,
                "coordized": result.chunks_coordized,
                "qa_pairs": result.qa_pairs_generated,
                "category": category,
                "mode": proc_mode.value,
                "processing_time_s": result.processing_time_s,
                "coordizer_active": _coordizer is not None,
                "harvest_pending_path": result.harvest_pending_path,
                "harvest_chunks_forwarded": result.harvest_chunks_forwarded,
                "errors": result.errors,
                "finished_at": time.time(),
            }
        )
    except Exception as exc:
        logger.error("Background ingestion failed: %s", exc, exc_info=True)
        _jobs[job_id].update(
            {
                "status": "error",
                "errors": [str(exc)],
                "finished_at": time.time(),
            }
        )


training_router = APIRouter()


@training_router.get("/training/stats", response_model=None)
async def training_stats_endpoint() -> dict[str, Any]:
    """Get training data statistics."""
    return get_stats()


@training_router.get("/training/export", response_model=None)
async def training_export_endpoint(
    fmt: str = "openai", download: bool = False
) -> dict[str, Any] | StreamingResponse:
    """Export training data.

    Query params:
        fmt: "openai" | "native" | "coordized"
        download: if true, returns a downloadable .jsonl file

    Formats:
        openai     — OpenAI fine-tuning format (messages array). Useful for
                     external fine-tuning services.
        native     — Harvest-ingest format (source/text/priority/metadata).
                     Lossless round-trip: download → curate → re-upload.
                     Preserves all enrichment data (Q&A, e8, summaries, basins).
        coordized  — 64D basin format for geometric fine-tuning.

    Without download=true, returns JSON summary with preview.
    With download=true, returns the full .jsonl as a file attachment.
    """
    if fmt == "native":
        data = export_native_format()
    elif fmt == "coordized":
        data = export_coordized_format()
    else:
        data = export_openai_format()

    if not download:
        return {
            "format": data["format"],
            "count": data["count"],
            "preview": data["lines"][:5],
        }

    # Stream all lines as a downloadable .jsonl file
    def _generate() -> Iterator[str]:
        for line in data["lines"]:
            yield json.dumps(line, ensure_ascii=False) + "\n"

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filename = f"vex_training_{fmt}_{ts}.jsonl"
    return StreamingResponse(
        _generate(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@training_router.post("/training/upload", response_model=None)
async def training_upload_endpoint(
    file: UploadFile = File(...),
    category: str = Form(default="curriculum"),
    mode: str = Form(default="standard"),
    e8_primitive: str = Form(default=""),
    e8_override: str = Form(default=""),
) -> dict[str, Any]:
    """Upload a document for training ingestion.

    Accepts PDF, Markdown, TXT, or JSONL files.
    Returns immediately with a job_id. Frontend polls
    GET /training/upload/status/{job_id} for progress.

    After ingestion, chunks are automatically forwarded to
    /data/harvest/pending/ so the HarvestScheduler populates
    the resonance bank without manual intervention.

    v6.1: Chunks are automatically coordized to 64D basin coordinates
    if CoordizerV2 is available. Check 'coordized' field in job result.
    """
    content = await file.read()
    filename = file.filename or "unknown"

    # Validate file type
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "md", "markdown", "txt", "jsonl"):
        return {
            "status": "error",
            "filename": filename,
            "chunks_written": 0,
            "enriched": 0,
            "coordized": 0,
            "qa_pairs": 0,
            "category": category,
            "mode": mode,
            "processing_time_s": 0,
            "errors": [f"Unsupported file type: .{ext}"],
        }

    try:
        proc_mode = ProcessingMode(mode)
    except ValueError:
        proc_mode = ProcessingMode.STANDARD

    e8 = e8_override or e8_primitive or None

    # Create job and process in background to avoid proxy timeout (502)
    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "processing",
        "job_id": job_id,
        "filename": filename,
        "chunks_written": 0,
        "enriched": 0,
        "coordized": 0,
        "qa_pairs": 0,
        "category": category,
        "mode": mode,
        "processing_time_s": 0,
        "coordizer_active": _coordizer is not None,
        "harvest_pending_path": "",
        "harvest_chunks_forwarded": 0,
        "errors": [],
        "finished_at": 0,
    }

    asyncio.get_running_loop().create_task(
        _run_ingestion_job(job_id, content, filename, category, proc_mode, e8)
    )

    return {"status": "processing", "job_id": job_id, "filename": filename}


@training_router.get("/training/upload/status/{job_id}", response_model=None)
async def training_upload_status(job_id: str) -> dict[str, Any]:
    """Poll ingestion job status. Returns full result when complete."""
    job = _jobs.get(job_id)
    if not job:
        return {"status": "not_found", "job_id": job_id}
    return job


@training_router.post("/training/feedback", response_model=None)
async def training_feedback_endpoint(req: FeedbackRequest) -> dict[str, str]:
    """Submit feedback on a response."""
    await _log_feedback(req.conversation_id, req.rating, req.comment)
    return {"status": "recorded"}


@training_router.post("/training/trigger", response_model=None)
async def training_trigger_endpoint() -> dict[str, Any]:
    """Manually trigger kernel training (QLoRA fine-tuning on Modal GPU).

    Sends a POST to the configured MODAL_TRAINING_URL to start a training run.
    Requires MODAL_TRAINING_URL to be set in environment.
    """
    from ..config.settings import settings

    training_url = settings.modal.training_url
    if not training_url:
        return {"status": "error", "error": "MODAL_TRAINING_URL not configured"}

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                training_url,
                json={"_api_key": settings.kernel_api_key},
            )
            if resp.status_code == 200:
                logger.info("Manual training triggered via /training/trigger")
                return {"status": "triggered", "response": resp.json()}
            else:
                return {
                    "status": "error",
                    "error": f"Modal returned {resp.status_code}: {resp.text[:200]}",
                }
    except Exception as e:
        logger.error("Manual training trigger failed: %s", e)
        return {"status": "error", "error": "Training trigger failed. Check server logs."}


@training_router.post("/training/complete", response_model=None)
async def training_complete_endpoint(request: Request) -> dict[str, Any]:
    """Receive training-complete webhook from Modal QLoRA.

    Closes the feedback loop:
      Modal training → this endpoint → bank rebuild + re-harvest → basins update.
    """
    app_state = request.app.state

    # Rebuild resonance bank with any new coordized data
    rebuilt = False
    try:
        rebuild_fn = getattr(app_state, "rebuild_bank", None)
        if rebuild_fn:
            await asyncio.to_thread(rebuild_fn)
            rebuilt = True
            logger.info("Bank rebuilt after training completion")
    except Exception as e:
        logger.error("Bank rebuild after training failed: %s", e)

    # Reset auto-training one-shot flag so future harvests can re-trigger
    reset_fn = getattr(app_state, "reset_training_flag", None)
    if reset_fn:
        reset_fn()
        logger.info("Auto-training flag reset")

    # Trigger a harvest cycle so the updated model produces fresh coordizations
    scheduler = getattr(app_state, "harvest_scheduler", None)
    harvest_queued = False
    if scheduler:
        try:
            await scheduler.run_once()
            harvest_queued = True
            logger.info("Re-harvest triggered after training completion")
        except Exception as e:
            logger.error("Post-training re-harvest failed: %s", e)

    return {
        "status": "ok",
        "bank_rebuilt": rebuilt,
        "training_flag_reset": reset_fn is not None,
        "harvest_queued": harvest_queued,
    }
