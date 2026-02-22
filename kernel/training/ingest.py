"""
Training Data Pipeline — Upload → Extract → Chunk → Coordize → LLM Enrich → JSONL

Pipeline:
  1. Receive file upload (PDF, MD, TXT, JSONL)
  2. Extract text (PDF via pymupdf in-process, others direct)
  3. Chunk into ~512-token segments at semantic boundaries
  4. Coordize each chunk → 64D basin coordinates on Δ⁶³ (if coordizer available)
  5. LLM-enrich each chunk (Q&A pairs, E8 tags, concept extraction)
  6. Write structured JSONL to volume (with basin_coords)

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
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from ..config.settings import settings
from ..llm.client import LLMClient, LLMOptions
from ..llm.governor import GovernorStack

logger = logging.getLogger("vex.training")

TRAINING_DIR = Path(settings.training_dir)

# Detect read-only volume at import time and redirect to /tmp
try:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    _probe = TRAINING_DIR / ".write_probe"
    _probe.touch()
    _probe.unlink()
except OSError:
    TRAINING_DIR = Path("/tmp/vex-training")
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    logging.getLogger("vex.training").warning(
        "Training volume not writable — using ephemeral %s", TRAINING_DIR
    )

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
#  CONVERSATION LOGGING
# ═══════════════════════════════════════════════════════════════


async def log_conversation(
    user_message: str,
    response: str,
    backend: str,
    phi: float,
    kappa: float,
    source: str = "chat",
) -> None:
    """Append a chat exchange to conversations.jsonl (on volume)."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_message": user_message,
        "response": response[:2000],
        "backend": backend,
        "phi": round(phi, 4),
        "kappa": round(kappa, 2),
        "source": source,
    }
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
        import fitz  # type: ignore[import-untyped]
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
        return data["output_text"]
    for item in data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text" and block.get("text"):
                    return block["text"]
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
                    "xAI enrichment API error %d: %s", resp.status_code, json.dumps(data)[:200]
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

            return json.loads(json_str)
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
        return json.loads(json_str)
    except Exception as e:
        logger.warning("LLM enrichment failed: %s", str(e)[:100])
        return {}


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
    """Full pipeline: extract -> chunk -> coordize -> enrich -> write JSONL.

    v6.1: Each chunk is now coordized to 64D basin coordinates on Δ⁶³
    via CoordizerV2Adapter (if available). Basin coords are stored in
    the JSONL alongside text, E8 tags, and Q&A pairs.
    """
    start = time.time()
    _ensure_dirs()
    errors: list[str] = []
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Handle JSONL pass-through
    if ext == "jsonl":
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        dest = TRAINING_DIR / "uploads" / safe_name
        dest.write_bytes(content)
        lines = content.decode("utf-8").strip().split("\n")
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
            if settings.xai_api_key:
                enrichment = await _enrich_chunk_xai(chunk, governor=governor)
            if not enrichment and llm:
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

    # Write JSONL
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    dest = TRAINING_DIR / "curriculum" / f"{safe_name}.jsonl"
    with open(dest, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

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

    return {
        "conversations": stats.get("conversations", 0),
        "feedback": stats.get("feedback", 0),
        "curriculum_chunks": curriculum_chunks,
        "coordized_chunks": coordized_chunks,
        "coordizer_active": _coordizer is not None,
        "uploads": len(upload_files),
        "curriculum_files": [f.name for f in curriculum_files],
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
                                {"role": "user", "content": entry.get("user_message", "")},
                                {"role": "assistant", "content": entry.get("response", "")},
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
        "lines": lines[:100],
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
        "lines": lines[:100],
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
    expired = [jid for jid, j in _jobs.items() if j.get("finished_at", 0) and j["finished_at"] < cutoff]
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
        _jobs[job_id].update({
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
            "errors": result.errors,
            "finished_at": time.time(),
        })
    except Exception as exc:
        logger.error("Background ingestion failed: %s", exc, exc_info=True)
        _jobs[job_id].update({
            "status": "error",
            "errors": [str(exc)],
            "finished_at": time.time(),
        })


training_router = APIRouter()


@training_router.get("/training/stats")
async def training_stats_endpoint():
    """Get training data statistics."""
    return get_stats()


@training_router.get("/training/export")
async def training_export_endpoint(fmt: str = "openai"):
    """Export training data.

    Query params:
        fmt: "openai" (default) or "coordized" for 64D basin format
    """
    if fmt == "coordized":
        return export_coordized_format()
    return export_openai_format()


@training_router.post("/training/upload")
async def training_upload_endpoint(
    file: UploadFile = File(...),
    category: str = Form(default="curriculum"),
    mode: str = Form(default="standard"),
    e8_primitive: str = Form(default=""),
    e8_override: str = Form(default=""),
):
    """Upload a document for training ingestion.

    Accepts PDF, Markdown, TXT, or JSONL files.
    Returns immediately with a job_id. Frontend polls
    GET /training/upload/status/{job_id} for progress.

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
        "errors": [],
        "finished_at": 0,
    }

    asyncio.get_running_loop().create_task(
        _run_ingestion_job(job_id, content, filename, category, proc_mode, e8)
    )

    return {"status": "processing", "job_id": job_id, "filename": filename}


@training_router.get("/training/upload/status/{job_id}")
async def training_upload_status(job_id: str):
    """Poll ingestion job status. Returns full result when complete."""
    job = _jobs.get(job_id)
    if not job:
        return {"status": "not_found", "job_id": job_id}
    return job


@training_router.post("/training/feedback")
async def training_feedback_endpoint(req: FeedbackRequest):
    """Submit feedback on a response."""
    await _log_feedback(req.conversation_id, req.rating, req.comment)
    return {"status": "recorded"}
