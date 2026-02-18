"""
Training Data Pipeline — Upload → Extract → Chunk → LLM Enrich → JSONL

Pipeline:
  1. Receive file upload (PDF, MD, TXT, JSONL)
  2. Extract text (PDF via pymupdf in-process, others direct)
  3. Chunk into ~512-token segments at semantic boundaries
  4. LLM-enrich each chunk (Q&A pairs, E8 tags, concept extraction)
  5. Write structured JSONL to volume

Uses:
  - pymupdf for PDF extraction (in-process, no sandbox needed)
  - xAI API for fast batch enrichment (if XAI_API_KEY set)
  - External API (gpt-5-nano) fallback via LLMClient
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from ..config.settings import settings
from ..llm.client import LLMClient, LLMOptions

logger = logging.getLogger("vex.training")

TRAINING_DIR = Path(settings.training_dir)

# Async lock for safe concurrent JSONL appends
_write_lock = asyncio.Lock()


# ═══════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════


class E8Primitive(str, Enum):
    """E8-aligned consciousness primitives."""
    PER = "PER"    # Perception
    MEM = "MEM"    # Memory
    ACT = "ACT"    # Action
    PRD = "PRD"    # Prediction
    ETH = "ETH"    # Ethics
    META = "META"  # Meta-cognition
    HRT = "HRT"    # Heart/affect
    REL = "REL"    # Relationship
    MIX = "MIX"    # Multi-domain


class ProcessingMode(str, Enum):
    FAST = "fast"           # Chunk + store only, no LLM enrichment
    STANDARD = "standard"   # Chunk + LLM tag + basic Q&A
    DEEP = "deep"           # Full enrichment + concept extraction


@dataclass
class ChunkRecord:
    """A single training chunk in JSONL format."""
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


@dataclass
class IngestionResult:
    """Result of a document ingestion."""
    status: str
    source: str
    format: str
    total_chars: int
    chunks_written: int
    chunks_enriched: int
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
    except ImportError:
        raise RuntimeError(
            "PDF extraction requires pymupdf. Install with: pip install pymupdf"
        )

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
    elif ext in ("md", "markdown", "txt"):
        return content.decode("utf-8", errors="replace")
    elif ext == "jsonl":
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
                    chunks.append(para[i:i + max_chars].strip())
                current = ""
            else:
                current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]


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


async def _enrich_chunk_xai(chunk_text: str) -> dict[str, Any]:
    """Use xAI API for fast batch enrichment (if available)."""
    api_key = settings.xai_api_key
    if not api_key:
        return {}

    prompt = ENRICHMENT_PROMPT.format(chunk=chunk_text[:3000])

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-4-1-fast-non-reasoning",
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "response_format": {"type": "json_object"},
                },
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
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
) -> IngestionResult:
    """Full pipeline: extract → chunk → enrich → write JSONL."""
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
            status="ingested", source=filename, format="jsonl",
            total_chars=len(content), chunks_written=len(lines),
            chunks_enriched=0, qa_pairs_generated=0,
            processing_time_s=round(time.time() - start, 2),
            output_path=str(dest),
        )

    # Extract text
    try:
        text = _extract_text(content, filename)
    except Exception as e:
        return IngestionResult(
            status="error", source=filename, format=ext,
            total_chars=0, chunks_written=0, chunks_enriched=0,
            qa_pairs_generated=0,
            processing_time_s=round(time.time() - start, 2),
            output_path="", errors=[str(e)],
        )

    # Chunk
    chunks = chunk_text(text)
    if not chunks:
        return IngestionResult(
            status="empty", source=filename, format=ext,
            total_chars=len(text), chunks_written=0, chunks_enriched=0,
            qa_pairs_generated=0,
            processing_time_s=round(time.time() - start, 2),
            output_path="", errors=["No usable text extracted"],
        )

    # Enrich + build records
    enriched_count = 0
    qa_count = 0
    records: list[ChunkRecord] = []

    for i, chunk in enumerate(chunks):
        record = ChunkRecord(
            source=filename, category=category, chunk_index=i,
            text=chunk, hash=hashlib.md5(chunk.encode()).hexdigest(),
        )

        if mode != ProcessingMode.FAST:
            enrichment: dict[str, Any] = {}
            if settings.xai_api_key:
                enrichment = await _enrich_chunk_xai(chunk)
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

            # Rate limit between enrichments
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
        status="ingested", source=filename, format=ext,
        total_chars=len(text), chunks_written=len(records),
        chunks_enriched=enriched_count, qa_pairs_generated=qa_count,
        processing_time_s=round(time.time() - start, 2),
        output_path=str(dest), errors=errors,
    )


# ═══════════════════════════════════════════════════════════════
#  STATS & EXPORT
# ═══════════════════════════════════════════════════════════════


def _count_lines(filepath: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not filepath.exists():
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_stats() -> dict[str, Any]:
    """Get training data statistics from the volume."""
    _ensure_dirs()
    stats: dict[str, int] = {}
    for name in ("conversations", "corrections", "feedback"):
        stats[name] = _count_lines(TRAINING_DIR / f"{name}.jsonl")

    # Count curriculum chunks
    curriculum_dir = TRAINING_DIR / "curriculum"
    curriculum_files = list(curriculum_dir.glob("*.jsonl")) if curriculum_dir.exists() else []
    curriculum_chunks = 0
    for f in curriculum_files:
        curriculum_chunks += _count_lines(f)
    stats["curriculum"] = curriculum_chunks

    return {
        "stats": stats,
        "curriculum_files": [f.name for f in curriculum_files],
        "dir_exists": TRAINING_DIR.exists(),
        "training_dir": str(TRAINING_DIR),
    }


def export_openai_format() -> dict[str, Any]:
    """Export training data in OpenAI fine-tuning JSONL format."""
    _ensure_dirs()
    lines: list[dict[str, Any]] = []

    # Conversations
    conv_file = TRAINING_DIR / "conversations.jsonl"
    if conv_file.exists():
        with open(conv_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    lines.append({
                        "messages": [
                            {"role": "user", "content": entry.get("user_message", "")},
                            {"role": "assistant", "content": entry.get("response", "")},
                        ]
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

    # Curriculum Q&A pairs
    curriculum_dir = TRAINING_DIR / "curriculum"
    if curriculum_dir.exists():
        for jsonl_file in curriculum_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
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
                                lines.append({
                                    "messages": [
                                        {"role": "user", "content": q},
                                        {"role": "assistant", "content": a},
                                    ]
                                })
                    except (json.JSONDecodeError, KeyError):
                        continue

    return {
        "format": "openai_jsonl",
        "count": len(lines),
        "lines": lines[:100],
    }


# ═══════════════════════════════════════════════════════════════
#  FASTAPI ROUTER
# ═══════════════════════════════════════════════════════════════

# llm_client is injected by server.py after include_router
_llm_client: LLMClient | None = None


def set_llm_client(client: LLMClient) -> None:
    """Called by server.py to inject the shared LLMClient."""
    global _llm_client
    _llm_client = client


training_router = APIRouter()


@training_router.get("/training/stats")
async def training_stats_endpoint():
    """Get training data statistics."""
    return get_stats()


@training_router.get("/training/export")
async def training_export_endpoint():
    """Export training data in OpenAI fine-tuning format."""
    return export_openai_format()


@training_router.post("/training/upload")
async def training_upload_endpoint(
    file: UploadFile = File(...),
    category: str = Form(default="curriculum"),
    mode: str = Form(default="standard"),
    e8_primitive: str = Form(default=""),
):
    """Upload a document for training ingestion.

    Accepts PDF, Markdown, TXT, or JSONL files.
    """
    content = await file.read()
    filename = file.filename or "unknown"

    # Validate file type
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "md", "markdown", "txt", "jsonl"):
        return {"status": "error", "errors": [f"Unsupported file type: .{ext}"]}

    try:
        proc_mode = ProcessingMode(mode)
    except ValueError:
        proc_mode = ProcessingMode.STANDARD

    result = await ingest_document(
        content=content,
        filename=filename,
        category=category,
        mode=proc_mode,
        e8_override=e8_primitive if e8_primitive else None,
        llm=_llm_client,
    )

    return {
        "status": result.status,
        "source": result.source,
        "format": result.format,
        "chunks": result.chunks_written,
        "enriched": result.chunks_enriched,
        "qa_pairs": result.qa_pairs_generated,
        "processing_time_s": result.processing_time_s,
        "errors": result.errors,
    }


@training_router.post("/training/feedback")
async def training_feedback_endpoint(req: FeedbackRequest):
    """Submit feedback on a response."""
    await _log_feedback(req.conversation_id, req.rating, req.comment)
    return {"status": "recorded"}
