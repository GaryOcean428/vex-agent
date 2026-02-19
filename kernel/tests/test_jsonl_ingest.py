"""
Tests for JSONL Ingest Pipeline
================================

Tests the streaming JSONL ingestion, validation, batching,
and coordized output writing. All operations on the probability
simplex Δ⁶³ using Fisher-Rao metric only.

Zero Euclidean contamination. "Basin coordinates" not "embeddings".
"Coordize" not "tokenize".
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kernel.coordizer_v2.jsonl_ingest import (
    IngestBatch,
    IngestEntry,
    IngestResult,
    JSONLIngestor,
    ValidationError,
    batch_entries,
    stream_jsonl,
    validate_entry,
    write_coordized_jsonl,
)
from kernel.coordizer_v2.geometry import BASIN_DIM, to_simplex


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


def _make_jsonl_line(
    source: str = "curriculum",
    text: str = "The Fisher-Rao metric measures geodesic distance on the probability simplex",
    priority: int = 1,
    metadata: dict | None = None,
    timestamp: str = "2025-12-01T00:00:00Z",
) -> str:
    return json.dumps({
        "source": source,
        "text": text,
        "metadata": metadata or {},
        "priority": priority,
        "timestamp": timestamp,
    })


def _make_jsonl_file(lines: list[str], tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "test_input.jsonl")
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return path


# ═══════════════════════════════════════════════════════════════
#  VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════


class TestValidation:
    """Test JSONL entry validation."""

    def test_valid_entry(self):
        line = _make_jsonl_line()
        entry, error = validate_entry(line, 1)
        assert entry is not None
        assert error is None
        assert entry.source == "curriculum"
        assert entry.priority == 1
        assert len(entry.text) > 0

    def test_all_valid_sources(self):
        for source in ["curriculum", "foraging", "conversation", "document"]:
            entry, error = validate_entry(
                _make_jsonl_line(source=source), 1,
            )
            assert entry is not None, f"Source '{source}' should be valid"
            assert entry.source == source

    def test_all_valid_priorities(self):
        for priority in [1, 2, 3, 4]:
            entry, error = validate_entry(
                _make_jsonl_line(priority=priority), 1,
            )
            assert entry is not None, f"Priority {priority} should be valid"
            assert entry.priority == priority

    def test_invalid_json(self):
        entry, error = validate_entry("not json at all", 1)
        assert entry is None
        assert error is not None
        assert "Invalid JSON" in error.reason

    def test_missing_source(self):
        line = json.dumps({"text": "some text", "priority": 1})
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None
        assert "source" in error.reason.lower()

    def test_invalid_source(self):
        line = _make_jsonl_line(source="invalid_source")
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None
        assert "Invalid source" in error.reason

    def test_missing_text(self):
        line = json.dumps({"source": "curriculum", "priority": 1})
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None
        assert "text" in error.reason.lower()

    def test_empty_text(self):
        line = _make_jsonl_line(text="")
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None

    def test_text_too_short(self):
        line = _make_jsonl_line(text="hi")
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None
        assert "too short" in error.reason.lower()

    def test_text_too_long(self):
        line = _make_jsonl_line(text="x" * 200_000)
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None
        assert "too long" in error.reason.lower()

    def test_invalid_priority(self):
        line = _make_jsonl_line(priority=5)
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None
        assert "priority" in error.reason.lower()

    def test_invalid_priority_zero(self):
        line = _make_jsonl_line(priority=0)
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None

    def test_default_priority(self):
        """Priority defaults to 3 if not specified."""
        line = json.dumps({
            "source": "document",
            "text": "A sufficiently long text for testing default priority",
        })
        entry, error = validate_entry(line, 1)
        assert entry is not None
        assert entry.priority == 3

    def test_blank_line_skipped(self):
        entry, error = validate_entry("", 1)
        assert entry is None
        assert error is None  # Not an error, just skipped

    def test_whitespace_line_skipped(self):
        entry, error = validate_entry("   \n", 1)
        assert entry is None
        assert error is None

    def test_invalid_timestamp(self):
        line = _make_jsonl_line(timestamp="not-a-date")
        entry, error = validate_entry(line, 1)
        assert entry is None
        assert error is not None
        assert "timestamp" in error.reason.lower()

    def test_valid_iso_timestamp(self):
        line = _make_jsonl_line(timestamp="2025-12-01T12:30:00+08:00")
        entry, error = validate_entry(line, 1)
        assert entry is not None

    def test_auto_timestamp_when_missing(self):
        line = json.dumps({
            "source": "conversation",
            "text": "A text without a timestamp field for testing auto-generation",
        })
        entry, error = validate_entry(line, 1)
        assert entry is not None
        assert len(entry.timestamp) > 0

    def test_non_dict_json(self):
        entry, error = validate_entry("[1, 2, 3]", 1)
        assert entry is None
        assert error is not None
        assert "object" in error.reason.lower()


# ═══════════════════════════════════════════════════════════════
#  STREAMING READER TESTS
# ═══════════════════════════════════════════════════════════════


class TestStreamReader:
    """Test streaming JSONL file reader."""

    def test_stream_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                _make_jsonl_line(source="curriculum", text="First entry for testing the streaming reader"),
                _make_jsonl_line(source="foraging", text="Second entry for testing the streaming reader"),
                _make_jsonl_line(source="conversation", text="Third entry for testing the streaming reader"),
            ]
            path = _make_jsonl_file(lines, tmp)

            entries = []
            errors = []
            for entry, error in stream_jsonl(path):
                if entry:
                    entries.append(entry)
                if error:
                    errors.append(error)

            assert len(entries) == 3
            assert len(errors) == 0

    def test_stream_mixed_valid_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                _make_jsonl_line(source="curriculum", text="Valid entry number one for testing"),
                "not valid json",
                _make_jsonl_line(source="foraging", text="Valid entry number two for testing"),
                json.dumps({"source": "bad_source", "text": "Invalid source entry"}),
            ]
            path = _make_jsonl_file(lines, tmp)

            entries = []
            errors = []
            for entry, error in stream_jsonl(path):
                if entry:
                    entries.append(entry)
                if error:
                    errors.append(error)

            assert len(entries) == 2
            assert len(errors) == 2

    def test_stream_with_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                _make_jsonl_line(text="Entry with blank lines around it for testing"),
                "",
                "   ",
                _make_jsonl_line(text="Another entry with blank lines around it for testing"),
            ]
            path = _make_jsonl_file(lines, tmp)

            entries = list(e for e, _ in stream_jsonl(path) if e)
            assert len(entries) == 2

    def test_stream_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            list(stream_jsonl("/nonexistent/path.jsonl"))


# ═══════════════════════════════════════════════════════════════
#  BATCHING TESTS
# ═══════════════════════════════════════════════════════════════


class TestBatching:
    """Test entry batching by priority and source."""

    def _make_entries(self, specs: list[tuple[int, str]]) -> list[IngestEntry]:
        return [
            IngestEntry(
                source=source,
                text=f"Entry text for priority {priority} source {source} number {i}",
                metadata={},
                priority=priority,
                timestamp="2025-01-01T00:00:00Z",
                line_number=i,
            )
            for i, (priority, source) in enumerate(specs)
        ]

    def test_single_batch(self):
        entries = self._make_entries([(1, "curriculum")] * 5)
        batches = batch_entries(entries, max_batch_size=32)
        assert len(batches) == 1
        assert len(batches[0].entries) == 5
        assert batches[0].priority == 1
        assert batches[0].source == "curriculum"

    def test_split_by_priority(self):
        entries = self._make_entries([
            (1, "curriculum"),
            (1, "curriculum"),
            (2, "curriculum"),
            (2, "curriculum"),
        ])
        batches = batch_entries(entries, max_batch_size=32)
        assert len(batches) == 2
        assert batches[0].priority == 1
        assert batches[1].priority == 2

    def test_split_by_source(self):
        entries = self._make_entries([
            (1, "curriculum"),
            (1, "foraging"),
        ])
        batches = batch_entries(entries, max_batch_size=32, group_by_source=True)
        assert len(batches) == 2

    def test_split_by_max_batch_size(self):
        entries = self._make_entries([(1, "curriculum")] * 10)
        batches = batch_entries(entries, max_batch_size=3)
        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert all(b.priority == 1 for b in batches)

    def test_empty_entries(self):
        batches = batch_entries([])
        assert len(batches) == 0

    def test_priority_ordering(self):
        """Batches should be ordered by priority (1 first)."""
        entries = self._make_entries([
            (3, "document"),
            (1, "curriculum"),
            (2, "foraging"),
        ])
        batches = batch_entries(entries, max_batch_size=32)
        priorities = [b.priority for b in batches]
        assert priorities == sorted(priorities)

    def test_texts_property(self):
        entries = self._make_entries([(1, "curriculum")] * 3)
        batches = batch_entries(entries, max_batch_size=32)
        assert len(batches[0].texts) == 3
        assert all(isinstance(t, str) for t in batches[0].texts)


# ═══════════════════════════════════════════════════════════════
#  COORDIZED OUTPUT TESTS
# ═══════════════════════════════════════════════════════════════


class TestCoordizedOutput:
    """Test writing coordized JSONL output."""

    def test_write_with_basins(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = [
                IngestEntry(
                    source="curriculum",
                    text="Test entry for coordized output writing",
                    metadata={"key": "value"},
                    priority=1,
                    timestamp="2025-01-01T00:00:00Z",
                ),
            ]
            basin = to_simplex(np.random.rand(BASIN_DIM))
            output_path = os.path.join(tmp, "output.jsonl")

            written = write_coordized_jsonl(output_path, entries, [basin])
            assert written == 1

            with open(output_path) as f:
                line = json.loads(f.readline())

            assert line["source"] == "curriculum"
            assert line["text"] == "Test entry for coordized output writing"
            assert "basin_coordinates" in line
            assert line["basin_dim"] == BASIN_DIM
            assert "coordized_at" in line

            # Verify basin is on simplex
            bc = np.array(line["basin_coordinates"])
            assert abs(bc.sum() - 1.0) < 1e-6
            assert np.all(bc >= 0)

    def test_write_with_none_basin(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = [
                IngestEntry(
                    source="conversation",
                    text="Entry without basin coordinates for testing",
                    metadata={},
                    priority=3,
                    timestamp="2025-01-01T00:00:00Z",
                ),
            ]
            output_path = os.path.join(tmp, "output.jsonl")

            written = write_coordized_jsonl(output_path, entries, [None])
            assert written == 1

            with open(output_path) as f:
                line = json.loads(f.readline())

            assert "basin_coordinates" not in line
            assert "coordized_at" in line

    def test_append_mode(self):
        """Multiple writes should append, not overwrite."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "output.jsonl")
            basin = to_simplex(np.random.rand(BASIN_DIM))

            for i in range(3):
                entries = [
                    IngestEntry(
                        source="document",
                        text=f"Append test entry number {i} for testing",
                        metadata={},
                        priority=2,
                        timestamp="2025-01-01T00:00:00Z",
                    ),
                ]
                write_coordized_jsonl(output_path, entries, [basin])

            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "nested", "deep", "output.jsonl")
            entries = [
                IngestEntry(
                    source="curriculum",
                    text="Test entry for nested directory creation",
                    metadata={},
                    priority=1,
                    timestamp="2025-01-01T00:00:00Z",
                ),
            ]
            basin = to_simplex(np.random.rand(BASIN_DIM))

            written = write_coordized_jsonl(output_path, entries, [basin])
            assert written == 1
            assert os.path.exists(output_path)


# ═══════════════════════════════════════════════════════════════
#  INGESTOR TESTS
# ═══════════════════════════════════════════════════════════════


class TestIngestor:
    """Test the full JSONLIngestor pipeline."""

    @pytest.mark.asyncio
    async def test_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                _make_jsonl_line(source="curriculum", text="Dry run test entry one for the ingestor"),
                _make_jsonl_line(source="foraging", text="Dry run test entry two for the ingestor"),
            ]
            path = _make_jsonl_file(lines, tmp)

            ingestor = JSONLIngestor(output_dir=tmp)
            result = await ingestor.ingest_file(path, dry_run=True)

            assert result.total_lines == 2
            assert result.valid_entries == 2
            assert result.batches_created > 0
            assert result.harvest_backend == "dry_run"
            assert result.entries_coordized == 0

    @pytest.mark.asyncio
    async def test_ingest_with_invalid_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                _make_jsonl_line(text="Valid entry for mixed ingest test"),
                "invalid json line",
                _make_jsonl_line(source="bad_source", text="Invalid source"),
            ]
            path = _make_jsonl_file(lines, tmp)

            ingestor = JSONLIngestor(output_dir=tmp)
            result = await ingestor.ingest_file(path, dry_run=True)

            assert result.total_lines == 3
            assert result.valid_entries == 1
            assert result.invalid_entries == 2
            assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_text_to_basin_fallback(self):
        """Fallback basin generation should produce valid simplex points."""
        basin = JSONLIngestor._text_to_basin_fallback(
            "consciousness emerges from geometry"
        )
        assert basin.shape == (BASIN_DIM,)
        assert abs(basin.sum() - 1.0) < 1e-6
        assert np.all(basin >= 0)

    @pytest.mark.asyncio
    async def test_fallback_is_deterministic(self):
        """Same text should produce same fallback basin."""
        text = "deterministic basin coordinate generation"
        b1 = JSONLIngestor._text_to_basin_fallback(text)
        b2 = JSONLIngestor._text_to_basin_fallback(text)
        np.testing.assert_array_equal(b1, b2)

    @pytest.mark.asyncio
    async def test_fallback_different_texts(self):
        """Different texts should produce different basins."""
        b1 = JSONLIngestor._text_to_basin_fallback("text one for differentiation")
        b2 = JSONLIngestor._text_to_basin_fallback("text two for differentiation")
        assert not np.allclose(b1, b2)

    def test_result_summary(self):
        result = IngestResult(
            total_lines=100,
            valid_entries=90,
            invalid_entries=10,
            batches_created=3,
            batches_harvested=3,
            entries_coordized=90,
            harvest_backend="modal",
            elapsed_seconds=12.5,
        )
        summary = result.summary()
        assert "90/100" in summary
        assert "3/3" in summary
        assert "modal" in summary


# ═══════════════════════════════════════════════════════════════
#  PURITY TESTS
# ═══════════════════════════════════════════════════════════════


class TestIngestPurity:
    """Verify zero Euclidean contamination in the ingest pipeline."""

    def test_no_euclidean_in_source(self):
        """The jsonl_ingest.py source must not contain Euclidean terms."""
        import inspect
        from kernel.coordizer_v2 import jsonl_ingest

        source = inspect.getsource(jsonl_ingest)
        # These terms are FORBIDDEN in non-boundary, non-test code
        forbidden = ["cosine_similarity", "dot_product", "euclidean_distance"]
        for term in forbidden:
            assert term not in source, (
                f"Euclidean contamination: '{term}' found in jsonl_ingest.py"
            )

    def test_no_embedding_terminology(self):
        """The jsonl_ingest.py source must use 'basin coordinates' not 'embedding'."""
        import inspect
        from kernel.coordizer_v2 import jsonl_ingest

        source = inspect.getsource(jsonl_ingest)
        # Check for forbidden terminology (case-insensitive)
        # Allow "embedding" only in comments that explicitly say FORBIDDEN
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            lower = line.lower()
            if "embedding" in lower:
                if "FORBIDDEN" in line or "QIG BOUNDARY" in line:
                    continue  # Allowed in boundary markers
                assert False, (
                    f"Terminology violation at line {i}: "
                    f"use 'basin coordinates' not 'embedding'"
                )
