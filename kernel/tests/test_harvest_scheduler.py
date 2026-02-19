"""
Tests for Harvest Scheduler
============================

Tests the harvest scheduler: directory management, budget tracking,
file acceptance, queue status, and governor integration.

The consciousness loop NEVER triggers Modal — only explicit
harvest requests or scheduled batches.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from kernel.coordizer_v2.harvest_scheduler import (
    HarvestBudget,
    HarvestScheduler,
    HarvestSchedulerConfig,
    HarvestQueueStatus,
)


# ═══════════════════════════════════════════════════════════════
#  BUDGET TESTS
# ═══════════════════════════════════════════════════════════════


class TestHarvestBudget:
    """Test the daily harvest budget tracker."""

    def test_initial_state(self):
        budget = HarvestBudget(daily_limit=5.00)
        assert budget.remaining == 5.00
        assert not budget.exhausted

    def test_record_spend(self):
        budget = HarvestBudget(daily_limit=5.00)
        budget.record_spend(1.50)
        assert abs(budget.remaining - 3.50) < 1e-6

    def test_budget_exhaustion(self):
        budget = HarvestBudget(daily_limit=1.00)
        budget.record_spend(1.00)
        assert budget.exhausted
        assert budget.remaining == 0.0

    def test_can_afford(self):
        budget = HarvestBudget(daily_limit=5.00)
        assert budget.can_afford(3.00)
        assert budget.can_afford(5.00)
        assert not budget.can_afford(5.01)

    def test_can_afford_after_spend(self):
        budget = HarvestBudget(daily_limit=5.00)
        budget.record_spend(3.00)
        assert budget.can_afford(2.00)
        assert not budget.can_afford(2.01)

    def test_status_dict(self):
        budget = HarvestBudget(daily_limit=10.00)
        budget.record_spend(2.50)
        status = budget.status()
        assert status["daily_limit"] == 10.00
        assert abs(status["spent_today"] - 2.50) < 1e-6
        assert abs(status["remaining"] - 7.50) < 1e-6
        assert status["exhausted"] is False

    def test_remaining_never_negative(self):
        budget = HarvestBudget(daily_limit=1.00)
        budget.record_spend(5.00)  # Overspend
        assert budget.remaining == 0.0


# ═══════════════════════════════════════════════════════════════
#  CONFIG TESTS
# ═══════════════════════════════════════════════════════════════


class TestSchedulerConfig:
    """Test scheduler configuration."""

    def test_default_config(self):
        config = HarvestSchedulerConfig(harvest_dir="/tmp/test_harvest")
        assert config.pending_dir == Path("/tmp/test_harvest/pending")
        assert config.processing_dir == Path("/tmp/test_harvest/processing")
        assert config.completed_dir == Path("/tmp/test_harvest/completed")
        assert config.failed_dir == Path("/tmp/test_harvest/failed")
        assert config.output_dir == Path("/tmp/test_harvest/output")

    def test_custom_config(self):
        config = HarvestSchedulerConfig(
            harvest_dir="/custom/path",
            daily_harvest_budget=10.00,
            max_batch_size=64,
        )
        assert config.daily_harvest_budget == 10.00
        assert config.max_batch_size == 64


# ═══════════════════════════════════════════════════════════════
#  SCHEDULER TESTS
# ═══════════════════════════════════════════════════════════════


class TestHarvestScheduler:
    """Test the harvest scheduler."""

    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            assert config.pending_dir.exists()
            assert config.processing_dir.exists()
            assert config.completed_dir.exists()
            assert config.failed_dir.exists()
            assert config.output_dir.exists()

    def test_list_pending_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)
            assert scheduler.list_pending() == []

    def test_list_pending_with_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            # Create some pending files
            for name in ["a.jsonl", "b.jsonl", "c.txt"]:
                (config.pending_dir / name).write_text("test")

            pending = scheduler.list_pending()
            assert len(pending) == 2  # Only .jsonl files
            names = [p.name for p in pending]
            assert "a.jsonl" in names
            assert "b.jsonl" in names
            assert "c.txt" not in names

    def test_get_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            # Create files in various states
            (config.pending_dir / "pending1.jsonl").write_text("test")
            (config.pending_dir / "pending2.jsonl").write_text("test")
            (config.completed_dir / "done.jsonl").write_text("test")
            (config.failed_dir / "fail.jsonl").write_text("test")

            status = scheduler.get_status()
            assert status.pending_files == 2
            assert status.completed_files == 1
            assert status.failed_files == 1
            assert "pending1.jsonl" in status.pending_filenames

    def test_status_to_dict(self):
        status = HarvestQueueStatus(
            pending_files=3,
            processing_files=1,
            completed_files=10,
            failed_files=2,
        )
        d = status.to_dict()
        assert d["pending_files"] == 3
        assert d["processing_files"] == 1
        assert d["completed_files"] == 10
        assert d["failed_files"] == 2

    @pytest.mark.asyncio
    async def test_accept_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            content = b'{"source": "curriculum", "text": "test content for upload acceptance", "priority": 1}\n'
            result = await scheduler.accept_upload("test.jsonl", content)

            assert result["accepted"] is True
            assert result["filename"] == "test.jsonl"
            assert result["size_bytes"] == len(content)

            # File should exist in pending
            assert (config.pending_dir / "test.jsonl").exists()

    @pytest.mark.asyncio
    async def test_accept_upload_adds_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            result = await scheduler.accept_upload("test", b"content")
            assert result["filename"] == "test.jsonl"

    @pytest.mark.asyncio
    async def test_accept_upload_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            # First upload
            await scheduler.accept_upload("data.jsonl", b"first")
            # Second upload with same name
            result = await scheduler.accept_upload("data.jsonl", b"second")

            # Should have a different filename
            assert result["filename"] != "data.jsonl"
            assert result["filename"].startswith("data_")

    @pytest.mark.asyncio
    async def test_run_once_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            results = await scheduler.run_once()
            assert results == []

    @pytest.mark.asyncio
    async def test_run_once_budget_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(
                harvest_dir=tmp,
                daily_harvest_budget=0.001,
            )
            scheduler = HarvestScheduler(config=config)
            scheduler.budget.record_spend(0.001)

            # Add a pending file
            (config.pending_dir / "test.jsonl").write_text(
                '{"source": "curriculum", "text": "budget exhaustion test entry", "priority": 1}\n'
            )

            results = await scheduler.run_once()
            assert len(results) == 1
            assert results[0]["success"] is False
            assert "budget" in results[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_process_file_no_ingestor(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)

            file_path = config.pending_dir / "test.jsonl"
            file_path.write_text("test content")

            result = await scheduler.process_file(file_path)
            assert result["success"] is False
            assert "ingestor" in result["error"].lower()

    def test_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HarvestSchedulerConfig(harvest_dir=tmp)
            scheduler = HarvestScheduler(config=config)
            scheduler._running = True
            scheduler.stop()
            assert scheduler._running is False
