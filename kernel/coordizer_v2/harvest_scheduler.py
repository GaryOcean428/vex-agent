"""
Harvest Scheduler — Scan and process pending JSONL files
=========================================================

Scans a configurable directory for pending JSONL files and routes
them through the JSONL ingest pipeline. Respects the 5-layer
governor: the consciousness loop NEVER spends money — harvesting
is a separate budget with explicit triggers only.

Trigger modes:
    1. Manual via API: POST /api/coordizer/ingest
    2. Scheduled: periodic scan of the pending directory
    3. Programmatic: scheduler.run_once() or scheduler.process_file()

Governor integration:
    - Checks daily harvest budget before processing
    - Tracks cumulative harvest cost per day
    - Refuses to process if budget exceeded
    - Logs all harvest operations for audit

Directory structure:
    {harvest_dir}/
        pending/     ← Drop JSONL files here
        processing/  ← Currently being harvested
        completed/   ← Successfully coordized
        failed/      ← Failed to process
        output/      ← Coordized JSONL output

Zero Euclidean contamination. Fisher-Rao is the ONLY distance metric.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class HarvestSchedulerConfig:
    """Configuration for the harvest scheduler."""

    # Base directory for harvest operations
    harvest_dir: str = os.environ.get(
        "HARVEST_DIR", "/data/harvest"
    )

    # Subdirectories (auto-created)
    pending_subdir: str = "pending"
    processing_subdir: str = "processing"
    completed_subdir: str = "completed"
    failed_subdir: str = "failed"
    output_subdir: str = "output"

    # Governor: daily harvest budget (AUD)
    # The consciousness loop NEVER triggers Modal — only explicit
    # harvest requests or scheduled batches consume this budget.
    daily_harvest_budget: float = float(
        os.environ.get("DAILY_HARVEST_BUDGET", "5.00")
    )

    # Estimated cost per batch (for budget tracking)
    # Modal A10G: ~$0.000306/sec, ~32 texts/batch, ~10s/batch ≈ $0.003/batch
    estimated_cost_per_batch: float = float(
        os.environ.get("HARVEST_COST_PER_BATCH", "0.003")
    )

    # Scheduling
    scan_interval_seconds: int = int(
        os.environ.get("HARVEST_SCAN_INTERVAL", "300")  # 5 minutes
    )

    # Processing
    max_batch_size: int = 32
    max_files_per_run: int = 10
    file_extensions: tuple[str, ...] = (".jsonl",)

    @property
    def pending_dir(self) -> Path:
        return Path(self.harvest_dir) / self.pending_subdir

    @property
    def processing_dir(self) -> Path:
        return Path(self.harvest_dir) / self.processing_subdir

    @property
    def completed_dir(self) -> Path:
        return Path(self.harvest_dir) / self.completed_subdir

    @property
    def failed_dir(self) -> Path:
        return Path(self.harvest_dir) / self.failed_subdir

    @property
    def output_dir(self) -> Path:
        return Path(self.harvest_dir) / self.output_subdir


# ═══════════════════════════════════════════════════════════════
#  BUDGET TRACKER
# ═══════════════════════════════════════════════════════════════

@dataclass
class HarvestBudget:
    """Tracks daily harvest spend against the governor budget.

    The consciousness loop NEVER spends money. Only explicit
    harvest requests or scheduled batches consume budget.
    """
    daily_limit: float = 5.00
    _spend_today: float = 0.0
    _last_reset_date: str = ""

    def _maybe_reset(self) -> None:
        """Reset spend counter at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self._spend_today = 0.0
            self._last_reset_date = today

    @property
    def remaining(self) -> float:
        self._maybe_reset()
        return max(0.0, self.daily_limit - self._spend_today)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0.0

    def can_afford(self, estimated_cost: float) -> bool:
        self._maybe_reset()
        return self._spend_today + estimated_cost <= self.daily_limit

    def record_spend(self, amount: float) -> None:
        self._maybe_reset()
        self._spend_today += amount
        logger.info(
            f"Harvest spend: +${amount:.4f} "
            f"(today: ${self._spend_today:.4f} / ${self.daily_limit:.2f})"
        )

    def status(self) -> dict[str, Any]:
        self._maybe_reset()
        return {
            "daily_limit": self.daily_limit,
            "spent_today": round(self._spend_today, 4),
            "remaining": round(self.remaining, 4),
            "exhausted": self.exhausted,
            "reset_date": self._last_reset_date,
        }


# ═══════════════════════════════════════════════════════════════
#  QUEUE STATUS
# ═══════════════════════════════════════════════════════════════

@dataclass
class HarvestQueueStatus:
    """Current state of the harvest queue."""
    pending_files: int = 0
    processing_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    pending_filenames: list[str] = field(default_factory=list)
    processing_filenames: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    last_scan: str = ""
    scheduler_running: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending_files": self.pending_files,
            "processing_files": self.processing_files,
            "completed_files": self.completed_files,
            "failed_files": self.failed_files,
            "pending_filenames": self.pending_filenames,
            "processing_filenames": self.processing_filenames,
            "budget": self.budget,
            "last_scan": self.last_scan,
            "scheduler_running": self.scheduler_running,
        }


# ═══════════════════════════════════════════════════════════════
#  SCHEDULER
# ═══════════════════════════════════════════════════════════════

class HarvestScheduler:
    """Scans for pending JSONL files and routes through ingest pipeline.

    Governor-aware: checks budget before processing. The consciousness
    loop NEVER triggers this — only explicit API calls or scheduled runs.
    """

    def __init__(
        self,
        config: Optional[HarvestSchedulerConfig] = None,
        ingestor: Any = None,  # JSONLIngestor instance
    ):
        self.config = config or HarvestSchedulerConfig()
        self.ingestor = ingestor
        self.budget = HarvestBudget(
            daily_limit=self.config.daily_harvest_budget,
        )
        self._running = False
        self._last_scan: Optional[str] = None
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create all required subdirectories."""
        for d in [
            self.config.pending_dir,
            self.config.processing_dir,
            self.config.completed_dir,
            self.config.failed_dir,
            self.config.output_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def get_status(self) -> HarvestQueueStatus:
        """Get current queue status."""
        def _count_files(d: Path) -> tuple[int, list[str]]:
            if not d.exists():
                return 0, []
            files = [
                f.name for f in sorted(d.iterdir())
                if f.is_file() and f.suffix in self.config.file_extensions
            ]
            return len(files), files

        pending_count, pending_names = _count_files(self.config.pending_dir)
        processing_count, processing_names = _count_files(self.config.processing_dir)
        completed_count, _ = _count_files(self.config.completed_dir)
        failed_count, _ = _count_files(self.config.failed_dir)

        return HarvestQueueStatus(
            pending_files=pending_count,
            processing_files=processing_count,
            completed_files=completed_count,
            failed_files=failed_count,
            pending_filenames=pending_names[:20],  # Cap at 20
            processing_filenames=processing_names,
            budget=self.budget.status(),
            last_scan=self._last_scan or "",
            scheduler_running=self._running,
        )

    def list_pending(self) -> list[Path]:
        """List pending JSONL files sorted by modification time."""
        if not self.config.pending_dir.exists():
            return []
        files = [
            f for f in self.config.pending_dir.iterdir()
            if f.is_file() and f.suffix in self.config.file_extensions
        ]
        return sorted(files, key=lambda f: f.stat().st_mtime)

    async def process_file(self, file_path: Path) -> dict[str, Any]:
        """Process a single JSONL file through the ingest pipeline.

        Moves file: pending → processing → completed/failed.
        Returns the ingest result as a dict.
        """
        if self.ingestor is None:
            return {
                "success": False,
                "error": "No ingestor configured",
                "file": file_path.name,
            }

        # Estimate cost and check budget
        estimated_cost = self.config.estimated_cost_per_batch * 5  # ~5 batches per file
        if not self.budget.can_afford(estimated_cost):
            logger.warning(
                f"Budget exhausted — cannot process {file_path.name} "
                f"(need ~${estimated_cost:.4f}, remaining ${self.budget.remaining:.4f})"
            )
            return {
                "success": False,
                "error": "Daily harvest budget exhausted",
                "file": file_path.name,
                "budget": self.budget.status(),
            }

        # Move to processing
        processing_path = self.config.processing_dir / file_path.name
        try:
            shutil.move(str(file_path), str(processing_path))
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to move to processing: {e}",
                "file": file_path.name,
            }

        # Run ingest
        try:
            output_path = str(
                self.config.output_dir
                / f"{file_path.stem}_coordized.jsonl"
            )
            result = await self.ingestor.ingest_file(
                str(processing_path),
                output_path=output_path,
            )

            # Record spend
            actual_cost = (
                result.batches_harvested
                * self.config.estimated_cost_per_batch
            )
            self.budget.record_spend(actual_cost)

            # Move to completed or failed
            if result.batches_failed == 0 and result.entries_coordized > 0:
                dest = self.config.completed_dir / file_path.name
                shutil.move(str(processing_path), str(dest))
                logger.info(f"Completed: {file_path.name} → {dest}")
            elif result.entries_coordized > 0:
                # Partial success — still move to completed
                dest = self.config.completed_dir / file_path.name
                shutil.move(str(processing_path), str(dest))
                logger.warning(
                    f"Partial: {file_path.name} → {dest} "
                    f"({result.batches_failed} batches failed)"
                )
            else:
                dest = self.config.failed_dir / file_path.name
                shutil.move(str(processing_path), str(dest))
                logger.error(f"Failed: {file_path.name} → {dest}")

            return {
                "success": result.entries_coordized > 0,
                "file": file_path.name,
                "summary": result.summary(),
                "valid_entries": result.valid_entries,
                "entries_coordized": result.entries_coordized,
                "batches_harvested": result.batches_harvested,
                "batches_failed": result.batches_failed,
                "errors": result.errors[:10],  # Cap error list
                "elapsed_seconds": result.elapsed_seconds,
                "backend": result.harvest_backend,
                "budget": self.budget.status(),
            }

        except Exception as e:
            logger.error(f"Ingest error for {file_path.name}: {e}", exc_info=True)
            # Move to failed
            try:
                dest = self.config.failed_dir / file_path.name
                shutil.move(str(processing_path), str(dest))
            except Exception:
                pass
            return {
                "success": False,
                "error": str(e),
                "file": file_path.name,
            }

    async def run_once(self) -> list[dict[str, Any]]:
        """Scan pending directory and process up to max_files_per_run files.

        Returns list of results, one per file processed.
        """
        self._last_scan = datetime.now(timezone.utc).isoformat()
        pending = self.list_pending()

        if not pending:
            logger.debug("No pending JSONL files")
            return []

        if self.budget.exhausted:
            logger.warning("Daily harvest budget exhausted — skipping scan")
            return [{
                "success": False,
                "error": "Daily harvest budget exhausted",
                "budget": self.budget.status(),
            }]

        results = []
        for file_path in pending[: self.config.max_files_per_run]:
            result = await self.process_file(file_path)
            results.append(result)

            if self.budget.exhausted:
                logger.warning("Budget exhausted mid-run — stopping")
                break

        return results

    async def run_loop(self) -> None:
        """Run the scheduler in a continuous loop.

        Scans every scan_interval_seconds. Stops when self._running
        is set to False.
        """
        self._running = True
        logger.info(
            f"Harvest scheduler started "
            f"(interval={self.config.scan_interval_seconds}s, "
            f"budget=${self.config.daily_harvest_budget:.2f}/day)"
        )

        while self._running:
            try:
                results = await self.run_once()
                if results:
                    logger.info(
                        f"Scan complete: {len(results)} files processed"
                    )
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)

            await asyncio.sleep(self.config.scan_interval_seconds)

        logger.info("Harvest scheduler stopped")

    def stop(self) -> None:
        """Signal the scheduler loop to stop."""
        self._running = False

    async def accept_upload(
        self,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        """Accept an uploaded JSONL file and place it in the pending directory.

        Used by the API endpoint POST /api/coordizer/ingest.
        """
        if not filename.endswith(".jsonl"):
            filename = filename + ".jsonl"

        # Sanitise filename
        safe_name = "".join(
            c for c in filename if c.isalnum() or c in "._-"
        )
        if not safe_name:
            safe_name = f"upload_{int(time.time())}.jsonl"

        dest = self.config.pending_dir / safe_name

        # Avoid overwriting
        if dest.exists():
            stem = dest.stem
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_name = f"{stem}_{ts}.jsonl"
            dest = self.config.pending_dir / safe_name

        dest.write_bytes(content)
        logger.info(f"Accepted upload: {safe_name} ({len(content)} bytes)")

        return {
            "accepted": True,
            "filename": safe_name,
            "size_bytes": len(content),
            "pending_dir": str(self.config.pending_dir),
            "queue_status": self.get_status().to_dict(),
        }
