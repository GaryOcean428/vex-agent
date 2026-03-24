from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

import httpx
from kernel.coordizer_v2.adapter import CoordizerV2Adapter


def _load_ingest_with_tmp_training_dir(tmp_path: Path):
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["TRAINING_DIR"] = str(tmp_path / "training")
    os.environ["HARVEST_DIR"] = str(tmp_path / "harvest")

    for module_name in [
        "kernel.training.ingest",
        "kernel.training",
        "kernel.config.settings",
    ]:
        sys.modules.pop(module_name, None)

    import kernel.training.ingest as ingest_module

    return importlib.reload(ingest_module)


def test_ingest_document_populates_local_coordizer_bank(tmp_path: Path) -> None:
    ingest = _load_ingest_with_tmp_training_dir(tmp_path)
    adapter = CoordizerV2Adapter(CoordizerV2Adapter._create_bootstrap_coordizer())
    ingest.set_coordizer(adapter)

    bank_before = len(adapter.coordizer.bank)
    result = asyncio.run(
        ingest.ingest_document(
            content=(
                b"Fisher-Rao geometry preserves simplex structure. "
                b"Curriculum uploads should enrich the resonance bank immediately. "
                b"This document is long enough to produce multiple semantic chunks. "
            )
            * 20,
            filename="curriculum.txt",
            category="curriculum",
            mode=ingest.ProcessingMode.FAST,
        )
    )

    assert result.status == "ingested"
    assert result.chunks_coordized > 0
    assert len(adapter.coordizer.bank) > bank_before
    assert sum(adapter.coordizer.bank.tier_distribution().values()) == len(adapter.coordizer.bank)


def test_training_modal_data_endpoint_handles_read_timeout(tmp_path: Path, monkeypatch) -> None:
    ingest = _load_ingest_with_tmp_training_dir(tmp_path)
    monkeypatch.setattr(ingest.settings.modal, "training_url", "https://example-qloratrainer-train.modal.run")

    class _TimeoutClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url: str):
            raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(ingest.httpx, "AsyncClient", _TimeoutClient)

    result = asyncio.run(ingest.training_modal_data_endpoint())

    assert result["status"] == "error"
    assert result["error"] == "Timed out fetching Modal training data"
