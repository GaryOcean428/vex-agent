from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest

from kernel.consciousness.kernel_voice import KernelVoice
from kernel.coordizer_v2.adapter import CoordizerV2Adapter
from kernel.coordizer_v2.geometry import random_basin
from kernel.governance import KernelSpecialization


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


@pytest.mark.asyncio
async def test_kernel_voice_sparse_bank_fails_closed() -> None:
    coordizer = CoordizerV2Adapter._create_bootstrap_coordizer()
    voice = KernelVoice(KernelSpecialization.GENERAL, coordizer)

    output = await voice.generate(
        input_basin=random_basin(),
        kernel_basin=random_basin(),
        user_message="What do you know about this?",
        quenched_gain=1.0,
        base_temperature=0.7,
        llm_client=None,
    )

    assert output.text == ""
    assert output.geometric_raw != ""
    assert output.llm_expanded is False
