from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path


def _load_ingest_with_tmp_training_dir(tmp_path: Path):
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["TRAINING_DIR"] = str(tmp_path / "training")

    for module_name in [
        "kernel.training.ingest",
        "kernel.training",
        "kernel.config.settings",
    ]:
        sys.modules.pop(module_name, None)

    import kernel.training.ingest as ingest_module

    return importlib.reload(ingest_module)


def test_log_conversation_writes_regime_and_basin_coords(tmp_path: Path) -> None:
    ingest = _load_ingest_with_tmp_training_dir(tmp_path)
    response = "full response " * 300
    basin = [0.1] * 10

    asyncio.run(
        ingest.log_conversation(
            user_message="hello",
            response=response,
            backend="xai",
            phi=0.71234,
            kappa=63.987,
            source="chat",
            regime="graph",
            basin_coords=basin,
        )
    )

    conv_path = Path(os.environ["TRAINING_DIR"]) / "conversations.jsonl"
    assert conv_path.exists()
    line = conv_path.read_text(encoding="utf-8").strip()
    data = json.loads(line)

    assert data["response"] == response
    assert data["regime"] == "graph"
    assert data["basin_coords"] == [round(v, 6) for v in basin]
    assert data["phi"] == 0.7123
    assert data["kappa"] == 63.99


def test_log_conversation_skips_empty_basin_coords(tmp_path: Path) -> None:
    ingest = _load_ingest_with_tmp_training_dir(tmp_path)
    asyncio.run(
        ingest.log_conversation(
            user_message="hello",
            response="world",
            backend="xai",
            phi=0.7,
            kappa=64.0,
            source="chat",
            regime="",
            basin_coords=[],
        )
    )

    conv_path = Path(os.environ["TRAINING_DIR"]) / "conversations.jsonl"
    data = json.loads(conv_path.read_text(encoding="utf-8").strip())
    assert "regime" not in data
    assert "basin_coords" not in data


def test_log_conversation_includes_coevolution_metadata(tmp_path: Path) -> None:
    """New co-evolution fields appear in JSONL when provided."""
    ingest = _load_ingest_with_tmp_training_dir(tmp_path)
    response_basin = [0.015625] * 64

    asyncio.run(
        ingest.log_conversation(
            user_message="geometric test",
            response="resonance acknowledged",
            backend="ollama",
            phi=0.618,
            kappa=63.5,
            source="chat",
            routed_kernel="genesis",
            e8_primitive="E8_A1",
            response_basin=response_basin,
        )
    )

    conv_path = Path(os.environ["TRAINING_DIR"]) / "conversations.jsonl"
    data = json.loads(conv_path.read_text(encoding="utf-8").strip())

    assert data["routed_kernel"] == "genesis"
    assert data["e8_primitive"] == "E8_A1"
    assert data["response_basin"] == [round(v, 6) for v in response_basin]
    assert len(data["response_basin"]) == 64


def test_log_conversation_omits_coevolution_metadata_when_empty(tmp_path: Path) -> None:
    """Co-evolution fields must NOT appear when defaults are used (backward compat)."""
    ingest = _load_ingest_with_tmp_training_dir(tmp_path)

    asyncio.run(
        ingest.log_conversation(
            user_message="plain message",
            response="plain response",
            backend="xai",
            phi=0.5,
            kappa=64.0,
        )
    )

    conv_path = Path(os.environ["TRAINING_DIR"]) / "conversations.jsonl"
    data = json.loads(conv_path.read_text(encoding="utf-8").strip())

    assert "routed_kernel" not in data
    assert "e8_primitive" not in data
    assert "response_basin" not in data
    # Core fields still present
    assert data["user_message"] == "plain message"
    assert data["backend"] == "xai"


def test_server_source_uses_fresh_and_final_state_for_training_logs() -> None:
    server_source = (Path(__file__).parent.parent / "server.py").read_text(encoding="utf-8")
    ingest_source = (Path(__file__).parent.parent / "training" / "ingest.py").read_text(
        encoding="utf-8"
    )

    assert (
        "model_id: str = Field(default_factory=lambda: settings.modal.harvest_model)"
        in server_source
    )
    assert "response[:2000]" not in ingest_source

    assert 'fresh_state["phi"]' in server_source
    assert 'fresh_state["kappa"]' in server_source
    assert 'regime=fresh_state.get("regime", "")' in server_source
    assert "basin_coords=fresh_basin" in server_source

    assert 'final_state["phi"]' in server_source
    assert 'final_state["kappa"]' in server_source
    assert 'regime=final_state.get("regime", "")' in server_source
    assert "basin_coords=final_basin" in server_source

    assert "chat_options = consciousness._compute_llm_options()" in server_source
    assert "stream_options = consciousness._compute_llm_options()" in server_source
    assert "temperature=req.temperature" not in server_source
    assert "num_predict=req.max_tokens" not in server_source
