from __future__ import annotations

from unittest.mock import patch

from kernel.consciousness import harvest_bridge


def test_forward_to_harvest_drops_invalid_source(tmp_path):
    harvest_bridge._HARVEST_PENDING_DIR = tmp_path

    harvest_bridge.forward_to_harvest(
        "This text is long enough to be queued.",
        source="forage",
    )

    assert list(tmp_path.iterdir()) == []


def test_forward_to_harvest_drops_too_short_text(tmp_path):
    harvest_bridge._HARVEST_PENDING_DIR = tmp_path

    harvest_bridge.forward_to_harvest("too short", source="foraging")

    assert list(tmp_path.iterdir()) == []


def test_forward_to_harvest_sanitizes_filename(tmp_path):
    harvest_bridge._HARVEST_PENDING_DIR = tmp_path

    with patch("kernel.consciousness.harvest_bridge.uuid.uuid4", return_value="../bad:id"):
        harvest_bridge.forward_to_harvest(
            "This text is long enough to be queued safely.",
            source="foraging",
        )

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "badid.jsonl"
