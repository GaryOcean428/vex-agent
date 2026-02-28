"""Tests for constants centralisation and Python ↔ TypeScript sync.

Verifies:
  1. ``kernel.constants`` re-exports every constant from ``kernel.config.frozen_facts``
  2. ``scripts/sync_constants.py`` passes (TS mirrors Python)
  3. ``kernel/constants.py`` does not define any new constants (re-export only)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kernel import constants as kernel_constants
from kernel.config import frozen_facts

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _frozen_names() -> list[str]:
    """All UPPER_CASE names exported by frozen_facts (the canonical source)."""
    return [name for name in dir(frozen_facts) if name.isupper() and not name.startswith("_")]


class TestKernelConstants:
    """Verify kernel/constants.py re-exports everything from frozen_facts."""

    def test_all_frozen_facts_available(self) -> None:
        """Every constant in frozen_facts must be accessible via kernel.constants."""
        missing = [name for name in _frozen_names() if not hasattr(kernel_constants, name)]
        assert not missing, f"kernel.constants missing re-exports: {missing}"

    def test_values_match(self) -> None:
        """Re-exported values must be identical to frozen_facts originals."""
        for name in _frozen_names():
            py_val = getattr(frozen_facts, name)
            re_val = getattr(kernel_constants, name)
            assert py_val is re_val, f"{name}: frozen_facts={py_val!r}, kernel.constants={re_val!r}"

    def test_no_extra_constants(self) -> None:
        """kernel.constants must not introduce new constants — re-export only."""
        frozen = set(_frozen_names())
        kernel = {
            name for name in dir(kernel_constants) if name.isupper() and not name.startswith("_")
        }
        extra = kernel - frozen
        assert not extra, (
            f"kernel.constants has constants not in frozen_facts: {extra}. "
            f"Add new constants to kernel/config/frozen_facts.py instead."
        )


class TestSyncScript:
    """Verify the sync_constants.py CI script works."""

    def test_sync_script_passes(self) -> None:
        """scripts/sync_constants.py must exit 0 (all constants in sync)."""
        script = REPO_ROOT / "scripts" / "sync_constants.py"
        if not script.exists():
            pytest.skip("sync_constants.py not found")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"sync_constants.py failed:\n{result.stdout}\n{result.stderr}"
        )
