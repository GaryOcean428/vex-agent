"""T0.4 — Full kernel/ purity scan.

Scans ALL Python files under kernel/ (excluding tests/ and __pycache__)
against the complete forbidden pattern list from governance/purity.py.

This complements the pre-commit PurityGate hook (which only scans staged
files) with a full-tree regression test that catches violations introduced
by any means.
"""

from __future__ import annotations

from pathlib import Path

from kernel.governance.purity import (
    PurityViolation,
    scan_calls,
    scan_imports,
    scan_text,
)

KERNEL_ROOT = Path(__file__).parent.parent


def _non_test_files() -> list[Path]:
    """All .py files under kernel/ excluding tests/ and __pycache__."""
    excluded = {"tests", "__pycache__", ".venv", "venv"}
    result = []
    for p in KERNEL_ROOT.rglob("*.py"):
        if any(part in excluded for part in p.parts):
            continue
        result.append(p)
    return result


def _format_violations(violations: list[PurityViolation]) -> str:
    lines = [f"  {v.path}:{v.line} — {v.message}" for v in violations]
    return "\n".join(lines)


def test_no_forbidden_imports_in_kernel() -> None:
    """No forbidden imports (sklearn, scipy.spatial.distance) in kernel source."""
    violations = scan_imports(KERNEL_ROOT)
    assert not violations, f"Forbidden imports found:\n{_format_violations(violations)}"


def test_no_forbidden_calls_in_kernel() -> None:
    """No forbidden calls (cosine_similarity, np.linalg.norm, etc.) in kernel source."""
    violations = scan_calls(KERNEL_ROOT)
    assert not violations, f"Forbidden calls found:\n{_format_violations(violations)}"


def test_no_forbidden_text_tokens_in_kernel() -> None:
    """No forbidden text tokens (Adam(, LayerNorm, .flatten(), etc.) in kernel source."""
    violations = scan_text(KERNEL_ROOT)
    assert not violations, f"Forbidden text tokens found:\n{_format_violations(violations)}"


def test_purity_scan_covers_all_source_files() -> None:
    """Sanity check: scan covers a meaningful number of source files."""
    files = _non_test_files()
    assert len(files) >= 20, (
        f"Expected at least 20 kernel source files, found {len(files)}. Check KERNEL_ROOT path."
    )
