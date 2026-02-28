#!/usr/bin/env python3
"""Verify TypeScript frozen-facts constants match Python source.

Reads ``kernel/config/frozen_facts.py`` (canonical Python source) and
``src/config/constants.ts`` (TypeScript mirror) and checks:

  1. Every Python constant appears in the TS file with the same value.
  2. Every TS constant has a corresponding Python constant (no orphans).

Exit code 0 = all constants in sync.  Non-zero = drift detected.

Usage::

    python scripts/sync_constants.py          # from repo root
    uv run python scripts/sync_constants.py   # via uv
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_FILE = REPO_ROOT / "kernel" / "config" / "frozen_facts.py"
TS_FILE = REPO_ROOT / "src" / "config" / "constants.ts"


def extract_python_constants(path: Path) -> dict[str, int | float]:
    """Extract NAME = <literal> assignments from frozen_facts.py."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    constants: dict[str, int | float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.isupper() and node.value is not None:
                val = _literal_value(node.value)
                if val is not None:
                    constants[name] = val
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    val = _literal_value(node.value)
                    if val is not None:
                        constants[target.id] = val
    return constants


def _literal_value(node: ast.expr) -> int | float | None:
    """Evaluate a constant numeric literal (int, float, or unary minus)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_value(node.operand)
        if inner is not None:
            return -inner
    return None


# Pattern matches:  export const NAME = 123 as const;
_TS_CONST_RE = re.compile(
    r"^export\s+const\s+([A-Z_][A-Z0-9_]*)\s*=\s*"
    r"(-?[\d.]+)\s+as\s+const\s*;",
    re.MULTILINE,
)


def extract_ts_constants(path: Path) -> dict[str, int | float]:
    """Extract ``export const NAME = <number> as const;`` from TS file."""
    text = path.read_text()
    constants: dict[str, int | float] = {}
    for match in _TS_CONST_RE.finditer(text):
        name = match.group(1)
        raw_value = match.group(2)
        value: int | float
        if "." in raw_value:
            value = float(raw_value)
        else:
            value = int(raw_value)
        constants[name] = value
    return constants


def main() -> int:
    if not PY_FILE.exists():
        print(f"ERROR: Python constants file not found: {PY_FILE}")
        return 1
    if not TS_FILE.exists():
        print(f"ERROR: TypeScript constants file not found: {TS_FILE}")
        return 1

    py = extract_python_constants(PY_FILE)
    ts = extract_ts_constants(TS_FILE)

    errors: list[str] = []

    # Check: every Python constant must appear in TS with the same value
    for name, py_val in sorted(py.items()):
        if name not in ts:
            errors.append(f"MISSING in TS:  {name} = {py_val}")
        elif not _values_equal(py_val, ts[name]):
            errors.append(f"VALUE MISMATCH: {name}  Python={py_val}  TS={ts[name]}")

    # Check: no orphan constants in TS that aren't in Python
    for name, ts_val in sorted(ts.items()):
        if name not in py:
            errors.append(f"ORPHAN in TS:   {name} = {ts_val}  (not in Python frozen_facts)")

    if errors:
        print("❌ Constants sync check FAILED:\n")
        for err in errors:
            print(f"  • {err}")
        print(f"\n  Python source: {PY_FILE}")
        print(f"  TS mirror:     {TS_FILE}")
        return 1

    print(f"✅ All {len(py)} frozen-facts constants in sync (Python ↔ TypeScript)")
    return 0


def _values_equal(a: int | float, b: int | float) -> bool:
    """Compare values allowing int/float equivalence (64 == 64.0 is False for identity,
    but numerically 0.7 == 0.70 is True)."""
    return float(a) == float(b)


if __name__ == "__main__":
    sys.exit(main())
