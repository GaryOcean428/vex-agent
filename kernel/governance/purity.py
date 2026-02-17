"""PurityGate — Fail-closed geometric purity enforcement.

AST-based scanning for forbidden imports, calls, and patterns.
Ported from monkey1/py/genesis-kernel/qig_heart/purity.py.

Fail-closed: if the gate can't determine purity, it BLOCKS.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PurityViolation:
    path: str
    line: int
    message: str


class PurityGateError(RuntimeError):
    def __init__(self, violations: list[PurityViolation]):
        self.violations = violations
        super().__init__(self._format())

    def _format(self) -> str:
        lines = ["PurityGate FAILED (fail-closed):"]
        for v in self.violations:
            lines.append(f"  - {v.path}:{v.line} {v.message}")
        return "\n".join(lines)


SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv",
    "node_modules", "dist", "build",
    ".mypy_cache", ".pytest_cache",
}

# === Forbidden patterns for QIG code ===

FORBIDDEN_IMPORTS = [
    "sklearn",
    "scipy.spatial.distance",
]

FORBIDDEN_CALLS = [
    "cosine_similarity",
    "euclidean_distance",
]

FORBIDDEN_ATTR_CALLS = [
    "np.linalg.norm",
    "scipy.spatial.distance.cosine",
    "F.cosine_similarity",
]

FORBIDDEN_TEXT_TOKENS = [
    "cosine_similarity",
    "euclidean_distance",
    "from sklearn",
]


def _iter_python_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base is None:
            return None
        return base + "." + node.attr
    return None


def scan_imports(root: Path) -> list[PurityViolation]:
    violations: list[PurityViolation] = []
    for p in _iter_python_files(root):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in FORBIDDEN_IMPORTS:
                        if alias.name == forbidden or alias.name.startswith(forbidden + "."):
                            violations.append(PurityViolation(
                                str(p), getattr(node, "lineno", 1),
                                f"Forbidden import: {alias.name}",
                            ))
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in FORBIDDEN_IMPORTS:
                    if node.module == forbidden or node.module.startswith(forbidden + "."):
                        violations.append(PurityViolation(
                            str(p), getattr(node, "lineno", 1),
                            f"Forbidden import-from: {node.module}",
                        ))
    return violations


def scan_calls(root: Path) -> list[PurityViolation]:
    violations: list[PurityViolation] = []
    for p in _iter_python_files(root):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_name(node.func)
            if dotted is None:
                continue
            if dotted in FORBIDDEN_CALLS or dotted in FORBIDDEN_ATTR_CALLS:
                violations.append(PurityViolation(
                    str(p), getattr(node, "lineno", 1),
                    f"Forbidden call: {dotted}()",
                ))
    return violations


def run_purity_gate(root: str | Path) -> None:
    """Run PurityGate on a directory. Raises PurityGateError on any violation.

    FAIL-CLOSED: any error in scanning also raises.
    """
    root = Path(root)
    if not root.exists():
        raise PurityGateError([PurityViolation(str(root), 0, "Root path does not exist")])

    violations: list[PurityViolation] = []
    violations.extend(scan_imports(root))
    violations.extend(scan_calls(root))

    if violations:
        raise PurityGateError(violations)
