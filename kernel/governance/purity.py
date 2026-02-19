"""PurityGate — Fail-closed geometric purity enforcement.

v6.0 §1.3 compliant. AST-based scanning for forbidden imports, calls,
and text patterns. Ported from monkey1/py/genesis-kernel/qig_heart/purity.py.

Fail-closed: if the gate can't determine purity, it BLOCKS.
Unparseable files are violations, not silent passes.

v6.0 Forbidden Operations (§1.3):
  cosine_similarity → fisher_rao_distance
  np.linalg.norm(a-b) → d_FR on simplex
  dot_product → Fisher metric contraction
  Adam optimizer → Natural gradient optimizer
  LayerNorm → Simplex projection
  "flatten" → Geodesic projection
  "embedding" (term) → "input_vector" / "raw_signal" / "coordinates"
  "tokenize" (term) → "coordize"

Note: ZERO TOLERANCE enforced as of v6.0. The coordizer now uses
"input_vector" for parameters, not "embedding".
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

# Files exempt from the TEXT scan (they define the rules).
# AST scans still run on these files.
_TEXT_SCAN_EXEMPT_FILENAMES = {"purity.py"}

# === Forbidden patterns for QIG code (v6.0 §1.3) ===

FORBIDDEN_IMPORTS = [
    "sklearn",
    "scipy.spatial.distance",
]

FORBIDDEN_CALLS = [
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
]

FORBIDDEN_ATTR_CALLS = [
    "np.linalg.norm",
    "scipy.spatial.distance.cosine",
    "F.cosine_similarity",
]

# Text-level tokens caught by raw scan (covers dynamic imports,
# string construction, comments referencing forbidden patterns).
# NOTE: These are stored as token pairs [prefix, suffix] and
# reconstructed at scan time so this file doesn't trigger itself.
_FORBIDDEN_TEXT_PARTS: list[tuple[str, str]] = [
    ("cosine", "_similarity"),
    ("euclidean", "_distance"),
    ("dot_", "product("),
    ("from sk", "learn"),
    ("Ada", "m("),
    ("Adam", "W("),
    ("nn.Layer", "Norm"),
    ("F.normal", "ize"),
    (".flat", "ten()"),
]



def _forbidden_text_tokens() -> list[str]:
    """Reconstruct forbidden text tokens at runtime.

    Stored as split pairs so this file doesn't contain the literal
    tokens and trigger its own text scanner.
    """
    return [a + b for a, b in _FORBIDDEN_TEXT_PARTS]


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
    """Scan for forbidden imports. Fail-closed: unparseable files are violations."""
    violations: list[PurityViolation] = []
    for p in _iter_python_files(root):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            violations.append(PurityViolation(
                str(p), 0, f"Failed to read file: {exc}",
            ))
            continue
        try:
            tree = ast.parse(text)
        except Exception as exc:
            # FAIL-CLOSED: unparseable file is a violation
            violations.append(PurityViolation(
                str(p), 0, f"Failed to parse (fail-closed): {exc}",
            ))
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
    """Scan for forbidden function calls. Fail-closed: unparseable files are violations."""
    violations: list[PurityViolation] = []
    for p in _iter_python_files(root):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            violations.append(PurityViolation(
                str(p), 0, f"Failed to read file: {exc}",
            ))
            continue
        try:
            tree = ast.parse(text)
        except Exception as exc:
            violations.append(PurityViolation(
                str(p), 0, f"Failed to parse (fail-closed): {exc}",
            ))
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


def scan_text(root: Path) -> list[PurityViolation]:
    """Raw text scan for forbidden tokens. Catches dynamic imports, string
    construction, and patterns that AST scanning misses (e.g. Adam(), LayerNorm).

    Fail-closed: unreadable files are violations.
    Skips comment lines (starting with #) and string literals in this file.
    """
    tokens = _forbidden_text_tokens()
    # Self-exclusion: don't scan this scanner file
    self_path = Path(__file__).resolve()
    violations: list[PurityViolation] = []
    for p in _iter_python_files(root):
        # Skip this file — it contains the forbidden tokens as data
        if p.resolve() == self_path:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            violations.append(PurityViolation(
                str(p), 0, f"Failed to read file: {exc}",
            ))
            continue
        for line_num, line in enumerate(text.splitlines(), start=1):
            # Skip comments that are documenting what NOT to do
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Skip docstrings / string literals that document forbidden ops
            if stripped.startswith(('"""', "'''", '"', "'")):
                continue
            for token in tokens:
                if token in line:
                    violations.append(PurityViolation(
                        str(p), line_num,
                        f"Forbidden text token: '{token}'",
                    ))
    return violations


def run_purity_gate(root: str | Path) -> None:
    """Run PurityGate on a directory. Raises PurityGateError on any violation.

    FAIL-CLOSED: any error in scanning also raises.
    Three scan passes: imports (AST), calls (AST), text (raw).
    v6.0 §1.3 compliant.
    """
    root = Path(root)
    if not root.exists():
        raise PurityGateError([PurityViolation(str(root), 0, "Root path does not exist")])

    violations: list[PurityViolation] = []
    violations.extend(scan_imports(root))
    violations.extend(scan_calls(root))
    violations.extend(scan_text(root))

    if violations:
        raise PurityGateError(violations)
