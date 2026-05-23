# src/agent/codebase_index.py
"""Lazy Python-only structural index of the project.

Walks `.py` files under the project root, parses each with the stdlib
`ast` module, and extracts top-level symbols (functions, classes, and
selected imports). The index is rendered as a markdown section that
the orchestrator injects into the planner's project context so plans
can reference symbols by name without first reading every file.

Scope decisions for this MVP:
- Python only. tree-sitter and other-language support is a follow-up.
- Top-level only (no nested classes / methods). The signal-to-noise
  ratio drops fast once you descend into method bodies.
- Imports collapsed to module names. Helps the planner spot deps
  without spamming the context with every `from foo import a, b, c`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".agent",
    "htmlcov",
}

MAX_FILES_IN_SUMMARY = 200


@dataclass(frozen=True)
class Symbol:
    kind: str  # "function" | "class" | "import"
    name: str
    line: int


@dataclass(frozen=True)
class FileEntry:
    path: str
    symbols: tuple[Symbol, ...]


def _extract_symbols(tree: ast.AST) -> list[Symbol]:
    symbols: list[Symbol] = []
    imports: list[Symbol] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(Symbol("function", node.name, node.lineno))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol("class", node.name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(Symbol("import", alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(Symbol("import", node.module, node.lineno))
    # Imports first (alphabetized), then functions/classes in source order.
    seen: set[str] = set()
    deduped_imports = []
    for imp in sorted(imports, key=lambda s: s.name):
        if imp.name in seen:
            continue
        seen.add(imp.name)
        deduped_imports.append(imp)
    return deduped_imports + symbols


class CodebaseIndex:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.entries: list[FileEntry] = []
        self._build()

    def _build(self) -> None:
        if not self.project_root.is_dir():
            return
        for path in sorted(self._iter_python_files()):
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8", errors="replace"),
                    filename=str(path),
                )
            except (SyntaxError, OSError):
                continue
            symbols = _extract_symbols(tree)
            if not symbols:
                continue
            rel = str(path.relative_to(self.project_root))
            self.entries.append(FileEntry(rel, tuple(symbols)))

    def _iter_python_files(self):
        for path in self.project_root.rglob("*.py"):
            if any(
                part in IGNORED_DIRS or part.startswith(".")
                for part in path.relative_to(self.project_root).parts[:-1]
            ):
                continue
            yield path

    def summary(self) -> str:
        if not self.entries:
            return ""
        lines = ["## Codebase symbol map", ""]
        for entry in self.entries[:MAX_FILES_IN_SUMMARY]:
            lines.append(f"### {entry.path}")
            for sym in entry.symbols:
                lines.append(f"- {sym.kind}: `{sym.name}` (L{sym.line})")
            lines.append("")
        if len(self.entries) > MAX_FILES_IN_SUMMARY:
            lines.append(
                f"_({len(self.entries) - MAX_FILES_IN_SUMMARY} more files omitted)_"
            )
        return "\n".join(lines)
