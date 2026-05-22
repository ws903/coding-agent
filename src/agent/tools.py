# src/agent/tools.py
import fnmatch
from pathlib import Path

from agent.sandbox import Sandbox


def _is_hidden_path(rel_path: Path) -> bool:
    """True if any component of the relative path starts with a dot."""
    return any(part.startswith(".") for part in rel_path.parts)


def _read_lines_safely(path: Path) -> list[str] | None:
    """Read file lines, returning None for binary/unreadable files."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (UnicodeDecodeError, PermissionError):
        return None


class FileTools:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.sandbox = Sandbox(project_root)

    def read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        full_path = self.sandbox.validate_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines(
            keepends=True
        )
        if start_line is not None or end_line is not None:
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            lines = lines[start:end]
        offset = (start_line or 1) - 1
        return "\n".join(
            f"{i:>4}| {line.rstrip()}" for i, line in enumerate(lines, start=offset + 1)
        )

    def write_file(self, path: str, content: str) -> None:
        full_path = self.sandbox.validate_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(encoding="utf-8", data=content)

    def edit_file(self, path: str, search: str, replace: str) -> bool:
        full_path = self.sandbox.validate_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = full_path.read_text(encoding="utf-8", errors="replace")

        if search in content:
            full_path.write_text(
                encoding="utf-8", data=content.replace(search, replace, 1)
            )
            return True

        match_result = self._whitespace_normalized_match(content, search, replace)
        if match_result is None:
            return False
        full_path.write_text(encoding="utf-8", data=match_result)
        return True

    def _whitespace_normalized_match(
        self, content: str, search: str, replace: str
    ) -> str | None:
        content_lines = content.splitlines(keepends=True)
        search_lines = search.splitlines()
        if not search_lines:
            return None
        search_stripped = [line.lstrip() for line in search_lines]

        for i in range(len(content_lines) - len(search_lines) + 1):
            window = content_lines[i : i + len(search_lines)]
            window_stripped = [line.rstrip().lstrip() for line in window]
            if window_stripped != search_stripped:
                continue

            indent_offset = self._indent_of(content_lines[i]) - self._indent_of(
                search_lines[0]
            )
            indented_replace = self._reindent(replace, indent_offset)
            return "".join(
                content_lines[:i]
                + indented_replace
                + content_lines[i + len(search_lines) :]
            )
        return None

    @staticmethod
    def _indent_of(line: str) -> int:
        return len(line) - len(line.lstrip())

    @staticmethod
    def _reindent(text: str, offset: int) -> list[str]:
        out = []
        for line in text.splitlines(keepends=True):
            indent = max(0, len(line) - len(line.lstrip()) + offset)
            out.append(" " * indent + line.lstrip())
        return out

    def list_files(self, directory: str = ".", pattern: str | None = None) -> list[str]:
        dir_path = self.sandbox.validate_path(directory)
        if not dir_path.is_dir():
            return []
        files = []
        for item in sorted(dir_path.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(self.project_root)
            if _is_hidden_path(rel):
                continue
            if pattern is not None and not fnmatch.fnmatch(item.name, pattern):
                continue
            files.append(str(rel))
        return files

    def search_text(self, query: str, path_filter: str | None = None) -> list[dict]:
        results = []
        for item in sorted(self.project_root.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(self.project_root)
            if _is_hidden_path(rel):
                continue
            if path_filter is not None and not fnmatch.fnmatch(item.name, path_filter):
                continue
            lines = _read_lines_safely(item)
            if lines is None:
                continue
            for i, line in enumerate(lines, 1):
                if query in line:
                    results.append(
                        {"file": str(rel), "line": i, "content": line.strip()}
                    )
        return results
