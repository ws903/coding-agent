# src/agent/tools.py
import fnmatch
from pathlib import Path

from agent.sandbox import Sandbox


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
        numbered = []
        offset = (start_line or 1) - 1
        for i, line in enumerate(lines, start=offset + 1):
            numbered.append(f"{i:>4}| {line.rstrip()}")
        return "\n".join(numbered)

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
            new_content = content.replace(search, replace, 1)
            full_path.write_text(encoding="utf-8", data=new_content)
            return True
        match_result = self._whitespace_normalized_match(content, search, replace)
        if match_result is not None:
            full_path.write_text(encoding="utf-8", data=match_result)
            return True
        return False

    def _whitespace_normalized_match(
        self, content: str, search: str, replace: str
    ) -> str | None:
        content_lines = content.splitlines(keepends=True)
        search_lines = search.splitlines()
        search_stripped = [line.lstrip() for line in search_lines]
        for i in range(len(content_lines) - len(search_lines) + 1):
            window = content_lines[i : i + len(search_lines)]
            window_stripped = [line.rstrip().lstrip() for line in window]
            if window_stripped == search_stripped:
                content_base = self._indent_of(content_lines[i])
                search_base = self._indent_of(search_lines[0]) if search_lines else 0
                indent_offset = content_base - search_base

                replace_lines = replace.splitlines(keepends=True)
                indented_replace = []
                for rline in replace_lines:
                    orig_indent = self._indent_of(rline)
                    new_indent = max(0, orig_indent + indent_offset)
                    indented_replace.append(" " * new_indent + rline.lstrip())
                result_lines = (
                    content_lines[:i]
                    + indented_replace
                    + content_lines[i + len(search_lines) :]
                )
                return "".join(result_lines)
        return None

    @staticmethod
    def _indent_of(line: str) -> int:
        return len(line) - len(line.lstrip())

    def list_files(self, directory: str = ".", pattern: str | None = None) -> list[str]:
        dir_path = self.sandbox.validate_path(directory)
        if not dir_path.is_dir():
            return []
        files = []
        for item in sorted(dir_path.rglob("*")):
            if item.is_file():
                rel = str(item.relative_to(self.project_root))
                if pattern is None or fnmatch.fnmatch(item.name, pattern):
                    if not any(
                        part.startswith(".")
                        for part in item.relative_to(self.project_root).parts
                    ):
                        files.append(rel)
        return files

    def search_text(self, query: str, path_filter: str | None = None) -> list[dict]:
        results = []
        for item in sorted(self.project_root.rglob("*")):
            if not item.is_file():
                continue
            if any(
                part.startswith(".")
                for part in item.relative_to(self.project_root).parts
            ):
                continue
            if path_filter and not fnmatch.fnmatch(item.name, path_filter):
                continue
            try:
                lines = item.read_text(encoding="utf-8", errors="replace").splitlines()
            except (UnicodeDecodeError, PermissionError):
                continue
            for i, line in enumerate(lines, 1):
                if query in line:
                    results.append(
                        {
                            "file": str(item.relative_to(self.project_root)),
                            "line": i,
                            "content": line.strip(),
                        }
                    )
        return results
