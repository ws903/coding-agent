# src/agent/tool_runner.py
"""Dispatches tool_call dicts from the LLM to FileTools/sandbox actions.

Read tools (read_file, list_files, search_text) execute live.
Edit tools (create_file, edit_file, replace_file) record FileEdit
intents in `edits` for the orchestrator to apply + lint-gate later.
run_command queues into `commands` for the orchestrator's sandbox.
"""

from __future__ import annotations

import json

from agent.models import FileEdit
from agent.tools import FileTools

MAX_OBSERVATION_CHARS = 3000


def _truncate(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n\n[... {omitted} chars truncated ...]\n\n{text[-half:]}"


class ToolRunner:
    def __init__(self, tools: FileTools):
        self.tools = tools
        self.edits: list[FileEdit] = []
        self.commands: list[str] = []

    def dispatch(self, tool_call: dict) -> str:
        """Execute a single tool call. Returns the result string for the LLM."""
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: could not parse arguments: {exc}"

        handler = self._HANDLERS.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return handler(self, args)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    def _read_file(self, args: dict) -> str:
        path = args["path"]
        start = args.get("start_line")
        end = args.get("end_line")
        try:
            content = self.tools.read_file(path, start_line=start, end_line=end)
            return _truncate(content)
        except FileNotFoundError:
            return f"File not found: {path}"

    def _list_files(self, args: dict) -> str:
        directory = args.get("directory", ".")
        pattern = args.get("pattern")
        files = self.tools.list_files(directory, pattern=pattern)
        if not files:
            return "(no files)"
        return "\n".join(files[:200])

    def _search_text(self, args: dict) -> str:
        query = args["query"]
        path_filter = args.get("path_filter")
        hits = self.tools.search_text(query, path_filter=path_filter)
        if not hits:
            return f"No matches for '{query}'"
        lines = [f"{h['file']}:{h['line']}: {h['content']}" for h in hits[:30]]
        more = f"\n... and {len(hits) - 30} more" if len(hits) > 30 else ""
        return "\n".join(lines) + more

    def _create_file(self, args: dict) -> str:
        edit = FileEdit(path=args["path"], action="create", content=args["content"])
        self.edits.append(edit)
        return f"Recorded create: {args['path']}"

    def _edit_file(self, args: dict) -> str:
        edit = FileEdit(
            path=args["path"],
            action="search_replace",
            search=args["search"],
            replace=args["replace"],
        )
        self.edits.append(edit)
        return f"Recorded edit: {args['path']}"

    def _replace_file(self, args: dict) -> str:
        edit = FileEdit(path=args["path"], action="rewrite", content=args["content"])
        self.edits.append(edit)
        return f"Recorded rewrite: {args['path']}"

    def _run_command(self, args: dict) -> str:
        cmd = args["command"]
        self.commands.append(cmd)
        return f"Recorded command: {cmd}"

    _HANDLERS = {
        "read_file": _read_file,
        "list_files": _list_files,
        "search_text": _search_text,
        "create_file": _create_file,
        "edit_file": _edit_file,
        "replace_file": _replace_file,
        "run_command": _run_command,
    }
