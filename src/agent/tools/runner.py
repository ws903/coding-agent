# src/agent/tool_runner.py
"""Dispatches tool_call dicts from the LLM to FileTools/sandbox actions.

Read tools (read_file, list_files, search_text) execute live.
Edit tools (create_file, edit_file, replace_file) record FileEdit
intents in `edits` for the orchestrator to apply + lint-gate later.
run_command queues into `commands` for the orchestrator's sandbox.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent.core.models import FileEdit
from agent.tools.filesystem import FileTools

if TYPE_CHECKING:
    # Type-only imports break the cycle:
    #   tools.runner <- core.executor <- core.__init__ <- core.subagent_runner
    # SubagentRunner / MCPManager / SkillsManager / AgentsManager are only
    # used as annotations on ToolRunner.__init__.
    from agent.core.subagent_runner import SubagentRunner
    from agent.extensions.agents import AgentsManager
    from agent.extensions.mcp import MCPManager
    from agent.extensions.skills import SkillsManager

MAX_OBSERVATION_CHARS = 3000


def _truncate(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n\n[... {omitted} chars truncated ...]\n\n{text[-half:]}"


class ToolRunner:
    def __init__(
        self,
        tools: FileTools,
        mcp: MCPManager | None = None,
        skills: SkillsManager | None = None,
        agents: AgentsManager | None = None,
        subagent_runner: SubagentRunner | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
    ):
        self.tools = tools
        self.mcp = mcp
        self.skills = skills
        self.agents = agents
        self.subagent_runner = subagent_runner
        self.on_tool_call = on_tool_call
        self.edits: list[FileEdit] = []
        self.commands: list[str] = []

    async def dispatch(self, tool_call: dict) -> str:
        """Execute a single tool call. Returns the result string for the LLM."""
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: could not parse arguments: {exc}"

        if self.on_tool_call is not None:
            try:
                self.on_tool_call(name, args)
            except Exception:  # noqa: BLE001 -- display callback must never break dispatch
                pass

        if self.mcp is not None and self.mcp.owns(name):
            return _truncate(await self.mcp.call(name, args))

        if name == "spawn_agent":
            return await self._spawn_agent(args)

        handler = self._HANDLERS.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return handler(self, args)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    async def _spawn_agent(self, args: dict) -> str:
        if self.agents is None or self.subagent_runner is None:
            return "Error: no subagent roles configured for this project"
        role_name = args.get("role", "")
        task = args.get("task", "")
        role = self.agents.get(role_name)
        if role is None:
            available = ", ".join(r.name for r in self.agents.roles) or "(none)"
            return (
                f"Error: subagent role '{role_name}' not found. Available: {available}"
            )
        try:
            return _truncate(await self.subagent_runner.run(role, task))
        except Exception as exc:
            return f"Error: subagent failed: {type(exc).__name__}: {exc}"

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

    def _read_skill(self, args: dict) -> str:
        if self.skills is None:
            return "Error: no skills configured for this project"
        name = args["name"]
        skill = self.skills.get(name)
        if skill is None:
            available = ", ".join(s.name for s in self.skills.skills) or "(none)"
            return f"Error: skill '{name}' not found. Available: {available}"
        return skill.body or "(skill body is empty)"

    _HANDLERS = {
        "read_file": _read_file,
        "list_files": _list_files,
        "search_text": _search_text,
        "create_file": _create_file,
        "edit_file": _edit_file,
        "replace_file": _replace_file,
        "run_command": _run_command,
        "read_skill": _read_skill,
    }
