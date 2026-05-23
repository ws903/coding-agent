# src/agent/mcp_manager.py
"""Lifecycle + dispatch for external MCP (Model Context Protocol) servers.

Reads a `.mcp.json` config file (Claude Code / standard format) from the
project root, spawns each declared stdio server, and exposes their tools
to the executor alongside the built-in tool set.

Config format:
    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
          "env": {"OPTIONAL": "value"}
        }
      }
    }

Tool names exposed to the LLM are prefixed `mcp__<server>__<tool>` so
they never collide with built-in tools.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

MCP_PREFIX = "mcp__"


def load_mcp_config(project_root: Path) -> dict:
    """Read .mcp.json from the project root. Returns {} if missing/invalid."""
    config_path = project_root / ".mcp.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read .mcp.json: %s", exc)
        return {}


def _qualified_name(server: str, tool: str) -> str:
    return f"{MCP_PREFIX}{server}__{tool}"


def _parse_qualified_name(name: str) -> tuple[str, str] | None:
    if not name.startswith(MCP_PREFIX):
        return None
    remainder = name[len(MCP_PREFIX) :]
    server, _, tool = remainder.partition("__")
    if not server or not tool:
        return None
    return server, tool


class MCPManager:
    """Holds open ClientSessions for each configured MCP server."""

    def __init__(self, config: dict):
        self.config = config or {}
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[dict] = []

    @property
    def tools(self) -> list[dict]:
        return list(self._tools)

    @property
    def connected_servers(self) -> list[str]:
        return list(self._sessions.keys())

    async def connect(self) -> None:
        """Spawn server processes and fetch tool lists."""
        servers = self.config.get("mcpServers") or {}
        if not servers:
            return

        self._stack = AsyncExitStack()
        for name, spec in servers.items():
            try:
                await self._connect_one(name, spec)
            except Exception as exc:
                logger.warning("Skipping MCP server '%s': %s", name, exc)

    async def _connect_one(self, name: str, spec: dict) -> None:
        params = StdioServerParameters(
            command=spec["command"],
            args=spec.get("args", []),
            env=spec.get("env"),
        )
        assert self._stack is not None
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tool_list = await session.list_tools()
        self._sessions[name] = session
        for t in tool_list.tools:
            self._tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": _qualified_name(name, t.name),
                        "description": t.description or "",
                        "parameters": t.inputSchema
                        or {"type": "object", "properties": {}},
                    },
                }
            )

    def owns(self, tool_name: str) -> bool:
        return tool_name.startswith(MCP_PREFIX)

    async def call(self, tool_name: str, arguments: dict) -> str:
        """Dispatch an mcp__-prefixed tool call. Returns the result as text."""
        parsed = _parse_qualified_name(tool_name)
        if parsed is None:
            return f"Error: malformed MCP tool name '{tool_name}'"
        server, tool = parsed
        session = self._sessions.get(server)
        if session is None:
            return f"Error: MCP server '{server}' not connected"
        try:
            result = await session.call_tool(name=tool, arguments=arguments)
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"
        return _result_to_text(result)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._sessions.clear()


def _result_to_text(result) -> str:
    """Coerce an MCP CallToolResult into a plain string for the LLM."""
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    if getattr(result, "isError", False):
        prefix = "Error: "
    else:
        prefix = ""
    return prefix + ("\n".join(parts) if parts else "(no content)")
