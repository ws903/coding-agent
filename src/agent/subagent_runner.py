# src/agent/subagent_runner.py
"""Run a focused subagent: fresh context, read-only tools, returns text."""

from __future__ import annotations

import json

from agent.agents_manager import AgentRole
from agent.llm_client import LLMClient
from agent.tool_schemas import TOOLS
from agent.tools import FileTools

MAX_SUBAGENT_ITERATIONS = 8

# Subagents may inspect but not modify. They also cannot spawn further
# subagents or call MCP servers (avoid runaway recursion / side effects).
ALLOWED_TOOL_NAMES = {"read_file", "list_files", "search_text"}

_SUBAGENT_TOOLS = [t for t in TOOLS if t["function"]["name"] in ALLOWED_TOOL_NAMES]


class SubagentRunner:
    def __init__(self, llm: LLMClient, tools: FileTools):
        self.llm = llm
        self.tools = tools

    async def run(self, role: AgentRole, task: str) -> str:
        from agent.tool_runner import ToolRunner

        runner = ToolRunner(self.tools)
        messages = [
            {"role": "system", "content": role.system_prompt},
            {"role": "user", "content": task},
        ]

        last_content = ""
        for _ in range(MAX_SUBAGENT_ITERATIONS):
            msg = await self.llm.chat_with_tools(
                messages, _SUBAGENT_TOOLS, temperature=0.2
            )
            last_content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            assistant_msg: dict = {"role": "assistant", "content": last_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "")
                if name not in ALLOWED_TOOL_NAMES:
                    result = (
                        f"Error: subagents may not call '{name}'. "
                        f"Allowed: {sorted(ALLOWED_TOOL_NAMES)}"
                    )
                else:
                    result = await runner.dispatch(tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    }
                )

        return last_content or "(subagent returned no content)"


def parse_arguments(arguments: str | None) -> dict:
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}
