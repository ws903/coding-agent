import json
from unittest.mock import AsyncMock

import pytest

from agent.agents_manager import AgentRole
from agent.subagent_runner import ALLOWED_TOOL_NAMES, SubagentRunner
from agent.tools import FileTools


def _role(prompt: str = "You are a reviewer.") -> AgentRole:
    from pathlib import Path

    return AgentRole(
        name="reviewer",
        description="Reviews code",
        system_prompt=prompt,
        source_path=Path("/tmp/_no_path"),
    )


def _msg(content: str = "", tool_calls: list | None = None) -> dict:
    out: dict = {"role": "assistant", "content": content}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _tc(name: str, args: dict, call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.mark.asyncio
async def test_subagent_returns_final_content(tmp_path):
    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(return_value=_msg("Review: looks fine."))
    runner = SubagentRunner(llm, FileTools(tmp_path))

    result = await runner.run(_role(), "Review the file.")
    assert result == "Review: looks fine."


@pytest.mark.asyncio
async def test_subagent_loops_through_tool_calls(tmp_path):
    (tmp_path / "code.py").write_text("def hello(): pass\n")
    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(
        side_effect=[
            _msg(tool_calls=[_tc("read_file", {"path": "code.py"})]),
            _msg("Reviewed -- function looks fine."),
        ]
    )
    runner = SubagentRunner(llm, FileTools(tmp_path))

    result = await runner.run(_role(), "Look at code.py.")
    assert "Reviewed" in result
    assert llm.chat_with_tools.call_count == 2


@pytest.mark.asyncio
async def test_subagent_blocks_disallowed_tools(tmp_path):
    """If the model tries to call edit_file, the subagent should refuse."""
    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(
        side_effect=[
            _msg(
                tool_calls=[
                    _tc("edit_file", {"path": "x", "search": "a", "replace": "b"})
                ]
            ),
            _msg("Acknowledged."),
        ]
    )
    runner = SubagentRunner(llm, FileTools(tmp_path))
    await runner.run(_role(), "Try to edit.")

    # The second LLM call should have received a tool result rejecting the edit.
    second_call_messages = llm.chat_with_tools.call_args_list[1].args[0]
    tool_results = [m for m in second_call_messages if m.get("role") == "tool"]
    assert tool_results
    assert "may not call" in tool_results[-1]["content"]


def test_allowed_tool_set_is_read_only():
    """Subagents must never have edit/run-command tools enabled."""
    forbidden = {"edit_file", "create_file", "replace_file", "run_command"}
    assert forbidden.isdisjoint(ALLOWED_TOOL_NAMES)
