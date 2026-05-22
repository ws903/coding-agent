import json

import pytest

from agent.tool_runner import ToolRunner
from agent.tools import FileTools


def _tc(name: str, args: dict, call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.fixture
def runner(tmp_path):
    return ToolRunner(FileTools(tmp_path))


@pytest.mark.asyncio
async def test_read_file_returns_content(runner, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    result = await runner.dispatch(_tc("read_file", {"path": "a.py"}))
    assert "x = 1" in result


@pytest.mark.asyncio
async def test_read_file_not_found(runner):
    result = await runner.dispatch(_tc("read_file", {"path": "missing.py"}))
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_list_files(runner, tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    result = await runner.dispatch(_tc("list_files", {"directory": "."}))
    assert "a.py" in result
    assert "b.py" in result


@pytest.mark.asyncio
async def test_search_text_hit(runner, tmp_path):
    (tmp_path / "code.py").write_text("def hello():\n    pass\n")
    result = await runner.dispatch(_tc("search_text", {"query": "hello"}))
    assert "code.py" in result
    assert "hello" in result


@pytest.mark.asyncio
async def test_search_text_no_matches(runner, tmp_path):
    (tmp_path / "code.py").write_text("x = 1\n")
    result = await runner.dispatch(_tc("search_text", {"query": "nonexistent_xyz"}))
    assert "No matches" in result


@pytest.mark.asyncio
async def test_create_file_records_edit(runner):
    result = await runner.dispatch(
        _tc("create_file", {"path": "new.py", "content": "x = 1"})
    )
    assert "Recorded create" in result
    assert len(runner.edits) == 1
    assert runner.edits[0].action == "create"
    assert runner.edits[0].path == "new.py"
    assert runner.edits[0].content == "x = 1"


@pytest.mark.asyncio
async def test_edit_file_records_search_replace(runner):
    result = await runner.dispatch(
        _tc(
            "edit_file",
            {"path": "foo.py", "search": "old", "replace": "new"},
        )
    )
    assert "Recorded edit" in result
    assert len(runner.edits) == 1
    assert runner.edits[0].action == "search_replace"
    assert runner.edits[0].search == "old"
    assert runner.edits[0].replace == "new"


@pytest.mark.asyncio
async def test_replace_file_records_rewrite(runner):
    result = await runner.dispatch(
        _tc("replace_file", {"path": "foo.py", "content": "x = 2"})
    )
    assert "Recorded rewrite" in result
    assert runner.edits[0].action == "rewrite"


@pytest.mark.asyncio
async def test_run_command_records_command(runner):
    result = await runner.dispatch(_tc("run_command", {"command": "pytest -q"}))
    assert "Recorded command" in result
    assert runner.commands == ["pytest -q"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error(runner):
    result = await runner.dispatch(_tc("bogus_tool", {}))
    assert "unknown tool" in result.lower()


@pytest.mark.asyncio
async def test_bad_json_args_returns_error(runner):
    tc = {
        "id": "x",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{not valid json"},
    }
    result = await runner.dispatch(tc)
    assert "could not parse" in result.lower()


@pytest.mark.asyncio
async def test_multiple_edits_accumulate(runner):
    await runner.dispatch(_tc("create_file", {"path": "a.py", "content": "1"}))
    await runner.dispatch(_tc("create_file", {"path": "b.py", "content": "2"}))
    assert len(runner.edits) == 2
    assert runner.edits[0].path == "a.py"
    assert runner.edits[1].path == "b.py"


@pytest.mark.asyncio
async def test_mcp_tool_dispatched_to_manager(tmp_path):
    """An mcp__-prefixed tool call routes through the MCPManager."""
    from unittest.mock import AsyncMock

    mcp = AsyncMock()
    mcp.owns = lambda name: name.startswith("mcp__")
    mcp.call = AsyncMock(return_value="mcp result text")

    runner = ToolRunner(FileTools(tmp_path), mcp=mcp)
    result = await runner.dispatch(
        _tc("mcp__filesystem__read_file", {"path": "/tmp/x"})
    )

    assert result == "mcp result text"
    mcp.call.assert_awaited_once_with("mcp__filesystem__read_file", {"path": "/tmp/x"})


@pytest.mark.asyncio
async def test_read_skill_returns_body(tmp_path):
    from agent.skills_manager import SkillsManager

    skill_dir = tmp_path / ".agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "review.md").write_text(
        "---\nname: review\ndescription: x\n---\nBe thorough."
    )
    runner = ToolRunner(FileTools(tmp_path), skills=SkillsManager(tmp_path))
    result = await runner.dispatch(_tc("read_skill", {"name": "review"}))
    assert "Be thorough." in result


@pytest.mark.asyncio
async def test_read_skill_unknown_returns_error(tmp_path):
    from agent.skills_manager import SkillsManager

    runner = ToolRunner(FileTools(tmp_path), skills=SkillsManager(tmp_path))
    result = await runner.dispatch(_tc("read_skill", {"name": "missing"}))
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_read_skill_without_manager_errors(tmp_path):
    runner = ToolRunner(FileTools(tmp_path))
    result = await runner.dispatch(_tc("read_skill", {"name": "any"}))
    assert "no skills configured" in result.lower()
