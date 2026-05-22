from unittest.mock import AsyncMock

import pytest

from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.models import Step
from agent.tools import FileTools


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _msg(content: str = "", tool_calls: list | None = None) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=LLMClient)
    return client


@pytest.fixture
def tools(tmp_path):
    return FileTools(tmp_path)


@pytest.fixture
def executor(mock_client, tools):
    return Executor(mock_client, tools)


@pytest.mark.asyncio
async def test_execute_search_replace(executor, mock_client, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        '@app.route("/")\ndef index():\n    return "hello"\n'
    )
    mock_client.chat_with_tools = AsyncMock(
        side_effect=[
            _msg(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "edit_file",
                        {
                            "path": "src/app.py",
                            "search": '@app.route("/")\ndef index():\n    return "hello"',
                            "replace": '@app.route("/")\ndef index():\n    return "hello"\n\n@app.route("/health")\ndef health():\n    return {"status": "ok"}',
                        },
                    )
                ]
            ),
            _msg(content="Added health endpoint."),
        ]
    )

    step = Step(id=1, action="Add health endpoint", files_needed=["src/app.py"])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "search_replace"
    assert result.file_edits[0].path == "src/app.py"


@pytest.mark.asyncio
async def test_execute_create_file(executor, mock_client, tmp_path):
    mock_client.chat_with_tools = AsyncMock(
        side_effect=[
            _msg(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "create_file",
                        {
                            "path": "tests/test_health.py",
                            "content": "def test_health(client):\n    pass\n",
                        },
                    )
                ]
            ),
            _msg(content="Test file created."),
        ]
    )

    step = Step(id=1, action="Create test file", files_needed=[])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "create"
    assert result.file_edits[0].path == "tests/test_health.py"


@pytest.mark.asyncio
async def test_execute_rewrite(executor, mock_client, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text("DEBUG = True\n")
    mock_client.chat_with_tools = AsyncMock(
        side_effect=[
            _msg(
                tool_calls=[
                    _tool_call(
                        "c1",
                        "replace_file",
                        {
                            "path": "src/config.py",
                            "content": "DEBUG = False\nPORT = 8080\n",
                        },
                    )
                ]
            ),
            _msg(content="Rewrote config."),
        ]
    )

    step = Step(id=1, action="Update config", files_needed=["src/config.py"])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "rewrite"


@pytest.mark.asyncio
async def test_execute_includes_step_action_in_prompt(executor, mock_client, tmp_path):
    mock_client.chat_with_tools = AsyncMock(return_value=_msg(content="no edits"))

    step = Step(id=1, action="Change x", files_needed=["code.py"])
    await executor.execute(step)

    messages = mock_client.chat_with_tools.call_args.args[0]
    # system(0) + user(1)
    assert "Change x" in messages[1]["content"]
    assert "code.py" in messages[1]["content"]


@pytest.mark.asyncio
async def test_execute_with_errors_includes_error(executor, mock_client, tmp_path):
    mock_client.chat_with_tools = AsyncMock(return_value=_msg(content="no edits"))
    step = Step(id=1, action="Fix bug", files_needed=[])
    await executor.execute(step, errors="NameError: name 'foo' is not defined")

    messages = mock_client.chat_with_tools.call_args.args[0]
    assert "NameError" in messages[1]["content"]


@pytest.mark.asyncio
async def test_execute_tool_loop_returns_tool_results(executor, mock_client, tmp_path):
    (tmp_path / "foo.py").write_text("x = 1\n")
    mock_client.chat_with_tools = AsyncMock(
        side_effect=[
            _msg(tool_calls=[_tool_call("c1", "read_file", {"path": "foo.py"})]),
            _msg(content="Inspected the file."),
        ]
    )

    step = Step(id=1, action="Look at foo.py", files_needed=[])
    result = await executor.execute(step)

    # Two LLM calls: one with read_file tool_call, one final with content.
    assert mock_client.chat_with_tools.call_count == 2
    assert result.file_edits == []
    assert "Inspected" in result.explanation


@pytest.mark.asyncio
async def test_execute_records_commands(executor, mock_client, tmp_path):
    mock_client.chat_with_tools = AsyncMock(
        side_effect=[
            _msg(
                tool_calls=[
                    _tool_call("c1", "run_command", {"command": "pytest tests/"})
                ]
            ),
            _msg(content="Queued tests."),
        ]
    )

    step = Step(id=1, action="Run tests", files_needed=[])
    result = await executor.execute(step)

    assert result.commands == ["pytest tests/"]
