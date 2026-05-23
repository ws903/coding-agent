# tests/test_cli.py

import json
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.cli import (
    SLASH_COMMANDS,
    _approve_plan,
    _find_project_root,
    _format_tool_call,
    _llm_classify_intent,
    _looks_like_chat,
    _route_input,
    _show_config,
    _show_history,
    build_orchestrator,
    parse_args,
    run_autonomous,
    run_interactive,
)
from agent.models import Plan, Step


def _mock_orch() -> MagicMock:
    """MagicMock orchestrator with awaitable mcp lifecycle methods."""
    m = MagicMock()
    m.executor.mcp.connect = AsyncMock()
    m.executor.mcp.close = AsyncMock()
    m.executor.mcp.connected_servers = []
    m.executor.skills.skills = []
    m.executor.agents.roles = []
    return m


def _all_printed(mock_console) -> str:
    """Flatten everything printed (including Panel/Markdown contents) for
    substring search in tests. Walks .renderable on Rich renderables so
    Panel-wrapped text is still discoverable."""
    parts: list[str] = []
    for call in mock_console.print.call_args_list:
        if not call.args:
            continue
        first = call.args[0]
        renderable = getattr(first, "renderable", None)
        if renderable is not None:
            parts.append(str(renderable))
        else:
            parts.append(str(first))
    return "\n".join(parts)


def test_find_project_root_walks_up_to_git_dir(tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "src" / "deep" / "nested"
    nested.mkdir(parents=True)
    assert _find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_returns_input_when_no_git(tmp_path):
    nested = tmp_path / "no_git_here"
    nested.mkdir()
    # No .git anywhere in the chain -- returns the original.
    assert _find_project_root(nested) == nested.resolve()


def test_find_project_root_uses_dir_itself_when_repo_root(tmp_path):
    (tmp_path / ".git").mkdir()
    assert _find_project_root(tmp_path) == tmp_path.resolve()


def test_looks_like_chat_greetings():
    for text in ["hi", "hello", "hey", "yo", "thanks", "thank you", "bye"]:
        assert _looks_like_chat(text), text


def test_looks_like_chat_punctuation_tolerated():
    assert _looks_like_chat("hi!")
    assert _looks_like_chat("Hey.")
    assert _looks_like_chat("thanks?")


def test_looks_like_chat_rejects_tasks():
    """Anything that smells like real work falls through to the planner."""
    for text in [
        "add a docstring to foo.py",
        "fix the bug in auth",
        "refactor the orchestrator",
        "what does the lint gate do?",
        "explain the architecture",
        "create a new test file",
    ]:
        assert not _looks_like_chat(text), text


def test_looks_like_chat_rejects_long_input():
    """Even if it starts with 'hi', long input is probably a real task."""
    assert not _looks_like_chat("hi can you add tests for foo")


def test_looks_like_chat_rejects_empty():
    assert not _looks_like_chat("")
    assert not _looks_like_chat("   ")


# --- intent routing tests ---


def _orch_with_classify_response(text: str) -> MagicMock:
    """A mock orch whose quick_chat_stream returns the given classifier verdict."""
    orch = _mock_orch()
    orch.executor.llm.quick_chat_stream = AsyncMock(return_value=text)
    return orch


@pytest.mark.asyncio
async def test_route_input_obvious_chat_skips_classifier():
    orch = _orch_with_classify_response("TASK")  # should NOT be called
    assert await _route_input(orch, "hi") == "chat"
    orch.executor.llm.quick_chat_stream.assert_not_called()


@pytest.mark.asyncio
async def test_route_input_long_input_skips_classifier():
    orch = _orch_with_classify_response("CHAT")  # should NOT be called
    long = "add a docstring to every function in src and then run the tests"
    assert await _route_input(orch, long) == "task"
    orch.executor.llm.quick_chat_stream.assert_not_called()


@pytest.mark.asyncio
async def test_route_input_ambiguous_calls_classifier_chat():
    orch = _orch_with_classify_response("CHAT")
    assert await _route_input(orch, "how are you") == "chat"
    orch.executor.llm.quick_chat_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_route_input_ambiguous_calls_classifier_task():
    orch = _orch_with_classify_response("TASK")
    assert await _route_input(orch, "explain the orchestrator") == "task"
    orch.executor.llm.quick_chat_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_route_input_classifier_failure_falls_back_to_task():
    orch = _mock_orch()
    orch.executor.llm.quick_chat_stream = AsyncMock(side_effect=RuntimeError("net"))
    # Ambiguous input. Classifier raises -- safer to treat as task than to
    # swallow real work into fast-chat.
    assert await _route_input(orch, "how about a refactor") == "task"


@pytest.mark.asyncio
async def test_llm_classify_intent_parses_case_insensitively():
    orch = _orch_with_classify_response("chat\n")
    assert await _llm_classify_intent(orch, "hey") == "chat"
    orch = _orch_with_classify_response("Task.")
    assert await _llm_classify_intent(orch, "fix it") == "task"


# --- tool-call icon rendering ---


def test_format_tool_call_known_tool_with_path():
    out = _format_tool_call("read_file", {"path": "src/agent/cli.py"})
    assert "read_file" in out
    assert "src/agent/cli.py" in out


def test_format_tool_call_command_arg():
    out = _format_tool_call("run_command", {"command": "pytest -q"})
    assert "pytest -q" in out


def test_format_tool_call_unknown_tool_still_renders():
    out = _format_tool_call("bogus_tool", {})
    assert "bogus_tool" in out


def test_format_tool_call_mcp_tool_uses_plug_icon():
    """MCP tools get a distinct icon since they're external."""
    out = _format_tool_call("mcp__filesystem__read", {"path": "/tmp/x"})
    assert "mcp__filesystem__read" in out
    assert "/tmp/x" in out


def test_format_tool_call_truncates_long_paths():
    long_path = "x" * 200
    out = _format_tool_call("read_file", {"path": long_path})
    # Should not contain the raw long string -- truncated with ellipsis.
    assert "..." in out
    assert "x" * 200 not in out


def test_parse_args_interactive():
    args = parse_args([])
    assert args.auto is False
    assert args.task is None


def test_parse_args_autonomous():
    args = parse_args(["--auto", "--task", "Fix the bug"])
    assert args.auto is True
    assert args.task == "Fix the bug"


def test_parse_args_max_steps():
    args = parse_args(["--auto", "--task", "Fix", "--max-steps", "10"])
    assert args.max_steps == 10


def test_parse_args_custom_model():
    args = parse_args(["--model", "phi4:14b"])
    assert args.model == "phi4:14b"


def test_parse_args_custom_base_url():
    args = parse_args(["--base-url", "http://localhost:5000/v1"])
    assert args.base_url == "http://localhost:5000/v1"


def test_parse_args_project_dir():
    args = parse_args(["--project", "/tmp/myproject"])
    assert args.project == "/tmp/myproject"


def test_parse_args_step_mode():
    args = parse_args(["--step"])
    assert args.step is True


def test_build_orchestrator(tmp_path):
    orch = build_orchestrator(
        project_root=tmp_path,
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
    )
    assert orch is not None
    assert orch.project_root == tmp_path


# --- _approve_plan tests ---


@patch("agent.cli.Prompt")
@patch("agent.cli.console")
def test_approve_plan_accepted(mock_console, mock_prompt):
    mock_prompt.ask.return_value = "y"
    plan = Plan(
        goal="Refactor module",
        steps=[
            Step(id=1, action="Read file", files_needed=["src/main.py"]),
            Step(id=2, action="Write tests", files_needed=[]),
        ],
    )
    result = _approve_plan(plan)
    assert result is True
    mock_prompt.ask.assert_called_once_with("Proceed?", choices=["y", "n"], default="y")
    # Goal text lives inside the Panel renderable.
    assert "Refactor module" in _all_printed(mock_console)


@patch("agent.cli.Prompt")
@patch("agent.cli.console")
def test_approve_plan_rejected(mock_console, mock_prompt):
    mock_prompt.ask.return_value = "n"
    plan = Plan(
        goal="Delete everything",
        steps=[Step(id=1, action="Delete all", files_needed=[])],
    )
    result = _approve_plan(plan)
    assert result is False


@patch("agent.cli.Prompt")
@patch("agent.cli.console")
def test_approve_plan_displays_files(mock_console, mock_prompt):
    mock_prompt.ask.return_value = "y"
    plan = Plan(
        goal="Update configs",
        steps=[
            Step(
                id=1,
                action="Edit config",
                files_needed=["config.yaml", "settings.json"],
            ),
        ],
    )
    _approve_plan(plan)
    printed = _all_printed(mock_console)
    assert "config.yaml" in printed
    assert "settings.json" in printed


@patch("agent.cli.Prompt")
@patch("agent.cli.console")
def test_approve_plan_step_without_files(mock_console, mock_prompt):
    mock_prompt.ask.return_value = "y"
    plan = Plan(
        goal="Simple task",
        steps=[Step(id=1, action="Think about it", files_needed=[])],
    )
    _approve_plan(plan)
    # Should not print any "Files:" line
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert not any("Files:" in c for c in print_calls)


# --- _show_config tests ---


@patch("agent.cli.console")
def test_show_config(mock_console):
    mock_db = MagicMock()
    mock_db.get_config.return_value = "(not set)"
    mock_orch = _mock_orch()
    mock_orch.db = mock_db
    _show_config(mock_orch)
    assert mock_console.print.call_count == 6  # 1 header + 5 config keys
    mock_db.get_config.assert_any_call("planner_base_url", "(not set)")
    mock_db.get_config.assert_any_call("planner_model", "(not set)")
    mock_db.get_config.assert_any_call("executor_base_url", "(not set)")
    mock_db.get_config.assert_any_call("executor_model", "(not set)")
    mock_db.get_config.assert_any_call("verify_commands", "(not set)")


@patch("agent.cli.console")
def test_show_config_with_values(mock_console):
    mock_db = MagicMock()
    mock_db.get_config.side_effect = lambda key, default: {
        "planner_base_url": "http://localhost:11434/v1",
        "planner_model": "qwen3:14b",
        "executor_base_url": "http://localhost:11434/v1",
        "executor_model": "qwen3:14b",
        "verify_commands": '["pytest"]',
    }.get(key, default)
    mock_orch = _mock_orch()
    mock_orch.db = mock_db
    _show_config(mock_orch)
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("qwen3:14b" in c for c in print_calls)


# --- _show_history tests ---


@patch("agent.cli.console")
def test_show_history_empty(mock_console):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_cursor
    mock_orch = _mock_orch()
    mock_orch.db = mock_db
    _show_history(mock_orch)
    mock_console.print.assert_called_once_with("No conversation history.")


@patch("agent.cli.console")
def test_show_history_with_rows(mock_console):
    rows = [
        {
            "id": "abc123",
            "started_at": "2025-01-01 12:00",
            "mode": "interactive",
            "task": "Fix bug",
            "status": "completed",
        },
        {
            "id": "def456",
            "started_at": "2025-01-02 14:00",
            "mode": "autonomous",
            "task": "Add tests",
            "status": "failed",
        },
    ]
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_cursor
    mock_orch = _mock_orch()
    mock_orch.db = mock_db
    _show_history(mock_orch)
    assert mock_console.print.call_count == 2
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("Fix bug" in c for c in print_calls)
    assert any("Add tests" in c for c in print_calls)
    assert any("completed" in c for c in print_calls)
    assert any("failed" in c for c in print_calls)


# --- run_interactive tests ---


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_quit(mock_console, mock_prompt, mock_build):
    mock_prompt.ask.return_value = "/quit"
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_build.return_value = _mock_orch()
    await run_interactive(args)
    mock_build.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_word", ["exit", "quit", "q", ":q", "/exit", "EXIT", "Quit"]
)
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_bare_word_exit(
    mock_console, mock_prompt, mock_build, exit_word
):
    """Bare 'exit'/'quit'/'q'/':q'/'/exit' (case-insensitive) all exit."""
    mock_prompt.ask.return_value = exit_word
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_orch = _mock_orch()
    mock_build.return_value = mock_orch
    await run_interactive(args)
    # Should exit without ever invoking the planner/run loop.
    mock_orch.run.assert_not_called()


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_exit_phrase_still_plans(
    mock_console, mock_prompt, mock_build
):
    """'exit the loop in main.py' is a real task and should reach the planner."""
    mock_prompt.ask.side_effect = ["exit the loop in main.py", "/quit"]
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_orch = _mock_orch()
    mock_orch.run = AsyncMock(return_value={"status": "completed"})
    mock_build.return_value = mock_orch
    await run_interactive(args)
    mock_orch.run.assert_called_once()


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_help(mock_console, mock_prompt, mock_build):
    mock_prompt.ask.side_effect = ["/help", "/quit"]
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_build.return_value = _mock_orch()
    await run_interactive(args)
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    # All slash commands should be printed
    for cmd in SLASH_COMMANDS:
        assert any(cmd in c for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli._show_config")
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_config(
    mock_console, mock_prompt, mock_build, mock_show_config
):
    mock_prompt.ask.side_effect = ["/config", "/quit"]
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_orch = _mock_orch()
    mock_build.return_value = mock_orch
    await run_interactive(args)
    mock_show_config.assert_called_once_with(mock_orch)


@pytest.mark.asyncio
@patch("agent.cli._show_history")
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_history(
    mock_console, mock_prompt, mock_build, mock_show_history
):
    mock_prompt.ask.side_effect = ["/history", "/quit"]
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_orch = _mock_orch()
    mock_build.return_value = mock_orch
    await run_interactive(args)
    mock_show_history.assert_called_once_with(mock_orch)


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_unknown_command(mock_console, mock_prompt, mock_build):
    mock_prompt.ask.side_effect = ["/foobar", "/quit"]
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_build.return_value = _mock_orch()
    await run_interactive(args)
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("Unknown command" in c and "/foobar" in c for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_empty_input(mock_console, mock_prompt, mock_build):
    mock_prompt.ask.side_effect = ["", "   ", "/quit"]
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_build.return_value = _mock_orch()
    await run_interactive(args)
    # Should not crash and should reach /quit


@pytest.mark.asyncio
@patch("agent.cli._approve_plan")
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_task_completed(
    mock_console, mock_prompt, mock_build, mock_approve
):
    mock_prompt.ask.side_effect = ["Fix the bug", "/quit"]
    mock_orch = _mock_orch()
    mock_orch.run = AsyncMock(return_value={"status": "completed"})
    mock_build.return_value = mock_orch
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    await run_interactive(args)
    mock_orch.run.assert_called_once_with(
        "Fix the bug", mode="interactive", approve_plan=mock_approve
    )
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("completed successfully" in c for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli._approve_plan")
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_task_failed(
    mock_console, mock_prompt, mock_build, mock_approve
):
    mock_prompt.ask.side_effect = ["Do something", "/quit"]
    mock_orch = _mock_orch()
    mock_orch.run = AsyncMock(
        return_value={"status": "failed", "reason": "syntax error"}
    )
    mock_build.return_value = mock_orch
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    await run_interactive(args)
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("syntax error" in c for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli._approve_plan")
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_task_aborted(
    mock_console, mock_prompt, mock_build, mock_approve
):
    mock_prompt.ask.side_effect = ["Do something", "/quit"]
    mock_orch = _mock_orch()
    mock_orch.run = AsyncMock(return_value={"status": "aborted"})
    mock_build.return_value = mock_orch
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    await run_interactive(args)
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("aborted" in c.lower() for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_eof(mock_console, mock_prompt, mock_build):
    mock_prompt.ask.side_effect = EOFError()
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_build.return_value = _mock_orch()
    await run_interactive(args)
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("Goodbye" in c for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.Prompt")
@patch("agent.cli.console")
async def test_run_interactive_keyboard_interrupt(
    mock_console, mock_prompt, mock_build
):
    mock_prompt.ask.side_effect = KeyboardInterrupt()
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    mock_build.return_value = _mock_orch()
    await run_interactive(args)
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("Goodbye" in c for c in print_calls)


# --- run_autonomous tests ---


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.console")
async def test_run_autonomous_no_task(mock_console, mock_build):
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        task=None,
        max_steps=20,
    )
    result = await run_autonomous(args)
    assert result == 1
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("--task is required" in c for c in print_calls)
    mock_build.assert_not_called()


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.console")
async def test_run_autonomous_completed(mock_console, mock_build):
    mock_orch = _mock_orch()
    mock_orch.run = AsyncMock(return_value={"status": "completed"})
    mock_build.return_value = mock_orch
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        task="Fix bug",
        max_steps=20,
    )
    result = await run_autonomous(args)
    assert result == 0
    mock_orch.run.assert_called_once_with("Fix bug", mode="autonomous")
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("completed successfully" in c for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.console")
async def test_run_autonomous_failed(mock_console, mock_build):
    mock_orch = _mock_orch()
    mock_orch.run = AsyncMock(return_value={"status": "failed", "reason": "timeout"})
    mock_build.return_value = mock_orch
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        task="Deploy app",
        max_steps=20,
    )
    result = await run_autonomous(args)
    assert result == 1
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("timeout" in c for c in print_calls)


@pytest.mark.asyncio
@patch("agent.cli.build_orchestrator")
@patch("agent.cli.console")
async def test_run_autonomous_failed_no_reason(mock_console, mock_build):
    mock_orch = _mock_orch()
    mock_orch.run = AsyncMock(return_value={"status": "failed"})
    mock_build.return_value = mock_orch
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        task="Deploy app",
        max_steps=20,
    )
    result = await run_autonomous(args)
    assert result == 1
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("unknown" in c for c in print_calls)


# --- build_orchestrator with stored verify commands ---


def test_build_orchestrator_with_stored_verify_commands(tmp_path):
    # First, build an orchestrator and store verify_commands in the db
    orch = build_orchestrator(
        project_root=tmp_path, base_url="http://localhost:11434/v1", model="qwen3:14b"
    )
    orch.db.set_config("verify_commands", json.dumps(["pytest", "ruff check ."]))
    orch.db.close()

    # Now build again -- stored commands should be loaded
    orch2 = build_orchestrator(
        project_root=tmp_path, base_url="http://localhost:11434/v1", model="qwen3:14b"
    )
    assert orch2.verifier.commands == ["pytest", "ruff check ."]
    orch2.db.close()


def test_build_orchestrator_verify_commands_override(tmp_path):
    # If stored commands exist, they override the passed-in verify_commands
    orch = build_orchestrator(
        project_root=tmp_path, base_url="http://localhost:11434/v1", model="qwen3:14b"
    )
    orch.db.set_config("verify_commands", json.dumps(["stored_cmd"]))
    orch.db.close()

    orch2 = build_orchestrator(
        project_root=tmp_path,
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        verify_commands=["passed_cmd"],
    )
    assert orch2.verifier.commands == ["stored_cmd"]
    orch2.db.close()


def test_build_orchestrator_verify_commands_passed(tmp_path):
    # When no stored commands, passed-in verify_commands are used
    orch = build_orchestrator(
        project_root=tmp_path,
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        verify_commands=["mypy ."],
    )
    assert orch.verifier.commands == ["mypy ."]
    orch.db.close()
