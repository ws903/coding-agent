# tests/test_cli.py

import json
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.cli import (
    SLASH_COMMANDS,
    _approve_plan,
    _build_repl_keybindings,
    _esc_aborts,
    _find_project_root,
    _is_lone_escape_unix,
    _format_tool_call,
    _llm_classify_intent,
    _looks_like_chat,
    _print_welcome_banner,
    _render_edit_diff,
    _render_status,
    _route_input,
    _show_change_summary,
    _show_config,
    _show_history,
    _show_status,
    _show_token_usage,
    build_orchestrator,
    parse_args,
    run_autonomous,
    run_interactive,
)
from agent.core.models import Plan, Step


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
    """Flatten everything printed (including Panel/Markdown/Tree contents)
    for substring search in tests.

    Uses isinstance checks against Rich's public classes rather than
    duck-typing on `.label`/`.children` -- duck-typing is fragile to
    future Rich refactors where new renderables might happen to expose
    the same attribute names.
    """
    from rich.panel import Panel
    from rich.tree import Tree

    def _walk(obj) -> list[str]:
        if isinstance(obj, Tree):
            out = [str(obj.label)]
            for child in obj.children:
                out.extend(_walk(child))
            return out
        if isinstance(obj, Panel):
            return [str(obj.renderable)]
        # Markdown and other renderables also expose `.renderable` /
        # `.markup`; surface either if present, otherwise stringify.
        renderable = getattr(obj, "renderable", None)
        if renderable is not None:
            return [str(renderable)]
        markup = getattr(obj, "markup", None)
        if markup is not None:
            return [str(markup)]
        return [str(obj)]

    parts: list[str] = []
    for call in mock_console.print.call_args_list:
        if not call.args:
            continue
        parts.extend(_walk(call.args[0]))
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


# --- diff rendering ---


@patch("agent.cli.console.console")
def test_render_edit_diff_prints_syntax_block(mock_console):
    """A non-empty edit produces a Syntax-rendered diff."""
    _render_edit_diff(
        "foo.py",
        "search_replace",
        "def hello():\n    pass\n",
        'def hello():\n    """Say hi."""\n    pass\n',
    )
    # console.print should fire exactly once with a Syntax renderable.
    assert mock_console.print.call_count == 1
    arg = mock_console.print.call_args.args[0]
    # Rich Syntax has a `.code` attribute holding the source string.
    code = getattr(arg, "code", "")
    assert "+    " in code  # the new docstring line
    assert "foo.py" in code  # filename header


@patch("agent.cli.console.console")
def test_render_edit_diff_skips_no_change(mock_console):
    """If before == after, nothing is printed."""
    text = "def hello():\n    pass\n"
    _render_edit_diff("foo.py", "search_replace", text, text)
    mock_console.print.assert_not_called()


@patch("agent.cli.console.console")
def test_render_edit_diff_truncates_huge_diffs(mock_console):
    """Diffs longer than _MAX_DIFF_LINES are truncated with a marker."""
    before = "\n".join(f"line {i}" for i in range(100))
    after = "\n".join(f"changed {i}" for i in range(100))
    _render_edit_diff("big.py", "rewrite", before, after)
    arg = mock_console.print.call_args.args[0]
    code = getattr(arg, "code", "")
    assert "truncated" in code


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


@patch("agent.cli.ui.Prompt")
@patch("agent.cli.console.console")
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


@patch("agent.cli.ui.Prompt")
@patch("agent.cli.console.console")
def test_approve_plan_rejected(mock_console, mock_prompt):
    mock_prompt.ask.return_value = "n"
    plan = Plan(
        goal="Delete everything",
        steps=[Step(id=1, action="Delete all", files_needed=[])],
    )
    result = _approve_plan(plan)
    assert result is False


@patch("agent.cli.ui.Prompt")
@patch("agent.cli.console.console")
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


@patch("agent.cli.ui.Prompt")
@patch("agent.cli.console.console")
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


@patch("agent.cli.console.console")
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


@patch("agent.cli.console.console")
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


@patch("agent.cli.console.console")
def test_show_history_empty(mock_console):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_cursor
    mock_orch = _mock_orch()
    mock_orch.db = mock_db
    _show_history(mock_orch)
    mock_console.print.assert_called_once_with("No conversation history.")


@patch("agent.cli.console.console")
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_quit(mock_console, mock_input, mock_build):
    mock_input.return_value = "/quit"
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_bare_word_exit(
    mock_console, mock_input, mock_build, exit_word
):
    """Bare 'exit'/'quit'/'q'/':q'/'/exit' (case-insensitive) all exit."""
    mock_input.return_value = exit_word
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_exit_phrase_still_plans(
    mock_console, mock_input, mock_build
):
    """'exit the loop in main.py' is a real task and should reach the planner."""
    mock_input.side_effect = ["exit the loop in main.py", "/quit"]
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_help(mock_console, mock_input, mock_build):
    mock_input.side_effect = ["/help", "/quit"]
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
@patch("agent.cli.main._show_config")
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_config(
    mock_console, mock_input, mock_build, mock_show_config
):
    mock_input.side_effect = ["/config", "/quit"]
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
@patch("agent.cli.main._show_history")
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_history(
    mock_console, mock_input, mock_build, mock_show_history
):
    mock_input.side_effect = ["/history", "/quit"]
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_unknown_command(mock_console, mock_input, mock_build):
    mock_input.side_effect = ["/foobar", "/quit"]
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_empty_input(mock_console, mock_input, mock_build):
    mock_input.side_effect = ["", "   ", "/quit"]
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
@patch("agent.cli.main._approve_plan")
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_task_completed(
    mock_console, mock_input, mock_build, mock_approve
):
    mock_input.side_effect = ["Fix the bug", "/quit"]
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
    # Completion indicator now lives inside a Rich Tree root label.
    assert "Task complete" in _all_printed(mock_console)


@pytest.mark.asyncio
@patch("agent.cli.main._approve_plan")
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_task_failed(
    mock_console, mock_input, mock_build, mock_approve
):
    mock_input.side_effect = ["Do something", "/quit"]
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
@patch("agent.cli.main._approve_plan")
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_task_aborted(
    mock_console, mock_input, mock_build, mock_approve
):
    mock_input.side_effect = ["Do something", "/quit"]
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_eof(mock_console, mock_input, mock_build):
    mock_input.side_effect = EOFError()
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_keyboard_interrupt(mock_console, mock_input, mock_build):
    mock_input.side_effect = KeyboardInterrupt()
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.console.console")
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.console.console")
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
    # Completion indicator now lives inside a Rich Tree root label.
    assert "Task complete" in _all_printed(mock_console)


@pytest.mark.asyncio
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.console.console")
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
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.console.console")
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


# --- Visual rendering helpers (coverage targeted) ---


@patch("agent.cli.console.console")
def test_render_status_executing_step_uses_rule(mock_console):
    _render_status("Executing step 1 [1/3]: do the thing")
    mock_console.rule.assert_called_once()


@patch("agent.cli.console.console")
def test_render_status_plan_message_uses_print(mock_console):
    _render_status("Plan: Add docstrings (3 steps)")
    mock_console.print.assert_called_once()


@patch("agent.cli.console.console")
def test_render_status_completion_uses_green_rule(mock_console):
    _render_status("All steps completed successfully.")
    args, kwargs = mock_console.rule.call_args
    assert "green" in str(kwargs.get("style", ""))


@patch("agent.cli.console.console")
def test_render_status_warnings_use_yellow_rule(mock_console):
    for msg in [
        "Max replans reached. Stopping.",
        "Rolling back failed step...",
        "Replanning (attempt 1/3)...",
        "Aborted by user.",
    ]:
        mock_console.reset_mock()
        _render_status(msg)
        args, kwargs = mock_console.rule.call_args
        assert "yellow" in str(kwargs.get("style", ""))


@patch("agent.cli.console.console")
def test_render_status_default_is_dim_print(mock_console):
    _render_status("something else entirely")
    mock_console.print.assert_called_once()
    assert mock_console.rule.call_count == 0


# --- _print_welcome_banner ---


@patch("agent.cli.console.console")
def test_print_welcome_banner_basic(mock_console, tmp_path):
    orch = _mock_orch()
    args = Namespace(model="qwen3.6:35b")
    _print_welcome_banner(orch, args, tmp_path)
    printed = _all_printed(mock_console)
    assert "qwen3.6:35b" in printed
    # tmp_path stringifies with the platform's native separators (/ on Linux,
    # \ on Windows) -- assert via its own str() rather than a hard-coded path.
    assert str(tmp_path) in printed


@patch("agent.cli.console.console")
def test_print_welcome_banner_with_mcp_and_skills(mock_console, tmp_path):
    orch = _mock_orch()
    orch.executor.skills.skills = ["skill-a", "skill-b"]
    orch.executor.agents.roles = ["reviewer"]
    orch.executor.mcp.connected_servers = ["filesystem"]
    args = Namespace(model="qwen3.6:35b")
    _print_welcome_banner(orch, args, tmp_path)
    printed = _all_printed(mock_console)
    assert "filesystem" in printed
    assert "2 skills" in printed
    assert "1 subagent" in printed


# --- _show_change_summary ---


@patch("agent.cli.console.console")
def test_show_change_summary_no_conv_id_prints_done(mock_console):
    orch = _mock_orch()
    _show_change_summary(orch, "")
    mock_console.print.assert_called_once()
    assert "completed successfully" in _all_printed(mock_console)


@patch("agent.cli.console.console")
def test_show_change_summary_empty_edits_prints_done(mock_console):
    orch = _mock_orch()
    orch.db.get_edits = MagicMock(return_value=[])
    _show_change_summary(orch, "c1")
    assert "completed successfully" in _all_printed(mock_console)


@patch("agent.cli.console.console")
def test_show_change_summary_renders_tree_with_deltas(mock_console):
    orch = _mock_orch()
    orch.db.get_edits = MagicMock(
        return_value=[
            {
                "file_path": "foo.py",
                "edit_type": "search_replace",
                "before": "def f():\n    pass\n",
                "after": 'def f():\n    """hi"""\n    pass\n',
            }
        ]
    )
    _show_change_summary(orch, "c1")
    printed = _all_printed(mock_console)
    assert "Task complete" in printed
    assert "foo.py" in printed


@patch("agent.cli.console.console")
def test_show_change_summary_created_file(mock_console):
    orch = _mock_orch()
    orch.db.get_edits = MagicMock(
        return_value=[
            {
                "file_path": "new.py",
                "edit_type": "create",
                "before": None,
                "after": "x = 1\n",
            }
        ]
    )
    _show_change_summary(orch, "c1")
    assert "new.py" in _all_printed(mock_console)


@patch("agent.cli.console.console")
def test_show_change_summary_db_failure_falls_back(mock_console):
    orch = _mock_orch()
    orch.db.get_edits = MagicMock(side_effect=AttributeError("boom"))
    _show_change_summary(orch, "c1")
    assert "completed successfully" in _all_printed(mock_console)


# --- _show_status ---


@patch("agent.cli.console.console")
def test_show_status_with_no_task(mock_console):
    orch = _mock_orch()
    orch.status = MagicMock(
        return_value={
            "task": "",
            "current_step": "",
            "steps_executed": 0,
            "total_steps": 0,
            "aborted": False,
        }
    )
    _show_status(orch)
    assert "No task running" in _all_printed(mock_console)


@patch("agent.cli.console.console")
def test_show_status_with_active_task(mock_console):
    orch = _mock_orch()
    orch.status = MagicMock(
        return_value={
            "task": "Refactor X",
            "current_step": "Step 2: Y",
            "steps_executed": 1,
            "total_steps": 3,
            "aborted": False,
        }
    )
    _show_status(orch)
    printed = _all_printed(mock_console)
    assert "Refactor X" in printed
    assert "Step 2: Y" in printed
    assert "1/3" in printed


# --- _show_token_usage table rendering ---


@patch("agent.cli.console.console")
def test_show_token_usage_renders_table_with_real_numbers(mock_console):
    orch = _mock_orch()
    orch.token_usage = MagicMock(
        return_value={
            "planner": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "calls": 1,
            },
            "executor": {
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "total_tokens": 280,
                "calls": 5,
            },
        }
    )
    _show_token_usage(orch)
    # First call is the Table renderable, second is a blank line.
    assert mock_console.print.call_count >= 1


@patch("agent.cli.console.console")
def test_show_token_usage_skips_when_zero(mock_console):
    orch = _mock_orch()
    orch.token_usage = MagicMock(
        return_value={
            "planner": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            },
            "executor": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            },
        }
    )
    _show_token_usage(orch)
    mock_console.print.assert_not_called()


# --- prompt_toolkit key bindings ---


def test_build_repl_keybindings_includes_escape_and_ctrl_c():
    """ESC and Ctrl+C must be bound; otherwise the user's UX regressed."""
    from prompt_toolkit.keys import Keys

    kb = _build_repl_keybindings()
    keys = []
    for binding in kb.bindings:
        for key in binding.keys:
            keys.append(key)

    assert Keys.Escape in keys
    assert Keys.ControlC in keys


# The PromptSession factory itself is one constructor call -- not worth a
# unit test, and instantiating it on Windows CI hits NoConsoleScreenBufferError
# because the runner has no attached console. _make_prompt_session is exercised
# at runtime via _get_session()'s lazy singleton.


# --- ESC abort behavior ---


def test_esc_aborts_is_noop_when_stdin_not_tty():
    """No TTY (CI, piped) -> context manager passes through with no watcher."""
    orch = _mock_orch()
    with patch("agent.cli.input.sys.stdin.isatty", return_value=False):
        with _esc_aborts(orch):
            pass  # No abort, no thread started.
    orch.abort.assert_not_called()


def test_esc_aborts_starts_and_stops_watcher_thread_on_tty():
    """When stdin is a TTY, the context manager spawns a daemon watcher."""
    orch = _mock_orch()
    with (
        patch("agent.cli.input.sys.stdin.isatty", return_value=True),
        patch("agent.cli.input.threading.Thread") as mock_thread_class,
    ):
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread
        with _esc_aborts(orch):
            mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once()


@pytest.mark.asyncio
@patch("agent.cli.main._approve_plan")
@patch("agent.cli.main.build_orchestrator")
@patch("agent.cli.main._get_user_input", new_callable=AsyncMock)
@patch("agent.cli.console.console")
async def test_run_interactive_ctrl_c_during_task_aborts_and_continues(
    mock_console, mock_input, mock_build, mock_approve
):
    """Ctrl+C during orch.run aborts the task and returns to the REPL prompt
    instead of exiting (the abort path is graceful, not terminal)."""
    mock_input.side_effect = ["fix it", "/quit"]
    mock_orch = _mock_orch()
    # First call raises KeyboardInterrupt (simulating Ctrl+C mid-task).
    mock_orch.run = AsyncMock(side_effect=KeyboardInterrupt())
    mock_build.return_value = mock_orch
    args = Namespace(
        project="/tmp/test",
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
        max_steps=20,
    )
    await run_interactive(args)
    # abort() called once via the KeyboardInterrupt handler.
    mock_orch.abort.assert_called_once()
    # Loop continued and reached /quit (i.e. didn't exit on the interrupt).
    assert "Interrupted" in _all_printed(mock_console)


# --- ESC vs escape sequence disambiguation ---


def test_is_lone_escape_returns_true_when_no_follow_up():
    """Bare ESC press: select returns no data within the timeout window."""
    with patch("agent.cli.input.select.select", return_value=([], [], [])):
        stream = MagicMock()
        assert _is_lone_escape_unix(stream, timeout=0.001) is True


def test_is_lone_escape_returns_false_for_arrow_key():
    """Arrow key: \\x1b[A. The leading \\x1b was already consumed; this
    helper sees [ and A follow. Should return False and drain both bytes."""
    stream = MagicMock()
    stream.read.side_effect = ["[", "A"]
    with patch(
        "agent.cli.input.select.select",
        side_effect=[
            ([stream], [], []),  # First peek: follow byte ready
            ([stream], [], []),  # Drain iter 1: '[' ready
            ([stream], [], []),  # Drain iter 2: 'A' ready
            ([], [], []),  # Drain done
        ],
    ):
        assert _is_lone_escape_unix(stream, timeout=0.001) is False
    assert stream.read.call_count == 2


def test_is_lone_escape_returns_false_for_alt_combo():
    """Alt+f sends \\x1b f. Single follow-up byte, should drain it."""
    stream = MagicMock()
    stream.read.side_effect = ["f"]
    with patch(
        "agent.cli.input.select.select",
        side_effect=[
            ([stream], [], []),  # Follow byte ready
            ([stream], [], []),  # Drain iter 1: 'f' ready
            ([], [], []),  # Drain done
        ],
    ):
        assert _is_lone_escape_unix(stream, timeout=0.001) is False
    assert stream.read.call_count == 1


# --- watcher bridge: fake watcher should call orch.abort() ---


def test_esc_aborts_bridge_fires_abort_via_watcher(monkeypatch):
    """Inject a fake watcher target into _esc_aborts and verify that when
    the watcher calls orch.abort(), the orchestrator is actually aborted.

    This closes the gap between platform-specific watcher implementations
    (pragma:no cover) and the orchestrator abort path: the watcher invokes
    abort() and the orchestrator side sees the flag.
    """
    orch = _mock_orch()
    orch.abort = MagicMock()

    def _fake_watch(orch, stop_event):
        # Simulate "user pressed ESC" -- immediately call abort.
        orch.abort()

    # _esc_aborts lives in cli_input and looks up these names locally there,
    # so the re-exports in agent.cli aren't what it sees.
    monkeypatch.setattr("agent.cli.input._watch_for_esc_unix", _fake_watch)
    monkeypatch.setattr("agent.cli.input._watch_for_esc_win", _fake_watch)

    with patch("agent.cli.input.sys.stdin.isatty", return_value=True):
        with _esc_aborts(orch):
            # Yield long enough for the daemon thread to run _fake_watch.
            import time as _t

            _t.sleep(0.05)

    orch.abort.assert_called_once()
