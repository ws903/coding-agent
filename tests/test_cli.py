# tests/test_cli.py

import json
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.cli import (
    SLASH_COMMANDS,
    _approve_plan,
    _show_config,
    _show_history,
    build_orchestrator,
    parse_args,
    run_autonomous,
    run_interactive,
)
from agent.models import Plan, Step


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
    # Verify plan goal was printed
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("Refactor module" in c for c in print_calls)


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
    print_calls = [str(c) for c in mock_console.print.call_args_list]
    assert any("config.yaml" in c for c in print_calls)
    assert any("settings.json" in c for c in print_calls)


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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_build.return_value = MagicMock()
    await run_interactive(args)
    mock_build.assert_called_once()


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
    mock_build.return_value = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_build.return_value = MagicMock()
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
    mock_build.return_value = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_build.return_value = MagicMock()
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
    mock_build.return_value = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
    mock_orch = MagicMock()
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
