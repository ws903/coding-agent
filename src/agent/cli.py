# src/agent/cli.py
"""CLI entry points.

Most of the body has been extracted into focused modules:

  agent.console     -- shared Rich console singleton
  agent.cli_ui      -- rendering helpers (banner, panels, diff, table, tree, ...)
  agent.cli_input   -- prompt_toolkit session + ESC-during-task watcher
  agent.cli_intent  -- chat-vs-task routing

This module owns: argument parsing, orchestrator wiring, the REPL loop,
slash-command dispatch, and the autonomous-mode entry. Everything else is
re-exported below so existing `from agent.cli import _foo` imports keep
working.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.markdown import Markdown

from agent.agents_manager import AgentsManager
from agent.cli_input import (
    _build_repl_keybindings,
    _esc_aborts,
    _get_session,
    _get_user_input,
    _is_lone_escape_unix,
    _make_prompt_session,
    _watch_for_esc_unix,
    _watch_for_esc_win,
)
from agent.cli_intent import (
    _CHAT_TOKENS,
    _CLASSIFIER_PROMPT,
    _fast_chat,
    _FAST_CHAT_PROMPT,
    _llm_classify_intent,
    _looks_like_chat,
    _route_input,
)
from agent.cli_ui import (
    _ACTION_GLYPH,
    _MAX_DIFF_LINES,
    _TOOL_DISPLAY,
    _approve_plan,
    _format_tool_call,
    _print_welcome_banner,
    _render_edit_diff,
    _render_status,
    _show_change_summary,
    _show_config,
    _show_history,
    _show_status,
    _show_token_usage,
)
from agent import console as _con  # See cli_ui for the attribute-access pattern
from agent.console import console  # re-exported for backwards-compat
from agent.db import AgentDB
from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.mcp_manager import MCPManager, load_mcp_config
from agent.orchestrator import Orchestrator
from agent.planner import Planner
from agent.sandbox import Sandbox
from agent.skills_manager import SkillsManager
from agent.tools import FileTools
from agent.verifier import Verifier

# Re-export names kept for tests + backwards-compat imports.
__all__ = [
    # Re-exports (UI)
    "_ACTION_GLYPH",
    "_MAX_DIFF_LINES",
    "_TOOL_DISPLAY",
    "_approve_plan",
    "_format_tool_call",
    "_print_welcome_banner",
    "_render_edit_diff",
    "_render_status",
    "_show_change_summary",
    "_show_config",
    "_show_history",
    "_show_status",
    "_show_token_usage",
    # Re-exports (input)
    "_build_repl_keybindings",
    "_esc_aborts",
    "_get_session",
    "_get_user_input",
    "_is_lone_escape_unix",
    "_make_prompt_session",
    "_watch_for_esc_unix",
    "_watch_for_esc_win",
    # Re-exports (intent)
    "_CHAT_TOKENS",
    "_CLASSIFIER_PROMPT",
    "_FAST_CHAT_PROMPT",
    "_fast_chat",
    "_llm_classify_intent",
    "_looks_like_chat",
    "_route_input",
    # Owned by this module
    "SLASH_COMMANDS",
    "_EXIT_INPUTS",
    "_find_project_root",
    "build_orchestrator",
    "console",
    "parse_args",
    "run_autonomous",
    "run_interactive",
]


# Load .env from the coding-agent repo root (gitignored) so users can set
# AGENT_BASE_URL / AGENT_MODEL once without touching shell config.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DEFAULT_MODEL = "qwen3.6:35b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"


SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/status": "Show current task status",
    "/config": "Show/edit project configuration",
    "/history": "Show conversation history",
    "/abort": "Abort current execution",
    "/quit": "Exit the agent (alias: /exit, or type 'exit'/'quit')",
}

# Bare-word exits. Only exact matches -- "exit the loop in main.py" still
# goes to the planner, because it's a real task description. The slash
# variants (/quit, /exit) work too.
_EXIT_INPUTS = {"exit", "quit", "q", ":q"}


def _find_project_root(start: Path) -> Path:
    """Walk up from `start` to the nearest directory containing `.git`.

    Falls back to `start` if no `.git` is found. Lets the user `cd` anywhere
    inside their repo (including subdirectories) and invoke the agent without
    needing to pass --project explicitly.
    """
    start = start.resolve()
    candidate = start if start.is_dir() else start.parent
    for path in [candidate, *candidate.parents]:
        if (path / ".git").exists():
            return path
    return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Local coding agent with planner/executor architecture",
    )
    parser.add_argument("--auto", action="store_true", help="Run in autonomous mode")
    parser.add_argument(
        "--task", type=str, help="Task description (required for --auto)"
    )
    parser.add_argument("--max-steps", type=int, default=20, help="Max execution steps")
    parser.add_argument(
        "--step", action="store_true", help="Approve each step individually"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
        help="Model name (env: AGENT_MODEL)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("AGENT_BASE_URL", DEFAULT_BASE_URL),
        help="LLM API base URL (env: AGENT_BASE_URL)",
    )
    parser.add_argument(
        "--project", type=str, default=".", help="Project root directory"
    )
    return parser.parse_args(argv)


def build_orchestrator(
    project_root: Path,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    verify_commands: list[str] | None = None,
    max_steps: int = 20,
    stream: bool = False,
) -> Orchestrator:
    project_root = project_root.resolve()
    db = AgentDB(project_root / ".agent" / "agent.db")

    stored_base_url = db.get_config("planner_base_url", base_url)
    stored_model = db.get_config("planner_model", model)

    planner_client = LLMClient(base_url=stored_base_url, model=stored_model)
    executor_client = LLMClient(
        base_url=db.get_config("executor_base_url", base_url),
        model=db.get_config("executor_model", model),
    )

    sandbox = Sandbox(project_root)
    tools = FileTools(project_root)
    planner = Planner(planner_client)

    on_token = None
    on_reasoning = None
    on_tool_call = None
    if stream:

        def on_token(chunk: str) -> None:
            _con.console.print(chunk, end="", soft_wrap=True)

        def on_reasoning(chunk: str) -> None:
            # Dim italic so the user sees the model is thinking, but the
            # reasoning visually separates from the final answer.
            _con.console.print(chunk, end="", soft_wrap=True, style="dim italic")

        def on_tool_call(name: str, args: dict) -> None:
            _con.console.print(_format_tool_call(name, args), soft_wrap=True)

    mcp = MCPManager(load_mcp_config(project_root))
    skills = SkillsManager(project_root)
    agents = AgentsManager(project_root)
    executor = Executor(
        executor_client,
        tools,
        on_token=on_token,
        on_reasoning=on_reasoning,
        on_tool_call=on_tool_call,
        mcp=mcp,
        skills=skills,
        agents=agents,
    )

    commands = verify_commands or []
    stored_commands = db.get_config("verify_commands")
    if stored_commands:
        commands = json.loads(stored_commands)

    verifier = Verifier(sandbox, commands=commands)

    return Orchestrator(
        planner=planner,
        executor=executor,
        verifier=verifier,
        tools=tools,
        db=db,
        project_root=project_root,
        on_status=_render_status,
        on_edit_applied=_render_edit_diff if stream else None,
        max_steps=max_steps,
    )


async def run_interactive(args: argparse.Namespace) -> None:
    project_root = _find_project_root(Path(args.project))
    orch = build_orchestrator(
        project_root,
        args.base_url,
        args.model,
        max_steps=args.max_steps,
        stream=True,
    )
    await orch.executor.mcp.connect()
    _print_welcome_banner(orch, args, project_root)

    try:
        await _interactive_loop(orch)
    finally:
        await orch.executor.mcp.close()


def _handle_slash_command(orch: Orchestrator, user_input: str) -> bool:
    """Dispatch a slash command. Returns True if the input was a (recognized
    or unknown) slash command -- caller should NOT pass it to the planner.
    Returns False if the input isn't a slash command at all."""
    if user_input == "/help":
        for cmd, desc in SLASH_COMMANDS.items():
            _con.console.print(f"  [bold]{cmd}[/bold] — {desc}")
        return True
    if user_input == "/status":
        _show_status(orch)
        return True
    if user_input == "/abort":
        orch.abort()
        _con.console.print("[yellow]Abort requested.[/yellow]")
        return True
    if user_input == "/config":
        _show_config(orch)
        return True
    if user_input == "/history":
        _show_history(orch)
        return True
    if user_input.startswith("/"):
        _con.console.print(f"[red]Unknown command: {user_input}[/red]")
        return True
    return False


async def _interactive_loop(orch: Orchestrator) -> None:
    while True:
        try:
            user_input = await _get_user_input()
        except (EOFError, KeyboardInterrupt):
            _con.console.print("\nGoodbye.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in _EXIT_INPUTS or user_input in {"/quit", "/exit"}:
            break

        if _handle_slash_command(orch, user_input):
            continue

        if await _route_input(orch, user_input) == "chat":
            await _fast_chat(orch, user_input)
            continue

        # Spinner runs in a background thread; on_status prints flow above it
        # via Rich's Live infrastructure. Auto-disabled on non-TTY (e.g. when
        # output is piped to a file), so this is safe for all environments.
        # _esc_aborts wires the ESC key to orch.abort() during the run.
        try:
            with (
                _con.console.status(
                    "[dim]Working...[/dim]", spinner="dots", spinner_style="cyan"
                ),
                _esc_aborts(orch),
            ):
                result = await orch.run(
                    user_input, mode="interactive", approve_plan=_approve_plan
                )
        except KeyboardInterrupt:
            # Ctrl+C during a running task: abort gracefully, stay in the REPL.
            # Same outcome as ESC; this catches the path where orch.run raises
            # before orch.abort() is checked.
            orch.abort()
            _con.console.print("\n[yellow]Interrupted.[/yellow]")
            continue
        status = result["status"]
        if status == "answered":
            # Render as markdown so headers, lists, code fences look right.
            # Planner answers often include markdown for codebase explanations.
            _con.console.print(Markdown(result["answer"]))
            _con.console.print()
        elif status == "completed":
            _show_change_summary(orch, result.get("conv_id", ""))
            _show_token_usage(orch)
        elif status == "failed":
            _con.console.print(
                f"[red]Task failed: {result.get('reason', 'unknown')}[/red]"
            )
            _show_change_summary(orch, result.get("conv_id", ""))
            _show_token_usage(orch)
        elif status == "aborted":
            _con.console.print("[yellow]Task aborted.[/yellow]\n")


async def run_autonomous(args: argparse.Namespace) -> int:
    if not args.task:
        _con.console.print("[red]--task is required for autonomous mode[/red]")
        return 1

    project_root = _find_project_root(Path(args.project))
    orch = build_orchestrator(
        project_root, args.base_url, args.model, max_steps=args.max_steps
    )

    _con.console.print(f"[bold]Autonomous mode[/bold] | Task: {args.task}")
    await orch.executor.mcp.connect()
    if orch.executor.mcp.connected_servers:
        _con.console.print(
            f"[dim]MCP servers: {', '.join(orch.executor.mcp.connected_servers)}[/dim]"
        )

    try:
        result = await orch.run(args.task, mode="autonomous")
    finally:
        await orch.executor.mcp.close()

    if result["status"] == "answered":
        _con.console.print(result["answer"])
        return 0
    if result["status"] == "completed":
        _show_change_summary(orch, result.get("conv_id", ""))
        _show_token_usage(orch)
        return 0
    _con.console.print(f"[red]Task failed: {result.get('reason', 'unknown')}[/red]")
    _show_change_summary(orch, result.get("conv_id", ""))
    _show_token_usage(orch)
    return 1
