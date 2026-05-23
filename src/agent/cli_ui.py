# src/agent/cli_ui.py
"""Rendering / display helpers for the CLI.

Pure output layer: every function in here writes to the shared Rich console
and never reads from stdin or makes LLM calls. Extracted from cli.py to keep
the main entry point focused on argparse + the REPL loop + orchestrator wiring.
"""

from __future__ import annotations

import argparse
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markdown import Markdown  # noqa: F401  # re-exported via agent.cli
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from agent import console as _con  # Use _con.console so tests can mock via
# `@patch("agent.console.console")` and the patch is honored at call time.

if TYPE_CHECKING:
    from agent.models import Plan
    from agent.orchestrator import Orchestrator

# Cap diff output so a 2000-line rewrite doesn't flood the terminal.
_MAX_DIFF_LINES = 40

# Per-tool display glyphs + colors for the on_tool_call callback. Emojis
# degrade to text on non-unicode terminals via Rich's automatic fallback.
_TOOL_DISPLAY: dict[str, tuple[str, str]] = {
    "read_file": ("📖", "cyan"),
    "list_files": ("📁", "cyan"),
    "search_text": ("🔍", "cyan"),
    "create_file": ("🆕", "green"),
    "edit_file": ("✏️ ", "yellow"),
    "replace_file": ("📝", "yellow"),
    "run_command": ("🏃", "magenta"),
    "read_skill": ("📜", "blue"),
    "spawn_agent": ("🤖", "blue"),
}

_ACTION_GLYPH = {
    "create": ("🆕", "green", "created"),
    "rewrite": ("📝", "yellow", "rewrote"),
    "search_replace": ("✏️ ", "yellow", "edited"),
}


def _render_edit_diff(
    path: str, action: str, before: str | None, after: str | None
) -> None:
    """Render an inline syntax-highlighted unified diff for a single edit."""
    before_lines = (before or "").splitlines(keepends=True)
    after_lines = (after or "").splitlines(keepends=True)
    diff_lines = list(
        unified_diff(
            before_lines, after_lines, fromfile=f"a/{path}", tofile=f"b/{path}", n=2
        )
    )
    if not diff_lines:
        return

    truncated = False
    if len(diff_lines) > _MAX_DIFF_LINES:
        diff_lines = diff_lines[:_MAX_DIFF_LINES]
        truncated = True
    diff_text = "".join(diff_lines)
    if truncated:
        diff_text += f"\n... [truncated at {_MAX_DIFF_LINES} lines]\n"

    _con.console.print(
        Syntax(
            diff_text,
            "diff",
            theme="ansi_dark",
            line_numbers=False,
            word_wrap=False,
            background_color="default",
        )
    )


def _render_status(msg: str) -> None:
    """Style orchestrator status messages by phase.

    Step boundaries get a `_con.console.rule()` for visual separation; everything
    else gets dim text. Keeps the orchestrator's on_status(str) interface
    unchanged -- the CLI just renders the strings smarter.
    """
    if msg.startswith("Executing step"):
        _con.console.rule(f"[bold cyan]{msg}[/bold cyan]", style="cyan", align="left")
    elif msg.startswith("Plan: "):
        # Already going to be rendered as a Panel by _approve_plan in
        # interactive mode; skip the dim duplicate to keep output clean.
        # In autonomous mode this is the only place the plan goal shows up.
        _con.console.print(f"[bold]{msg}[/bold]")
    elif msg.startswith("All steps completed"):
        _con.console.rule("[bold green]✓ " + msg + "[/bold green]", style="green")
    elif msg.startswith(("Max ", "Rolling back", "Replanning", "Aborted")):
        _con.console.rule(f"[yellow]{msg}[/yellow]", style="yellow", align="left")
    else:
        _con.console.print(f"[dim]{msg}[/dim]")


def _approve_plan(plan: Plan) -> bool:
    lines = [f"[bold green]{plan.goal}[/bold green]", ""]
    for step in plan.steps:
        lines.append(f"  [cyan]{step.id}.[/cyan] {step.action}")
        if step.files_needed:
            lines.append(f"     [dim]files: {', '.join(step.files_needed)}[/dim]")
        if step.verify_command:
            lines.append(f"     [dim]verify: {step.verify_command}[/dim]")
    _con.console.print()
    _con.console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold]Proposed Plan[/bold] [dim]({len(plan.steps)} step{'s' if len(plan.steps) != 1 else ''})[/dim]",
            border_style="green",
            padding=(0, 1),
        )
    )
    answer = Prompt.ask("Proceed?", choices=["y", "n"], default="y")
    return answer == "y"


def _format_tool_call(name: str, args: dict) -> str:
    """Render a tool call as `<emoji> <name>(<key arg>)` for inline display."""
    icon, color = _TOOL_DISPLAY.get(name, ("⚙️ ", "white"))
    if name.startswith("mcp__"):
        icon, color = "🔌", "blue"

    # Pull the most informative single argument as a hint.
    summary = ""
    if "path" in args:
        summary = args["path"]
    elif "command" in args:
        summary = args["command"]
    elif "query" in args:
        summary = repr(args["query"])
    elif "name" in args:
        summary = args["name"]
    elif "role" in args:
        summary = args["role"]
    elif "directory" in args:
        summary = args["directory"]
    if len(summary) > 60:
        summary = summary[:57] + "..."

    label = f"[{color}]{name}[/{color}]"
    if summary:
        return f"{icon} {label} [dim]{summary}[/dim]"
    return f"{icon} {label}"


def _print_welcome_banner(
    orch: Orchestrator, args: argparse.Namespace, project_root: Path
) -> None:
    """Top-of-REPL banner: model, project, loaded extensions."""
    skills_count = len(orch.executor.skills.skills) if orch.executor.skills else 0
    agents_count = len(orch.executor.agents.roles) if orch.executor.agents else 0
    mcp_servers = orch.executor.mcp.connected_servers if orch.executor.mcp else []

    lines = [
        f"[bold]Model:[/bold]   {args.model}",
        f"[bold]Project:[/bold] {project_root}",
    ]
    if mcp_servers:
        lines.append(f"[bold]MCP:[/bold]     {', '.join(mcp_servers)}")
    if skills_count or agents_count:
        ext = []
        if skills_count:
            ext.append(f"{skills_count} skill{'s' if skills_count != 1 else ''}")
        if agents_count:
            ext.append(f"{agents_count} subagent{'s' if agents_count != 1 else ''}")
        lines.append(f"[bold]Loaded:[/bold]  {', '.join(ext)}")

    _con.console.print(
        Panel(
            "\n".join(lines),
            title="[bold cyan]Coding Agent[/bold cyan] [dim]v0.1.0[/dim]",
            subtitle="[dim]/help for commands · /quit or 'exit' to leave[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    _con.console.print()


def _show_change_summary(orch: Orchestrator, conv_id: str) -> None:
    """Print a Tree of files changed during the task with +/- line counts.

    Falls back to a one-line completion message when there's nothing to
    enumerate (no conv_id, no edits recorded, or the DB call fails).
    """
    edits = []
    if conv_id:
        try:
            edits = orch.db.get_edits(conv_id) or []
        except (AttributeError, TypeError):
            edits = []

    if not edits:
        _con.console.print("[green]✓ Task completed successfully.[/green]")
        return

    # Collapse multiple edits to the same file into a single entry that uses
    # the earliest 'before' and the latest 'after' for a clean total delta.
    per_file: dict[str, dict] = {}
    for row in edits:
        path = row.get("file_path", "")
        if not path:
            continue
        entry = per_file.setdefault(
            path,
            {
                "action": row.get("edit_type", ""),
                "before": row.get("before"),
                "after": row.get("after"),
            },
        )
        entry["after"] = row.get("after")  # latest after wins

    tree = Tree(
        f"[bold green]✓[/bold green] [bold]Task complete[/bold] "
        f"[dim]({len(per_file)} file{'s' if len(per_file) != 1 else ''} changed)[/dim]",
        guide_style="dim",
    )
    for path, info in per_file.items():
        before_lines = info["before"].count("\n") + 1 if info["before"] else 0
        after_lines = info["after"].count("\n") + 1 if info["after"] else 0
        added = max(0, after_lines - before_lines)
        removed = max(0, before_lines - after_lines)

        icon, color, verb = _ACTION_GLYPH.get(
            info["action"], ("•", "white", info["action"])
        )
        delta_parts = []
        if added:
            delta_parts.append(f"[green]+{added}[/green]")
        if removed:
            delta_parts.append(f"[red]-{removed}[/red]")
        delta = " ".join(delta_parts) or "[dim]·[/dim]"

        tree.add(f"{icon} [{color}]{verb}[/{color}] {path} [dim]{delta}[/dim]")

    _con.console.print(tree)
    _con.console.print()


def _show_token_usage(orch: Orchestrator) -> None:
    try:
        usage = orch.token_usage()
        p = usage["planner"]
        e = usage["executor"]
        # Validate shape early so MagicMock objects in tests fall to the
        # except branch instead of crashing in the format string below.
        total = int(p["total_tokens"]) + int(e["total_tokens"])
        if total == 0:
            return

        table = Table(
            title="[dim]Token usage[/dim]",
            title_justify="left",
            show_header=True,
            header_style="dim",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("", style="dim")
        table.add_column("Prompt", justify="right")
        table.add_column("Completion", justify="right")
        table.add_column("Total", justify="right", style="bold")
        table.add_column("Calls", justify="right", style="dim")
        table.add_row(
            "Planner",
            f"{int(p['prompt_tokens']):,}",
            f"{int(p['completion_tokens']):,}",
            f"{int(p['total_tokens']):,}",
            str(int(p["calls"])),
        )
        table.add_row(
            "Executor",
            f"{int(e['prompt_tokens']):,}",
            f"{int(e['completion_tokens']):,}",
            f"{int(e['total_tokens']):,}",
            str(int(e["calls"])),
        )
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{int(p['prompt_tokens']) + int(e['prompt_tokens']):,}[/bold]",
            f"[bold]{int(p['completion_tokens']) + int(e['completion_tokens']):,}[/bold]",
            f"[bold]{total:,}[/bold]",
            f"[bold]{int(p['calls']) + int(e['calls'])}[/bold]",
        )
        _con.console.print(table)
        _con.console.print()
    except (TypeError, ValueError, KeyError, AttributeError):
        # Graceful no-op for tests that mock orch.token_usage(), or for
        # backends that don't report usage in their response.
        return


def _show_status(orch: Orchestrator) -> None:
    s = orch.status()
    if not s["task"]:
        _con.console.print("No task running.")
        return
    _con.console.print(f"[bold]Task:[/bold] {s['task']}")
    _con.console.print(f"[bold]Step:[/bold] {s['current_step']}")
    _con.console.print(
        f"[bold]Progress:[/bold] {s['steps_executed']}/{s['total_steps']} steps executed"
    )


def _show_config(orch: Orchestrator) -> None:
    _con.console.print("[bold]Configuration:[/bold]")
    for key in [
        "planner_base_url",
        "planner_model",
        "executor_base_url",
        "executor_model",
        "verify_commands",
    ]:
        val = orch.db.get_config(key, "(not set)")
        _con.console.print(f"  {key} = {val}")


def _show_history(orch: Orchestrator) -> None:
    rows = orch.db.execute(
        "SELECT id, started_at, mode, task, status FROM conversations "
        "ORDER BY started_at DESC LIMIT 10"
    ).fetchall()
    if not rows:
        _con.console.print("No conversation history.")
        return
    for row in rows:
        _con.console.print(
            f"  [{row['status']}] {row['started_at']} ({row['mode']}) — {row['task']}"
        )
