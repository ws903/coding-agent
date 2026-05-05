# src/agent/cli.py
import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from agent.db import AgentDB
from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.models import Plan
from agent.orchestrator import Orchestrator
from agent.planner import Planner
from agent.sandbox import Sandbox
from agent.tools import FileTools
from agent.verifier import Verifier


console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Local coding agent with planner/executor architecture",
    )
    parser.add_argument("--auto", action="store_true", help="Run in autonomous mode")
    parser.add_argument("--task", type=str, help="Task description (required for --auto)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max execution steps")
    parser.add_argument("--step", action="store_true", help="Approve each step individually")
    parser.add_argument(
        "--model", type=str, default="qwen3:14b", help="Model name"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="LLM API base URL",
    )
    parser.add_argument(
        "--project", type=str, default=".", help="Project root directory"
    )
    return parser.parse_args(argv)


def build_orchestrator(
    project_root: Path,
    base_url: str = "http://localhost:11434/v1",
    model: str = "qwen3:14b",
    verify_commands: list[str] | None = None,
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
    executor = Executor(executor_client, tools)

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
        on_status=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )


def _approve_plan(plan: Plan) -> bool:
    console.print("\n[bold]Proposed Plan:[/bold]")
    console.print(f"[green]{plan.goal}[/green]\n")
    for step in plan.steps:
        console.print(f"  {step.id}. {step.action}")
        if step.files_needed:
            console.print(f"     Files: {', '.join(step.files_needed)}")
    console.print()
    answer = Prompt.ask("Proceed?", choices=["y", "n"], default="y")
    return answer == "y"


SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/status": "Show current task status",
    "/config": "Show/edit project configuration",
    "/history": "Show conversation history",
    "/abort": "Abort current execution",
    "/quit": "Exit the agent",
}


async def run_interactive(args: argparse.Namespace) -> None:
    project_root = Path(args.project).resolve()
    orch = build_orchestrator(project_root, args.base_url, args.model)

    console.print(f"[bold]Agent v0.1.0[/bold] | Model: {args.model} | Project: {project_root}")
    console.print("Type a task or /help for commands.\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input == "/help":
            for cmd, desc in SLASH_COMMANDS.items():
                console.print(f"  [bold]{cmd}[/bold] — {desc}")
            continue
        elif user_input == "/config":
            _show_config(orch)
            continue
        elif user_input == "/history":
            _show_history(orch)
            continue
        elif user_input.startswith("/"):
            console.print(f"[red]Unknown command: {user_input}[/red]")
            continue

        result = await orch.run(user_input, mode="interactive", approve_plan=_approve_plan)
        status = result["status"]
        if status == "completed":
            console.print("[green]Task completed successfully.[/green]\n")
        elif status == "failed":
            console.print(f"[red]Task failed: {result.get('reason', 'unknown')}[/red]\n")
        elif status == "aborted":
            console.print("[yellow]Task aborted.[/yellow]\n")


async def run_autonomous(args: argparse.Namespace) -> int:
    if not args.task:
        console.print("[red]--task is required for autonomous mode[/red]")
        return 1

    project_root = Path(args.project).resolve()
    orch = build_orchestrator(project_root, args.base_url, args.model)

    console.print(f"[bold]Autonomous mode[/bold] | Task: {args.task}")
    result = await orch.run(args.task, mode="autonomous")

    if result["status"] == "completed":
        console.print("[green]Task completed successfully.[/green]")
        return 0
    else:
        console.print(f"[red]Task failed: {result.get('reason', 'unknown')}[/red]")
        return 1


def _show_config(orch: Orchestrator) -> None:
    console.print("[bold]Configuration:[/bold]")
    for key in [
        "planner_base_url", "planner_model",
        "executor_base_url", "executor_model",
        "verify_commands",
    ]:
        val = orch.db.get_config(key, "(not set)")
        console.print(f"  {key} = {val}")


def _show_history(orch: Orchestrator) -> None:
    rows = orch.db.execute(
        "SELECT id, started_at, mode, task, status FROM conversations ORDER BY started_at DESC LIMIT 10"
    ).fetchall()
    if not rows:
        console.print("No conversation history.")
        return
    for row in rows:
        console.print(
            f"  [{row['status']}] {row['started_at']} ({row['mode']}) — {row['task']}"
        )
