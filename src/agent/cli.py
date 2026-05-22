# src/agent/cli.py
import argparse
import json
import os
from pathlib import Path

from rich.console import Console
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

DEFAULT_MODEL = "qwen3.6:35b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"

console = Console()


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
        max_steps=max_steps,
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
    orch = build_orchestrator(
        project_root, args.base_url, args.model, max_steps=args.max_steps
    )

    console.print(
        f"[bold]Agent v0.1.0[/bold] | Model: {args.model} | Project: {project_root}"
    )
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
        elif user_input == "/status":
            _show_status(orch)
            continue
        elif user_input == "/abort":
            orch.abort()
            console.print("[yellow]Abort requested.[/yellow]")
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

        result = await orch.run(
            user_input, mode="interactive", approve_plan=_approve_plan
        )
        status = result["status"]
        if status == "answered":
            console.print(result["answer"] + "\n")
        elif status == "completed":
            console.print("[green]Task completed successfully.[/green]")
            _show_token_usage(orch)
        elif status == "failed":
            console.print(f"[red]Task failed: {result.get('reason', 'unknown')}[/red]")
            _show_token_usage(orch)
        elif status == "aborted":
            console.print("[yellow]Task aborted.[/yellow]\n")


async def run_autonomous(args: argparse.Namespace) -> int:
    if not args.task:
        console.print("[red]--task is required for autonomous mode[/red]")
        return 1

    project_root = Path(args.project).resolve()
    orch = build_orchestrator(
        project_root, args.base_url, args.model, max_steps=args.max_steps
    )

    console.print(f"[bold]Autonomous mode[/bold] | Task: {args.task}")
    result = await orch.run(args.task, mode="autonomous")

    if result["status"] == "answered":
        console.print(result["answer"])
        return 0
    if result["status"] == "completed":
        console.print("[green]Task completed successfully.[/green]")
        _show_token_usage(orch)
        return 0
    console.print(f"[red]Task failed: {result.get('reason', 'unknown')}[/red]")
    _show_token_usage(orch)
    return 1


def _show_token_usage(orch: Orchestrator) -> None:
    try:
        usage = orch.token_usage()
        p = usage["planner"]
        e = usage["executor"]
        total = p["total_tokens"] + e["total_tokens"]
        calls = p["calls"] + e["calls"]
        if total > 0:
            console.print(
                f"[dim]Tokens: {total:,} ({p['total_tokens']:,} planner + "
                f"{e['total_tokens']:,} executor) | {calls} LLM calls[/dim]\n"
            )
    except (TypeError, KeyError, AttributeError):
        pass


def _show_status(orch: Orchestrator) -> None:
    s = orch.status()
    if not s["task"]:
        console.print("No task running.")
        return
    console.print(f"[bold]Task:[/bold] {s['task']}")
    console.print(f"[bold]Step:[/bold] {s['current_step']}")
    console.print(
        f"[bold]Progress:[/bold] {s['steps_executed']}/{s['total_steps']} steps executed"
    )


def _show_config(orch: Orchestrator) -> None:
    console.print("[bold]Configuration:[/bold]")
    for key in [
        "planner_base_url",
        "planner_model",
        "executor_base_url",
        "executor_model",
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
