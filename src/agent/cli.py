# src/agent/cli.py
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from agent.db import AgentDB
from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.agents_manager import AgentsManager
from agent.mcp_manager import MCPManager, load_mcp_config
from agent.models import Plan
from agent.skills_manager import SkillsManager
from agent.orchestrator import Orchestrator
from agent.planner import Planner
from agent.sandbox import Sandbox
from agent.tools import FileTools
from agent.verifier import Verifier

# Load .env from the coding-agent repo root (gitignored) so users can set
# AGENT_BASE_URL / AGENT_MODEL once without touching shell config.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

DEFAULT_MODEL = "qwen3.6:35b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"

console = Console()


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
    if stream:

        def on_token(chunk: str) -> None:
            console.print(chunk, end="", soft_wrap=True)

        def on_reasoning(chunk: str) -> None:
            # Dim italic so the user sees the model is thinking, but the
            # reasoning visually separates from the final answer.
            console.print(chunk, end="", soft_wrap=True, style="dim italic")

    mcp = MCPManager(load_mcp_config(project_root))
    skills = SkillsManager(project_root)
    agents = AgentsManager(project_root)
    executor = Executor(
        executor_client,
        tools,
        on_token=on_token,
        on_reasoning=on_reasoning,
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

# Short conversational inputs that should bypass the planner entirely.
# Conservative -- we only fast-path obvious greetings; anything ambiguous
# falls through to the planner so we don't skip real work.
_CHAT_TOKENS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "thanks",
    "thank",
    "ok",
    "okay",
    "cool",
    "great",
    "nice",
    "bye",
    "goodbye",
    "morning",
}

_FAST_CHAT_PROMPT = (
    "You are a friendly assistant embedded in a coding agent CLI. "
    "Reply briefly (1-2 sentences). The user is making conversation, "
    "not asking for code work."
)


def _looks_like_chat(text: str) -> bool:
    """Detect trivial conversational input that doesn't need the full planner."""
    lower = text.lower().strip().rstrip("?!.,'\"")
    if not lower:
        return False
    words = lower.split()
    if len(words) > 3:
        return False
    return words[0] in _CHAT_TOKENS


async def _fast_chat(orch: Orchestrator, user_input: str) -> None:
    """Bypass planner+executor AND the model's reasoning phase.

    Uses Ollama's native /api/chat with `think: false`. For thinking models
    like qwen3.6, this cuts "hi"-style replies from ~14s to ~3s by skipping
    the silent reasoning pass.
    """
    messages = [
        {"role": "system", "content": _FAST_CHAT_PROMPT},
        {"role": "user", "content": user_input},
    ]

    def on_token(chunk: str) -> None:
        console.print(chunk, end="", soft_wrap=True)

    await orch.executor.llm.quick_chat_stream(
        messages, on_token=on_token, temperature=0.7
    )
    console.print()


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
    if orch.executor.mcp.connected_servers:
        console.print(
            f"[dim]MCP servers: {', '.join(orch.executor.mcp.connected_servers)}[/dim]"
        )

    console.print(
        f"[bold]Agent v0.1.0[/bold] | Model: {args.model} | Project: {project_root}"
    )
    console.print("Type a task or /help for commands.\n")

    try:
        await _interactive_loop(orch)
    finally:
        await orch.executor.mcp.close()


async def _interactive_loop(orch: Orchestrator) -> None:
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

        if _looks_like_chat(user_input):
            await _fast_chat(orch, user_input)
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

    project_root = _find_project_root(Path(args.project))
    orch = build_orchestrator(
        project_root, args.base_url, args.model, max_steps=args.max_steps
    )

    console.print(f"[bold]Autonomous mode[/bold] | Task: {args.task}")
    await orch.executor.mcp.connect()
    if orch.executor.mcp.connected_servers:
        console.print(
            f"[dim]MCP servers: {', '.join(orch.executor.mcp.connected_servers)}[/dim]"
        )

    try:
        result = await orch.run(args.task, mode="autonomous")
    finally:
        await orch.executor.mcp.close()

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
