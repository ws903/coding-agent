# src/agent/cli.py
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

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
    on_tool_call = None
    if stream:

        def on_token(chunk: str) -> None:
            console.print(chunk, end="", soft_wrap=True)

        def on_reasoning(chunk: str) -> None:
            # Dim italic so the user sees the model is thinking, but the
            # reasoning visually separates from the final answer.
            console.print(chunk, end="", soft_wrap=True, style="dim italic")

        def on_tool_call(name: str, args: dict) -> None:
            console.print(_format_tool_call(name, args), soft_wrap=True)

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


# Cap diff output so a 2000-line rewrite doesn't flood the terminal.
_MAX_DIFF_LINES = 40


def _render_edit_diff(
    path: str, action: str, before: str | None, after: str | None
) -> None:
    """Render an inline syntax-highlighted unified diff for a single edit."""
    from difflib import unified_diff

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

    console.print(
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

    Step boundaries get a `console.rule()` for visual separation; everything
    else gets dim text. Keeps the orchestrator's on_status(str) interface
    unchanged -- the CLI just renders the strings smarter.
    """
    if msg.startswith("Executing step"):
        console.rule(f"[bold cyan]{msg}[/bold cyan]", style="cyan", align="left")
    elif msg.startswith("Plan: "):
        # Already going to be rendered as a Panel by _approve_plan in
        # interactive mode; skip the dim duplicate to keep output clean.
        # In autonomous mode this is the only place the plan goal shows up.
        console.print(f"[bold]{msg}[/bold]")
    elif msg.startswith("All steps completed"):
        console.rule("[bold green]✓ " + msg + "[/bold green]", style="green")
    elif msg.startswith(("Max ", "Rolling back", "Replanning", "Aborted")):
        console.rule(f"[yellow]{msg}[/yellow]", style="yellow", align="left")
    else:
        console.print(f"[dim]{msg}[/dim]")


def _approve_plan(plan: Plan) -> bool:
    lines = [f"[bold green]{plan.goal}[/bold green]", ""]
    for step in plan.steps:
        lines.append(f"  [cyan]{step.id}.[/cyan] {step.action}")
        if step.files_needed:
            lines.append(f"     [dim]files: {', '.join(step.files_needed)}[/dim]")
        if step.verify_command:
            lines.append(f"     [dim]verify: {step.verify_command}[/dim]")
    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold]Proposed Plan[/bold] [dim]({len(plan.steps)} step{'s' if len(plan.steps) != 1 else ''})[/dim]",
            border_style="green",
            padding=(0, 1),
        )
    )
    answer = Prompt.ask("Proceed?", choices=["y", "n"], default="y")
    return answer == "y"


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
    """Cheap heuristic for obvious conversational input. Only matches single-
    word greetings; ambiguous short input falls through to the LLM classifier."""
    lower = text.lower().strip().rstrip("?!.,'\"")
    if not lower:
        return False
    words = lower.split()
    if len(words) > 3:
        return False
    return words[0] in _CHAT_TOKENS


_CLASSIFIER_PROMPT = (
    "Classify the user message as either CHAT or TASK.\n\n"
    "CHAT: small talk, greetings, expressions of feeling, simple questions "
    "about the agent itself (not the codebase). Examples: 'hi', 'how are "
    "you', 'thanks', 'what's up', 'are you online', 'you're cool'.\n\n"
    "TASK: anything about a codebase, file system, or software work -- "
    "explanations, edits, refactors, debugging, file lookups, exploration. "
    "Examples: 'add a docstring', 'what does this codebase do', 'list the "
    "files in src/', 'fix the bug in auth', 'explain the orchestrator'.\n\n"
    "When in doubt, prefer TASK -- the planner can decline or return a "
    "direct answer if no work is needed.\n\n"
    "Reply with exactly one word: CHAT or TASK"
)


async def _llm_classify_intent(orch: Orchestrator, user_input: str) -> str:
    """Returns 'chat' or 'task'. ~0.3s round-trip with think:false."""
    messages = [
        {"role": "system", "content": _CLASSIFIER_PROMPT},
        {"role": "user", "content": user_input},
    ]
    try:
        result = await orch.executor.llm.quick_chat_stream(
            messages, on_token=None, temperature=0.0
        )
    except Exception:
        # On any failure fall back to the safer TASK path so we never
        # accidentally route real work to fast-chat.
        return "task"
    return "chat" if "CHAT" in result.upper() else "task"


async def _route_input(orch: Orchestrator, user_input: str) -> str:
    """Tier-based routing: heuristic shortcuts first, LLM classifier for
    ambiguous middle. Long inputs (>10 words) skip the classifier."""
    if _looks_like_chat(user_input):
        return "chat"
    if len(user_input.split()) > 10:
        return "task"
    return await _llm_classify_intent(orch, user_input)


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


# Per-tool display glyphs + colors. Emojis degrade to text on non-unicode
# terminals via Rich's automatic fallback; we provide ASCII tags as the
# second element so the line still reads cleanly if emoji are stripped.
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

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold cyan]Coding Agent[/bold cyan] [dim]v0.1.0[/dim]",
            subtitle="[dim]/help for commands · /quit or 'exit' to leave[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
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
    _print_welcome_banner(orch, args, project_root)

    try:
        await _interactive_loop(orch)
    finally:
        await orch.executor.mcp.close()


def _build_repl_keybindings() -> KeyBindings:
    """Key bindings for the main REPL prompt.

    ESC clears the input line (matches Claude Code behavior at the prompt).
    Ctrl+C raises KeyboardInterrupt so the outer loop can exit cleanly.
    """
    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _esc_clear(event):
        event.app.current_buffer.reset()

    @kb.add("c-c")
    def _ctrl_c(event):
        event.app.exit(exception=KeyboardInterrupt)

    return kb


def _make_prompt_session() -> PromptSession:
    """Build the REPL PromptSession.

    Replaces Rich's Prompt.ask (which used input() underneath and produced
    garbled escape sequences for arrow keys, ESC, etc.). prompt_toolkit gives
    us cross-platform line editing, history within a session, and bindable
    keys -- including ESC to clear the input line.
    """
    return PromptSession(
        message=HTML("<ansicyan><b>></b></ansicyan> "),
        key_bindings=_build_repl_keybindings(),
    )


_PROMPT_SESSION: PromptSession | None = None


def _get_session() -> PromptSession:
    """Lazy singleton. Avoids constructing a PromptSession at import or loop
    entry, which triggers Windows-console probing that fails in headless CI.
    Tests mock `_get_user_input` so this never runs in unit tests."""
    global _PROMPT_SESSION
    if _PROMPT_SESSION is None:
        _PROMPT_SESSION = _make_prompt_session()
    return _PROMPT_SESSION


async def _get_user_input() -> str:
    """One-line shim that tests patch instead of mocking PromptSession itself."""
    return await _get_session().prompt_async()


async def _interactive_loop(orch: Orchestrator) -> None:
    while True:
        try:
            user_input = await _get_user_input()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in _EXIT_INPUTS or user_input in {"/quit", "/exit"}:
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

        if await _route_input(orch, user_input) == "chat":
            await _fast_chat(orch, user_input)
            continue

        # Spinner runs in a background thread; on_status prints flow above it
        # via Rich's Live infrastructure. Auto-disabled on non-TTY (e.g. when
        # output is piped to a file), so this is safe for all environments.
        with console.status(
            "[dim]Working...[/dim]", spinner="dots", spinner_style="cyan"
        ):
            result = await orch.run(
                user_input, mode="interactive", approve_plan=_approve_plan
            )
        status = result["status"]
        if status == "answered":
            # Render as markdown so headers, lists, code fences look right.
            # Planner answers often include markdown for codebase explanations.
            console.print(Markdown(result["answer"]))
            console.print()
        elif status == "completed":
            _show_change_summary(orch, result.get("conv_id", ""))
            _show_token_usage(orch)
        elif status == "failed":
            console.print(f"[red]Task failed: {result.get('reason', 'unknown')}[/red]")
            _show_change_summary(orch, result.get("conv_id", ""))
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
        _show_change_summary(orch, result.get("conv_id", ""))
        _show_token_usage(orch)
        return 0
    console.print(f"[red]Task failed: {result.get('reason', 'unknown')}[/red]")
    _show_change_summary(orch, result.get("conv_id", ""))
    _show_token_usage(orch)
    return 1


_ACTION_GLYPH = {
    "create": ("🆕", "green", "created"),
    "rewrite": ("📝", "yellow", "rewrote"),
    "search_replace": ("✏️ ", "yellow", "edited"),
}


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
        console.print("[green]✓ Task completed successfully.[/green]")
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

    console.print(tree)
    console.print()


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
        console.print(table)
        console.print()
    except (TypeError, ValueError, KeyError, AttributeError):
        # Graceful no-op for tests that mock orch.token_usage(), or for
        # backends that don't report usage in their response.
        return


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
