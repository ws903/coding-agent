# Local Coding Agent

A terminal-based coding agent that runs entirely on local hardware at zero cost. Uses open-source LLMs via Ollama (or any OpenAI-compatible backend) for inference. Implements a planner/executor architecture with automated verification.

Built from scratch in Python with no framework dependencies.

## Features

- **Planner/Executor Architecture** — A planning LLM breaks tasks into steps, an executor LLM implements each step, a deterministic verifier validates the result
- **Automated Verification** — Runs tests, linters, and type checkers after every step. Retries on failure, replans when stuck
- **Adaptive Edit Format** — Whole-file rewrites for small files (<300 lines), search/replace blocks for large files
- **Sandboxed Execution** — All file operations scoped to project root. No path traversal, no escapes
- **Two Modes** — Interactive REPL with plan approval, or autonomous fire-and-forget
- **Pluggable Models** — Swap models by changing a URL and model name. No code changes
- **SQLite Storage** — Conversations, plans, edits, and config all persisted per-project

## Requirements

- **Python 3.12+**
- **Ollama** (or any OpenAI-compatible API server)
- **GPU** — Recommended: NVIDIA RTX 5070 Ti (16GB VRAM) or similar. A 14B parameter model at Q4 quantization needs ~9GB VRAM

## Installation (Windows)

### 1. Install Python

Download Python 3.12+ from [python.org](https://www.python.org/downloads/). During installation, check **"Add Python to PATH"**.

Verify:

```powershell
python --version
```

### 2. Install Ollama

Download and install from [ollama.com](https://ollama.com/download). After installation, Ollama runs as a service automatically.

Pull the default model:

```powershell
ollama pull qwen3:14b
```

This downloads ~9GB. Verify it's running:

```powershell
ollama list
```

### 3. Install the Agent

Clone the repository and install in a virtual environment:

```powershell
git clone https://github.com/ws903/coding-agent.git
cd coding-agent

python -m venv .venv
.venv\Scripts\activate

pip install -e ".[dev]"
```

Verify the installation:

```powershell
python -m pytest tests/ -q
```

All 80 tests should pass.

## Usage

### Interactive Mode

Navigate to any project directory and launch the agent:

```powershell
python -m agent --project C:\path\to\your\project
```

The agent starts a REPL. Type a task in natural language:

```
Agent v0.1.0 | Model: qwen3:14b | Project: C:\Users\dave\myproject
Type a task or /help for commands.

> Add input validation to the user registration endpoint
```

The agent will:
1. Generate a plan and show it for approval
2. Execute each step (edit files, run commands)
3. Run verification after each step (tests, lint)
4. Retry or replan on failure

#### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show current task status |
| `/config` | Show/edit project configuration |
| `/history` | Show conversation history |
| `/abort` | Abort current execution |
| `/quit` | Exit the agent |

### Autonomous Mode

Run a task unattended. The agent executes the full plan-execute-verify loop, then exits with a status code (0 = success, 1 = failure):

```powershell
python -m agent --task "refactor the database module to use connection pooling" --auto --project C:\path\to\your\project
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--project` | `.` (current dir) | Project root directory |
| `--auto` | off | Run in autonomous mode |
| `--task` | — | Task description (required with `--auto`) |
| `--model` | `qwen3:14b` | Model name |
| `--base-url` | `http://localhost:11434/v1` | LLM API base URL |
| `--max-steps` | `20` | Maximum execution steps |
| `--step` | off | Approve each step individually |

## Configuration

Settings are stored per-project in SQLite at `{project}/.agent/agent.db`. Set them via the `/config` command or directly:

### Verification Commands

Configure which commands run after every step to validate changes:

```
/config verify_commands ["pytest", "ruff check .", "mypy ."]
```

If no verification commands are configured, the verifier is a no-op (steps always pass).

### Separate Planner/Executor Models

Use different models for planning vs execution:

```
/config planner_model qwen3:14b
/config executor_model qwen3-coder:14b
/config planner_base_url http://localhost:11434/v1
/config executor_base_url http://localhost:11434/v1
```

This lets you use a reasoning-focused model for planning and a code-optimized model for execution, with no code changes.

## Using Different Backends

The agent works with any server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

### Ollama (Default)

Zero-friction setup. Ollama runs as a service and auto-manages models:

```powershell
ollama pull qwen3:14b
python -m agent --base-url http://localhost:11434/v1 --model qwen3:14b
```

### TabbyAPI (ExLlamaV3)

Higher throughput for NVIDIA GPUs. Install [TabbyAPI](https://github.com/theroyallab/tabbyAPI) separately, then point the agent at it:

```powershell
python -m agent --base-url http://localhost:5000/v1 --model qwen3-14b-exl2
```

### LM Studio

```powershell
python -m agent --base-url http://localhost:1234/v1 --model qwen3-14b
```

### vLLM

```powershell
python -m agent --base-url http://localhost:8000/v1 --model qwen3-14b
```

## Recommended Models

| Model | VRAM (Q4) | Best For |
|-------|-----------|----------|
| Qwen3-14B | ~9 GB | All-rounder, supports `/think` toggle |
| Phi-4-Reasoning-Plus 14B | ~9 GB | Planning — strong reasoning benchmarks |
| Qwen3-Coder-14B | ~9 GB | Execution — code-optimized |
| Devstral Small 2 (24B) | ~14 GB | Execution — highest SWE-bench at this size |
| Qwen3-Coder-30B-A3B (MoE) | ~15-17 GB | Best local coding model (tight on 16GB) |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  CLI / REPL                      │
│  Interactive: user types, sees responses          │
│  Autonomous: user gives task, agent runs solo     │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │  ORCHESTRATOR   │
              │  plan → execute │
              │  → verify →     │
              │  replan/done    │
              └───┬─────────┬───┘
                  │         │
        ┌─────────▼──┐  ┌──▼──────────┐
        │  PLANNER   │  │  EXECUTOR   │
        │  Breaks    │  │  Edits files │
        │  task into │  │  Runs cmds   │
        │  steps     │  │  One step    │
        │            │  │  at a time   │
        └─────┬──────┘  └──────┬──────┘
              │                │
              └────────┬───────┘
                       │
              ┌────────▼────────┐
              │   LLM CLIENT   │
              │  base_url +    │
              │  model string  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   VERIFIER     │
              │  No LLM.       │
              │  Runs tests,   │
              │  lint, types.  │
              │  Pass/fail.    │
              └────────────────┘
```

### How It Works

1. **Plan** — The planner LLM receives the task + project file tree and outputs a structured step-by-step plan in Markdown
2. **Execute** — For each step, the executor LLM receives only the relevant files and produces file edits (create, rewrite, or search/replace)
3. **Apply** — Edits are applied to disk. Search/replace uses exact matching with a whitespace-normalized fallback
4. **Verify** — Configured commands (tests, lint, type-check) run. If all pass, move to the next step
5. **Retry** — On verification failure, the executor retries with the error output (up to 2 attempts)
6. **Replan** — After 2 failed retries, the planner generates a revised plan (up to 3 replans before giving up)

### Error Recovery

```
Step execution
    │
    ├── Verification passes → next step
    │
    ├── Verification fails → executor retry (attempt 2)
    │       │
    │       ├── passes → next step
    │       └── fails → replan (planner generates new plan)
    │
    └── Edit fails to apply → retry with actual file contents
            │
            ├── applies → verify
            └── fails again → replan
```

Every file edit is recorded with before/after snapshots in SQLite for full auditability.

## Project Structure

```
coding-agent/
├── src/agent/
│   ├── __main__.py        # Entry point
│   ├── cli.py             # REPL, argument parsing, slash commands
│   ├── orchestrator.py    # Plan-execute-verify state machine
│   ├── planner.py         # Planner LLM calls + plan parsing
│   ├── executor.py        # Executor LLM calls + edit parsing
│   ├── verifier.py        # Deterministic command runner
│   ├── llm_client.py      # Async HTTP client (OpenAI-compatible)
│   ├── sandbox.py         # Path validation, command sandboxing
│   ├── tools.py           # File operations (read, write, edit, list, search)
│   ├── parser.py          # Markdown plan parser, edit format parser
│   ├── db.py              # SQLite storage
│   ├── models.py          # Shared dataclasses
│   └── prompts/
│       ├── planner.md     # Planner system prompt
│       └── executor.md    # Executor system prompt
├── tests/                 # 80 tests
├── pyproject.toml
└── README.md
```

## Development

```powershell
# Create venv and install with dev dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_parser.py -v
```

## Remote Access

Since the agent runs in a terminal, you can access it from any machine via SSH into your Windows box:

```bash
ssh user@windows-machine
cd C:\path\to\coding-agent
.venv\Scripts\activate
python -m agent --project C:\path\to\target-project
```

## License

MIT
