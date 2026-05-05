# Local Coding Agent

[![CI](https://github.com/ws903/coding-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ws903/coding-agent/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ws903/d00dd6063816632622587ade226668a3/raw/coverage-badge.json)](https://github.com/ws903/coding-agent/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fws903%2Fcoding-agent%2Fmain%2Fpyproject.toml)](https://github.com/ws903/coding-agent)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A terminal-based coding agent that runs entirely on local hardware at zero cost. Uses open-source LLMs via [Ollama](https://ollama.com) (or any OpenAI-compatible backend) for inference. Implements a planner/executor architecture with automated verification.

Built from scratch in Python. No LangChain, no frameworks, no API fees.

## Features

- **Planner/Executor Architecture** -- A planning LLM breaks tasks into steps, an executor LLM implements each step, a deterministic verifier validates the result
- **Automated Verification** -- Runs tests, linters, and type checkers after every step. Retries on failure, replans when stuck
- **Adaptive Edit Format** -- Whole-file rewrites for small files (<300 lines), search/replace blocks for large files
- **Sandboxed Execution** -- All file operations scoped to project root. No path traversal, no escapes
- **Two Modes** -- Interactive REPL with plan approval, or autonomous fire-and-forget
- **Pluggable Models** -- Swap models by changing a URL and model name. No code changes
- **Pluggable Backends** -- Ollama, TabbyAPI/ExLlamaV3, LM Studio, vLLM, or any OpenAI-compatible server
- **SQLite Storage** -- Conversations, plans, edits, and config all persisted per-project

## Quickstart

```powershell
# 1. Install uv (manages Python for you)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install Ollama and pull a model
#    Download from https://ollama.com/download, then:
ollama pull qwen3:14b

# 3. Clone and install
git clone https://github.com/ws903/coding-agent.git
cd coding-agent
uv sync

# 4. Run
uv run agent --project C:\path\to\your\project
```

## Requirements

- **[uv](https://docs.astral.sh/uv/)** -- Installs and manages Python automatically
- **[Ollama](https://ollama.com)** (or any OpenAI-compatible API server)
- **GPU** -- Recommended: NVIDIA GPU with 10GB+ VRAM. A 14B parameter model at Q4 quantization needs ~9GB

## Installation

### Windows

Install uv:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Then

```bash
git clone https://github.com/ws903/coding-agent.git
cd coding-agent
uv sync
```

`uv sync` creates the venv, installs Python 3.13+ if needed, and resolves all dependencies from the lockfile.

Verify:

```bash
uv run pytest tests/ -q
```

## Usage

### Interactive Mode

```bash
uv run agent --project /path/to/your/project
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
3. Verify after each step (tests, lint, type-check)
4. Retry or replan on failure

#### Slash Commands

| Command    | Description                       |
|------------|-----------------------------------|
| `/help`    | Show available commands           |
| `/status`  | Show current task status          |
| `/config`  | Show/edit project configuration   |
| `/history` | Show conversation history         |
| `/abort`   | Abort current execution           |
| `/quit`    | Exit the agent                    |

### Autonomous Mode

Run a task unattended. Exits with status code 0 (success) or 1 (failure):

```bash
uv run agent --task "refactor the database module to use connection pooling" --auto --project /path/to/project
```

### CLI Reference

| Flag         | Default                          | Description                          |
|--------------|----------------------------------|--------------------------------------|
| `--project`  | `.`                              | Project root directory               |
| `--auto`     | off                              | Run in autonomous mode               |
| `--task`     | --                               | Task description (required w/ `--auto`) |
| `--model`    | `qwen3:14b`                      | Model name                           |
| `--base-url` | `http://localhost:11434/v1`       | LLM API base URL                     |
| `--max-steps`| `20`                             | Maximum execution steps              |
| `--step`     | off                              | Approve each step individually       |

## Configuration

Settings are stored per-project in SQLite at `{project}/.agent/agent.db`. Set them via `/config` in interactive mode.

### Verification Commands

Configure which commands run after every step:

```
/config verify_commands ["pytest", "ruff check .", "mypy ."]
```

If no verification commands are configured, the verifier is a no-op.

### Dual-Model Setup

Use a reasoning-focused model for planning and a code-optimized model for execution:

```
/config planner_model qwen3:14b
/config executor_model qwen3-coder:14b
```

Both point at the same backend by default. You can also split backends:

```
/config planner_base_url http://localhost:11434/v1
/config executor_base_url http://localhost:5000/v1
```

## Inference Backends

The agent works with any server exposing an OpenAI-compatible `/v1/chat/completions` endpoint.

| Backend | URL | Notes |
|---------|-----|-------|
| [Ollama](https://ollama.com) | `http://localhost:11434/v1` | Zero-friction default. Auto-manages models |
| [TabbyAPI](https://github.com/theroyallab/tabbyAPI) (ExLlamaV3) | `http://localhost:5000/v1` | Fastest on NVIDIA. EXL2/EXL3 quantization |
| [LM Studio](https://lmstudio.ai) | `http://localhost:1234/v1` | GUI-based model management |
| [vLLM](https://github.com/vllm-project/vllm) | `http://localhost:8000/v1` | Production-grade serving |

Switch backends with `--base-url`:

```bash
uv run agent --base-url http://localhost:5000/v1 --model qwen3-14b-exl2
```

## Recommended Models

| Model | VRAM (Q4) | Best For |
|-------|-----------|----------|
| **Qwen3-14B** | ~9 GB | All-rounder, `/think` toggle for reasoning vs speed |
| Phi-4-Reasoning-Plus 14B | ~9 GB | Planning -- strongest reasoning at 14B |
| Qwen3-Coder-14B | ~9 GB | Execution -- code-optimized |
| Devstral Small 2 (24B) | ~14 GB | Execution -- highest SWE-bench at this size |
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
              │  plan -> execute │
              │  -> verify ->    │
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

1. **Plan** -- The planner LLM receives the task + project file tree and outputs a step-by-step plan in Markdown
2. **Execute** -- For each step, the executor receives only the relevant files and produces file edits
3. **Apply** -- Edits are applied to disk with exact matching and a whitespace-normalized fallback
4. **Verify** -- Configured commands (tests, lint, type-check) run. All must pass to proceed
5. **Retry** -- On failure, the executor retries with error output (up to 2 attempts)
6. **Replan** -- After 2 failed retries, the planner generates a revised plan (up to 3 replans)

### Error Recovery

```
Step execution
    |
    |-- Verification passes --> next step
    |
    |-- Verification fails --> executor retry (attempt 2)
    |       |
    |       |-- passes --> next step
    |       +-- fails --> replan (planner generates new plan)
    |
    +-- Edit fails to apply --> retry with actual file contents
            |
            |-- applies --> verify
            +-- fails again --> replan
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
├── uv.lock
└── README.md
```

## Development

```bash
uv sync                              # Install all dependencies
uv run pytest tests/ -v              # Run tests
uv run pytest tests/test_parser.py   # Run a specific test file
uvx ruff check src/ tests/           # Lint
uvx ruff format src/ tests/          # Format
```

## Roadmap

- [ ] **Native tool calling** -- Ollama and TabbyAPI now support OpenAI-compatible function calling. Migrate from text-parsed edits to structured tool use for higher reliability
- [ ] **Prompt caching** -- TabbyAPI/ExLlamaV3 supports prefix caching. Reuse KV cache across executor calls on the same file to cut time-to-first-token
- [ ] **Structured output** -- Use grammar-constrained generation (GBNF/JSON schema) to guarantee valid edit blocks instead of regex parsing
- [ ] **Codebase indexing** -- Embed files with a local model for retrieval. Give the planner semantic search over the project instead of just a file tree
- [ ] **Parallel step execution** -- When planner identifies independent steps, execute them concurrently
- [ ] **Slash commands as markdown files** -- Discoverable, user-extensible commands (`.agent/commands/*.md`)
- [ ] **Layered configuration** -- Global (`~/.agent/settings.json`) + project (`.agent/settings.json`) config, modeled after Claude Code
- [ ] **Hooks** -- Pre/post tool use event system for validation and logging
- [ ] **MCP integration** -- Connect external tools via Model Context Protocol servers
- [ ] **Web UI** -- Browser-based chat interface accessible from phone/laptop, sharing the same inference backend
- [ ] **Git integration** -- Auto-commit after successful steps, branch management

## Remote Access

The agent runs in a terminal, so you can access it from any machine via SSH:

```bash
ssh user@windows-machine
cd /path/to/coding-agent
uv run agent --project /path/to/target-project
```

## License

[MIT](LICENSE)
