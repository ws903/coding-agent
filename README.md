# Local Coding Agent

[![CI](https://github.com/ws903/coding-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ws903/coding-agent/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ws903/d00dd6063816632622587ade226668a3/raw/coverage-badge.json)](https://github.com/ws903/coding-agent/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fws903%2Fcoding-agent%2Fmain%2Fpyproject.toml)](https://github.com/ws903/coding-agent)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A terminal-based coding agent that runs entirely on local hardware at zero cost. Uses open-source LLMs via [Ollama](https://ollama.com) (or any OpenAI-compatible backend) for inference. Implements a planner/executor architecture with automated verification, git-based rollback, and lint-gated edits.

Built from scratch in Python. No LangChain, no frameworks, no API fees.

## Features

- **Planner/Executor Architecture** -- A planning LLM breaks tasks into steps, an executor LLM implements each step, a deterministic verifier validates the result
- **Automated Verification** -- Runs tests, linters, and type checkers after every step. Retries on failure, replans when stuck
- **Lint-Gated Edits** -- Runs [ruff](https://docs.astral.sh/ruff/) on every edited Python file. Only newly introduced lint errors trigger rollback; pre-existing errors pass through
- **Git Snapshots & Rollback** -- Snapshots working tree before each step. Automatically rolls back on failure before replanning
- **Adaptive Edit Format** -- Whole-file rewrites for small files (<300 lines), search/replace blocks for large files. Whitespace-normalized matching with relative indent preservation
- **Sandboxed Execution** -- All file operations scoped to project root. Command allowlist blocks destructive operations (`rm -rf`, `sudo`, force-push, etc.)
- **Token Tracking** -- Displays prompt/completion tokens and LLM call counts after each task
- **Two Modes** -- Interactive REPL with plan approval, or autonomous fire-and-forget
- **Pluggable Models** -- Swap models by changing a URL and model name. No code changes
- **Pluggable Backends** -- Ollama, TabbyAPI/ExLlamaV3, LM Studio, vLLM, or any OpenAI-compatible server
- **SQLite Storage** -- Conversations, plans, edits, and config all persisted per-project

## Quickstart

```powershell
# 1. Install uv (manages Python for you)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install Ollama and pull the default model
#    Download from https://ollama.com/download, then:
ollama pull qwen3.6:35b

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
- **GPU** -- Recommended: NVIDIA GPU with 16GB+ VRAM. The default model (`qwen3.6:35b`, MoE 35B/3B-active) needs ~16GB at Q3 or ~22GB at Q4 with RAM offload. 32GB+ system RAM recommended for offload. For lower VRAM, see [Recommended Models](#recommended-models) -- `qwen3:14b` (~9GB) is the small-VRAM fallback

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
Agent v0.1.0 | Model: qwen3.6:35b | Project: C:\Users\dave\myproject
Type a task or /help for commands.

> Add input validation to the user registration endpoint
```

The agent will:
1. Generate a plan and show it for approval
2. Execute each step (edit files, run commands)
3. Lint-check every edited Python file (rollback on new errors)
4. Verify after each step (tests, lint, type-check)
5. Retry or replan on failure

#### Slash Commands

| Command    | Description                       |
|------------|-----------------------------------|
| `/help`    | Show available commands           |
| `/status`  | Show current task progress        |
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
| `--model`    | `qwen3.6:35b` (env: `AGENT_MODEL`)   | Model name                           |
| `--base-url` | `http://localhost:11434/v1` (env: `AGENT_BASE_URL`) | LLM API base URL                     |
| `--max-steps`| `20`                             | Maximum execution steps              |
| `--step`     | off                              | Approve each step individually       |

## Configuration

Three layers, in precedence order: **CLI flag → env var → `.env` file → built-in default**.

For things that vary per-project (e.g. `verify_commands`), use SQLite via `/config` in interactive mode (stored at `{project}/.agent/agent.db`).

For things that should follow you across all projects (e.g. pointing at a remote Ollama backend), drop a gitignored `.env` in the coding-agent repo root:

```
# coding-agent/.env  (already in .gitignore)
AGENT_BASE_URL=http://192.168.1.42:11434/v1
AGENT_MODEL=qwen3.6:35b
```

The agent loads this on every invocation, so `uv run agent` from any project picks up your config.

### Verification Commands

Configure which commands run after every step:

```
/config verify_commands ["pytest", "ruff check .", "mypy ."]
```

If no verification commands are configured, the verifier is a no-op.

### Dual-Model Setup

Use a reasoning-focused model for planning and a code-optimized model for execution:

```
/config planner_model qwen3.6:35b
/config executor_model qwen3.6:35b
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

For 16GB VRAM with 32GB+ system RAM (e.g. NVIDIA RTX 5070 Ti, RTX 4080):

| Model | VRAM | SWE-bench Verified | Best For |
|-------|------|-------|----------|
| **Qwen3.6-35B-A3B** | ~16 GB (Q3) or ~22 GB (Q4, RAM offload) | **73.4** | **Default.** MoE 35B/3B-active. Unified planner+executor, thinking mode built-in, 262K context, multimodal |
| Qwen3-14B (dense) | ~9 GB | ~45 | **Small-VRAM fallback.** No offload, predictable. Use if 35B's offload doesn't suit your CPU/RAM |
| Phi-4-Reasoning-Plus 14B | ~9 GB | -- | Planner-only specialist if you split planner/executor |
| Devstral Small 2 (24B) | ~14 GB | ~46 | Executor-only specialist if you split |

The default (`qwen3.6:35b`) handles both planner and executor roles in a single model -- the MoE router internally specializes per token, so a manual split into separate planner/executor models gives little benefit and adds model-swap latency.

For 24GB+ VRAM (RTX 4090, RTX 5080+):

| Model | VRAM (Q4) | Best For |
|-------|-----------|----------|
| Qwen3.6-27B (dense) | ~17 GB | Dense alternative to 35B MoE, no offload, multimodal |
| Qwen3-Coder-30B-A3B (MoE) | ~19 GB | Code-specialized MoE if you don't need vision |

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
              │  -> lint-gate -> │
              │  verify -> next/ │
              │  rollback/replan │
              └───┬────┬────┬───┘
                  │    │    │
        ┌─────────▼┐ ┌▼────▼──────────┐
        │ PLANNER  │ │   EXECUTOR     │
        │ Breaks   │ │  Edits files    │
        │ task into│ │  Runs cmds      │
        │ steps    │ │  One step       │
        │          │ │  at a time      │
        └────┬─────┘ └───────┬────────┘
             │               │
             └───────┬───────┘
                     │
            ┌────────▼────────┐
            │   LLM CLIENT   │
            │  Shared client  │
            │  Retry+backoff  │
            │  Token tracking │
            └────────┬────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
   ┌─────▼─────┐ ┌──▼────┐ ┌───▼──────┐
   │ VERIFIER  │ │ LINT  │ │ GIT OPS  │
   │ Tests,    │ │ GATE  │ │ Snapshot │
   │ lint,     │ │ ruff  │ │ before   │
   │ types.    │ │ diff  │ │ each     │
   │ Pass/fail │ │ gate  │ │ step,    │
   │           │ │       │ │ rollback │
   └───────────┘ └───────┘ └──────────┘
```

### How It Works

1. **Plan** -- The planner LLM receives the task + project file tree + environment info and outputs a step-by-step plan
2. **Snapshot** -- Git snapshots the working tree before each step for safe rollback
3. **Execute** -- The executor receives only the relevant files and produces file edits
4. **Lint Gate** -- Every edited Python file is checked with ruff. Pre-existing errors are ignored; only newly introduced errors trigger rollback
5. **Apply** -- Edits are applied with exact matching and a whitespace-normalized fallback that preserves relative indentation
6. **Verify** -- Configured commands (tests, lint, type-check) run. All must pass to proceed
7. **Retry** -- On failure, the executor retries with error output (up to 2 attempts)
8. **Rollback & Replan** -- After 2 failed retries, git rolls back, and the planner generates a revised plan. Previously completed steps are skipped (up to 3 replans)

### Error Recovery

```
Step execution
    |
    ├── Lint gate fails (new errors) --> rollback file, retry
    |
    ├── Verification passes --> next step
    |
    ├── Verification fails --> executor retry (attempt 2)
    |       |
    |       ├── passes --> next step
    |       └── fails --> git rollback + replan
    |
    └── Edit fails to apply --> retry with actual file contents
            |
            ├── applies --> lint gate --> verify
            └── fails again --> git rollback + replan
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
│   ├── llm_client.py      # Async HTTP client with retry + backoff
│   ├── sandbox.py         # Path validation, command sandboxing
│   ├── tools.py           # File operations (read, write, edit, list, search)
│   ├── parser.py          # Markdown plan parser, edit format parser
│   ├── lint_gate.py       # Ruff-based lint gating with pre/post diffing
│   ├── git_ops.py         # Git snapshot, rollback, diff
│   ├── command_policy.py  # Command allowlist and block patterns
│   ├── db.py              # SQLite storage
│   ├── models.py          # Shared dataclasses
│   └── prompts/
│       ├── planner.md     # Planner system prompt
│       └── executor.md    # Executor system prompt
├── tests/                 # 245 tests
├── pyproject.toml
├── uv.lock
└── README.md
```

## Development

```bash
uv sync                              # Install all dependencies
uv run pytest tests/ -v              # Run tests
uv run pytest tests/ --cov           # Run tests with coverage
uv run pytest tests/test_parser.py   # Run a specific test file
uvx ruff check src/ tests/           # Lint
uvx ruff format src/ tests/          # Format
```

## Roadmap

- [ ] **Native tool calling** -- Ollama and TabbyAPI now support OpenAI-compatible function calling. Migrate from text-parsed edits to structured tool use for higher reliability
- [ ] **Structured output** -- Use grammar-constrained generation (GBNF/JSON schema) to guarantee valid edit blocks instead of regex parsing
- [ ] **Prompt caching** -- TabbyAPI/ExLlamaV3 supports prefix caching. Reuse KV cache across executor calls on the same file to cut time-to-first-token
- [ ] **Streaming output** -- Stream executor responses to show progress in real-time
- [ ] **Step-level auto-commits** -- Git commit after each successful step with descriptive message
- [ ] **Cross-step context** -- Feed executor a summary of prior steps to reduce redundant reads
- [ ] **Codebase indexing** -- Embed files with a local model for retrieval. Give the planner semantic search over the project instead of just a file tree
- [ ] **Parallel step execution** -- When planner identifies independent steps, execute them concurrently
- [ ] **MCP integration** -- Connect external tools via Model Context Protocol servers
- [ ] **Web UI** -- Browser-based chat interface accessible from phone/laptop, sharing the same inference backend

## Remote Access

The agent runs in a terminal, so you can access it from any machine via SSH:

```bash
ssh user@your-machine
cd /path/to/coding-agent
uv run agent --project /path/to/target-project
```

## Web Chat UI (Open WebUI)

[Open WebUI](https://docs.openwebui.com/) runs alongside Ollama for browser-based chat against the same models -- useful for phone access (pair with Tailscale) or general non-coding chat. Requires Docker Desktop and `OLLAMA_HOST=0.0.0.0` set so the container can reach the host's Ollama.

```powershell
docker run -d `
  -p 8080:8080 `
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
  -v open-webui:/app/backend/data `
  --name open-webui `
  --restart always `
  ghcr.io/open-webui/open-webui:main
```

First-run setup at `http://localhost:8080` -- the first account created becomes admin.

For a desktop launcher on Windows, use [`scripts/start-open-webui.bat`](scripts/start-open-webui.bat) -- double-click to start the container (creates it on first run) and open the browser.

## License

[MIT](LICENSE)
