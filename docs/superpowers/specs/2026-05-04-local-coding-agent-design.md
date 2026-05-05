# Local Coding Agent — Design Spec

## Overview

A terminal-based coding agent that runs entirely on local hardware (Windows, RTX 5070 Ti, 16GB VRAM) at zero cost. Uses open-source LLMs via Ollama or TabbyAPI for inference. Implements a planner/executor architecture with automated verification. Supports both interactive and autonomous modes.

Built from scratch in Python with no framework dependencies. Models are pluggable via a configurable OpenAI-compatible API endpoint.

Future phases will add a web-based chatbot UI accessible from any device, sharing the same inference backend.

## Constraints

- **Hardware**: Windows, NVIDIA RTX 5070 Ti, 16GB GDDR7 VRAM
- **Cost**: $0 forever — fully local, no API fees
- **Models**: Qwen3-14B primary (9GB Q4, fits comfortably). Phi-4-Reasoning-Plus and Qwen3-Coder-14B as future planner/executor split candidates
- **Inference backend**: Any OpenAI-compatible endpoint. Ollama for day one, TabbyAPI/ExLlamaV3 for performance later
- **Language**: Python, no framework dependencies (no LangChain, no smolagents)
- **Storage**: SQLite for everything (config, conversations, history)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLI / REPL                        │
│  Interactive mode: user types, sees responses        │
│  Autonomous mode: user gives task, agent runs solo   │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  ORCHESTRATOR   │
              │  Owns the loop: │
              │  plan → execute │
              │  → verify →     │
              │  replan/done    │
              └───┬─────────┬───┘
                  │         │
        ┌─────────▼──┐  ┌──▼──────────┐
        │  PLANNER   │  │  EXECUTOR   │
        │            │  │             │
        │ Reasons    │  │ Edits files │
        │ about task │  │ Runs cmds   │
        │ Outputs    │  │ One step    │
        │ step list  │  │ at a time   │
        └─────┬──────┘  └──────┬──────┘
              │                │
              └────────┬───────┘
                       │
              ┌────────▼────────┐
              │   LLM CLIENT   │
              │  base_url +    │
              │  model string  │
              │  That's it     │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   VERIFIER     │
              │  No LLM        │
              │  Runs tests,   │
              │  lint, types   │
              │  Pass/fail     │
              └────────────────┘
```

### Components

**1. CLI/REPL**

Entry point. Two modes:

- **Interactive**: Terminal REPL with streaming output. Shows plan for approval before execution. Supports step-by-step approval with `--step` flag. Commands: `/status`, `/plan`, `/abort`, `/config`, `/history`.
- **Autonomous**: `python -m agent --task "description" --auto`. Runs full loop unattended, logs everything to SQLite, exits with status code 0 (success) or 1 (failure). Optional `--max-steps N` to cap execution.

**2. Orchestrator**

State machine that drives the plan-execute-verify loop:

```
receive task
    → planner.generate_plan(task, project_context)
    → for each step in plan:
        → executor.execute(step, relevant_files)
        → apply file edits
        → verifier.run()
        → if verify fails:
            → executor.retry(step, relevant_files, error_output)  [attempt 2]
            → if still fails:
                → planner.replan(task, plan, failed_step, error)
                → restart loop with new plan
    → report results
```

In interactive mode, the orchestrator pauses after planning to show the plan and wait for approval. In autonomous mode, it runs straight through.

**3. Planner**

An LLM call with a reasoning-focused system prompt. Receives the task description and a project summary (file tree, key file snippets). Outputs a structured plan in Markdown:

```markdown
## Plan: Add user authentication

### Step 1: Create User model
- Files needed: src/models.py, src/config.py
- Verify: pytest tests/test_models.py

### Step 2: Add login/register routes
- Files needed: src/routes.py, src/models.py
- Verify: pytest tests/test_auth.py
```

Parsed with simple regex/string matching. Markdown chosen over JSON because:
- Models produce better reasoning when not constrained to JSON
- 34-38% fewer tokens than JSON
- JSON mode during reasoning degrades performance 10-15%

The planner also handles replanning: receives the original task, current plan, failed step, and error output. Outputs a revised plan.

**4. Executor**

An LLM call with a coding-focused system prompt. Receives one step + only the relevant file contents (identified by the planner's `files_needed` list). Outputs file edits and optional shell commands.

**Adaptive edit format based on file size:**

| File size | Format | Rationale |
|-----------|--------|-----------|
| New file | Full content | No choice |
| < 300 lines | Whole-file rewrite | Higher accuracy, tokens are free locally |
| 300+ lines | Search/replace blocks | Token-efficient for large files |

**Search/replace format** (Aider's battle-tested delimiters):

```
src/routes.py
<<<<<<< SEARCH
original code exactly as it appears
=======
replacement code
>>>>>>> REPLACE
```

**Matching cascade** for search/replace:
1. Exact string match
2. Whitespace-normalized match (strip leading whitespace, apply offset to replacement)
3. On failure: feed actual file contents back to executor with error message showing the real lines. No fuzzy matching — false positives are worse than retries.

**5. Verifier**

No LLM. Runs a configured list of shell commands (e.g., `["pytest", "ruff check .", "mypy ."]`). Returns structured results:

```python
VerificationResult(
    passed=True/False,
    details=[
        CommandResult(cmd="pytest", exit_code=0, stdout="...", stderr="..."),
        CommandResult(cmd="ruff check .", exit_code=1, stdout="...", stderr="..."),
    ]
)
```

Commands are configured per-project in SQLite. If none configured, verifier is a no-op (always passes).

**6. LLM Client**

Thin HTTP client. Two config fields: `base_url` and `model`. That's the entire abstraction.

```python
class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "local"):
        ...
    def chat(self, messages, temperature=0.7, max_tokens=4096, stream=False):
        # POST to {base_url}/chat/completions
```

The orchestrator creates two instances — one for planner, one for executor. In v1 both point at the same model. Splitting to different models or different backends is a config change, zero code change.

Supported backends (anything OpenAI-compatible):
- Ollama (`http://localhost:11434/v1`)
- TabbyAPI (`http://localhost:5000/v1`)
- LM Studio (`http://localhost:1234/v1`)
- vLLM, LocalAI, or any future backend

## Error Recovery

```
Executor produces edit
    │
    ├── Edit applies successfully
    │       │
    │       ▼
    │   Verifier runs ──── pass ──── next step
    │       │
    │       fail
    │       │
    │       ▼
    │   Executor retry with error output (attempt 2)
    │       │
    │       ├── pass ──── next step
    │       └── fail ──── escalate to planner
    │
    └── Edit fails (no match)
            │
            ▼
        Feed actual file contents to executor (attempt 2)
            │
            ├── applies ──── verifier
            └── fails again ──── escalate to planner
```

Escalation to planner: the planner receives the original task, current plan, which step failed, and the error output. It generates a revised plan, and the orchestrator restarts execution from the revised plan.

Maximum replan attempts: 3. After 3 replans, the orchestrator stops and reports failure. In interactive mode, it asks the user for guidance. In autonomous mode, it exits with status code 1 and logs the full history for review.

## Sandbox & Security

All file operations are scoped to the project root:

```python
def validate_path(path: str) -> Path:
    resolved = (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise SecurityError("Path escapes project root")
    return resolved
```

Shell commands run in subprocess with:
- `cwd` = project root
- Configurable timeout (default 60 seconds)
- Inherited environment with no modifications
- All commands logged to SQLite

**Tools available to the executor:**

| Tool | Purpose |
|------|---------|
| `read_file(path, start_line?, end_line?)` | Read file contents, optionally a line range |
| `write_file(path, content)` | Create or overwrite a file |
| `edit_file(path, search, replace)` | Search/replace within a file |
| `list_files(directory, pattern?)` | List files, optionally filtered by glob |
| `search_text(query, path_filter?)` | Grep-like search across project |
| `run_command(command)` | Run a shell command in sandbox |

## Data Model (SQLite)

Single database file per project at `{project_root}/.agent/agent.db`.

```sql
CREATE TABLE config (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- Keys: planner_base_url, planner_model, executor_base_url, executor_model,
--        verify_commands (JSON array), project_name, max_steps, timeout

CREATE TABLE conversations (
    id          TEXT PRIMARY KEY,
    started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode        TEXT CHECK(mode IN ('interactive', 'autonomous')),
    task        TEXT,
    status      TEXT CHECK(status IN ('active', 'completed', 'failed', 'aborted'))
);

CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    role        TEXT CHECK(role IN ('user', 'planner', 'executor', 'verifier', 'system')),
    content     TEXT,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    version     INTEGER DEFAULT 1,
    content     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE edits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    step_id     INTEGER,
    file_path   TEXT,
    edit_type   TEXT CHECK(edit_type IN ('create', 'rewrite', 'search_replace')),
    before      TEXT,
    after       TEXT,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

The `edits` table stores before/after snapshots of every file change, enabling undo and audit. The `plans` table stores every plan version (including replans) for debugging.

## Project Structure

```
coding-agent/
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── __main__.py          # Entry point, CLI argument parsing
│       ├── cli.py               # REPL loop, input handling, output rendering
│       ├── orchestrator.py      # Plan-execute-verify state machine
│       ├── planner.py           # Planner LLM calls + plan parsing
│       ├── executor.py          # Executor LLM calls + edit parsing
│       ├── verifier.py          # Command runner for verification
│       ├── llm_client.py        # HTTP client for OpenAI-compatible APIs
│       ├── sandbox.py           # Path validation, command sandboxing
│       ├── tools.py             # File tools (read, write, edit, list, search)
│       ├── db.py                # SQLite schema, queries, migrations
│       ├── parser.py            # Plan markdown parser, edit format parser
│       └── prompts/
│           ├── planner.md       # Planner system prompt
│           └── executor.md      # Executor system prompt
├── tests/
│   ├── test_orchestrator.py
│   ├── test_planner.py
│   ├── test_executor.py
│   ├── test_parser.py
│   ├── test_sandbox.py
│   └── test_tools.py
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-04-local-coding-agent-design.md
├── pyproject.toml
└── README.md
```

## Dependencies

Minimal:
- `httpx` — async HTTP client for LLM API calls
- `rich` — terminal formatting and streaming output
- Standard library only: `sqlite3`, `subprocess`, `pathlib`, `json`, `re`, `asyncio`, `argparse`

No LangChain, no smolagents, no heavy frameworks.

## Context Management

The orchestrator manages what context each component sees:

- **Planner receives**: task description + project summary (file tree, optionally key file snippets). Total target: < 4K tokens.
- **Executor receives**: one step description + only the files listed in `files_needed`. For files > 300 lines, only a relevant window (+-50 lines around area of interest). Total target: < 8K tokens per call.
- **Verifier receives**: nothing from the LLM. Just runs commands.

This keeps each LLM call focused and well within context limits, avoiding the quality degradation that occurs past 25K tokens.

## v1 Scope

**In scope:**
- Terminal REPL with interactive and autonomous modes
- Planner/executor split with separate system prompts
- Adaptive edit format (whole-file < 300 lines, search/replace >= 300 lines)
- Matching cascade with error feedback loop
- Sandboxed file and command tools
- Configurable verification commands
- SQLite storage for config, conversations, plans, edits
- Ollama as primary inference backend
- Streaming output
- Model pluggability via base_url + model config

**Out of scope (future phases):**
- ExLlamaV3/TabbyAPI backend optimization (Phase 2 — works today with nightly PyTorch, just not default)
- Dual-model hot-swap with different models for planner vs executor (Phase 2 — config already supports it)
- Web UI / chatbot accessible from phone/laptop (Phase 3 — reads from same SQLite)
- RAG / codebase indexing
- LSP integration
- MCP tools
- Git integration (auto-commit, branch management)

## Model Recommendations

| Model | VRAM (Q4) | Use case |
|-------|-----------|----------|
| Qwen3-14B | ~9 GB | v1 default — all-rounder with /think toggle |
| Phi-4-Reasoning-Plus 14B | ~9 GB | Future planner — best planning benchmarks at 14B |
| Qwen3-Coder-14B | ~9 GB | Future executor — coding-optimized |
| Qwen3-Coder-30B-A3B (MoE) | ~15-17 GB | Stretch — tight on 16GB, best local coding agent model |
| Devstral Small 2 (24B) | ~14 GB | Alternative executor — highest SWE-bench in class |
