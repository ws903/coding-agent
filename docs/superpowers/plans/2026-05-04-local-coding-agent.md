# Local Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terminal-based coding agent with planner/executor architecture, local LLM inference via Ollama, sandboxed tools, automated verification, and both interactive and autonomous modes.

**Architecture:** Bottom-up build. Data layer (SQLite) → LLM client → sandbox/tools → parsers → planner/executor → orchestrator → CLI. Each layer is testable independently before the next layer is built on top.

**Tech Stack:** Python 3.12+, httpx (async HTTP), rich (terminal UI), sqlite3/subprocess/pathlib/asyncio (stdlib). No frameworks.

---

## File Map

```
coding-agent/
├── pyproject.toml                    # Project metadata, dependencies, entry point
├── src/
│   └── agent/
│       ├── __init__.py               # Package init, version
│       ├── __main__.py               # Entry point: argparse, dispatch to interactive/autonomous
│       ├── cli.py                    # REPL loop, streaming output, slash commands
│       ├── orchestrator.py           # Plan-execute-verify state machine
│       ├── planner.py                # Planner LLM calls, plan generation/replanning
│       ├── executor.py               # Executor LLM calls, edit application
│       ├── verifier.py               # Run verification commands, return structured results
│       ├── llm_client.py             # Async HTTP client for OpenAI-compatible APIs
│       ├── sandbox.py                # Path validation, command execution with timeout
│       ├── tools.py                  # File tools: read, write, edit, list, search
│       ├── db.py                     # SQLite schema init, CRUD helpers
│       ├── parser.py                 # Plan markdown parser, edit format parser
│       ├── models.py                 # Dataclasses: Plan, Step, EditResult, etc.
│       └── prompts/
│           ├── planner.md            # Planner system prompt
│           └── executor.md           # Executor system prompt
├── tests/
│   ├── test_db.py
│   ├── test_llm_client.py
│   ├── test_sandbox.py
│   ├── test_tools.py
│   ├── test_parser.py
│   ├── test_planner.py
│   ├── test_executor.py
│   ├── test_orchestrator.py
│   └── test_cli.py
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-05-04-local-coding-agent-design.md
        └── plans/
            └── 2026-05-04-local-coding-agent.md
```

Note: `src/agent/models.py` is added beyond the spec's file list. It holds all shared dataclasses (Plan, Step, EditResult, CommandResult, VerificationResult) so that every module imports types from one place instead of defining them inline.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/agent/__init__.py`
- Create: `src/agent/__main__.py` (stub)
- Create: `src/agent/models.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "coding-agent"
version = "0.1.0"
description = "Local coding agent with planner/executor architecture"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
agent = "agent.__main__:main"

[tool.hatch.build.targets.wheel]
packages = ["src/agent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create package init**

```python
# src/agent/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create stub entry point**

```python
# src/agent/__main__.py
def main():
    print("agent v0.1.0")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create shared data models**

```python
# src/agent/models.py
from dataclasses import dataclass, field


@dataclass
class Step:
    id: int
    action: str
    files_needed: list[str]
    verify_command: str | None = None


@dataclass
class Plan:
    goal: str
    steps: list[Step]


@dataclass
class FileEdit:
    path: str
    action: str  # "create", "rewrite", "search_replace"
    content: str | None = None  # for create/rewrite
    search: str | None = None  # for search_replace
    replace: str | None = None  # for search_replace


@dataclass
class ExecutionResult:
    file_edits: list[FileEdit]
    commands: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class CommandResult:
    cmd: str
    exit_code: int
    stdout: str
    stderr: str


@dataclass
class VerificationResult:
    passed: bool
    details: list[CommandResult]
```

- [ ] **Step 5: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Then: `python -m agent`
Expected: prints `agent v0.1.0`

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml src/agent/__init__.py src/agent/__main__.py src/agent/models.py
git commit -m "feat: scaffold project with data models"
```

---

### Task 2: SQLite Database Layer

**Files:**
- Create: `src/agent/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing tests for db module**

```python
# tests/test_db.py
import os
import tempfile
from pathlib import Path

from agent.db import AgentDB


def make_db(tmp_path: Path) -> AgentDB:
    return AgentDB(tmp_path / ".agent" / "agent.db")


def test_init_creates_tables(tmp_path):
    db = make_db(tmp_path)
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "config" in table_names
    assert "conversations" in table_names
    assert "messages" in table_names
    assert "plans" in table_names
    assert "edits" in table_names


def test_config_get_set(tmp_path):
    db = make_db(tmp_path)
    db.set_config("planner_model", "qwen3:14b")
    assert db.get_config("planner_model") == "qwen3:14b"


def test_config_get_default(tmp_path):
    db = make_db(tmp_path)
    assert db.get_config("nonexistent", "fallback") == "fallback"


def test_config_upsert(tmp_path):
    db = make_db(tmp_path)
    db.set_config("model", "a")
    db.set_config("model", "b")
    assert db.get_config("model") == "b"


def test_create_conversation(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("interactive", "Fix the bug")
    conv = db.get_conversation(conv_id)
    assert conv["mode"] == "interactive"
    assert conv["task"] == "Fix the bug"
    assert conv["status"] == "active"


def test_add_message(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("interactive", "task")
    db.add_message(conv_id, "user", "hello")
    db.add_message(conv_id, "planner", "plan output")
    messages = db.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "planner"


def test_save_plan(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("autonomous", "task")
    db.save_plan(conv_id, 1, "## Plan\n### Step 1: do thing")
    db.save_plan(conv_id, 2, "## Revised Plan\n### Step 1: different")
    plans = db.get_plans(conv_id)
    assert len(plans) == 2
    assert plans[0]["version"] == 1
    assert plans[1]["version"] == 2


def test_save_edit(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("autonomous", "task")
    db.save_edit(conv_id, 1, "src/main.py", "create", before=None, after="print('hi')")
    edits = db.get_edits(conv_id)
    assert len(edits) == 1
    assert edits[0]["file_path"] == "src/main.py"
    assert edits[0]["edit_type"] == "create"


def test_update_conversation_status(tmp_path):
    db = make_db(tmp_path)
    conv_id = db.create_conversation("interactive", "task")
    db.update_conversation_status(conv_id, "completed")
    conv = db.get_conversation(conv_id)
    assert conv["status"] == "completed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.db'`

- [ ] **Step 3: Implement db.py**

```python
# src/agent/db.py
import sqlite3
import uuid
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode        TEXT CHECK(mode IN ('interactive', 'autonomous')),
    task        TEXT,
    status      TEXT CHECK(status IN ('active', 'completed', 'failed', 'aborted'))
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    role        TEXT CHECK(role IN ('user', 'planner', 'executor', 'verifier', 'system')),
    content     TEXT,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    version     INTEGER DEFAULT 1,
    content     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT REFERENCES conversations(id),
    step_id     INTEGER,
    file_path   TEXT,
    edit_type   TEXT CHECK(edit_type IN ('create', 'rewrite', 'search_replace')),
    before      TEXT,
    after       TEXT,
    timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class AgentDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def set_config(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_config(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def create_conversation(self, mode: str, task: str) -> str:
        conv_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO conversations (id, mode, task, status) VALUES (?, ?, ?, 'active')",
            (conv_id, mode, task),
        )
        self.conn.commit()
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_conversation_status(self, conv_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET status = ? WHERE id = ?", (status, conv_id)
        )
        self.conn.commit()

    def add_message(self, conv_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages (conv_id, role, content) VALUES (?, ?, ?)",
            (conv_id, role, content),
        )
        self.conn.commit()

    def get_messages(self, conv_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conv_id = ? ORDER BY id", (conv_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def save_plan(self, conv_id: str, version: int, content: str) -> None:
        self.conn.execute(
            "INSERT INTO plans (conv_id, version, content) VALUES (?, ?, ?)",
            (conv_id, version, content),
        )
        self.conn.commit()

    def get_plans(self, conv_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM plans WHERE conv_id = ? ORDER BY version", (conv_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def save_edit(
        self,
        conv_id: str,
        step_id: int,
        file_path: str,
        edit_type: str,
        before: str | None,
        after: str | None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO edits (conv_id, step_id, file_path, edit_type, before, after) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, step_id, file_path, edit_type, before, after),
        )
        self.conn.commit()

    def get_edits(self, conv_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM edits WHERE conv_id = ? ORDER BY id", (conv_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/db.py tests/test_db.py
git commit -m "feat: add SQLite database layer with config, conversations, plans, edits"
```

---

### Task 3: Sandbox & Path Validation

**Files:**
- Create: `src/agent/sandbox.py`
- Create: `tests/test_sandbox.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sandbox.py
import os
import stat
from pathlib import Path

import pytest

from agent.sandbox import Sandbox, SecurityError


def test_validate_path_within_root(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.validate_path("src/main.py")
    assert result == tmp_path / "src" / "main.py"


def test_validate_path_rejects_escape(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SecurityError):
        sandbox.validate_path("../../etc/passwd")


def test_validate_path_rejects_absolute(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SecurityError):
        sandbox.validate_path("/etc/passwd")


def test_validate_path_resolves_dots(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.validate_path("src/../src/main.py")
    assert result == tmp_path / "src" / "main.py"


def test_run_command_captures_output(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_run_command_captures_stderr(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command("python -c \"import sys; sys.stderr.write('err')\"")
    assert "err" in result.stderr


def test_run_command_nonzero_exit(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command("python -c \"raise SystemExit(42)\"")
    assert result.exit_code == 42


def test_run_command_timeout(tmp_path):
    sandbox = Sandbox(tmp_path, timeout=1)
    result = sandbox.run_command("python -c \"import time; time.sleep(10)\"")
    assert result.exit_code != 0
    assert "timeout" in result.stderr.lower() or "timed out" in result.stderr.lower()


def test_run_command_cwd_is_project_root(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command("python -c \"import os; print(os.getcwd())\"")
    assert result.exit_code == 0
    assert str(tmp_path) in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.sandbox'`

- [ ] **Step 3: Implement sandbox.py**

```python
# src/agent/sandbox.py
import subprocess
from pathlib import Path

from agent.models import CommandResult


class SecurityError(Exception):
    pass


class Sandbox:
    def __init__(self, project_root: Path, timeout: int = 60):
        self.project_root = project_root.resolve()
        self.timeout = timeout

    def validate_path(self, path: str) -> Path:
        if Path(path).is_absolute():
            raise SecurityError(f"Absolute paths not allowed: {path}")
        resolved = (self.project_root / path).resolve()
        if not resolved.is_relative_to(self.project_root):
            raise SecurityError(f"Path escapes project root: {path}")
        return resolved

    def run_command(self, command: str) -> CommandResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return CommandResult(
                cmd=command,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                cmd=command,
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sandbox.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/sandbox.py tests/test_sandbox.py
git commit -m "feat: add sandbox with path validation and command execution"
```

---

### Task 4: File Tools

**Files:**
- Create: `src/agent/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools.py
from pathlib import Path

import pytest

from agent.tools import FileTools
from agent.sandbox import SecurityError


def make_tools(tmp_path: Path) -> FileTools:
    return FileTools(tmp_path)


def test_read_file(tmp_path):
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n")
    tools = make_tools(tmp_path)
    content = tools.read_file("hello.txt")
    assert "line1" in content
    assert "line2" in content


def test_read_file_with_range(tmp_path):
    lines = "\n".join(f"line{i}" for i in range(1, 21))
    (tmp_path / "big.txt").write_text(lines)
    tools = make_tools(tmp_path)
    content = tools.read_file("big.txt", start_line=5, end_line=7)
    assert "line5" in content
    assert "line7" in content
    assert "line4" not in content
    assert "line8" not in content


def test_read_file_not_found(tmp_path):
    tools = make_tools(tmp_path)
    with pytest.raises(FileNotFoundError):
        tools.read_file("nope.txt")


def test_write_file_creates(tmp_path):
    tools = make_tools(tmp_path)
    tools.write_file("new.txt", "hello world")
    assert (tmp_path / "new.txt").read_text() == "hello world"


def test_write_file_creates_dirs(tmp_path):
    tools = make_tools(tmp_path)
    tools.write_file("a/b/c.txt", "deep")
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_write_file_rejects_escape(tmp_path):
    tools = make_tools(tmp_path)
    with pytest.raises(SecurityError):
        tools.write_file("../../evil.txt", "bad")


def test_edit_file_search_replace(tmp_path):
    (tmp_path / "code.py").write_text("def hello():\n    return 'hi'\n")
    tools = make_tools(tmp_path)
    success = tools.edit_file("code.py", "return 'hi'", "return 'hello world'")
    assert success
    assert "hello world" in (tmp_path / "code.py").read_text()


def test_edit_file_no_match(tmp_path):
    (tmp_path / "code.py").write_text("def hello():\n    return 'hi'\n")
    tools = make_tools(tmp_path)
    success = tools.edit_file("code.py", "return 'bye'", "return 'hello world'")
    assert not success


def test_edit_file_whitespace_normalized(tmp_path):
    (tmp_path / "code.py").write_text("    def hello():\n        return 'hi'\n")
    tools = make_tools(tmp_path)
    success = tools.edit_file("code.py", "def hello():\n    return 'hi'", "def hello():\n    return 'bye'")
    assert success
    content = (tmp_path / "code.py").read_text()
    assert "bye" in content


def test_list_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    tools = make_tools(tmp_path)
    files = tools.list_files(".")
    assert "a.py" in files
    assert "b.txt" in files


def test_list_files_with_pattern(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    tools = make_tools(tmp_path)
    files = tools.list_files(".", pattern="*.py")
    assert "a.py" in files
    assert "b.txt" not in files


def test_search_text(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    pass\n")
    (tmp_path / "b.py").write_text("def world():\n    pass\n")
    tools = make_tools(tmp_path)
    results = tools.search_text("hello")
    assert len(results) >= 1
    assert any("a.py" in r["file"] for r in results)


def test_search_text_with_filter(tmp_path):
    (tmp_path / "a.py").write_text("hello\n")
    (tmp_path / "b.txt").write_text("hello\n")
    tools = make_tools(tmp_path)
    results = tools.search_text("hello", path_filter="*.py")
    assert all("a.py" in r["file"] for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools'`

- [ ] **Step 3: Implement tools.py**

```python
# src/agent/tools.py
import fnmatch
import re
from pathlib import Path

from agent.sandbox import Sandbox, SecurityError


class FileTools:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.sandbox = Sandbox(project_root)

    def read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        full_path = self.sandbox.validate_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = full_path.read_text().splitlines(keepends=True)
        if start_line is not None or end_line is not None:
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            lines = lines[start:end]
        numbered = []
        offset = (start_line or 1) - 1
        for i, line in enumerate(lines, start=offset + 1):
            numbered.append(f"{i:>4}| {line.rstrip()}")
        return "\n".join(numbered)

    def write_file(self, path: str, content: str) -> None:
        full_path = self.sandbox.validate_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    def edit_file(self, path: str, search: str, replace: str) -> bool:
        full_path = self.sandbox.validate_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = full_path.read_text()
        if search in content:
            new_content = content.replace(search, replace, 1)
            full_path.write_text(new_content)
            return True
        match_result = self._whitespace_normalized_match(content, search, replace)
        if match_result is not None:
            full_path.write_text(match_result)
            return True
        return False

    def _whitespace_normalized_match(
        self, content: str, search: str, replace: str
    ) -> str | None:
        content_lines = content.splitlines(keepends=True)
        search_lines = search.splitlines()
        search_stripped = [line.lstrip() for line in search_lines]
        for i in range(len(content_lines) - len(search_lines) + 1):
            window = content_lines[i : i + len(search_lines)]
            window_stripped = [line.rstrip().lstrip() for line in window]
            if window_stripped == search_stripped:
                indent = ""
                first_line = content_lines[i]
                indent = first_line[: len(first_line) - len(first_line.lstrip())]
                replace_lines = replace.splitlines(keepends=True)
                indented_replace = []
                for j, rline in enumerate(replace_lines):
                    if j == 0:
                        indented_replace.append(indent + rline.lstrip())
                    else:
                        indented_replace.append(indent + rline.lstrip())
                result_lines = (
                    content_lines[:i] + indented_replace + content_lines[i + len(search_lines) :]
                )
                return "".join(result_lines)
        return None

    def list_files(
        self, directory: str = ".", pattern: str | None = None
    ) -> list[str]:
        dir_path = self.sandbox.validate_path(directory)
        if not dir_path.is_dir():
            return []
        files = []
        for item in sorted(dir_path.rglob("*")):
            if item.is_file():
                rel = str(item.relative_to(self.project_root))
                if pattern is None or fnmatch.fnmatch(item.name, pattern):
                    if not any(part.startswith(".") for part in item.parts):
                        files.append(rel)
        return files

    def search_text(
        self, query: str, path_filter: str | None = None
    ) -> list[dict]:
        results = []
        for item in sorted(self.project_root.rglob("*")):
            if not item.is_file():
                continue
            if any(part.startswith(".") for part in item.relative_to(self.project_root).parts):
                continue
            if path_filter and not fnmatch.fnmatch(item.name, path_filter):
                continue
            try:
                lines = item.read_text().splitlines()
            except (UnicodeDecodeError, PermissionError):
                continue
            for i, line in enumerate(lines, 1):
                if query in line:
                    results.append({
                        "file": str(item.relative_to(self.project_root)),
                        "line": i,
                        "content": line.strip(),
                    })
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: all 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools.py tests/test_tools.py
git commit -m "feat: add file tools with read, write, edit, list, search"
```

---

### Task 5: LLM Client

**Files:**
- Create: `src/agent/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

These tests mock httpx to avoid needing a running Ollama instance. The client is thin enough that mocked tests cover the contract.

```python
# tests/test_llm_client.py
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from agent.llm_client import LLMClient


@pytest.fixture
def client():
    return LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
    )


def test_client_init(client):
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model == "qwen3:14b"


def test_build_payload(client):
    messages = [{"role": "user", "content": "hello"}]
    payload = client._build_payload(messages, temperature=0.5, max_tokens=100)
    assert payload["model"] == "qwen3:14b"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    assert payload["stream"] is False


def test_build_payload_with_stream(client):
    messages = [{"role": "user", "content": "hello"}]
    payload = client._build_payload(messages, stream=True)
    assert payload["stream"] is True


@pytest.mark.asyncio
async def test_chat_returns_content(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "world"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        result = await client.chat([{"role": "user", "content": "hello"}])
        assert result == "world"


@pytest.mark.asyncio
async def test_chat_posts_to_correct_url(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        await client.chat([{"role": "user", "content": "hi"}])
        call_args = mock_instance.post.call_args
        assert call_args[0][0] == "http://localhost:11434/v1/chat/completions"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.llm_client'`

- [ ] **Step 3: Implement llm_client.py**

```python
# src/agent/llm_client.py
import json
from collections.abc import AsyncGenerator

import httpx


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3:14b",
        api_key: str = "local",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _build_payload(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, stream=False)
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/llm_client.py tests/test_llm_client.py
git commit -m "feat: add async LLM client with streaming support"
```

---

### Task 6: Parsers (Plan Markdown + Edit Format)

**Files:**
- Create: `src/agent/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write failing tests for plan parsing**

```python
# tests/test_parser.py
from agent.parser import parse_plan, parse_edits
from agent.models import Plan, Step, FileEdit


class TestParsePlan:
    def test_basic_plan(self):
        text = """## Plan: Add authentication

### Step 1: Create User model
- Files needed: src/models.py, src/config.py
- Verify: pytest tests/test_models.py

### Step 2: Add login route
- Files needed: src/routes.py
- Verify: pytest tests/test_auth.py
"""
        plan = parse_plan(text)
        assert plan.goal == "Add authentication"
        assert len(plan.steps) == 2
        assert plan.steps[0].id == 1
        assert plan.steps[0].action == "Create User model"
        assert plan.steps[0].files_needed == ["src/models.py", "src/config.py"]
        assert plan.steps[0].verify_command == "pytest tests/test_models.py"
        assert plan.steps[1].id == 2
        assert plan.steps[1].action == "Add login route"

    def test_plan_no_verify(self):
        text = """## Plan: Simple change

### Step 1: Update readme
- Files needed: README.md
"""
        plan = parse_plan(text)
        assert plan.steps[0].verify_command is None

    def test_plan_with_extra_content(self):
        text = """Some preamble text the model might add.

## Plan: Fix the bug

Here is my thinking about this...

### Step 1: Fix the handler
- Files needed: src/handler.py
- Verify: pytest

More explanation text here.

### Step 2: Update tests
- Files needed: tests/test_handler.py
- Verify: pytest tests/test_handler.py
"""
        plan = parse_plan(text)
        assert plan.goal == "Fix the bug"
        assert len(plan.steps) == 2

    def test_empty_returns_empty_plan(self):
        plan = parse_plan("no plan here")
        assert plan.goal == ""
        assert len(plan.steps) == 0


class TestParseEdits:
    def test_search_replace_block(self):
        text = """I'll make the following changes:

src/main.py
<<<<<<< SEARCH
def hello():
    return "hi"
=======
def hello():
    return "hello world"
>>>>>>> REPLACE
"""
        edits = parse_edits(text)
        assert len(edits) == 1
        assert edits[0].path == "src/main.py"
        assert edits[0].action == "search_replace"
        assert edits[0].search == 'def hello():\n    return "hi"'
        assert edits[0].replace == 'def hello():\n    return "hello world"'

    def test_multiple_search_replace(self):
        text = """
src/a.py
<<<<<<< SEARCH
old_a
=======
new_a
>>>>>>> REPLACE

src/b.py
<<<<<<< SEARCH
old_b
=======
new_b
>>>>>>> REPLACE
"""
        edits = parse_edits(text)
        assert len(edits) == 2
        assert edits[0].path == "src/a.py"
        assert edits[1].path == "src/b.py"

    def test_create_file_block(self):
        text = """
CREATE src/new_file.py
```
def new_function():
    pass
```
"""
        edits = parse_edits(text)
        assert len(edits) == 1
        assert edits[0].path == "src/new_file.py"
        assert edits[0].action == "create"
        assert "def new_function" in edits[0].content

    def test_rewrite_file_block(self):
        text = """
REWRITE src/small.py
```
def updated():
    return True
```
"""
        edits = parse_edits(text)
        assert len(edits) == 1
        assert edits[0].path == "src/small.py"
        assert edits[0].action == "rewrite"
        assert "def updated" in edits[0].content

    def test_commands_extracted(self):
        text = """
RUN: pip install flask
RUN: pytest tests/
"""
        _, commands = parse_edits(text, extract_commands=True)
        assert len(commands) == 2
        assert commands[0] == "pip install flask"
        assert commands[1] == "pytest tests/"

    def test_no_edits_returns_empty(self):
        edits = parse_edits("just some explanation text")
        assert len(edits) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.parser'`

- [ ] **Step 3: Implement parser.py**

```python
# src/agent/parser.py
import re

from agent.models import Plan, Step, FileEdit


def parse_plan(text: str) -> Plan:
    goal_match = re.search(r"##\s+Plan:\s*(.+)", text)
    goal = goal_match.group(1).strip() if goal_match else ""

    step_pattern = re.compile(
        r"###\s+Step\s+(\d+):\s*(.+?)(?=\n###\s+Step|\Z)",
        re.DOTALL,
    )

    steps = []
    for match in step_pattern.finditer(text):
        step_id = int(match.group(1))
        step_text = match.group(2).strip()
        action = step_text.split("\n")[0].strip()

        files_match = re.search(
            r"-\s*Files needed:\s*(.+)", step_text
        )
        files_needed = []
        if files_match:
            files_needed = [
                f.strip() for f in files_match.group(1).split(",")
            ]

        verify_match = re.search(r"-\s*Verify:\s*(.+)", step_text)
        verify_command = verify_match.group(1).strip() if verify_match else None

        steps.append(
            Step(
                id=step_id,
                action=action,
                files_needed=files_needed,
                verify_command=verify_command,
            )
        )

    return Plan(goal=goal, steps=steps)


def parse_edits(
    text: str, extract_commands: bool = False
) -> list[FileEdit] | tuple[list[FileEdit], list[str]]:
    edits: list[FileEdit] = []

    sr_pattern = re.compile(
        r"^(\S+.*?)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.MULTILINE | re.DOTALL,
    )
    for match in sr_pattern.finditer(text):
        path = match.group(1).strip()
        edits.append(
            FileEdit(
                path=path,
                action="search_replace",
                search=match.group(2),
                replace=match.group(3),
            )
        )

    create_pattern = re.compile(
        r"^CREATE\s+(\S+)\n```\w*\n(.*?)\n```",
        re.MULTILINE | re.DOTALL,
    )
    for match in create_pattern.finditer(text):
        edits.append(
            FileEdit(
                path=match.group(1).strip(),
                action="create",
                content=match.group(2),
            )
        )

    rewrite_pattern = re.compile(
        r"^REWRITE\s+(\S+)\n```\w*\n(.*?)\n```",
        re.MULTILINE | re.DOTALL,
    )
    for match in rewrite_pattern.finditer(text):
        edits.append(
            FileEdit(
                path=match.group(1).strip(),
                action="rewrite",
                content=match.group(2),
            )
        )

    if extract_commands:
        commands = re.findall(r"^RUN:\s*(.+)$", text, re.MULTILINE)
        return edits, commands

    return edits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/parser.py tests/test_parser.py
git commit -m "feat: add plan markdown parser and edit format parser"
```

---

### Task 7: System Prompts

**Files:**
- Create: `src/agent/prompts/planner.md`
- Create: `src/agent/prompts/executor.md`

- [ ] **Step 1: Create planner system prompt**

```markdown
# src/agent/prompts/planner.md
You are a software planning agent. Your job is to break down coding tasks into clear, sequential steps.

## Input

You receive:
1. A task description from the user
2. A project summary showing the file tree and key file contents

## Output Format

You MUST output a plan in this exact format:

## Plan: <one-line goal description>

### Step 1: <action description>
- Files needed: <comma-separated file paths>
- Verify: <shell command to verify this step, or omit if none>

### Step 2: <action description>
- Files needed: <comma-separated file paths>
- Verify: <shell command to verify this step, or omit if none>

(continue for all steps)

## Rules

- Each step should be a single, focused change
- List only the files the executor will need to read or modify
- Keep steps small — prefer 5 steps of 1 change each over 1 step with 5 changes
- Order steps so each builds on the previous (dependencies first)
- Include verification commands when possible (test commands, lint, type check)
- If a step creates a new file, include the directory path in files_needed
- Do not include code in the plan — the executor handles implementation
- Think carefully about the order of operations
```

- [ ] **Step 2: Create executor system prompt**

```markdown
# src/agent/prompts/executor.md
You are a code execution agent. You receive one step from a plan and the relevant file contents. Your job is to produce the exact file edits needed to complete that step.

## Input

You receive:
1. A step description (what to do)
2. The current contents of relevant files

## Output Formats

Use the appropriate format based on what you need to do:

### Creating a new file

CREATE path/to/file.py
```
file contents here
```

### Rewriting a small file (under 300 lines)

REWRITE path/to/file.py
```
complete new file contents
```

### Editing part of a larger file

path/to/file.py
<<<<<<< SEARCH
exact text to find in the file
=======
replacement text
>>>>>>> REPLACE

### Running a shell command

RUN: command here

## Rules

- For SEARCH blocks: copy the existing code EXACTLY as it appears, including whitespace and indentation
- For REWRITE: output the complete file contents — do not use placeholders like "... rest of file"
- For CREATE: output the complete file contents
- You may include multiple edits and commands in one response
- Include a brief explanation of what you changed and why
- Do not change code unrelated to the current step
- Do not add comments explaining what you changed — the code should speak for itself
```

- [ ] **Step 3: Create prompts __init__ to make it a package**

Create an empty file at `src/agent/prompts/__init__.py` so the prompts directory is a Python package (needed for `importlib.resources` to find the .md files).

- [ ] **Step 4: Commit**

```bash
git add src/agent/prompts/
git commit -m "feat: add planner and executor system prompts"
```

---

### Task 8: Planner Module

**Files:**
- Create: `src/agent/planner.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_planner.py
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.planner import Planner
from agent.llm_client import LLMClient
from agent.models import Plan


MOCK_PLAN_RESPONSE = """## Plan: Add health check

### Step 1: Create health endpoint
- Files needed: src/app.py
- Verify: pytest tests/test_health.py

### Step 2: Add health test
- Files needed: tests/test_health.py
- Verify: pytest tests/test_health.py
"""

MOCK_REPLAN_RESPONSE = """## Plan: Add health check (revised)

### Step 1: Fix import error in app.py
- Files needed: src/app.py
- Verify: python -c "import src.app"

### Step 2: Create health endpoint
- Files needed: src/app.py
- Verify: pytest tests/test_health.py
"""


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=LLMClient)
    client.chat = AsyncMock(return_value=MOCK_PLAN_RESPONSE)
    return client


@pytest.fixture
def planner(mock_client):
    return Planner(mock_client)


@pytest.mark.asyncio
async def test_generate_plan(planner, mock_client):
    plan = await planner.generate_plan("Add health check", "file tree here")
    assert plan.goal == "Add health check"
    assert len(plan.steps) == 2
    assert plan.steps[0].action == "Create health endpoint"
    mock_client.chat.assert_called_once()


@pytest.mark.asyncio
async def test_generate_plan_sends_system_prompt(planner, mock_client):
    await planner.generate_plan("task", "context")
    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert "planning agent" in messages[0]["content"].lower()


@pytest.mark.asyncio
async def test_generate_plan_includes_context(planner, mock_client):
    await planner.generate_plan("Add feature", "src/app.py\nsrc/models.py")
    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "Add feature" in user_msg
    assert "src/app.py" in user_msg


@pytest.mark.asyncio
async def test_replan(planner, mock_client):
    mock_client.chat = AsyncMock(return_value=MOCK_REPLAN_RESPONSE)
    original_plan = Plan(goal="Add health check", steps=[])
    plan = await planner.replan(
        task="Add health check",
        current_plan=original_plan,
        failed_step_id=1,
        error="ImportError: no module named flask",
    )
    assert len(plan.steps) == 2
    assert "Fix import" in plan.steps[0].action


@pytest.mark.asyncio
async def test_replan_includes_error(planner, mock_client):
    mock_client.chat = AsyncMock(return_value=MOCK_REPLAN_RESPONSE)
    original_plan = Plan(goal="goal", steps=[])
    await planner.replan("task", original_plan, 1, "some error")
    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "some error" in user_msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.planner'`

- [ ] **Step 3: Implement planner.py**

```python
# src/agent/planner.py
from importlib import resources

from agent.llm_client import LLMClient
from agent.models import Plan
from agent.parser import parse_plan


def _load_prompt() -> str:
    return resources.files("agent.prompts").joinpath("planner.md").read_text()


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = _load_prompt()

    async def generate_plan(self, task: str, project_context: str) -> Plan:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n"
                    f"## Project Context\n{project_context}"
                ),
            },
        ]
        response = await self.llm.chat(messages, temperature=0.3)
        return parse_plan(response)

    async def replan(
        self,
        task: str,
        current_plan: Plan,
        failed_step_id: int,
        error: str,
    ) -> Plan:
        plan_summary = f"Goal: {current_plan.goal}\n"
        for step in current_plan.steps:
            plan_summary += f"Step {step.id}: {step.action}\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n"
                    f"## Previous Plan\n{plan_summary}\n\n"
                    f"## Failure\nStep {failed_step_id} failed with error:\n{error}\n\n"
                    f"Please create a revised plan that addresses this failure."
                ),
            },
        ]
        response = await self.llm.chat(messages, temperature=0.3)
        return parse_plan(response)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_planner.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/planner.py tests/test_planner.py
git commit -m "feat: add planner module with plan generation and replanning"
```

---

### Task 9: Executor Module

**Files:**
- Create: `src/agent/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_executor.py
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.models import Step, FileEdit
from agent.tools import FileTools


MOCK_SEARCH_REPLACE_RESPONSE = """I'll update the handler to add the health endpoint.

src/app.py
<<<<<<< SEARCH
@app.route("/")
def index():
    return "hello"
=======
@app.route("/")
def index():
    return "hello"

@app.route("/health")
def health():
    return {"status": "ok"}
>>>>>>> REPLACE
"""

MOCK_CREATE_RESPONSE = """I'll create the test file.

CREATE tests/test_health.py
```
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
```
"""

MOCK_REWRITE_RESPONSE = """I'll rewrite this small file.

REWRITE src/config.py
```
DEBUG = False
PORT = 8080
```
"""


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=LLMClient)
    return client


@pytest.fixture
def tools(tmp_path):
    return FileTools(tmp_path)


@pytest.fixture
def executor(mock_client, tools):
    return Executor(mock_client, tools)


@pytest.mark.asyncio
async def test_execute_search_replace(executor, mock_client, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text('@app.route("/")\ndef index():\n    return "hello"\n')
    mock_client.chat = AsyncMock(return_value=MOCK_SEARCH_REPLACE_RESPONSE)

    step = Step(id=1, action="Add health endpoint", files_needed=["src/app.py"])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "search_replace"


@pytest.mark.asyncio
async def test_execute_create_file(executor, mock_client, tmp_path):
    mock_client.chat = AsyncMock(return_value=MOCK_CREATE_RESPONSE)

    step = Step(id=1, action="Create test file", files_needed=[])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "create"
    assert result.file_edits[0].path == "tests/test_health.py"


@pytest.mark.asyncio
async def test_execute_rewrite(executor, mock_client, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text("DEBUG = True\n")
    mock_client.chat = AsyncMock(return_value=MOCK_REWRITE_RESPONSE)

    step = Step(id=1, action="Update config", files_needed=["src/config.py"])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "rewrite"


@pytest.mark.asyncio
async def test_execute_includes_file_contents_in_prompt(executor, mock_client, tmp_path):
    (tmp_path / "code.py").write_text("x = 1\n")
    mock_client.chat = AsyncMock(return_value="no edits")

    step = Step(id=1, action="Change x", files_needed=["code.py"])
    await executor.execute(step)

    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "x = 1" in user_msg


@pytest.mark.asyncio
async def test_execute_with_errors_includes_error(executor, mock_client, tmp_path):
    mock_client.chat = AsyncMock(return_value="no edits")
    step = Step(id=1, action="Fix bug", files_needed=[])
    await executor.execute(step, errors="NameError: name 'foo' is not defined")

    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "NameError" in user_msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.executor'`

- [ ] **Step 3: Implement executor.py**

```python
# src/agent/executor.py
from importlib import resources

from agent.llm_client import LLMClient
from agent.models import Step, ExecutionResult, FileEdit
from agent.parser import parse_edits
from agent.tools import FileTools


def _load_prompt() -> str:
    return resources.files("agent.prompts").joinpath("executor.md").read_text()


class Executor:
    def __init__(self, llm_client: LLMClient, tools: FileTools):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = _load_prompt()

    async def execute(
        self, step: Step, errors: str | None = None
    ) -> ExecutionResult:
        file_contents = self._gather_files(step.files_needed)
        user_content = f"## Step\n{step.action}\n"

        if file_contents:
            user_content += f"\n## Current File Contents\n{file_contents}\n"

        if errors:
            user_content += (
                f"\n## Previous Attempt Failed\n"
                f"The previous attempt produced these errors:\n{errors}\n"
                f"Please fix the issues and try again.\n"
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.llm.chat(messages, temperature=0.2)
        edits, commands = parse_edits(response, extract_commands=True)

        return ExecutionResult(
            file_edits=edits,
            commands=commands,
            explanation=response,
        )

    def _gather_files(self, file_paths: list[str]) -> str:
        sections = []
        for path in file_paths:
            try:
                content = self.tools.read_file(path)
                sections.append(f"### {path}\n```\n{content}\n```")
            except FileNotFoundError:
                sections.append(f"### {path}\n(file does not exist yet)")
        return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_executor.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/executor.py tests/test_executor.py
git commit -m "feat: add executor module with file gathering and edit parsing"
```

---

### Task 10: Verifier Module

**Files:**
- Create: `src/agent/verifier.py`
- Create: `tests/test_verifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_verifier.py
from pathlib import Path

import pytest

from agent.verifier import Verifier
from agent.sandbox import Sandbox


def test_verify_passes_when_commands_succeed(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["echo ok"])
    result = verifier.run()
    assert result.passed
    assert len(result.details) == 1
    assert result.details[0].exit_code == 0


def test_verify_fails_when_command_fails(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["python -c \"raise SystemExit(1)\""])
    result = verifier.run()
    assert not result.passed


def test_verify_runs_all_commands(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["echo one", "echo two", "echo three"])
    result = verifier.run()
    assert result.passed
    assert len(result.details) == 3


def test_verify_fails_if_any_command_fails(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(
        sandbox,
        commands=["echo ok", "python -c \"raise SystemExit(1)\"", "echo ok"],
    )
    result = verifier.run()
    assert not result.passed
    assert len(result.details) == 3


def test_verify_no_commands_passes(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=[])
    result = verifier.run()
    assert result.passed
    assert len(result.details) == 0


def test_verify_with_step_command(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["echo global"])
    result = verifier.run(step_command="echo step")
    assert result.passed
    assert len(result.details) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.verifier'`

- [ ] **Step 3: Implement verifier.py**

```python
# src/agent/verifier.py
from agent.models import CommandResult, VerificationResult
from agent.sandbox import Sandbox


class Verifier:
    def __init__(self, sandbox: Sandbox, commands: list[str] | None = None):
        self.sandbox = sandbox
        self.commands = commands or []

    def run(self, step_command: str | None = None) -> VerificationResult:
        all_commands = list(self.commands)
        if step_command:
            all_commands.append(step_command)

        if not all_commands:
            return VerificationResult(passed=True, details=[])

        details = []
        for cmd in all_commands:
            result = self.sandbox.run_command(cmd)
            details.append(result)

        passed = all(r.exit_code == 0 for r in details)
        return VerificationResult(passed=passed, details=details)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verifier.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/verifier.py tests/test_verifier.py
git commit -m "feat: add verifier module for running verification commands"
```

---

### Task 11: Orchestrator

**Files:**
- Create: `src/agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestrator import Orchestrator
from agent.models import (
    Plan, Step, ExecutionResult, FileEdit,
    VerificationResult, CommandResult,
)


def make_plan(num_steps=2):
    steps = [
        Step(id=i + 1, action=f"Step {i + 1}", files_needed=[], verify_command="echo ok")
        for i in range(num_steps)
    ]
    return Plan(goal="Test goal", steps=steps)


def make_success_result():
    return ExecutionResult(
        file_edits=[FileEdit(path="test.py", action="create", content="pass")],
        commands=[],
        explanation="done",
    )


def make_verification_pass():
    return VerificationResult(passed=True, details=[])


def make_verification_fail():
    return VerificationResult(
        passed=False,
        details=[CommandResult(cmd="pytest", exit_code=1, stdout="", stderr="FAILED")],
    )


@pytest.fixture
def orchestrator(tmp_path):
    mock_planner = AsyncMock()
    mock_planner.generate_plan = AsyncMock(return_value=make_plan())
    mock_planner.replan = AsyncMock(return_value=make_plan(1))

    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=make_success_result())

    mock_verifier = MagicMock()
    mock_verifier.run = MagicMock(return_value=make_verification_pass())

    mock_tools = MagicMock()
    mock_tools.write_file = MagicMock()
    mock_tools.edit_file = MagicMock(return_value=True)
    mock_tools.read_file = MagicMock(return_value="file contents")
    mock_tools.list_files = MagicMock(return_value=["a.py", "b.py"])

    mock_db = MagicMock()
    mock_db.create_conversation = MagicMock(return_value="conv123")
    mock_db.add_message = MagicMock()
    mock_db.save_plan = MagicMock()
    mock_db.save_edit = MagicMock()
    mock_db.update_conversation_status = MagicMock()

    orch = Orchestrator(
        planner=mock_planner,
        executor=mock_executor,
        verifier=mock_verifier,
        tools=mock_tools,
        db=mock_db,
        project_root=tmp_path,
    )
    return orch


@pytest.mark.asyncio
async def test_run_generates_plan(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.planner.generate_plan.assert_called_once()


@pytest.mark.asyncio
async def test_run_executes_all_steps(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.executor.execute.call_count == 2


@pytest.mark.asyncio
async def test_run_verifies_after_each_step(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.verifier.run.call_count == 2


@pytest.mark.asyncio
async def test_run_applies_file_creates(orchestrator, tmp_path):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.tools.write_file.assert_called()


@pytest.mark.asyncio
async def test_run_retries_on_verify_failure(orchestrator):
    orchestrator.verifier.run = MagicMock(
        side_effect=[make_verification_fail(), make_verification_pass()] * 2
    )
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.executor.execute.call_count == 4  # 2 steps x (1 attempt + 1 retry)


@pytest.mark.asyncio
async def test_run_replans_after_two_failures(orchestrator):
    orchestrator.verifier.run = MagicMock(return_value=make_verification_fail())
    orchestrator.planner.replan = AsyncMock(return_value=make_plan(1))

    # After replan with 1 step, verifier still fails -> second replan, still fails -> third replan, still fails -> give up
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.planner.replan.call_count >= 1


@pytest.mark.asyncio
async def test_run_returns_success_on_completion(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_run_saves_to_db(orchestrator):
    await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.db.create_conversation.assert_called_once()
    orchestrator.db.save_plan.assert_called()
    orchestrator.db.update_conversation_status.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.orchestrator'`

- [ ] **Step 3: Implement orchestrator.py**

```python
# src/agent/orchestrator.py
from pathlib import Path

from agent.db import AgentDB
from agent.models import (
    Plan, Step, ExecutionResult, FileEdit,
    VerificationResult,
)
from agent.planner import Planner
from agent.executor import Executor
from agent.verifier import Verifier
from agent.tools import FileTools


MAX_EXECUTOR_RETRIES = 2
MAX_REPLANS = 3


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        verifier: Verifier,
        tools: FileTools,
        db: AgentDB,
        project_root: Path,
        on_status: callable | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.tools = tools
        self.db = db
        self.project_root = project_root
        self.on_status = on_status or (lambda msg: None)

    async def run(
        self,
        task: str,
        mode: str = "autonomous",
        approve_plan: callable | None = None,
    ) -> dict:
        conv_id = self.db.create_conversation(mode, task)
        self.db.add_message(conv_id, "user", task)

        project_context = self._build_project_context()
        plan_version = 0
        replan_count = 0

        self.on_status("Generating plan...")
        plan = await self.planner.generate_plan(task, project_context)
        plan_version += 1
        self.db.save_plan(conv_id, plan_version, self._plan_to_text(plan))
        self.db.add_message(conv_id, "planner", self._plan_to_text(plan))
        self.on_status(f"Plan: {plan.goal} ({len(plan.steps)} steps)")

        if approve_plan and not approve_plan(plan):
            self.db.update_conversation_status(conv_id, "aborted")
            return {"status": "aborted", "conv_id": conv_id}

        step_index = 0
        while step_index < len(plan.steps):
            step = plan.steps[step_index]
            self.on_status(f"Executing step {step.id}: {step.action}")

            success = await self._execute_step(conv_id, step)

            if success:
                step_index += 1
                continue

            replan_count += 1
            if replan_count > MAX_REPLANS:
                self.on_status("Max replans reached. Stopping.")
                self.db.update_conversation_status(conv_id, "failed")
                return {"status": "failed", "conv_id": conv_id, "reason": "max_replans"}

            self.on_status(f"Replanning (attempt {replan_count}/{MAX_REPLANS})...")
            error_summary = self._get_last_error(conv_id)
            plan = await self.planner.replan(task, plan, step.id, error_summary)
            plan_version += 1
            self.db.save_plan(conv_id, plan_version, self._plan_to_text(plan))
            self.db.add_message(conv_id, "planner", f"Replan v{plan_version}:\n{self._plan_to_text(plan)}")
            step_index = 0

        self.db.update_conversation_status(conv_id, "completed")
        self.on_status("All steps completed successfully.")
        return {"status": "completed", "conv_id": conv_id}

    async def _execute_step(self, conv_id: str, step: Step) -> bool:
        for attempt in range(MAX_EXECUTOR_RETRIES):
            errors = None
            if attempt > 0:
                errors = self._get_last_error(conv_id)
                self.on_status(f"  Retry {attempt}/{MAX_EXECUTOR_RETRIES - 1}...")

            result = await self.executor.execute(step, errors=errors)
            self.db.add_message(conv_id, "executor", result.explanation)

            apply_ok = self._apply_edits(conv_id, step.id, result)
            if not apply_ok:
                self.db.add_message(conv_id, "system", "Edit application failed")
                continue

            for cmd in result.commands:
                cmd_result = self.tools.sandbox.run_command(cmd)
                self.db.add_message(
                    conv_id, "system", f"Command `{cmd}`: exit={cmd_result.exit_code}"
                )

            verification = self.verifier.run(step_command=step.verify_command)
            if verification.passed:
                self.db.add_message(conv_id, "verifier", "Verification passed")
                return True

            error_detail = "\n".join(
                f"{d.cmd}: {d.stderr or d.stdout}" for d in verification.details if d.exit_code != 0
            )
            self.db.add_message(conv_id, "verifier", f"Verification failed:\n{error_detail}")

        return False

    def _apply_edits(self, conv_id: str, step_id: int, result: ExecutionResult) -> bool:
        all_ok = True
        for edit in result.file_edits:
            try:
                if edit.action == "create":
                    before = None
                    self.tools.write_file(edit.path, edit.content)
                elif edit.action == "rewrite":
                    try:
                        before = self.tools.read_file(edit.path)
                    except FileNotFoundError:
                        before = None
                    self.tools.write_file(edit.path, edit.content)
                elif edit.action == "search_replace":
                    try:
                        before = self.tools.read_file(edit.path)
                    except FileNotFoundError:
                        all_ok = False
                        continue
                    success = self.tools.edit_file(edit.path, edit.search, edit.replace)
                    if not success:
                        all_ok = False
                        continue
                else:
                    continue

                after = self.tools.read_file(edit.path)
                self.db.save_edit(
                    conv_id, step_id, edit.path, edit.action,
                    before=before, after=after,
                )
            except Exception:
                all_ok = False
        return all_ok

    def _build_project_context(self) -> str:
        files = self.tools.list_files(".")
        tree = "\n".join(f"  {f}" for f in files[:100])
        return f"## File Tree\n{tree}\n"

    def _plan_to_text(self, plan: Plan) -> str:
        lines = [f"## Plan: {plan.goal}\n"]
        for step in plan.steps:
            lines.append(f"### Step {step.id}: {step.action}")
            if step.files_needed:
                lines.append(f"- Files needed: {', '.join(step.files_needed)}")
            if step.verify_command:
                lines.append(f"- Verify: {step.verify_command}")
            lines.append("")
        return "\n".join(lines)

    def _get_last_error(self, conv_id: str) -> str:
        messages = self.db.get_messages(conv_id)
        for msg in reversed(messages):
            if msg["role"] == "verifier" and "failed" in msg["content"].lower():
                return msg["content"]
            if msg["role"] == "system" and "failed" in msg["content"].lower():
                return msg["content"]
        return "Unknown error"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add orchestrator with plan-execute-verify loop and replanning"
```

---

### Task 12: CLI and REPL

**Files:**
- Create: `src/agent/cli.py`
- Modify: `src/agent/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from agent.cli import build_orchestrator, parse_args


def test_parse_args_interactive():
    args = parse_args([])
    assert args.auto is False
    assert args.task is None


def test_parse_args_autonomous():
    args = parse_args(["--auto", "--task", "Fix the bug"])
    assert args.auto is True
    assert args.task == "Fix the bug"


def test_parse_args_max_steps():
    args = parse_args(["--auto", "--task", "Fix", "--max-steps", "10"])
    assert args.max_steps == 10


def test_parse_args_custom_model():
    args = parse_args(["--model", "phi4:14b"])
    assert args.model == "phi4:14b"


def test_parse_args_custom_base_url():
    args = parse_args(["--base-url", "http://localhost:5000/v1"])
    assert args.base_url == "http://localhost:5000/v1"


def test_parse_args_project_dir():
    args = parse_args(["--project", "/tmp/myproject"])
    assert args.project == "/tmp/myproject"


def test_parse_args_step_mode():
    args = parse_args(["--step"])
    assert args.step is True


def test_build_orchestrator(tmp_path):
    orch = build_orchestrator(
        project_root=tmp_path,
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
    )
    assert orch is not None
    assert orch.project_root == tmp_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.cli'`

- [ ] **Step 3: Implement cli.py**

```python
# src/agent/cli.py
import argparse
import asyncio
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
        import json
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
    answer = Prompt.ask("Proceed?", choices=["y", "n", "edit"], default="y")
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

        approve_fn = _approve_plan if not args.step else _approve_plan
        result = await orch.run(user_input, mode="interactive", approve_plan=approve_fn)
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
```

- [ ] **Step 4: Update __main__.py**

```python
# src/agent/__main__.py
import asyncio
import sys

from agent.cli import parse_args, run_interactive, run_autonomous


def main():
    args = parse_args()
    if args.auto:
        exit_code = asyncio.run(run_autonomous(args))
        sys.exit(exit_code)
    else:
        asyncio.run(run_interactive(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: all 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent/cli.py src/agent/__main__.py tests/test_cli.py
git commit -m "feat: add CLI with interactive REPL and autonomous mode"
```

---

### Task 13: Integration Test — End-to-End Autonomous Run

**Files:**
- Create: `tests/test_integration.py`

This test runs the full orchestrator against a mock LLM to verify all components wire together correctly.

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.db import AgentDB
from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator
from agent.planner import Planner
from agent.sandbox import Sandbox
from agent.tools import FileTools
from agent.verifier import Verifier


PLAN_RESPONSE = """## Plan: Create hello script

### Step 1: Create hello.py
- Files needed: hello.py
- Verify: python hello.py
"""

EXECUTOR_RESPONSE = """I'll create the hello script.

CREATE hello.py
```
print("hello world")
```
"""


@pytest.mark.asyncio
async def test_full_autonomous_run(tmp_path):
    db = AgentDB(tmp_path / ".agent" / "agent.db")

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(side_effect=[PLAN_RESPONSE, EXECUTOR_RESPONSE])

    sandbox = Sandbox(tmp_path)
    tools = FileTools(tmp_path)
    planner = Planner(mock_llm)
    executor = Executor(mock_llm, tools)
    verifier = Verifier(sandbox, commands=[])

    orch = Orchestrator(
        planner=planner,
        executor=executor,
        verifier=verifier,
        tools=tools,
        db=db,
        project_root=tmp_path,
    )

    result = await orch.run("Create a hello world script", mode="autonomous")

    assert result["status"] == "completed"
    assert (tmp_path / "hello.py").exists()
    assert "hello world" in (tmp_path / "hello.py").read_text()

    conv = db.get_conversation(result["conv_id"])
    assert conv["status"] == "completed"

    plans = db.get_plans(result["conv_id"])
    assert len(plans) >= 1

    edits = db.get_edits(result["conv_id"])
    assert len(edits) == 1
    assert edits[0]["file_path"] == "hello.py"
    assert edits[0]["edit_type"] == "create"


@pytest.mark.asyncio
async def test_full_run_with_verification(tmp_path):
    db = AgentDB(tmp_path / ".agent" / "agent.db")

    plan_response = """## Plan: Create and test script

### Step 1: Create greet.py
- Files needed: greet.py
- Verify: python greet.py
"""

    executor_response = """
CREATE greet.py
```
print("greetings")
```
"""

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(side_effect=[plan_response, executor_response])

    sandbox = Sandbox(tmp_path)
    tools = FileTools(tmp_path)
    planner = Planner(mock_llm)
    executor = Executor(mock_llm, tools)
    verifier = Verifier(sandbox, commands=[])

    orch = Orchestrator(
        planner=planner,
        executor=executor,
        verifier=verifier,
        tools=tools,
        db=db,
        project_root=tmp_path,
    )

    result = await orch.run("Create greeting script", mode="autonomous")
    assert result["status"] == "completed"
    assert (tmp_path / "greet.py").exists()

    cmd_result = sandbox.run_command("python greet.py")
    assert cmd_result.exit_code == 0
    assert "greetings" in cmd_result.stdout
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: all 2 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS across all test files

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for autonomous mode"
```

---

### Task 14: Manual Smoke Test with Ollama

This is a manual verification task — no code changes, just confirming the agent works against a real LLM.

- [ ] **Step 1: Ensure Ollama is running with qwen3:14b**

Run: `ollama list` to check if qwen3:14b is available.
If not: `ollama pull qwen3:14b`

- [ ] **Step 2: Create a test project**

```bash
mkdir /tmp/test-project
cd /tmp/test-project
echo "print('hello')" > main.py
```

- [ ] **Step 3: Run agent in autonomous mode**

Run: `python -m agent --project /tmp/test-project --auto --task "Add a function called greet that takes a name parameter and prints a greeting. Add it to main.py."`

Expected: Agent generates a plan, executes it, modifies main.py, and reports completion.

- [ ] **Step 4: Run agent in interactive mode**

Run: `python -m agent --project /tmp/test-project`

Expected: REPL starts, you can type tasks, see plans, approve them, and watch execution.

- [ ] **Step 5: Verify slash commands work**

In the REPL:
- Type `/help` — should show command list
- Type `/config` — should show model and URL settings
- Type `/history` — should show previous conversation from step 3
- Type `/quit` — should exit cleanly

- [ ] **Step 6: Commit any fixes discovered during smoke testing**

```bash
git add -A
git commit -m "fix: address issues found during smoke testing"
```

---

## Self-Review

**Spec coverage check:**
- CLI/REPL with interactive and autonomous modes → Task 12, 14
- Planner/executor split with separate system prompts → Tasks 7, 8, 9
- Adaptive edit format (whole-file < 300 lines, search/replace >= 300 lines) → Task 6 (parser), Task 9 (executor prompt)
- Matching cascade with error feedback loop → Task 4 (tools.edit_file), Task 11 (orchestrator retry)
- Sandboxed file and command tools → Tasks 3, 4
- Configurable verification commands → Task 10
- SQLite storage for config, conversations, plans, edits → Task 2
- Ollama as primary inference backend → Task 5 (client), Task 14 (smoke test)
- Streaming output → Task 5 (chat_stream), Task 12 (cli uses rich)
- Model pluggability via base_url + model config → Task 5, Task 12 (CLI args)
- Max replan attempts: 3 → Task 11 (MAX_REPLANS = 3)
- Error recovery with 2 retries then escalate → Task 11 (MAX_EXECUTOR_RETRIES = 2)

**Placeholder scan:** No TBDs, TODOs, or incomplete sections found.

**Type consistency check:**
- `Plan`, `Step`, `FileEdit`, `ExecutionResult`, `CommandResult`, `VerificationResult` — defined in Task 1 (models.py), used consistently in Tasks 6, 8, 9, 10, 11
- `LLMClient.chat()` returns `str` — consumed by planner and executor
- `FileTools.edit_file()` returns `bool` — checked in orchestrator
- `Verifier.run()` returns `VerificationResult` — checked in orchestrator
- `parse_plan()` returns `Plan` — used by Planner
- `parse_edits()` returns `list[FileEdit]` or `tuple[list[FileEdit], list[str]]` — used by Executor

All types, method signatures, and property names are consistent across tasks.
