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
