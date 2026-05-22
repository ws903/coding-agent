# src/agent/models.py
from dataclasses import dataclass, field
from typing import Literal, TypedDict

EditAction = Literal["create", "rewrite", "search_replace"]


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
class Answer:
    text: str


@dataclass
class FileEdit:
    path: str
    action: EditAction
    content: str | None = None  # for create/rewrite
    search: str | None = None  # for search_replace
    replace: str | None = None  # for search_replace


class CompletedStep(TypedDict):
    """Record of a successfully completed step. Used in cross-step context."""

    step_id: int
    action: str


class ModelUsage(TypedDict):
    """Per-model token usage stats."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    calls: int


class AgentTokenUsage(TypedDict):
    """The combined planner+executor usage shape returned by orchestrator."""

    planner: ModelUsage
    executor: ModelUsage


class AgentStatus(TypedDict):
    """Live orchestrator status snapshot."""

    task: str
    current_step: str
    steps_executed: int
    total_steps: int
    aborted: bool


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


@dataclass
class ParseError:
    errors: list[str]
    raw_output: str


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
