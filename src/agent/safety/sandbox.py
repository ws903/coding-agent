# src/agent/sandbox.py
import subprocess
from pathlib import Path

from agent.safety.command_policy import CommandBlocked, check_command
from agent.core.models import CommandResult


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
        verdict, reason = check_command(command)
        if verdict == "block":
            raise CommandBlocked(command, reason)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
