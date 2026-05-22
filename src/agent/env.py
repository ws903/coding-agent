# src/agent/env.py
"""Environment summary -- OS, shell, tool versions, git state.

Extracted from Orchestrator so the state machine doesn't have to know
how to introspect the host. The summary is cheap to compute once per
agent run and is fed into the planner's project context.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Common dev tools we check for. If any are present we list them with
# version so the planner can adapt (e.g. "this project has rg, use it").
PROBED_TOOLS = (
    "git",
    "python3",
    "python",
    "node",
    "npm",
    "pip",
    "pip3",
    "cargo",
    "make",
    "docker",
    "rg",
)


class EnvironmentDetector:
    """Probes the host and returns a fenced `<env>` block for prompts."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self._cache: str | None = None

    def detect(self) -> str:
        """Return the cached env summary, computing it on first call."""
        if self._cache is None:
            self._cache = self._compute()
        return self._cache

    def _compute(self) -> str:
        shell_path = shutil.which("bash") or shutil.which("zsh") or shutil.which("sh")
        shell = Path(shell_path).name if shell_path else "unknown"

        tools = []
        for cmd in PROBED_TOOLS:
            if shutil.which(cmd) is None:
                continue
            ver = _tool_version(cmd)
            tools.append(f"{cmd} {ver}" if ver else cmd)

        lines = [
            "<env>",
            f"os: {platform.system()} {platform.machine()}",
            f"shell: {shell}",
            f"cwd: {self.project_root}",
        ]
        git_info = _detect_git(self.project_root)
        if git_info:
            lines.append(f"git: {git_info}")
        if tools:
            lines.append(f"tools: {', '.join(tools)}")
        lines.append("</env>")
        return "\n".join(lines)


def _tool_version(cmd: str) -> str:
    """Best-effort version string. Returns "" if unprobeable."""
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = result.stdout.strip().split("\n")[0]
        for part in first_line.split():
            if part[0].isdigit():
                return part.rstrip(",")
    except (subprocess.SubprocessError, OSError, IndexError) as exc:
        logger.debug("Could not get version of %s: %s", cmd, exc)
    return ""


def _detect_git(project_root: Path) -> str:
    """Return `<branch> (clean|dirty)` or "" if not a git repo."""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(project_root),
        ).stdout.strip()
        if not branch:
            return ""
        diff_stat = subprocess.run(
            ["git", "diff", "--stat", "--quiet"],
            capture_output=True,
            timeout=5,
            cwd=str(project_root),
        )
        status = "clean" if diff_stat.returncode == 0 else "dirty"
        return f"{branch} ({status})"
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("Could not detect git state: %s", exc)
        return ""
