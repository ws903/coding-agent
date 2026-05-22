# src/agent/lint_gate.py
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


LINT_RULES = ["F821", "F822", "E111", "E112", "E113"]

LINT_TIMEOUT = 30


@dataclass
class LintError:
    file: str
    row: int
    col: int
    code: str
    message: str


@dataclass
class LintResult:
    passed: bool
    new_errors: list[LintError] = field(default_factory=list)
    pre_existing: int = 0


class LintGate:
    def __init__(self, project_root: Path, rules: list[str] | None = None):
        self.project_root = project_root.resolve()
        self.rules = rules or LINT_RULES
        self._ruff = shutil.which("ruff")

    @property
    def available(self) -> bool:
        return self._ruff is not None

    def check_file(self, path: str) -> list[LintError]:
        if not self.available:
            return []
        full_path = self.project_root / path
        if not full_path.exists() or full_path.suffix != ".py":
            return []
        try:
            result = subprocess.run(
                [
                    self._ruff,
                    "check",
                    "--select",
                    ",".join(self.rules),
                    "--output-format=json",
                    "--no-cache",
                    str(full_path),
                ],
                capture_output=True,
                text=True,
                timeout=LINT_TIMEOUT,
                cwd=str(self.project_root),
            )
            if result.stdout.strip():
                return self._parse_output(result.stdout, path)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return []

    def gate_edit(
        self, path: str, before_errors: list[LintError] | None = None
    ) -> LintResult:
        if not self.available or not path.endswith(".py"):
            return LintResult(passed=True)

        after_errors = self.check_file(path)
        if not after_errors:
            return LintResult(passed=True)

        if before_errors is None:
            before_errors = []

        before_set = {(e.row, e.code, e.message) for e in before_errors}
        new = []
        pre_existing = 0
        for err in after_errors:
            if (err.row, err.code, err.message) in before_set:
                pre_existing += 1
            else:
                new.append(err)

        return LintResult(
            passed=len(new) == 0,
            new_errors=new,
            pre_existing=pre_existing,
        )

    def _parse_output(self, raw: str, rel_path: str) -> list[LintError]:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []
        errors = []
        for item in items:
            loc = item.get("location", {})
            errors.append(
                LintError(
                    file=rel_path,
                    row=loc.get("row", 0),
                    col=loc.get("column", 0),
                    code=item.get("code", "unknown"),
                    message=item.get("message", ""),
                )
            )
        return errors
