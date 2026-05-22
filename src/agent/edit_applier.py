# src/agent/edit_applier.py
"""Apply FileEdits with lint gating + DB audit trail.

Extracted from Orchestrator so the state machine focuses on plan/execute
flow and delegates the actual "write to disk, lint, roll back" mechanics
here. EditApplier captures one well-defined transactional concern: apply
a list of edits atomically (per-edit), respecting the lint gate, and
persist a before/after audit row for each.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.db import AgentDB
    from agent.lint_gate import LintGate
    from agent.models import ExecutionResult, FileEdit
    from agent.tools import FileTools

logger = logging.getLogger(__name__)


class EditApplier:
    """Applies file edits with lint gating and DB audit logging."""

    def __init__(self, tools: FileTools, lint: LintGate, db: AgentDB):
        self.tools = tools
        self.lint = lint
        self.db = db

    def apply(self, conv_id: str, step_id: int, result: ExecutionResult) -> bool:
        """Apply every edit in `result`. Returns True iff all succeeded.

        Each edit is: lint-before, write, lint-after, record-audit. If
        the post-edit lint introduces NEW errors, the file is rolled back
        to its pre-edit content and the edit is marked failed. Other
        edits in the batch still run.
        """
        all_ok = True
        for edit in result.file_edits:
            try:
                if not self._apply_one(conv_id, step_id, edit):
                    all_ok = False
            except Exception as exc:
                # Apply layer is the safety net -- catch broadly but log
                # so failures are visible. Expected errors are OSError
                # (disk), ValueError (path), but unexpected failures
                # shouldn't crash the whole run.
                logger.warning(
                    "Apply edit failed for %s: %s", edit.path, exc, exc_info=True
                )
                all_ok = False
        return all_ok

    def _apply_one(self, conv_id: str, step_id: int, edit: FileEdit) -> bool:
        lint_before = self.lint.check_file(edit.path)
        raw_before = self._read_raw(edit.path)

        before = self._write_edit(edit, raw_before)
        if before is False:
            return False

        lint_result = self.lint.gate_edit(edit.path, lint_before)
        if not lint_result.passed:
            self._record_lint_failure(conv_id, edit, lint_result, raw_before)
            return False

        after = self.tools.read_file(edit.path)
        self.db.save_edit(
            conv_id, step_id, edit.path, edit.action, before=before, after=after
        )
        return True

    def _write_edit(self, edit: FileEdit, raw_before: str | None) -> str | None | bool:
        """Apply the write. Returns the pre-edit content snippet for audit,
        or False if the edit can't be performed (e.g. search/replace miss)."""
        if edit.action == "create":
            self.tools.write_file(edit.path, edit.content or "")
            return None
        if edit.action == "rewrite":
            before = self.tools.read_file(edit.path) if raw_before is not None else None
            self.tools.write_file(edit.path, edit.content or "")
            return before
        if edit.action == "search_replace":
            if raw_before is None:
                return False
            before = self.tools.read_file(edit.path)
            success = self.tools.edit_file(
                edit.path, edit.search or "", edit.replace or ""
            )
            return before if success else False
        return False

    def _record_lint_failure(
        self,
        conv_id: str,
        edit: FileEdit,
        lint_result,
        raw_before: str | None,
    ) -> None:
        error_lines = [
            f"  {e.code} L{e.row}: {e.message}" for e in lint_result.new_errors
        ]
        self.db.add_message(
            conv_id,
            "lint",
            f"Lint errors in {edit.path}:\n" + "\n".join(error_lines),
        )
        if raw_before is not None:
            self.tools.write_file(edit.path, raw_before)

    def _read_raw(self, path: str) -> str | None:
        try:
            full_path = self.tools.sandbox.validate_path(path)
            if full_path.exists():
                return full_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError) as exc:
            logger.debug("Could not read %s: %s", path, exc)
        return None
