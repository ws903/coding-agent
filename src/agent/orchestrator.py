# src/agent/orchestrator.py
from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from agent.command_policy import CommandBlocked
from agent.db import AgentDB
from agent.git_ops import GitOps
from agent.lint_gate import LintGate
from agent.models import (
    Answer,
    Plan,
    Step,
    ExecutionResult,
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
        on_status: Callable[[str], None] | None = None,
        git_ops: GitOps | None = None,
        lint_gate: LintGate | None = None,
        max_steps: int = 20,
    ):
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.tools = tools
        self.db = db
        self.project_root = project_root
        self.on_status = on_status or (lambda msg: None)
        self.git = git_ops or GitOps(project_root)
        self.lint = lint_gate or LintGate(project_root)
        self.max_steps = max_steps
        self._aborted = False
        self._current_task: str = ""
        self._current_step: str = ""
        self._steps_executed = 0
        self._total_steps = 0

    def abort(self) -> None:
        self._aborted = True

    def status(self) -> dict:
        return {
            "task": self._current_task,
            "current_step": self._current_step,
            "steps_executed": self._steps_executed,
            "total_steps": self._total_steps,
            "aborted": self._aborted,
        }

    def token_usage(self) -> dict:
        planner_usage = self.planner.llm.total_usage
        executor_usage = self.executor.llm.total_usage
        return {
            "planner": {
                "prompt_tokens": planner_usage.prompt_tokens,
                "completion_tokens": planner_usage.completion_tokens,
                "total_tokens": planner_usage.total_tokens,
                "calls": self.planner.llm.call_count,
            },
            "executor": {
                "prompt_tokens": executor_usage.prompt_tokens,
                "completion_tokens": executor_usage.completion_tokens,
                "total_tokens": executor_usage.total_tokens,
                "calls": self.executor.llm.call_count,
            },
        }

    async def run(
        self,
        task: str,
        mode: str = "autonomous",
        approve_plan: Callable[..., bool] | None = None,
    ) -> dict:
        conv_id = self.db.create_conversation(mode, task)
        self.db.add_message(conv_id, "user", task)

        self._aborted = False
        self._current_task = task
        self._steps_executed = 0

        project_context = self._build_project_context()
        plan_version = 0
        replan_count = 0

        self.on_status("Generating plan...")
        response = await self.planner.generate_plan(task, project_context)

        if isinstance(response, Answer):
            self.db.add_message(conv_id, "planner", response.text)
            self.db.update_conversation_status(conv_id, "completed")
            return {
                "status": "answered",
                "conv_id": conv_id,
                "answer": response.text,
            }

        plan = response
        if not plan.steps:
            self.db.add_message(conv_id, "planner", f"Empty plan: {plan.goal}")
            self.db.update_conversation_status(conv_id, "failed")
            return {
                "status": "failed",
                "conv_id": conv_id,
                "reason": "empty_plan",
            }

        plan_version += 1
        self.db.save_plan(conv_id, plan_version, self._plan_to_text(plan))
        self.db.add_message(conv_id, "planner", self._plan_to_text(plan))
        self.on_status(f"Plan: {plan.goal} ({len(plan.steps)} steps)")

        if approve_plan and not approve_plan(plan):
            self.db.update_conversation_status(conv_id, "aborted")
            return {"status": "aborted", "conv_id": conv_id}

        step_index = 0
        completed_steps: list[dict] = []
        self._total_steps = len(plan.steps)
        while step_index < len(plan.steps):
            if self._aborted:
                self.on_status("Aborted by user.")
                self.db.update_conversation_status(conv_id, "aborted")
                return {"status": "aborted", "conv_id": conv_id}

            if self._steps_executed >= self.max_steps:
                self.on_status(f"Max steps ({self.max_steps}) reached.")
                self.db.update_conversation_status(conv_id, "failed")
                return {"status": "failed", "conv_id": conv_id, "reason": "max_steps"}

            step = plan.steps[step_index]
            self._current_step = f"Step {step.id}: {step.action}"
            self.on_status(f"Executing step {step.id}: {step.action}")

            snapshot_sha = self._snapshot(f"agent: before step {step.id}")
            success = await self._execute_step(conv_id, step)
            self._steps_executed += 1

            if success:
                completed_steps.append({"step_id": step.id, "action": step.action})
                step_index += 1
                continue

            if snapshot_sha:
                self.on_status("Rolling back failed step...")
                self.git.rollback(snapshot_sha)

            replan_count += 1
            if replan_count > MAX_REPLANS:
                self.on_status("Max replans reached. Stopping.")
                self.db.update_conversation_status(conv_id, "failed")
                return {"status": "failed", "conv_id": conv_id, "reason": "max_replans"}

            self.on_status(f"Replanning (attempt {replan_count}/{MAX_REPLANS})...")
            error_summary = self._get_last_error(conv_id)
            plan = await self.planner.replan(
                task,
                plan,
                step.id,
                error_summary,
                project_context,
                completed_steps=completed_steps,
            )
            plan_version += 1
            self.db.save_plan(conv_id, plan_version, self._plan_to_text(plan))
            self.db.add_message(
                conv_id,
                "planner",
                f"Replan v{plan_version}:\n{self._plan_to_text(plan)}",
            )
            step_index = 0
            completed_steps = []

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
                try:
                    cmd_result = self.tools.sandbox.run_command(cmd)
                except CommandBlocked as exc:
                    self.db.add_message(
                        conv_id, "system", f"Command blocked: `{cmd}` — {exc.reason}"
                    )
                    continue
                self.db.add_message(
                    conv_id, "system", f"Command `{cmd}`: exit={cmd_result.exit_code}"
                )

            verification = self.verifier.run(step_command=step.verify_command)
            if verification.passed:
                self.db.add_message(conv_id, "verifier", "Verification passed")
                return True

            error_detail = "\n".join(
                f"{d.cmd}: {d.stderr or d.stdout}"
                for d in verification.details
                if d.exit_code != 0
            )
            self.db.add_message(
                conv_id, "verifier", f"Verification failed:\n{error_detail}"
            )

        return False

    def _apply_edits(self, conv_id: str, step_id: int, result: ExecutionResult) -> bool:
        all_ok = True
        for edit in result.file_edits:
            try:
                lint_before = self.lint.check_file(edit.path)
                raw_before = self._read_raw(edit.path)

                if edit.action == "create":
                    before = None
                    self.tools.write_file(edit.path, edit.content)
                elif edit.action == "rewrite":
                    before = (
                        self.tools.read_file(edit.path)
                        if raw_before is not None
                        else None
                    )
                    self.tools.write_file(edit.path, edit.content)
                elif edit.action == "search_replace":
                    if raw_before is None:
                        all_ok = False
                        continue
                    before = self.tools.read_file(edit.path)
                    success = self.tools.edit_file(edit.path, edit.search, edit.replace)
                    if not success:
                        all_ok = False
                        continue
                else:
                    continue

                lint_result = self.lint.gate_edit(edit.path, lint_before)
                if not lint_result.passed:
                    error_lines = [
                        f"  {e.code} L{e.row}: {e.message}"
                        for e in lint_result.new_errors
                    ]
                    self.db.add_message(
                        conv_id,
                        "lint",
                        f"Lint errors in {edit.path}:\n" + "\n".join(error_lines),
                    )
                    if raw_before is not None:
                        self.tools.write_file(edit.path, raw_before)
                    all_ok = False
                    continue

                after = self.tools.read_file(edit.path)
                self.db.save_edit(
                    conv_id,
                    step_id,
                    edit.path,
                    edit.action,
                    before=before,
                    after=after,
                )
            except Exception:
                all_ok = False
        return all_ok

    def _read_raw(self, path: str) -> str | None:
        try:
            full_path = self.tools.sandbox.validate_path(path)
            if full_path.exists():
                return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        return None

    def _build_project_context(self) -> str:
        env = self._detect_environment()
        files = self.tools.list_files(".")
        tree = "\n".join(f"  {f}" for f in files[:100])
        return f"{env}\n## File Tree\n{tree}\n"

    def _detect_environment(self) -> str:
        if hasattr(self, "_env_cache"):
            return self._env_cache

        os_name = platform.system()
        arch = platform.machine()

        shell = "unknown"
        shell_path = shutil.which("bash") or shutil.which("zsh") or shutil.which("sh")
        if shell_path:
            shell = Path(shell_path).name

        tools = []
        for cmd in [
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
        ]:
            path = shutil.which(cmd)
            if path:
                ver = self._tool_version(cmd)
                tools.append(f"{cmd} {ver}" if ver else cmd)

        git_info = self._detect_git()

        lines = [
            "<env>",
            f"os: {os_name} {arch}",
            f"shell: {shell}",
            f"cwd: {self.project_root}",
        ]
        if git_info:
            lines.append(f"git: {git_info}")
        if tools:
            lines.append(f"tools: {', '.join(tools)}")
        lines.append("</env>")

        self._env_cache = "\n".join(lines)
        return self._env_cache

    def _tool_version(self, cmd: str) -> str:
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
        except Exception:
            pass
        return ""

    def _detect_git(self) -> str:
        try:
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(self.project_root),
            ).stdout.strip()
            if not branch:
                return ""
            diff_stat = subprocess.run(
                ["git", "diff", "--stat", "--quiet"],
                capture_output=True,
                timeout=5,
                cwd=str(self.project_root),
            )
            status = "clean" if diff_stat.returncode == 0 else "dirty"
            return f"{branch} ({status})"
        except Exception:
            return ""

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

    def _snapshot(self, message: str) -> str | None:
        try:
            return self.git.snapshot(message)
        except Exception:
            return None

    def _get_last_error(self, conv_id: str) -> str:
        messages = self.db.get_messages(conv_id)
        for msg in reversed(messages):
            if msg["role"] == "verifier" and "failed" in msg["content"].lower():
                return msg["content"]
            if msg["role"] == "system" and "failed" in msg["content"].lower():
                return msg["content"]
        return "Unknown error"
