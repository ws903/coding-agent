# src/agent/orchestrator.py
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from agent.codebase_index import CodebaseIndex
from agent.command_policy import CommandBlocked
from agent.db import AgentDB
from agent.edit_applier import EditApplier
from agent.env import EnvironmentDetector
from agent.executor import Executor
from agent.git_ops import GitOps
from agent.lint_gate import LintGate
from agent.models import (
    AgentStatus,
    AgentTokenUsage,
    Answer,
    ExecutionResult,
    ModelUsage,
    Plan,
    Step,
)
from agent.planner import Planner
from agent.tools import FileTools
from agent.verifier import Verifier

logger = logging.getLogger(__name__)

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
        on_edit_applied: Callable[[str, str, str | None, str | None], None]
        | None = None,
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
        self.env = EnvironmentDetector(project_root)
        self.edit_applier = EditApplier(
            tools, self.lint, db, on_edit_applied=on_edit_applied
        )
        self._aborted = False
        self._task_description: str = ""
        self._current_step: str = ""
        self._steps_executed = 0
        self._total_steps = 0
        # Conversation id of the active run. Initialized here so the cancel
        # handler in run() can safely read it even if cancellation fires
        # before _run_inner() got far enough to set it.
        self._conv_id: str = ""
        # asyncio handles for thread-safe cancellation. Captured in run()
        # so abort() can cancel an in-flight LLM HTTP call mid-await from
        # another thread (e.g. the ESC watcher).
        self._run_asyncio_task: asyncio.Task | None = None
        self._run_loop: asyncio.AbstractEventLoop | None = None

    def abort(self) -> None:
        self._aborted = True
        # Cancel the in-flight asyncio task from any thread. Without this,
        # abort() just sets a flag that's only checked between steps -- the
        # planner's HTTP call would finish naturally before we noticed.
        loop = self._run_loop
        task = self._run_asyncio_task
        if loop is not None and task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)

    def status(self) -> AgentStatus:
        return AgentStatus(
            task=self._task_description,
            current_step=self._current_step,
            steps_executed=self._steps_executed,
            total_steps=self._total_steps,
            aborted=self._aborted,
        )

    def token_usage(self) -> AgentTokenUsage:
        planner_usage = self.planner.llm.total_usage
        executor_usage = self.executor.llm.total_usage
        return AgentTokenUsage(
            planner=ModelUsage(
                prompt_tokens=planner_usage.prompt_tokens,
                completion_tokens=planner_usage.completion_tokens,
                total_tokens=planner_usage.total_tokens,
                calls=self.planner.llm.call_count,
            ),
            executor=ModelUsage(
                prompt_tokens=executor_usage.prompt_tokens,
                completion_tokens=executor_usage.completion_tokens,
                total_tokens=executor_usage.total_tokens,
                calls=self.executor.llm.call_count,
            ),
        )

    async def run(
        self,
        task: str,
        mode: str = "autonomous",
        approve_plan: Callable[..., bool] | None = None,
    ) -> dict:
        # Reset per-run state BEFORE capturing the asyncio handles, so an
        # abort fired immediately after run() returns to the caller doesn't
        # see stale conv_id from a previous run.
        self._conv_id = ""
        self._aborted = False
        # Capture handles so abort() can cancel an in-flight HTTP call.
        self._run_asyncio_task = asyncio.current_task()
        self._run_loop = asyncio.get_running_loop()
        try:
            return await self._run_inner(task, mode, approve_plan)
        except asyncio.CancelledError:
            # Only treat as a clean abort if WE cancelled (via abort()).
            # Other cancellations (e.g. supervisor shutdown) must propagate.
            if not self._aborted:
                raise
            # _conv_id may be "" if cancellation fired before _run_inner
            # created the conversation -- skip the DB update in that case.
            if self._conv_id:
                self.db.update_conversation_status(self._conv_id, "aborted")
            return {"status": "aborted", "conv_id": self._conv_id}
        finally:
            self._run_asyncio_task = None
            self._run_loop = None

    async def _run_inner(
        self,
        task: str,
        mode: str,
        approve_plan: Callable[..., bool] | None,
    ) -> dict:
        conv_id = self.db.create_conversation(mode, task)
        # Stash conv_id so the cancellation handler in run() can return it.
        self._conv_id = conv_id
        self.db.add_message(conv_id, "user", task)

        self._aborted = False
        self._task_description = task
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
            # Embed progress [N/total] for the CLI's status renderer to surface.
            self.on_status(
                f"Executing step {step.id} "
                f"[{step_index + 1}/{len(plan.steps)}]: {step.action}"
            )

            snapshot_sha = self._snapshot(f"agent: before step {step.id}")
            success = await self._execute_step(conv_id, step, completed_steps)
            self._steps_executed += 1

            if success:
                self._snapshot(f"agent: step {step.id} - {step.action[:60]}")
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

    async def _execute_step(
        self,
        conv_id: str,
        step: Step,
        completed_steps: list[dict] | None = None,
    ) -> bool:
        for attempt in range(MAX_EXECUTOR_RETRIES):
            errors = None
            if attempt > 0:
                errors = self._get_last_error(conv_id)
                self.on_status(f"  Retry {attempt}/{MAX_EXECUTOR_RETRIES - 1}...")

            result = await self.executor.execute(
                step, errors=errors, completed_steps=completed_steps
            )
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
        return self.edit_applier.apply(conv_id, step_id, result)

    def _build_project_context(self) -> str:
        env = self.env.detect()
        files = self.tools.list_files(".")
        tree = "\n".join(f"  {f}" for f in files[:100])
        index = CodebaseIndex(self.project_root).summary()
        sections = [env, f"## File Tree\n{tree}"]
        if index:
            sections.append(index)
        return "\n".join(sections) + "\n"

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
        except Exception as exc:
            # Snapshot is best-effort -- if git is unreachable or in a bad
            # state we want to keep going and skip rollback rather than crash
            # the whole run. Log so failures are visible.
            logger.warning("git snapshot %r failed: %s", message, exc)
            return None

    def _get_last_error(self, conv_id: str) -> str:
        messages = self.db.get_messages(conv_id)
        for msg in reversed(messages):
            if msg["role"] == "verifier" and "failed" in msg["content"].lower():
                return msg["content"]
            if msg["role"] == "system" and "failed" in msg["content"].lower():
                return msg["content"]
        return "Unknown error"
