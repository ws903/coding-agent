# src/agent/orchestrator.py
from __future__ import annotations

from pathlib import Path

from agent.db import AgentDB
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
            self.db.add_message(
                conv_id,
                "planner",
                f"Replan v{plan_version}:\n{self._plan_to_text(plan)}",
            )
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
