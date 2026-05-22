# src/agent/planner.py
"""Planner using JSON-structured output.

The planner asks the model for a JSON object of the form:

    {"kind": "plan", "goal": "...", "steps": [{"id": 1, "action": "...",
     "files_needed": [...], "verify_command": ...}, ...]}

or

    {"kind": "answer", "answer": "..."}

The model is told this in the system prompt and Ollama enforces JSON
output via response_format. We still keep a retry loop for the rare
case the model emits an invalid shape (missing required field, etc.)
or non-JSON despite the constraint.
"""

import json
from importlib import resources

from agent.llm_client import LLMClient
from agent.models import Answer, Plan, Step

MAX_PARSE_RETRIES = 2

RETRY_PROMPT = (
    "Your previous response was missing required fields or had the wrong "
    "shape:\n{errors}\n\nReply again with a single JSON object matching the "
    "schema in your instructions. No prose, no markdown fences -- just the JSON."
)


def _load_prompt() -> str:
    return (
        resources.files("agent.prompts")
        .joinpath("planner.md")
        .read_text(encoding="utf-8")
    )


def _validate_response(data: dict) -> tuple[Plan | Answer | None, list[str]]:
    """Convert a parsed JSON dict into Plan or Answer, or list field errors."""
    errors: list[str] = []
    kind = data.get("kind")

    if kind == "answer":
        text = data.get("answer")
        if not isinstance(text, str) or not text.strip():
            errors.append("kind=answer but 'answer' is missing or empty")
            return None, errors
        return Answer(text=text.strip()), []

    if kind == "plan":
        goal = data.get("goal")
        steps_raw = data.get("steps")
        if not isinstance(goal, str) or not goal.strip():
            errors.append("kind=plan but 'goal' is missing or empty")
        if not isinstance(steps_raw, list) or not steps_raw:
            errors.append("kind=plan but 'steps' is missing or empty")
        if errors:
            return None, errors

        steps: list[Step] = []
        for i, raw in enumerate(steps_raw):
            if not isinstance(raw, dict):
                errors.append(f"step {i} is not an object")
                continue
            sid = raw.get("id")
            action = raw.get("action")
            files_needed = raw.get("files_needed") or []
            verify = raw.get("verify_command")
            if not isinstance(sid, int):
                errors.append(f"step {i} missing integer 'id'")
                continue
            if not isinstance(action, str) or not action.strip():
                errors.append(f"step {sid} missing string 'action'")
                continue
            if not isinstance(files_needed, list):
                errors.append(f"step {sid} 'files_needed' must be a list")
                continue
            steps.append(
                Step(
                    id=sid,
                    action=action.strip(),
                    files_needed=[str(p) for p in files_needed],
                    verify_command=verify if isinstance(verify, str) else None,
                )
            )
        if errors:
            return None, errors
        return Plan(goal=goal.strip(), steps=steps), []

    errors.append(
        f"'kind' must be 'plan' or 'answer', got {kind!r}"
        if kind is not None
        else "missing required field 'kind'"
    )
    return None, errors


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = _load_prompt()

    async def generate_plan(self, task: str, project_context: str) -> Plan | Answer:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n## Project Context\n{project_context}"
                ),
            },
        ]
        return await self._planner_loop(messages)

    async def replan(
        self,
        task: str,
        current_plan: Plan,
        failed_step_id: int,
        error: str,
        project_context: str = "",
        completed_steps: list[dict] | None = None,
    ) -> Plan:
        plan_summary = f"Goal: {current_plan.goal}\n"
        for step in current_plan.steps:
            plan_summary += f"Step {step.id}: {step.action}\n"

        completed_section = ""
        if completed_steps:
            items = "\n".join(
                f"- Step {s['step_id']}: {s['action']} (DONE)" for s in completed_steps
            )
            completed_section = f"## Completed Steps\n{items}\n\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n"
                    f"## Project Context\n{project_context}\n\n"
                    f"## Previous Plan\n{plan_summary}\n\n"
                    f"{completed_section}"
                    f"## Failure\nStep {failed_step_id} failed:\n{error}\n\n"
                    f"The completed steps above are already done -- do not repeat them. "
                    f"Reply with a JSON plan starting from the failed step."
                ),
            },
        ]
        result = await self._planner_loop(messages)
        # On replan, an "answer" doesn't make sense -- coerce to an empty plan
        # so the orchestrator's max-replans guard kicks in cleanly.
        if isinstance(result, Answer):
            return Plan(goal="(planner returned non-plan)", steps=[])
        return result

    async def _planner_loop(self, messages: list[dict]) -> Plan | Answer:
        try:
            data = await self.llm.chat_json(messages, temperature=0.3)
            result, errors = _validate_response(data)
        except (json.JSONDecodeError, ValueError) as exc:
            data, result, errors = {}, None, [f"JSON parse failed: {exc}"]

        for _ in range(MAX_PARSE_RETRIES):
            if result is not None:
                return result
            messages.append(
                {"role": "assistant", "content": json.dumps(data) if data else ""}
            )
            messages.append(
                {
                    "role": "user",
                    "content": RETRY_PROMPT.format(errors="\n".join(errors)),
                }
            )
            try:
                data = await self.llm.chat_json(messages, temperature=0.2)
                result, errors = _validate_response(data)
            except (json.JSONDecodeError, ValueError) as exc:
                data, result, errors = {}, None, [f"JSON parse failed: {exc}"]

        # All retries exhausted -- return whatever we have (likely empty Plan)
        if result is not None:
            return result
        return Plan(goal="(planner failed to produce valid plan)", steps=[])
