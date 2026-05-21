from importlib import resources

from agent.llm_client import LLMClient
from agent.models import Answer, Plan
from agent.parser import parse_plan, parse_planner_response, validate_plan
from agent.prompts.examples import PLANNER_EXAMPLES

MAX_PARSE_RETRIES = 2

RETRY_PROMPT = (
    "Your previous response could not be parsed. Errors:\n{errors}\n\n"
    "Please respond again using EXACTLY the format from your instructions. "
    "Do not add text outside the format."
)


def _load_prompt() -> str:
    return (
        resources.files("agent.prompts")
        .joinpath("planner.md")
        .read_text(encoding="utf-8")
    )


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = _load_prompt()

    async def generate_plan(self, task: str, project_context: str) -> Plan | Answer:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *PLANNER_EXAMPLES,
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n## Project Context\n{project_context}"
                ),
            },
        ]
        response = await self.llm.chat(messages, temperature=0.3)
        result = parse_planner_response(response)

        if isinstance(result, Answer):
            return result

        errors = validate_plan(result)
        if not errors:
            return result

        for _ in range(MAX_PARSE_RETRIES):
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": RETRY_PROMPT.format(errors="\n".join(errors)),
                }
            )
            response = await self.llm.chat(messages, temperature=0.2)
            result = parse_planner_response(response)
            if isinstance(result, Answer):
                return result
            errors = validate_plan(result)
            if not errors:
                return result

        return result

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
            *PLANNER_EXAMPLES,
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n"
                    f"## Project Context\n{project_context}\n\n"
                    f"## Previous Plan\n{plan_summary}\n\n"
                    f"{completed_section}"
                    f"## Failure\nStep {failed_step_id} failed with error:\n{error}\n\n"
                    f"The completed steps above are already done — do not repeat them. "
                    f"Create a revised plan starting from the failed step."
                ),
            },
        ]
        response = await self.llm.chat(messages, temperature=0.3)
        plan = parse_plan(response)

        errors = validate_plan(plan)
        if not errors:
            return plan

        for _ in range(MAX_PARSE_RETRIES):
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": RETRY_PROMPT.format(errors="\n".join(errors)),
                }
            )
            response = await self.llm.chat(messages, temperature=0.2)
            plan = parse_plan(response)
            errors = validate_plan(plan)
            if not errors:
                return plan

        return plan
