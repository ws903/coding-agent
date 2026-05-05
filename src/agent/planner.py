from importlib import resources

from agent.llm_client import LLMClient
from agent.models import Plan
from agent.parser import parse_plan


def _load_prompt() -> str:
    return resources.files("agent.prompts").joinpath("planner.md").read_text()


class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.system_prompt = _load_prompt()

    async def generate_plan(self, task: str, project_context: str) -> Plan:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n"
                    f"## Project Context\n{project_context}"
                ),
            },
        ]
        response = await self.llm.chat(messages, temperature=0.3)
        return parse_plan(response)

    async def replan(
        self,
        task: str,
        current_plan: Plan,
        failed_step_id: int,
        error: str,
    ) -> Plan:
        plan_summary = f"Goal: {current_plan.goal}\n"
        for step in current_plan.steps:
            plan_summary += f"Step {step.id}: {step.action}\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"## Task\n{task}\n\n"
                    f"## Previous Plan\n{plan_summary}\n\n"
                    f"## Failure\nStep {failed_step_id} failed with error:\n{error}\n\n"
                    f"Please create a revised plan that addresses this failure."
                ),
            },
        ]
        response = await self.llm.chat(messages, temperature=0.3)
        return parse_plan(response)
