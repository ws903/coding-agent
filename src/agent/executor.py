# src/agent/executor.py
from importlib import resources

from agent.llm_client import LLMClient
from agent.models import Step, ExecutionResult
from agent.parser import parse_edits
from agent.tools import FileTools


def _load_prompt() -> str:
    return resources.files("agent.prompts").joinpath("executor.md").read_text(encoding="utf-8")


class Executor:
    def __init__(self, llm_client: LLMClient, tools: FileTools):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = _load_prompt()

    async def execute(self, step: Step, errors: str | None = None) -> ExecutionResult:
        file_contents = self._gather_files(step.files_needed)
        user_content = f"## Step\n{step.action}\n"

        if file_contents:
            user_content += f"\n## Current File Contents\n{file_contents}\n"

        if errors:
            user_content += (
                f"\n## Previous Attempt Failed\n"
                f"The previous attempt produced these errors:\n{errors}\n"
                f"Please fix the issues and try again.\n"
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.llm.chat(messages, temperature=0.2)
        edits, commands = parse_edits(response, extract_commands=True)

        return ExecutionResult(
            file_edits=edits,
            commands=commands,
            explanation=response,
        )

    def _gather_files(self, file_paths: list[str]) -> str:
        sections = []
        for path in file_paths:
            try:
                content = self.tools.read_file(path)
                sections.append(f"### {path}\n```\n{content}\n```")
            except FileNotFoundError:
                sections.append(f"### {path}\n(file does not exist yet)")
        return "\n\n".join(sections)
