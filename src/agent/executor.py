# src/agent/executor.py
from collections.abc import Callable
from importlib import resources

from agent.llm_client import LLMClient
from agent.models import ExecutionResult, Step
from agent.tool_runner import ToolRunner
from agent.tool_schemas import TOOLS
from agent.tools import FileTools

MAX_TOOL_ITERATIONS = 10


def _load_prompt() -> str:
    return (
        resources.files("agent.prompts")
        .joinpath("executor.md")
        .read_text(encoding="utf-8")
    )


class Executor:
    def __init__(
        self,
        llm_client: LLMClient,
        tools: FileTools,
        on_token: Callable[[str], None] | None = None,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = _load_prompt()
        self.on_token = on_token

    async def execute(self, step: Step, errors: str | None = None) -> ExecutionResult:
        runner = ToolRunner(self.tools)
        user_content = self._build_user_content(step, errors)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_content = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            if self.on_token is not None:
                msg = await self.llm.chat_with_tools_stream(
                    messages, TOOLS, on_token=self.on_token, temperature=0.2
                )
            else:
                msg = await self.llm.chat_with_tools(messages, TOOLS, temperature=0.2)
            last_content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            messages.append(self._assistant_message(msg))

            if not tool_calls:
                break

            for tc in tool_calls:
                result = runner.dispatch(tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    }
                )

        return ExecutionResult(
            file_edits=runner.edits,
            commands=runner.commands,
            explanation=last_content,
        )

    def _build_user_content(self, step: Step, errors: str | None) -> str:
        parts = [f"## Step\n{step.action}"]
        if step.files_needed:
            parts.append(
                "\n## Files likely relevant\n"
                + "\n".join(f"- {p}" for p in step.files_needed)
            )
        if errors:
            parts.append(
                f"\n## Previous attempt failed\n{errors}\n"
                "Inspect the current state with read_file/search_text before retrying."
            )
        return "\n".join(parts)

    @staticmethod
    def _assistant_message(msg: dict) -> dict:
        """Trim assistant message to fields the Ollama endpoint expects on replay."""
        out: dict = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return out
