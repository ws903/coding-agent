# src/agent/executor.py
from collections.abc import Callable
from importlib import resources

from agent.agents_manager import AgentsManager
from agent.llm_client import LLMClient
from agent.mcp_manager import MCPManager
from agent.models import ExecutionResult, Step
from agent.skills_manager import SkillsManager
from agent.subagent_runner import SubagentRunner
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


def _build_system_prompt(
    skills: SkillsManager | None,
    agents: AgentsManager | None,
) -> str:
    base = _load_prompt()
    sections = []
    if skills is not None:
        s = skills.catalog_section()
        if s:
            sections.append(s)
    if agents is not None:
        a = agents.catalog_section()
        if a:
            sections.append(a)
    if not sections:
        return base
    return base + "\n\n" + "\n\n".join(sections)


class Executor:
    def __init__(
        self,
        llm_client: LLMClient,
        tools: FileTools,
        on_token: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
        mcp: MCPManager | None = None,
        skills: SkillsManager | None = None,
        agents: AgentsManager | None = None,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = _build_system_prompt(skills, agents)
        self.on_token = on_token
        self.on_reasoning = on_reasoning
        self.on_tool_call = on_tool_call
        self.mcp = mcp
        self.skills = skills
        self.agents = agents
        self.subagent_runner = (
            SubagentRunner(llm_client, tools) if agents is not None else None
        )

    async def execute(
        self,
        step: Step,
        errors: str | None = None,
        completed_steps: list[dict] | None = None,
    ) -> ExecutionResult:
        runner = ToolRunner(
            self.tools,
            mcp=self.mcp,
            skills=self.skills,
            agents=self.agents,
            subagent_runner=self.subagent_runner,
            on_tool_call=self.on_tool_call,
        )
        user_content = self._build_user_content(step, errors, completed_steps)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        tools = TOOLS + (self.mcp.tools if self.mcp else [])

        last_content = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            if self.on_token is not None:
                msg = await self.llm.chat_with_tools_stream(
                    messages,
                    tools,
                    on_token=self.on_token,
                    on_reasoning=self.on_reasoning,
                    temperature=0.2,
                )
            else:
                msg = await self.llm.chat_with_tools(messages, tools, temperature=0.2)
            last_content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            messages.append(self._assistant_message(msg))

            if not tool_calls:
                break

            for tc in tool_calls:
                result = await runner.dispatch(tc)
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

    def _build_user_content(
        self,
        step: Step,
        errors: str | None,
        completed_steps: list[dict] | None = None,
    ) -> str:
        parts = [f"## Step\n{step.action}"]
        if completed_steps:
            done = "\n".join(
                f"- Step {s['step_id']}: {s['action']}" for s in completed_steps
            )
            parts.append(f"\n## Already completed in this task\n{done}")
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
