# src/agent/executor.py
import re
from importlib import resources

from agent.llm_client import LLMClient
from agent.models import Step, ExecutionResult
from agent.parser import parse_edits, validate_edits
from agent.prompts.examples import EXECUTOR_EXAMPLES
from agent.tools import FileTools

MAX_TOOL_ITERATIONS = 10
MAX_PARSE_RETRIES = 2
MAX_OBSERVATION_CHARS = 3000

RETRY_PROMPT = (
    "Your previous response could not be parsed. Errors:\n{errors}\n\n"
    "Please respond again using EXACTLY the edit format from your instructions "
    "(SEARCH/REPLACE, CREATE, or REWRITE blocks). "
    "Do not add text outside the format."
)

_READ_PATTERN = re.compile(r"^READ:\s*(.+)$", re.MULTILINE)
_SEARCH_PATTERN = re.compile(r"^SEARCH_CODE:\s*(.+)$", re.MULTILINE)
_LIST_PATTERN = re.compile(r"^LIST:\s*(.+)$", re.MULTILINE)


def _truncate(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n\n[... {omitted} chars truncated ...]\n\n{text[-half:]}"


def _load_prompt() -> str:
    return (
        resources.files("agent.prompts")
        .joinpath("executor.md")
        .read_text(encoding="utf-8")
    )


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
            *EXECUTOR_EXAMPLES,
            {"role": "user", "content": user_content},
        ]

        response = await self.llm.chat(messages, temperature=0.2)

        for _ in range(MAX_TOOL_ITERATIONS):
            tool_results = self._process_tool_commands(response)
            if not tool_results:
                break
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": tool_results})
            response = await self.llm.chat(messages, temperature=0.2)

        edits, commands = parse_edits(response, extract_commands=True)

        parse_errors = validate_edits(edits)
        for _ in range(MAX_PARSE_RETRIES):
            if not parse_errors:
                break
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": RETRY_PROMPT.format(errors="\n".join(parse_errors)),
                }
            )
            response = await self.llm.chat(messages, temperature=0.2)
            edits, commands = parse_edits(response, extract_commands=True)
            parse_errors = validate_edits(edits)

        return ExecutionResult(
            file_edits=edits,
            commands=commands,
            explanation=response,
        )

    def _process_tool_commands(self, response: str) -> str:
        results = []

        for match in _READ_PATTERN.finditer(response):
            path = match.group(1).strip()
            try:
                content = self.tools.read_file(path)
                results.append(f"## File: {path}\n```\n{_truncate(content)}\n```")
            except FileNotFoundError:
                results.append(f"## File: {path}\n(file not found)")

        for match in _SEARCH_PATTERN.finditer(response):
            pattern = match.group(1).strip()
            try:
                hits = self.tools.search_text(pattern)
                if hits:
                    formatted = "\n".join(f"  {h}" for h in hits[:20])
                    results.append(f"## Search results for '{pattern}':\n{formatted}")
                else:
                    results.append(f"## Search results for '{pattern}':\n(no matches)")
            except Exception:
                results.append(f"## Search results for '{pattern}':\n(search failed)")

        for match in _LIST_PATTERN.finditer(response):
            path = match.group(1).strip()
            try:
                files = self.tools.list_files(path)
                formatted = "\n".join(f"  {f}" for f in files[:50])
                results.append(f"## Files in {path}:\n{formatted}")
            except Exception:
                results.append(f"## Files in {path}:\n(listing failed)")

        if not results:
            return ""
        return "\n\n".join(results)

    def _gather_files(self, file_paths: list[str]) -> str:
        sections = []
        for path in file_paths:
            try:
                content = self.tools.read_file(path)
                content = _truncate(content)
                sections.append(f"### {path}\n```\n{content}\n```")
            except FileNotFoundError:
                sections.append(f"### {path}\n(file does not exist yet)")
        return "\n\n".join(sections)
