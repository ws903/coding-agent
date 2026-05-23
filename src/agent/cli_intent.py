# src/agent/cli_intent.py
"""Chat-vs-task routing for REPL input.

Three-tier classifier:
  1. obvious chat (hi, hey, thanks ...) -> fast-chat, 0 overhead
  2. long input (>10 words) -> planner, 0 overhead
  3. ambiguous short -> LLM classifier (~1s with think:false)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent import console as _con  # See cli_ui.py for why we use attribute access

if TYPE_CHECKING:
    from agent.orchestrator import Orchestrator

# Short conversational inputs that should bypass the planner entirely.
# Conservative -- we only fast-path obvious greetings; anything ambiguous
# falls through to the planner so we don't skip real work.
_CHAT_TOKENS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "thanks",
    "thank",
    "ok",
    "okay",
    "cool",
    "great",
    "nice",
    "bye",
    "goodbye",
    "morning",
}

_FAST_CHAT_PROMPT = (
    "You are a friendly assistant embedded in a coding agent CLI. "
    "Reply briefly (1-2 sentences). The user is making conversation, "
    "not asking for code work."
)

_CLASSIFIER_PROMPT = (
    "Classify the user message as either CHAT or TASK.\n\n"
    "CHAT: small talk, greetings, expressions of feeling, simple questions "
    "about the agent itself (not the codebase). Examples: 'hi', 'how are "
    "you', 'thanks', 'what's up', 'are you online', 'you're cool'.\n\n"
    "TASK: anything about a codebase, file system, or software work -- "
    "explanations, edits, refactors, debugging, file lookups, exploration. "
    "Examples: 'add a docstring', 'what does this codebase do', 'list the "
    "files in src/', 'fix the bug in auth', 'explain the orchestrator'.\n\n"
    "When in doubt, prefer TASK -- the planner can decline or return a "
    "direct answer if no work is needed.\n\n"
    "Reply with exactly one word: CHAT or TASK"
)


def _looks_like_chat(text: str) -> bool:
    """Cheap heuristic for obvious conversational input. Only matches single-
    word greetings; ambiguous short input falls through to the LLM classifier."""
    lower = text.lower().strip().rstrip("?!.,'\"")
    if not lower:
        return False
    words = lower.split()
    if len(words) > 3:
        return False
    return words[0] in _CHAT_TOKENS


async def _llm_classify_intent(orch: Orchestrator, user_input: str) -> str:
    """Returns 'chat' or 'task'. ~0.3s round-trip with think:false."""
    messages = [
        {"role": "system", "content": _CLASSIFIER_PROMPT},
        {"role": "user", "content": user_input},
    ]
    try:
        result = await orch.executor.llm.quick_chat_stream(
            messages, on_token=None, temperature=0.0
        )
    except Exception:
        # On any failure fall back to the safer TASK path so we never
        # accidentally route real work to fast-chat.
        return "task"
    return "chat" if "CHAT" in result.upper() else "task"


async def _route_input(orch: Orchestrator, user_input: str) -> str:
    """Tier-based routing: heuristic shortcuts first, LLM classifier for
    ambiguous middle. Long inputs (>10 words) skip the classifier."""
    if _looks_like_chat(user_input):
        return "chat"
    if len(user_input.split()) > 10:
        return "task"
    return await _llm_classify_intent(orch, user_input)


async def _fast_chat(orch: Orchestrator, user_input: str) -> None:
    """Bypass planner+executor AND the model's reasoning phase.

    Uses Ollama's native /api/chat with `think: false`. For thinking models
    like qwen3.6, this cuts "hi"-style replies from ~14s to ~3s by skipping
    the silent reasoning pass.
    """
    messages = [
        {"role": "system", "content": _FAST_CHAT_PROMPT},
        {"role": "user", "content": user_input},
    ]

    def on_token(chunk: str) -> None:
        _con.console.print(chunk, end="", soft_wrap=True)

    await orch.executor.llm.quick_chat_stream(
        messages, on_token=on_token, temperature=0.7
    )
    _con.console.print()
