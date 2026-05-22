# ADR 0002 — Native tool calling instead of regex-parsed edit blocks

* Status: Accepted
* Date: 2026-05-22

## Context

Until mid-2026 the executor asked the LLM to emit text blocks (`<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE`, `CREATE path\n...`, `REWRITE path\n...`) plus inline pseudo-tool commands (`READ:`, `SEARCH_CODE:`, `LIST:`). A regex parser converted text to `FileEdit` records. This is the Aider-era pattern, and it has known failure modes — 2026 research summary: *"String-based replacement is particularly wasteful and brittle... 'no occurrence' / 'multiple occurrence' errors are a known failure mode."*

Modern thinking models (qwen3.6, llama 3.x, gpt-oss) natively support OpenAI-format `tool_calls` reliably.

## Decision

Replace text-block edits with native tool calling. Define seven tools in `agent/tool_schemas.py`:

| Tool | Behavior |
|---|---|
| `read_file`, `list_files`, `search_text` | Execute live during the loop |
| `create_file`, `edit_file`, `replace_file` | Record a `FileEdit` for the orchestrator to apply + lint-gate |
| `run_command` | Queue for the orchestrator's sandboxed runner |

Executor calls `LLMClient.chat_with_tools(messages, TOOLS)` in a loop until `finish_reason=stop`. `ToolRunner.dispatch` routes each `tool_call` to the right handler.

## Rationale

- **Reliability.** qwen3.6 emits structured `tool_calls` consistently. No "search string doesn't match" loops. Verified end-to-end against the live model.
- **Self-describing.** Tool schemas tell the model the contract; the system prompt becomes short. We dropped 4 few-shot examples that previously taught the edit format.
- **Cleaner separation.** Read tools execute immediately; edit tools record intent. The orchestrator's existing `_apply_edits` + lint-gate stays as the safety layer; no new architecture needed.
- **MCP compatibility.** External MCP servers are themselves tool providers — adding MCP becomes a list-merge instead of a parser extension. See ADR 0004.

## Consequences

- **Old text-edit parser is dead.** `parser.py` removed entirely; its tests dropped.
- **Executor prompt is much shorter** (`prompts/executor.md` ~25 lines vs ~60 before). Lower input-token cost per call.
- **Subagents share the schema set.** `SubagentRunner` reuses `TOOLS` but filters down to read-only via `ALLOWED_TOOL_NAMES`. The same dispatch code; no parallel parser.

## Alternatives considered

- **GBNF / grammar-constrained generation.** Would force valid JSON tool calls structurally. Skipped for now: model adherence is high enough without it, and the constraint adds backend-specific complexity (TabbyAPI/ExLlamaV3 only).
- **Keep parser as a fallback** if `tool_calls` are empty. Rejected: empty tool_calls + content is a valid "done" state, and the old parser added 100+ lines of code we'd have to keep maintaining. If a non-tool-calling backend appears, add the fallback then.
