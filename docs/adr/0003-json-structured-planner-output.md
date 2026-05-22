# ADR 0003 — JSON-structured planner output

* Status: Accepted
* Date: 2026-05-22

## Context

The planner originally emitted markdown (`## Plan: ...`, `### Step N: ...`, or `## Answer\n...`) parsed by regex. Behavior was fragile: malformed headers, prose around the structured section, or model deviations all triggered retries or returned empty answers. We had a real incident where a complex prompt produced `answered` with empty content because the parser couldn't extract a section it expected.

Ollama's OpenAI-compat endpoint supports `response_format: {"type": "json_object"}` reliably (verified live: qwen3.6:35b returns valid JSON every call on first attempt).

## Decision

Planner uses `LLMClient.chat_json(messages)` which sets `response_format=json_object`. Model is told (in `prompts/planner.md`) to return:

```json
{"kind": "plan", "goal": "...", "steps": [{"id": 1, "action": "...", "files_needed": [...], "verify_command": "..."}]}
```

or

```json
{"kind": "answer", "answer": "..."}
```

Python validates the shape with field-by-field error messages, and retries with targeted feedback when the model produces a wrong shape (rare in practice).

## Rationale

- **No regex parser to maintain.** Replaces ~80 lines of parse_planner_response / validate_plan code with structural validation against known fields.
- **Empty-answer bug fixed.** The original failure was the markdown parser bailing out silently. JSON validation explicitly identifies which field is missing and routes to retry.
- **Forward-compatible.** Adding new optional fields (e.g. `priority`, `parallelizable`) doesn't break the parser — unknown keys are ignored.
- **`replan` coerces nonsensical answer responses to an empty plan** so the orchestrator's max-replans guard fires cleanly instead of looping forever.

## Consequences

- **`parser.py` becomes dead** for planner use (also became dead for executor edits via ADR 0002 — confirmed and removed).
- **The system prompt is longer.** It now contains a schema and three examples. Acceptable trade: input tokens are cheap; reliability isn't.
- **`prompts/planner.md` rewrite** drops the old "decide answer mode vs plan mode" English description; the JSON `kind` field carries that meaning structurally.

## Alternatives considered

- **GBNF schema constraint** — Tighter than `response_format`. Ollama doesn't expose grammar-constrained output via the OpenAI-compat path. Could add via native `/api/chat` later if needed.
- **Function calling for the planner too** — Define a `propose_plan` and `propose_answer` tool. Awkward: the planner doesn't *act*, it just outputs a structure. JSON output is the natural fit.
