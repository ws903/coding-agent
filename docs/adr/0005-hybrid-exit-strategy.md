# ADR 0005 — Hybrid bare-word + slash exit strategy

* Status: Accepted
* Date: 2026-05-22

## Context

REPL exit conventions vary:

| Tool | Exit mechanism |
|---|---|
| Claude Code | `/exit`, `/quit` — slash-only |
| Codex CLI | `/exit` — slash-only |
| Aider | `/exit`, `/quit` — slash-only |
| Python REPL, ipython, sqlite | `exit`, `quit` — bare word |
| Most shells | `exit` — bare word |

Slash-only is safer for tools serving millions of users: typing bare `exit` as part of `"remove the exit handler"` shouldn't silently quit. Bare-word is more ergonomic for personal-use CLIs.

User reported: typing bare `exit` in the REPL triggered the full planner pipeline instead of leaving, because the chat-classifier heuristic only matched greetings.

## Decision

Both work. Specifically:

- **Exact-match bare words** (case-insensitive, no other tokens): `exit`, `quit`, `q`, `:q` → exit
- **Slash commands**: `/quit`, `/exit` → exit (same outcome)
- **Anything else with those words in it** (e.g. `exit the loop in main.py`) → falls through to the planner as a regular task

Implementation: `_EXIT_INPUTS = {"exit", "quit", "q", ":q"}`; check is `user_input.lower() in _EXIT_INPUTS or user_input in {"/quit", "/exit"}`. Exact-match means *no substring matching*.

## Rationale

- **Single-user CLI optimizes for keystrokes.** Slash-only adds 2 chars per exit; bare-word costs nothing in clarity for solo use.
- **Exact-match removes the substring ambiguity** that motivated slash-only in larger products. `"exit the loop"` is 4 tokens; it can't possibly match `_EXIT_INPUTS`.
- **Slash variants stay** so muscle memory from Claude Code / Aider users carries over.
- **Vim refugees get `:q`** — same idea, costs one line of code.

## Consequences

- **No way to accidentally exit on a task that legitimately uses the word `exit`** — the exact-match guard catches this.
- **`bye` stays in the chat path**, not the exit path. Typing `bye` triggers fast-chat → friendly response → user follows with `exit` or Ctrl+C if they actually want to leave.
- **`/quit` is documented as the primary exit** in `SLASH_COMMANDS` so help output is unambiguous.

## Alternatives considered

- **Slash-only** (Claude Code's approach). Rejected: too cautious for a personal CLI. The friction is unjustified by single-user scale.
- **Substring matching on `exit`** — Rejected: would catch `exit the loop in main.py` and break legitimate tasks.
- **Confirmation prompt before exit on ambiguous input** — Rejected: more friction, not less. Exact-match removes the ambiguity entirely.
