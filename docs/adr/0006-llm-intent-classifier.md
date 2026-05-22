# ADR 0006 — LLM-based intent classifier for ambiguous input

* Status: Accepted
* Date: 2026-05-22

## Context

Trivial chat input ("hi", "how are you", "what's up") was triggering the full planner pipeline — system prompt + project context + JSON-structured output + the model's full thinking phase. End-to-end ~10–13s for what should be a one-line response.

A first pass added a keyword-list heuristic (`_looks_like_chat` matching the first word of short inputs against `_CHAT_TOKENS = {"hi", "hello", "hey", ...}`). It caught `hi` but not `how are you` or `what's up`. The user reported the gap.

Keyword expansion gets messy fast — "how" starts both `"how are you"` (chat) and `"how does the auth module work?"` (legit codebase question). Substring matching can't disambiguate without semantic understanding.

## Decision

**Three-tier routing** for REPL input:

| Tier | Trigger | Behavior |
|---|---|---|
| 1 | Obvious chat (`_CHAT_TOKENS` heuristic, ≤3 words starting with known greeting) | Fast-chat path, 0 overhead |
| 2 | Long input (>10 words) | Planner path, 0 overhead |
| 3 | Short ambiguous input (none of the above) | LLM classifier via `quick_chat_stream` (~1s with `think:false`) → routes to tier 1 or planner |

The classifier uses `LLMClient.quick_chat_stream` (native `/api/chat` with `think: false`, no project context, no JSON schema overhead) and asks for a single-word reply: `CHAT` or `TASK`. Failure modes (network error, parse error) default to `TASK` so real work never silently gets dropped into fast-chat.

## Rationale

- **Heuristics alone hit a ceiling.** Keyword lists can't disambiguate "how does this codebase work" from "how are you". The LLM can.
- **Tier ordering keeps the common cases free.** Most inputs are obvious one way or the other; only short ambiguous middle-ground pays the ~1s classifier cost.
- **`quick_chat_stream` is cheap.** Native `/api/chat` with `think:false` skips qwen3.6's reasoning phase — measured ~1s warm vs ~10s through the planner.
- **Failsafe to TASK.** A misclassification that sends real work to fast-chat is much worse than the inverse — fast-chat has no tools, no file access. Default to the safer path on any uncertainty.

## Consequences

- **One extra LLM call per ambiguous turn.** ~1s overhead on short-but-not-obviously-chat input. Not measurable on long inputs (skipped) or obvious chat (skipped).
- **`_CHAT_TOKENS` heuristic stays.** It's the zero-cost fast path for the most common case ("hi", "thanks", etc.).
- **Classifier prompt is in `cli.py`** (`_CLASSIFIER_PROMPT`). Could move to `prompts/` if it grows; small enough to inline for now.

## Trade-offs

| Scenario | Before | After |
|---|---|---|
| `hi` (heuristic catches) | ~10s planner | ~1s fast-chat |
| `how are you` (classifier routes) | ~10s planner | ~2s (classifier + fast-chat) |
| `add a docstring` (planner) | ~10s | ~11s (1s classifier overhead) |
| `add docstrings to every function and run tests` (>10 words, skipped) | ~10s | ~10s |

The 1s classifier tax on short tasks is acceptable; the chat speedup is the win.

## Alternatives considered

- **More keywords.** Adding "how", "what's", "good" as prefix tokens overshoots — false-matches on real questions about the codebase.
- **Make the planner faster** (run it with `think:false`). Rejected: the planner *needs* thinking for non-trivial tasks. Disabling it would hurt plan quality.
- **Run the classifier in parallel with the planner** and abort whichever path the classifier doesn't pick. Rejected: implementation complexity, GPU contention, hard to reason about.
- **No classifier, full planner for everything.** Status quo before this ADR. Rejected: 10s for "hi" is the literal user complaint that triggered this.
