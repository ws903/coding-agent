# ADR 0007 — Planner runs with `think:false` by default

* Status: Accepted
* Date: 2026-05-22

## Context

`qwen3.6:35b` (the default model — see [ADR 0001](0001-qwen3.6-as-default-model.md)) is a thinking model. On a single 5070 Ti with MoE RAM-offload, its reasoning phase is the dominant latency source:

- Cloud thinking models typically add 1–3s of latency
- On local 16 GB VRAM + DDR5 RAM offload (this project's reference hardware): **measured 104s of first-token delay and 126s total** for a single planner call

The 9× bandwidth gap between VRAM (~896 GB/s GDDR7) and DDR5 (~96 GB/s) means every reasoning token is memory-bound. The GPU stays at ~20% utilization while waiting on expert weights to stream in.

For the planner specifically, this hits every interaction: even codebase questions ("what does this project do?") take ~2 minutes because they detour through the planner returning `kind: "answer"`.

## Decision

The Planner calls `LLMClient.chat_json(messages, think=False)` by default. The new `think` parameter routes the call to Ollama's native `/api/chat` endpoint with `think: false` + `format: "json"` instead of the OpenAI-compat `/v1/chat/completions` endpoint with `response_format: {"type": "json_object"}` (the OpenAI-compat path silently ignores `think:false`; only the native endpoint actually disables reasoning).

Users can opt back into thinking-mode planning via the `AGENT_PLANNER_THINK` env var:

```
AGENT_PLANNER_THINK=true uv run agent --auto --task "refactor the auth module to use bcrypt"
```

The env var accepts `1`, `true`, `yes`, `True`, `YES` (case-insensitive). Anything else (including unset) keeps thinking off.

## Rationale

### Live measurement on this project's hardware

| Config | Total | First content | Output |
|---|---|---|---|
| qwen3.6:35b `/v1` thinking ON (previous default) | 126.4s | 104.2s | 1428 chars |
| qwen3.6:35b native `think:false` (new default) | **17.5s** | **0.6s** | 874 chars |

**~7× speedup, visible progress immediately instead of staring at a frozen prompt for ~100s.**

### Research backs "default off" for planning tasks

A [2026 benchmark of 15 LLMs on 38 real coding tasks](https://ianlpaterson.com/blog/llm-benchmark-2026-38-actual-tasks-15-models-for-2-29/) found:

> "Planning and Code are essentially solved at 0-1% failure rates across all 15 models."
>
> "High-end reasoner models are needed for multi-step causal chains... cheaper models drop to 60-80% on those."

Thinking helps on intricate refactors with multi-step causal dependencies — not on routine planning. Most planner invocations are routine; the env var covers the rest.

### Industry pattern

Qwen3 ships with [three explicit modes](https://qwenlm.github.io/blog/qwen3/): `non-think` (fast, intuitive), `think-high` (deliberate), `think-max` (maximum reasoning). The pattern Alibaba ships and Anthropic mirrors is **default to fast, opt-in to thinking for hard tasks**.

[Production lens, Augment Code 2026 routing guide](https://www.augmentcode.com/guides/ai-model-routing-guide):

> "In real-world agent workflows, consistency, latency, and usability often matter more than peak reasoning."

### Why bypass the OpenAI-compat path

Verified live: `think: false` passed to `/v1/chat/completions` has **zero effect** on Ollama. The model still emits its full reasoning pass (~120 chunks of `delta.reasoning` for "hi"). Only the native `/api/chat` endpoint honors the flag. So this ADR is also a small protocol-layering decision: structured planner output now uses native Ollama for the no-think path, OpenAI-compat for the think path.

## Consequences

- **Planner runs ~7× faster on the reference hardware.** Visible content starts in <1s, total typically 5-20s.
- **Plan quality drops slightly on complex multi-step tasks.** The 15-LLM benchmark suggests this is marginal for most plans. The env var covers the cases where it matters.
- **Token usage tracking is preserved** by translating Ollama's `prompt_eval_count`/`eval_count` → OpenAI-style `prompt_tokens`/`completion_tokens` inside `_chat_json_native_no_think`.
- **The native path is non-retried** (no transport-level retry on connection errors). Justified: cheap call, transient failures bubble up to the planner's parse-retry loop anyway.
- **`chat_json` is now a thin dispatcher** that picks endpoints based on the `think` flag. Each path is small and self-contained.

## Alternatives considered

- **Thinking budget instead of binary off** — Qwen3 supports limited-budget thinking (e.g. 500 reasoning tokens max). Would give a middle ground. Skipped for now: binary is simpler, the speedup is huge, and the env var covers the few cases where partial thinking helps. Revisit if quality regressions surface.
- **Three-way intent routing (chat/question/task)** — Add a "codebase question" lane that skips the planner entirely. Considered and rejected: the planner already handles `kind: "answer"` correctly, and the slowness wasn't from routing — it was from thinking. Making the planner fast is the right fix; adding a router in front of an unchanged planner would do the same thing with more plumbing.
- **Always stream the planner** including reasoning tokens — would surface progress but not change wall-clock time. The user reported "too slow," not "looks frozen." This ADR fixes the underlying speed; a future streaming pass is independent.
- **Switch planner to a smaller dense model** (e.g. `qwen3:14b`) — would avoid MoE offload entirely, possibly faster still. But adds model-swap latency between planner→executor (5-15s per switch) and a second model to keep pulled. Higher complexity, less clear win.

## Sources

- [Live benchmark of think:true vs think:false on this hardware](https://github.com/ws903/coding-agent/pull/36)
- [I Tested 15 LLMs on 38 Real Coding Tasks (2026)](https://ianlpaterson.com/blog/llm-benchmark-2026-38-actual-tasks-15-models-for-2-29/)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Augment Code 2026 routing guide](https://www.augmentcode.com/guides/ai-model-routing-guide)
- [Ollama API docs — `think` parameter](https://github.com/ollama/ollama/blob/main/docs/api.md)
