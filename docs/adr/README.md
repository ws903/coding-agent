# Architecture Decision Records

Lightweight records of the non-obvious design decisions baked into this codebase. Format: [MADR](https://github.com/adr/madr), one decision per file.

Read these when you're considering a change that touches the same surface — the rationale lives here, not just in git log.

| # | Decision |
|---|---|
| [0001](0001-qwen3.6-as-default-model.md) | qwen3.6:35b as the default model |
| [0002](0002-native-tool-calling-over-text-parsing.md) | Native tool calling instead of regex-parsed edit blocks |
| [0003](0003-json-structured-planner-output.md) | JSON-structured planner output |
| [0004](0004-filesystem-discovered-managers.md) | Filesystem-discovered managers for skills/agents/MCP |
| [0005](0005-hybrid-exit-strategy.md) | Hybrid bare-word + slash exit |
| [0006](0006-llm-intent-classifier.md) | LLM-based intent classifier for ambiguous input |
| [0007](0007-planner-no-think-default.md) | Planner runs with `think:false` by default (~7× speedup) |
