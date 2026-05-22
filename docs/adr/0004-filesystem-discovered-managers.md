# ADR 0004 — Filesystem-discovered managers for skills, agents, and MCP

* Status: Accepted
* Date: 2026-05-22

## Context

The agent grew three orthogonal extensibility points in mid-2026:

1. **Skills** — reusable executor instructions (Claude-Code-style)
2. **Subagents** — focused role-based sub-LLM calls with their own context
3. **MCP servers** — external tool providers via the Model Context Protocol

Each could live as Python plugin classes, decorator-registered modules, an entry-point system, or filesystem-discovered files. Three independent additions in close succession risks one becoming the load-bearing pattern by accident.

## Decision

All three are **filesystem-discovered managers** following a shared shape:

| Manager | Source | Layout |
|---|---|---|
| `SkillsManager` | `<project>/.agent/skills/` | `name.md` or `name/SKILL.md` |
| `AgentsManager` | `<project>/.agent/agents/` | `name.md` or `name/AGENT.md` |
| `MCPManager` | `<project>/.mcp.json` (Claude Code format) | JSON config that spawns external processes |

Each scans the source at agent startup, parses lightweight frontmatter (skills/agents) or JSON (MCP), and exposes a `catalog_section()` that gets injected into the executor's system prompt as a one-line summary per item. The full body is loaded on demand via a dedicated tool (`read_skill`, `spawn_agent`) — **progressive disclosure** so the context stays small until something is actually used.

## Rationale

- **User-extensible without code changes.** The user drops a markdown file in `.agent/skills/` and the agent picks it up. Matches Claude Code / OpenCode / Aider authoring patterns.
- **One mental model.** Same discovery shape, same frontmatter parser (shared via `skills_manager._parse_frontmatter`), same catalog injection convention. Reading one manager teaches you all three.
- **Progressive disclosure controls token cost.** N skills cost N description lines in context, not N full bodies. Empirically a 5-line skill description gets the model to call `read_skill(...)` correctly the ~10-line full body is needed.
- **MCP follows the same model even though it's external.** A `.mcp.json` server is "discovered" the same way: read config at startup, fetch tools, expose them. The fact that the server is an external process is an implementation detail.

## Consequences

- **Authoring is plain markdown.** Skills/agents are markdown files with YAML-ish frontmatter (no PyYAML dep — manual `key: value` parse).
- **No central registry.** Adding a skill is `mkdir .agent/skills && echo ... > my-skill.md`. No imports, no decorators, no setup hooks.
- **Subagents inherit the same parser.** `agents_manager._load_one` imports `skills_manager._parse_frontmatter`. If frontmatter needs to evolve, it changes in one place.
- **MCP servers are namespaced** (`mcp__<server>__<tool>`) to avoid name collisions with built-in tools or each other. The ToolRunner routes by prefix.
- **Subagents have a restricted tool set** (`read_file`, `list_files`, `search_text` only). Enforced in `SubagentRunner.run`. Prevents recursive `spawn_agent` and any edit / shell side effects. Documented in ADR-adjacent code comments in `subagent_runner.py`.

## Alternatives considered

- **Python plugins via entry points** — Standard, type-safe, IDE-friendly. Rejected: forces users to write Python and reinstall to add a skill. Slower iteration loop.
- **Single combined config file** (e.g. `.agent/config.toml` declaring everything) — Rejected: skills/agents/MCP have different shapes; combining them in one schema would be a worse mental model.
- **Code-based subagent roles** (decorator-registered Python classes) — Rejected for symmetry with skills. If a power user needs Python-defined logic in a subagent later, can be added as a separate "code-skill" path; the markdown path stays the default.
