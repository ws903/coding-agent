# src/agent/agents_manager.py
"""Filesystem-based subagent role discovery.

A subagent role is a markdown file with YAML-ish frontmatter (name,
description) plus a body that becomes the role's system prompt. The
executor sees the catalog (one line per role) and spawns a role on
demand via the `spawn_agent` tool.

Subagents share the same LLM backend but run with a fresh context and
a read-only tool subset (no edits, no recursive spawn). They return
text that the executor can incorporate into its own reasoning.

Layouts under `.agent/agents/`:
  - `<name>.md`                       (single file)
  - `<name>/AGENT.md`                 (directory layout)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.extensions.skills import _parse_frontmatter


@dataclass(frozen=True)
class AgentRole:
    name: str
    description: str
    system_prompt: str
    source_path: Path


def _load_one(path: Path, default_name: str) -> AgentRole | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    frontmatter, body = _parse_frontmatter(text)
    name = frontmatter.get("name") or default_name
    description = frontmatter.get("description") or ""
    return AgentRole(
        name=name,
        description=description,
        system_prompt=body.strip(),
        source_path=path,
    )


class AgentsManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.agents_dir = project_root / ".agent" / "agents"
        self._roles: dict[str, AgentRole] = {}
        self._load()

    def _load(self) -> None:
        if not self.agents_dir.is_dir():
            return
        for entry in sorted(self.agents_dir.iterdir()):
            if entry.is_file() and entry.suffix == ".md":
                role = _load_one(entry, default_name=entry.stem)
            elif entry.is_dir():
                candidate = entry / "AGENT.md"
                if not candidate.is_file():
                    continue
                role = _load_one(candidate, default_name=entry.name)
            else:
                continue
            if role is not None:
                self._roles[role.name] = role

    @property
    def roles(self) -> list[AgentRole]:
        return sorted(self._roles.values(), key=lambda r: r.name)

    def get(self, name: str) -> AgentRole | None:
        return self._roles.get(name)

    def catalog_section(self) -> str:
        if not self._roles:
            return ""
        lines = ["## Available subagents", ""]
        for role in self.roles:
            desc = role.description or "(no description)"
            lines.append(f"- **{role.name}** -- {desc}")
        lines.append("")
        lines.append(
            "Call `spawn_agent(role=..., task=...)` to dispatch a focused "
            "subagent with a fresh context. Subagents are read-only and "
            "return text. Use them for code review, devil's advocate, or "
            "investigation -- not for making edits."
        )
        return "\n".join(lines)
