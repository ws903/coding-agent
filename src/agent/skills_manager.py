# src/agent/skills_manager.py
"""Filesystem-based skill discovery.

A skill is a markdown file with YAML-ish frontmatter that captures
reusable instructions for the executor. The executor sees only each
skill's `name` + `description` in its system prompt; it loads the full
body on demand via the `read_skill` tool. This is progressive disclosure
-- many small skills cost only a line each in context until used.

Layouts supported under `.agent/skills/`:
  - single file: `code-review.md`
  - directory:   `code-review/SKILL.md`  (lets you bundle resources alongside)

Frontmatter (only `name` and `description` are read; other keys are
ignored so future Claude Code-style fields don't break us):

  ---
  name: code-review
  description: Review the current diff for correctness, security, style.
  ---
  <markdown body>

If `name` is missing, the filename (or directory name) is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    source_path: Path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter_dict, body). Tolerant of missing frontmatter."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip()
    body_start = end + len("\n---")
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    body = text[body_start:]
    frontmatter: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    return frontmatter, body


def _load_one(path: Path, default_name: str) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    frontmatter, body = _parse_frontmatter(text)
    name = frontmatter.get("name") or default_name
    description = frontmatter.get("description") or ""
    return Skill(
        name=name, description=description, body=body.strip(), source_path=path
    )


class SkillsManager:
    """Discovers and serves skill files under `<project>/.agent/skills/`."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.skills_dir = project_root / ".agent" / "skills"
        self._skills: dict[str, Skill] = {}
        self._load()

    def _load(self) -> None:
        if not self.skills_dir.is_dir():
            return
        for entry in sorted(self.skills_dir.iterdir()):
            if entry.is_file() and entry.suffix == ".md":
                skill = _load_one(entry, default_name=entry.stem)
            elif entry.is_dir():
                candidate = entry / "SKILL.md"
                if not candidate.is_file():
                    continue
                skill = _load_one(candidate, default_name=entry.name)
            else:
                continue
            if skill is not None:
                self._skills[skill.name] = skill

    @property
    def skills(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def catalog_section(self) -> str:
        """Markdown section to inject into the executor system prompt."""
        if not self._skills:
            return ""
        lines = ["## Available skills", ""]
        for skill in self.skills:
            desc = skill.description or "(no description)"
            lines.append(f"- **{skill.name}** -- {desc}")
        lines.append("")
        lines.append(
            "If a skill applies to the current step, call "
            "`read_skill(name=...)` to load its full instructions, then follow them."
        )
        return "\n".join(lines)
