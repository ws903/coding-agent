from agent.extensions.skills import SkillsManager, _parse_frontmatter


def test_parse_frontmatter_full():
    text = """---
name: foo
description: Does foo things
extra: ignored
---
body content here
"""
    frontmatter, body = _parse_frontmatter(text)
    assert frontmatter == {
        "name": "foo",
        "description": "Does foo things",
        "extra": "ignored",
    }
    assert body == "body content here\n"


def test_parse_frontmatter_quoted_values():
    text = """---
name: "foo"
description: 'with quotes'
---
body
"""
    frontmatter, _ = _parse_frontmatter(text)
    assert frontmatter["name"] == "foo"
    assert frontmatter["description"] == "with quotes"


def test_parse_frontmatter_missing_returns_empty():
    text = "just a markdown body with no frontmatter"
    frontmatter, body = _parse_frontmatter(text)
    assert frontmatter == {}
    assert body == text


def test_parse_frontmatter_unterminated_returns_empty():
    text = "---\nname: oops\nno closing fence"
    frontmatter, body = _parse_frontmatter(text)
    assert frontmatter == {}
    assert body == text


def test_manager_with_no_skills_dir(tmp_path):
    mgr = SkillsManager(tmp_path)
    assert mgr.skills == []
    assert mgr.catalog_section() == ""


def test_manager_loads_single_file_skill(tmp_path):
    skill_dir = tmp_path / ".agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "code-review.md").write_text(
        "---\nname: code-review\ndescription: Review the diff\n---\nDo the review carefully.\n"
    )
    mgr = SkillsManager(tmp_path)
    assert [s.name for s in mgr.skills] == ["code-review"]
    skill = mgr.get("code-review")
    assert skill is not None
    assert skill.description == "Review the diff"
    assert skill.body == "Do the review carefully."


def test_manager_loads_directory_skill(tmp_path):
    skill_dir = tmp_path / ".agent" / "skills" / "debug"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: debug\ndescription: Systematic debugging\n---\n1. Reproduce. 2. Bisect.\n"
    )
    mgr = SkillsManager(tmp_path)
    skill = mgr.get("debug")
    assert skill is not None
    assert skill.description == "Systematic debugging"
    assert "Bisect" in skill.body


def test_manager_uses_filename_when_no_name(tmp_path):
    skill_dir = tmp_path / ".agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "my-skill.md").write_text(
        "---\ndescription: No explicit name\n---\nbody\n"
    )
    mgr = SkillsManager(tmp_path)
    assert mgr.get("my-skill") is not None


def test_manager_skips_skill_dir_without_SKILL_md(tmp_path):
    skill_dir = tmp_path / ".agent" / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "other.md").write_text("not a skill")
    mgr = SkillsManager(tmp_path)
    assert mgr.get("broken") is None


def test_catalog_section_lists_each_skill(tmp_path):
    skill_dir = tmp_path / ".agent" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "a.md").write_text("---\nname: a\ndescription: alpha\n---\n")
    (skill_dir / "b.md").write_text("---\nname: b\ndescription: beta\n---\n")
    mgr = SkillsManager(tmp_path)
    section = mgr.catalog_section()
    assert "## Available skills" in section
    assert "**a** -- alpha" in section
    assert "**b** -- beta" in section
    assert "read_skill" in section
