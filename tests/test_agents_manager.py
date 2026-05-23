from agent.extensions.agents import AgentsManager


def test_manager_with_no_agents_dir(tmp_path):
    mgr = AgentsManager(tmp_path)
    assert mgr.roles == []
    assert mgr.catalog_section() == ""


def test_manager_loads_single_file_role(tmp_path):
    agents_dir = tmp_path / ".agent" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "code-review.md").write_text(
        "---\nname: code-review\ndescription: Review the diff\n---\n"
        "You are a code reviewer. Be thorough."
    )
    mgr = AgentsManager(tmp_path)
    assert [r.name for r in mgr.roles] == ["code-review"]
    role = mgr.get("code-review")
    assert role is not None
    assert role.description == "Review the diff"
    assert "thorough" in role.system_prompt


def test_manager_loads_directory_role(tmp_path):
    agents_dir = tmp_path / ".agent" / "agents" / "debug"
    agents_dir.mkdir(parents=True)
    (agents_dir / "AGENT.md").write_text(
        "---\nname: debug\ndescription: Investigate failures\n---\n"
        "Approach bugs scientifically."
    )
    mgr = AgentsManager(tmp_path)
    role = mgr.get("debug")
    assert role is not None
    assert role.description == "Investigate failures"
    assert "scientifically" in role.system_prompt


def test_manager_skips_role_dir_without_AGENT_md(tmp_path):
    role_dir = tmp_path / ".agent" / "agents" / "broken"
    role_dir.mkdir(parents=True)
    (role_dir / "README.md").write_text("not the right name")
    mgr = AgentsManager(tmp_path)
    assert mgr.get("broken") is None


def test_catalog_section_lists_each_role(tmp_path):
    agents_dir = tmp_path / ".agent" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "a.md").write_text("---\nname: a\ndescription: alpha\n---\nbody")
    (agents_dir / "b.md").write_text("---\nname: b\ndescription: beta\n---\nbody")
    mgr = AgentsManager(tmp_path)
    section = mgr.catalog_section()
    assert "## Available subagents" in section
    assert "**a** -- alpha" in section
    assert "**b** -- beta" in section
    assert "spawn_agent" in section
