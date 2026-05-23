from agent.env.codebase_index import IGNORED_DIRS, CodebaseIndex


def test_empty_project(tmp_path):
    idx = CodebaseIndex(tmp_path)
    assert idx.entries == []
    assert idx.summary() == ""


def test_extracts_functions_and_classes(tmp_path):
    (tmp_path / "a.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def helper(x):\n"
        "    return x + 1\n"
        "\n"
        "class Foo:\n"
        "    pass\n"
    )
    idx = CodebaseIndex(tmp_path)
    assert len(idx.entries) == 1
    entry = idx.entries[0]
    assert entry.path == "a.py"
    kinds = [(s.kind, s.name) for s in entry.symbols]
    assert ("function", "helper") in kinds
    assert ("class", "Foo") in kinds
    assert ("import", "os") in kinds
    assert ("import", "pathlib") in kinds


def test_imports_deduplicated_and_sorted(tmp_path):
    (tmp_path / "a.py").write_text(
        "import sys\nimport os\nimport sys\n\ndef f(): pass\n"
    )
    idx = CodebaseIndex(tmp_path)
    imports = [s.name for s in idx.entries[0].symbols if s.kind == "import"]
    # sorted, deduped
    assert imports == ["os", "sys"]


def test_ignores_hidden_and_common_dirs(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("def hidden(): pass\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "mod.py").write_text("def nope(): pass\n")
    (tmp_path / "real.py").write_text("def visible(): pass\n")

    idx = CodebaseIndex(tmp_path)
    paths = [e.path for e in idx.entries]
    assert paths == ["real.py"]


def test_malformed_python_skipped(tmp_path):
    (tmp_path / "broken.py").write_text("def oops(\n")  # syntax error
    (tmp_path / "ok.py").write_text("def good(): pass\n")

    idx = CodebaseIndex(tmp_path)
    paths = [e.path for e in idx.entries]
    assert paths == ["ok.py"]


def test_files_with_no_symbols_skipped(tmp_path):
    (tmp_path / "blank.py").write_text("# only a comment\n")
    (tmp_path / "ok.py").write_text("def hi(): pass\n")
    idx = CodebaseIndex(tmp_path)
    assert [e.path for e in idx.entries] == ["ok.py"]


def test_summary_renders_markdown(tmp_path):
    (tmp_path / "main.py").write_text("def start(): pass\nclass App: pass\n")
    summary = CodebaseIndex(tmp_path).summary()
    assert "## Codebase symbol map" in summary
    assert "main.py" in summary
    assert "`start`" in summary
    assert "`App`" in summary


def test_async_functions_recognized(tmp_path):
    (tmp_path / "a.py").write_text("async def aio(): pass\n")
    idx = CodebaseIndex(tmp_path)
    assert [s.kind for s in idx.entries[0].symbols] == ["function"]
    assert idx.entries[0].symbols[0].name == "aio"


def test_ignored_dirs_include_agent_state():
    """Don't index files inside the agent's own state directory."""
    assert ".agent" in IGNORED_DIRS
    assert ".git" in IGNORED_DIRS
