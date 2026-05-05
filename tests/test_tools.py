# tests/test_tools.py
from pathlib import Path

import pytest

from agent.tools import FileTools
from agent.sandbox import SecurityError


def make_tools(tmp_path: Path) -> FileTools:
    return FileTools(tmp_path)


def test_read_file(tmp_path):
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n")
    tools = make_tools(tmp_path)
    content = tools.read_file("hello.txt")
    assert "line1" in content
    assert "line2" in content


def test_read_file_with_range(tmp_path):
    lines = "\n".join(f"line{i}" for i in range(1, 21))
    (tmp_path / "big.txt").write_text(lines)
    tools = make_tools(tmp_path)
    content = tools.read_file("big.txt", start_line=5, end_line=7)
    assert "line5" in content
    assert "line7" in content
    assert "line4" not in content
    assert "line8" not in content


def test_read_file_not_found(tmp_path):
    tools = make_tools(tmp_path)
    with pytest.raises(FileNotFoundError):
        tools.read_file("nope.txt")


def test_write_file_creates(tmp_path):
    tools = make_tools(tmp_path)
    tools.write_file("new.txt", "hello world")
    assert (tmp_path / "new.txt").read_text() == "hello world"


def test_write_file_creates_dirs(tmp_path):
    tools = make_tools(tmp_path)
    tools.write_file("a/b/c.txt", "deep")
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_write_file_rejects_escape(tmp_path):
    tools = make_tools(tmp_path)
    with pytest.raises(SecurityError):
        tools.write_file("../../evil.txt", "bad")


def test_edit_file_search_replace(tmp_path):
    (tmp_path / "code.py").write_text("def hello():\n    return 'hi'\n")
    tools = make_tools(tmp_path)
    success = tools.edit_file("code.py", "return 'hi'", "return 'hello world'")
    assert success
    assert "hello world" in (tmp_path / "code.py").read_text()


def test_edit_file_no_match(tmp_path):
    (tmp_path / "code.py").write_text("def hello():\n    return 'hi'\n")
    tools = make_tools(tmp_path)
    success = tools.edit_file("code.py", "return 'bye'", "return 'hello world'")
    assert not success


def test_edit_file_whitespace_normalized(tmp_path):
    (tmp_path / "code.py").write_text("    def hello():\n        return 'hi'\n")
    tools = make_tools(tmp_path)
    success = tools.edit_file("code.py", "def hello():\n    return 'hi'", "def hello():\n    return 'bye'")
    assert success
    content = (tmp_path / "code.py").read_text()
    assert "bye" in content


def test_list_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    tools = make_tools(tmp_path)
    files = tools.list_files(".")
    assert "a.py" in files
    assert "b.txt" in files


def test_list_files_with_pattern(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    tools = make_tools(tmp_path)
    files = tools.list_files(".", pattern="*.py")
    assert "a.py" in files
    assert "b.txt" not in files


def test_search_text(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    pass\n")
    (tmp_path / "b.py").write_text("def world():\n    pass\n")
    tools = make_tools(tmp_path)
    results = tools.search_text("hello")
    assert len(results) >= 1
    assert any("a.py" in r["file"] for r in results)


def test_search_text_with_filter(tmp_path):
    (tmp_path / "a.py").write_text("hello\n")
    (tmp_path / "b.txt").write_text("hello\n")
    tools = make_tools(tmp_path)
    results = tools.search_text("hello", path_filter="*.py")
    assert all("a.py" in r["file"] for r in results)
