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
    success = tools.edit_file(
        "code.py", "def hello():\n    return 'hi'", "def hello():\n    return 'bye'"
    )
    assert success
    content = (tmp_path / "code.py").read_text()
    assert "bye" in content


def test_edit_file_whitespace_preserves_relative_indent(tmp_path):
    original = (
        "class Foo:\n    def bar(self):\n        if True:\n            return 1\n"
    )
    (tmp_path / "code.py").write_text(original)
    tools = make_tools(tmp_path)
    success = tools.edit_file(
        "code.py",
        "def bar(self):\n    if True:\n        return 1",
        "def bar(self):\n    if True:\n        return 2\n    else:\n        return 0",
    )
    assert success
    content = (tmp_path / "code.py").read_text()
    lines = content.splitlines()
    assert lines[1] == "    def bar(self):"
    assert lines[2] == "        if True:"
    assert lines[3] == "            return 2"
    assert lines[4] == "        else:"
    assert lines[5] == "            return 0"


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


def test_edit_file_not_found(tmp_path):
    """edit_file raises FileNotFoundError when file doesn't exist."""
    tools = make_tools(tmp_path)
    with pytest.raises(FileNotFoundError):
        tools.edit_file("nonexistent.py", "old", "new")


def test_list_files_non_directory(tmp_path):
    """list_files returns empty list when path is not a directory."""
    (tmp_path / "afile.txt").write_text("content")
    tools = make_tools(tmp_path)
    result = tools.list_files("afile.txt")
    assert result == []


def test_search_text_skips_hidden_dirs(tmp_path):
    """search_text skips files in hidden directories."""
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("hello\n")
    (tmp_path / "visible.py").write_text("hello\n")
    tools = make_tools(tmp_path)
    results = tools.search_text("hello")
    files = [r["file"] for r in results]
    assert any("visible.py" in f for f in files)
    assert not any(".hidden" in f for f in files)


def test_search_text_path_filter_skips_non_matching(tmp_path):
    """search_text with path_filter skips non-matching files."""
    (tmp_path / "a.py").write_text("hello\n")
    (tmp_path / "b.txt").write_text("hello\n")
    tools = make_tools(tmp_path)
    results = tools.search_text("hello", path_filter="*.py")
    files = [r["file"] for r in results]
    assert any("a.py" in f for f in files)
    assert not any("b.txt" in f for f in files)


def test_search_text_handles_unicode_decode_error(tmp_path):
    """search_text skips files that can't be decoded."""
    (tmp_path / "binary.bin").write_bytes(b"\x80\x81\x82\x83")
    (tmp_path / "text.py").write_text("hello\n")
    tools = make_tools(tmp_path)
    results = tools.search_text("hello")
    # Should not crash, and should find text.py
    assert any("text.py" in r["file"] for r in results)


def test_search_text_handles_permission_error(tmp_path):
    """search_text skips files with permission errors."""
    import os

    restricted = tmp_path / "noperm.py"
    restricted.write_text("hello\n")
    os.chmod(str(restricted), 0o000)
    (tmp_path / "readable.py").write_text("hello\n")
    tools = make_tools(tmp_path)
    try:
        results = tools.search_text("hello")
        # Should not crash
        assert any("readable.py" in r["file"] for r in results)
    finally:
        os.chmod(str(restricted), 0o644)
