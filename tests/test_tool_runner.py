import json

import pytest

from agent.tool_runner import ToolRunner
from agent.tools import FileTools


def _tc(name: str, args: dict, call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.fixture
def runner(tmp_path):
    return ToolRunner(FileTools(tmp_path))


def test_read_file_returns_content(runner, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    result = runner.dispatch(_tc("read_file", {"path": "a.py"}))
    assert "x = 1" in result


def test_read_file_not_found(runner):
    result = runner.dispatch(_tc("read_file", {"path": "missing.py"}))
    assert "not found" in result.lower()


def test_list_files(runner, tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    result = runner.dispatch(_tc("list_files", {"directory": "."}))
    assert "a.py" in result
    assert "b.py" in result


def test_search_text_hit(runner, tmp_path):
    (tmp_path / "code.py").write_text("def hello():\n    pass\n")
    result = runner.dispatch(_tc("search_text", {"query": "hello"}))
    assert "code.py" in result
    assert "hello" in result


def test_search_text_no_matches(runner, tmp_path):
    (tmp_path / "code.py").write_text("x = 1\n")
    result = runner.dispatch(_tc("search_text", {"query": "nonexistent_xyz"}))
    assert "No matches" in result


def test_create_file_records_edit(runner):
    result = runner.dispatch(_tc("create_file", {"path": "new.py", "content": "x = 1"}))
    assert "Recorded create" in result
    assert len(runner.edits) == 1
    assert runner.edits[0].action == "create"
    assert runner.edits[0].path == "new.py"
    assert runner.edits[0].content == "x = 1"


def test_edit_file_records_search_replace(runner):
    result = runner.dispatch(
        _tc(
            "edit_file",
            {"path": "foo.py", "search": "old", "replace": "new"},
        )
    )
    assert "Recorded edit" in result
    assert len(runner.edits) == 1
    assert runner.edits[0].action == "search_replace"
    assert runner.edits[0].search == "old"
    assert runner.edits[0].replace == "new"


def test_replace_file_records_rewrite(runner):
    result = runner.dispatch(
        _tc("replace_file", {"path": "foo.py", "content": "x = 2"})
    )
    assert "Recorded rewrite" in result
    assert runner.edits[0].action == "rewrite"


def test_run_command_records_command(runner):
    result = runner.dispatch(_tc("run_command", {"command": "pytest -q"}))
    assert "Recorded command" in result
    assert runner.commands == ["pytest -q"]


def test_unknown_tool_returns_error(runner):
    result = runner.dispatch(_tc("bogus_tool", {}))
    assert "unknown tool" in result.lower()


def test_bad_json_args_returns_error(runner):
    tc = {
        "id": "x",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{not valid json"},
    }
    result = runner.dispatch(tc)
    assert "could not parse" in result.lower()


def test_multiple_edits_accumulate(runner):
    runner.dispatch(_tc("create_file", {"path": "a.py", "content": "1"}))
    runner.dispatch(_tc("create_file", {"path": "b.py", "content": "2"}))
    assert len(runner.edits) == 2
    assert runner.edits[0].path == "a.py"
    assert runner.edits[1].path == "b.py"
