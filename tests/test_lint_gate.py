from unittest.mock import patch, MagicMock

import pytest

from agent.safety.lint_gate import LintGate, LintError


@pytest.fixture
def gate(tmp_path):
    with patch("agent.safety.lint_gate.shutil.which", return_value="/usr/bin/ruff"):
        return LintGate(tmp_path)


@pytest.fixture
def py_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    return f


def test_available_when_ruff_found(tmp_path):
    with patch("agent.safety.lint_gate.shutil.which", return_value="/usr/bin/ruff"):
        g = LintGate(tmp_path)
    assert g.available is True


def test_not_available_when_ruff_missing(tmp_path):
    with patch("agent.safety.lint_gate.shutil.which", return_value=None):
        g = LintGate(tmp_path)
    assert g.available is False


def test_check_file_skips_non_python(gate, tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Hello")
    assert gate.check_file("readme.md") == []


def test_check_file_skips_missing(gate):
    assert gate.check_file("no_such.py") == []


def test_check_file_returns_errors(gate, py_file):
    mock_result = MagicMock()
    mock_result.stdout = '[{"code":"F821","message":"Undefined name","location":{"row":1,"column":5},"end_location":{"row":1,"column":10},"filename":"test.py","fix":null,"cell":null,"noqa_row":1,"severity":"error","url":"https://example.com"}]'
    with patch("agent.safety.lint_gate.subprocess.run", return_value=mock_result):
        errors = gate.check_file("test.py")
    assert len(errors) == 1
    assert errors[0].code == "F821"
    assert errors[0].row == 1
    assert errors[0].message == "Undefined name"


def test_check_file_returns_empty_on_no_errors(gate, py_file):
    mock_result = MagicMock()
    mock_result.stdout = "[]"
    with patch("agent.safety.lint_gate.subprocess.run", return_value=mock_result):
        errors = gate.check_file("test.py")
    assert errors == []


def test_check_file_handles_timeout(gate, py_file):
    import subprocess

    with patch(
        "agent.safety.lint_gate.subprocess.run",
        side_effect=subprocess.TimeoutExpired("ruff", 30),
    ):
        errors = gate.check_file("test.py")
    assert errors == []


def test_check_file_not_available(tmp_path, py_file):
    with patch("agent.safety.lint_gate.shutil.which", return_value=None):
        g = LintGate(tmp_path)
    assert g.check_file("test.py") == []


def test_gate_edit_passes_when_no_errors(gate):
    with patch.object(gate, "check_file", return_value=[]):
        result = gate.gate_edit("test.py")
    assert result.passed is True


def test_gate_edit_passes_for_non_python(gate):
    result = gate.gate_edit("style.css")
    assert result.passed is True


def test_gate_edit_detects_new_errors(gate):
    new_error = LintError(
        file="test.py", row=5, col=1, code="F821", message="Undefined name `x`"
    )
    with patch.object(gate, "check_file", return_value=[new_error]):
        result = gate.gate_edit("test.py", before_errors=[])
    assert result.passed is False
    assert len(result.new_errors) == 1
    assert result.new_errors[0].code == "F821"


def test_gate_edit_ignores_preexisting_errors(gate):
    err = LintError(
        file="test.py", row=5, col=1, code="F821", message="Undefined name `x`"
    )
    with patch.object(gate, "check_file", return_value=[err]):
        result = gate.gate_edit("test.py", before_errors=[err])
    assert result.passed is True
    assert result.pre_existing == 1
    assert len(result.new_errors) == 0


def test_gate_edit_mixed_old_and_new(gate):
    old = LintError(file="test.py", row=3, col=1, code="F821", message="old error")
    new = LintError(file="test.py", row=10, col=1, code="F822", message="new error")
    with patch.object(gate, "check_file", return_value=[old, new]):
        result = gate.gate_edit("test.py", before_errors=[old])
    assert result.passed is False
    assert result.pre_existing == 1
    assert len(result.new_errors) == 1
    assert result.new_errors[0].code == "F822"


def test_gate_edit_no_before_errors_arg(gate):
    err = LintError(
        file="test.py", row=5, col=1, code="F821", message="Undefined name `x`"
    )
    with patch.object(gate, "check_file", return_value=[err]):
        result = gate.gate_edit("test.py")
    assert result.passed is False
    assert len(result.new_errors) == 1


def test_parse_output_bad_json(gate):
    errors = gate._parse_output("not json", "test.py")
    assert errors == []


def test_custom_rules(tmp_path):
    g = LintGate(tmp_path, rules=["F821"])
    assert g.rules == ["F821"]
