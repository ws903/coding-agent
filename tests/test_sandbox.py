# tests/test_sandbox.py

import pytest

from agent.safety.command_policy import CommandBlocked
from agent.safety.sandbox import Sandbox, SecurityError


def test_validate_path_within_root(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.validate_path("src/main.py")
    assert result == tmp_path / "src" / "main.py"


def test_validate_path_rejects_escape(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SecurityError):
        sandbox.validate_path("../../etc/passwd")


def test_validate_path_rejects_absolute(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SecurityError):
        sandbox.validate_path("/etc/passwd")


def test_validate_path_resolves_dots(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.validate_path("src/../src/main.py")
    assert result == tmp_path / "src" / "main.py"


def test_run_command_captures_output(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_run_command_captures_stderr(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command("python3 -c \"import sys; sys.stderr.write('err')\"")
    assert "err" in result.stderr


def test_run_command_nonzero_exit(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command('python3 -c "raise SystemExit(42)"')
    assert result.exit_code == 42


def test_run_command_timeout(tmp_path):
    sandbox = Sandbox(tmp_path, timeout=1)
    result = sandbox.run_command('python3 -c "import time; time.sleep(10)"')
    assert result.exit_code != 0
    assert "timeout" in result.stderr.lower() or "timed out" in result.stderr.lower()


def test_run_command_cwd_is_project_root(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command('python3 -c "import os; print(os.getcwd())"')
    assert result.exit_code == 0
    assert str(tmp_path) in result.stdout


def test_run_command_blocks_dangerous_command(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(CommandBlocked):
        sandbox.run_command("rm -rf /")


def test_run_command_blocks_sudo(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(CommandBlocked):
        sandbox.run_command("sudo apt-get install git")


def test_run_command_allows_safe_command(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_command("echo safe")
    assert result.exit_code == 0
    assert "safe" in result.stdout
