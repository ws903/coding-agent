from pathlib import Path

import pytest

from agent.verifier import Verifier
from agent.sandbox import Sandbox


def test_verify_passes_when_commands_succeed(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["echo ok"])
    result = verifier.run()
    assert result.passed
    assert len(result.details) == 1
    assert result.details[0].exit_code == 0


def test_verify_fails_when_command_fails(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["python3 -c \"raise SystemExit(1)\""])
    result = verifier.run()
    assert not result.passed


def test_verify_runs_all_commands(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["echo one", "echo two", "echo three"])
    result = verifier.run()
    assert result.passed
    assert len(result.details) == 3


def test_verify_fails_if_any_command_fails(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(
        sandbox,
        commands=["echo ok", "python3 -c \"raise SystemExit(1)\"", "echo ok"],
    )
    result = verifier.run()
    assert not result.passed
    assert len(result.details) == 3


def test_verify_no_commands_passes(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=[])
    result = verifier.run()
    assert result.passed
    assert len(result.details) == 0


def test_verify_with_step_command(tmp_path):
    sandbox = Sandbox(tmp_path)
    verifier = Verifier(sandbox, commands=["echo global"])
    result = verifier.run(step_command="echo step")
    assert result.passed
    assert len(result.details) == 2
