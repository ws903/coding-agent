# tests/test_orchestrator.py
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestrator import Orchestrator
from agent.models import (
    Plan, Step, ExecutionResult, FileEdit,
    VerificationResult, CommandResult,
)


def make_plan(num_steps=2):
    steps = [
        Step(id=i + 1, action=f"Step {i + 1}", files_needed=[], verify_command="echo ok")
        for i in range(num_steps)
    ]
    return Plan(goal="Test goal", steps=steps)


def make_success_result():
    return ExecutionResult(
        file_edits=[FileEdit(path="test.py", action="create", content="pass")],
        commands=[],
        explanation="done",
    )


def make_verification_pass():
    return VerificationResult(passed=True, details=[])


def make_verification_fail():
    return VerificationResult(
        passed=False,
        details=[CommandResult(cmd="pytest", exit_code=1, stdout="", stderr="FAILED")],
    )


@pytest.fixture
def orchestrator(tmp_path):
    mock_planner = AsyncMock()
    mock_planner.generate_plan = AsyncMock(return_value=make_plan())
    mock_planner.replan = AsyncMock(return_value=make_plan(1))

    mock_executor = AsyncMock()
    mock_executor.execute = AsyncMock(return_value=make_success_result())

    mock_verifier = MagicMock()
    mock_verifier.run = MagicMock(return_value=make_verification_pass())

    mock_tools = MagicMock()
    mock_tools.write_file = MagicMock()
    mock_tools.edit_file = MagicMock(return_value=True)
    mock_tools.read_file = MagicMock(return_value="file contents")
    mock_tools.list_files = MagicMock(return_value=["a.py", "b.py"])
    mock_tools.sandbox = MagicMock()
    mock_tools.sandbox.run_command = MagicMock(return_value=CommandResult(cmd="echo", exit_code=0, stdout="", stderr=""))

    mock_db = MagicMock()
    mock_db.create_conversation = MagicMock(return_value="conv123")
    mock_db.add_message = MagicMock()
    mock_db.save_plan = MagicMock()
    mock_db.save_edit = MagicMock()
    mock_db.update_conversation_status = MagicMock()
    mock_db.get_messages = MagicMock(return_value=[])

    orch = Orchestrator(
        planner=mock_planner,
        executor=mock_executor,
        verifier=mock_verifier,
        tools=mock_tools,
        db=mock_db,
        project_root=tmp_path,
    )
    return orch


@pytest.mark.asyncio
async def test_run_generates_plan(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.planner.generate_plan.assert_called_once()


@pytest.mark.asyncio
async def test_run_executes_all_steps(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.executor.execute.call_count == 2


@pytest.mark.asyncio
async def test_run_verifies_after_each_step(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.verifier.run.call_count == 2


@pytest.mark.asyncio
async def test_run_applies_file_creates(orchestrator, tmp_path):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.tools.write_file.assert_called()


@pytest.mark.asyncio
async def test_run_retries_on_verify_failure(orchestrator):
    orchestrator.verifier.run = MagicMock(
        side_effect=[make_verification_fail(), make_verification_pass()] * 2
    )
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.executor.execute.call_count == 4  # 2 steps x (1 attempt + 1 retry)


@pytest.mark.asyncio
async def test_run_replans_after_two_failures(orchestrator):
    orchestrator.verifier.run = MagicMock(return_value=make_verification_fail())
    orchestrator.planner.replan = AsyncMock(return_value=make_plan(1))

    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.planner.replan.call_count >= 1


@pytest.mark.asyncio
async def test_run_returns_success_on_completion(orchestrator):
    result = await orchestrator.run("Fix the bug", mode="autonomous")
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_run_saves_to_db(orchestrator):
    await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.db.create_conversation.assert_called_once()
    orchestrator.db.save_plan.assert_called()
    orchestrator.db.update_conversation_status.assert_called()
