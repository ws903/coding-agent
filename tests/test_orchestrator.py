# tests/test_orchestrator.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.orchestrator import Orchestrator
from agent.models import (
    Answer,
    Plan,
    Step,
    ExecutionResult,
    FileEdit,
    VerificationResult,
    CommandResult,
)


def make_plan(num_steps=2):
    steps = [
        Step(
            id=i + 1, action=f"Step {i + 1}", files_needed=[], verify_command="echo ok"
        )
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
    mock_tools.sandbox.run_command = MagicMock(
        return_value=CommandResult(cmd="echo", exit_code=0, stdout="", stderr="")
    )

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
    await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.planner.generate_plan.assert_called_once()


@pytest.mark.asyncio
async def test_run_executes_all_steps(orchestrator):
    await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.executor.execute.call_count == 2


@pytest.mark.asyncio
async def test_run_verifies_after_each_step(orchestrator):
    await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.verifier.run.call_count == 2


@pytest.mark.asyncio
async def test_run_applies_file_creates(orchestrator, tmp_path):
    await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.tools.write_file.assert_called()


@pytest.mark.asyncio
async def test_run_retries_on_verify_failure(orchestrator):
    orchestrator.verifier.run = MagicMock(
        side_effect=[make_verification_fail(), make_verification_pass()] * 2
    )
    await orchestrator.run("Fix the bug", mode="autonomous")
    assert (
        orchestrator.executor.execute.call_count == 4
    )  # 2 steps x (1 attempt + 1 retry)


@pytest.mark.asyncio
async def test_run_replans_after_two_failures(orchestrator):
    orchestrator.verifier.run = MagicMock(return_value=make_verification_fail())
    orchestrator.planner.replan = AsyncMock(return_value=make_plan(1))

    await orchestrator.run("Fix the bug", mode="autonomous")
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


@pytest.mark.asyncio
async def test_run_aborts_when_plan_rejected(orchestrator):
    result = await orchestrator.run(
        "Fix the bug", mode="autonomous", approve_plan=lambda plan: False
    )
    assert result["status"] == "aborted"
    orchestrator.db.update_conversation_status.assert_called_with("conv123", "aborted")


@pytest.mark.asyncio
async def test_apply_edits_failure_continues_to_next_attempt(orchestrator):
    """When _apply_edits returns False, the step retries."""
    orchestrator.tools.write_file = MagicMock(side_effect=Exception("disk full"))
    orchestrator.verifier.run = MagicMock(return_value=make_verification_pass())
    # The first attempt will fail in _apply_edits (exception), second too,
    # so _execute_step returns False, triggering replan.
    await orchestrator.run("Fix the bug", mode="autonomous")
    # Should have called add_message with "Edit application failed"
    calls = [
        c
        for c in orchestrator.db.add_message.call_args_list
        if len(c[0]) >= 3 and c[0][2] == "Edit application failed"
    ]
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_execute_step_runs_commands(orchestrator):
    """Commands from executor result are run via sandbox."""
    result_with_commands = ExecutionResult(
        file_edits=[FileEdit(path="test.py", action="create", content="pass")],
        commands=["echo hello", "echo world"],
        explanation="done",
    )
    orchestrator.executor.execute = AsyncMock(return_value=result_with_commands)
    await orchestrator.run("Fix the bug", mode="autonomous")
    # sandbox.run_command should be called for each command, for each step
    assert orchestrator.tools.sandbox.run_command.call_count >= 2


@pytest.mark.asyncio
async def test_apply_edits_rewrite_action(orchestrator, tmp_path):
    """Test rewrite action reads before, writes new content."""
    rewrite_result = ExecutionResult(
        file_edits=[
            FileEdit(path="existing.py", action="rewrite", content="new content")
        ],
        commands=[],
        explanation="rewritten",
    )
    orchestrator.executor.execute = AsyncMock(return_value=rewrite_result)
    await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.tools.read_file.assert_called()
    orchestrator.tools.write_file.assert_called()


@pytest.mark.asyncio
async def test_apply_edits_rewrite_file_not_found(orchestrator):
    """Test rewrite when file doesn't exist yet -- before should be None."""
    orchestrator.tools.read_file = MagicMock(side_effect=FileNotFoundError("nope"))
    rewrite_result = ExecutionResult(
        file_edits=[FileEdit(path="new.py", action="rewrite", content="content")],
        commands=[],
        explanation="rewritten",
    )
    orchestrator.executor.execute = AsyncMock(return_value=rewrite_result)
    # read_file raises on first call (for before), but write_file succeeds,
    # then read_file is called again for after -- we need it to succeed the second time
    call_count = 0

    def read_side_effect(path):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise FileNotFoundError("not found")
        return "content"

    orchestrator.tools.read_file = MagicMock(side_effect=read_side_effect)
    await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.tools.write_file.assert_called()


@pytest.mark.asyncio
async def test_apply_edits_search_replace_success(orchestrator):
    """Test search_replace action path."""
    sr_result = ExecutionResult(
        file_edits=[
            FileEdit(
                path="code.py",
                action="search_replace",
                search="old_code",
                replace="new_code",
            )
        ],
        commands=[],
        explanation="replaced",
    )
    orchestrator.executor.execute = AsyncMock(return_value=sr_result)
    orchestrator.tools.edit_file = MagicMock(return_value=True)
    await orchestrator.run("Fix the bug", mode="autonomous")
    orchestrator.tools.edit_file.assert_called()
    orchestrator.db.save_edit.assert_called()


@pytest.mark.asyncio
async def test_apply_edits_search_replace_file_not_found(orchestrator):
    """search_replace on missing file sets all_ok=False and continues."""
    sr_result = ExecutionResult(
        file_edits=[
            FileEdit(
                path="missing.py",
                action="search_replace",
                search="old",
                replace="new",
            )
        ],
        commands=[],
        explanation="replaced",
    )
    orchestrator.executor.execute = AsyncMock(return_value=sr_result)
    orchestrator.tools.read_file = MagicMock(side_effect=FileNotFoundError("not found"))
    # _apply_edits returns False, so "Edit application failed" is logged
    await orchestrator.run("Fix the bug", mode="autonomous")
    calls = [
        c
        for c in orchestrator.db.add_message.call_args_list
        if len(c[0]) >= 3 and c[0][2] == "Edit application failed"
    ]
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_apply_edits_search_replace_no_match(orchestrator):
    """search_replace where edit_file returns False."""
    sr_result = ExecutionResult(
        file_edits=[
            FileEdit(
                path="code.py",
                action="search_replace",
                search="nonexistent",
                replace="new",
            )
        ],
        commands=[],
        explanation="replaced",
    )
    orchestrator.executor.execute = AsyncMock(return_value=sr_result)
    orchestrator.tools.edit_file = MagicMock(return_value=False)
    await orchestrator.run("Fix the bug", mode="autonomous")
    calls = [
        c
        for c in orchestrator.db.add_message.call_args_list
        if len(c[0]) >= 3 and c[0][2] == "Edit application failed"
    ]
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_apply_edits_unknown_action_skipped(orchestrator):
    """Unknown edit action is skipped via continue."""
    unknown_result = ExecutionResult(
        file_edits=[FileEdit(path="x.py", action="delete", content="")],
        commands=[],
        explanation="deleted",
    )
    orchestrator.executor.execute = AsyncMock(return_value=unknown_result)
    # Should not crash, just skip. But save_edit won't be called for the unknown action.
    # _apply_edits returns True (no failures, just skipped)
    await orchestrator.run("Fix the bug", mode="autonomous")
    # The edit was skipped, so save_edit should NOT be called for it
    # (save_edit is only called after the action block)


@pytest.mark.asyncio
async def test_apply_edits_exception_sets_all_ok_false(orchestrator):
    """Exception during edit sets all_ok=False."""
    orchestrator.tools.write_file = MagicMock(side_effect=PermissionError("denied"))
    create_result = ExecutionResult(
        file_edits=[FileEdit(path="x.py", action="create", content="hello")],
        commands=[],
        explanation="created",
    )
    orchestrator.executor.execute = AsyncMock(return_value=create_result)
    await orchestrator.run("Fix the bug", mode="autonomous")
    calls = [
        c
        for c in orchestrator.db.add_message.call_args_list
        if len(c[0]) >= 3 and c[0][2] == "Edit application failed"
    ]
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_get_last_error_returns_system_failed_message(orchestrator):
    """_get_last_error returns system message containing 'failed'."""
    orchestrator.db.get_messages = MagicMock(
        return_value=[
            {"role": "user", "content": "do something"},
            {"role": "system", "content": "Edit application failed"},
        ]
    )
    error = orchestrator._get_last_error("conv123")
    assert error == "Edit application failed"


@pytest.mark.asyncio
async def test_get_last_error_returns_verifier_failed_message(orchestrator):
    """_get_last_error returns verifier message containing 'failed'."""
    orchestrator.db.get_messages = MagicMock(
        return_value=[
            {"role": "user", "content": "do something"},
            {"role": "verifier", "content": "Verification failed:\npytest: FAILED"},
        ]
    )
    error = orchestrator._get_last_error("conv123")
    assert error == "Verification failed:\npytest: FAILED"


@pytest.mark.asyncio
async def test_run_returns_answered_when_planner_returns_answer(orchestrator):
    orchestrator.planner.generate_plan = AsyncMock(
        return_value=Answer(text="This app is a local coding agent.")
    )
    result = await orchestrator.run("what is this app?", mode="autonomous")
    assert result["status"] == "answered"
    assert "local coding agent" in result["answer"]
    orchestrator.executor.execute.assert_not_called()
    orchestrator.verifier.run.assert_not_called()
    orchestrator.db.update_conversation_status.assert_called_with(
        "conv123", "completed"
    )


@pytest.mark.asyncio
async def test_run_skips_plan_approval_for_answer(orchestrator):
    orchestrator.planner.generate_plan = AsyncMock(return_value=Answer(text="hello"))
    approve_calls = []

    def approve(plan):
        approve_calls.append(plan)
        return True

    result = await orchestrator.run(
        "explain this", mode="interactive", approve_plan=approve
    )
    assert result["status"] == "answered"
    assert approve_calls == []


@pytest.mark.asyncio
async def test_run_returns_failed_for_empty_plan(orchestrator):
    orchestrator.planner.generate_plan = AsyncMock(return_value=Plan(goal="", steps=[]))
    result = await orchestrator.run("do something", mode="autonomous")
    assert result["status"] == "failed"
    assert result["reason"] == "empty_plan"
    orchestrator.executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_replan_receives_project_context(orchestrator):
    orchestrator.verifier.run = MagicMock(return_value=make_verification_fail())
    orchestrator.planner.replan = AsyncMock(return_value=make_plan(1))

    await orchestrator.run("Fix the bug", mode="autonomous")
    assert orchestrator.planner.replan.call_count >= 1
    call_args = orchestrator.planner.replan.call_args
    assert "File Tree" in call_args[0][4] or "File Tree" in call_args.kwargs.get(
        "project_context", ""
    )


@pytest.mark.asyncio
async def test_blocked_command_logged_and_skipped(orchestrator):
    from agent.command_policy import CommandBlocked

    result_with_cmd = ExecutionResult(
        file_edits=[FileEdit(path="test.py", action="create", content="pass")],
        commands=["rm -rf /"],
        explanation="done",
    )
    orchestrator.executor.execute = AsyncMock(return_value=result_with_cmd)
    orchestrator.tools.sandbox.run_command = MagicMock(
        side_effect=CommandBlocked("rm -rf /", "destructive")
    )
    await orchestrator.run("Fix the bug", mode="autonomous")
    blocked_calls = [
        c
        for c in orchestrator.db.add_message.call_args_list
        if len(c[0]) >= 3 and "Command blocked" in c[0][2]
    ]
    assert len(blocked_calls) >= 1


def test_build_project_context_includes_env(orchestrator):
    context = orchestrator._build_project_context()
    assert "<env>" in context
    assert "os:" in context
    assert "File Tree" in context


def test_detect_environment_cached(orchestrator):
    env1 = orchestrator._detect_environment()
    env2 = orchestrator._detect_environment()
    assert env1 is env2


@pytest.mark.asyncio
async def test_get_last_error_returns_unknown_when_no_failures(orchestrator):
    """_get_last_error returns 'Unknown error' when no failure messages exist."""
    orchestrator.db.get_messages = MagicMock(
        return_value=[
            {"role": "user", "content": "do something"},
            {"role": "executor", "content": "all good"},
        ]
    )
    error = orchestrator._get_last_error("conv123")
    assert error == "Unknown error"
