from unittest.mock import AsyncMock

import pytest

from agent.llm_client import LLMClient
from agent.models import Answer, Plan, Step
from agent.planner import Planner


PLAN_RESPONSE = {
    "kind": "plan",
    "goal": "Add health check",
    "steps": [
        {
            "id": 1,
            "action": "Create health endpoint",
            "files_needed": ["src/app.py"],
            "verify_command": "pytest tests/test_health.py",
        },
        {
            "id": 2,
            "action": "Add health test",
            "files_needed": ["tests/test_health.py"],
            "verify_command": "pytest tests/test_health.py",
        },
    ],
}

REPLAN_RESPONSE = {
    "kind": "plan",
    "goal": "Add health check (revised)",
    "steps": [
        {
            "id": 1,
            "action": "Fix import error in app.py",
            "files_needed": ["src/app.py"],
            "verify_command": None,
        },
        {
            "id": 2,
            "action": "Create health endpoint",
            "files_needed": ["src/app.py"],
            "verify_command": "pytest tests/test_health.py",
        },
    ],
}


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=LLMClient)
    client.chat_json = AsyncMock(return_value=PLAN_RESPONSE)
    return client


@pytest.fixture
def planner(mock_client):
    return Planner(mock_client)


@pytest.mark.asyncio
async def test_generate_plan(planner, mock_client):
    plan = await planner.generate_plan("Add health check", "file tree here")
    assert plan.goal == "Add health check"
    assert len(plan.steps) == 2
    assert plan.steps[0].action == "Create health endpoint"
    mock_client.chat_json.assert_called_once()


@pytest.mark.asyncio
async def test_generate_plan_defaults_to_think_false(planner, mock_client, monkeypatch):
    """Default behavior: planner calls chat_json with think=False so qwen3.6
    skips its 60-120s thinking phase on local hardware."""
    monkeypatch.delenv("AGENT_PLANNER_THINK", raising=False)
    await planner.generate_plan("anything", "context")
    assert mock_client.chat_json.call_args.kwargs["think"] is False


@pytest.mark.asyncio
async def test_generate_plan_respects_AGENT_PLANNER_THINK_env(
    planner, mock_client, monkeypatch
):
    """Setting AGENT_PLANNER_THINK=true re-enables thinking-mode planning
    for users who prefer quality over latency on complex tasks."""
    monkeypatch.setenv("AGENT_PLANNER_THINK", "true")
    await planner.generate_plan("anything", "context")
    assert mock_client.chat_json.call_args.kwargs["think"] is True


@pytest.mark.asyncio
async def test_AGENT_PLANNER_THINK_accepts_multiple_truthy_values(
    planner, mock_client, monkeypatch
):
    """Common truthy values all enable thinking (matches Python env-flag idiom)."""
    for value in ["1", "true", "True", "TRUE", "yes", "YES"]:
        monkeypatch.setenv("AGENT_PLANNER_THINK", value)
        mock_client.chat_json.reset_mock()
        mock_client.chat_json.return_value = PLAN_RESPONSE
        await planner.generate_plan("anything", "context")
        assert mock_client.chat_json.call_args.kwargs["think"] is True, value


@pytest.mark.asyncio
async def test_AGENT_PLANNER_THINK_falsy_values_keep_default_off(
    planner, mock_client, monkeypatch
):
    """Falsy values keep thinking off (the secure default)."""
    for value in ["", "0", "false", "no", "off", "anything-else"]:
        monkeypatch.setenv("AGENT_PLANNER_THINK", value)
        mock_client.chat_json.reset_mock()
        mock_client.chat_json.return_value = PLAN_RESPONSE
        await planner.generate_plan("anything", "context")
        assert mock_client.chat_json.call_args.kwargs["think"] is False, value


@pytest.mark.asyncio
async def test_generate_plan_sends_system_prompt(planner, mock_client):
    await planner.generate_plan("task", "context")
    messages = mock_client.chat_json.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert "planning agent" in messages[0]["content"].lower()


@pytest.mark.asyncio
async def test_generate_plan_includes_context(planner, mock_client):
    await planner.generate_plan("Add feature", "src/app.py\nsrc/models.py")
    messages = mock_client.chat_json.call_args.args[0]
    user_msg = messages[-1]["content"]
    assert "Add feature" in user_msg
    assert "src/app.py" in user_msg


@pytest.mark.asyncio
async def test_replan(planner, mock_client):
    mock_client.chat_json = AsyncMock(return_value=REPLAN_RESPONSE)
    plan = await planner.replan(
        task="Add health check",
        current_plan=Plan(goal="Add health check", steps=[]),
        failed_step_id=1,
        error="ImportError",
    )
    assert len(plan.steps) == 2
    assert "Fix import" in plan.steps[0].action


@pytest.mark.asyncio
async def test_replan_includes_error(planner, mock_client):
    mock_client.chat_json = AsyncMock(return_value=REPLAN_RESPONSE)
    await planner.replan("task", Plan(goal="g", steps=[]), 1, "some error")
    messages = mock_client.chat_json.call_args.args[0]
    assert "some error" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_generate_plan_returns_answer_for_question(planner, mock_client):
    mock_client.chat_json = AsyncMock(
        return_value={"kind": "answer", "answer": "This app is a local coding agent."}
    )
    result = await planner.generate_plan("what is this app?", "file tree")
    assert isinstance(result, Answer)
    assert "local coding agent" in result.text


@pytest.mark.asyncio
async def test_replan_with_steps_includes_plan_summary(planner, mock_client):
    mock_client.chat_json = AsyncMock(return_value=REPLAN_RESPONSE)
    original_plan = Plan(
        goal="Add health check",
        steps=[
            Step(id=1, action="Create endpoint", files_needed=["app.py"]),
            Step(id=2, action="Write tests", files_needed=["test.py"]),
        ],
    )
    plan = await planner.replan(
        task="Add health check",
        current_plan=original_plan,
        failed_step_id=1,
        error="ImportError",
    )
    messages = mock_client.chat_json.call_args.args[0]
    user_msg = messages[-1]["content"]
    assert "Step 1: Create endpoint" in user_msg
    assert "Step 2: Write tests" in user_msg
    assert len(plan.steps) == 2


@pytest.mark.asyncio
async def test_generate_plan_retries_on_invalid_shape(planner, mock_client):
    bad = {"kind": "plan", "goal": "do something"}  # missing steps
    mock_client.chat_json = AsyncMock(side_effect=[bad, PLAN_RESPONSE])
    result = await planner.generate_plan("Add feature", "file tree")
    assert isinstance(result, Plan)
    assert len(result.steps) == 2
    assert mock_client.chat_json.call_count == 2


@pytest.mark.asyncio
async def test_generate_plan_returns_best_effort_after_max_retries(
    planner, mock_client
):
    bad = {"kind": "plan", "goal": "x"}
    mock_client.chat_json = AsyncMock(return_value=bad)
    result = await planner.generate_plan("Add feature", "file tree")
    # 1 initial + 2 retries
    assert mock_client.chat_json.call_count == 3
    assert isinstance(result, Plan)
    assert result.steps == []


@pytest.mark.asyncio
async def test_generate_plan_retries_on_json_decode_error(planner, mock_client):
    import json

    mock_client.chat_json = AsyncMock(
        side_effect=[json.JSONDecodeError("bad", "doc", 0), PLAN_RESPONSE]
    )
    result = await planner.generate_plan("Add feature", "file tree")
    assert isinstance(result, Plan)
    assert mock_client.chat_json.call_count == 2


@pytest.mark.asyncio
async def test_replan_retries_on_invalid_shape(planner, mock_client):
    bad = {"kind": "plan", "goal": "revised"}
    mock_client.chat_json = AsyncMock(side_effect=[bad, REPLAN_RESPONSE])
    result = await planner.replan("task", Plan(goal="g", steps=[]), 1, "err")
    assert len(result.steps) == 2
    assert mock_client.chat_json.call_count == 2


@pytest.mark.asyncio
async def test_replan_includes_completed_steps(planner, mock_client):
    mock_client.chat_json = AsyncMock(return_value=REPLAN_RESPONSE)
    completed = [
        {"step_id": 1, "action": "Create endpoint"},
        {"step_id": 2, "action": "Write tests"},
    ]
    await planner.replan(
        "task", Plan(goal="g", steps=[]), 3, "err", completed_steps=completed
    )
    messages = mock_client.chat_json.call_args.args[0]
    user_msg = messages[-1]["content"]
    assert "Completed Steps" in user_msg
    assert "Create endpoint" in user_msg
    assert "(DONE)" in user_msg
    assert "do not repeat them" in user_msg


@pytest.mark.asyncio
async def test_replan_without_completed_steps(planner, mock_client):
    mock_client.chat_json = AsyncMock(return_value=REPLAN_RESPONSE)
    await planner.replan("task", Plan(goal="g", steps=[]), 1, "err")
    messages = mock_client.chat_json.call_args.args[0]
    assert "Completed Steps" not in messages[-1]["content"]


@pytest.mark.asyncio
async def test_replan_coerces_answer_to_empty_plan(planner, mock_client):
    """An 'answer' response during replan is nonsensical -- return empty plan."""
    mock_client.chat_json = AsyncMock(
        return_value={"kind": "answer", "answer": "not a plan"}
    )
    result = await planner.replan("task", Plan(goal="g", steps=[]), 1, "err")
    assert isinstance(result, Plan)
    assert result.steps == []
