from unittest.mock import AsyncMock

import pytest

from agent.planner import Planner
from agent.llm_client import LLMClient
from agent.models import Answer, Plan


MOCK_PLAN_RESPONSE = """## Plan: Add health check

### Step 1: Create health endpoint
- Files needed: src/app.py
- Verify: pytest tests/test_health.py

### Step 2: Add health test
- Files needed: tests/test_health.py
- Verify: pytest tests/test_health.py
"""

MOCK_REPLAN_RESPONSE = """## Plan: Add health check (revised)

### Step 1: Fix import error in app.py
- Files needed: src/app.py
- Verify: python -c "import src.app"

### Step 2: Create health endpoint
- Files needed: src/app.py
- Verify: pytest tests/test_health.py
"""


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=LLMClient)
    client.chat = AsyncMock(return_value=MOCK_PLAN_RESPONSE)
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
    mock_client.chat.assert_called_once()


@pytest.mark.asyncio
async def test_generate_plan_sends_system_prompt(planner, mock_client):
    await planner.generate_plan("task", "context")
    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert "planning agent" in messages[0]["content"].lower()


@pytest.mark.asyncio
async def test_generate_plan_includes_context(planner, mock_client):
    await planner.generate_plan("Add feature", "src/app.py\nsrc/models.py")
    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "Add feature" in user_msg
    assert "src/app.py" in user_msg


@pytest.mark.asyncio
async def test_replan(planner, mock_client):
    mock_client.chat = AsyncMock(return_value=MOCK_REPLAN_RESPONSE)
    original_plan = Plan(goal="Add health check", steps=[])
    plan = await planner.replan(
        task="Add health check",
        current_plan=original_plan,
        failed_step_id=1,
        error="ImportError: no module named flask",
    )
    assert len(plan.steps) == 2
    assert "Fix import" in plan.steps[0].action


@pytest.mark.asyncio
async def test_replan_includes_error(planner, mock_client):
    mock_client.chat = AsyncMock(return_value=MOCK_REPLAN_RESPONSE)
    original_plan = Plan(goal="goal", steps=[])
    await planner.replan("task", original_plan, 1, "some error")
    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "some error" in user_msg


@pytest.mark.asyncio
async def test_generate_plan_returns_answer_for_question(planner, mock_client):
    mock_client.chat = AsyncMock(
        return_value="## Answer\n\nThis app is a local coding agent.\n"
    )
    result = await planner.generate_plan("what is this app?", "file tree")
    assert isinstance(result, Answer)
    assert "local coding agent" in result.text


@pytest.mark.asyncio
async def test_replan_with_steps_includes_plan_summary(planner, mock_client):
    """replan with a plan that has steps includes step summaries in the message."""
    from agent.models import Step

    mock_client.chat = AsyncMock(return_value=MOCK_REPLAN_RESPONSE)
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
    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "Step 1: Create endpoint" in user_msg
    assert "Step 2: Write tests" in user_msg
    assert len(plan.steps) == 2


@pytest.mark.asyncio
async def test_generate_plan_retries_on_empty_parse(planner, mock_client):
    """When the first response has ## Plan: but no steps, planner retries."""
    bad_response = "## Plan: do something\n\nno steps here\n"
    mock_client.chat = AsyncMock(side_effect=[bad_response, MOCK_PLAN_RESPONSE])
    result = await planner.generate_plan("Add feature", "file tree")
    assert isinstance(result, Plan)
    assert len(result.steps) == 2
    assert mock_client.chat.call_count == 2


@pytest.mark.asyncio
async def test_generate_plan_returns_best_effort_after_max_retries(
    planner, mock_client
):
    """After MAX_PARSE_RETRIES, returns whatever was last parsed."""
    bad_response = "## Plan: do something\n\nno steps\n"
    mock_client.chat = AsyncMock(return_value=bad_response)
    result = await planner.generate_plan("Add feature", "file tree")
    assert isinstance(result, Plan)
    assert mock_client.chat.call_count == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_replan_retries_on_empty_parse(planner, mock_client):
    bad_response = "## Plan: revised\n\nnothing\n"
    mock_client.chat = AsyncMock(side_effect=[bad_response, MOCK_REPLAN_RESPONSE])
    original_plan = Plan(goal="goal", steps=[])
    result = await planner.replan("task", original_plan, 1, "error")
    assert len(result.steps) == 2
    assert mock_client.chat.call_count == 2
