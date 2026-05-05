import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.planner import Planner
from agent.llm_client import LLMClient
from agent.models import Plan


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
