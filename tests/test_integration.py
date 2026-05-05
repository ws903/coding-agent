# tests/test_integration.py
from unittest.mock import AsyncMock

import pytest

from agent.db import AgentDB
from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator
from agent.planner import Planner
from agent.sandbox import Sandbox
from agent.tools import FileTools
from agent.verifier import Verifier


PLAN_RESPONSE = """## Plan: Create hello script

### Step 1: Create hello.py
- Files needed: hello.py
- Verify: python3 hello.py
"""

EXECUTOR_RESPONSE = """I'll create the hello script.

CREATE hello.py
```
print("hello world")
```
"""


@pytest.mark.asyncio
async def test_full_autonomous_run(tmp_path):
    db = AgentDB(tmp_path / ".agent" / "agent.db")

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(side_effect=[PLAN_RESPONSE, EXECUTOR_RESPONSE])

    sandbox = Sandbox(tmp_path)
    tools = FileTools(tmp_path)
    planner = Planner(mock_llm)
    executor = Executor(mock_llm, tools)
    verifier = Verifier(sandbox, commands=[])

    orch = Orchestrator(
        planner=planner,
        executor=executor,
        verifier=verifier,
        tools=tools,
        db=db,
        project_root=tmp_path,
    )

    result = await orch.run("Create a hello world script", mode="autonomous")

    assert result["status"] == "completed"
    assert (tmp_path / "hello.py").exists()
    assert "hello world" in (tmp_path / "hello.py").read_text()

    conv = db.get_conversation(result["conv_id"])
    assert conv["status"] == "completed"

    plans = db.get_plans(result["conv_id"])
    assert len(plans) >= 1

    edits = db.get_edits(result["conv_id"])
    assert len(edits) == 1
    assert edits[0]["file_path"] == "hello.py"
    assert edits[0]["edit_type"] == "create"


@pytest.mark.asyncio
async def test_full_run_with_verification(tmp_path):
    db = AgentDB(tmp_path / ".agent" / "agent.db")

    plan_response = """## Plan: Create and test script

### Step 1: Create greet.py
- Files needed: greet.py
- Verify: python3 greet.py
"""

    executor_response = """
CREATE greet.py
```
print("greetings")
```
"""

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat = AsyncMock(side_effect=[plan_response, executor_response])

    sandbox = Sandbox(tmp_path)
    tools = FileTools(tmp_path)
    planner = Planner(mock_llm)
    executor = Executor(mock_llm, tools)
    verifier = Verifier(sandbox, commands=[])

    orch = Orchestrator(
        planner=planner,
        executor=executor,
        verifier=verifier,
        tools=tools,
        db=db,
        project_root=tmp_path,
    )

    result = await orch.run("Create greeting script", mode="autonomous")
    assert result["status"] == "completed"
    assert (tmp_path / "greet.py").exists()

    cmd_result = sandbox.run_command("python3 greet.py")
    assert cmd_result.exit_code == 0
    assert "greetings" in cmd_result.stdout
