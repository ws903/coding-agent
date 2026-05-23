# tests/test_integration.py
import json
from unittest.mock import AsyncMock

import pytest

from agent.persistence.db import AgentDB
from agent.core.executor import Executor
from agent.llm.client import LLMClient
from agent.core.orchestrator import Orchestrator
from agent.core.planner import Planner
from agent.safety.sandbox import Sandbox
from agent.tools.filesystem import FileTools
from agent.core.verifier import Verifier


PLAN_RESPONSE = {
    "kind": "plan",
    "goal": "Create hello script",
    "steps": [
        {
            "id": 1,
            "action": "Create hello.py",
            "files_needed": ["hello.py"],
            "verify_command": "python3 hello.py",
        }
    ],
}


def _create_file_msg(path: str, content: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "create_file",
                    "arguments": json.dumps({"path": path, "content": content}),
                },
            }
        ],
    }


def _final_msg(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": []}


@pytest.mark.asyncio
async def test_full_autonomous_run(tmp_path):
    db = AgentDB(tmp_path / ".agent" / "agent.persistence.db")

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(return_value=PLAN_RESPONSE)
    mock_llm.chat_with_tools = AsyncMock(
        side_effect=[
            _create_file_msg("hello.py", 'print("hello world")\n'),
            _final_msg("Created hello.py."),
        ]
    )

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
    db = AgentDB(tmp_path / ".agent" / "agent.persistence.db")

    plan_response = {
        "kind": "plan",
        "goal": "Create and test script",
        "steps": [
            {
                "id": 1,
                "action": "Create greet.py",
                "files_needed": ["greet.py"],
                "verify_command": "python3 greet.py",
            }
        ],
    }

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_json = AsyncMock(return_value=plan_response)
    mock_llm.chat_with_tools = AsyncMock(
        side_effect=[
            _create_file_msg("greet.py", 'print("greetings")\n'),
            _final_msg("Created greet.py."),
        ]
    )

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
