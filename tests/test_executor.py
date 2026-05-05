import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.executor import Executor
from agent.llm_client import LLMClient
from agent.models import Step, FileEdit
from agent.tools import FileTools


MOCK_SEARCH_REPLACE_RESPONSE = """I'll update the handler to add the health endpoint.

src/app.py
<<<<<<< SEARCH
@app.route("/")
def index():
    return "hello"
=======
@app.route("/")
def index():
    return "hello"

@app.route("/health")
def health():
    return {"status": "ok"}
>>>>>>> REPLACE
"""

MOCK_CREATE_RESPONSE = """I'll create the test file.

CREATE tests/test_health.py
```
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
```
"""

MOCK_REWRITE_RESPONSE = """I'll rewrite this small file.

REWRITE src/config.py
```
DEBUG = False
PORT = 8080
```
"""


@pytest.fixture
def mock_client():
    client = AsyncMock(spec=LLMClient)
    return client


@pytest.fixture
def tools(tmp_path):
    return FileTools(tmp_path)


@pytest.fixture
def executor(mock_client, tools):
    return Executor(mock_client, tools)


@pytest.mark.asyncio
async def test_execute_search_replace(executor, mock_client, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text('@app.route("/")\ndef index():\n    return "hello"\n')
    mock_client.chat = AsyncMock(return_value=MOCK_SEARCH_REPLACE_RESPONSE)

    step = Step(id=1, action="Add health endpoint", files_needed=["src/app.py"])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "search_replace"


@pytest.mark.asyncio
async def test_execute_create_file(executor, mock_client, tmp_path):
    mock_client.chat = AsyncMock(return_value=MOCK_CREATE_RESPONSE)

    step = Step(id=1, action="Create test file", files_needed=[])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "create"
    assert result.file_edits[0].path == "tests/test_health.py"


@pytest.mark.asyncio
async def test_execute_rewrite(executor, mock_client, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text("DEBUG = True\n")
    mock_client.chat = AsyncMock(return_value=MOCK_REWRITE_RESPONSE)

    step = Step(id=1, action="Update config", files_needed=["src/config.py"])
    result = await executor.execute(step)

    assert len(result.file_edits) == 1
    assert result.file_edits[0].action == "rewrite"


@pytest.mark.asyncio
async def test_execute_includes_file_contents_in_prompt(executor, mock_client, tmp_path):
    (tmp_path / "code.py").write_text("x = 1\n")
    mock_client.chat = AsyncMock(return_value="no edits")

    step = Step(id=1, action="Change x", files_needed=["code.py"])
    await executor.execute(step)

    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "x = 1" in user_msg


@pytest.mark.asyncio
async def test_execute_with_errors_includes_error(executor, mock_client, tmp_path):
    mock_client.chat = AsyncMock(return_value="no edits")
    step = Step(id=1, action="Fix bug", files_needed=[])
    await executor.execute(step, errors="NameError: name 'foo' is not defined")

    call_args = mock_client.chat.call_args
    messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
    user_msg = messages[1]["content"]
    assert "NameError" in user_msg
