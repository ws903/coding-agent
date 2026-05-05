from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from agent.llm_client import LLMClient


@pytest.fixture
def client():
    return LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
    )


def test_client_init(client):
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model == "qwen3:14b"


def test_build_payload(client):
    messages = [{"role": "user", "content": "hello"}]
    payload = client._build_payload(messages, temperature=0.5, max_tokens=100)
    assert payload["model"] == "qwen3:14b"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    assert payload["stream"] is False


def test_build_payload_with_stream(client):
    messages = [{"role": "user", "content": "hello"}]
    payload = client._build_payload(messages, stream=True)
    assert payload["stream"] is True


@pytest.mark.asyncio
async def test_chat_returns_content(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "world"}}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        result = await client.chat([{"role": "user", "content": "hello"}])
        assert result == "world"


@pytest.mark.asyncio
async def test_chat_posts_to_correct_url(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        await client.chat([{"role": "user", "content": "hi"}])
        call_args = mock_instance.post.call_args
        assert call_args[0][0] == "http://localhost:11434/v1/chat/completions"


@pytest.mark.asyncio
async def test_chat_stream_yields_content(client):
    """Test chat_stream async generator with SSE-style lines."""

    async def mock_aiter_lines():
        lines = [
            "event: message",
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{}}]}',
            "data: [DONE]",
        ]
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.stream = MagicMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        chunks = []
        async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_chat_stream_skips_non_data_lines(client):
    """Test that chat_stream skips lines not starting with 'data: '."""

    async def mock_aiter_lines():
        lines = [
            "",
            ": keep-alive",
            'data: {"choices":[{"delta":{"content":"only"}}]}',
            "data: [DONE]",
        ]
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.stream = MagicMock(return_value=mock_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        chunks = []
        async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["only"]
