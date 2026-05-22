from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from agent.llm_client import (
    LLMClient,
    _backoff_delay,
    estimate_tokens,
    estimate_messages_tokens,
    MAX_RETRIES,
    RESPONSE_RESERVE,
)


@pytest.fixture
def client():
    c = LLMClient(
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
    )
    c._context_limit = 8192
    return c


def _make_ok_response(content="world"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.raise_for_status = MagicMock()
    return resp


def _make_error_response(status_code):
    resp = MagicMock()
    resp.status_code = status_code
    resp.request = MagicMock()
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            str(status_code), request=resp.request, response=resp
        )
    )
    return resp


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
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_ok_response())
    mock_client.is_closed = False
    client._client = mock_client

    result = await client.chat([{"role": "user", "content": "hello"}])
    assert result == "world"


@pytest.mark.asyncio
async def test_chat_posts_to_correct_url(client):
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_ok_response("ok"))
    mock_client.is_closed = False
    client._client = mock_client

    await client.chat([{"role": "user", "content": "hi"}])
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "http://localhost:11434/v1/chat/completions"


@pytest.mark.asyncio
async def test_chat_retries_on_500(client):
    error_resp = MagicMock()
    error_resp.status_code = 500
    error_resp.request = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=[error_resp, _make_ok_response("recovered")]
    )
    mock_client.is_closed = False
    client._client = mock_client

    with patch("agent.llm_client.asyncio.sleep", new_callable=AsyncMock):
        result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "recovered"
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_chat_retries_on_connection_error(client):
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=[httpx.ConnectError("refused"), _make_ok_response("ok")]
    )
    mock_client.is_closed = False
    client._client = mock_client

    with patch("agent.llm_client.asyncio.sleep", new_callable=AsyncMock):
        result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_chat_raises_after_max_retries(client):
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.is_closed = False
    client._client = mock_client

    with (
        patch("agent.llm_client.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(httpx.ConnectError),
    ):
        await client.chat([{"role": "user", "content": "hi"}])
    assert mock_client.post.call_count == MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_chat_raises_after_max_retries_on_500(client):
    error_resp = MagicMock()
    error_resp.status_code = 500
    error_resp.request = MagicMock()
    error_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "500", request=error_resp.request, response=error_resp
        )
    )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=error_resp)
    mock_client.is_closed = False
    client._client = mock_client

    with (
        patch("agent.llm_client.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await client.chat([{"role": "user", "content": "hi"}])
    assert mock_client.post.call_count == MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_chat_no_retry_on_400(client):
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_error_response(400))
    mock_client.is_closed = False
    client._client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_chat_retries_on_429(client):
    rate_limit_resp = MagicMock()
    rate_limit_resp.status_code = 429
    rate_limit_resp.request = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[rate_limit_resp, _make_ok_response("ok")])
    mock_client.is_closed = False
    client._client = mock_client

    with patch("agent.llm_client.asyncio.sleep", new_callable=AsyncMock):
        result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "ok"


def test_dynamic_max_tokens_caps_at_response_reserve(client):
    messages = [{"role": "user", "content": "short"}]
    result = client._dynamic_max_tokens(messages, 100000)
    assert result == RESPONSE_RESERVE


def test_dynamic_max_tokens_minimum_256(client):
    messages = [{"role": "user", "content": "x" * 30000}]
    result = client._dynamic_max_tokens(messages, 100)
    assert result == 256


def test_backoff_delay_increases():
    d0 = _backoff_delay(0)
    d2 = _backoff_delay(2)
    assert d0 < 2.0
    assert d2 > d0


def test_estimate_tokens():
    assert estimate_tokens("hello world") == 3
    assert estimate_tokens("") == 0


def test_estimate_messages_tokens():
    msgs = [{"role": "user", "content": "hello"}]
    result = estimate_messages_tokens(msgs)
    assert result > 0


@pytest.mark.asyncio
async def test_shared_client_reused(client):
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_ok_response())
    mock_client.is_closed = False
    client._client = mock_client

    await client.chat([{"role": "user", "content": "a"}])
    await client.chat([{"role": "user", "content": "b"}])
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_close_client(client):
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.aclose = AsyncMock()
    client._client = mock_client

    await client.close()
    mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_chat_records_token_usage(client):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20},
    }
    resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.is_closed = False
    client._client = mock_client

    await client.chat([{"role": "user", "content": "test"}])
    assert client.total_usage.prompt_tokens == 50
    assert client.total_usage.completion_tokens == 20
    assert client.total_usage.total_tokens == 70
    assert client.call_count == 1

    await client.chat([{"role": "user", "content": "test2"}])
    assert client.total_usage.prompt_tokens == 100
    assert client.total_usage.total_tokens == 140
    assert client.call_count == 2


@pytest.mark.asyncio
async def test_chat_records_usage_without_usage_field(client):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.is_closed = False
    client._client = mock_client

    await client.chat([{"role": "user", "content": "test"}])
    assert client.total_usage.total_tokens == 0
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_chat_stream_yields_content(client):
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

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.is_closed = False
    client._client = mock_client

    chunks = []
    async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_chat_stream_skips_non_data_lines(client):
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

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_response)
    mock_client.is_closed = False
    client._client = mock_client

    chunks = []
    async for chunk in client.chat_stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert chunks == ["only"]
