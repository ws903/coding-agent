import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator, Callable

import httpx

from agent.models import TokenUsage

DEFAULT_CONTEXT_LIMIT = 8192
RESPONSE_RESERVE = 4096

MAX_RETRIES = 3
BACKOFF_BASE = 0.5
BACKOFF_MAX = 8.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += 4
        total += estimate_tokens(msg.get("content", ""))
    return total + 2


def _backoff_delay(attempt: int) -> float:
    delay = min(BACKOFF_BASE * (2**attempt), BACKOFF_MAX)
    jitter = random.uniform(0, delay * 0.5)
    return delay + jitter


def _parse_json_content(content: str) -> dict:
    """Tolerant JSON parser: strips ```json fences and finds the outer object."""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    if s.startswith("json\n"):
        s = s[5:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3.6:35b",
        api_key: str = "local",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._context_limit: int | None = None
        self._client: httpx.AsyncClient | None = None
        self.total_usage = TokenUsage()
        self.call_count = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, headers=self._headers()
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _record_usage(self, data: dict) -> None:
        usage = data.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        self.total_usage.prompt_tokens += prompt
        self.total_usage.completion_tokens += completion
        self.total_usage.total_tokens += prompt + completion
        self.call_count += 1

    async def get_context_limit(self) -> int:
        if self._context_limit is not None:
            return self._context_limit
        try:
            ollama_base = self.base_url.replace("/v1", "")
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{ollama_base}/api/show",
                    json={"model": self.model},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    model_info = data.get("model_info", {})
                    for key, value in model_info.items():
                        if "context_length" in key:
                            self._context_limit = int(value)
                            return self._context_limit
        except Exception:
            pass
        self._context_limit = DEFAULT_CONTEXT_LIMIT
        return self._context_limit

    def _compute_num_ctx(self, messages: list[dict], max_tokens: int) -> int:
        prompt_tokens = estimate_messages_tokens(messages)
        return int(prompt_tokens * 1.25) + max_tokens

    def _dynamic_max_tokens(self, messages: list[dict], context_limit: int) -> int:
        prompt_tokens = estimate_messages_tokens(messages)
        available = context_limit - prompt_tokens
        return max(256, min(available, RESPONSE_RESERVE))

    def _build_payload(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> dict:
        num_ctx = self._compute_num_ctx(messages, max_tokens)
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "num_ctx": num_ctx,
        }

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        data = await self._post_chat(messages, temperature, max_tokens)
        return data["choices"][0]["message"]["content"]

    async def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """Chat with response_format=json_object; returns parsed dict.

        Raises ValueError if the response isn't valid JSON. Caller is
        expected to retry or fall back as appropriate.
        """
        data = await self._post_chat(
            messages, temperature, max_tokens, response_format="json_object"
        )
        content = data["choices"][0]["message"]["content"] or ""
        return _parse_json_content(content)

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict:
        """Returns the full assistant message dict including any tool_calls."""
        data = await self._post_chat(messages, temperature, max_tokens, tools=tools)
        return data["choices"][0]["message"]

    async def quick_chat_stream(
        self,
        messages: list[dict],
        on_token: Callable[[str], None] | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Native Ollama /api/chat with `think: false` -- skips the model's
        reasoning phase entirely. ~3-5x faster than chat_stream for thinking
        models when you just need a quick chat reply.

        Returns the assembled content string. Tokens are streamed via on_token.
        Note: this bypasses the OpenAI-compat endpoint and the LLM-call retry
        path; intended for cheap "hi"-style chat where retries don't matter.
        """
        ollama_base = self.base_url.replace("/v1", "")
        url = f"{ollama_base}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {"temperature": temperature},
        }
        client = await self._get_client()
        parts: list[str] = []
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                content = msg.get("content")
                if content:
                    parts.append(content)
                    if on_token is not None:
                        on_token(content)
                if chunk.get("done"):
                    break
        self.call_count += 1
        return "".join(parts)

    async def chat_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict:
        """Stream content chunks via on_token; return the assembled assistant message.

        For thinking models (qwen3.6, deepseek-r1, etc.), reasoning tokens
        stream via `on_reasoning` if provided. This gives users visibility
        during the model's silent thinking phase, which can be 10-30s for
        non-trivial requests.
        """
        context_limit = await self.get_context_limit()
        max_tokens = self._dynamic_max_tokens(messages, context_limit)
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)
        payload["tools"] = tools
        url = f"{self.base_url}/chat/completions"
        client = await self._get_client()

        content_parts: list[str] = []
        tool_calls: list[dict] = []
        prompt_tokens = 0
        completion_tokens = 0

        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                chunk = json.loads(data_str)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                reasoning = delta.get("reasoning")
                if reasoning and on_reasoning is not None:
                    on_reasoning(reasoning)

                content = delta.get("content")
                if content:
                    content_parts.append(content)
                    if on_token is not None:
                        on_token(content)

                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )

        if prompt_tokens or completion_tokens:
            self._record_usage(
                {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                }
            )
        else:
            self.call_count += 1

        msg: dict = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    async def _post_chat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
        response_format: str | None = None,
    ) -> dict:
        context_limit = await self.get_context_limit()
        max_tokens = self._dynamic_max_tokens(messages, context_limit)
        payload = self._build_payload(messages, temperature, max_tokens, stream=False)
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = {"type": response_format}
        url = f"{self.base_url}/chat/completions"
        client = await self._get_client()

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.post(url, json=payload)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = httpx.HTTPStatusError(
                        f"{response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    if attempt < MAX_RETRIES:
                        delay = _backoff_delay(attempt)
                        logger.warning(
                            "LLM call returned %d, retrying in %.1fs (attempt %d/%d)",
                            response.status_code,
                            delay,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
                response.raise_for_status()
                data = response.json()
                self._record_usage(data)
                return data
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "LLM call failed (%s), retrying in %.1fs (attempt %d/%d)",
                        type(exc).__name__,
                        delay,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
        raise last_error  # type: ignore[misc]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        context_limit = await self.get_context_limit()
        max_tokens = self._dynamic_max_tokens(messages, context_limit)
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)
        url = f"{self.base_url}/chat/completions"
        client = await self._get_client()
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
