import json
from collections.abc import AsyncGenerator

import httpx

DEFAULT_CONTEXT_LIMIT = 8192
RESPONSE_RESERVE = 4096


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += 4
        total += estimate_tokens(msg.get("content", ""))
    return total + 2


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3:14b",
        api_key: str = "local",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._context_limit: int | None = None

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
        payload = self._build_payload(messages, temperature, max_tokens, stream=False)
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
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
