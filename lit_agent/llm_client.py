"""Minimal OpenAI-compatible chat client used by the agent layer."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .llm_config import ProviderConfig


class LLMClientError(RuntimeError):
    """Raised when an LLM provider request fails."""


class OpenAICompatibleClient:
    def __init__(self, provider: ProviderConfig) -> None:
        self.provider = provider

    def complete(self, messages: list[dict[str, str]], *, response_format: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.provider.model,
            "messages": messages,
            "temperature": self.provider.temperature,
            "max_tokens": self.provider.max_tokens,
        }
        if response_format:
            payload["response_format"] = {"type": response_format}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.provider.api_key:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"

        url = f"{self.provider.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(1, self.provider.retry_count + 2):
            try:
                request = Request(url, data=body, headers=headers, method="POST")
                with urlopen(request, timeout=self.provider.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return str(data["choices"][0]["message"]["content"])
            except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(0.5 * attempt, 2.0))
        raise LLMClientError(f"LLM request failed: {last_error}")


def build_llm_client(provider: ProviderConfig) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(provider)
