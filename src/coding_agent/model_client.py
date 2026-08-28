from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_message_value(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None

    def as_assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = [call.as_message_value() for call in self.tool_calls]
        return message


class DeepSeekClient:
    """Small Chat Completions adapter with retries confined to API requests."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        thinking: bool = False,
        max_attempts: int = 3,
        request_callable: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.thinking = thinking
        self.max_attempts = max_attempts
        self._request_callable = request_callable
        self._sleep = sleep
        self._jitter = jitter

    @classmethod
    def from_env(cls, **overrides: Any) -> "DeepSeekClient":
        thinking_value = os.getenv("DEEPSEEK_THINKING", "false").strip().lower()
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            thinking=thinking_value in {"1", "true", "yes", "on"},
            **overrides,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        request = self._get_request_callable()
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_response = request(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    extra_body={
                        "thinking": {"type": "enabled" if self.thinking else "disabled"}
                    },
                )
                return self._parse_response(raw_response)
            except Exception as exc:
                last_error = exc
                if attempt == self.max_attempts or not self._is_retriable(exc):
                    raise
                delay = (2 ** (attempt - 1)) + self._jitter(0.0, 0.2)
                self._sleep(delay)
        raise RuntimeError("模型请求进入了不可达的重试状态") from last_error

    def _get_request_callable(self) -> Callable[..., Any]:
        if self._request_callable is not None:
            return self._request_callable
        if not self.api_key:
            raise RuntimeError(
                "未设置 DEEPSEEK_API_KEY。请先在当前环境中配置该变量，再运行智能体。"
            )
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._request_callable = client.chat.completions.create
        return self._request_callable

    @staticmethod
    def _parse_response(response: Any) -> ModelResponse:
        message = response.choices[0].message
        parsed_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            parsed_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )
        return ModelResponse(
            content=message.content,
            tool_calls=tuple(parsed_calls),
            reasoning_content=getattr(message, "reasoning_content", None),
        )

    @staticmethod
    def _is_retriable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is None and getattr(exc, "response", None) is not None:
            status_code = getattr(exc.response, "status_code", None)
        if status_code is not None:
            return status_code == 429 or 500 <= status_code < 600
        return exc.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "TimeoutException",
        }
