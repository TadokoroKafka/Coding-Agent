from __future__ import annotations

from types import SimpleNamespace

import pytest

from coding_agent.model_client import DeepSeekClient


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def response(*, content="done", reasoning="reason", calls=()):
    tool_calls = [
        SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        for call_id, name, arguments in calls
    ]
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_parses_tool_calls_and_preserves_reasoning_content():
    captured = []
    client = DeepSeekClient(
        api_key="test-key",
        thinking=True,
        request_callable=lambda **kwargs: captured.append(kwargs) or response(
            content=None,
            calls=(("call-1", "read_file", '{"path":"a.py"}'),),
        ),
    )
    result = client.complete([{"role": "user", "content": "read"}], [])

    assert result.tool_calls[0].name == "read_file"
    assert result.as_assistant_message()["reasoning_content"] == "reason"
    assert captured[0]["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_rate_limits_and_server_errors(status):
    attempts = []
    sleeps = []

    def request(**_):
        attempts.append(1)
        if len(attempts) < 3:
            raise StatusError(status)
        return response()

    client = DeepSeekClient(
        api_key="test-key",
        request_callable=request,
        sleep=sleeps.append,
        jitter=lambda _a, _b: 0,
    )
    assert client.complete([], []).content == "done"
    assert len(attempts) == 3
    assert sleeps == [1, 2]


def test_authentication_error_is_not_retried():
    attempts = []

    def request(**_):
        attempts.append(1)
        raise StatusError(401)

    client = DeepSeekClient(api_key="bad", request_callable=request, sleep=lambda _: None)
    with pytest.raises(StatusError):
        client.complete([], [])
    assert len(attempts) == 1


def test_retry_limit_propagates_last_error():
    attempts = []

    def request(**_):
        attempts.append(1)
        raise StatusError(429)

    client = DeepSeekClient(
        api_key="test-key",
        request_callable=request,
        sleep=lambda _: None,
        jitter=lambda _a, _b: 0,
    )
    with pytest.raises(StatusError):
        client.complete([], [])
    assert len(attempts) == 3


def test_missing_key_is_reported_before_first_real_request():
    client = DeepSeekClient(api_key=None)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        client.complete([], [])
