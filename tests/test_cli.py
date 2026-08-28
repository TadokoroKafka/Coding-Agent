from __future__ import annotations

import json
import sys

import pytest

from coding_agent.cli import main
from coding_agent.model_client import ModelResponse, ToolCall


class QueueModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((messages, tools))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def tool_call(call_id, name, arguments):
    return ToolCall(call_id, name, json.dumps(arguments, ensure_ascii=False))


def test_help_lists_supported_options_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--workspace" in output
    assert "--approval-mode" in output
    assert "--max-steps" in output
    assert "--verbose" in output


def test_workspace_is_required_and_max_steps_must_be_positive(tmp_path):
    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as invalid:
        main(["--workspace", str(tmp_path), "--max-steps", "0"])
    assert invalid.value.code == 2


def test_empty_task_returns_usage_error_without_creating_model(tmp_path):
    errors = []
    factory_calls = []
    exit_code = main(
        ["--workspace", str(tmp_path)],
        input_func=lambda _: "   ",
        error_func=errors.append,
        model_client_factory=lambda: factory_calls.append(1),
    )
    assert exit_code == 2
    assert factory_calls == []
    assert any("task" in message.lower() for message in errors)


def test_auto_mode_runs_real_write_and_command_tools_end_to_end(tmp_path):
    model = QueueModel(
        [
            ModelResponse(
                None,
                (tool_call("write-1", "write_file", {"path": "hello.py", "content": "print('ok')\n"}),),
            ),
            ModelResponse(
                None,
                (
                    tool_call(
                        "run-1",
                        "run_command",
                        {
                            "argv": [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; assert Path('hello.py').is_file()",
                            ],
                            "cwd": ".",
                            "timeout_seconds": 10,
                        },
                    ),
                ),
            ),
            ModelResponse("created and verified"),
        ]
    )
    output = []

    exit_code = main(
        ["--workspace", str(tmp_path), "--approval-mode", "auto"],
        input_func=lambda _: "create the file",
        output_func=output.append,
        model_client_factory=lambda: model,
    )

    assert exit_code == 0
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert output[-1] == "created and verified"
    second_tool_result = json.loads(model.requests[1][0][-1]["content"])
    third_tool_result = json.loads(model.requests[2][0][-1]["content"])
    assert second_tool_result["status"] == "ok"
    assert third_tool_result["status"] == "ok"


def test_ask_mode_denial_is_returned_to_model_without_writing(tmp_path):
    answers = iter(["write a file", "n"])
    model = QueueModel(
        [
            ModelResponse(
                None,
                (tool_call("write-1", "write_file", {"path": "denied.txt", "content": "no"}),),
            ),
            ModelResponse("permission was denied"),
        ]
    )

    exit_code = main(
        ["--workspace", str(tmp_path)],
        input_func=lambda _: next(answers),
        output_func=lambda _: None,
        model_client_factory=lambda: model,
    )

    assert exit_code == 0
    assert not (tmp_path / "denied.txt").exists()
    result = json.loads(model.requests[1][0][-1]["content"])
    assert result["status"] == "permission_denied"


def test_missing_api_key_is_reported_on_first_request(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    errors = []
    exit_code = main(
        ["--workspace", str(tmp_path)],
        input_func=lambda _: "inspect",
        error_func=errors.append,
    )
    assert exit_code == 1
    assert any("DEEPSEEK_API_KEY" in message for message in errors)


def test_keyboard_interrupt_returns_130(tmp_path):
    errors = []
    exit_code = main(
        ["--workspace", str(tmp_path)],
        input_func=lambda _: "task",
        error_func=errors.append,
        model_client_factory=lambda: QueueModel([KeyboardInterrupt()]),
    )
    assert exit_code == 130
    assert errors == ["Interrupted by user."]


def test_non_completed_agent_result_returns_one(tmp_path):
    errors = []
    exit_code = main(
        ["--workspace", str(tmp_path)],
        input_func=lambda _: "task",
        error_func=errors.append,
        model_client_factory=lambda: QueueModel([RuntimeError("API unavailable")]),
    )
    assert exit_code == 1
    assert errors == ["Agent stopped (api_error): API unavailable"]


def test_verbose_output_summarizes_content_without_leaking_it(tmp_path):
    secret_content = "sk-example-secret-value"
    model = QueueModel(
        [
            ModelResponse(
                None,
                (
                    tool_call(
                        "write-1",
                        "write_file",
                        {"path": "safe.txt", "content": secret_content},
                    ),
                ),
            ),
            ModelResponse("done"),
        ]
    )
    output = []

    exit_code = main(
        ["--workspace", str(tmp_path), "--approval-mode", "auto", "--verbose"],
        input_func=lambda _: "task",
        output_func=output.append,
        model_client_factory=lambda: model,
    )

    rendered = "\n".join(output)
    assert exit_code == 0
    assert secret_content not in rendered
    assert "<23 characters>" in rendered
    assert "write_file" in rendered
    assert "status=ok" in rendered


def test_verbose_output_redacts_api_key_from_command_arguments(tmp_path, monkeypatch):
    api_key = "key-verbose-secret-value"
    monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    model = QueueModel(
        [
            ModelResponse(
                None,
                (
                    tool_call(
                        "run-1",
                        "run_command",
                        {
                            "argv": [sys.executable, "-c", "pass", api_key],
                            "timeout_seconds": 10,
                        },
                    ),
                ),
            ),
            ModelResponse("done"),
        ]
    )
    output = []

    exit_code = main(
        ["--workspace", str(tmp_path), "--approval-mode", "auto", "--verbose"],
        input_func=lambda _: "task",
        output_func=output.append,
        model_client_factory=lambda: model,
    )

    rendered = "\n".join(output)
    assert exit_code == 0
    assert api_key not in rendered
    assert "[REDACTED]" in rendered
