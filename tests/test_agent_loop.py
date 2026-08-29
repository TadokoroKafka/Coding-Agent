from __future__ import annotations

import json

from coding_agent.agent import SYSTEM_PROMPT, AgentResult, CodingAgent
from coding_agent.approval import ApprovalPolicy
from coding_agent.model_client import ModelResponse, ToolCall
from coding_agent.run_log import RunLog
from coding_agent.tools.files import ToolError


class FakeModelClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((messages, tools))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingRegistry:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def definitions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "write",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return {"status": "ok", "tool": name, "arguments": arguments}


def call(call_id, name="read_file", arguments='{"path":"a.py"}'):
    return ToolCall(id=call_id, name=name, arguments=arguments)


def agent(responses, *, registry=None, approval=None, max_steps=20):
    model = FakeModelClient(responses)
    registry = registry or RecordingRegistry()
    approval = approval or ApprovalPolicy(mode="auto")
    return CodingAgent(model, registry, approval, max_steps=max_steps), model, registry


def test_single_tool_result_is_returned_to_model_with_reasoning_content():
    coding_agent, model, registry = agent(
        [
            ModelResponse(None, (call("call-1"),), "need the file"),
            ModelResponse("finished"),
        ]
    )

    result = coding_agent.run("inspect the file")

    assert result == AgentResult(status="completed", message="finished", steps=2)
    assert registry.calls == [("read_file", {"path": "a.py"})]
    second_messages = model.requests[1][0]
    assert second_messages[-2]["reasoning_content"] == "need the file"
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "call-1"
    assert json.loads(second_messages[-1]["content"])["status"] == "ok"


def test_multiple_tools_execute_in_model_order():
    coding_agent, _, registry = agent(
        [
            ModelResponse(
                None,
                (
                    call("call-1", arguments='{"path":"first.py"}'),
                    call("call-2", arguments='{"path":"second.py"}'),
                ),
            ),
            ModelResponse("done"),
        ]
    )

    assert coding_agent.run("read both").status == "completed"
    assert registry.calls == [
        ("read_file", {"path": "first.py"}),
        ("read_file", {"path": "second.py"}),
    ]


def test_permission_denial_is_structured_and_does_not_execute_tool():
    approval = ApprovalPolicy(mode="ask", input_func=lambda _: "n")
    coding_agent, model, registry = agent(
        [
            ModelResponse(None, (call("call-1", "write_file", '{"path":"a.py","content":"x"}'),)),
            ModelResponse("understood"),
        ],
        approval=approval,
    )

    assert coding_agent.run("write").status == "completed"
    assert registry.calls == []
    denied = json.loads(model.requests[1][0][-1]["content"])
    assert denied == {
        "status": "permission_denied",
        "error": "permission_denied",
        "reason": "user_denied",
    }


def test_duplicate_tool_call_id_is_never_executed_twice():
    coding_agent, model, registry = agent(
        [
            ModelResponse(None, (call("same-id"),)),
            ModelResponse(None, (call("same-id", arguments='{"path":"other.py"}'),)),
            ModelResponse("done"),
        ]
    )

    assert coding_agent.run("read").status == "completed"
    assert len(registry.calls) == 1
    duplicate = json.loads(model.requests[2][0][-1]["content"])
    assert duplicate["error"] == "duplicate_tool_call"


def test_tool_failure_is_returned_once_without_automatic_replay():
    registry = RecordingRegistry([ToolError("not_a_file", "missing")])
    coding_agent, model, _ = agent(
        [ModelResponse(None, (call("call-1"),)), ModelResponse("cannot continue")],
        registry=registry,
    )

    assert coding_agent.run("read").status == "completed"
    assert len(registry.calls) == 1
    failure = json.loads(model.requests[1][0][-1]["content"])
    assert failure["error"] == "not_a_file"


def test_two_consecutive_malformed_argument_rounds_terminate():
    coding_agent, _, registry = agent(
        [
            ModelResponse(None, (call("call-1", arguments="{"),)),
            ModelResponse(None, (call("call-2", arguments="not json"),)),
        ]
    )

    result = coding_agent.run("bad calls")
    assert result.status == "invalid_tool_calls"
    assert result.steps == 2
    assert registry.calls == []


def test_two_consecutive_unknown_tool_rounds_terminate():
    coding_agent, _, registry = agent(
        [
            ModelResponse(None, (call("call-1", name="missing_tool", arguments="{}"),)),
            ModelResponse(None, (call("call-2", name="still_missing", arguments="{}"),)),
        ]
    )

    assert coding_agent.run("unknown calls").status == "invalid_tool_calls"
    assert registry.calls == []


def test_third_consecutive_same_call_terminates_before_third_execution():
    coding_agent, _, registry = agent(
        [
            ModelResponse(None, (call("call-1"),)),
            ModelResponse(None, (call("call-2"),)),
            ModelResponse(None, (call("call-3"),)),
        ]
    )

    result = coding_agent.run("loop")
    assert result.status == "repeated_tool_call"
    assert result.steps == 3
    assert len(registry.calls) == 2


def test_three_consecutive_tool_errors_terminate():
    registry = RecordingRegistry(
        [
            ToolError("failure-1", "one"),
            ToolError("failure-2", "two"),
            ToolError("failure-3", "three"),
        ]
    )
    coding_agent, _, _ = agent(
        [
            ModelResponse(None, (call("call-1", arguments='{"path":"1"}'),)),
            ModelResponse(None, (call("call-2", arguments='{"path":"2"}'),)),
            ModelResponse(None, (call("call-3", arguments='{"path":"3"}'),)),
        ],
        registry=registry,
    )

    result = coding_agent.run("errors")
    assert result.status == "consecutive_tool_errors"
    assert result.steps == 3


def test_model_argument_errors_do_not_count_as_local_tool_execution_errors():
    registry = RecordingRegistry([ToolError("read_failed", "cannot read")])
    coding_agent, _, _ = agent(
        [
            ModelResponse(None, (call("call-1", arguments="{"),)),
            ModelResponse(None, (call("call-2", arguments='{"path":"valid"}'),)),
            ModelResponse(None, (call("call-3", arguments="not json"),)),
            ModelResponse("done"),
        ],
        registry=registry,
    )

    result = coding_agent.run("mixed failures")
    assert result.status == "completed"
    assert result.steps == 4


def test_step_limit_terminates_after_configured_number_of_model_calls():
    coding_agent, model, _ = agent(
        [
            ModelResponse(None, (call("call-1", arguments='{"path":"1"}'),)),
            ModelResponse(None, (call("call-2", arguments='{"path":"2"}'),)),
        ],
        max_steps=2,
    )

    result = coding_agent.run("long task")
    assert result.status == "max_steps"
    assert result.steps == 2
    assert len(model.requests) == 2


def test_api_failure_after_client_retries_becomes_terminal_result():
    coding_agent, _, _ = agent([RuntimeError("API unavailable")])
    result = coding_agent.run("task")
    assert result.status == "api_error"
    assert "API unavailable" in result.message


def test_agent_writes_start_tool_result_and_finish_events(tmp_path):
    model = FakeModelClient(
        [ModelResponse(None, (call("call-1"),)), ModelResponse("done")]
    )
    registry = RecordingRegistry()
    run_log = RunLog(tmp_path, filename="agent.jsonl")
    coding_agent = CodingAgent(
        model,
        registry,
        ApprovalPolicy(mode="auto"),
        run_log=run_log,
    )

    coding_agent.run("inspect")

    records = [json.loads(line) for line in run_log.path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "agent_started",
        "tool_result",
        "agent_finished",
    ]
    assert records[1]["payload"]["tool_call_id"] == "call-1"
    assert records[2]["payload"]["status"] == "completed"


def test_event_handler_observes_steps_and_tool_execution():
    events = []
    model = FakeModelClient(
        [ModelResponse(None, (call("call-1"),)), ModelResponse("done")]
    )
    coding_agent = CodingAgent(
        model,
        RecordingRegistry(),
        ApprovalPolicy(mode="auto"),
        event_handler=lambda event, payload: events.append((event, payload)),
    )

    result = coding_agent.run("inspect")

    assert result.status == "completed"
    assert [event for event, _ in events] == [
        "model_step",
        "tool_call",
        "tool_result",
        "model_step",
    ]
    assert events[0][1] == {"step": 1}
    assert events[1][1] == {
        "tool_call_id": "call-1",
        "tool_name": "read_file",
        "arguments": {"path": "a.py"},
    }
    assert events[2][1]["result"]["status"] == "ok"


def test_result_contains_deterministic_execution_summary():
    registry = RecordingRegistry(
        [{"status": "ok", "count": 2, "truncated": False, "matches": []}]
    )
    coding_agent, _, _ = agent(
        [
            ModelResponse(
                None,
                (call("search-1", "search_text", '{"query":"target","path":"src"}'),),
            ),
            ModelResponse("done"),
        ],
        registry=registry,
    )

    result = coding_agent.run("locate target")

    assert result.summary["recent_searches"] == [
        {
            "query": "target",
            "path": "src",
            "pattern": "**/*",
            "count": 2,
            "truncated": False,
        }
    ]
    assert "search_text" in SYSTEM_PROMPT
