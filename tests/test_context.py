from __future__ import annotations

from coding_agent.context import ContextManager


def group(number):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"call-{number}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": f"call-{number}", "content": f"result-{number}"},
    ]


def test_group_limit_never_splits_tool_call_and_result():
    context = ContextManager("system", "task", max_groups=2)
    for number in range(4):
        context.add_group(group(number))

    messages = context.messages()
    call_ids = {
        call["id"]
        for message in messages
        for call in message.get("tool_calls", [])
    }
    result_ids = {message["tool_call_id"] for message in messages if message["role"] == "tool"}
    assert call_ids == result_ids == {"call-2", "call-3"}


def test_character_limit_prunes_complete_early_groups():
    context = ContextManager("system", "task", max_groups=12, max_chars=450)
    for number in range(4):
        current = group(number)
        current[1]["content"] = "x" * 200
        context.add_group(current)
    messages = context.messages()
    assert any("执行状态快照" in str(message.get("content")) for message in messages)
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "tool"


def test_snapshot_preserves_modified_files_and_test_status():
    context = ContextManager("system", "task", max_groups=1)
    context.record_tool_result("write_file", {"path": "solution.py"}, {"status": "ok"})
    context.record_tool_result(
        "run_command",
        {"argv": ["python", "-m", "pytest"]},
        {"status": "error", "exit_code": 1, "timed_out": False},
    )
    context.add_group(group(1))
    context.add_group(group(2))
    snapshot = context.messages()[2]["content"]
    assert "solution.py" in snapshot
    assert '"exit_code": 1' in snapshot
    assert "run_command" in snapshot


def test_system_prompt_and_original_task_are_always_first():
    context = ContextManager("rules", "original task", max_groups=1)
    for number in range(3):
        context.add_group(group(number))
    messages = context.messages()
    assert messages[0] == {"role": "system", "content": "rules"}
    assert messages[1] == {"role": "user", "content": "original task"}


def test_snapshot_records_recent_text_searches():
    context = ContextManager("system", "task")

    context.record_tool_result(
        "search_text",
        {"query": "CodingAgent", "path": "src", "pattern": "**/*.py"},
        {"status": "ok", "count": 2, "truncated": False},
    )

    assert context.state.snapshot()["recent_searches"] == [
        {
            "query": "CodingAgent",
            "path": "src",
            "pattern": "**/*.py",
            "count": 2,
            "truncated": False,
        }
    ]
