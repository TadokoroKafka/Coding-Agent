from __future__ import annotations

from coding_agent.approval import ApprovalDecision, ApprovalPolicy


def test_ask_mode_reads_without_confirmation():
    asked = []
    policy = ApprovalPolicy(input_func=lambda prompt: asked.append(prompt) or "n")
    assert policy.authorize("read_file", {"path": "a.py"}) == ApprovalDecision(True)
    assert asked == []


def test_ask_mode_searches_without_confirmation():
    asked = []
    policy = ApprovalPolicy(input_func=lambda prompt: asked.append(prompt) or "n")
    assert policy.authorize("search_text", {"query": "target"}) == ApprovalDecision(True)
    assert asked == []


def test_ask_mode_requires_confirmation_for_write_and_command():
    answers = iter(["y", "n"])
    policy = ApprovalPolicy(input_func=lambda _: next(answers))
    assert policy.authorize("write_file", {"path": "a.py", "content": "secret"}) == ApprovalDecision(True)
    assert policy.authorize("run_command", {"argv": ["python", "-m", "pytest"]}) == ApprovalDecision(
        False,
        "user_denied",
    )


def test_auto_mode_allows_safe_commands_and_rejects_dangerous_ones():
    policy = ApprovalPolicy(mode="auto")
    assert policy.authorize("run_command", {"argv": ["python", "-m", "pytest"]}) == ApprovalDecision(True)
    assert policy.authorize("run_command", {"argv": ["git", "diff"]}) == ApprovalDecision(True)
    assert policy.authorize(
        "run_command",
        {"argv": ["git", "commit", "-am", "x"]},
    ).allowed is False
    assert policy.authorize("run_command", {"argv": ["pip", "install", "x"]}).allowed is False


def test_prompt_does_not_echo_full_file_content():
    prompts = []
    policy = ApprovalPolicy(input_func=lambda prompt: prompts.append(prompt) or "n")
    policy.authorize("write_file", {"path": "a.py", "content": "TOP_SECRET"})
    assert "TOP_SECRET" not in prompts[0]
    assert "10 个字符" in prompts[0]


def test_ask_mode_uses_chinese_confirmation_prompt():
    prompts = []
    policy = ApprovalPolicy(
        mode="ask",
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    decision = policy.authorize("run_command", {"argv": ["pytest"]})

    assert decision == ApprovalDecision(False, "user_denied")
    assert prompts == ['允许执行 run_command {"argv": ["pytest"]}？[y/N/f] ']


def test_ask_mode_displays_redacted_preview_before_confirmation():
    events = []
    policy = ApprovalPolicy(
        input_func=lambda prompt: events.append(("input", prompt)) or "y",
        output_func=lambda message: events.append(("output", message)),
        redact_func=lambda value: value.replace("sk-preview-secret", "[REDACTED]"),
    )

    decision = policy.authorize(
        "write_file",
        {"path": "a.py", "content": "sk-preview-secret"},
        preview={
            "path": "a.py",
            "diff": "--- /dev/null\n+++ b/a.py\n+sk-preview-secret",
            "truncated": False,
        },
    )

    assert decision == ApprovalDecision(True)
    assert events[0] == (
        "output",
        "[变更预览] a.py\n--- /dev/null\n+++ b/a.py\n+[REDACTED]",
    )
    assert events[1][0] == "input"


def test_feedback_denies_current_call_and_returns_user_message():
    answers = iter(["f", "请保留原有函数名"])
    prompts = []
    policy = ApprovalPolicy(input_func=lambda prompt: prompts.append(prompt) or next(answers))

    decision = policy.authorize(
        "replace_in_file",
        {"path": "a.py", "old_text": "old", "new_text": "new", "expected_count": 1},
    )

    assert decision == ApprovalDecision(False, "user_feedback", "请保留原有函数名")
    assert prompts[-1] == "请输入给模型的修改意见："


def test_empty_feedback_falls_back_to_plain_denial():
    answers = iter(["feedback", "   "])
    policy = ApprovalPolicy(input_func=lambda _: next(answers))

    decision = policy.authorize("write_file", {"path": "a.py", "content": "x"})

    assert decision == ApprovalDecision(False, "user_denied")


def test_auto_mode_does_not_render_change_preview():
    output = []
    policy = ApprovalPolicy(mode="auto", output_func=output.append)

    decision = policy.authorize(
        "write_file",
        {"path": "a.py", "content": "x"},
        preview={"path": "a.py", "diff": "+x", "truncated": False},
    )

    assert decision == ApprovalDecision(True)
    assert output == []
