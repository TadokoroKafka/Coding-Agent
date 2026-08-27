from __future__ import annotations

from coding_agent.approval import ApprovalPolicy


def test_ask_mode_reads_without_confirmation():
    asked = []
    policy = ApprovalPolicy(input_func=lambda prompt: asked.append(prompt) or "n")
    assert policy.authorize("read_file", {"path": "a.py"}) == (True, None)
    assert asked == []


def test_ask_mode_requires_confirmation_for_write_and_command():
    answers = iter(["y", "n"])
    policy = ApprovalPolicy(input_func=lambda _: next(answers))
    assert policy.authorize("write_file", {"path": "a.py", "content": "secret"}) == (True, None)
    assert policy.authorize("run_command", {"argv": ["python", "-m", "pytest"]}) == (False, "user_denied")


def test_auto_mode_allows_safe_commands_and_rejects_dangerous_ones():
    policy = ApprovalPolicy(mode="auto")
    assert policy.authorize("run_command", {"argv": ["python", "-m", "pytest"]}) == (True, None)
    assert policy.authorize("run_command", {"argv": ["git", "diff"]}) == (True, None)
    assert policy.authorize("run_command", {"argv": ["git", "commit", "-am", "x"]})[0] is False
    assert policy.authorize("run_command", {"argv": ["pip", "install", "x"]})[0] is False


def test_prompt_does_not_echo_full_file_content():
    prompts = []
    policy = ApprovalPolicy(input_func=lambda prompt: prompts.append(prompt) or "n")
    policy.authorize("write_file", {"path": "a.py", "content": "TOP_SECRET"})
    assert "TOP_SECRET" not in prompts[0]
    assert "10 characters" in prompts[0]
