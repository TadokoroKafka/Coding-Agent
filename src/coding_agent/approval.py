from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


SIDE_EFFECT_TOOLS = {"write_file", "replace_in_file", "run_command"}


@dataclass(frozen=True)
class ApprovalDecision:
    allowed: bool
    reason: str | None = None
    feedback: str | None = None


class ApprovalPolicy:
    def __init__(
        self,
        mode: str = "ask",
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
        redact_func: Callable[[Any], Any] | None = None,
    ) -> None:
        if mode not in {"ask", "auto"}:
            raise ValueError("approval mode 必须是 'ask' 或 'auto'")
        self.mode = mode
        self.input_func = input_func
        self.output_func = output_func
        self.redact_func = redact_func or (lambda value: value)

    def authorize(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        preview: dict[str, Any] | None = None,
    ) -> ApprovalDecision:
        if tool_name not in SIDE_EFFECT_TOOLS:
            return ApprovalDecision(True)
        if self.mode == "auto":
            if tool_name == "run_command" and not self._is_safe_auto_command(arguments):
                return ApprovalDecision(False, "command_not_allowed_in_auto_mode")
            return ApprovalDecision(True)

        if preview is not None:
            path = self.redact_func(preview.get("path", "未知路径"))
            diff = self.redact_func(preview.get("diff", "（无文本差异）"))
            self.output_func(f"[变更预览] {path}\n{diff}")

        summary = self._summarize(tool_name, arguments)
        response = self.input_func(f"允许执行 {tool_name} {summary}？[y/N/f] ").strip().lower()
        if response in {"y", "yes"}:
            return ApprovalDecision(True)
        if response in {"f", "feedback"}:
            feedback = self.input_func("请输入给模型的修改意见：").strip()
            if feedback:
                return ApprovalDecision(False, "user_feedback", feedback)
        return ApprovalDecision(False, "user_denied")

    @staticmethod
    def _is_safe_auto_command(arguments: dict[str, Any]) -> bool:
        argv = arguments.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            return False
        executable = os.path.basename(argv[0]).lower()
        if executable in {"python", "python.exe", "pytest", "pytest.exe"}:
            return True
        if executable in {"git", "git.exe"}:
            return len(argv) >= 2 and argv[1].lower() in {"status", "diff"}
        return False

    @staticmethod
    def _summarize(tool_name: str, arguments: dict[str, Any]) -> str:
        safe = dict(arguments)
        if "content" in safe:
            safe["content"] = f"<{len(str(safe['content']))} 个字符>"
        if "old_text" in safe:
            safe["old_text"] = f"<{len(str(safe['old_text']))} 个字符>"
        if "new_text" in safe:
            safe["new_text"] = f"<{len(str(safe['new_text']))} 个字符>"
        rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True)
        return rendered[:500] + ("..." if len(rendered) > 500 else "")
