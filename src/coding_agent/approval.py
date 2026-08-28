from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any


SIDE_EFFECT_TOOLS = {"write_file", "replace_in_file", "run_command"}


class ApprovalPolicy:
    def __init__(
        self,
        mode: str = "ask",
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        if mode not in {"ask", "auto"}:
            raise ValueError("approval mode 必须是 'ask' 或 'auto'")
        self.mode = mode
        self.input_func = input_func
        self.output_func = output_func

    def authorize(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        if tool_name not in SIDE_EFFECT_TOOLS:
            return True, None
        if self.mode == "auto":
            if tool_name == "run_command" and not self._is_safe_auto_command(arguments):
                return False, "command_not_allowed_in_auto_mode"
            return True, None

        summary = self._summarize(tool_name, arguments)
        response = self.input_func(f"允许执行 {tool_name} {summary}？[y/N] ").strip().lower()
        if response in {"y", "yes"}:
            return True, None
        return False, "user_denied"

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
