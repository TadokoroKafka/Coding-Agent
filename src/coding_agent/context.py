from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionState:
    read_files: set[str] = field(default_factory=set)
    modified_files: set[str] = field(default_factory=set)
    searches: list[dict[str, Any]] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    latest_test_result: dict[str, Any] | None = None
    failures: list[str] = field(default_factory=list)

    def record(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        path = arguments.get("path")
        if tool_name == "read_file" and isinstance(path, str) and result.get("status") == "ok":
            self.read_files.add(path)
        if tool_name in {"write_file", "replace_in_file"} and isinstance(path, str) and result.get("status") == "ok":
            self.modified_files.add(path)
        if tool_name == "search_text" and result.get("status") == "ok":
            self.searches.append(
                {
                    "query": arguments.get("query"),
                    "path": arguments.get("path", "."),
                    "pattern": arguments.get("pattern", "**/*"),
                    "count": result.get("count", 0),
                    "truncated": result.get("truncated", False),
                }
            )
            self.searches = self.searches[-12:]
        if tool_name == "run_command":
            argv = arguments.get("argv")
            if isinstance(argv, list):
                command = " ".join(str(part) for part in argv)
                self.commands.append(command)
                self.commands = self.commands[-12:]
                if any("pytest" in str(part).lower() for part in argv):
                    self.latest_test_result = self._result_summary(result)
        if result.get("status") != "ok":
            failure = result.get("error") or result.get("message") or "未知错误"
            self.failures.append(f"{tool_name}: {failure}")
            self.failures = self.failures[-8:]

    def snapshot(self) -> dict[str, Any]:
        return {
            "read_files": sorted(self.read_files),
            "modified_files": sorted(self.modified_files),
            "recent_searches": self.searches,
            "recent_commands": self.commands,
            "latest_test_result": self.latest_test_result,
            "recent_failures": self.failures,
        }

    @staticmethod
    def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": result.get("status"),
            "exit_code": result.get("exit_code"),
            "timed_out": result.get("timed_out", False),
        }


class ContextManager:
    """Keep complete assistant/tool groups and prune only at group boundaries."""

    def __init__(
        self,
        system_prompt: str,
        user_task: str,
        *,
        max_groups: int = 12,
        max_chars: int = 60_000,
    ) -> None:
        self.system_message = {"role": "system", "content": system_prompt}
        self.user_message = {"role": "user", "content": user_task}
        self.max_groups = max_groups
        self.max_chars = max_chars
        self.groups: list[list[dict[str, Any]]] = []
        self.state = ExecutionState()

    def add_group(self, messages: list[dict[str, Any]]) -> None:
        if not messages or messages[0].get("role") != "assistant":
            raise ValueError("交互组必须以 assistant 消息开始")
        self.groups.append(messages)

    def record_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.state.record(tool_name, arguments, result)

    def messages(self) -> list[dict[str, Any]]:
        selected = self.groups[-self.max_groups :]
        dropped = len(selected) < len(self.groups)
        base = [self.system_message, self.user_message]

        while len(selected) > 1 and self._character_count(base, selected) > self.max_chars:
            selected = selected[1:]
            dropped = True

        messages = list(base)
        if dropped:
            snapshot = json.dumps(self.state.snapshot(), ensure_ascii=False, sort_keys=True)
            messages.append(
                {
                    "role": "system",
                    "content": f"历史裁剪后的确定性执行状态快照：\n{snapshot}",
                }
            )
        for group in selected:
            messages.extend(group)
        return messages

    @staticmethod
    def _character_count(
        base: list[dict[str, Any]],
        groups: list[list[dict[str, Any]]],
    ) -> int:
        return len(json.dumps([*base, *[message for group in groups for message in group]], ensure_ascii=False))
