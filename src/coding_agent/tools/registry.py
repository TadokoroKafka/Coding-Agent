from __future__ import annotations

from typing import Any

from .command import CommandRunner
from .files import ToolError, Workspace


class ToolRegistry:
    def __init__(self, workspace: Workspace, command_runner: CommandRunner | None = None) -> None:
        self.workspace = workspace
        self.command_runner = command_runner or CommandRunner(workspace)

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        schemas = [
            ("list_files", "递归列出工作区内的文件。", {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "**/*"},
            }, []),
            ("read_file", "按行号读取 UTF-8 文本。", {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1, "default": 1},
                "end_line": {"type": ["integer", "null"], "minimum": 1},
            }, ["path"]),
            ("search_text", "在工作区 UTF-8 文本中按字面量搜索并返回文件、行号和匹配行。", {
                "query": {"type": "string", "minLength": 1},
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "**/*"},
                "case_sensitive": {"type": "boolean", "default": False},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
            }, ["query"]),
            ("write_file", "创建或完整覆盖 UTF-8 文本文件。", {
                "path": {"type": "string"},
                "content": {"type": "string"},
            }, ["path", "content"]),
            ("replace_in_file", "按照指定匹配次数精确替换文本。", {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "expected_count": {"type": "integer", "minimum": 1},
            }, ["path", "old_text", "new_text", "expected_count"]),
            ("run_command", "在工作区内不经过 shell 执行 argv 参数数组。", {
                "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "cwd": {"type": "string", "default": "."},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 300, "default": 60},
            }, ["argv"]),
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
            for name, description, properties, required in schemas
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ToolError("invalid_arguments", "工具参数必须是 JSON 对象。")
        try:
            if name == "list_files":
                return self.workspace.list_files(**arguments)
            if name == "read_file":
                return self.workspace.read_file(**arguments)
            if name == "search_text":
                return self.workspace.search_text(**arguments)
            if name == "write_file":
                return self.workspace.write_file(**arguments)
            if name == "replace_in_file":
                return self.workspace.replace_in_file(**arguments)
            if name == "run_command":
                return self.command_runner.run(**arguments)
        except TypeError as exc:
            raise ToolError("invalid_arguments", str(exc)) from exc
        raise ToolError("unknown_tool", f"未知工具：{name}")

    def preview(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(arguments, dict):
            raise ToolError("invalid_arguments", "工具参数必须是 JSON 对象。")
        try:
            if name == "write_file":
                return self.workspace.preview_write(**arguments)
            if name == "replace_in_file":
                return self.workspace.preview_replace(**arguments)
        except TypeError as exc:
            raise ToolError("invalid_arguments", str(exc)) from exc
        return None
