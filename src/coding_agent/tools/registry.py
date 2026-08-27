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
            ("list_files", "List files inside the workspace.", {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string", "default": "*"},
            }, []),
            ("read_file", "Read UTF-8 text with line numbers.", {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1, "default": 1},
                "end_line": {"type": ["integer", "null"], "minimum": 1},
            }, ["path"]),
            ("write_file", "Create or completely overwrite a UTF-8 text file.", {
                "path": {"type": "string"},
                "content": {"type": "string"},
            }, ["path", "content"]),
            ("replace_in_file", "Replace an exact number of text occurrences.", {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "expected_count": {"type": "integer", "minimum": 1},
            }, ["path", "old_text", "new_text", "expected_count"]),
            ("run_command", "Run an argv array without a shell inside the workspace.", {
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
            raise ToolError("invalid_arguments", "Tool arguments must be a JSON object")
        try:
            if name == "list_files":
                return self.workspace.list_files(**arguments)
            if name == "read_file":
                return self.workspace.read_file(**arguments)
            if name == "write_file":
                return self.workspace.write_file(**arguments)
            if name == "replace_in_file":
                return self.workspace.replace_in_file(**arguments)
            if name == "run_command":
                return self.command_runner.run(**arguments)
        except TypeError as exc:
            raise ToolError("invalid_arguments", str(exc)) from exc
        raise ToolError("unknown_tool", f"Unknown tool: {name}")
