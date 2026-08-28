from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .context import ContextManager
from .tools.files import ToolError


SYSTEM_PROMPT = """You are a coding agent operating inside one restricted workspace.
Inspect relevant files before editing. Use only the supplied tools. Prefer precise
replacements over full rewrites. Run relevant tests after changes. Never assume a
tool succeeded: inspect its structured result. When the task is complete, respond
with a concise summary and do not call another tool."""


@dataclass(frozen=True)
class AgentResult:
    status: str
    message: str
    steps: int


class CodingAgent:
    def __init__(
        self,
        model_client: Any,
        registry: Any,
        approval: Any,
        *,
        max_steps: int = 20,
        max_context_groups: int = 12,
        max_context_chars: int = 60_000,
        run_log: Any | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least one")
        self.model_client = model_client
        self.registry = registry
        self.approval = approval
        self.max_steps = max_steps
        self.max_context_groups = max_context_groups
        self.max_context_chars = max_context_chars
        self.run_log = run_log

    def run(self, task: str) -> AgentResult:
        if self.run_log is not None:
            self.run_log.write("agent_started", {"task": task})
        context = ContextManager(
            SYSTEM_PROMPT,
            task,
            max_groups=self.max_context_groups,
            max_chars=self.max_context_chars,
        )
        definitions = self.registry.definitions()
        known_tools = {
            item["function"]["name"]
            for item in definitions
            if item.get("type") == "function" and "function" in item
        }
        executed_ids: set[str] = set()
        previous_signature: str | None = None
        repeated_signature_count = 0
        consecutive_errors = 0
        invalid_rounds = 0

        for step in range(1, self.max_steps + 1):
            try:
                response = self.model_client.complete(context.messages(), definitions)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                return self._finish("api_error", str(exc), step)

            if not response.tool_calls:
                return self._finish("completed", response.content or "", step)

            group = [response.as_assistant_message()]
            round_invalid = False
            stop_status: str | None = None

            for tool_call in response.tool_calls:
                if stop_status is not None:
                    result = {
                        "status": "error",
                        "error": "skipped_due_to_termination",
                        "message": "A loop termination condition was already reached.",
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    continue

                if tool_call.id in executed_ids:
                    result = {
                        "status": "error",
                        "error": "duplicate_tool_call",
                        "message": "This tool_call_id has already been handled.",
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    continue
                executed_ids.add(tool_call.id)

                try:
                    arguments = json.loads(tool_call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must decode to an object")
                except (json.JSONDecodeError, ValueError) as exc:
                    round_invalid = True
                    result = {
                        "status": "error",
                        "error": "invalid_arguments_json",
                        "message": str(exc),
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    continue

                if tool_call.name not in known_tools:
                    round_invalid = True
                    result = {
                        "status": "error",
                        "error": "unknown_tool",
                        "message": f"Unknown tool: {tool_call.name}",
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    continue

                signature = json.dumps(
                    {"name": tool_call.name, "arguments": arguments},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if signature == previous_signature:
                    repeated_signature_count += 1
                else:
                    previous_signature = signature
                    repeated_signature_count = 1
                if repeated_signature_count >= 3:
                    result = {
                        "status": "error",
                        "error": "repeated_tool_call",
                        "message": "The same normalized tool call appeared three times consecutively.",
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    stop_status = "repeated_tool_call"
                    continue

                allowed, denial_reason = self.approval.authorize(tool_call.name, arguments)
                if not allowed:
                    result = {
                        "status": "permission_denied",
                        "error": "permission_denied",
                        "reason": denial_reason,
                    }
                else:
                    try:
                        result = self.registry.execute(tool_call.name, arguments)
                    except ToolError as exc:
                        result = exc.as_result()
                    except Exception as exc:
                        result = {
                            "status": "error",
                            "error": "tool_execution_error",
                            "message": str(exc),
                        }

                context.record_tool_result(tool_call.name, arguments, result)
                if self.run_log is not None:
                    self.run_log.write(
                        "tool_result",
                        {
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_call.name,
                            "arguments": arguments,
                            "result": result,
                        },
                    )
                group.append(self._tool_message(tool_call.id, result))
                if result.get("status") == "error":
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        stop_status = "consecutive_tool_errors"
                elif result.get("status") == "ok":
                    consecutive_errors = 0

            context.add_group(group)
            if stop_status is not None:
                return self._finish(stop_status, self._termination_message(stop_status), step)

            if round_invalid:
                invalid_rounds += 1
                if invalid_rounds >= 2:
                    return self._finish(
                        "invalid_tool_calls",
                        "The model returned invalid arguments or unknown tools twice consecutively.",
                        step,
                    )
            else:
                invalid_rounds = 0

        return self._finish(
            "max_steps",
            f"The agent reached the configured limit of {self.max_steps} model steps.",
            self.max_steps,
        )

    def _finish(self, status: str, message: str, steps: int) -> AgentResult:
        result = AgentResult(status, message, steps)
        if self.run_log is not None:
            self.run_log.write(
                "agent_finished",
                {"status": result.status, "message": result.message, "steps": result.steps},
            )
        return result

    @staticmethod
    def _tool_message(tool_call_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
        }

    @staticmethod
    def _termination_message(status: str) -> str:
        if status == "repeated_tool_call":
            return "The same normalized tool call appeared three times consecutively."
        return "Three consecutive tool errors occurred."
