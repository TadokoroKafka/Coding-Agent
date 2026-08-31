from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .context import ContextManager
from .tools.files import ToolError


SYSTEM_PROMPT = """你是一个只能在指定工作区内操作的编程智能体。
修改前先检查相关文件，只使用提供的工具。优先进行精确替换，避免不必要地重写整个文件。
定位代码时优先使用 search_text 搜索相关文本，再按需读取具体文件，避免无目的地广泛读取。
修改后运行相关测试。不要假设工具已经成功执行，必须检查其结构化结果。
任务完成后，用中文给出简洁总结，不要再调用工具。"""


@dataclass(frozen=True)
class AgentResult:
    status: str
    message: str
    steps: int
    summary: dict[str, Any] = field(default_factory=dict, compare=False)


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
        event_handler: Any | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须至少为 1")
        self.model_client = model_client
        self.registry = registry
        self.approval = approval
        self.max_steps = max_steps
        self.max_context_groups = max_context_groups
        self.max_context_chars = max_context_chars
        self.run_log = run_log
        self.event_handler = event_handler

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
            self._emit("model_step", {"step": step})
            try:
                response = self.model_client.complete(context.messages(), definitions)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                return self._finish("api_error", str(exc), step, context.state.snapshot())

            if not response.tool_calls:
                if response.content is None or not response.content.strip():
                    return self._finish(
                        "empty_response",
                        "模型未返回最终文本或工具调用，请重试。",
                        step,
                        context.state.snapshot(),
                    )
                return self._finish("completed", response.content, step, context.state.snapshot())

            group = [response.as_assistant_message()]
            round_invalid = False
            stop_status: str | None = None

            for tool_call in response.tool_calls:
                if stop_status is not None:
                    result = {
                        "status": "error",
                        "error": "skipped_due_to_termination",
                        "message": "已经触发循环终止条件，因此跳过此工具调用。",
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    continue

                if tool_call.id in executed_ids:
                    result = {
                        "status": "error",
                        "error": "duplicate_tool_call",
                        "message": "此 tool_call_id 已经处理过。",
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    continue
                executed_ids.add(tool_call.id)

                try:
                    arguments = json.loads(tool_call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments 解析后必须是 JSON 对象")
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
                        "message": f"未知工具：{tool_call.name}",
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
                        "message": "规范化后的同一工具调用连续出现了三次。",
                    }
                    group.append(self._tool_message(tool_call.id, result))
                    stop_status = "repeated_tool_call"
                    continue

                self._emit(
                    "tool_call",
                    {
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "arguments": arguments,
                    },
                )
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
                self._emit(
                    "tool_result",
                    {
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "result": result,
                    },
                )
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
                return self._finish(
                    stop_status,
                    self._termination_message(stop_status),
                    step,
                    context.state.snapshot(),
                )

            if round_invalid:
                invalid_rounds += 1
                if invalid_rounds >= 2:
                    return self._finish(
                        "invalid_tool_calls",
                        "模型连续两次返回了非法参数或未知工具。",
                        step,
                        context.state.snapshot(),
                    )
            else:
                invalid_rounds = 0

        return self._finish(
            "max_steps",
            f"智能体已达到配置的 {self.max_steps} 次模型请求上限。",
            self.max_steps,
            context.state.snapshot(),
        )

    def _finish(
        self,
        status: str,
        message: str,
        steps: int,
        summary: dict[str, Any],
    ) -> AgentResult:
        result = AgentResult(status, message, steps, summary)
        if self.run_log is not None:
            self.run_log.write(
                "agent_finished",
                {
                    "status": result.status,
                    "message": result.message,
                    "steps": result.steps,
                    "summary": result.summary,
                },
            )
        return result

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.event_handler is not None:
            self.event_handler(event, payload)

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
            return "规范化后的同一工具调用连续出现了三次。"
        return "连续发生了三次工具错误。"
