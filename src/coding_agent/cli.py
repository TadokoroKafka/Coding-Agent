from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from .agent import CodingAgent
from .approval import ApprovalPolicy
from .model_client import DeepSeekClient
from .run_log import RunLog
from .tools.command import CommandRunner
from .tools.files import Workspace
from .tools.registry import ToolRegistry


class _ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace("usage: ", "用法：", 1)
            .replace("options:\n", "选项：\n", 1)
        )

    def format_usage(self) -> str:
        return super().format_usage().replace("usage: ", "用法：", 1)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须大于零")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = _ChineseArgumentParser(
        prog="coding-agent",
        description="运行一个限制在指定工作区内的 DeepSeek 编程智能体。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--workspace", required=True, help="智能体可以访问的工作区目录。")
    parser.add_argument(
        "--approval-mode",
        choices=("ask", "auto"),
        default="ask",
        help="对副作用操作进行确认，或使用受限的演示命令白名单。",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=20,
        help="模型请求的最大次数（默认：20）。",
    )
    parser.add_argument("--verbose", action="store_true", help="显示经过脱敏的执行进度。")
    return parser


def _verbose_handler(
    output_func: Callable[[str], None],
    redact_func: Callable[[Any], Any],
) -> Callable[[str, dict[str, Any]], None]:
    def handle(event: str, payload: dict[str, Any]) -> None:
        if event == "model_step":
            output_func(f"[步骤 {payload['step']}] 正在请求模型")
        elif event == "tool_call":
            summary = ApprovalPolicy._summarize(payload["tool_name"], payload["arguments"])
            output_func(f"[工具] {payload['tool_name']} {redact_func(summary)}")
        elif event == "tool_result":
            status = payload["result"].get("status", "未知")
            output_func(f"[工具结果] {payload['tool_name']} 状态={status}")

    return handle


def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[[str], None] | None = None,
    error_func: Callable[[str], None] | None = None,
    model_client_factory: Callable[[], Any] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    read_input = input_func or input
    write_output = output_func or print
    write_error = error_func or (lambda message: print(message, file=sys.stderr))

    try:
        task = read_input("任务：").strip()
    except EOFError:
        write_error("任务不能为空。")
        return 2
    except KeyboardInterrupt:
        write_error("用户已中断运行。")
        return 130
    if not task:
        write_error("任务不能为空。")
        return 2

    try:
        workspace = Workspace(args.workspace)
    except (OSError, ValueError) as exc:
        write_error(f"工作区无效：{exc}")
        return 2

    try:
        model_client = (model_client_factory or DeepSeekClient.from_env)()
        command_runner = CommandRunner(workspace)
        registry = ToolRegistry(workspace, command_runner)
        approval = ApprovalPolicy(
            mode=args.approval_mode,
            input_func=read_input,
            output_func=write_output,
        )
        run_log = RunLog(workspace.root)
        event_handler = _verbose_handler(write_output, run_log.redact) if args.verbose else None
        coding_agent = CodingAgent(
            model_client,
            registry,
            approval,
            max_steps=args.max_steps,
            run_log=run_log,
            event_handler=event_handler,
        )
        result = coding_agent.run(task)
    except KeyboardInterrupt:
        write_error("用户已中断运行。")
        return 130
    except Exception as exc:
        write_error(f"智能体初始化失败：{exc}")
        return 1

    if result.status == "completed":
        write_output(result.message)
        return 0
    write_error(f"智能体已停止（{result.status}）：{result.message}")
    return 1
