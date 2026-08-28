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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Run a workspace-restricted DeepSeek coding agent.",
    )
    parser.add_argument("--workspace", required=True, help="Workspace directory the agent may access.")
    parser.add_argument(
        "--approval-mode",
        choices=("ask", "auto"),
        default="ask",
        help="Require confirmation for side effects, or use the restricted demo allowlist.",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=20,
        help="Maximum number of model requests (default: 20).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show redacted execution progress.")
    return parser


def _verbose_handler(
    output_func: Callable[[str], None],
    redact_func: Callable[[Any], Any],
) -> Callable[[str, dict[str, Any]], None]:
    def handle(event: str, payload: dict[str, Any]) -> None:
        if event == "model_step":
            output_func(f"[step {payload['step']}] requesting model")
        elif event == "tool_call":
            summary = ApprovalPolicy._summarize(payload["tool_name"], payload["arguments"])
            output_func(f"[tool] {payload['tool_name']} {redact_func(summary)}")
        elif event == "tool_result":
            status = payload["result"].get("status", "unknown")
            output_func(f"[tool result] {payload['tool_name']} status={status}")

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
        task = read_input("Task: ").strip()
    except EOFError:
        write_error("Task must not be empty.")
        return 2
    except KeyboardInterrupt:
        write_error("Interrupted by user.")
        return 130
    if not task:
        write_error("Task must not be empty.")
        return 2

    try:
        workspace = Workspace(args.workspace)
    except (OSError, ValueError) as exc:
        write_error(f"Invalid workspace: {exc}")
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
        write_error("Interrupted by user.")
        return 130
    except Exception as exc:
        write_error(f"Agent initialization failed: {exc}")
        return 1

    if result.status == "completed":
        write_output(result.message)
        return 0
    write_error(f"Agent stopped ({result.status}): {result.message}")
    return 1
