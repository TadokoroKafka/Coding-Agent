from __future__ import annotations

import locale
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .files import ToolError, Workspace


class CommandRunner:
    """Run one command without a shell and clean up its process tree on timeout."""

    def __init__(self, workspace: Workspace, max_output_chars: int = 20_000) -> None:
        self.workspace = workspace
        self.max_output_chars = max_output_chars

    def run(
        self,
        argv: list[str],
        cwd: str = ".",
        timeout_seconds: float = 60,
    ) -> dict[str, Any]:
        if not isinstance(argv, list) or not argv or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise ToolError("invalid_arguments", "argv must be a non-empty string array")
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 300:
            raise ToolError("invalid_arguments", "timeout_seconds must be between 0 and 300")

        command_cwd = self.workspace.resolve_path(cwd)
        if not command_cwd.is_dir():
            raise ToolError("not_a_directory", f"Command directory does not exist: {cwd}")

        popen_options: dict[str, Any] = {
            "args": argv,
            "cwd": command_cwd,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
        }
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True

        started = time.monotonic()
        try:
            process = subprocess.Popen(**popen_options)
        except (OSError, ValueError) as exc:
            return self._result(
                started=started,
                status="error",
                exit_code=None,
                stdout=b"",
                stderr=str(exc).encode("utf-8", errors="replace"),
                timed_out=False,
                error="command_start_failed",
            )

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_tree(process)
            stdout, stderr = process.communicate()

        status = "ok" if process.returncode == 0 and not timed_out else "error"
        error = "command_timed_out" if timed_out else (
            "command_failed" if process.returncode else None
        )
        return self._result(
            started=started,
            status=status,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            error=error,
        )

    def _terminate_tree(self, process: subprocess.Popen[bytes]) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=0.5)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    def _result(
        self,
        *,
        started: float,
        status: str,
        exit_code: int | None,
        stdout: bytes,
        stderr: bytes,
        timed_out: bool,
        error: str | None,
    ) -> dict[str, Any]:
        encoding = locale.getpreferredencoding(False) or "utf-8"
        result: dict[str, Any] = {
            "status": status,
            "exit_code": exit_code,
            "stdout": self._truncate(stdout.decode(encoding, errors="replace")),
            "stderr": self._truncate(stderr.decode(encoding, errors="replace")),
            "duration_ms": round((time.monotonic() - started) * 1000),
            "timed_out": timed_out,
        }
        if error:
            result["error"] = error
        return result

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        omitted = len(text) - self.max_output_chars
        return f"{text[: self.max_output_chars]}\n... [{omitted} characters omitted]"
