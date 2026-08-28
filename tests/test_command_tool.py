from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from coding_agent.tools.command import CommandRunner
from coding_agent.tools.files import Workspace


def runner(tmp_path, max_output_chars=20_000):
    return CommandRunner(Workspace(tmp_path), max_output_chars=max_output_chars)


def test_command_success_and_nonzero_exit(tmp_path):
    command = runner(tmp_path)
    success = command.run([sys.executable, "-c", "print('你好')"])
    failure = command.run([sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(3)"])

    assert success["status"] == "ok"
    assert "你好" in success["stdout"]
    assert failure["status"] == "error"
    assert failure["exit_code"] == 3
    assert failure["error"] == "command_failed"


def test_command_timeout_is_reported_once(tmp_path):
    result = runner(tmp_path).run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_seconds=0.2,
    )
    assert result["status"] == "error"
    assert result["timed_out"] is True
    assert result["error"] == "command_timed_out"


def test_command_output_is_truncated_on_character_boundary(tmp_path):
    result = runner(tmp_path, max_output_chars=5).run(
        [sys.executable, "-c", "print('你好世界再见', end='')"]
    )
    assert result["stdout"].startswith("你好世界再")
    assert "已省略" in result["stdout"]


def test_command_rejects_cwd_outside_workspace(tmp_path):
    with pytest.raises(Exception) as exc_info:
        runner(tmp_path).run([sys.executable, "-c", "print(1)"], cwd="..")
    assert getattr(exc_info.value, "code", None) == "path_outside_workspace"


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree behavior")
def test_windows_timeout_kills_child_process_tree(tmp_path):
    marker = tmp_path / "child.pid"
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        f"p=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    result = runner(tmp_path).run([sys.executable, "-c", parent_code], timeout_seconds=1)
    assert result["timed_out"] is True
    child_pid = int(marker.read_text())
    time.sleep(0.2)
    probe = subprocess.run(
        ["tasklist", "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert str(child_pid) not in probe.stdout
