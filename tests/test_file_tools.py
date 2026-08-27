from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_agent.tools import ToolError, Workspace


def test_list_read_write_and_replace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    result = workspace.write_file("src/hello.py", "greeting = '你好'\n")
    assert result == {"status": "ok", "path": "src/hello.py", "characters": 16}
    assert workspace.list_files(pattern="**/*.py")["files"] == ["src/hello.py"]

    read_result = workspace.read_file("src/hello.py")
    assert read_result["content"] == "1: greeting = '你好'"

    replace_result = workspace.replace_in_file("src/hello.py", "你好", "南京大学")
    assert replace_result["replacements"] == 1
    assert (tmp_path / "src" / "hello.py").read_text(encoding="utf-8") == "greeting = '南京大学'\n"


def test_replace_count_mismatch_does_not_modify_file(tmp_path: Path) -> None:
    source = tmp_path / "value.txt"
    source.write_text("same same", encoding="utf-8")
    workspace = Workspace(tmp_path)

    with pytest.raises(ToolError, match="Expected 1 matches but found 2") as captured:
        workspace.replace_in_file("value.txt", "same", "changed", expected_count=1)

    assert captured.value.code == "match_count_mismatch"
    assert source.read_text(encoding="utf-8") == "same same"


@pytest.mark.parametrize("path", ["../outside.txt", "..\\outside.txt"])
def test_parent_traversal_is_rejected(tmp_path: Path, path: str) -> None:
    workspace = Workspace(tmp_path)
    with pytest.raises(ToolError) as captured:
        workspace.write_file(path, "unsafe")
    assert captured.value.code == "path_outside_workspace"


def test_absolute_path_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    with pytest.raises(ToolError) as captured:
        workspace.read_file(str(tmp_path / "absolute.txt"))
    assert captured.value.code == "path_outside_workspace"


@pytest.mark.skipif(os.name == "nt", reason="Creating symlinks may require Windows developer mode")
def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-agent-workspace"
    outside.mkdir(exist_ok=True)
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    workspace = Workspace(tmp_path)

    with pytest.raises(ToolError) as captured:
        workspace.write_file("escape/file.txt", "unsafe")
    assert captured.value.code == "path_outside_workspace"


def test_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\xfa")
    workspace = Workspace(tmp_path)
    with pytest.raises(ToolError) as captured:
        workspace.read_file("binary.txt")
    assert captured.value.code == "unsupported_encoding"


def test_unicode_output_is_truncated_on_character_boundary(tmp_path: Path) -> None:
    (tmp_path / "unicode.txt").write_text("你好世界", encoding="utf-8")
    workspace = Workspace(tmp_path, max_read_chars=6)

    result = workspace.read_file("unicode.txt")

    assert result["truncated"] is True
    assert result["content"].startswith("1: 你好世")
    result["content"].encode("utf-8")


def test_line_range_validation(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    with pytest.raises(ToolError) as captured:
        workspace.read_file("anything.txt", start_line=3, end_line=2)
    assert captured.value.code == "invalid_line_range"
