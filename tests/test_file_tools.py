from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_agent.tools import ToolError, Workspace
from coding_agent.tools.registry import ToolRegistry


def test_list_files_defaults_to_recursive_discovery(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("value = 1\n", encoding="utf-8")

    result = Workspace(tmp_path).list_files()

    assert result["files"] == ["src/package/module.py"]


def test_list_files_tool_schema_uses_recursive_default(tmp_path: Path) -> None:
    definitions = ToolRegistry(Workspace(tmp_path)).definitions()
    list_files = next(
        item["function"]
        for item in definitions
        if item["function"]["name"] == "list_files"
    )

    assert list_files["parameters"]["properties"]["pattern"]["default"] == "**/*"
    assert "递归" in list_files["description"]


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

    with pytest.raises(ToolError, match="预期匹配 1 次，实际匹配 2 次") as captured:
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


def test_tool_error_message_is_chinese(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(ToolError) as captured:
        workspace.write_file("../escape.txt", "unsafe")

    assert captured.value.code == "path_outside_workspace"
    assert captured.value.message == "不允许使用绝对路径或 '..'。"


def test_search_text_finds_literal_matches_with_glob_and_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("first\nNeedle here\n", encoding="utf-8")
    (tmp_path / "src" / "b.txt").write_text("Needle ignored\n", encoding="utf-8")

    result = Workspace(tmp_path).search_text(
        "Needle",
        pattern="**/*.py",
        case_sensitive=True,
    )

    assert result == {
        "status": "ok",
        "matches": [{"path": "src/a.py", "line": 2, "text": "Needle here"}],
        "count": 1,
        "truncated": False,
        "skipped_files": 0,
    }


def test_search_text_is_case_insensitive_and_obeys_result_limit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("needle one\nNEEDLE two\n", encoding="utf-8")

    result = Workspace(tmp_path).search_text("Needle", max_results=1)

    assert result["matches"] == [{"path": "a.py", "line": 1, "text": "needle one"}]
    assert result["count"] == 1
    assert result["truncated"] is True


def test_search_text_skips_non_utf8_and_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / "valid.txt").write_text("目标文本\n", encoding="utf-8")
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\xfa")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "hidden.py").write_text("目标文本\n", encoding="utf-8")

    result = Workspace(tmp_path).search_text("目标文本")

    assert result["matches"] == [{"path": "valid.txt", "line": 1, "text": "目标文本"}]
    assert result["skipped_files"] == 1


@pytest.mark.parametrize(
    ("arguments", "error_code"),
    [
        ({"query": ""}, "invalid_query"),
        ({"query": "x", "max_results": 0}, "invalid_max_results"),
        ({"query": "x", "max_results": 201}, "invalid_max_results"),
        ({"query": "x", "path": ".."}, "path_outside_workspace"),
    ],
)
def test_search_text_rejects_invalid_inputs(tmp_path: Path, arguments: dict, error_code: str) -> None:
    with pytest.raises(ToolError) as captured:
        Workspace(tmp_path).search_text(**arguments)

    assert captured.value.code == error_code
