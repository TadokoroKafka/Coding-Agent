from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePath
from typing import Any


class ToolError(RuntimeError):
    """A safe, user-facing failure from a local tool."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_result(self) -> dict[str, Any]:
        return {"status": "error", "error": self.code, "message": self.message}


class Workspace:
    """UTF-8 text operations restricted to one resolved directory."""

    IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".coding_agent"}

    def __init__(self, root: str | Path, *, max_read_chars: int = 20_000) -> None:
        candidate = Path(root).expanduser().resolve(strict=True)
        if not candidate.is_dir():
            raise ValueError(f"工作区不是目录：{candidate}")
        self.root = candidate
        self.max_read_chars = max_read_chars

    def resolve_path(self, value: str, *, must_exist: bool = False) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ToolError("invalid_path", "路径必须是非空字符串。")
        raw = Path(value)
        if raw.is_absolute() or ".." in PurePath(value).parts:
            raise ToolError("path_outside_workspace", "不允许使用绝对路径或 '..'。")
        resolved = (self.root / raw).resolve(strict=must_exist)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ToolError("path_outside_workspace", "解析后的路径超出了工作区。") from exc
        return resolved

    def list_files(self, path: str = ".", pattern: str = "**/*") -> dict[str, Any]:
        base = self.resolve_path(path, must_exist=True)
        if not base.is_dir():
            raise ToolError("not_a_directory", f"不是目录：{path}")
        if Path(pattern).is_absolute() or ".." in PurePath(pattern).parts:
            raise ToolError("invalid_pattern", "匹配模式必须限制在所选目录内。")

        files: list[str] = []
        for item in base.glob(pattern):
            if not item.is_file():
                continue
            relative = item.relative_to(self.root)
            if any(part in self.IGNORED_PARTS for part in relative.parts):
                continue
            files.append(relative.as_posix())
            if len(files) == 500:
                break
        files.sort()
        return {"status": "ok", "files": files, "count": len(files), "truncated": len(files) == 500}

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise ToolError("invalid_line_range", "行号必须满足 1 <= start_line <= end_line。")
        file_path = self.resolve_path(path, must_exist=True)
        if not file_path.is_file():
            raise ToolError("not_a_file", f"不是文件：{path}")
        try:
            text = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolError("unsupported_encoding", "仅支持 UTF-8 文本文件。") from exc

        all_lines = text.splitlines()
        selected = all_lines[start_line - 1 : end_line]
        numbered = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start=start_line))
        truncated = len(numbered) > self.max_read_chars
        if truncated:
            numbered = numbered[: self.max_read_chars] + "\n[输出已截断]"
        return {
            "status": "ok",
            "path": file_path.relative_to(self.root).as_posix(),
            "content": numbered,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
            "total_lines": len(all_lines),
            "truncated": truncated,
        }

    def search_text(
        self,
        query: str,
        path: str = ".",
        pattern: str = "**/*",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query:
            raise ToolError("invalid_query", "搜索文本必须是非空字符串。")
        if not isinstance(case_sensitive, bool):
            raise ToolError("invalid_case_sensitive", "case_sensitive 必须是布尔值。")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 200:
            raise ToolError("invalid_max_results", "max_results 必须是 1 到 200 之间的整数。")

        base = self.resolve_path(path, must_exist=True)
        if not base.is_dir():
            raise ToolError("not_a_directory", f"不是目录：{path}")
        if Path(pattern).is_absolute() or ".." in PurePath(pattern).parts:
            raise ToolError("invalid_pattern", "匹配模式必须限制在所选目录内。")

        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        skipped_files = 0
        truncated = False

        for item in sorted(base.glob(pattern)):
            try:
                relative = item.relative_to(self.root)
            except ValueError:
                skipped_files += 1
                continue
            if any(part in self.IGNORED_PARTS for part in relative.parts):
                continue
            try:
                file_path = self.resolve_path(relative.as_posix(), must_exist=True)
            except (OSError, ToolError):
                skipped_files += 1
                continue
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                skipped_files += 1
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle not in haystack:
                    continue
                if len(matches) >= max_results:
                    truncated = True
                    break
                matches.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
            if truncated:
                break

        return {
            "status": "ok",
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
            "skipped_files": skipped_files,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ToolError("invalid_content", "文件内容必须是字符串。")
        file_path = self.resolve_path(path)
        self._atomic_write(file_path, content)
        return {
            "status": "ok",
            "path": file_path.relative_to(self.root).as_posix(),
            "characters": len(content),
        }

    def replace_in_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_count: int = 1,
    ) -> dict[str, Any]:
        if not old_text:
            raise ToolError("invalid_old_text", "old_text 不能为空。")
        if expected_count < 1:
            raise ToolError("invalid_expected_count", "expected_count 必须至少为 1。")
        file_path = self.resolve_path(path, must_exist=True)
        try:
            content = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolError("unsupported_encoding", "仅支持 UTF-8 文本文件。") from exc
        actual_count = content.count(old_text)
        if actual_count != expected_count:
            raise ToolError(
                "match_count_mismatch",
                f"预期匹配 {expected_count} 次，实际匹配 {actual_count} 次；文件未修改。",
            )
        self._atomic_write(file_path, content.replace(old_text, new_text))
        return {
            "status": "ok",
            "path": file_path.relative_to(self.root).as_posix(),
            "replacements": actual_count,
        }

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_name = temporary.name
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
