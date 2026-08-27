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
            raise ValueError(f"Workspace is not a directory: {candidate}")
        self.root = candidate
        self.max_read_chars = max_read_chars

    def resolve_path(self, value: str, *, must_exist: bool = False) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ToolError("invalid_path", "Path must be a non-empty string.")
        raw = Path(value)
        if raw.is_absolute() or ".." in PurePath(value).parts:
            raise ToolError("path_outside_workspace", "Absolute paths and '..' are not allowed.")
        resolved = (self.root / raw).resolve(strict=must_exist)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ToolError("path_outside_workspace", "Resolved path escapes the workspace.") from exc
        return resolved

    def list_files(self, path: str = ".", pattern: str = "**/*") -> dict[str, Any]:
        base = self.resolve_path(path, must_exist=True)
        if not base.is_dir():
            raise ToolError("not_a_directory", f"Not a directory: {path}")
        if Path(pattern).is_absolute() or ".." in PurePath(pattern).parts:
            raise ToolError("invalid_pattern", "Pattern must stay inside the selected directory.")

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
            raise ToolError("invalid_line_range", "Line numbers must satisfy 1 <= start_line <= end_line.")
        file_path = self.resolve_path(path, must_exist=True)
        if not file_path.is_file():
            raise ToolError("not_a_file", f"Not a file: {path}")
        try:
            text = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolError("unsupported_encoding", "Only UTF-8 text files are supported.") from exc

        all_lines = text.splitlines()
        selected = all_lines[start_line - 1 : end_line]
        numbered = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, start=start_line))
        truncated = len(numbered) > self.max_read_chars
        if truncated:
            numbered = numbered[: self.max_read_chars] + "\n[output truncated]"
        return {
            "status": "ok",
            "path": file_path.relative_to(self.root).as_posix(),
            "content": numbered,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
            "total_lines": len(all_lines),
            "truncated": truncated,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ToolError("invalid_content", "File content must be a string.")
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
            raise ToolError("invalid_old_text", "old_text must not be empty.")
        if expected_count < 1:
            raise ToolError("invalid_expected_count", "expected_count must be at least one.")
        file_path = self.resolve_path(path, must_exist=True)
        try:
            content = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolError("unsupported_encoding", "Only UTF-8 text files are supported.") from exc
        actual_count = content.count(old_text)
        if actual_count != expected_count:
            raise ToolError(
                "match_count_mismatch",
                f"Expected {expected_count} matches but found {actual_count}; file was not changed.",
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
