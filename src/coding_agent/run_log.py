from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SENSITIVE_KEY = re.compile(r"api.?key|authorization|bearer|password|secret|token|environment", re.I)
SECRET_VALUE = re.compile(r"\b(?:sk|key)-[A-Za-z0-9_-]{6,}\b", re.I)


class RunLog:
    def __init__(self, workspace: str | Path, *, filename: str | None = None) -> None:
        directory = Path(workspace) / ".coding_agent"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (filename or "runs.jsonl")
        self._known_secret = os.getenv("DEEPSEEK_API_KEY")

    def write(self, event: str, payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": self.redact(payload),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if SENSITIVE_KEY.search(str(key)) else self.redact(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            redacted = value
            if self._known_secret:
                redacted = redacted.replace(self._known_secret, "[REDACTED]")
            return SECRET_VALUE.sub("[REDACTED]", redacted)
        return value
