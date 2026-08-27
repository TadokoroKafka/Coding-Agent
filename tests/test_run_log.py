from __future__ import annotations

import json

from coding_agent.run_log import RunLog


def test_log_redacts_keys_headers_environment_and_secret_values(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key-current-secret")
    log = RunLog(tmp_path, filename="test.jsonl")
    log.write(
        "request",
        {
            "api_key": "key-current-secret",
            "Authorization": "Bearer abc",
            "environment": {"PATH": "many values"},
            "message": "do not leak sk-abcdef123456 or key-current-secret",
            "safe": "visible",
        },
    )
    raw = log.path.read_text(encoding="utf-8")
    assert "key-current-secret" not in raw
    assert "sk-abcdef123456" not in raw
    assert "Bearer abc" not in raw
    assert "many values" not in raw
    assert json.loads(raw)["payload"]["safe"] == "visible"
