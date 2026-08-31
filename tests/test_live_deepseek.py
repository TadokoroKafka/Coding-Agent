from __future__ import annotations

import json
import os
from typing import Any

import pytest

from coding_agent.agent import CodingAgent
from coding_agent.approval import ApprovalPolicy
from coding_agent.model_client import DeepSeekClient, ModelResponse
from coding_agent.tools import Workspace
from coding_agent.tools.registry import ToolRegistry


class RecordingClient:
    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client
        self.responses: list[ModelResponse] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        response = self.client.complete(messages, tools)
        self.responses.append(response)
        return response


class ReadOnlyToolRegistry(ToolRegistry):
    def definitions(self) -> list[dict[str, Any]]:
        return [
            definition
            for definition in super().definitions()
            if definition["function"]["name"] == "read_file"
        ]


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="未设置 DEEPSEEK_API_KEY；跳过真实 DeepSeek 测试。",
)
def test_real_deepseek_thinking_tool_call_round_trip(tmp_path):
    (tmp_path / "probe.txt").write_text("live tool calling probe", encoding="utf-8")
    workspace = Workspace(tmp_path)
    client = RecordingClient(
        DeepSeekClient(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            thinking=True,
            max_attempts=1,
        )
    )
    agent = CodingAgent(
        client,
        ReadOnlyToolRegistry(workspace),
        ApprovalPolicy(mode="ask"),
        max_steps=4,
    )

    result = agent.run(
        "请严格按以下步骤执行：第一步必须且只调用 read_file 读取 probe.txt；"
        "读取成功后不要再调用任何工具，最终回答中必须包含 LIVE_OK。"
    )

    assert result.status == "completed"
    assert "LIVE_OK" in result.message
    assert result.summary["read_files"] == ["probe.txt"]
    assert len(client.responses) >= 2

    first_response = client.responses[0]
    assert first_response.reasoning_content
    assert len(first_response.tool_calls) == 1
    read_call = first_response.tool_calls[0]
    assert read_call.name == "read_file"
    assert read_call.id
    arguments = json.loads(read_call.arguments)
    assert isinstance(arguments, dict)
    assert arguments["path"] == "probe.txt"
