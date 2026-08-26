"""Conversation Event Store Phase 4B Runtime 集成回归测试。

运行：
    python -m scripts.test_conversation_event_runtime_integration

本测试：
- 不调用真实 LLM；
- 不执行真实 Tool；
- 模拟一次 HITL interrupt -> /approve resume；
- 验证 SecureAgentRuntime 自动归档消息。
"""

from __future__ import annotations

import tempfile

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.runtime_security import (
    SecureAgentRuntime,
)


@tool
def dummy_tool(
    value: str,
) -> str:
    """Phase 4B 测试用 Tool。"""

    return value


class FakePolicyStore:
    """本测试不会真正执行 SecureToolNode。"""

    pass


class FakeGraph:
    def __init__(
        self,
    ) -> None:
        self.pending = False

        self.human = HumanMessage(
            content="执行一次需要审批的测试操作",
            id="human-runtime-001",
        )

        self.ai_call = AIMessage(
            content="",
            id="ai-runtime-001",
            tool_calls=[
                {
                    "id": "call-runtime-001",
                    "name": "dummy_tool",
                    "args": {
                        "value": "hello"
                    },
                    "type": "tool_call",
                }
            ],
        )

    def get_state(
        self,
        config: Any,
    ) -> Any:
        if not self.pending:
            return SimpleNamespace(
                tasks=[]
            )

        interrupt_value = {
            "tool_name": "dummy_tool",
            "tool_call_id": (
                "call-runtime-001"
            ),
        }

        return SimpleNamespace(
            tasks=[
                SimpleNamespace(
                    interrupts=[
                        SimpleNamespace(
                            value=(
                                interrupt_value
                            )
                        )
                    ]
                )
            ]
        )

    def invoke(
        self,
        command: Any,
        *,
        config: Any,
        context: Any,
    ) -> dict[str, Any]:
        if not self.pending:
            raise RuntimeError(
                "FakeGraph 当前没有待恢复 HITL。"
            )

        self.pending = False

        return {
            "messages": [
                self.human,
                self.ai_call,
                ToolMessage(
                    content=(
                        "dummy_tool 的完整原始结果"
                    ),
                    tool_call_id=(
                        "call-runtime-001"
                    ),
                    name="dummy_tool",
                    id="tool-runtime-001",
                ),
                AIMessage(
                    content=(
                        "审批后已根据 Tool Result "
                        "生成最终回答。"
                    ),
                    id="ai-runtime-002",
                ),
            ],
            "turn_llm_calls": 2,
            "turn_tool_calls": 1,
            "turn_summary_calls": 0,
            "summary_updated": False,
            "summarized_turns_this_run": 0,
            "stopped_by_max_steps": False,
            "model_trace": [],
            "tool_trace": [],
        }


class FakeBaseAgent:
    """只提供 SecureAgentRuntime 本测试所需接口。"""

    def __init__(
        self,
    ) -> None:
        self.tools = [
            dummy_tool
        ]

        self.graph = FakeGraph()

        self.tool_node = None

    def _refresh_tool_bindings(
        self,
    ):
        return list(
            self.tools
        )

    def _build_config(
        self,
        thread_id: str,
    ) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": thread_id
            }
        }

    def run(
        self,
        question: str,
        *,
        thread_id: str,
        user_id: str,
    ) -> Any:
        # 模拟：
        #
        # Human
        #   ↓
        # AI(tool_call)
        #   ↓
        # interrupt
        self.graph.pending = True

        return SimpleNamespace(
            thread_id=thread_id,
            question=question,
            answer="",
            messages=[
                self.graph.human,
                self.graph.ai_call,
            ],
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = (
            Path(temp_dir)
            / "conversation_events.sqlite3"
        )

        event_store = (
            ConversationEventStore(
                database_path
            )
        )

        runtime = SecureAgentRuntime(
            FakeBaseAgent(),
            policy_store=(
                FakePolicyStore()
            ),
            conversation_event_store=(
                event_store
            ),
        )

        thread_id = (
            "thread-runtime-integration"
        )

        user_id = "local-user"

        print(
            "=" * 80
        )
        print(
            "1. 普通 run() 停在 HITL 前"
        )
        print(
            "=" * 80
        )

        first_result = runtime.run(
            "执行一次需要审批的测试操作",
            thread_id=thread_id,
            user_id=user_id,
        )

        assert len(
            first_result.messages
        ) == 2

        first_events = (
            event_store
            .list_thread_events(
                thread_id
            )
        )

        assert len(
            first_events
        ) == 2

        assert [
            event.role
            for event in first_events
        ] == [
            "human",
            "assistant",
        ]

        print(
            "HITL 前 Event 数：",
            len(
                first_events
            ),
        )

        assert (
            runtime
            .last_conversation_archive_report
            .inserted_event_count
            == 2
        )

        print(
            "[PASS] HITL 前 Human + AI(tool_call) "
            "已经自动归档"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "2. /approve 恢复"
        )
        print(
            "=" * 80
        )

        resumed = runtime.run(
            "/approve",
            thread_id=thread_id,
            user_id=user_id,
        )

        assert (
            resumed.answer
            == (
                "审批后已根据 Tool Result "
                "生成最终回答。"
            )
        )

        all_events = (
            event_store
            .list_thread_events(
                thread_id
            )
        )

        assert len(
            all_events
        ) == 4

        assert [
            event.role
            for event in all_events
        ] == [
            "human",
            "assistant",
            "tool",
            "assistant",
        ]

        print(
            "HITL 恢复后 Event 数：",
            len(
                all_events
            ),
        )

        archive_report = (
            runtime
            .last_conversation_archive_report
        )

        assert (
            archive_report
            is not None
        )

        assert (
            archive_report
            .inserted_event_count
            == 2
        )

        assert (
            archive_report
            .existing_event_count
            == 2
        )

        print(
            "[PASS] resume 时旧消息幂等跳过，"
            "只补入 Tool + AI(final)"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "3. 验证 Tool Evidence 可从 Event Store 恢复"
        )
        print(
            "=" * 80
        )

        tool_events = (
            event_store
            .get_tool_evidence(
                thread_id=thread_id,
                turn_id=(
                    "turn:human-runtime-001"
                ),
            )
        )

        assert len(
            tool_events
        ) == 1

        assert (
            tool_events[0]
            .tool_call_id
            == "call-runtime-001"
        )

        assert (
            tool_events[0]
            .content_text
            == (
                "dummy_tool 的完整原始结果"
            )
        )

        print(
            "[PASS] Tool Result 已按原始内容持久化"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "Conversation Event Runtime "
            "Phase 4B 回归测试通过"
        )
        print(
            "=" * 80
        )

        event_store.close()


if __name__ == "__main__":
    main()