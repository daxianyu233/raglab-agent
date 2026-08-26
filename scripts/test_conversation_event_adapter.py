"""Conversation Event Adapter Phase 4A 回归测试。

运行：
    python -m scripts.test_conversation_event_adapter

本测试不调用 LLM，不接正式 Agent。
"""

from __future__ import annotations

import tempfile

from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from raglab.agent.conversation_event_adapter import (
    archive_messages_to_event_store,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = (
            Path(temp_dir)
            / "conversation_events.sqlite3"
        )

        store = ConversationEventStore(
            database_path
        )

        user_id = "local-user"
        thread_id = "thread-adapter-test"

        large_tool_result = (
            "GitHub RAG 原始结果："
            + "A" * 12000
        )

        messages = [
            HumanMessage(
                content=(
                    "比较 Agent-Reach 和 ai-memory"
                ),
                id="human-001",
            ),
            AIMessage(
                content="",
                id="ai-001",
                tool_calls=[
                    {
                        "id": "call-001",
                        "name": (
                            "search_github_intelligence"
                        ),
                        "args": {
                            "query": (
                                "Agent-Reach"
                            )
                        },
                        "type": "tool_call",
                    },
                    {
                        "id": "call-002",
                        "name": (
                            "search_github_intelligence"
                        ),
                        "args": {
                            "query": (
                                "ai-memory"
                            )
                        },
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                content=(
                    large_tool_result
                ),
                tool_call_id="call-001",
                name=(
                    "search_github_intelligence"
                ),
                id="tool-001",
            ),
            ToolMessage(
                content=(
                    "ai-memory 原始资料："
                    "通过 markdown/git repository "
                    "管理可持久化信息。"
                ),
                tool_call_id="call-002",
                name=(
                    "search_github_intelligence"
                ),
                id="tool-002",
            ),
            AIMessage(
                content=(
                    "已根据两份原始资料完成比较。"
                ),
                id="ai-002",
            ),
            HumanMessage(
                content=(
                    "把刚才结论改得正式一点。"
                ),
                id="human-002",
            ),
            # 故意不给 id，验证 fallback event_id 幂等。
            AIMessage(
                content=(
                    "已改写为正式表达。"
                ),
            ),
        ]

        print(
            "=" * 80
        )
        print(
            "1. 第一次归档 LangChain Messages"
        )
        print(
            "=" * 80
        )

        report_1 = (
            archive_messages_to_event_store(
                store=store,
                user_id=user_id,
                thread_id=thread_id,
                messages=messages,
            )
        )

        print(
            "扫描消息数：",
            report_1.scanned_message_count,
        )

        print(
            "新增事件数：",
            report_1.inserted_event_count,
        )

        print(
            "已存在事件数：",
            report_1.existing_event_count,
        )

        print(
            "turn_ids：",
            report_1.turn_ids,
        )

        assert (
            report_1.scanned_message_count
            == 7
        )

        assert (
            report_1.inserted_event_count
            == 7
        )

        assert (
            report_1.existing_event_count
            == 0
        )

        assert (
            report_1.turn_ids
            == [
                "turn:human-001",
                "turn:human-002",
            ]
        )

        print()
        print(
            "=" * 80
        )
        print(
            "2. 验证同轮 Human / AI / Tool 归属"
        )
        print(
            "=" * 80
        )

        turn_1_events = (
            store.list_turn_events(
                thread_id=thread_id,
                turn_id="turn:human-001",
            )
        )

        assert len(
            turn_1_events
        ) == 5

        assert [
            event.role
            for event in turn_1_events
        ] == [
            "human",
            "assistant",
            "tool",
            "tool",
            "assistant",
        ]

        print(
            "[PASS] 第一轮消息顺序和 turn 归属正确"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "3. 验证多 Tool Call 完整保存在 payload"
        )
        print(
            "=" * 80
        )

        ai_tool_event = (
            turn_1_events[1]
        )

        tool_calls = (
            ai_tool_event
            .payload
            .get(
                "tool_calls",
                [],
            )
        )

        assert len(
            tool_calls
        ) == 2

        assert (
            ai_tool_event
            .metadata[
                "tool_call_ids"
            ]
            == [
                "call-001",
                "call-002",
            ]
        )

        print(
            "[PASS] 多 Tool Call 没有因顶层单值索引而丢失"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "4. 验证超长 Tool Result 原文完整"
        )
        print(
            "=" * 80
        )

        tool_events = (
            store.get_tool_evidence(
                thread_id=thread_id,
                turn_id="turn:human-001",
            )
        )

        assert len(
            tool_events
        ) == 2

        assert (
            tool_events[0]
            .content_text
            == large_tool_result
        )

        print(
            "原文字符数：",
            len(
                large_tool_result
            ),
        )

        print(
            "恢复字符数：",
            len(
                tool_events[0]
                .content_text
            ),
        )

        print(
            "[PASS] Tool Result 未截断"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "5. 第二次扫描相同 messages，验证幂等"
        )
        print(
            "=" * 80
        )

        report_2 = (
            archive_messages_to_event_store(
                store=store,
                user_id=user_id,
                thread_id=thread_id,
                messages=messages,
            )
        )

        assert (
            report_2.inserted_event_count
            == 0
        )

        assert (
            report_2.existing_event_count
            == 7
        )

        assert (
            store.count_events(
                thread_id=thread_id
            )
            == 7
        )

        print(
            "[PASS] 有 message_id 和无 message_id "
            "两种消息都实现了重复扫描幂等"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "Conversation Event Adapter "
            "Phase 4A 回归测试通过"
        )
        print(
            "=" * 80
        )

        store.close()


if __name__ == "__main__":
    main()