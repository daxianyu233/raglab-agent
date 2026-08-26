"""Conversation Event Store Phase 3 回归测试。

运行：
    python -m scripts.test_conversation_event_store

本测试不调用 LLM，不接主 Agent。
"""

from __future__ import annotations

import tempfile

from pathlib import Path

from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = (
            Path(temp_dir)
            / "conversation_events.sqlite3"
        )

        thread_id = "thread-context-test"
        user_id = "local-user"

        turn_1 = "turn-001"
        turn_2 = "turn-002"

        original_tool_result = (
            "Agent-Reach 原始 RAG 资料："
            + "A" * 12000
            + "；ai-memory 不使用向量数据库，"
            "其信息通过 markdown/git repository 管理。"
        )

        store = ConversationEventStore(
            database_path
        )

        print(
            "=" * 80
        )
        print(
            "1. 写入第一轮原始事件"
        )
        print(
            "=" * 80
        )

        human_event, inserted = (
            store.append_event(
                user_id=user_id,
                thread_id=thread_id,
                turn_id=turn_1,
                event_type="message",
                role="human",
                message_id="human-001",
                content_text=(
                    "查询 Agent-Reach 和 ai-memory"
                ),
                payload={
                    "type": "human",
                    "content": (
                        "查询 Agent-Reach 和 ai-memory"
                    ),
                },
            )
        )

        assert inserted is True

        ai_tool_call, inserted = (
            store.append_event(
                user_id=user_id,
                thread_id=thread_id,
                turn_id=turn_1,
                event_type=(
                    "message"
                ),
                role="assistant",
                message_id="ai-001",
                tool_call_id="call-001",
                tool_name=(
                    "search_github_intelligence"
                ),
                content_text="",
                payload={
                    "type": "ai",
                    "tool_calls": [
                        {
                            "id": "call-001",
                            "name": (
                                "search_github_intelligence"
                            ),
                            "args": {
                                "query": (
                                    "Agent-Reach ai-memory"
                                )
                            },
                        }
                    ],
                },
            )
        )

        assert inserted is True

        tool_event, inserted = (
            store.append_event(
                user_id=user_id,
                thread_id=thread_id,
                turn_id=turn_1,
                event_type="message",
                role="tool",
                message_id="tool-001",
                tool_call_id="call-001",
                tool_name=(
                    "search_github_intelligence"
                ),
                content_text=(
                    original_tool_result
                ),
                payload={
                    "type": "tool",
                    "content": (
                        original_tool_result
                    ),
                    "tool_call_id": (
                        "call-001"
                    ),
                },
                metadata={
                    "status": "success"
                },
            )
        )

        assert inserted is True

        ai_final, inserted = (
            store.append_event(
                user_id=user_id,
                thread_id=thread_id,
                turn_id=turn_1,
                event_type="message",
                role="assistant",
                message_id="ai-002",
                content_text=(
                    "已根据 RAG 原始资料完成整理。"
                ),
                payload={
                    "type": "ai",
                    "content": (
                        "已根据 RAG 原始资料完成整理。"
                    ),
                },
            )
        )

        assert inserted is True

        print(
            "第一轮事件数：",
            len(
                store.list_turn_events(
                    thread_id=thread_id,
                    turn_id=turn_1,
                )
            ),
        )

        print()
        print(
            "=" * 80
        )
        print(
            "2. 验证重复归档幂等"
        )
        print(
            "=" * 80
        )

        duplicate, inserted = (
            store.append_event(
                user_id=user_id,
                thread_id=thread_id,
                turn_id=turn_1,
                event_type="message",
                role="tool",
                message_id="tool-001",
                tool_call_id="call-001",
                tool_name=(
                    "search_github_intelligence"
                ),
                content_text=(
                    original_tool_result
                ),
                payload={
                    "type": "tool",
                    "content": (
                        original_tool_result
                    ),
                },
            )
        )

        assert inserted is False

        assert (
            duplicate.event_id
            == tool_event.event_id
        )

        assert (
            store.count_events(
                thread_id=thread_id
            )
            == 4
        )

        print(
            "[PASS] 同一 message_id 重复写入未产生重复事件"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "3. 写入第二轮"
        )
        print(
            "=" * 80
        )

        store.append_event(
            user_id=user_id,
            thread_id=thread_id,
            turn_id=turn_2,
            event_type="message",
            role="human",
            message_id="human-002",
            content_text=(
                "不要重新搜索，"
                "根据刚才的原始资料重新整理。"
            ),
            payload={
                "type": "human",
                "content": (
                    "不要重新搜索，"
                    "根据刚才的原始资料重新整理。"
                ),
            },
        )

        store.append_event(
            user_id=user_id,
            thread_id=thread_id,
            turn_id=turn_2,
            event_type="message",
            role="assistant",
            message_id="ai-003",
            content_text=(
                "已重新整理上一轮原始资料。"
            ),
            payload={
                "type": "ai",
                "content": (
                    "已重新整理上一轮原始资料。"
                ),
            },
        )

        assert (
            store.list_turn_ids(
                thread_id=thread_id
            )
            == [
                turn_1,
                turn_2,
            ]
        )

        print(
            "thread 总事件数：",
            store.count_events(
                thread_id=thread_id
            ),
        )

        print()
        print(
            "=" * 80
        )
        print(
            "4. 验证 Tool 原始正文没有被截断"
        )
        print(
            "=" * 80
        )

        tool_evidence = (
            store.get_tool_evidence(
                thread_id=thread_id,
                turn_id=turn_1,
            )
        )

        assert len(
            tool_evidence
        ) == 1

        restored_tool_text = (
            tool_evidence[0]
            .content_text
        )

        assert (
            restored_tool_text
            == original_tool_result
        )

        print(
            "原始 Tool 字符数：",
            len(
                original_tool_result
            ),
        )

        print(
            "恢复 Tool 字符数：",
            len(
                restored_tool_text
            ),
        )

        print(
            "[PASS] Tool Result 完整恢复"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "5. 验证历史文本可再次检索"
        )
        print(
            "=" * 80
        )

        search_results = (
            store.search_thread_events(
                thread_id=thread_id,
                query="向量数据库",
            )
        )

        assert len(
            search_results
        ) == 1

        assert (
            search_results[0].role
            == "tool"
        )

        print(
            "命中 event_id：",
            search_results[0]
            .event_id,
        )

        print()
        print(
            "=" * 80
        )
        print(
            "6. 关闭并重新打开 SQLite，验证持久化"
        )
        print(
            "=" * 80
        )

        store.close()

        reopened = (
            ConversationEventStore(
                database_path
            )
        )

        assert (
            reopened.count_events(
                thread_id=thread_id
            )
            == 6
        )

        reopened_tool = (
            reopened.get_tool_evidence(
                thread_id=thread_id,
                turn_id=turn_1,
            )
        )

        assert (
            len(
                reopened_tool
            )
            == 1
        )

        assert (
            reopened_tool[0]
            .content_text
            == original_tool_result
        )

        print(
            "[PASS] 进程级重新打开后仍可恢复完整 Tool Result"
        )

        reopened.close()

        print()
        print(
            "=" * 80
        )
        print(
            "Conversation Event Store "
            "Phase 3 回归测试通过"
        )
        print(
            "=" * 80
        )


if __name__ == "__main__":
    main()