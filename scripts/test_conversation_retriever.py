"""Conversation Retriever Phase 5 回归测试。

运行：
    python -m scripts.test_conversation_retriever

不调用 LLM。
"""

from __future__ import annotations

import tempfile

from pathlib import Path

from raglab.agent.context_plan import (
    ContextPlan,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.conversation_retriever import (
    ConversationRetriever,
)


def make_plan(
    *,
    history_required: bool,
    history_scope: str,
    history_query: str | None = None,
    previous_answer_required: bool = False,
    raw_tool_evidence_required: bool = False,
) -> ContextPlan:
    return ContextPlan(
        task_intent="test",
        response_goal="test",
        history_required=(
            history_required
        ),
        history_scope=(
            history_scope
        ),
        history_query=(
            history_query
        ),
        previous_answer_required=(
            previous_answer_required
        ),
        raw_tool_evidence_required=(
            raw_tool_evidence_required
        ),
        external_retrieval_required=False,
        external_retrieval_allowed=False,
        long_term_memory_required=False,
        long_term_memory_query=None,
        referenced_entities=[],
        temporal_scope=None,
        confidence=1.0,
    )


def append_message(
    store: ConversationEventStore,
    *,
    user_id: str,
    thread_id: str,
    turn_id: str,
    message_id: str,
    role: str,
    content: str,
    payload: dict,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> None:
    store.append_event(
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_id,
        event_type="message",
        role=role,
        message_id=message_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content_text=content,
        payload=payload,
    )


def build_history(
    store: ConversationEventStore,
    *,
    user_id: str,
    thread_id: str,
) -> None:
    # --------------------------------------------------------
    # Turn 1：更早的 ai-memory 讨论
    # --------------------------------------------------------
    turn_1 = "turn:human-001"

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="human-001",
        role="human",
        content=(
            "分析 ai-memory 为什么不使用传统向量数据库。"
        ),
        payload={
            "type": "human",
            "content": (
                "分析 ai-memory 为什么不使用传统向量数据库。"
            ),
        },
    )

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="ai-001",
        role="assistant",
        content="",
        payload={
            "type": "ai",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-001",
                    "name": (
                        "search_github_intelligence"
                    ),
                    "args": {
                        "query": "ai-memory"
                    },
                }
            ],
        },
        tool_call_id="call-001",
        tool_name=(
            "search_github_intelligence"
        ),
    )

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="tool-001",
        role="tool",
        content=(
            "ai-memory 的原始资料："
            "该项目使用 markdown 和 git repository "
            "维护长期信息，而不是传统向量数据库。"
        ),
        payload={
            "type": "tool",
            "content": (
                "ai-memory 的原始资料："
                "该项目使用 markdown 和 git repository "
                "维护长期信息，而不是传统向量数据库。"
            ),
            "tool_call_id": "call-001",
        },
        tool_call_id="call-001",
        tool_name=(
            "search_github_intelligence"
        ),
    )

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="ai-002",
        role="assistant",
        content=(
            "ai-memory 采用 markdown/git 的可审计存储思路。"
        ),
        payload={
            "type": "ai",
            "content": (
                "ai-memory 采用 markdown/git 的可审计存储思路。"
            ),
            "tool_calls": [],
        },
    )

    # --------------------------------------------------------
    # Turn 2：另一个主题，有 Tool Evidence
    # --------------------------------------------------------
    turn_2 = "turn:human-002"

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_2,
        message_id="human-002",
        role="human",
        content=(
            "查询 Agent-Reach 的工具路由设计。"
        ),
        payload={
            "type": "human",
            "content": (
                "查询 Agent-Reach 的工具路由设计。"
            ),
        },
    )

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_2,
        message_id="ai-003",
        role="assistant",
        content="",
        payload={
            "type": "ai",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-002",
                    "name": (
                        "search_github_intelligence"
                    ),
                    "args": {
                        "query": (
                            "Agent-Reach"
                        )
                    },
                }
            ],
        },
        tool_call_id="call-002",
        tool_name=(
            "search_github_intelligence"
        ),
    )

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_2,
        message_id="tool-002",
        role="tool",
        content=(
            "Agent-Reach 原始 Tool Evidence："
            "支持工具路由和互联网访问。"
        ),
        payload={
            "type": "tool",
            "content": (
                "Agent-Reach 原始 Tool Evidence："
                "支持工具路由和互联网访问。"
            ),
            "tool_call_id": "call-002",
        },
        tool_call_id="call-002",
        tool_name=(
            "search_github_intelligence"
        ),
    )

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_2,
        message_id="ai-004",
        role="assistant",
        content=(
            "Agent-Reach 的核心是工具路由与外部访问。"
        ),
        payload={
            "type": "ai",
            "content": (
                "Agent-Reach 的核心是工具路由与外部访问。"
            ),
            "tool_calls": [],
        },
    )

    # --------------------------------------------------------
    # Turn 3：最近一轮纯对话
    # --------------------------------------------------------
    turn_3 = "turn:human-003"

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_3,
        message_id="human-003",
        role="human",
        content=(
            "把刚才结论写得正式一点。"
        ),
        payload={
            "type": "human",
            "content": (
                "把刚才结论写得正式一点。"
            ),
        },
    )

    append_message(
        store,
        user_id=user_id,
        thread_id=thread_id,
        turn_id=turn_3,
        message_id="ai-005",
        role="assistant",
        content=(
            "已将 Agent-Reach 的结论改写为正式表达。"
        ),
        payload={
            "type": "ai",
            "content": (
                "已将 Agent-Reach 的结论改写为正式表达。"
            ),
            "tool_calls": [],
        },
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
        thread_id = (
            "thread-retriever-test"
        )

        build_history(
            store,
            user_id=user_id,
            thread_id=thread_id,
        )

        retriever = ConversationRetriever(
            store=store,
            recent_turn_limit=2,
            historical_turn_limit=2,
        )

        # ----------------------------------------------------
        # Case 1：上一轮最终 Answer
        # ----------------------------------------------------
        print(
            "=" * 80
        )
        print(
            "Case 1：previous_turn + previous_answer"
        )
        print(
            "=" * 80
        )

        result_1 = retriever.retrieve(
            thread_id=thread_id,
            plan=make_plan(
                history_required=True,
                history_scope=(
                    "previous_turn"
                ),
                previous_answer_required=True,
            ),
        )

        assert (
            result_1.selected_turn_ids
            == [
                "turn:human-003"
            ]
        )

        roles_1 = [
            event.role
            for event
            in result_1.turns[
                0
            ].selected_events
        ]

        assert roles_1 == [
            "human",
            "assistant",
        ]

        assert (
            result_1.turns[
                0
            ].selected_events[
                -1
            ].content_text
            == (
                "已将 Agent-Reach 的结论改写为正式表达。"
            )
        )

        print(
            "[PASS] 精确恢复上一轮 Human + 最终 AI Answer"
        )

        # ----------------------------------------------------
        # Case 2：上一轮原始 Tool Evidence
        # 当前轮已经存入时也必须正确排除。
        # ----------------------------------------------------
        print()
        print(
            "=" * 80
        )
        print(
            "Case 2：current_turn 已入库时恢复上一轮 Tool Evidence"
        )
        print(
            "=" * 80
        )

        current_turn = (
            "turn:human-004"
        )

        append_message(
            store,
            user_id=user_id,
            thread_id=thread_id,
            turn_id=current_turn,
            message_id="human-004",
            role="human",
            content=(
                "不要重新搜索，直接使用上一轮原始资料。"
            ),
            payload={
                "type": "human",
                "content": (
                    "不要重新搜索，直接使用上一轮原始资料。"
                ),
            },
        )

        # 上一轮是 turn-003，没有 Tool。
        # 因此这里显式测试 current_turn 排除是否正确，
        # 并允许结果中只有 Human / AI tool pair 选择规则的实际值。
        result_2a = retriever.retrieve(
            thread_id=thread_id,
            current_turn_id=(
                current_turn
            ),
            plan=make_plan(
                history_required=True,
                history_scope=(
                    "previous_turn"
                ),
                raw_tool_evidence_required=True,
            ),
        )

        assert (
            result_2a.selected_turn_ids
            == [
                "turn:human-003"
            ]
        )

        # 再直接指定历史搜索 Agent-Reach，
        # 验证 Tool Pair + Tool Evidence。
        result_2 = retriever.retrieve(
            thread_id=thread_id,
            current_turn_id=(
                current_turn
            ),
            plan=make_plan(
                history_required=True,
                history_scope=(
                    "historical_search"
                ),
                history_query=(
                    "Agent-Reach 工具路由 原始资料"
                ),
                raw_tool_evidence_required=True,
            ),
        )

        assert (
            "turn:human-002"
            in result_2.selected_turn_ids
        )

        agent_turn = next(
            turn
            for turn
            in result_2.turns
            if turn.turn_id
            == "turn:human-002"
        )

        roles_2 = [
            event.role
            for event
            in agent_turn.selected_events
        ]

        assert roles_2 == [
            "human",
            "assistant",
            "tool",
        ]

        assert (
            agent_turn
            .selected_events[
                -1
            ].content_text
            == (
                "Agent-Reach 原始 Tool Evidence："
                "支持工具路由和互联网访问。"
            )
        )

        print(
            "[PASS] raw_tool_evidence 会恢复 AI tool_call + Tool 原文"
        )

        # ----------------------------------------------------
        # Case 3：更早历史 ai-memory
        # ----------------------------------------------------
        print()
        print(
            "=" * 80
        )
        print(
            "Case 3：historical_search 找回十几轮前类型的历史"
        )
        print(
            "=" * 80
        )

        result_3 = retriever.retrieve(
            thread_id=thread_id,
            current_turn_id=(
                current_turn
            ),
            plan=make_plan(
                history_required=True,
                history_scope=(
                    "historical_search"
                ),
                history_query=(
                    "ai-memory 不用向量数据库 原始资料"
                ),
                raw_tool_evidence_required=True,
            ),
        )

        assert result_3.turns

        assert (
            result_3.turns[
                0
            ].turn_id
            == "turn:human-001"
        )

        assert any(
            event.role == "tool"
            and "markdown"
            in event.content_text
            for event
            in result_3.turns[
                0
            ].selected_events
        )

        print(
            "Top-1 turn：",
            result_3.turns[
                0
            ].turn_id,
        )

        print(
            "Top-1 score：",
            round(
                result_3.turns[
                    0
                ].retrieval_score,
                4,
            ),
        )

        print(
            "matched_terms：",
            result_3.turns[
                0
            ].matched_terms,
        )

        print(
            "[PASS] 更早 ai-memory 原始 Tool Evidence 被重新找回"
        )

        # ----------------------------------------------------
        # Case 4：none 不读取任何历史
        # ----------------------------------------------------
        print()
        print(
            "=" * 80
        )
        print(
            "Case 4：history_scope=none"
        )
        print(
            "=" * 80
        )

        result_4 = retriever.retrieve(
            thread_id=thread_id,
            current_turn_id=(
                current_turn
            ),
            plan=make_plan(
                history_required=False,
                history_scope="none",
            ),
        )

        assert (
            result_4.selected_turn_ids
            == []
        )

        assert (
            result_4.selected_event_count
            == 0
        )

        print(
            "[PASS] 独立新任务不会读取任何 Conversation History"
        )

        # ----------------------------------------------------
        # Case 5：recent_turns
        # ----------------------------------------------------
        print()
        print(
            "=" * 80
        )
        print(
            "Case 5：recent_turns"
        )
        print(
            "=" * 80
        )

        result_5 = retriever.retrieve(
            thread_id=thread_id,
            current_turn_id=(
                current_turn
            ),
            plan=make_plan(
                history_required=True,
                history_scope=(
                    "recent_turns"
                ),
                previous_answer_required=True,
            ),
        )

        assert (
            result_5.selected_turn_ids
            == [
                "turn:human-002",
                "turn:human-003",
            ]
        )

        print(
            "最近历史 turns：",
            result_5.selected_turn_ids,
        )

        print(
            "[PASS] recent_turns 只恢复最近指定数量历史"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "Conversation Retriever Phase 5 回归测试通过"
        )
        print(
            "=" * 80
        )

        store.close()


if __name__ == "__main__":
    main()