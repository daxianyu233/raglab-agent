"""Context Assembler Phase 6A 回归测试。

运行：
    python -m scripts.test_context_assembler

不调用真实 LLM。
"""

from __future__ import annotations

import tempfile

from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from raglab.agent.context_assembler import (
    ContextAssembler,
)
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
    history_query: str = "",
    previous_answer_required: bool = False,
    raw_tool_evidence_required: bool = False,
) -> ContextPlan:
    return ContextPlan(
        task_intent="test",
        response_goal="test",
        history_required=history_required,
        history_scope=history_scope,
        history_query=history_query,
        previous_answer_required=previous_answer_required,
        raw_tool_evidence_required=raw_tool_evidence_required,
        external_retrieval_required=False,
        external_retrieval_allowed=False,
        long_term_memory_required=False,
        long_term_memory_query="",
        referenced_entities=[],
        temporal_scope="current",
        confidence=1.0,
    )


def append(
    store: ConversationEventStore,
    *,
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
        user_id="local-user",
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
    thread_id: str,
) -> None:
    # Turn 1：带 Tool Evidence。
    turn_1 = "turn:human-001"

    append(
        store,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="human-001",
        role="human",
        content="查询 ai-memory。",
        payload={
            "type": "human",
            "content": "查询 ai-memory。",
            "id": "human-001",
        },
    )

    append(
        store,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="ai-001",
        role="assistant",
        content="",
        payload={
            "type": "ai",
            "content": "",
            "id": "ai-001",
            "tool_calls": [
                {
                    "id": "call-001",
                    "name": "search_history",
                    "args": {
                        "query": "ai-memory"
                    },
                    "type": "tool_call",
                }
            ],
            "invalid_tool_calls": [],
        },
        tool_call_id="call-001",
        tool_name="search_history",
    )

    append(
        store,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="tool-001",
        role="tool",
        content=(
            "ai-memory 原始 Tool Evidence："
            "使用 markdown/git 管理长期信息。"
        ),
        payload={
            "type": "tool",
            "content": (
                "ai-memory 原始 Tool Evidence："
                "使用 markdown/git 管理长期信息。"
            ),
            "id": "tool-001",
            "tool_call_id": "call-001",
            "name": "search_history",
        },
        tool_call_id="call-001",
        tool_name="search_history",
    )

    append(
        store,
        thread_id=thread_id,
        turn_id=turn_1,
        message_id="ai-002",
        role="assistant",
        content=(
            "ai-memory 采用可审计的文本化长期信息方案。"
        ),
        payload={
            "type": "ai",
            "content": (
                "ai-memory 采用可审计的文本化长期信息方案。"
            ),
            "id": "ai-002",
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    )

    # Turn 2：纯问答，作为 previous_turn。
    turn_2 = "turn:human-002"

    append(
        store,
        thread_id=thread_id,
        turn_id=turn_2,
        message_id="human-002",
        role="human",
        content="把结论写正式一点。",
        payload={
            "type": "human",
            "content": "把结论写正式一点。",
            "id": "human-002",
        },
    )

    append(
        store,
        thread_id=thread_id,
        turn_id=turn_2,
        message_id="ai-003",
        role="assistant",
        content="已改写为正式表达。",
        payload={
            "type": "ai",
            "content": "已改写为正式表达。",
            "id": "ai-003",
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = (
            Path(temp_dir)
            / "conversation_events.sqlite3"
        )

        store = ConversationEventStore(
            db_path
        )

        thread_id = "thread-assembler-test"
        build_history(store, thread_id)

        retriever = ConversationRetriever(
            store=store,
            recent_turn_limit=2,
            historical_turn_limit=2,
        )
        assembler = ContextAssembler()

        print("=" * 80)
        print("Case 1：previous answer -> model_input")
        print("=" * 80)

        retrieval_1 = retriever.retrieve(
            thread_id=thread_id,
            plan=make_plan(
                history_required=True,
                history_scope="previous_turn",
                previous_answer_required=True,
            ),
        )

        result_1 = assembler.assemble(
            system_prompt="你是测试 Agent。",
            current_messages=[
                HumanMessage(
                    content="再简洁一点。",
                    id="human-current-001",
                )
            ],
            conversation_retrieval=retrieval_1,
        )

        assert result_1.source_turn_ids == [
            "turn:human-002"
        ]
        assert result_1.history_message_count == 2
        assert result_1.messages[-1].content == "再简洁一点。"
        assert result_1.tool_pair_integrity_ok is True

        print(
            "组装消息数：",
            len(result_1.messages),
        )
        print(
            "估算 tokens：",
            result_1.estimated_message_tokens,
        )
        print(
            "[PASS] previous answer 已进入真正 model_input"
        )

        print()
        print("=" * 80)
        print("Case 2：raw Tool Evidence -> Hydrate -> model_input")
        print("=" * 80)

        retrieval_2 = retriever.retrieve(
            thread_id=thread_id,
            plan=make_plan(
                history_required=True,
                history_scope="historical_search",
                history_query="ai-memory markdown 原始资料",
                raw_tool_evidence_required=True,
            ),
        )

        result_2 = assembler.assemble(
            system_prompt="你是测试 Agent。",
            current_messages=[
                HumanMessage(
                    content="根据当时原始资料重新解释。",
                    id="human-current-002",
                )
            ],
            conversation_retrieval=retrieval_2,
        )

        history_types = [
            type(message).__name__
            for message in result_2.messages[-4:-1]
        ]

        assert history_types == [
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
        ]
        assert result_2.tool_pair_integrity_ok is True

        hydrated_ai = result_2.messages[-3]
        hydrated_tool = result_2.messages[-2]

        assert isinstance(hydrated_ai, AIMessage)
        assert isinstance(hydrated_tool, ToolMessage)
        assert hydrated_ai.tool_calls[0]["id"] == "call-001"
        assert hydrated_tool.tool_call_id == "call-001"

        print(
            "[PASS] Conversation Event 已反向恢复为完整 Tool Pair"
        )

        print()
        print("=" * 80)
        print("Case 3：当前轮 Tool Pair + 历史 Tool Pair")
        print("=" * 80)

        current_messages = [
            HumanMessage(
                content="当前轮再查一个资料。",
                id="human-current-003",
            ),
            AIMessage(
                content="",
                id="ai-current-003",
                tool_calls=[
                    {
                        "id": "call-current-003",
                        "name": "search_current",
                        "args": {
                            "query": "current"
                        },
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="当前轮 Tool Result",
                id="tool-current-003",
                tool_call_id="call-current-003",
                name="search_current",
            ),
        ]

        result_3 = assembler.assemble(
            system_prompt="你是测试 Agent。",
            current_messages=current_messages,
            conversation_retrieval=retrieval_2,
            long_term_memory_text="用户偏好简洁技术说明。",
            thread_summary="此前线程讨论过 Agent 架构。",
        )

        assert result_3.tool_pair_integrity_ok is True
        assert (
            result_3.context_audit[
                "unresolved_tool_call_count"
            ]
            == 0
        )
        assert (
            result_3.context_audit[
                "orphan_tool_message_count"
            ]
            == 0
        )

        print(
            "[PASS] 历史 Tool Pair 和当前 Tool Pair "
            "可以同时存在且协议完整"
        )

        print()
        print("=" * 80)
        print("Case 4：独立新请求，不加入 Conversation History")
        print("=" * 80)

        result_4 = assembler.assemble(
            system_prompt="你是测试 Agent。",
            current_messages=[
                HumanMessage(
                    content="给我今天的日报。",
                    id="human-current-004",
                )
            ],
            conversation_retrieval=None,
        )

        assert result_4.history_message_count == 0
        assert result_4.source_turn_ids == []
        assert len(result_4.messages) == 2

        print(
            "[PASS] 独立任务只有 System + 当前 Human"
        )

        print()
        print("=" * 80)
        print(
            "Context Assembler Phase 6A 回归测试通过"
        )
        print("=" * 80)

        store.close()


if __name__ == "__main__":
    main()