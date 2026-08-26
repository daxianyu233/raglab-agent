"""Context Compression Phase 6C 回归测试。

运行：
    python -m scripts.test_context_compression

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
from raglab.agent.context_budget import (
    ContextBudgetConfig,
)
from raglab.agent.context_compression import (
    ContextBudgetExceededError,
    ContextCompressor,
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
    history_scope: str = "none",
    history_query: str | None = None,
    previous_answer_required: bool = False,
    raw_tool_evidence_required: bool = False,
) -> ContextPlan:
    history_required = (
        history_scope
        != "none"
    )

    return ContextPlan(
        task_intent="test",
        response_goal="test",
        history_required=history_required,
        history_scope=history_scope,
        history_query=history_query,
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


def append_event(
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


def build_large_tool_history(
    store: ConversationEventStore,
    *,
    thread_id: str,
) -> str:
    turn_id = "turn:history-tool-001"
    raw_tool_text = (
        "原始历史资料："
        + "A" * 12000
        + "：原始资料结束"
    )

    append_event(
        store,
        thread_id=thread_id,
        turn_id=turn_id,
        message_id="human-history-tool-001",
        role="human",
        content="查询历史大型 Tool Evidence。",
        payload={
            "type": "human",
            "content": "查询历史大型 Tool Evidence。",
            "id": "human-history-tool-001",
        },
    )

    append_event(
        store,
        thread_id=thread_id,
        turn_id=turn_id,
        message_id="ai-history-tool-001",
        role="assistant",
        content="",
        payload={
            "type": "ai",
            "content": "",
            "id": "ai-history-tool-001",
            "tool_calls": [
                {
                    "id": "call-history-tool-001",
                    "name": "search_history",
                    "args": {
                        "query": "大型 Tool Evidence"
                    },
                    "type": "tool_call",
                }
            ],
            "invalid_tool_calls": [],
        },
        tool_call_id=(
            "call-history-tool-001"
        ),
        tool_name="search_history",
    )

    append_event(
        store,
        thread_id=thread_id,
        turn_id=turn_id,
        message_id="tool-history-tool-001",
        role="tool",
        content=raw_tool_text,
        payload={
            "type": "tool",
            "content": raw_tool_text,
            "id": "tool-history-tool-001",
            "tool_call_id": (
                "call-history-tool-001"
            ),
            "name": "search_history",
        },
        tool_call_id=(
            "call-history-tool-001"
        ),
        tool_name="search_history",
    )

    append_event(
        store,
        thread_id=thread_id,
        turn_id=turn_id,
        message_id="ai-history-tool-final",
        role="assistant",
        content="历史资料查询完成。",
        payload={
            "type": "ai",
            "content": "历史资料查询完成。",
            "id": "ai-history-tool-final",
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    )

    return raw_tool_text


def build_recent_turns(
    store: ConversationEventStore,
    *,
    thread_id: str,
) -> None:
    for index in range(
        1,
        4,
    ):
        turn_id = (
            f"turn:recent-{index:03d}"
        )

        human_id = (
            f"human-recent-{index:03d}"
        )

        ai_id = (
            f"ai-recent-{index:03d}"
        )

        append_event(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            message_id=human_id,
            role="human",
            content=(
                f"最近历史问题 {index}"
            ),
            payload={
                "type": "human",
                "content": (
                    f"最近历史问题 {index}"
                ),
                "id": human_id,
            },
        )

        answer = (
            f"最近历史回答 {index}："
            + (
                chr(
                    ord("a")
                    + index
                )
                * 2400
            )
        )

        append_event(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            message_id=ai_id,
            role="assistant",
            content=answer,
            payload={
                "type": "ai",
                "content": answer,
                "id": ai_id,
                "tool_calls": [],
                "invalid_tool_calls": [],
            },
        )


def main() -> None:
    assembler = ContextAssembler()
    compressor = ContextCompressor()

    # ========================================================
    # Case 1：历史巨大 Tool Result
    # ========================================================
    print("=" * 80)
    print("Case 1：历史 Tool Result 超预算 -> 压 Context View，不改 Event Store")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ConversationEventStore(
            Path(temp_dir)
            / "events.sqlite3"
        )

        thread_id = "thread-compress-history-tool"

        original_tool_text = (
            build_large_tool_history(
                store,
                thread_id=thread_id,
            )
        )

        retriever = ConversationRetriever(
            store=store
        )

        plan = make_plan(
            history_scope=(
                "historical_search"
            ),
            history_query=(
                "大型 Tool Evidence 原始历史资料"
            ),
            raw_tool_evidence_required=True,
        )

        retrieval = retriever.retrieve(
            thread_id=thread_id,
            plan=plan,
        )

        assembly = assembler.assemble(
            system_prompt="你是测试 Agent。",
            thread_summary=(
                "这是很长的派生摘要。"
                + "摘要" * 500
            ),
            conversation_retrieval=retrieval,
            current_messages=[
                HumanMessage(
                    content=(
                        "请基于当时原始 Tool Evidence 重新说明。"
                    ),
                    id="human-current-001",
                )
            ],
        )

        result = compressor.compress_to_fit(
            assembly=assembly,
            budget_config=ContextBudgetConfig(
                model_context_limit_tokens=2200,
                reserved_output_tokens=400,
                tool_schema_tokens=150,
                safety_margin_tokens=100,
            ),
            plan=plan,
        )

        assert result.final_budget.fits is True

        action_names = [
            action.action
            for action in result.actions
        ]

        assert (
            "remove_message"
            in action_names
        )

        assert (
            "truncate_tool_result"
            in action_names
        )

        assert (
            result.assembly
            .tool_pair_integrity_ok
            is True
        )

        hydrated_tool = next(
            message
            for message
            in result.assembly.messages
            if isinstance(
                message,
                ToolMessage,
            )
        )

        assert (
            hydrated_tool.tool_call_id
            == "call-history-tool-001"
        )

        assert (
            "Conversation Event Store"
            in str(
                hydrated_tool.content
            )
        )

        # 最关键：SQLite 原始 Event 完全没改。
        raw_events = (
            store
            .get_tool_evidence(
                thread_id=thread_id,
                turn_id=(
                    "turn:history-tool-001"
                ),
            )
        )

        assert len(
            raw_events
        ) == 1

        assert (
            raw_events[0].content_text
            == original_tool_text
        )

        print(
            "压缩前消息 tokens：",
            result.initial_budget
            .estimated_message_tokens,
        )

        print(
            "压缩后消息 tokens：",
            result.final_budget
            .estimated_message_tokens,
        )

        print(
            "节省 tokens：",
            result.tokens_saved,
        )

        print(
            "Actions：",
            action_names,
        )

        print(
            "[PASS] 仅压缩本轮 Context View，Event Store 原始 Tool Result 完整"
        )

        store.close()

    # ========================================================
    # Case 2：多个历史 turn -> 整轮裁剪
    # ========================================================
    print()
    print("=" * 80)
    print("Case 2：多个 recent turns 超预算 -> 删除低优先级整轮")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ConversationEventStore(
            Path(temp_dir)
            / "events.sqlite3"
        )

        thread_id = "thread-compress-recent"

        build_recent_turns(
            store,
            thread_id=thread_id,
        )

        retriever = ConversationRetriever(
            store=store,
            recent_turn_limit=3,
        )

        plan = make_plan(
            history_scope="recent_turns",
            previous_answer_required=True,
        )

        retrieval = retriever.retrieve(
            thread_id=thread_id,
            plan=plan,
        )

        assembly = assembler.assemble(
            system_prompt="你是测试 Agent。",
            conversation_retrieval=retrieval,
            current_messages=[
                HumanMessage(
                    content="综合最近讨论继续回答。",
                    id="human-current-002",
                )
            ],
        )

        result = compressor.compress_to_fit(
            assembly=assembly,
            budget_config=ContextBudgetConfig(
                model_context_limit_tokens=1500,
                reserved_output_tokens=300,
                tool_schema_tokens=100,
                safety_margin_tokens=100,
            ),
            plan=plan,
        )

        assert result.final_budget.fits is True

        assert (
            len(
                result.removed_turn_ids
            )
            >= 1
        )

        assert (
            "turn:recent-001"
            in result.removed_turn_ids
        )

        assert (
            "turn:recent-003"
            in result.assembly.source_turn_ids
        )

        assert (
            result.assembly
            .tool_pair_integrity_ok
            is True
        )

        print(
            "删除 turns：",
            result.removed_turn_ids,
        )

        print(
            "保留 turns：",
            result.assembly.source_turn_ids,
        )

        print(
            "[PASS] recent_turns 按整轮裁剪，较新的历史优先保留"
        )

        store.close()

    # ========================================================
    # Case 3：当前巨大 Tool Result
    # ========================================================
    print()
    print("=" * 80)
    print("Case 3：当前轮巨大 Tool Result -> 压内容但不破坏 Tool Pair")
    print("=" * 80)

    current_human_text = (
        "查询当前大型资料，但必须保留我的这个完整问题。"
    )

    assembly = assembler.assemble(
        system_prompt="你是测试 Agent。",
        current_messages=[
            HumanMessage(
                content=(
                    current_human_text
                ),
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
                            "query": "large"
                        },
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=(
                    "B" * 14000
                ),
                id="tool-current-003",
                tool_call_id=(
                    "call-current-003"
                ),
                name="search_current",
            ),
        ],
    )

    result = compressor.compress_to_fit(
        assembly=assembly,
        budget_config=ContextBudgetConfig(
            model_context_limit_tokens=1800,
            reserved_output_tokens=350,
            tool_schema_tokens=150,
            safety_margin_tokens=100,
        ),
    )

    assert result.final_budget.fits is True

    current_human = next(
        message
        for message
        in result.assembly.messages
        if isinstance(
            message,
            HumanMessage,
        )
    )

    assert (
        current_human.content
        == current_human_text
    )

    current_tool = next(
        message
        for message
        in result.assembly.messages
        if isinstance(
            message,
            ToolMessage,
        )
    )

    assert (
        current_tool.tool_call_id
        == "call-current-003"
    )

    assert (
        result.assembly
        .tool_pair_integrity_ok
        is True
    )

    print(
        "[PASS] 当前 Human 原文不动，Tool Result 可压缩，Tool Pair 仍完整"
    )

    # ========================================================
    # Case 4：强保护内容本身就超预算
    # ========================================================
    print()
    print("=" * 80)
    print("Case 4：Base System + Current Human 本身超限 -> 明确失败")
    print("=" * 80)

    impossible = assembler.assemble(
        system_prompt=(
            "你是测试 Agent。"
        ),
        current_messages=[
            HumanMessage(
                content=(
                    "必须完整保留的当前用户输入："
                    + "超长用户正文" * 3000
                ),
                id="human-impossible-001",
            )
        ],
    )

    try:
        compressor.compress_to_fit(
            assembly=impossible,
            budget_config=ContextBudgetConfig(
                model_context_limit_tokens=900,
                reserved_output_tokens=200,
                tool_schema_tokens=100,
                safety_margin_tokens=100,
            ),
        )

    except ContextBudgetExceededError:
        print(
            "[PASS] 不会为了凑预算静默截断当前 Human"
        )

    else:
        raise AssertionError(
            "预期 ContextBudgetExceededError，"
            "但压缩器错误地声称可以装入。"
        )

    print()
    print("=" * 80)
    print(
        "Context Compression Phase 6C 回归测试通过"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()