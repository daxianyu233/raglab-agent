"""Token Budget Manager Phase 6B 回归测试。

运行：
    python -m scripts.test_context_budget

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
    SOURCE_BASE_SYSTEM,
    SOURCE_CONVERSATION_HISTORY,
    SOURCE_CONVERSATION_HISTORY_HEADER,
    SOURCE_CURRENT_TURN,
    SOURCE_LONG_TERM_MEMORY,
    SOURCE_SKILL_RUNTIME,
    SOURCE_THREAD_SUMMARY,
)
from raglab.agent.context_budget import (
    ContextBudgetConfig,
    ContextBudgetManager,
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


def make_previous_plan() -> ContextPlan:
    return ContextPlan(
        task_intent="rewrite_previous",
        response_goal="改写上一轮回答",
        history_required=True,
        history_scope="previous_turn",
        history_query=None,
        previous_answer_required=True,
        raw_tool_evidence_required=False,
        external_retrieval_required=False,
        external_retrieval_allowed=False,
        long_term_memory_required=False,
        long_term_memory_query=None,
        referenced_entities=[],
        temporal_scope=None,
        confidence=1.0,
    )


def build_history(
    store: ConversationEventStore,
    *,
    thread_id: str,
) -> None:
    turn_id = "turn:human-001"

    store.append_event(
        user_id="local-user",
        thread_id=thread_id,
        turn_id=turn_id,
        event_type="message",
        role="human",
        message_id="human-001",
        content_text="解释一下 Context Planner。",
        payload={
            "type": "human",
            "content": "解释一下 Context Planner。",
            "id": "human-001",
        },
    )

    store.append_event(
        user_id="local-user",
        thread_id=thread_id,
        turn_id=turn_id,
        event_type="message",
        role="assistant",
        message_id="ai-001",
        content_text=(
            "Context Planner 负责判断本轮需要哪些上下文来源。"
        ),
        payload={
            "type": "ai",
            "content": (
                "Context Planner 负责判断本轮需要哪些上下文来源。"
            ),
            "id": "ai-001",
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    )


def main() -> None:
    manager = ContextBudgetManager()
    assembler = ContextAssembler()

    # ========================================================
    # Case 1：来源标记 + 正常预算
    # ========================================================
    print("=" * 80)
    print("Case 1：Context Source 成本拆分 + fits")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        store = ConversationEventStore(
            Path(temp_dir)
            / "events.sqlite3"
        )

        thread_id = "thread-budget-test"
        build_history(
            store,
            thread_id=thread_id,
        )

        retriever = ConversationRetriever(
            store=store
        )

        retrieval = retriever.retrieve(
            thread_id=thread_id,
            plan=make_previous_plan(),
        )

        assembly = assembler.assemble(
            system_prompt="你是测试 Agent。",
            skill_runtime_prompt=(
                "当前已加载 github-intelligence Skill。"
            ),
            long_term_memory_text=(
                "用户偏好架构优先的技术解释。"
            ),
            thread_summary=(
                "此前讨论过 Agent Context Management。"
            ),
            conversation_retrieval=retrieval,
            current_messages=[
                HumanMessage(
                    content="把上一轮再讲简洁一点。",
                    id="human-current-001",
                )
            ],
        )

        expected_sources = {
            SOURCE_BASE_SYSTEM,
            SOURCE_SKILL_RUNTIME,
            SOURCE_LONG_TERM_MEMORY,
            SOURCE_THREAD_SUMMARY,
            SOURCE_CONVERSATION_HISTORY_HEADER,
            SOURCE_CONVERSATION_HISTORY,
            SOURCE_CURRENT_TURN,
        }

        assert expected_sources.issubset(
            set(
                assembly.message_sources
            )
        )

        report = manager.evaluate(
            assembly=assembly,
            config=ContextBudgetConfig(
                model_context_limit_tokens=4096,
                reserved_output_tokens=512,
                tool_schema_tokens=256,
                safety_margin_tokens=128,
            ),
        )

        assert report.fits is True
        assert (
            report.compression_required
            is False
        )

        assert (
            sum(
                report
                .source_estimated_tokens
                .values()
            )
            == report.estimated_message_tokens
        )

        print(
            "消息预算：",
            report.available_message_tokens,
        )
        print(
            "消息估算：",
            report.estimated_message_tokens,
        )
        print(
            "剩余预算：",
            report.remaining_message_tokens,
        )
        print(
            "按来源 tokens：",
            report.source_estimated_tokens,
        )
        print(
            "[PASS] 每条消息来源可追踪，正常上下文可直接发送"
        )

        store.close()

    # ========================================================
    # Case 2：巨大 Tool Result 导致超预算
    # ========================================================
    print()
    print("=" * 80)
    print("Case 2：巨大 Tool Result -> overflow")
    print("=" * 80)

    large_tool_result = "A" * 12000

    large_assembly = assembler.assemble(
        system_prompt="你是测试 Agent。",
        current_messages=[
            HumanMessage(
                content="查询一个大型资料。",
                id="human-large-001",
            ),
            AIMessage(
                content="",
                id="ai-large-001",
                tool_calls=[
                    {
                        "id": "call-large-001",
                        "name": "search_large",
                        "args": {
                            "query": "large"
                        },
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=large_tool_result,
                id="tool-large-001",
                tool_call_id="call-large-001",
                name="search_large",
            ),
        ],
    )

    overflow_report = manager.evaluate(
        assembly=large_assembly,
        config=ContextBudgetConfig(
            model_context_limit_tokens=2500,
            reserved_output_tokens=500,
            tool_schema_tokens=200,
            safety_margin_tokens=100,
        ),
    )

    assert overflow_report.fits is False
    assert (
        overflow_report
        .compression_required
        is True
    )
    assert (
        overflow_report.overflow_tokens
        > 0
    )
    assert (
        overflow_report.largest_source
        == SOURCE_CURRENT_TURN
    )
    assert (
        overflow_report.recommended_action
        == "compression_or_pruning_required"
    )

    print(
        "可用消息预算：",
        overflow_report.available_message_tokens,
    )
    print(
        "消息估算：",
        overflow_report.estimated_message_tokens,
    )
    print(
        "超出 tokens：",
        overflow_report.overflow_tokens,
    )
    print(
        "最大来源：",
        overflow_report.largest_source,
        overflow_report.largest_source_tokens,
    )
    print(
        "[PASS] Tool Result 过大时只报告超限，不擅自裁剪"
    )

    # ========================================================
    # Case 3：Tool Schema 独立计费
    # ========================================================
    print()
    print("=" * 80)
    print("Case 3：Tool Schema 不在 messages 中，但必须占预算")
    print("=" * 80)

    small_assembly = assembler.assemble(
        system_prompt="你是测试 Agent。",
        current_messages=[
            HumanMessage(
                content=(
                    "这是一个很短的独立请求，"
                    "用于验证 Tool Schema 预算。"
                ),
                id="human-schema-001",
            )
        ],
    )

    no_schema_report = manager.evaluate(
        assembly=small_assembly,
        config=ContextBudgetConfig(
            model_context_limit_tokens=1000,
            reserved_output_tokens=200,
            tool_schema_tokens=0,
            safety_margin_tokens=100,
        ),
    )

    heavy_schema_report = manager.evaluate(
        assembly=small_assembly,
        config=ContextBudgetConfig(
            model_context_limit_tokens=1000,
            reserved_output_tokens=200,
            tool_schema_tokens=690,
            safety_margin_tokens=100,
        ),
    )

    assert no_schema_report.fits is True
    assert heavy_schema_report.fits is False

    print(
        "无 Tool Schema 时可用：",
        no_schema_report.available_message_tokens,
    )
    print(
        "Tool Schema=690 时可用：",
        heavy_schema_report.available_message_tokens,
    )
    print(
        "[PASS] Tool Schema 成本已从消息预算中单独扣除"
    )

    # ========================================================
    # Case 4：完整预算恒等式
    # ========================================================
    print()
    print("=" * 80)
    print("Case 4：预算恒等式")
    print("=" * 80)

    report = no_schema_report

    expected_total = (
        report.estimated_message_tokens
        + report.reserved_output_tokens
        + report.tool_schema_tokens
        + report.safety_margin_tokens
    )

    assert (
        report.estimated_total_reserved_tokens
        == expected_total
    )

    assert (
        report.available_message_tokens
        == (
            report.model_context_limit_tokens
            - report.reserved_output_tokens
            - report.tool_schema_tokens
            - report.safety_margin_tokens
        )
    )

    print(
        "[PASS] Context Window 的输入/输出/Schema/安全余量账目一致"
    )

    print()
    print("=" * 80)
    print(
        "Token Budget Manager Phase 6B 回归测试通过"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()