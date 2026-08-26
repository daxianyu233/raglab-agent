"""Phase 7E-2 Token-aware Working Memory Compaction regression.

运行：
    python -m scripts.test_working_memory_compaction
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from raglab.agent.persistent_langgraph_agent import (
    PersistentLangGraphRetrievalAgent,
)
from raglab.agent.working_memory import (
    WorkingMemoryAuditConfig,
    audit_working_memory,
    plan_working_memory_compaction,
)


CONFIG = WorkingMemoryAuditConfig(
    soft_limit_tokens=12000,
    target_tokens=8000,
    oversized_tool_threshold_tokens=4000,
    minimum_recent_turns=1,
)


@dataclass
class FakeEvent:
    message_id: str


class FakeEventStore:
    def __init__(
        self,
        archived: dict[
            tuple[str, str],
            list[str],
        ],
    ) -> None:
        self.archived = archived

    def list_turn_events(
        self,
        *,
        thread_id: str,
        turn_id: str,
    ):
        return [
            FakeEvent(
                message_id=current
            )
            for current
            in self.archived.get(
                (
                    thread_id,
                    turn_id,
                ),
                [],
            )
        ]


class FakeSummaryModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        return AIMessage(
            content=(
                "【当前目标】继续完成 Context 管理。\n"
                "【已完成进度】巨大历史 Tool Turn 已归档并压缩。\n"
                "【待继续事项】继续后续 Agent 工程。"
            ),
            id=(
                "summary-ai"
            ),
        )


def tiny_turns(
    count: int,
):
    messages = []

    for index in range(
        count
    ):
        messages.extend(
            [
                HumanMessage(
                    content=(
                        f"短问题{index}"
                    ),
                    id=(
                        f"human-{index}"
                    ),
                ),
                AIMessage(
                    content=(
                        f"短回答{index}"
                    ),
                    id=(
                        f"ai-{index}"
                    ),
                ),
            ]
        )

    return messages


def huge_historical_turn_plus_current():
    return [
        HumanMessage(
            content="查询巨大历史资料",
            id="human-old",
        ),
        AIMessage(
            content="",
            id="ai-old-call",
            tool_calls=[
                {
                    "name":
                        "search_knowledge_base",
                    "args": {
                        "query":
                            "历史资料",
                    },
                    "id":
                        "call-old",
                    "type":
                        "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=(
                "X"
                * 60000
            ),
            tool_call_id=(
                "call-old"
            ),
            name=(
                "search_knowledge_base"
            ),
            id=(
                "tool-old"
            ),
        ),
        AIMessage(
            content="历史最终回答",
            id="ai-old-final",
        ),
        HumanMessage(
            content="继续下一步",
            id="human-current",
        ),
        AIMessage(
            content="当前回答",
            id="ai-current",
        ),
    ]


def pending_historical_plus_current():
    return [
        HumanMessage(
            content="旧查询",
            id="human-pending",
        ),
        AIMessage(
            content="",
            id="ai-pending",
            tool_calls=[
                {
                    "name":
                        "search_knowledge_base",
                    "args": {
                        "query":
                            "测试",
                    },
                    "id":
                        "call-pending",
                    "type":
                        "tool_call",
                }
            ],
        ),
        HumanMessage(
            content=(
                "当前轮"
                + "Y"
                * 60000
            ),
            id="human-current",
        ),
        AIMessage(
            content="当前回答",
            id="ai-current",
        ),
    ]


def archived_old_turn():
    return FakeEventStore(
        {
            (
                "thread-test",
                "turn:human-old",
            ): [
                "human-old",
                "ai-old-call",
                "tool-old",
                "ai-old-final",
            ],
        }
    )


def main() -> None:
    print("=" * 80)
    print(
        "Working Memory Phase 7E-2 回归测试"
    )
    print("=" * 80)

    # ========================================================
    # Case 1
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 1：10 个极短 Turn -> "
        "不再因为达到第 7 轮而压缩"
    )
    print("=" * 80)

    case1 = plan_working_memory_compaction(
        tiny_turns(10),
        config=CONFIG,
        event_store=(
            FakeEventStore({})
        ),
        thread_id="thread-test",
        legacy_keep_recent_turns=4,
        legacy_summarize_trigger_turns=7,
    )

    print(
        "legacy trigger：",
        case1.audit_before[
            "legacy_turn_trigger_would_fire"
        ],
    )
    print(
        "token should compact：",
        case1.should_compact,
    )
    print(
        "reason：",
        case1.reason,
    )

    assert (
        case1.audit_before[
            "legacy_turn_trigger_would_fire"
        ]
        is True
    )

    assert (
        case1.should_compact
        is False
    )

    assert (
        case1.reason
        == "below_soft_limit"
    )

    print(
        "[PASS] 固定 7 Turn 触发规则已不再控制真实行为"
    )

    # ========================================================
    # Case 2
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 2：巨大历史 Turn 已归档 -> "
        "按 Token 选择并压缩旧 Turn"
    )
    print("=" * 80)

    messages2 = (
        huge_historical_turn_plus_current()
    )

    case2 = plan_working_memory_compaction(
        messages2,
        summary="旧摘要",
        config=CONFIG,
        event_store=(
            archived_old_turn()
        ),
        thread_id="thread-test",
        legacy_keep_recent_turns=4,
        legacy_summarize_trigger_turns=7,
    )

    print(
        "before：",
        case2.before_tokens,
    )
    print(
        "selected：",
        list(
            case2.selected_turn_keys
        ),
    )
    print(
        "predicted after：",
        case2.predicted_after_tokens,
    )
    print(
        "target reached：",
        case2.predicted_target_reached,
    )

    assert case2.should_compact
    assert case2.can_compact
    assert (
        case2.selected_turn_keys
        == (
            "turn:human-old",
        )
    )
    assert (
        case2.archive_verified_turn_keys
        == (
            "turn:human-old",
        )
    )
    assert (
        case2.predicted_after_tokens
        < case2.before_tokens
    )

    print(
        "[PASS] oldest archived completed Turn 被选为压缩候选"
    )

    # ========================================================
    # Case 3
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 3：巨大历史 Turn 未完整归档 -> Fail Closed 不删除"
    )
    print("=" * 80)

    case3 = plan_working_memory_compaction(
        messages2,
        summary="旧摘要",
        config=CONFIG,
        event_store=(
            FakeEventStore(
                {
                    (
                        "thread-test",
                        "turn:human-old",
                    ): [
                        "human-old",
                        # 故意缺少 AI / Tool / final
                    ],
                }
            )
        ),
        thread_id="thread-test",
    )

    print(
        "can compact：",
        case3.can_compact,
    )
    print(
        "reason：",
        case3.reason,
    )
    print(
        "archive rejected：",
        list(
            case3.archive_rejected_turn_keys
        ),
    )

    assert case3.should_compact
    assert (
        case3.can_compact
        is False
    )
    assert (
        case3.reason
        == "historical_turns_not_fully_archived"
    )
    assert (
        case3.selected_turn_count
        == 0
    )

    print(
        "[PASS] Event Store 不完整时绝不删除 Checkpoint 历史"
    )

    # ========================================================
    # Case 4
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 4：未完成 Tool Pair 不进入安全候选"
    )
    print("=" * 80)

    messages4 = (
        pending_historical_plus_current()
    )

    case4 = plan_working_memory_compaction(
        messages4,
        config=CONFIG,
        event_store=(
            FakeEventStore({})
        ),
        thread_id="thread-test",
    )

    print(
        "selected：",
        list(
            case4.selected_turn_keys
        ),
    )
    print(
        "reason：",
        case4.reason,
    )

    assert case4.should_compact
    assert (
        case4.selected_turn_count
        == 0
    )

    print(
        "[PASS] 未完成 Tool Pair 不会为了省 Token 被误删"
    )

    # ========================================================
    # Case 5
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 5：memory_manager 先更新 Summary，"
        "再删除已归档历史 Turn"
    )
    print("=" * 80)

    agent = object.__new__(
        PersistentLangGraphRetrievalAgent
    )

    agent.keep_recent_turns = 4
    agent.summarize_trigger_turns = 7
    agent.working_memory_audit_config = (
        CONFIG
    )
    agent.conversation_event_store = (
        archived_old_turn()
    )
    agent.chat_model = (
        FakeSummaryModel()
    )

    result = (
        agent
        ._memory_manager_node(
            {
                "messages":
                    messages2,
                "summary":
                    "旧摘要",
                "working_memory_thread_id":
                    "thread-test",
                "turn_llm_calls":
                    0,
                "turn_summary_calls":
                    0,
                "total_summarized_turns":
                    0,
                "model_trace":
                    [],
            }
        )
    )

    print(
        "summary_updated：",
        result[
            "summary_updated"
        ],
    )
    print(
        "summarized turns：",
        result[
            "summarized_turns_this_run"
        ],
    )
    print(
        "token compaction applied：",
        result[
            "working_memory_audit"
        ][
            "token_compaction_applied"
        ],
    )
    print(
        "selected：",
        result[
            "working_memory_audit"
        ][
            "plan"
        ][
            "selected_turn_keys"
        ],
    )

    assert (
        result[
            "summary_updated"
        ]
        is True
    )
    assert (
        result[
            "summarized_turns_this_run"
        ]
        == 1
    )
    assert (
        result[
            "working_memory_audit"
        ][
            "token_compaction_applied"
        ]
        is True
    )
    assert (
        "已归档并压缩"
        in result[
            "summary"
        ]
    )
    assert (
        agent.chat_model.calls
        == 1
    )

    print(
        "[PASS] Summary 更新成功后才返回 RemoveMessage + retained messages"
    )

    # ========================================================
    # Case 6
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 6：当前 Turn 自己巨大但尚未成为历史 -> "
        "本轮延迟压缩"
    )
    print("=" * 80)

    current_only = [
        HumanMessage(
            content=(
                "Z"
                * 60000
            ),
            id="human-current-only",
        ),
        AIMessage(
            content="回答",
            id="ai-current-only",
        ),
    ]

    case6 = plan_working_memory_compaction(
        current_only,
        config=CONFIG,
        event_store=(
            FakeEventStore({})
        ),
        thread_id="thread-test",
    )

    print(
        "should compact：",
        case6.should_compact,
    )
    print(
        "can compact：",
        case6.can_compact,
    )
    print(
        "reason：",
        case6.reason,
    )

    assert case6.should_compact
    assert (
        case6.can_compact
        is False
    )
    assert (
        case6.reason
        == "only_protected_recent_turns_available"
    )

    print(
        "[PASS] 当前 Turn 未归档时不删除，下一轮成为历史后再整理"
    )

    print()
    print("=" * 80)
    print(
        "Working Memory Phase 7E-2 回归测试通过"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()