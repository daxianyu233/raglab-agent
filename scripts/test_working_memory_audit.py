"""Phase 7E-1 Working Memory Audit regression.

运行：
    python -m scripts.test_working_memory_audit
"""

from __future__ import annotations

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
)


CONFIG = WorkingMemoryAuditConfig(
    soft_limit_tokens=12000,
    target_tokens=8000,
    oversized_tool_threshold_tokens=4000,
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


def huge_tool_two_turns():
    return [
        HumanMessage(
            content="第一轮",
            id="human-1",
        ),
        AIMessage(
            content="第一轮回答",
            id="ai-1",
        ),
        HumanMessage(
            content="第二轮查询",
            id="human-2",
        ),
        AIMessage(
            content="",
            id="ai-2-call",
            tool_calls=[
                {
                    "name":
                        "search_knowledge_base",
                    "args": {
                        "query":
                            "测试",
                    },
                    "id":
                        "call-huge",
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
                "call-huge"
            ),
            name=(
                "search_knowledge_base"
            ),
            id=(
                "tool-huge"
            ),
        ),
        AIMessage(
            content="第二轮最终回答",
            id="ai-2-final",
        ),
    ]


def pending_tool_turn():
    return [
        HumanMessage(
            content="执行查询",
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
    ]


def main() -> None:
    print("=" * 80)
    print(
        "Working Memory Phase 7E-1 回归测试"
    )
    print("=" * 80)

    print()
    print("=" * 80)
    print(
        "Case 1：10 个极短 Turn -> "
        "旧规则会压，新 Token 规则不建议压"
    )
    print("=" * 80)

    case1 = audit_working_memory(
        tiny_turns(10),
        summary="",
        config=CONFIG,
        legacy_keep_recent_turns=4,
        legacy_summarize_trigger_turns=7,
    )

    print(
        "estimated tokens：",
        case1[
            "checkpoint_message_estimated_tokens"
        ],
    )
    print(
        "legacy trigger：",
        case1[
            "legacy_turn_trigger_would_fire"
        ],
    )
    print(
        "token trigger：",
        case1[
            "token_compaction_recommended"
        ],
    )

    assert (
        case1[
            "legacy_turn_trigger_would_fire"
        ]
        is True
    )
    assert (
        case1[
            "token_compaction_recommended"
        ]
        is False
    )

    print(
        "[PASS] 很多短 Turn 不应只因为轮数多就强制压缩"
    )

    print()
    print("=" * 80)
    print(
        "Case 2：只有 2 个 Turn，但 Tool Result 巨大 -> "
        "旧规则不压，新 Token 规则建议压"
    )
    print("=" * 80)

    messages2 = (
        huge_tool_two_turns()
    )

    case2 = audit_working_memory(
        messages2,
        summary="",
        config=CONFIG,
        legacy_keep_recent_turns=4,
        legacy_summarize_trigger_turns=7,
    )

    print(
        "estimated tokens：",
        case2[
            "checkpoint_message_estimated_tokens"
        ],
    )
    print(
        "oversized tools：",
        case2[
            "oversized_tool_message_count"
        ],
    )
    print(
        "legacy trigger：",
        case2[
            "legacy_turn_trigger_would_fire"
        ],
    )
    print(
        "token trigger：",
        case2[
            "token_compaction_recommended"
        ],
    )
    print(
        "remove to target：",
        case2[
            "estimated_tokens_to_remove_to_target"
        ],
    )

    assert (
        case2[
            "legacy_turn_trigger_would_fire"
        ]
        is False
    )
    assert (
        case2[
            "token_compaction_recommended"
        ]
        is True
    )
    assert (
        case2[
            "oversized_tool_message_count"
        ]
        == 1
    )

    print(
        "[PASS] 少轮次也可能因为巨大 Tool Result 需要压缩"
    )

    print()
    print("=" * 80)
    print(
        "Case 3：未完成 Tool Pair -> "
        "必须识别为不可安全清理"
    )
    print("=" * 80)

    case3 = audit_working_memory(
        pending_tool_turn(),
        summary="",
        config=CONFIG,
        legacy_keep_recent_turns=4,
        legacy_summarize_trigger_turns=7,
    )

    print(
        "unresolved：",
        case3[
            "unresolved_tool_call_ids"
        ],
    )

    assert (
        case3[
            "tool_pair_integrity_ok"
        ]
        is False
    )
    assert (
        case3[
            "unresolved_tool_call_ids"
        ]
        == [
            "call-pending"
        ]
    )

    print(
        "[PASS] Pending Tool Call 能被识别，7E-2 不能误删"
    )

    print()
    print("=" * 80)
    print(
        "Case 4：已完成历史 Turn 与当前 Turn 能分开"
    )
    print("=" * 80)

    case4 = audit_working_memory(
        messages2,
        summary="已有线程摘要",
        config=CONFIG,
        legacy_keep_recent_turns=4,
        legacy_summarize_trigger_turns=7,
    )

    print(
        "turn count：",
        case4[
            "turn_count"
        ],
    )
    print(
        "completed historical turns：",
        case4[
            "completed_historical_turn_keys"
        ],
    )

    assert (
        case4[
            "turn_count"
        ]
        == 2
    )
    assert (
        case4[
            "completed_historical_turn_count"
        ]
        == 1
    )
    assert (
        case4[
            "turns"
        ][-1][
            "is_current_turn"
        ]
        is True
    )

    print(
        "[PASS] 7E-2 可只从已完成历史 Turn 中选择清理候选"
    )

    print()
    print("=" * 80)
    print(
        "Case 5：正式 memory_manager 已接入 Audit，"
        "但 7E-1 不改变旧行为"
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

    result = (
        agent
        ._memory_manager_node(
            {
                "messages":
                    messages2,
                "summary":
                    "",
            }
        )
    )

    audit = (
        result[
            "working_memory_audit"
        ][
            "before"
        ]
    )

    print(
        "summary_updated：",
        result[
            "summary_updated"
        ],
    )
    print(
        "legacy applied：",
        result[
            "working_memory_audit"
        ][
            "legacy_compaction_applied"
        ],
    )
    print(
        "token recommendation：",
        audit[
            "token_compaction_recommended"
        ],
    )

    assert (
        result[
            "summary_updated"
        ]
        is False
    )
    assert (
        result[
            "working_memory_audit"
        ][
            "legacy_compaction_applied"
        ]
        is False
    )
    assert (
        audit[
            "token_compaction_recommended"
        ]
        is True
    )

    print(
        "[PASS] 7E-1 只观测，不提前改变 Rolling Summary / Checkpoint"
    )

    print()
    print("=" * 80)
    print(
        "Working Memory Phase 7E-1 回归测试通过"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()