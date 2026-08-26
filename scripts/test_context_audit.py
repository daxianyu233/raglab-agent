from __future__ import annotations

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from raglab.agent.context_manager import (
    audit_model_input,
)


def main() -> None:
    paired_messages = [
        SystemMessage(
            content="你是测试 Agent。"
        ),
        HumanMessage(
            content="请查询测试资料。"
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_test",
                    "args": {
                        "query": "context audit"
                    },
                    "id": "call-001",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="A" * 12000,
            tool_call_id="call-001",
            name="search_test",
        ),
    ]

    report = audit_model_input(
        paired_messages,
        oversized_tool_threshold_tokens=1000,
    )

    print("=" * 80)
    print("1. 完整 Tool Pair")
    print("=" * 80)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    assert (
        report[
            "tool_pair_integrity_ok"
        ]
        is True
    )

    assert (
        report[
            "unresolved_tool_call_count"
        ]
        == 0
    )

    assert (
        report[
            "orphan_tool_message_count"
        ]
        == 0
    )

    assert (
        report[
            "oversized_tool_message_count"
        ]
        >= 1
    )

    broken_messages = (
        paired_messages[:-1]
    )

    broken_report = (
        audit_model_input(
            broken_messages,
            oversized_tool_threshold_tokens=1000,
        )
    )

    print()
    print("=" * 80)
    print("2. 缺失 ToolMessage")
    print("=" * 80)
    print(
        json.dumps(
            broken_report,
            ensure_ascii=False,
            indent=2,
        )
    )

    assert (
        broken_report[
            "tool_pair_integrity_ok"
        ]
        is False
    )

    assert (
        broken_report[
            "unresolved_tool_call_count"
        ]
        == 1
    )

    print()
    print("=" * 80)
    print("Context Audit Phase 1 回归测试通过")
    print("=" * 80)


if __name__ == "__main__":
    main()