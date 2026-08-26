"""ContextPlan -> Tool Permission Phase 7B 回归测试。

运行：
    python -m scripts.test_context_tool_policy

不调用真实 LLM。
"""

from __future__ import annotations

import tempfile

from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from raglab.agent.runtime_security import (
    SQLiteToolPolicyStore,
    SecureToolNode,
)


def make_tool(
    *,
    name: str,
    calls: list[str],
) -> StructuredTool:

    def implementation(
        query: str = "",
    ) -> str:
        calls.append(
            name
        )
        return (
            f"EXECUTED:{name}:{query}"
        )

    return StructuredTool.from_function(
        func=implementation,
        name=name,
        description=f"测试 Tool {name}",
    )


def tool_call(
    *,
    call_id: str,
    name: str,
    query: str = "demo",
) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": name,
        "args": {
            "query": query,
        },
        "type": "tool_call",
    }


def invoke(
    node: SecureToolNode,
    *,
    calls: list[dict[str, Any]],
    external_retrieval_allowed: bool,
) -> dict[str, Any]:

    return node.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=calls,
                )
            ],
            "context_pipeline_enabled": True,
            "context_plan": {
                "external_retrieval_allowed":
                    external_retrieval_allowed,
            },
        }
    )


def main() -> None:

    with tempfile.TemporaryDirectory() as temp_dir:

        policy_store = (
            SQLiteToolPolicyStore(
                Path(temp_dir)
                / "control.sqlite3"
            )
        )

        executed: list[str] = []

        retrieval_tool = make_tool(
            name="search_knowledge_base",
            calls=executed,
        )

        control_tool = make_tool(
            name="list_skills",
            calls=executed,
        )

        node = SecureToolNode(
            [
                retrieval_tool,
                control_tool,
            ],
            policy_store=(
                policy_store
            ),
        )

        # ====================================================
        # Case 1：Retrieval forbidden -> hard block
        # ====================================================

        print("=" * 80)
        print(
            "Case 1：external_retrieval_allowed=false "
            "-> Retrieval Tool 硬拦截"
        )
        print("=" * 80)

        output = invoke(
            node,
            calls=[
                tool_call(
                    call_id="call-r-001",
                    name=(
                        "search_knowledge_base"
                    ),
                )
            ],
            external_retrieval_allowed=False,
        )

        messages = list(
            output.get(
                "messages",
                [],
            )
        )

        assert (
            "search_knowledge_base"
            not in executed
        )

        assert len(
            messages
        ) == 1

        assert isinstance(
            messages[0],
            ToolMessage,
        )

        assert (
            messages[0].tool_call_id
            == "call-r-001"
        )

        assert (
            "external_retrieval_allowed=false"
            in str(
                messages[0].content
            )
        )

        print(
            "[PASS] LLM 即使主动请求 Retrieval Tool，执行层也不会运行"
        )

        # ====================================================
        # Case 2：Retrieval allowed -> actual execution
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 2：external_retrieval_allowed=true "
            "-> Retrieval Tool 正常执行"
        )
        print("=" * 80)

        output = invoke(
            node,
            calls=[
                tool_call(
                    call_id="call-r-002",
                    name=(
                        "search_knowledge_base"
                    ),
                    query="allowed",
                )
            ],
            external_retrieval_allowed=True,
        )

        assert (
            executed.count(
                "search_knowledge_base"
            )
            == 1
        )

        allowed_messages = list(
            output.get(
                "messages",
                [],
            )
        )

        assert (
            allowed_messages[
                0
            ].tool_call_id
            == "call-r-002"
        )

        assert (
            "EXECUTED:"
            in str(
                allowed_messages[
                    0
                ].content
            )
        )

        print(
            "[PASS] ContextPlan 允许时，不影响正常 Retrieval"
        )

        # ====================================================
        # Case 3：CONTROL 不属于 Retrieval
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 3：禁止 Retrieval 时 CONTROL Tool 仍可执行"
        )
        print("=" * 80)

        output = invoke(
            node,
            calls=[
                tool_call(
                    call_id="call-c-001",
                    name="list_skills",
                )
            ],
            external_retrieval_allowed=False,
        )

        assert (
            executed.count(
                "list_skills"
            )
            == 1
        )

        control_messages = list(
            output.get(
                "messages",
                [],
            )
        )

        assert (
            control_messages[
                0
            ].tool_call_id
            == "call-c-001"
        )

        print(
            "[PASS] Context Retrieval Permission 不会误伤 Skill 控制面"
        )

        # ====================================================
        # Case 4：Batch fail-closed 必须解析全部 Tool Call ID
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 4：一个 batch 同时含 Retrieval + CONTROL"
        )
        print("=" * 80)

        executed_before = list(
            executed
        )

        output = invoke(
            node,
            calls=[
                tool_call(
                    call_id="call-batch-r",
                    name=(
                        "search_knowledge_base"
                    ),
                ),
                tool_call(
                    call_id="call-batch-c",
                    name="list_skills",
                ),
            ],
            external_retrieval_allowed=False,
        )

        batch_messages = list(
            output.get(
                "messages",
                [],
            )
        )

        assert (
            executed
            == executed_before
        )

        assert len(
            batch_messages
        ) == 2

        assert {
            message.tool_call_id
            for message
            in batch_messages
        } == {
            "call-batch-r",
            "call-batch-c",
        }

        assert all(
            isinstance(
                message,
                ToolMessage,
            )
            for message
            in batch_messages
        )

        print(
            "[PASS] 一个 call 被阻止时整批不执行，"
            "并为所有 tool_call_id 返回 ToolMessage"
        )

        # ====================================================
        # Case 5：Policy Schema migration / classification
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 5：Tool Policy 独立 Context Access 分类"
        )
        print("=" * 80)

        retrieval_policy = (
            policy_store.get_policy(
                "search_knowledge_base"
            )
        )

        control_policy = (
            policy_store.get_policy(
                "list_skills"
            )
        )

        action_policy = (
            policy_store.get_policy(
                "update_github_intelligence"
            )
        )

        assert (
            retrieval_policy
            is not None
            and
            retrieval_policy
            .context_access_type
            == "RETRIEVAL"
        )

        assert (
            control_policy
            is not None
            and
            control_policy
            .context_access_type
            == "CONTROL"
        )

        assert (
            action_policy
            is not None
            and
            action_policy
            .context_access_type
            == "ACTION"
        )

        print(
            "[PASS] RETRIEVAL / CONTROL / ACTION "
            "与 external side-effect 语义已经分离"
        )

        print()
        print("=" * 80)
        print(
            "Context Tool Policy Phase 7B 回归测试通过"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()