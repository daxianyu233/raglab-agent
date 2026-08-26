"""Tool Schema Exposure Phase 7C 回归测试。

运行：
    python -m scripts.test_context_tool_exposure

不调用真实 LLM。
"""

from __future__ import annotations

import tempfile

from pathlib import Path
from typing import Any

from langchain_core.tools import (
    StructuredTool,
)

from raglab.agent.context_runtime import (
    estimate_active_tool_schema_tokens,
)
from raglab.agent.long_term_memory_agent import (
    LongTermMemoryRetrievalAgent,
)
from raglab.agent.runtime_security import (
    SQLiteToolPolicyStore,
)
from raglab.agent.tool_exposure import (
    select_tools_for_context,
)


def make_tool(
    name: str,
    description_size: int = 20,
) -> StructuredTool:

    def implementation(
        query: str = "",
    ) -> str:
        return (
            f"{name}:{query}"
        )

    return StructuredTool.from_function(
        func=implementation,
        name=name,
        description=(
            f"{name} "
            + "schema-description-"
            * description_size
        ),
    )


def main() -> None:

    with tempfile.TemporaryDirectory() as temp_dir:

        policy_store = (
            SQLiteToolPolicyStore(
                Path(temp_dir)
                / "control.sqlite3"
            )
        )

        retrieval_1 = make_tool(
            "search_knowledge_base",
            description_size=30,
        )

        retrieval_2 = make_tool(
            "search_github_intelligence",
            description_size=30,
        )

        control_1 = make_tool(
            "list_skills",
            description_size=10,
        )

        control_2 = make_tool(
            "load_skill",
            description_size=10,
        )

        active_tools = [
            retrieval_1,
            retrieval_2,
            control_1,
            control_2,
        ]

        # ====================================================
        # Case 1：禁止 Retrieval -> schema 隐藏
        # ====================================================

        print("=" * 80)
        print(
            "Case 1：external_retrieval_allowed=false "
            "-> Retrieval Schemas 不暴露给 LLM"
        )
        print("=" * 80)

        decision = (
            select_tools_for_context(
                active_tools=(
                    active_tools
                ),
                context_pipeline_enabled=True,
                context_plan={
                    "external_retrieval_allowed":
                        False,
                },
                policy_resolver=(
                    policy_store
                    .get_policy
                ),
            )
        )

        assert set(
            decision
            .hidden_tool_names
        ) == {
            "search_knowledge_base",
            "search_github_intelligence",
        }

        assert set(
            decision
            .exposed_tool_names
        ) == {
            "list_skills",
            "load_skill",
        }

        print(
            "Active：",
            decision.active_tool_names,
        )

        print(
            "Exposed：",
            decision.exposed_tool_names,
        )

        print(
            "Hidden：",
            decision.hidden_tool_names,
        )

        print(
            "[PASS] Retrieval Tool 仍 Active，"
            "但 schema 不再展示给 LLM"
        )

        # ====================================================
        # Case 2：Schema Token 真实下降
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 2：Tool Schema Budget 只计算 Exposed Tools"
        )
        print("=" * 80)

        active_schema_tokens = (
            estimate_active_tool_schema_tokens(
                active_tools
            )
        )

        exposed_schema_tokens = (
            estimate_active_tool_schema_tokens(
                decision
                .exposed_tools
            )
        )

        assert (
            exposed_schema_tokens
            < active_schema_tokens
        )

        saved = (
            active_schema_tokens
            - exposed_schema_tokens
        )

        print(
            "Active schema tokens：",
            active_schema_tokens,
        )

        print(
            "Exposed schema tokens：",
            exposed_schema_tokens,
        )

        print(
            "节省 schema tokens：",
            saved,
        )

        print(
            "[PASS] 禁止 Retrieval 时，"
            "Context Budget 不再为隐藏 schema 付费"
        )

        # ====================================================
        # Case 3：允许 Retrieval -> 完整暴露
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 3：external_retrieval_allowed=true "
            "-> 所有 Active Tools 正常暴露"
        )
        print("=" * 80)

        allowed = (
            select_tools_for_context(
                active_tools=(
                    active_tools
                ),
                context_pipeline_enabled=True,
                context_plan={
                    "external_retrieval_allowed":
                        True,
                },
                policy_resolver=(
                    policy_store
                    .get_policy
                ),
            )
        )

        assert (
            allowed
            .exposed_tool_names
            ==
            allowed
            .active_tool_names
        )

        assert (
            allowed
            .hidden_tool_names
            == []
        )

        print(
            "[PASS] ContextPlan 允许 Retrieval 时不损失能力"
        )

        # ====================================================
        # Case 4：真正重新 bind 模型侧 tools
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 4：LongTermMemoryAgent 模型侧只 bind Exposed Tools"
        )
        print("=" * 80)

        # 不需要构造完整 Agent/Graph。
        # 这里只直接验证 Phase 7C 新增的绑定边界。
        harness = object.__new__(
            LongTermMemoryRetrievalAgent
        )

        harness.context_pipeline_enabled = True
        harness.context_tool_policy_resolver = (
            policy_store.get_policy
        )

        bound_names: list[
            list[str]
        ] = []

        class BoundModel:
            pass

        harness.chat_model = (
            BoundModel()
        )

        def fake_bind_tools(
            tools: list[Any],
        ) -> Any:
            bound_names.append(
                [
                    str(tool.name)
                    for tool in tools
                ]
            )
            return BoundModel()

        harness._bind_tools = (
            fake_bind_tools
        )

        state = {
            "context_pipeline_enabled": True,
            "context_plan": {
                "external_retrieval_allowed":
                    False,
            },
        }

        exposure = (
            harness
            ._apply_context_tool_exposure(
                state=state,
                active_tools=(
                    active_tools
                ),
            )
        )

        assert bound_names[-1] == [
            "list_skills",
            "load_skill",
        ]

        assert (
            exposure
            .hidden_tool_names
            == [
                "search_knowledge_base",
                "search_github_intelligence",
            ]
        )

        print(
            "bind_tools 收到：",
            bound_names[-1],
        )

        print(
            "[PASS] LLM 实际 Tool Binding 已从 Active 集合缩成 Exposed 子集"
        )

        # ====================================================
        # Case 5：Exposure 层故障不替代 Security Backstop
        # ====================================================

        print()
        print("=" * 80)
        print(
            "Case 5：Policy Resolver 缺失 -> Exposure fail-open，"
            "执行层继续兜底"
        )
        print("=" * 80)

        fallback = (
            select_tools_for_context(
                active_tools=(
                    active_tools
                ),
                context_pipeline_enabled=True,
                context_plan={
                    "external_retrieval_allowed":
                        False,
                },
                policy_resolver=None,
            )
        )

        assert (
            fallback
            .exposed_tool_names
            ==
            fallback
            .active_tool_names
        )

        assert (
            fallback.reason
            ==
            "policy_resolver_unavailable_"
            "runtime_backstop_required"
        )

        print(
            "[PASS] Schema Exposure 是优化层，"
            "SecureToolNode 才是最终硬安全边界"
        )

        print()
        print("=" * 80)
        print(
            "Tool Schema Exposure Phase 7C 回归测试通过"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()