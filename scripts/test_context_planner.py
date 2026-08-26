"""真实 DeepSeek Context Planner 回归测试。

运行：
    python -m scripts.test_context_planner

本测试不会接入主 Agent，也不会改写 Checkpoint。
"""

from __future__ import annotations

import json

from raglab.agent.context_plan import (
    NavigationContext,
    TurnIndexItem,
)
from raglab.agent.context_planner import (
    ContextPlanner,
)
from raglab.settings import (
    CONFIG_DIR,
)
from scripts.ask_rag import (
    create_deepseek_model,
    load_yaml_config,
    require_mapping,
)


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "agent.yaml"
)


def build_planner() -> ContextPlanner:
    config = load_yaml_config(
        DEFAULT_CONFIG_PATH
    )

    model_config = require_mapping(
        config,
        "model",
    )

    chat_model = create_deepseek_model(
        model_config
    )

    return ContextPlanner(
        chat_model=chat_model
    )


def print_result(
    title: str,
    result,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        "Navigation JSON 字符数："
        f"{result.navigation_characters}"
    )

    print(
        "Planner 延迟："
        f"{result.latency_ms:.2f} ms"
    )

    print(
        "usage_metadata："
        f"{result.usage_metadata}"
    )

    print()

    print(
        json.dumps(
            result.plan.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    planner = build_planner()

    previous_turn = TurnIndexItem(
        turn_id="turn-017",
        user_goal=(
            "查询今天 GitHub 技术热点，"
            "重点查看 Agent 项目"
        ),
        assistant_outcome=(
            "根据 GitHub RAG 返回结果，"
            "整理了多个 Agent 项目"
        ),
        entities=[
            "Agent-Reach",
            "ai-memory",
            "Needle",
        ],
        has_tool_evidence=True,
        tool_names=[
            "search_github_intelligence"
        ],
    )

    old_turn = TurnIndexItem(
        turn_id="turn-006",
        user_goal=(
            "分析 ai-memory 的长期记忆实现"
        ),
        assistant_outcome=(
            "讨论了 markdown git repository、"
            "MCP 和跨代理交接"
        ),
        entities=[
            "ai-memory",
            "MCP",
        ],
        has_tool_evidence=True,
        tool_names=[
            "search_github_intelligence"
        ],
    )

    result_1 = planner.plan(
        NavigationContext(
            current_user_input=(
                "你刚才那段总结写得太口语了，"
                "改得正式一点。"
            ),
            thread_summary=(
                "当前线程主要讨论 GitHub Agent 技术情报。"
            ),
            recent_turns=[
                previous_turn
            ],
            history_candidates=[
                old_turn
            ],
        )
    )

    print_result(
        "Case 1：只改写上一轮回答",
        result_1,
    )

    assert (
        result_1.plan.history_required
        is True
    )

    assert (
        result_1.plan.history_scope
        == "previous_turn"
    )

    assert (
        result_1.plan.previous_answer_required
        is True
    )

    assert (
        result_1.plan.raw_tool_evidence_required
        is False
    )

    assert (
        result_1.plan.external_retrieval_required
        is False
    )

    result_2 = planner.plan(
        NavigationContext(
            current_user_input=(
                "刚才整理得不好。不要重新搜索，"
                "直接根据刚才 RAG 查到的原始资料"
                "重新整理那些项目。"
            ),
            thread_summary=(
                "当前线程主要讨论 GitHub Agent 技术情报。"
            ),
            recent_turns=[
                previous_turn
            ],
            history_candidates=[
                old_turn
            ],
        )
    )

    print_result(
        "Case 2：重用上一轮原始 Tool Evidence",
        result_2,
    )

    assert (
        result_2.plan.history_scope
        == "previous_turn"
    )

    assert (
        result_2.plan.raw_tool_evidence_required
        is True
    )

    assert (
        result_2.plan.external_retrieval_required
        is False
    )

    assert (
        result_2.plan.external_retrieval_allowed
        is False
    )

    result_3 = planner.plan(
        NavigationContext(
            current_user_input=(
                "十几轮前我们聊过 ai-memory "
                "不用向量数据库这件事，"
                "把当时的原始资料重新找出来再解释一下。"
            ),
            thread_summary=(
                "线程中讨论过多个 Agent 工程项目。"
            ),
            recent_turns=[
                previous_turn
            ],
            history_candidates=[
                old_turn
            ],
        )
    )

    print_result(
        "Case 3：搜索更早历史",
        result_3,
    )

    assert (
        result_3.plan.history_required
        is True
    )

    assert (
        result_3.plan.history_scope
        == "historical_search"
    )

    assert bool(
        result_3.plan.history_query
    )

    assert (
        result_3.plan.raw_tool_evidence_required
        is True
    )

    result_4 = planner.plan(
        NavigationContext(
            current_user_input=(
                "给我今天的 GitHub 技术日报。"
            ),
            thread_summary=(
                "此前讨论过 ai-memory 和 Agent-Reach。"
            ),
            recent_turns=[
                previous_turn
            ],
            history_candidates=[
                old_turn
            ],
        )
    )

    print_result(
        "Case 4：与旧历史无关的新查询",
        result_4,
    )

    assert (
        result_4.plan.history_required
        is False
    )

    assert (
        result_4.plan.history_scope
        == "none"
    )

    assert (
        result_4.plan.external_retrieval_required
        is True
    )

    print()
    print("=" * 80)
    print(
        "Context Planner Phase 2 "
        "真实 LLM 回归测试通过"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()