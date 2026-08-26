"""RAGLab Context Planner 的结构化数据模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


HistoryScope = Literal[
    "none",
    "previous_turn",
    "recent_turns",
    "historical_search",
]


class TurnIndexItem(BaseModel):
    """一轮历史的轻量目录项，而不是完整历史正文。"""

    turn_id: str = Field(
        description="该轮在 Conversation Event Store 中的稳定标识。"
    )

    user_goal: str = Field(
        description="该轮用户目标的短摘要。"
    )

    assistant_outcome: str = Field(
        default="",
        description="该轮最终结果的短摘要。"
    )

    entities: list[str] = Field(
        default_factory=list,
        description="该轮涉及的关键实体。"
    )

    has_tool_evidence: bool = Field(
        default=False,
        description="该轮是否存在可恢复的原始 Tool Result。"
    )

    tool_names: list[str] = Field(
        default_factory=list,
        description="该轮实际使用过的 Tool 名称。"
    )


class NavigationContext(BaseModel):
    """轻量 Planner 的输入。"""

    current_user_input: str = Field(
        description="当前这一轮用户的原始输入。"
    )

    thread_summary: str = Field(
        default="",
        description="当前线程已有滚动摘要的精简版本。"
    )

    recent_turns: list[TurnIndexItem] = Field(
        default_factory=list,
        description="最近少量轮次的轻量目录项。"
    )

    history_candidates: list[TurnIndexItem] = Field(
        default_factory=list,
        description="从更早历史索引中粗召回出的少量候选目录项。"
    )

    capability_catalog: list[str] = Field(
        default_factory=list,
        description="系统可获取的信息类别摘要，不放完整 Tool Schema。"
    )

    runtime_notes: list[str] = Field(
        default_factory=list,
        description="少量运行时事实。"
    )


class ContextPlan(BaseModel):
    """Planner 输出：主 Agent 本轮所需的上下文需求计划。"""

    task_intent: str = Field(
        description=(
            "开放式、简短的任务意图标签，例如 rewrite_previous_answer、"
            "compare_previous_evidence、query_daily_report。"
        )
    )

    response_goal: str = Field(
        description="一句话说明主 Agent 最终需要完成什么。"
    )

    history_required: bool = Field(
        description="本轮是否需要当前线程中的历史信息。"
    )

    history_scope: HistoryScope = Field(
        description=(
            "none=不需要历史；previous_turn=上一轮；"
            "recent_turns=最近若干轮；historical_search=搜索更早历史。"
        )
    )

    history_query: str | None = Field(
        default=None,
        description="需要检索更早历史时使用的短查询。"
    )

    previous_answer_required: bool = Field(
        description="是否需要历史中的 Assistant 最终回答。"
    )

    raw_tool_evidence_required: bool = Field(
        description="是否需要恢复历史中的原始 Tool Result / Evidence。"
    )

    external_retrieval_required: bool = Field(
        description="是否需要新的 RAG / SQL / 文件等外部检索。"
    )

    external_retrieval_allowed: bool = Field(
        description="用户是否允许新的外部检索。"
    )

    long_term_memory_required: bool = Field(
        description="是否需要跨线程长期记忆。"
    )

    long_term_memory_query: str | None = Field(
        default=None,
        description="需要长期记忆时使用的短检索查询。"
    )

    referenced_entities: list[str] = Field(
        default_factory=list,
        description="本轮引用的关键实体。"
    )

    temporal_scope: str | None = Field(
        default=None,
        description="用户明确要求的时间范围。"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Planner 对这份 ContextPlan 的置信度。"
    )


def validate_context_plan(
    plan: ContextPlan,
) -> ContextPlan:
    """只做结构一致性校验，不做意图猜测。"""

    if not plan.task_intent.strip():
        raise ValueError("task_intent 不能为空。")

    if not plan.response_goal.strip():
        raise ValueError("response_goal 不能为空。")

    if not plan.history_required:
        if plan.history_scope != "none":
            raise ValueError(
                "history_required=false 时 history_scope 必须为 none。"
            )

        if plan.previous_answer_required:
            raise ValueError(
                "不需要历史时 previous_answer_required 不能为 true。"
            )

        if plan.raw_tool_evidence_required:
            raise ValueError(
                "不需要历史时 raw_tool_evidence_required 不能为 true。"
            )

        if plan.history_query:
            raise ValueError(
                "不需要历史时 history_query 必须为空。"
            )

    if plan.history_required and plan.history_scope == "none":
        raise ValueError(
            "history_required=true 时 history_scope 不能为 none。"
        )

    if plan.history_scope == "historical_search":
        if not str(plan.history_query or "").strip():
            raise ValueError(
                "historical_search 必须提供 history_query。"
            )

    if (
        plan.external_retrieval_required
        and not plan.external_retrieval_allowed
    ):
        raise ValueError(
            "external_retrieval_required=true 与 "
            "external_retrieval_allowed=false 冲突。"
        )

    if plan.long_term_memory_required:
        if not str(plan.long_term_memory_query or "").strip():
            raise ValueError(
                "需要长期记忆时必须提供 long_term_memory_query。"
            )

    if (
        not plan.long_term_memory_required
        and plan.long_term_memory_query
    ):
        raise ValueError(
            "不需要长期记忆时 long_term_memory_query 应为空。"
        )

    plan.task_intent = plan.task_intent.strip()
    plan.response_goal = plan.response_goal.strip()

    if plan.history_query:
        plan.history_query = plan.history_query.strip()

    if plan.long_term_memory_query:
        plan.long_term_memory_query = (
            plan.long_term_memory_query.strip()
        )

    plan.referenced_entities = [
        str(item).strip()
        for item in plan.referenced_entities
        if str(item).strip()
    ]

    return plan


def compact_navigation_context(
    context: NavigationContext,
    *,
    maximum_summary_characters: int = 1200,
    maximum_goal_characters: int = 220,
    maximum_outcome_characters: int = 260,
    maximum_recent_turns: int = 3,
    maximum_history_candidates: int = 6,
    maximum_capabilities: int = 8,
    maximum_runtime_notes: int = 6,
) -> NavigationContext:
    """生成 Planner 使用的轻量 Navigation Context。"""

    def truncate(
        value: Any,
        maximum: int,
    ) -> str:
        text = str(value or "").strip()

        if len(text) <= maximum:
            return text

        return text[:maximum] + "……"

    def compact_turn(
        item: TurnIndexItem,
    ) -> TurnIndexItem:
        return TurnIndexItem(
            turn_id=item.turn_id,
            user_goal=truncate(
                item.user_goal,
                maximum_goal_characters,
            ),
            assistant_outcome=truncate(
                item.assistant_outcome,
                maximum_outcome_characters,
            ),
            entities=[
                str(entity).strip()
                for entity in item.entities[:8]
                if str(entity).strip()
            ],
            has_tool_evidence=item.has_tool_evidence,
            tool_names=[
                str(name).strip()
                for name in item.tool_names[:6]
                if str(name).strip()
            ],
        )

    return NavigationContext(
        current_user_input=context.current_user_input.strip(),
        thread_summary=truncate(
            context.thread_summary,
            maximum_summary_characters,
        ),
        recent_turns=[
            compact_turn(item)
            for item in context.recent_turns[
                -maximum_recent_turns:
            ]
        ],
        history_candidates=[
            compact_turn(item)
            for item in context.history_candidates[
                :maximum_history_candidates
            ]
        ],
        capability_catalog=[
            str(item).strip()
            for item in context.capability_catalog[
                :maximum_capabilities
            ]
            if str(item).strip()
        ],
        runtime_notes=[
            str(item).strip()
            for item in context.runtime_notes[
                :maximum_runtime_notes
            ]
            if str(item).strip()
        ],
    )