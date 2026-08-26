"""RAGLab Context Pipeline Runtime - Phase 7A.

新 Human Turn:
    Context Planner（仅一次）
        -> Conversation Retriever（仅一次）
        -> Plan / Retrieval 写入 Graph State

每次 Main LLM 调用:
    读取同一份 Plan / Retrieval
        -> 提取当前 turn 动态 messages
        -> Context Assembler
        -> Token Budget
        -> Compression（必要时）
        -> 最终 model_input
"""

from __future__ import annotations

import json

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.tools import BaseTool

from raglab.agent.context_assembler import (
    ContextAssembler,
)
from raglab.agent.context_budget import (
    ContextBudgetConfig,
    ContextBudgetManager,
)
from raglab.agent.context_compression import (
    ContextCompressor,
)
from raglab.agent.context_manager import (
    estimate_text_tokens,
)
from raglab.agent.context_plan import (
    ContextPlan,
    NavigationContext,
    TurnIndexItem,
)
from raglab.agent.context_planner import (
    ContextPlanner,
    ContextPlannerResult,
)
from raglab.agent.conversation_event_store import (
    ConversationEvent,
    ConversationEventStore,
)
from raglab.agent.conversation_retriever import (
    ConversationRetrievalResult,
    ConversationRetriever,
    RetrievedConversationTurn,
)


@dataclass(frozen=True)
class PreparedTurnContext:
    context_plan: dict[str, Any]
    context_retrieval: dict[str, Any]
    context_planner_trace: dict[str, Any]


@dataclass(frozen=True)
class BuiltModelContext:
    messages: list[BaseMessage]
    diagnostics: dict[str, Any]


def _event_to_dict(
    event: ConversationEvent,
) -> dict[str, Any]:
    return asdict(event)


def _event_from_dict(
    payload: dict[str, Any],
) -> ConversationEvent:
    return ConversationEvent(
        event_id=str(payload.get("event_id", "")),
        user_id=str(payload.get("user_id", "")),
        thread_id=str(payload.get("thread_id", "")),
        turn_id=str(payload.get("turn_id", "")),
        sequence_no=int(payload.get("sequence_no", 0)),
        event_type=str(payload.get("event_type", "message")),
        role=str(payload.get("role", "")),
        message_id=(
            str(payload["message_id"])
            if payload.get("message_id") is not None
            else None
        ),
        tool_call_id=(
            str(payload["tool_call_id"])
            if payload.get("tool_call_id") is not None
            else None
        ),
        tool_name=(
            str(payload["tool_name"])
            if payload.get("tool_name") is not None
            else None
        ),
        content_text=str(payload.get("content_text", "")),
        payload=dict(payload.get("payload", {}) or {}),
        metadata=dict(payload.get("metadata", {}) or {}),
        created_at=str(payload.get("created_at", "")),
    )


def serialize_retrieval_result(
    result: ConversationRetrievalResult,
) -> dict[str, Any]:
    return {
        "thread_id": result.thread_id,
        "history_scope": result.history_scope,
        "history_query": result.history_query,
        "selected_turn_ids": list(result.selected_turn_ids),
        "selected_event_count": int(result.selected_event_count),
        "candidate_turn_count": int(result.candidate_turn_count),
        "retrieval_strategy": result.retrieval_strategy,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "events": [
                    _event_to_dict(event)
                    for event in turn.events
                ],
                "selected_events": [
                    _event_to_dict(event)
                    for event in turn.selected_events
                ],
                "retrieval_score": float(turn.retrieval_score),
                "matched_terms": list(turn.matched_terms),
            }
            for turn in result.turns
        ],
    }


def deserialize_retrieval_result(
    payload: dict[str, Any],
) -> ConversationRetrievalResult:
    turns: list[RetrievedConversationTurn] = []

    for raw_turn in payload.get("turns", []) or []:
        if not isinstance(raw_turn, dict):
            continue

        turns.append(
            RetrievedConversationTurn(
                turn_id=str(raw_turn.get("turn_id", "")),
                events=[
                    _event_from_dict(item)
                    for item in raw_turn.get("events", []) or []
                    if isinstance(item, dict)
                ],
                selected_events=[
                    _event_from_dict(item)
                    for item in raw_turn.get("selected_events", []) or []
                    if isinstance(item, dict)
                ],
                retrieval_score=float(
                    raw_turn.get("retrieval_score", 0.0)
                ),
                matched_terms=[
                    str(item)
                    for item in raw_turn.get("matched_terms", []) or []
                ],
            )
        )

    return ConversationRetrievalResult(
        thread_id=str(payload.get("thread_id", "")),
        history_scope=str(payload.get("history_scope", "none")),
        history_query=(
            str(payload["history_query"])
            if payload.get("history_query") is not None
            else None
        ),
        selected_turn_ids=[
            str(item)
            for item in payload.get("selected_turn_ids", []) or []
        ],
        turns=turns,
        selected_event_count=int(
            payload.get("selected_event_count", 0)
        ),
        candidate_turn_count=int(
            payload.get("candidate_turn_count", 0)
        ),
        retrieval_strategy=str(
            payload.get("retrieval_strategy", "none")
        ),
    )


def _model_dump(
    model: Any,
) -> dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())

    dict_method = getattr(model, "dict", None)
    if callable(dict_method):
        return dict(dict_method())

    raise TypeError("对象不支持 Pydantic dump。")


def _tool_schema_payload(
    tool: BaseTool,
) -> dict[str, Any]:
    args_schema: Any = getattr(
        tool,
        "args_schema",
        None,
    )

    schema_payload: Any = None

    if args_schema is not None:
        model_json_schema = getattr(
            args_schema,
            "model_json_schema",
            None,
        )
        if callable(model_json_schema):
            try:
                schema_payload = model_json_schema()
            except Exception:
                schema_payload = None

        if schema_payload is None:
            schema = getattr(
                args_schema,
                "schema",
                None,
            )
            if callable(schema):
                try:
                    schema_payload = schema()
                except Exception:
                    schema_payload = None

    return {
        "name": str(getattr(tool, "name", "") or ""),
        "description": str(
            getattr(tool, "description", "") or ""
        ),
        "args_schema": (
            schema_payload
            if schema_payload is not None
            else str(args_schema or "")
        ),
    }


def estimate_active_tool_schema_tokens(
    tools: Sequence[BaseTool],
) -> int:
    if not tools:
        return 0

    payload = [
        _tool_schema_payload(tool)
        for tool in tools
    ]

    text = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
        sort_keys=True,
    )

    return int(
        estimate_text_tokens(text)
    )


def _extract_current_turn_messages(
    *,
    messages: Sequence[BaseMessage],
    current_human_message_id: str,
) -> list[BaseMessage]:
    normalized_id = str(
        current_human_message_id
    ).strip()

    if not normalized_id:
        return list(messages)

    start_index: int | None = None

    for index, message in enumerate(messages):
        if (
            str(
                getattr(message, "id", "")
                or ""
            )
            == normalized_id
        ):
            start_index = index
            break

    if start_index is None:
        # 兼容旧 checkpoint / 特殊恢复路径。
        return list(messages)

    return list(
        messages[start_index:]
    )


def _planner_guidance(
    plan: ContextPlan,
) -> str:
    retrieval_policy = (
        "允许并且本轮需要新的外部检索。"
        if plan.external_retrieval_required
        else (
            "允许在确有必要时进行新的外部检索。"
            if plan.external_retrieval_allowed
            else (
                "本轮不要重新执行新的外部检索；"
                "优先使用已经恢复的历史证据和当前上下文。"
            )
        )
    )

    return (
        "# 本轮 Context Plan\n\n"
        f"- response_goal: {plan.response_goal}\n"
        f"- history_scope: {plan.history_scope}\n"
        f"- raw_tool_evidence_required: "
        f"{plan.raw_tool_evidence_required}\n"
        f"- long_term_memory_required: "
        f"{plan.long_term_memory_required}\n"
        f"- external_retrieval_required: "
        f"{plan.external_retrieval_required}\n"
        f"- external_retrieval_allowed: "
        f"{plan.external_retrieval_allowed}\n\n"
        f"执行约束：{retrieval_policy}"
    )


class AgentContextPipeline:
    """单轮 Agent Context 的统一运行时。"""

    def __init__(
        self,
        *,
        planner: ContextPlanner,
        event_store: ConversationEventStore,
        model_context_limit_tokens: int = 32768,
        reserved_output_tokens: int = 4096,
        safety_margin_tokens: int = 1024,
        recent_turn_limit: int = 3,
        historical_turn_limit: int = 3,
    ) -> None:
        self.planner = planner
        self.event_store = event_store

        self.retriever = ConversationRetriever(
            store=event_store,
            recent_turn_limit=recent_turn_limit,
            historical_turn_limit=historical_turn_limit,
        )

        self.assembler = ContextAssembler()
        self.budget_manager = ContextBudgetManager()

        self.compressor = ContextCompressor(
            budget_manager=self.budget_manager
        )

        self.model_context_limit_tokens = int(
            model_context_limit_tokens
        )
        self.reserved_output_tokens = int(
            reserved_output_tokens
        )
        self.safety_margin_tokens = int(
            safety_margin_tokens
        )

    def prepare_new_turn(
        self,
        *,
        question: str,
        thread_id: str,
        thread_summary: str = "",
    ) -> PreparedTurnContext:
        turn_ids = self.event_store.list_turn_ids(
            thread_id=thread_id
        )

        availability_item: TurnIndexItem | None = None

        if turn_ids:
            availability_item = TurnIndexItem(
                turn_id=turn_ids[-1],
                user_goal="history_available",
                assistant_outcome="",
                entities=[],
                has_tool_evidence=False,
                tool_names=[],
            )

        navigation = NavigationContext(
            current_user_input=question,
            # Planner fix2 只把它作为 availability 信号，
            # 不把正文送给 Planner。
            thread_summary=thread_summary,
            recent_turns=(
                [availability_item]
                if availability_item is not None
                else []
            ),
            history_candidates=(
                [availability_item]
                if availability_item is not None
                else []
            ),
            capability_catalog=[],
            runtime_notes=[],
        )

        planner_result: ContextPlannerResult = (
            self.planner.plan(
                navigation
            )
        )

        plan = planner_result.plan

        retrieval = self.retriever.retrieve(
            thread_id=thread_id,
            plan=plan,
        )

        return PreparedTurnContext(
            context_plan=_model_dump(plan),
            context_retrieval=(
                serialize_retrieval_result(
                    retrieval
                )
            ),
            context_planner_trace={
                "latency_ms": float(
                    planner_result.latency_ms
                ),
                "usage_metadata": dict(
                    planner_result.usage_metadata
                    or {}
                ),
                "navigation_characters": int(
                    planner_result.navigation_characters
                ),
                "raw_model_output": str(
                    planner_result.raw_model_output
                ),
            },
        )

    def build_for_model(
        self,
        *,
        state: dict[str, Any],
        base_system_prompt: str,
        skill_runtime_prompt: str | None,
        long_term_memory_text: str | None,
        thread_summary: str | None,
        active_tools: Sequence[BaseTool],
        finalize_instruction: str | None = None,
    ) -> BuiltModelContext:
        raw_plan = state.get(
            "context_plan"
        )
        raw_retrieval = state.get(
            "context_retrieval"
        )

        if not isinstance(raw_plan, dict):
            raise ValueError(
                "Graph State 缺少 context_plan。"
            )

        if not isinstance(raw_retrieval, dict):
            raise ValueError(
                "Graph State 缺少 context_retrieval。"
            )

        plan = ContextPlan.model_validate(
            raw_plan
        )

        retrieval = deserialize_retrieval_result(
            raw_retrieval
        )

        current_messages = (
            _extract_current_turn_messages(
                messages=list(
                    state.get(
                        "messages",
                        [],
                    )
                    or []
                ),
                current_human_message_id=str(
                    state.get(
                        "context_current_human_message_id",
                        "",
                    )
                    or ""
                ),
            )
        )

        if finalize_instruction:
            current_messages.append(
                HumanMessage(
                    content=finalize_instruction
                )
            )

        # 有原始 Event Store 历史时，不再重复加入 Rolling Summary。
        # 只有历史检索为空时才把 Summary 当 legacy fallback。
        summary_fallback = ""

        if (
            plan.history_required
            and not retrieval.turns
        ):
            summary_fallback = str(
                thread_summary or ""
            ).strip()

        effective_system_prompt = (
            str(base_system_prompt).strip()
            + "\n\n"
            + _planner_guidance(plan)
        )

        memory_text = (
            str(
                long_term_memory_text
                or ""
            ).strip()
            if plan.long_term_memory_required
            else ""
        )

        assembly = self.assembler.assemble(
            system_prompt=effective_system_prompt,
            current_messages=current_messages,
            conversation_retrieval=retrieval,
            skill_runtime_prompt=skill_runtime_prompt,
            long_term_memory_text=memory_text,
            thread_summary=summary_fallback,
        )

        tool_schema_tokens = (
            estimate_active_tool_schema_tokens(
                active_tools
            )
        )

        budget_config = ContextBudgetConfig(
            model_context_limit_tokens=(
                self.model_context_limit_tokens
            ),
            reserved_output_tokens=(
                self.reserved_output_tokens
            ),
            tool_schema_tokens=(
                tool_schema_tokens
            ),
            safety_margin_tokens=(
                self.safety_margin_tokens
            ),
        )

        compressed = (
            self.compressor.compress_to_fit(
                assembly=assembly,
                budget_config=budget_config,
                plan=plan,
            )
        )

        final_assembly = compressed.assembly

        diagnostics = {
            "enabled": True,
            "plan": _model_dump(plan),
            "retrieval": {
                "strategy": (
                    retrieval.retrieval_strategy
                ),
                "selected_turn_ids": list(
                    retrieval.selected_turn_ids
                ),
                "selected_event_count": int(
                    retrieval.selected_event_count
                ),
                "candidate_turn_count": int(
                    retrieval.candidate_turn_count
                ),
            },
            "budget": {
                "model_context_limit_tokens": (
                    compressed.final_budget
                    .model_context_limit_tokens
                ),
                "available_message_tokens": (
                    compressed.final_budget
                    .available_message_tokens
                ),
                "estimated_message_tokens": (
                    compressed.final_budget
                    .estimated_message_tokens
                ),
                "tool_schema_tokens": (
                    compressed.final_budget
                    .tool_schema_tokens
                ),
                "remaining_message_tokens": (
                    compressed.final_budget
                    .remaining_message_tokens
                ),
                "fits": (
                    compressed.final_budget.fits
                ),
                "source_estimated_tokens": dict(
                    compressed.final_budget
                    .source_estimated_tokens
                ),
            },
            "compression": {
                "compressed": compressed.compressed,
                "tokens_saved": compressed.tokens_saved,
                "removed_turn_ids": list(
                    compressed.removed_turn_ids
                ),
                "actions": [
                    {
                        "action": action.action,
                        "source": action.source,
                        "message_id": action.message_id,
                        "context_ref": action.context_ref,
                        "before_tokens": action.before_tokens,
                        "after_tokens": action.after_tokens,
                        "detail": action.detail,
                    }
                    for action in compressed.actions
                ],
            },
            "tool_pair_integrity_ok": (
                final_assembly
                .tool_pair_integrity_ok
            ),
            "final_message_count": len(
                final_assembly.messages
            ),
        }

        return BuiltModelContext(
            messages=list(
                final_assembly.messages
            ),
            diagnostics=diagnostics,
        )