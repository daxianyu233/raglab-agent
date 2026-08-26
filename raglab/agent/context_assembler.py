"""RAGLab Context Assembler - Phase 6A/6B/6C.

Phase 6C 增量：
- message_context_refs 与 messages 一一对应；
- Conversation History 消息记录其 turn_id；
- history_turn_priorities 记录 Retriever 给出的 turn 优先级；
- 为后续安全的“整轮历史裁剪”提供结构信息。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)

from raglab.agent.context_manager import (
    audit_model_input,
)
from raglab.agent.conversation_event_hydrator import (
    hydrate_conversation_events,
)
from raglab.agent.conversation_retriever import (
    ConversationRetrievalResult,
)


SOURCE_BASE_SYSTEM = "base_system"
SOURCE_SKILL_RUNTIME = "skill_runtime"
SOURCE_LONG_TERM_MEMORY = "long_term_memory"
SOURCE_THREAD_SUMMARY = "thread_summary"
SOURCE_CONVERSATION_HISTORY_HEADER = (
    "conversation_history_header"
)
SOURCE_CONVERSATION_HISTORY = "conversation_history"
SOURCE_CURRENT_TURN = "current_turn"


@dataclass(frozen=True)
class ContextAssemblyResult:
    messages: list[BaseMessage]

    # 与 messages 一一对应。
    message_sources: list[str]

    # Conversation History 消息填 turn_id；
    # 其他来源为 None。
    message_context_refs: list[str | None]

    history_message_count: int
    current_message_count: int
    system_message_count: int

    source_turn_ids: list[str]

    # turn_id -> retrieval priority。
    # historical_search 使用 retrieval_score；
    # recent_turns 同分时后面的 turn 获得轻微 recency bonus。
    history_turn_priorities: dict[str, float]

    context_audit: dict

    @property
    def estimated_message_tokens(
        self,
    ) -> int:
        return int(
            self.context_audit.get(
                "estimated_message_tokens",
                0,
            )
        )

    @property
    def tool_pair_integrity_ok(
        self,
    ) -> bool:
        return bool(
            self.context_audit.get(
                "tool_pair_integrity_ok",
                False,
            )
        )

    @property
    def source_message_counts(
        self,
    ) -> dict[str, int]:
        return dict(
            Counter(
                self.message_sources
            )
        )


class ContextAssembler:
    def __init__(
        self,
        *,
        history_header: str = (
            "以下消息来自当前聊天线程的 Conversation Event Store，"
            "是根据本轮 ContextPlan 按需恢复的历史。"
            "它们只用于恢复当前任务所依赖的历史上下文；"
            "不要把它们误认为本轮新执行的外部检索结果。"
        ),
    ) -> None:
        self.history_header = str(
            history_header
        ).strip()

    def assemble(
        self,
        *,
        system_prompt: str,
        current_messages: Sequence[BaseMessage],
        conversation_retrieval: (
            ConversationRetrievalResult | None
        ) = None,
        skill_runtime_prompt: str | None = None,
        long_term_memory_text: str | None = None,
        thread_summary: str | None = None,
    ) -> ContextAssemblyResult:
        normalized_system_prompt = str(
            system_prompt
        ).strip()

        if not normalized_system_prompt:
            raise ValueError(
                "system_prompt 不能为空。"
            )

        current = list(
            current_messages
        )

        if not current:
            raise ValueError(
                "current_messages 不能为空。"
            )

        assembled: list[BaseMessage] = []
        message_sources: list[str] = []
        message_context_refs: list[
            str | None
        ] = []

        def append_message(
            message: BaseMessage,
            source: str,
            context_ref: str | None = None,
        ) -> None:
            assembled.append(message)
            message_sources.append(source)
            message_context_refs.append(
                context_ref
            )

        append_message(
            SystemMessage(
                content=normalized_system_prompt
            ),
            SOURCE_BASE_SYSTEM,
        )

        system_message_count = 1

        normalized_skill_prompt = str(
            skill_runtime_prompt or ""
        ).strip()

        if normalized_skill_prompt:
            append_message(
                SystemMessage(
                    content=normalized_skill_prompt
                ),
                SOURCE_SKILL_RUNTIME,
            )
            system_message_count += 1

        normalized_memory = str(
            long_term_memory_text or ""
        ).strip()

        if normalized_memory:
            append_message(
                SystemMessage(
                    content=(
                        "以下是调用方已经根据本轮 ContextPlan "
                        "选择出的跨会话长期记忆。"
                        "它仅用于理解用户稳定背景和偏好，"
                        "不能替代知识库或 Tool Evidence：\n\n"
                        + normalized_memory
                    )
                ),
                SOURCE_LONG_TERM_MEMORY,
            )
            system_message_count += 1

        normalized_summary = str(
            thread_summary or ""
        ).strip()

        if normalized_summary:
            append_message(
                SystemMessage(
                    content=(
                        "以下是当前线程较早历史的工作摘要。"
                        "它是派生视图，不是原始历史 Source of Truth；"
                        "如果与按需恢复的原始 Conversation Events 冲突，"
                        "应优先采用原始 Events：\n\n"
                        + normalized_summary
                    )
                ),
                SOURCE_THREAD_SUMMARY,
            )
            system_message_count += 1

        history_messages: list[
            BaseMessage
        ] = []

        source_turn_ids: list[
            str
        ] = []

        history_turn_priorities: dict[
            str,
            float,
        ] = {}

        if (
            conversation_retrieval is not None
            and conversation_retrieval.turns
        ):
            if self.history_header:
                append_message(
                    SystemMessage(
                        content=self.history_header
                    ),
                    SOURCE_CONVERSATION_HISTORY_HEADER,
                )
                system_message_count += 1

            total_turns = len(
                conversation_retrieval.turns
            )

            for turn_index, turn in enumerate(
                conversation_retrieval.turns
            ):
                if not turn.selected_events:
                    continue

                source_turn_ids.append(
                    turn.turn_id
                )

                # retrieval_score 是主要优先级。
                # recent_turns 通常同为 1.0，此时给较新的 turn
                # 极小 recency bonus，只用于预算裁剪时破同分。
                recency_bonus = (
                    (turn_index + 1)
                    / max(1, total_turns)
                    * 1e-6
                )

                history_turn_priorities[
                    turn.turn_id
                ] = (
                    float(
                        turn.retrieval_score
                    )
                    + recency_bonus
                )

                hydrated = (
                    hydrate_conversation_events(
                        turn.selected_events
                    )
                )

                history_messages.extend(
                    hydrated
                )

                for message in hydrated:
                    append_message(
                        message,
                        SOURCE_CONVERSATION_HISTORY,
                        turn.turn_id,
                    )

        for message in current:
            append_message(
                message,
                SOURCE_CURRENT_TURN,
            )

        if not (
            len(assembled)
            == len(message_sources)
            == len(message_context_refs)
        ):
            raise RuntimeError(
                "Context Assembler 内部错误："
                "messages/source/ref 数量不一致。"
            )

        audit = audit_model_input(
            assembled
        )

        if not bool(
            audit.get(
                "tool_pair_integrity_ok",
                False,
            )
        ):
            raise ValueError(
                "Context Assembly 后 Tool Pair 不完整："
                f"unresolved={audit.get('unresolved_tool_call_ids', [])}；"
                f"orphan={audit.get('orphan_tool_message_ids', [])}"
            )

        return ContextAssemblyResult(
            messages=assembled,
            message_sources=message_sources,
            message_context_refs=(
                message_context_refs
            ),
            history_message_count=len(
                history_messages
            ),
            current_message_count=len(
                current
            ),
            system_message_count=system_message_count,
            source_turn_ids=source_turn_ids,
            history_turn_priorities=(
                history_turn_priorities
            ),
            context_audit=audit,
        )