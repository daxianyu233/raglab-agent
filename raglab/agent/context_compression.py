"""RAGLab Context Compression / Pruning - Phase 6C.

核心原则：
1. Event Store 是 Source of Truth，本模块永远不修改原始历史；
2. Selection first, Compression second；
3. 当前 Human 和 Base System 不做静默截断；
4. Tool Result 可以压缩内容，但 Tool Pair 外壳必须保留；
5. Conversation History 按 turn 原子裁剪，避免拆坏协议；
6. 如果安全压缩后仍超预算，明确抛出错误，不伪装成功。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from raglab.agent.context_assembler import (
    ContextAssemblyResult,
    SOURCE_BASE_SYSTEM,
    SOURCE_CONVERSATION_HISTORY,
    SOURCE_CONVERSATION_HISTORY_HEADER,
    SOURCE_CURRENT_TURN,
    SOURCE_LONG_TERM_MEMORY,
    SOURCE_THREAD_SUMMARY,
)
from raglab.agent.context_budget import (
    ContextBudgetConfig,
    ContextBudgetManager,
    ContextBudgetReport,
)
from raglab.agent.context_manager import (
    audit_model_input,
    estimate_message_tokens,
    estimate_text_tokens,
    message_content_to_text,
)
from raglab.agent.context_plan import (
    ContextPlan,
)


TOOL_COMPRESSION_MARKER = (
    "\n\n[... Tool Result 已在本次模型上下文中压缩；"
    "完整原文仍保存在 Conversation Event Store，可按需重新恢复 ...]\n\n"
)

GENERIC_COMPRESSION_MARKER = (
    "\n\n[... 本段仅在本次模型上下文中压缩；"
    "原始数据未被修改 ...]\n\n"
)


class ContextBudgetExceededError(
    RuntimeError
):
    """安全压缩后仍无法满足 Context Budget。"""


@dataclass(frozen=True)
class ContextCompressionPolicy:
    """Phase 6C 的保守压缩策略。"""

    min_tool_message_tokens: int = 256
    min_long_term_memory_tokens: int = 192
    min_thread_summary_tokens: int = 192

    # 无论如何至少保留一个 Planner/Retriever
    # 已选择出的历史 turn，避免把任务依赖完全删光。
    min_history_turns: int = 1

    def __post_init__(
        self,
    ) -> None:
        for name in (
            "min_tool_message_tokens",
            "min_long_term_memory_tokens",
            "min_thread_summary_tokens",
            "min_history_turns",
        ):
            value = int(
                getattr(
                    self,
                    name,
                )
            )

            if value < 0:
                raise ValueError(
                    f"{name} 不能小于 0。"
                )


@dataclass(frozen=True)
class ContextCompressionAction:
    action: str
    source: str

    message_id: str | None = None
    context_ref: str | None = None

    before_tokens: int = 0
    after_tokens: int = 0

    detail: str = ""


@dataclass(frozen=True)
class ContextCompressionResult:
    assembly: ContextAssemblyResult

    initial_budget: ContextBudgetReport
    final_budget: ContextBudgetReport

    actions: list[
        ContextCompressionAction
    ]

    removed_turn_ids: list[str]

    @property
    def tokens_saved(
        self,
    ) -> int:
        return (
            self.initial_budget
            .estimated_message_tokens
            - self.final_budget
            .estimated_message_tokens
        )

    @property
    def compressed(
        self,
    ) -> bool:
        return bool(
            self.actions
        )


def _copy_message_with_content(
    message: BaseMessage,
    content: str,
) -> BaseMessage:
    """兼容 Pydantic v1/v2 的 LangChain Message copy。"""

    model_copy = getattr(
        message,
        "model_copy",
        None,
    )

    if callable(
        model_copy
    ):
        return model_copy(
            update={
                "content": content,
            }
        )

    copy_method = getattr(
        message,
        "copy",
        None,
    )

    if callable(
        copy_method
    ):
        return copy_method(
            update={
                "content": content,
            }
        )

    raise TypeError(
        "当前 LangChain Message 不支持安全复制。"
    )


def _truncate_text_to_token_target(
    *,
    text: str,
    target_tokens: int,
    marker: str,
) -> str:
    """头尾保留式压缩。

    这里只压 Context View，不修改 Event Store。
    """

    original = str(
        text
    )

    if (
        target_tokens <= 0
        or not original
    ):
        return marker.strip()

    if (
        estimate_text_tokens(
            original
        )
        <= target_tokens
    ):
        return original

    marker_tokens = (
        estimate_text_tokens(
            marker
        )
    )

    # 即使目标非常小，也必须留下明确的“这是压缩视图”标记。
    if (
        target_tokens
        <= marker_tokens + 4
    ):
        return marker.strip()

    low = 0
    high = len(
        original
    )

    best = marker.strip()

    while low <= high:
        keep_chars = (
            low + high
        ) // 2

        head_chars = int(
            keep_chars
            * 0.7
        )

        tail_chars = (
            keep_chars
            - head_chars
        )

        if tail_chars > 0:
            candidate = (
                original[
                    :head_chars
                ]
                + marker
                + original[
                    -tail_chars:
                ]
            )
        else:
            candidate = (
                original[
                    :head_chars
                ]
                + marker
            )

        estimated = (
            estimate_text_tokens(
                candidate
            )
        )

        if estimated <= target_tokens:
            best = candidate
            low = (
                keep_chars
                + 1
            )
        else:
            high = (
                keep_chars
                - 1
            )

    return best


def _rebuild_assembly(
    *,
    original: ContextAssemblyResult,
    messages: list[
        BaseMessage
    ],
    sources: list[str],
    refs: list[
        str | None
    ],
) -> ContextAssemblyResult:
    if not (
        len(messages)
        == len(sources)
        == len(refs)
    ):
        raise RuntimeError(
            "Compression 内部错误："
            "messages/source/ref 数量不一致。"
        )

    audit = audit_model_input(
        messages
    )

    if not bool(
        audit.get(
            "tool_pair_integrity_ok",
            False,
        )
    ):
        raise RuntimeError(
            "Compression 破坏了 Tool Pair："
            f"unresolved={audit.get('unresolved_tool_call_ids', [])}；"
            f"orphan={audit.get('orphan_tool_message_ids', [])}"
        )

    retained_turn_ids: list[
        str
    ] = []

    for ref, source in zip(
        refs,
        sources,
    ):
        if (
            source
            == SOURCE_CONVERSATION_HISTORY
            and ref
            and ref not in retained_turn_ids
        ):
            retained_turn_ids.append(
                ref
            )

    retained_priorities = {
        turn_id: priority
        for (
            turn_id,
            priority,
        ) in (
            original
            .history_turn_priorities
            .items()
        )
        if turn_id in retained_turn_ids
    }

    system_count = sum(
        isinstance(
            message,
            SystemMessage,
        )
        for message in messages
    )

    return ContextAssemblyResult(
        messages=messages,
        message_sources=sources,
        message_context_refs=refs,
        history_message_count=sum(
            source
            == SOURCE_CONVERSATION_HISTORY
            for source in sources
        ),
        current_message_count=sum(
            source
            == SOURCE_CURRENT_TURN
            for source in sources
        ),
        system_message_count=(
            system_count
        ),
        source_turn_ids=(
            retained_turn_ids
        ),
        history_turn_priorities=(
            retained_priorities
        ),
        context_audit=audit,
    )


class ContextCompressor:
    """仅在 Budget 超限时对已选择 Context 做安全压缩。"""

    def __init__(
        self,
        *,
        budget_manager: (
            ContextBudgetManager
            | None
        ) = None,
        policy: (
            ContextCompressionPolicy
            | None
        ) = None,
    ) -> None:
        self.budget_manager = (
            budget_manager
            or ContextBudgetManager()
        )

        self.policy = (
            policy
            or ContextCompressionPolicy()
        )

    def compress_to_fit(
        self,
        *,
        assembly: ContextAssemblyResult,
        budget_config: ContextBudgetConfig,
        plan: ContextPlan | None = None,
    ) -> ContextCompressionResult:
        initial_budget = (
            self.budget_manager
            .evaluate(
                assembly=assembly,
                config=budget_config,
            )
        )

        if initial_budget.fits:
            return ContextCompressionResult(
                assembly=assembly,
                initial_budget=initial_budget,
                final_budget=initial_budget,
                actions=[],
                removed_turn_ids=[],
            )

        working = assembly
        actions: list[
            ContextCompressionAction
        ] = []
        removed_turn_ids: list[
            str
        ] = []

        # ----------------------------------------------------
        # Step 1：删除纯说明 Header。
        # ----------------------------------------------------
        working, step_actions = (
            self._remove_source(
                assembly=working,
                source=(
                    SOURCE_CONVERSATION_HISTORY_HEADER
                ),
                detail=(
                    "历史来源说明属于辅助文本，"
                    "超预算时优先移除。"
                ),
            )
        )
        actions.extend(
            step_actions
        )

        if self._fits(
            working,
            budget_config,
        ):
            return self._finish(
                assembly=working,
                initial_budget=initial_budget,
                budget_config=budget_config,
                actions=actions,
                removed_turn_ids=removed_turn_ids,
            )

        # ----------------------------------------------------
        # Step 2：已有按需恢复的原始历史时，
        # Rolling Summary 属于派生/重复视图，可优先删除。
        # ----------------------------------------------------
        has_raw_history = any(
            source
            == SOURCE_CONVERSATION_HISTORY
            for source
            in working.message_sources
        )

        if has_raw_history:
            (
                working,
                step_actions,
            ) = self._remove_source(
                assembly=working,
                source=(
                    SOURCE_THREAD_SUMMARY
                ),
                detail=(
                    "已有 Planner/Retriever 按需恢复的原始历史，"
                    "Rolling Summary 作为派生视图优先删除。"
                ),
            )
            actions.extend(
                step_actions
            )

        if self._fits(
            working,
            budget_config,
        ):
            return self._finish(
                assembly=working,
                initial_budget=initial_budget,
                budget_config=budget_config,
                actions=actions,
                removed_turn_ids=removed_turn_ids,
            )

        # ----------------------------------------------------
        # Step 3：压缩历史 Tool Result。
        # 保留 AI tool_call + ToolMessage 外壳。
        # ----------------------------------------------------
        (
            working,
            step_actions,
        ) = self._compress_tool_messages(
            assembly=working,
            budget_config=budget_config,
            source_filter=(
                SOURCE_CONVERSATION_HISTORY
            ),
            minimum_tokens=(
                self.policy
                .min_tool_message_tokens
            ),
        )
        actions.extend(
            step_actions
        )

        if self._fits(
            working,
            budget_config,
        ):
            return self._finish(
                assembly=working,
                initial_budget=initial_budget,
                budget_config=budget_config,
                actions=actions,
                removed_turn_ids=removed_turn_ids,
            )

        # ----------------------------------------------------
        # Step 4：压缩当前轮巨大 Tool Result。
        # 当前 Human 不动，Tool Pair id 不动。
        # ----------------------------------------------------
        (
            working,
            step_actions,
        ) = self._compress_tool_messages(
            assembly=working,
            budget_config=budget_config,
            source_filter=(
                SOURCE_CURRENT_TURN
            ),
            minimum_tokens=(
                self.policy
                .min_tool_message_tokens
            ),
        )
        actions.extend(
            step_actions
        )

        if self._fits(
            working,
            budget_config,
        ):
            return self._finish(
                assembly=working,
                initial_budget=initial_budget,
                budget_config=budget_config,
                actions=actions,
                removed_turn_ids=removed_turn_ids,
            )

        # ----------------------------------------------------
        # Step 5：如果 Retriever 选了多个历史 turn，
        # 整轮删除最低优先级 turn。
        # 至少保留 min_history_turns。
        # ----------------------------------------------------
        (
            working,
            step_actions,
            step_removed_turns,
        ) = self._drop_low_priority_history_turns(
            assembly=working,
            budget_config=budget_config,
        )

        actions.extend(
            step_actions
        )

        removed_turn_ids.extend(
            step_removed_turns
        )

        if self._fits(
            working,
            budget_config,
        ):
            return self._finish(
                assembly=working,
                initial_budget=initial_budget,
                budget_config=budget_config,
                actions=actions,
                removed_turn_ids=removed_turn_ids,
            )

        # ----------------------------------------------------
        # Step 6：压缩 Long-term Memory Context View。
        # Event/Memory 原始数据不修改。
        # ----------------------------------------------------
        (
            working,
            step_actions,
        ) = self._compress_single_source_message(
            assembly=working,
            budget_config=budget_config,
            source=(
                SOURCE_LONG_TERM_MEMORY
            ),
            minimum_tokens=(
                self.policy
                .min_long_term_memory_tokens
            ),
            marker=(
                GENERIC_COMPRESSION_MARKER
            ),
            detail=(
                "压缩本轮 Long-term Memory Context View。"
            ),
        )
        actions.extend(
            step_actions
        )

        if self._fits(
            working,
            budget_config,
        ):
            return self._finish(
                assembly=working,
                initial_budget=initial_budget,
                budget_config=budget_config,
                actions=actions,
                removed_turn_ids=removed_turn_ids,
            )

        # ----------------------------------------------------
        # Step 7：如果没有原始历史可以替代 Summary，
        # 最后再压缩 Summary 本身。
        # ----------------------------------------------------
        (
            working,
            step_actions,
        ) = self._compress_single_source_message(
            assembly=working,
            budget_config=budget_config,
            source=(
                SOURCE_THREAD_SUMMARY
            ),
            minimum_tokens=(
                self.policy
                .min_thread_summary_tokens
            ),
            marker=(
                GENERIC_COMPRESSION_MARKER
            ),
            detail=(
                "压缩本轮 Rolling Summary Context View。"
            ),
        )
        actions.extend(
            step_actions
        )

        final_budget = (
            self.budget_manager
            .evaluate(
                assembly=working,
                config=budget_config,
            )
        )

        if not final_budget.fits:
            raise ContextBudgetExceededError(
                "经过安全压缩后仍超出模型 Context Budget："
                f"overflow_tokens={final_budget.overflow_tokens}。"
                "Base System、当前 Human 或必要协议内容不会被静默截断；"
                "应降低检索 Top-K、提高模型 Context Window，"
                "或进入更高级的语义压缩/分段处理。"
            )

        return ContextCompressionResult(
            assembly=working,
            initial_budget=initial_budget,
            final_budget=final_budget,
            actions=actions,
            removed_turn_ids=(
                removed_turn_ids
            ),
        )

    def _fits(
        self,
        assembly: ContextAssemblyResult,
        config: ContextBudgetConfig,
    ) -> bool:
        return (
            self.budget_manager
            .evaluate(
                assembly=assembly,
                config=config,
            )
            .fits
        )

    def _finish(
        self,
        *,
        assembly: ContextAssemblyResult,
        initial_budget: ContextBudgetReport,
        budget_config: ContextBudgetConfig,
        actions: list[
            ContextCompressionAction
        ],
        removed_turn_ids: list[str],
    ) -> ContextCompressionResult:
        final_budget = (
            self.budget_manager
            .evaluate(
                assembly=assembly,
                config=budget_config,
            )
        )

        return ContextCompressionResult(
            assembly=assembly,
            initial_budget=initial_budget,
            final_budget=final_budget,
            actions=list(actions),
            removed_turn_ids=list(
                removed_turn_ids
            ),
        )

    def _remove_source(
        self,
        *,
        assembly: ContextAssemblyResult,
        source: str,
        detail: str,
    ) -> tuple[
        ContextAssemblyResult,
        list[
            ContextCompressionAction
        ],
    ]:
        messages: list[
            BaseMessage
        ] = []
        sources: list[str] = []
        refs: list[
            str | None
        ] = []

        actions: list[
            ContextCompressionAction
        ] = []

        for (
            message,
            current_source,
            ref,
        ) in zip(
            assembly.messages,
            assembly.message_sources,
            assembly.message_context_refs,
        ):
            if (
                current_source
                == source
            ):
                actions.append(
                    ContextCompressionAction(
                        action="remove_message",
                        source=source,
                        message_id=(
                            getattr(
                                message,
                                "id",
                                None,
                            )
                        ),
                        context_ref=ref,
                        before_tokens=(
                            estimate_message_tokens(
                                message
                            )
                        ),
                        after_tokens=0,
                        detail=detail,
                    )
                )
                continue

            messages.append(
                message
            )
            sources.append(
                current_source
            )
            refs.append(
                ref
            )

        if not actions:
            return (
                assembly,
                [],
            )

        rebuilt = _rebuild_assembly(
            original=assembly,
            messages=messages,
            sources=sources,
            refs=refs,
        )

        return (
            rebuilt,
            actions,
        )

    def _compress_tool_messages(
        self,
        *,
        assembly: ContextAssemblyResult,
        budget_config: ContextBudgetConfig,
        source_filter: str,
        minimum_tokens: int,
    ) -> tuple[
        ContextAssemblyResult,
        list[
            ContextCompressionAction
        ],
    ]:
        working = assembly
        actions: list[
            ContextCompressionAction
        ] = []

        while True:
            report = (
                self.budget_manager
                .evaluate(
                    assembly=working,
                    config=budget_config,
                )
            )

            if report.fits:
                break

            candidates: list[
                tuple[
                    int,
                    int,
                    ToolMessage,
                ]
            ] = []

            for index, (
                message,
                source,
            ) in enumerate(
                zip(
                    working.messages,
                    working.message_sources,
                )
            ):
                if (
                    source
                    != source_filter
                    or not isinstance(
                        message,
                        ToolMessage,
                    )
                ):
                    continue

                token_count = (
                    estimate_message_tokens(
                        message
                    )
                )

                if (
                    token_count
                    <= minimum_tokens
                ):
                    continue

                candidates.append(
                    (
                        token_count,
                        index,
                        message,
                    )
                )

            if not candidates:
                break

            # 每次先压最大的 Tool Result。
            (
                before_tokens,
                index,
                message,
            ) = max(
                candidates,
                key=lambda item: item[0],
            )

            desired_tokens = max(
                minimum_tokens,
                before_tokens
                - report.overflow_tokens
                - 16,
            )

            original_text = (
                message_content_to_text(
                    message.content
                )
            )

            new_text = (
                _truncate_text_to_token_target(
                    text=original_text,
                    target_tokens=(
                        desired_tokens
                    ),
                    marker=(
                        TOOL_COMPRESSION_MARKER
                    ),
                )
            )

            new_message = (
                _copy_message_with_content(
                    message,
                    new_text,
                )
            )

            after_tokens = (
                estimate_message_tokens(
                    new_message
                )
            )

            if (
                after_tokens
                >= before_tokens
            ):
                break

            new_messages = list(
                working.messages
            )

            new_messages[
                index
            ] = new_message

            working = _rebuild_assembly(
                original=working,
                messages=new_messages,
                sources=list(
                    working.message_sources
                ),
                refs=list(
                    working.message_context_refs
                ),
            )

            actions.append(
                ContextCompressionAction(
                    action="truncate_tool_result",
                    source=source_filter,
                    message_id=(
                        getattr(
                            message,
                            "id",
                            None,
                        )
                    ),
                    context_ref=(
                        working
                        .message_context_refs[
                            index
                        ]
                    ),
                    before_tokens=(
                        before_tokens
                    ),
                    after_tokens=(
                        after_tokens
                    ),
                    detail=(
                        "仅压缩 ToolMessage.content；"
                        "tool_call_id/name 和原始 Event Store 数据保持不变。"
                    ),
                )
            )

        return (
            working,
            actions,
        )

    def _drop_low_priority_history_turns(
        self,
        *,
        assembly: ContextAssemblyResult,
        budget_config: ContextBudgetConfig,
    ) -> tuple[
        ContextAssemblyResult,
        list[
            ContextCompressionAction
        ],
        list[str],
    ]:
        working = assembly
        actions: list[
            ContextCompressionAction
        ] = []
        removed_turn_ids: list[
            str
        ] = []

        while True:
            report = (
                self.budget_manager
                .evaluate(
                    assembly=working,
                    config=budget_config,
                )
            )

            if report.fits:
                break

            current_turn_ids = list(
                working.source_turn_ids
            )

            if (
                len(current_turn_ids)
                <= self.policy
                .min_history_turns
            ):
                break

            lowest_priority_turn = min(
                current_turn_ids,
                key=lambda turn_id: (
                    working
                    .history_turn_priorities
                    .get(
                        turn_id,
                        0.0,
                    )
                ),
            )

            messages: list[
                BaseMessage
            ] = []
            sources: list[str] = []
            refs: list[
                str | None
            ] = []

            removed_tokens = 0

            for (
                message,
                source,
                ref,
            ) in zip(
                working.messages,
                working.message_sources,
                working.message_context_refs,
            ):
                if (
                    source
                    == SOURCE_CONVERSATION_HISTORY
                    and ref
                    == lowest_priority_turn
                ):
                    removed_tokens += (
                        estimate_message_tokens(
                            message
                        )
                    )
                    continue

                messages.append(
                    message
                )
                sources.append(
                    source
                )
                refs.append(
                    ref
                )

            if (
                len(messages)
                == len(
                    working.messages
                )
            ):
                break

            working = _rebuild_assembly(
                original=working,
                messages=messages,
                sources=sources,
                refs=refs,
            )

            removed_turn_ids.append(
                lowest_priority_turn
            )

            actions.append(
                ContextCompressionAction(
                    action=(
                        "drop_history_turn"
                    ),
                    source=(
                        SOURCE_CONVERSATION_HISTORY
                    ),
                    context_ref=(
                        lowest_priority_turn
                    ),
                    before_tokens=(
                        removed_tokens
                    ),
                    after_tokens=0,
                    detail=(
                        "整轮移除最低优先级历史，"
                        "不拆分 Human/AI/Tool 关系。"
                    ),
                )
            )

        return (
            working,
            actions,
            removed_turn_ids,
        )

    def _compress_single_source_message(
        self,
        *,
        assembly: ContextAssemblyResult,
        budget_config: ContextBudgetConfig,
        source: str,
        minimum_tokens: int,
        marker: str,
        detail: str,
    ) -> tuple[
        ContextAssemblyResult,
        list[
            ContextCompressionAction
        ],
    ]:
        report = (
            self.budget_manager
            .evaluate(
                assembly=assembly,
                config=budget_config,
            )
        )

        if report.fits:
            return (
                assembly,
                [],
            )

        candidates = [
            (
                estimate_message_tokens(
                    message
                ),
                index,
                message,
            )
            for index, (
                message,
                current_source,
            ) in enumerate(
                zip(
                    assembly.messages,
                    assembly.message_sources,
                )
            )
            if (
                current_source
                == source
                and not isinstance(
                    message,
                    HumanMessage,
                )
            )
        ]

        if not candidates:
            return (
                assembly,
                [],
            )

        (
            before_tokens,
            index,
            message,
        ) = max(
            candidates,
            key=lambda item: item[0],
        )

        if (
            before_tokens
            <= minimum_tokens
        ):
            return (
                assembly,
                [],
            )

        target_tokens = max(
            minimum_tokens,
            before_tokens
            - report.overflow_tokens
            - 16,
        )

        original_text = (
            message_content_to_text(
                message.content
            )
        )

        new_text = (
            _truncate_text_to_token_target(
                text=original_text,
                target_tokens=(
                    target_tokens
                ),
                marker=marker,
            )
        )

        new_message = (
            _copy_message_with_content(
                message,
                new_text,
            )
        )

        after_tokens = (
            estimate_message_tokens(
                new_message
            )
        )

        if (
            after_tokens
            >= before_tokens
        ):
            return (
                assembly,
                [],
            )

        messages = list(
            assembly.messages
        )

        messages[
            index
        ] = new_message

        rebuilt = _rebuild_assembly(
            original=assembly,
            messages=messages,
            sources=list(
                assembly.message_sources
            ),
            refs=list(
                assembly.message_context_refs
            ),
        )

        return (
            rebuilt,
            [
                ContextCompressionAction(
                    action=(
                        "truncate_context_view"
                    ),
                    source=source,
                    message_id=(
                        getattr(
                            message,
                            "id",
                            None,
                        )
                    ),
                    context_ref=(
                        assembly
                        .message_context_refs[
                            index
                        ]
                    ),
                    before_tokens=(
                        before_tokens
                    ),
                    after_tokens=(
                        after_tokens
                    ),
                    detail=detail,
                )
            ],
        )