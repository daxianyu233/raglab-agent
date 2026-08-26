"""RAGLab Conversation Retriever.

职责：
1. 接收 ContextPlan；
2. 根据 history_scope 选择需要的历史 turn；
3. 从 Conversation Event Store 恢复结构化历史事件；
4. 根据 previous_answer_required / raw_tool_evidence_required
   标记本轮真正需要交给后续 Context Assembler 的事件。

不负责：
- 调用 LLM；
- 判断用户意图；
- 重新执行业务 RAG / SQL；
- 修改 Event Store；
- 直接拼最终模型输入；
- Token Budget / Compression。
"""

from __future__ import annotations

import math
import re

from dataclasses import dataclass
from typing import Iterable

from raglab.agent.context_plan import (
    ContextPlan,
)
from raglab.agent.conversation_event_store import (
    ConversationEvent,
    ConversationEventStore,
)


@dataclass(frozen=True)
class RetrievedConversationTurn:
    """Conversation Retriever 返回的一轮历史。"""

    turn_id: str

    # 原始完整事件，保持 Event Store 中顺序。
    events: list[ConversationEvent]

    # 根据 ContextPlan 选中的事件。
    selected_events: list[ConversationEvent]

    retrieval_score: float
    matched_terms: list[str]


@dataclass(frozen=True)
class ConversationRetrievalResult:
    """一次历史上下文检索结果。"""

    thread_id: str
    history_scope: str
    history_query: str | None

    selected_turn_ids: list[str]
    turns: list[RetrievedConversationTurn]

    selected_event_count: int

    # 便于 Context Audit / Evaluation。
    candidate_turn_count: int
    retrieval_strategy: str


def _event_has_tool_calls(
    event: ConversationEvent,
) -> bool:
    if event.role != "assistant":
        return False

    tool_calls = event.payload.get(
        "tool_calls",
        [],
    )

    return bool(
        isinstance(
            tool_calls,
            list,
        )
        and tool_calls
    )


def _is_final_assistant_event(
    event: ConversationEvent,
) -> bool:
    return (
        event.role == "assistant"
        and not _event_has_tool_calls(
            event
        )
    )


def _select_events_for_plan(
    events: list[ConversationEvent],
    plan: ContextPlan,
) -> list[ConversationEvent]:
    """根据 ContextPlan 选择后续 Context Assembler 需要的事件。

    原则：
    1. Human 作为该历史轮次的语义锚点，始终保留；
    2. previous_answer_required=true 时保留最终 AI Answer；
    3. raw_tool_evidence_required=true 时保留
       AI tool_call + ToolMessage，确保 Tool Pair 可恢复；
    4. 如果历史被要求，但两个 evidence flag 都是 false，
       默认保留 Human + 最终 AI Answer，支持一般性“继续/展开”。
    """

    selected: list[
        ConversationEvent
    ] = []

    include_final_answer = (
        plan.previous_answer_required
        or not (
            plan.raw_tool_evidence_required
        )
    )

    for event in events:
        if event.role == "human":
            selected.append(
                event
            )
            continue

        if (
            plan.raw_tool_evidence_required
            and (
                event.role == "tool"
                or _event_has_tool_calls(
                    event
                )
            )
        ):
            selected.append(
                event
            )
            continue

        if (
            include_final_answer
            and _is_final_assistant_event(
                event
            )
        ):
            selected.append(
                event
            )

    return selected


def _normalize_search_units(
    text: str,
) -> set[str]:
    """生成轻量 lexical units。

    Phase 5 目标不是建立最终 Conversation RAG，
    只是让 historical_search 比单个 SQL LIKE 更可靠。

    同时支持：
    - 英文/数字 token；
    - 中文连续字符串；
    - 中文 2-gram。

    后续可以替换为 BM25 + Dense + RRF。
    """

    normalized = str(
        text
        or ""
    ).strip().lower()

    if not normalized:
        return set()

    units: set[str] = set()

    # 英文、数字、下划线、连字符等。
    for token in re.findall(
        r"[a-z0-9][a-z0-9_\-\.]*",
        normalized,
    ):
        if len(token) >= 2:
            units.add(
                token
            )

    # 中文连续片段。
    cjk_sequences = re.findall(
        r"[\u4e00-\u9fff]+",
        normalized,
    )

    for sequence in cjk_sequences:
        if len(sequence) <= 4:
            units.add(
                sequence
            )

        # 加入 2-gram，避免 query 和正文分词形式不一致。
        if len(sequence) >= 2:
            for index in range(
                len(sequence) - 1
            ):
                units.add(
                    sequence[
                        index:
                        index + 2
                    ]
                )

    return units


def _turn_search_text(
    events: Iterable[
        ConversationEvent
    ],
) -> str:
    parts: list[str] = []

    for event in events:
        if event.content_text:
            parts.append(
                event.content_text
            )

        if event.tool_name:
            parts.append(
                event.tool_name
            )

    return "\n".join(
        parts
    )


def _score_turn(
    *,
    query: str,
    events: list[ConversationEvent],
) -> tuple[
    float,
    list[str],
]:
    query_units = (
        _normalize_search_units(
            query
        )
    )

    if not query_units:
        return (
            0.0,
            [],
        )

    text_units = (
        _normalize_search_units(
            _turn_search_text(
                events
            )
        )
    )

    matched = sorted(
        query_units
        & text_units
    )

    if not matched:
        return (
            0.0,
            [],
        )

    # 类似轻量 IDF-free lexical coverage：
    # 更看重 query 覆盖率，而不是长 Tool 文本长度。
    coverage = (
        len(
            matched
        )
        / len(
            query_units
        )
    )

    # 防止极少 query units 时所有结果完全同分。
    match_bonus = math.log1p(
        len(
            matched
        )
    ) * 0.05

    score = (
        coverage
        + match_bonus
    )

    return (
        float(
            score
        ),
        matched,
    )


class ConversationRetriever:
    """从 Conversation Event Store 按 ContextPlan 读取历史。"""

    def __init__(
        self,
        *,
        store: ConversationEventStore,
        recent_turn_limit: int = 3,
        historical_turn_limit: int = 3,
    ) -> None:
        self.store = store

        self.recent_turn_limit = int(
            recent_turn_limit
        )

        self.historical_turn_limit = int(
            historical_turn_limit
        )

        if self.recent_turn_limit <= 0:
            raise ValueError(
                "recent_turn_limit 必须大于 0。"
            )

        if self.historical_turn_limit <= 0:
            raise ValueError(
                "historical_turn_limit 必须大于 0。"
            )

    def retrieve(
        self,
        *,
        thread_id: str,
        plan: ContextPlan,
        current_turn_id: str | None = None,
    ) -> ConversationRetrievalResult:
        """根据 ContextPlan 检索当前线程历史。

        current_turn_id:
            如果当前 Human 已经提前写入 Event Store，
            用它排除当前轮；
            如果当前 Human 尚未归档，则传 None，
            此时 Event Store 最后一个 turn 就是上一轮。
        """

        normalized_thread_id = str(
            thread_id
        ).strip()

        if not normalized_thread_id:
            raise ValueError(
                "thread_id 不能为空。"
            )

        if (
            not plan.history_required
            or plan.history_scope == "none"
        ):
            return ConversationRetrievalResult(
                thread_id=(
                    normalized_thread_id
                ),
                history_scope="none",
                history_query=None,
                selected_turn_ids=[],
                turns=[],
                selected_event_count=0,
                candidate_turn_count=0,
                retrieval_strategy="none",
            )

        all_turn_ids = (
            self.store.list_turn_ids(
                thread_id=(
                    normalized_thread_id
                )
            )
        )

        prior_turn_ids = (
            self._prior_turn_ids(
                all_turn_ids=(
                    all_turn_ids
                ),
                current_turn_id=(
                    current_turn_id
                ),
            )
        )

        if plan.history_scope == "previous_turn":
            selected_ids = (
                prior_turn_ids[-1:]
            )

            turns = (
                self._load_turns(
                    thread_id=(
                        normalized_thread_id
                    ),
                    turn_ids=selected_ids,
                    plan=plan,
                )
            )

            return self._build_result(
                thread_id=(
                    normalized_thread_id
                ),
                plan=plan,
                turns=turns,
                candidate_turn_count=len(
                    prior_turn_ids
                ),
                retrieval_strategy=(
                    "previous_turn"
                ),
            )

        if plan.history_scope == "recent_turns":
            selected_ids = (
                prior_turn_ids[
                    -self.recent_turn_limit:
                ]
            )

            turns = (
                self._load_turns(
                    thread_id=(
                        normalized_thread_id
                    ),
                    turn_ids=selected_ids,
                    plan=plan,
                )
            )

            return self._build_result(
                thread_id=(
                    normalized_thread_id
                ),
                plan=plan,
                turns=turns,
                candidate_turn_count=len(
                    prior_turn_ids
                ),
                retrieval_strategy=(
                    "recent_turns"
                ),
            )

        if plan.history_scope == "historical_search":
            history_query = str(
                plan.history_query
                or ""
            ).strip()

            if not history_query:
                raise ValueError(
                    "historical_search 缺少 history_query。"
                )

            ranked_turns: list[
                RetrievedConversationTurn
            ] = []

            for turn_id in prior_turn_ids:
                events = (
                    self.store
                    .list_turn_events(
                        thread_id=(
                            normalized_thread_id
                        ),
                        turn_id=turn_id,
                    )
                )

                score, matched_terms = (
                    _score_turn(
                        query=(
                            history_query
                        ),
                        events=events,
                    )
                )

                if score <= 0.0:
                    continue

                ranked_turns.append(
                    RetrievedConversationTurn(
                        turn_id=turn_id,
                        events=events,
                        selected_events=(
                            _select_events_for_plan(
                                events,
                                plan,
                            )
                        ),
                        retrieval_score=(
                            score
                        ),
                        matched_terms=(
                            matched_terms
                        ),
                    )
                )

            ranked_turns.sort(
                key=lambda item: (
                    item.retrieval_score
                ),
                reverse=True,
            )

            turns = ranked_turns[
                :self.historical_turn_limit
            ]

            return self._build_result(
                thread_id=(
                    normalized_thread_id
                ),
                plan=plan,
                turns=turns,
                candidate_turn_count=len(
                    prior_turn_ids
                ),
                retrieval_strategy=(
                    "historical_lexical"
                ),
            )

        raise ValueError(
            "不支持的 history_scope："
            f"{plan.history_scope!r}"
        )

    @staticmethod
    def _prior_turn_ids(
        *,
        all_turn_ids: list[str],
        current_turn_id: str | None,
    ) -> list[str]:
        if not current_turn_id:
            return list(
                all_turn_ids
            )

        normalized_current = str(
            current_turn_id
        ).strip()

        if not normalized_current:
            return list(
                all_turn_ids
            )

        # 当前轮已经在 Event Store：
        # 只允许读取它之前的 turns。
        if normalized_current in all_turn_ids:
            index = all_turn_ids.index(
                normalized_current
            )

            return all_turn_ids[
                :index
            ]

        # 当前轮尚未入库：
        # 已有全部 turns 都是历史。
        return list(
            all_turn_ids
        )

    def _load_turns(
        self,
        *,
        thread_id: str,
        turn_ids: list[str],
        plan: ContextPlan,
    ) -> list[
        RetrievedConversationTurn
    ]:
        turns: list[
            RetrievedConversationTurn
        ] = []

        for turn_id in turn_ids:
            events = (
                self.store
                .list_turn_events(
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            )

            turns.append(
                RetrievedConversationTurn(
                    turn_id=turn_id,
                    events=events,
                    selected_events=(
                        _select_events_for_plan(
                            events,
                            plan,
                        )
                    ),
                    retrieval_score=1.0,
                    matched_terms=[],
                )
            )

        return turns

    @staticmethod
    def _build_result(
        *,
        thread_id: str,
        plan: ContextPlan,
        turns: list[
            RetrievedConversationTurn
        ],
        candidate_turn_count: int,
        retrieval_strategy: str,
    ) -> ConversationRetrievalResult:
        return ConversationRetrievalResult(
            thread_id=thread_id,
            history_scope=(
                plan.history_scope
            ),
            history_query=(
                plan.history_query
            ),
            selected_turn_ids=[
                turn.turn_id
                for turn in turns
            ],
            turns=turns,
            selected_event_count=sum(
                len(
                    turn.selected_events
                )
                for turn in turns
            ),
            candidate_turn_count=(
                candidate_turn_count
            ),
            retrieval_strategy=(
                retrieval_strategy
            ),
        )