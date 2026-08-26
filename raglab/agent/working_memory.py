"""Phase 7E-2: Token-aware Working Memory management.

Responsibilities
----------------
1. Audit checkpoint Working Memory by estimated tokens.
2. Trigger compaction by token footprint, not fixed turn count.
3. Compact only completed historical Turns.
4. Fail closed: a Turn is removable only after it is confirmed archived
   in Conversation Event Store.
5. Never remove the current Turn or unresolved Tool Call protocol state.
6. Return a deterministic compaction plan. The Agent's memory_manager
   updates the Thread Summary before applying message deletion.

Conversation Event Store remains the full raw-history source of truth.
Thread Summary remains the compact "where are we now?" map.
Checkpoint messages remain recent raw Working Memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from raglab.agent.context_manager import (
    estimate_message_tokens,
    estimate_text_tokens,
)


@dataclass(frozen=True)
class WorkingMemoryAuditConfig:
    """Token-aware Working Memory thresholds.

    soft_limit_tokens
        Compaction starts only when checkpoint messages reach/exceed this
        threshold.

    target_tokens
        Once compaction starts, remove oldest safe historical Turns until
        checkpoint messages are near/below this target, or until no more
        safe candidates remain.

    oversized_tool_threshold_tokens
        Diagnostic threshold only.

    minimum_recent_turns
        Safety preference. Keep at least this many newest Turns in the
        checkpoint, including the current Turn. This is NOT the old
        fixed "keep exactly N turns" behavior; it is only a minimum floor.
    """

    soft_limit_tokens: int = 12000
    target_tokens: int = 8000
    oversized_tool_threshold_tokens: int = 4000
    minimum_recent_turns: int = 1

    def __post_init__(self) -> None:
        if self.soft_limit_tokens <= 0:
            raise ValueError(
                "soft_limit_tokens 必须大于 0。"
            )

        if self.target_tokens <= 0:
            raise ValueError(
                "target_tokens 必须大于 0。"
            )

        if self.target_tokens >= self.soft_limit_tokens:
            raise ValueError(
                "target_tokens 必须小于 soft_limit_tokens。"
            )

        if self.oversized_tool_threshold_tokens <= 0:
            raise ValueError(
                "oversized_tool_threshold_tokens 必须大于 0。"
            )

        if self.minimum_recent_turns <= 0:
            raise ValueError(
                "minimum_recent_turns 必须大于 0。"
            )


@dataclass(frozen=True)
class WorkingMemoryCompactionPlan:
    """Deterministic plan produced before any checkpoint mutation."""

    should_compact: bool
    can_compact: bool
    reason: str

    before_tokens: int
    target_tokens: int
    predicted_after_tokens: int
    predicted_target_reached: bool

    selected_turn_keys: tuple[str, ...]
    selected_turn_count: int
    selected_message_count: int
    selected_estimated_tokens: int

    retained_turn_keys: tuple[str, ...]
    retained_turn_count: int
    retained_message_count: int

    archive_verified_turn_keys: tuple[str, ...]
    archive_rejected_turn_keys: tuple[str, ...]

    # Historical Turns that could not be safely removed.
    # They remain raw in Checkpoint and will be retried by
    # Phase 7E-3 reconciliation on later Human Turns.
    pinned_turn_keys: tuple[str, ...]
    pinned_turn_count: int

    messages_to_summarize: tuple[BaseMessage, ...]
    retained_messages: tuple[BaseMessage, ...]

    audit_before: dict[str, Any]

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "should_compact": self.should_compact,
            "can_compact": self.can_compact,
            "reason": self.reason,
            "before_tokens": self.before_tokens,
            "target_tokens": self.target_tokens,
            "predicted_after_tokens": (
                self.predicted_after_tokens
            ),
            "predicted_target_reached": (
                self.predicted_target_reached
            ),
            "selected_turn_keys": list(
                self.selected_turn_keys
            ),
            "selected_turn_count": (
                self.selected_turn_count
            ),
            "selected_message_count": (
                self.selected_message_count
            ),
            "selected_estimated_tokens": (
                self.selected_estimated_tokens
            ),
            "retained_turn_keys": list(
                self.retained_turn_keys
            ),
            "retained_turn_count": (
                self.retained_turn_count
            ),
            "retained_message_count": (
                self.retained_message_count
            ),
            "archive_verified_turn_keys": list(
                self.archive_verified_turn_keys
            ),
            "archive_rejected_turn_keys": list(
                self.archive_rejected_turn_keys
            ),
            "pinned_turn_keys": list(
                self.pinned_turn_keys
            ),
            "pinned_turn_count": (
                self.pinned_turn_count
            ),
        }


def _message_role(
    message: BaseMessage,
) -> str:
    if isinstance(message, HumanMessage):
        return "human"

    if isinstance(message, AIMessage):
        return "ai"

    if isinstance(message, ToolMessage):
        return "tool"

    if isinstance(message, SystemMessage):
        return "system"

    return type(message).__name__.lower()


def _message_id(
    message: BaseMessage,
    index: int,
) -> str:
    value = str(
        getattr(
            message,
            "id",
            "",
        )
        or ""
    ).strip()

    return value or f"message-index:{index}"


def _tool_call_id(
    call: Any,
) -> str:
    if isinstance(call, dict):
        return str(
            call.get(
                "id",
                "",
            )
            or ""
        ).strip()

    return str(
        getattr(
            call,
            "id",
            "",
        )
        or ""
    ).strip()


def _ai_tool_calls(
    message: AIMessage,
) -> list[Any]:
    raw = getattr(
        message,
        "tool_calls",
        None,
    )

    if raw:
        return list(raw)

    additional = getattr(
        message,
        "additional_kwargs",
        {},
    )

    if isinstance(
        additional,
        dict,
    ):
        raw = additional.get(
            "tool_calls",
            [],
        )

        if raw:
            return list(raw)

    return []


def _audit_turn(
    indexed_messages: Sequence[
        tuple[
            int,
            BaseMessage,
        ]
    ],
    *,
    ordinal: int,
) -> dict[str, Any]:

    unresolved: set[str] = set()
    tool_message_ids: set[str] = set()

    estimated_tokens = 0
    message_ids: list[str] = []

    for index, message in (
        indexed_messages
    ):
        estimated_tokens += (
            estimate_message_tokens(
                message
            )
        )

        message_ids.append(
            _message_id(
                message,
                index,
            )
        )

        if isinstance(
            message,
            AIMessage,
        ):
            for call in (
                _ai_tool_calls(
                    message
                )
            ):
                call_id = (
                    _tool_call_id(
                        call
                    )
                )

                if call_id:
                    unresolved.add(
                        call_id
                    )

        if isinstance(
            message,
            ToolMessage,
        ):
            call_id = str(
                getattr(
                    message,
                    "tool_call_id",
                    "",
                )
                or ""
            ).strip()

            if call_id:
                tool_message_ids.add(
                    call_id
                )

    unresolved -= (
        tool_message_ids
    )

    final_message = (
        indexed_messages[
            -1
        ][1]
    )

    is_complete = bool(
        isinstance(
            final_message,
            AIMessage,
        )
        and not (
            _ai_tool_calls(
                final_message
            )
        )
        and not unresolved
    )

    first_index = (
        indexed_messages[
            0
        ][0]
    )

    first_message = (
        indexed_messages[
            0
        ][1]
    )

    return {
        "turn_ordinal": ordinal,
        "turn_key": (
            "turn:"
            + _message_id(
                first_message,
                first_index,
            )
        ),
        "start_message_index": (
            first_index
        ),
        "end_message_index": (
            indexed_messages[
                -1
            ][0]
        ),
        "message_count": len(
            indexed_messages
        ),
        "message_ids": (
            message_ids
        ),
        "estimated_tokens": (
            estimated_tokens
        ),
        "is_complete": (
            is_complete
        ),
        "is_current_turn": False,
        "unresolved_tool_call_ids": (
            sorted(
                unresolved
            )
        ),
    }


def _segment_turns(
    messages: Sequence[
        BaseMessage
    ],
) -> list[dict[str, Any]]:
    """Split checkpoint messages by HumanMessage boundaries."""

    turns: list[
        dict[str, Any]
    ] = []

    current: list[
        tuple[
            int,
            BaseMessage,
        ]
    ] = []

    for index, message in enumerate(
        messages
    ):
        if isinstance(
            message,
            HumanMessage,
        ):
            if current:
                turns.append(
                    _audit_turn(
                        current,
                        ordinal=(
                            len(turns)
                            + 1
                        ),
                    )
                )

            current = [
                (
                    index,
                    message,
                )
            ]

            continue

        if current:
            current.append(
                (
                    index,
                    message,
                )
            )

    if current:
        turns.append(
            _audit_turn(
                current,
                ordinal=(
                    len(turns)
                    + 1
                ),
            )
        )

    if turns:
        turns[-1][
            "is_current_turn"
        ] = True

    return turns


def audit_working_memory(
    messages: Sequence[
        BaseMessage
    ],
    *,
    summary: str = "",
    config: (
        WorkingMemoryAuditConfig
        | None
    ) = None,
    legacy_keep_recent_turns: (
        int
        | None
    ) = None,
    legacy_summarize_trigger_turns: (
        int
        | None
    ) = None,
) -> dict[str, Any]:
    """Audit checkpoint Working Memory without mutating it."""

    current_config = (
        config
        or WorkingMemoryAuditConfig()
    )

    message_list = list(
        messages
    )

    role_tokens: dict[
        str,
        int
    ] = {}

    role_counts: dict[
        str,
        int
    ] = {}

    message_details: list[
        dict[str, Any]
    ] = []

    oversized_tool_messages: list[
        dict[str, Any]
    ] = []

    unresolved_tool_call_ids: set[
        str
    ] = set()

    seen_tool_message_ids: set[
        str
    ] = set()

    total_message_tokens = 0

    largest_message_index: (
        int
        | None
    ) = None

    largest_message_tokens = 0
    largest_message_role = ""

    for index, message in enumerate(
        message_list
    ):
        role = _message_role(
            message
        )

        tokens = (
            estimate_message_tokens(
                message
            )
        )

        total_message_tokens += (
            tokens
        )

        role_tokens[role] = (
            role_tokens.get(
                role,
                0,
            )
            + tokens
        )

        role_counts[role] = (
            role_counts.get(
                role,
                0,
            )
            + 1
        )

        message_details.append(
            {
                "index": index,
                "role": role,
                "message_id": (
                    _message_id(
                        message,
                        index,
                    )
                ),
                "estimated_tokens": (
                    tokens
                ),
            }
        )

        if tokens > (
            largest_message_tokens
        ):
            largest_message_index = (
                index
            )
            largest_message_tokens = (
                tokens
            )
            largest_message_role = (
                role
            )

        if isinstance(
            message,
            AIMessage,
        ):
            for call in (
                _ai_tool_calls(
                    message
                )
            ):
                call_id = (
                    _tool_call_id(
                        call
                    )
                )

                if call_id:
                    unresolved_tool_call_ids.add(
                        call_id
                    )

        if isinstance(
            message,
            ToolMessage,
        ):
            call_id = str(
                getattr(
                    message,
                    "tool_call_id",
                    "",
                )
                or ""
            ).strip()

            if call_id:
                seen_tool_message_ids.add(
                    call_id
                )

            if tokens >= (
                current_config
                .oversized_tool_threshold_tokens
            ):
                oversized_tool_messages.append(
                    {
                        "index": index,
                        "message_id": (
                            _message_id(
                                message,
                                index,
                            )
                        ),
                        "tool_call_id": (
                            call_id
                        ),
                        "name": str(
                            getattr(
                                message,
                                "name",
                                "",
                            )
                            or ""
                        ),
                        "estimated_tokens": (
                            tokens
                        ),
                    }
                )

    unresolved_tool_call_ids -= (
        seen_tool_message_ids
    )

    turns = _segment_turns(
        message_list
    )

    completed_historical_turns = [
        current
        for current
        in turns
        if (
            current.get(
                "is_complete",
                False,
            )
            and not current.get(
                "is_current_turn",
                False,
            )
        )
    ]

    summary_tokens = (
        estimate_text_tokens(
            str(
                summary
                or ""
            )
        )
    )

    token_trigger = bool(
        total_message_tokens
        >= current_config
        .soft_limit_tokens
    )

    legacy_trigger = bool(
        legacy_summarize_trigger_turns
        is not None
        and len(turns)
        >= int(
            legacy_summarize_trigger_turns
        )
    )

    return {
        "config": asdict(
            current_config
        ),
        "message_count": len(
            message_list
        ),
        "turn_count": len(
            turns
        ),
        "summary_estimated_tokens": (
            summary_tokens
        ),
        "checkpoint_message_estimated_tokens": (
            total_message_tokens
        ),
        "checkpoint_plus_summary_estimated_tokens": (
            total_message_tokens
            + summary_tokens
        ),
        "message_counts_by_role": (
            role_counts
        ),
        "estimated_tokens_by_role": (
            role_tokens
        ),
        "largest_message_index": (
            largest_message_index
        ),
        "largest_message_role": (
            largest_message_role
        ),
        "largest_message_tokens": (
            largest_message_tokens
        ),
        "oversized_tool_message_count": len(
            oversized_tool_messages
        ),
        "oversized_tool_messages": (
            oversized_tool_messages
        ),
        "unresolved_tool_call_ids": (
            sorted(
                unresolved_tool_call_ids
            )
        ),
        "unresolved_tool_call_count": len(
            unresolved_tool_call_ids
        ),
        "tool_pair_integrity_ok": (
            not unresolved_tool_call_ids
        ),
        "turns": turns,
        "completed_historical_turn_count": len(
            completed_historical_turns
        ),
        "completed_historical_turn_keys": [
            str(
                current.get(
                    "turn_key",
                    "",
                )
            )
            for current
            in completed_historical_turns
        ],
        "legacy_keep_recent_turns": (
            legacy_keep_recent_turns
        ),
        "legacy_summarize_trigger_turns": (
            legacy_summarize_trigger_turns
        ),
        "legacy_turn_trigger_would_fire": (
            legacy_trigger
        ),
        "token_compaction_recommended": (
            token_trigger
        ),
        "estimated_tokens_to_remove_to_target": max(
            0,
            total_message_tokens
            - current_config
            .target_tokens,
        ),
        "phase7e_action": (
            "token_aware_compaction"
        ),
        "message_details": (
            message_details
        ),
    }


def _turn_archived_completely(
    *,
    event_store: Any,
    thread_id: str,
    turn: dict[str, Any],
) -> bool:
    """Require every checkpoint message id in the Turn to exist in Event Store."""

    if event_store is None:
        return False

    normalized_thread_id = str(
        thread_id
        or ""
    ).strip()

    turn_key = str(
        turn.get(
            "turn_key",
            "",
        )
        or ""
    ).strip()

    if (
        not normalized_thread_id
        or not turn_key
    ):
        return False

    list_turn_events = getattr(
        event_store,
        "list_turn_events",
        None,
    )

    if not callable(
        list_turn_events
    ):
        return False

    try:
        archived_events = list(
            list_turn_events(
                thread_id=(
                    normalized_thread_id
                ),
                turn_id=(
                    turn_key
                ),
            )
            or []
        )
    except Exception:
        return False

    if not archived_events:
        return False

    expected_ids = {
        str(
            current
            or ""
        ).strip()
        for current
        in (
            turn.get(
                "message_ids",
                [],
            )
            or []
        )
        if str(
            current
            or ""
        ).strip()
    }

    if not expected_ids:
        return False

    archived_ids = {
        str(
            getattr(
                event,
                "message_id",
                "",
            )
            or ""
        ).strip()
        for event
        in archived_events
        if str(
            getattr(
                event,
                "message_id",
                "",
            )
            or ""
        ).strip()
    }

    return (
        expected_ids
        <= archived_ids
    )


def plan_working_memory_compaction(
    messages: Sequence[
        BaseMessage
    ],
    *,
    summary: str = "",
    config: (
        WorkingMemoryAuditConfig
        | None
    ) = None,
    event_store: Any = None,
    thread_id: str = "",
    legacy_keep_recent_turns: (
        int
        | None
    ) = None,
    legacy_summarize_trigger_turns: (
        int
        | None
    ) = None,
) -> WorkingMemoryCompactionPlan:
    """Build a safe oldest-first compaction plan.

    The planner is deterministic. It does not call an LLM and does not
    mutate checkpoint state.
    """

    current_config = (
        config
        or WorkingMemoryAuditConfig()
    )

    message_list = list(
        messages
    )

    audit = audit_working_memory(
        message_list,
        summary=summary,
        config=current_config,
        legacy_keep_recent_turns=(
            legacy_keep_recent_turns
        ),
        legacy_summarize_trigger_turns=(
            legacy_summarize_trigger_turns
        ),
    )

    before_tokens = int(
        audit[
            "checkpoint_message_estimated_tokens"
        ]
    )

    turns = list(
        audit.get(
            "turns",
            [],
        )
        or []
    )

    all_turn_keys = tuple(
        str(
            current.get(
                "turn_key",
                "",
            )
            or ""
        )
        for current
        in turns
    )

    if not bool(
        audit[
            "token_compaction_recommended"
        ]
    ):
        return WorkingMemoryCompactionPlan(
            should_compact=False,
            can_compact=False,
            reason="below_soft_limit",
            before_tokens=before_tokens,
            target_tokens=(
                current_config
                .target_tokens
            ),
            predicted_after_tokens=(
                before_tokens
            ),
            predicted_target_reached=(
                before_tokens
                <= current_config
                .target_tokens
            ),
            selected_turn_keys=(),
            selected_turn_count=0,
            selected_message_count=0,
            selected_estimated_tokens=0,
            retained_turn_keys=(
                all_turn_keys
            ),
            retained_turn_count=len(
                turns
            ),
            retained_message_count=len(
                message_list
            ),
            archive_verified_turn_keys=(),
            archive_rejected_turn_keys=(),
            pinned_turn_keys=(),
            pinned_turn_count=0,
            messages_to_summarize=(),
            retained_messages=tuple(
                message_list
            ),
            audit_before=audit,
        )

    if len(turns) <= (
        current_config
        .minimum_recent_turns
    ):
        return WorkingMemoryCompactionPlan(
            should_compact=True,
            can_compact=False,
            reason=(
                "only_protected_recent_turns_available"
            ),
            before_tokens=before_tokens,
            target_tokens=(
                current_config
                .target_tokens
            ),
            predicted_after_tokens=(
                before_tokens
            ),
            predicted_target_reached=False,
            selected_turn_keys=(),
            selected_turn_count=0,
            selected_message_count=0,
            selected_estimated_tokens=0,
            retained_turn_keys=(
                all_turn_keys
            ),
            retained_turn_count=len(
                turns
            ),
            retained_message_count=len(
                message_list
            ),
            archive_verified_turn_keys=(),
            archive_rejected_turn_keys=(),
            pinned_turn_keys=(),
            pinned_turn_count=0,
            messages_to_summarize=(),
            retained_messages=tuple(
                message_list
            ),
            audit_before=audit,
        )

    removable_end_index = max(
        0,
        len(turns)
        - current_config
        .minimum_recent_turns,
    )

    selected: list[
        dict[str, Any]
    ] = []

    archive_verified: list[str] = []
    archive_rejected: list[str] = []
    pinned_turns: list[str] = []

    predicted_tokens = (
        before_tokens
    )

    for turn in (
        turns[
            :removable_end_index
        ]
    ):
        if (
            predicted_tokens
            <= current_config
            .target_tokens
        ):
            break

        turn_key = str(
            turn.get(
                "turn_key",
                "",
            )
            or ""
        )

        if not bool(
            turn.get(
                "is_complete",
                False,
            )
        ):
            # 不能删除，但也不能让一个坏 Turn
            # 永久阻塞所有更晚的安全历史。
            #
            # Pin:
            # - 原文继续留在 Checkpoint；
            # - 本轮不参与 Summary/RemoveMessage；
            # - 后续 reconciliation 会再次尝试恢复。
            pinned_turns.append(
                turn_key
            )
            continue

        archived = (
            _turn_archived_completely(
                event_store=(
                    event_store
                ),
                thread_id=(
                    thread_id
                ),
                turn=turn,
            )
        )

        if not archived:
            archive_rejected.append(
                turn_key
            )
            pinned_turns.append(
                turn_key
            )

            # Repair 失败后 Pin 当前 Turn，
            # 继续检查更晚的历史 Turn。
            continue

        archive_verified.append(
            turn_key
        )

        selected.append(
            turn
        )

        predicted_tokens -= int(
            turn.get(
                "estimated_tokens",
                0,
            )
            or 0
        )

    selected_keys = {
        str(
            current.get(
                "turn_key",
                "",
            )
            or ""
        )
        for current
        in selected
    }

    selected_indexes: set[int] = set()

    for turn in selected:
        start_index = int(
            turn[
                "start_message_index"
            ]
        )

        end_index = int(
            turn[
                "end_message_index"
            ]
        )

        selected_indexes.update(
            range(
                start_index,
                end_index + 1,
            )
        )

    messages_to_summarize = tuple(
        message
        for index, message
        in enumerate(
            message_list
        )
        if index in selected_indexes
    )

    retained_messages = tuple(
        message
        for index, message
        in enumerate(
            message_list
        )
        if index not in selected_indexes
    )

    retained_turn_keys = tuple(
        str(
            current.get(
                "turn_key",
                "",
            )
            or ""
        )
        for current
        in turns
        if str(
            current.get(
                "turn_key",
                "",
            )
            or ""
        )
        not in selected_keys
    )

    selected_estimated_tokens = sum(
        int(
            current.get(
                "estimated_tokens",
                0,
            )
            or 0
        )
        for current
        in selected
    )

    if selected and pinned_turns:
        reason = (
            "selected_safe_turns_"
            "with_pinned_holes"
        )
        can_compact = True
    elif selected:
        reason = (
            "selected_archived_historical_turns"
        )
        can_compact = True
    elif pinned_turns:
        reason = (
            "only_pinned_historical_turns_"
            "available"
        )
        can_compact = False
    else:
        reason = (
            "no_safe_historical_turn_candidates"
        )
        can_compact = False

    return WorkingMemoryCompactionPlan(
        should_compact=True,
        can_compact=(
            can_compact
        ),
        reason=reason,
        before_tokens=(
            before_tokens
        ),
        target_tokens=(
            current_config
            .target_tokens
        ),
        predicted_after_tokens=max(
            0,
            predicted_tokens,
        ),
        predicted_target_reached=bool(
            predicted_tokens
            <= current_config
            .target_tokens
        ),
        selected_turn_keys=tuple(
            str(
                current.get(
                    "turn_key",
                    "",
                )
                or ""
            )
            for current
            in selected
        ),
        selected_turn_count=len(
            selected
        ),
        selected_message_count=len(
            messages_to_summarize
        ),
        selected_estimated_tokens=(
            selected_estimated_tokens
        ),
        retained_turn_keys=(
            retained_turn_keys
        ),
        retained_turn_count=len(
            retained_turn_keys
        ),
        retained_message_count=len(
            retained_messages
        ),
        archive_verified_turn_keys=tuple(
            archive_verified
        ),
        archive_rejected_turn_keys=tuple(
            archive_rejected
        ),
        pinned_turn_keys=tuple(
            pinned_turns
        ),
        pinned_turn_count=len(
            pinned_turns
        ),
        messages_to_summarize=(
            messages_to_summarize
        ),
        retained_messages=(
            retained_messages
        ),
        audit_before=audit,
    )