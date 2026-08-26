"""Phase 7E-3: Conversation Archive reconciliation / self-healing.

The normal post-run archive remains the primary write path.

This module adds a repair path:
before a new Human Turn starts, replay the previous checkpoint messages
through the idempotent Conversation Event adapter.

Existing events are skipped by the Event Store's idempotency rules;
missing events are backfilled.

A reconciliation failure must NOT make the whole Agent unavailable.
Working Memory compaction remains fail-closed and will pin any still
unverified Turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from raglab.agent.conversation_event_adapter import (
    ConversationArchiveReport,
    archive_messages_to_event_store,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.long_term_memory_agent import (
    normalize_user_id,
)
from raglab.agent.persistent_langgraph_agent import (
    normalize_thread_id,
)


@dataclass(frozen=True)
class ArchiveReconciliationResult:
    """One pre-run reconciliation attempt."""

    attempted: bool
    success: bool
    reason: str

    user_id: str
    thread_id: str

    checkpoint_message_count: int

    report: (
        ConversationArchiveReport
        | None
    )

    error_type: str = ""
    error_message: str = ""

    def to_diagnostics(
        self,
    ) -> dict[str, Any]:
        return {
            "attempted": (
                self.attempted
            ),
            "success": (
                self.success
            ),
            "reason": (
                self.reason
            ),
            "user_id": (
                self.user_id
            ),
            "thread_id": (
                self.thread_id
            ),
            "checkpoint_message_count": (
                self
                .checkpoint_message_count
            ),
            "report": (
                repr(
                    self.report
                )
                if self.report
                is not None
                else None
            ),
            "error_type": (
                self.error_type
            ),
            "error_message": (
                self.error_message
            ),
        }


def reconcile_checkpoint_archive(
    *,
    base_agent: Any,
    store: ConversationEventStore,
    user_id: str,
    thread_id: str,
) -> ArchiveReconciliationResult:
    """Idempotently backfill Event Store from checkpoint messages.

    This is deliberately fail-open for normal conversation availability:
    an exception is returned as diagnostics instead of being raised.

    Data deletion remains protected elsewhere:
    Working Memory compaction still requires a successful per-Turn
    archive verification before removing raw checkpoint messages.
    """

    normalized_user_id = (
        normalize_user_id(
            user_id
        )
    )

    normalized_thread_id = (
        normalize_thread_id(
            thread_id
        )
    )

    get_thread_messages = getattr(
        base_agent,
        "get_thread_messages",
        None,
    )

    if not callable(
        get_thread_messages
    ):
        return (
            ArchiveReconciliationResult(
                attempted=False,
                success=False,
                reason=(
                    "base_agent_has_no_"
                    "get_thread_messages"
                ),
                user_id=(
                    normalized_user_id
                ),
                thread_id=(
                    normalized_thread_id
                ),
                checkpoint_message_count=0,
                report=None,
            )
        )

    try:
        messages = list(
            get_thread_messages(
                normalized_thread_id
            )
            or []
        )
    except Exception as exc:
        return (
            ArchiveReconciliationResult(
                attempted=True,
                success=False,
                reason=(
                    "checkpoint_read_failed"
                ),
                user_id=(
                    normalized_user_id
                ),
                thread_id=(
                    normalized_thread_id
                ),
                checkpoint_message_count=0,
                report=None,
                error_type=(
                    type(exc).__name__
                ),
                error_message=str(
                    exc
                ),
            )
        )

    if not messages:
        return (
            ArchiveReconciliationResult(
                attempted=False,
                success=True,
                reason=(
                    "no_checkpoint_messages"
                ),
                user_id=(
                    normalized_user_id
                ),
                thread_id=(
                    normalized_thread_id
                ),
                checkpoint_message_count=0,
                report=None,
            )
        )

    try:
        report = (
            archive_messages_to_event_store(
                store=store,
                user_id=(
                    normalized_user_id
                ),
                thread_id=(
                    normalized_thread_id
                ),
                messages=messages,
            )
        )
    except Exception as exc:
        return (
            ArchiveReconciliationResult(
                attempted=True,
                success=False,
                reason=(
                    "archive_backfill_failed"
                ),
                user_id=(
                    normalized_user_id
                ),
                thread_id=(
                    normalized_thread_id
                ),
                checkpoint_message_count=len(
                    messages
                ),
                report=None,
                error_type=(
                    type(exc).__name__
                ),
                error_message=str(
                    exc
                ),
            )
        )

    return (
        ArchiveReconciliationResult(
            attempted=True,
            success=True,
            reason=(
                "checkpoint_replayed_"
                "idempotently"
            ),
            user_id=(
                normalized_user_id
            ),
            thread_id=(
                normalized_thread_id
            ),
            checkpoint_message_count=len(
                messages
            ),
            report=report,
        )
    )