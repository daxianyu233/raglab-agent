"""Phase 7E-3 Archive Reconciliation / Pin-Skip regression.

运行：
    python -m scripts.test_archive_reconciliation
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from raglab.agent.conversation_archive_reconciler import (
    reconcile_checkpoint_archive,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.working_memory import (
    WorkingMemoryAuditConfig,
    plan_working_memory_compaction,
)


CONFIG = WorkingMemoryAuditConfig(
    soft_limit_tokens=500,
    target_tokens=180,
    oversized_tool_threshold_tokens=200,
    minimum_recent_turns=1,
)


class FakeBaseAgent:
    def __init__(
        self,
        messages,
    ) -> None:
        self.messages = list(
            messages
        )

    def get_thread_messages(
        self,
        thread_id: str,
    ):
        return list(
            self.messages
        )


class FailingBaseAgent:
    def get_thread_messages(
        self,
        thread_id: str,
    ):
        raise RuntimeError(
            "simulated checkpoint read failure"
        )


def build_four_turns():
    # Turn 1: small, archived
    # Turn 2: small, intentionally unarchived hole
    # Turn 3: large, archived
    # Turn 4: current, protected
    return [
        HumanMessage(
            content="Turn1",
            id="human-1",
        ),
        AIMessage(
            content="A1",
            id="ai-1",
        ),

        HumanMessage(
            content="Turn2",
            id="human-2",
        ),
        AIMessage(
            content="A2",
            id="ai-2",
        ),

        HumanMessage(
            content=(
                "Turn3 "
                + "X" * 2600
            ),
            id="human-3",
        ),
        AIMessage(
            content="A3",
            id="ai-3",
        ),

        HumanMessage(
            content="Turn4 current",
            id="human-4",
        ),
        AIMessage(
            content="A4",
            id="ai-4",
        ),
    ]


def archive_subset(
    store,
    *,
    user_id: str,
    thread_id: str,
    messages,
):
    from raglab.agent.conversation_event_adapter import (
        archive_messages_to_event_store,
    )

    archive_messages_to_event_store(
        store=store,
        user_id=user_id,
        thread_id=thread_id,
        messages=messages,
    )


def main() -> None:
    print("=" * 80)
    print(
        "Archive Reconciliation Phase 7E-3 回归测试"
    )
    print("=" * 80)

    # ========================================================
    # Case 1: backfill missing history from checkpoint
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 1：Event Store 缺历史 -> "
        "Checkpoint 幂等补归档"
    )
    print("=" * 80)

    with TemporaryDirectory() as temp_dir:
        db_path = (
            Path(temp_dir)
            / "events.sqlite3"
        )

        store = (
            ConversationEventStore(
                database_path=db_path
            )
        )

        messages = [
            HumanMessage(
                content="第一轮",
                id="human-r1",
            ),
            AIMessage(
                content="第一轮回答",
                id="ai-r1",
            ),
        ]

        base = FakeBaseAgent(
            messages
        )

        before = (
            store.list_thread_events(
                thread_id="thread-r"
            )
        )

        result = (
            reconcile_checkpoint_archive(
                base_agent=base,
                store=store,
                user_id="user-r",
                thread_id="thread-r",
            )
        )

        after = (
            store.list_thread_events(
                thread_id="thread-r"
            )
        )

        print(
            "before：",
            len(before),
        )
        print(
            "after：",
            len(after),
        )
        print(
            "success：",
            result.success,
        )
        print(
            "reason：",
            result.reason,
        )

        assert len(before) == 0
        assert len(after) == 2
        assert result.success

        # 再跑一次必须幂等，不重复增长。
        result2 = (
            reconcile_checkpoint_archive(
                base_agent=base,
                store=store,
                user_id="user-r",
                thread_id="thread-r",
            )
        )

        after2 = (
            store.list_thread_events(
                thread_id="thread-r"
            )
        )

        assert len(after2) == 2
        assert result2.success

        print(
            "[PASS] 缺失 Event 自动补齐，重复 repair 不产生重复数据"
        )

        store.close()

    # ========================================================
    # Case 2: repair failure does not crash normal availability
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 2：Reconciliation 自身失败 -> "
        "返回诊断，不抛异常"
    )
    print("=" * 80)

    with TemporaryDirectory() as temp_dir:
        store = (
            ConversationEventStore(
                database_path=(
                    Path(temp_dir)
                    / "events.sqlite3"
                )
            )
        )

        result = (
            reconcile_checkpoint_archive(
                base_agent=(
                    FailingBaseAgent()
                ),
                store=store,
                user_id="user-fail",
                thread_id="thread-fail",
            )
        )

        print(
            "success：",
            result.success,
        )
        print(
            "reason：",
            result.reason,
        )
        print(
            "error：",
            result.error_type,
            result.error_message,
        )

        assert (
            result.success
            is False
        )
        assert (
            result.reason
            == "checkpoint_read_failed"
        )

        print(
            "[PASS] Repair 失败不让整个 Agent 不可用，删除侧仍 Fail Closed"
        )

        store.close()

    # ========================================================
    # Case 3: one hole is pinned, later safe turn can compact
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 3：Turn2 未归档 -> Pin Turn2，"
        "但 Turn3 安全时继续压缩"
    )
    print("=" * 80)

    with TemporaryDirectory() as temp_dir:
        store = (
            ConversationEventStore(
                database_path=(
                    Path(temp_dir)
                    / "events.sqlite3"
                )
            )
        )

        messages = (
            build_four_turns()
        )

        # Archive Turn1 only.
        archive_subset(
            store,
            user_id="user-p",
            thread_id="thread-p",
            messages=messages[0:2],
        )

        # Archive Turn3 separately.
        # Adapter needs a Human boundary; this produces turn:human-3.
        archive_subset(
            store,
            user_id="user-p",
            thread_id="thread-p",
            messages=messages[4:6],
        )

        plan = (
            plan_working_memory_compaction(
                messages,
                summary="",
                config=CONFIG,
                event_store=store,
                thread_id="thread-p",
            )
        )

        print(
            "selected：",
            list(
                plan.selected_turn_keys
            ),
        )
        print(
            "pinned：",
            list(
                plan.pinned_turn_keys
            ),
        )
        print(
            "reason：",
            plan.reason,
        )

        assert (
            "turn:human-2"
            in plan.pinned_turn_keys
        )

        assert (
            "turn:human-3"
            in plan.selected_turn_keys
        )

        assert plan.can_compact

        print(
            "[PASS] 单个未归档 Hole 不再永久阻塞后面的安全 Turn"
        )

        store.close()

    # ========================================================
    # Case 4: reconciliation heals the pinned hole
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Case 4：下一轮 Reconcile 后 -> "
        "原 Pinned Turn 恢复为可验证历史"
    )
    print("=" * 80)

    with TemporaryDirectory() as temp_dir:
        store = (
            ConversationEventStore(
                database_path=(
                    Path(temp_dir)
                    / "events.sqlite3"
                )
            )
        )

        messages = (
            build_four_turns()
        )

        # 初始只归档 Turn1 / Turn3，制造 Turn2 hole。
        archive_subset(
            store,
            user_id="user-heal",
            thread_id="thread-heal",
            messages=messages[0:2],
        )
        archive_subset(
            store,
            user_id="user-heal",
            thread_id="thread-heal",
            messages=messages[4:6],
        )

        before_plan = (
            plan_working_memory_compaction(
                messages,
                config=CONFIG,
                event_store=store,
                thread_id="thread-heal",
            )
        )

        assert (
            "turn:human-2"
            in before_plan.pinned_turn_keys
        )

        # 新 Human Turn 之前，SecureRuntime 会做的事情：
        repair = (
            reconcile_checkpoint_archive(
                base_agent=(
                    FakeBaseAgent(
                        messages
                    )
                ),
                store=store,
                user_id="user-heal",
                thread_id="thread-heal",
            )
        )

        assert repair.success

        after_plan = (
            plan_working_memory_compaction(
                messages,
                config=CONFIG,
                event_store=store,
                thread_id="thread-heal",
            )
        )

        print(
            "before pinned：",
            list(
                before_plan.pinned_turn_keys
            ),
        )
        print(
            "after pinned：",
            list(
                after_plan.pinned_turn_keys
            ),
        )

        assert (
            "turn:human-2"
            not in after_plan.pinned_turn_keys
        )

        print(
            "[PASS] Pinned Turn 不是永久拒绝；下一轮自动 repair 后恢复"
        )

        store.close()

    print()
    print("=" * 80)
    print(
        "Archive Reconciliation Phase 7E-3 回归测试通过"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()