"""Replay Branch External Effect Reconciliation。

职责：

1. 找到 Replay 前的旧 Branch；
2. 找到 Replay 后的新 Branch；
3. 从两条 Branch 中提取 Tool Call；
4. 将 Tool Call 映射到 External Effect Ledger；
5. 比较旧、新 Effect；
6. 生成 Compensation Plan；
7. 经人工确认后执行可自动补偿项。

重要原则：

Replay != Commit

Replay 完成以后这里只生成计划，
不会自动修改外部系统。
"""

from __future__ import annotations

import sqlite3
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import (
    AIMessage,
)

from raglab.control.compensation import (
    ExternalEffectCompensationManager,
)

from raglab.control.external_effect import (
    ExternalEffectRecord,
    ExternalEffectStatus,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
)

from raglab.control.tool_policy import (
    ToolEffectType,
)

from raglab.control.tool_policy_repository import (
    DEFAULT_CONTROL_DATABASE_PATH,
)


# ============================================================
# Enum
# ============================================================


class ReconciliationDisposition(
    str,
    Enum,
):
    """新旧 Branch 对一个 Effect 的处理决定。"""

    # 两条分支都使用同一个 Effect。
    KEEP = "KEEP"

    # 旧分支独有，并且系统可以自动补偿。
    COMPENSATE = "COMPENSATE"

    # 旧分支独有，但无法安全自动补偿。
    MANUAL_REVIEW = "MANUAL_REVIEW"

    # 新分支新增的 Effect。
    NEW_EFFECT = "NEW_EFFECT"

    # 不需要进一步处理。
    IGNORE = "IGNORE"


class ReconciliationItemStatus(
    str,
    Enum,
):
    """单个计划项状态。"""

    PENDING = "PENDING"

    NO_ACTION = "NO_ACTION"

    COMPENSATED = "COMPENSATED"

    MANUAL_REQUIRED = (
        "MANUAL_REQUIRED"
    )

    FAILED = "FAILED"


class ReconciliationPlanStatus(
    str,
    Enum,
):
    """整个补偿计划状态。"""

    DRAFT = "DRAFT"

    APPLIED = "APPLIED"

    PARTIAL = "PARTIAL"

    FAILED = "FAILED"


# ============================================================
# Model
# ============================================================


@dataclass(
    frozen=True,
)
class ReconciliationItem:
    """一个 Effect 的 Branch 对账结果。"""

    item_id: str

    plan_id: str

    effect_id: str

    tool_name: str

    disposition: (
        ReconciliationDisposition
    )

    status: (
        ReconciliationItemStatus
    )

    reason: str

    created_at: str

    updated_at: str


@dataclass(
    frozen=True,
)
class ReconciliationPlan:
    """一次 Replay 对应的补偿计划。"""

    plan_id: str

    thread_id: str

    replay_checkpoint_id: str

    old_head_checkpoint_id: str

    new_head_checkpoint_id: str

    status: (
        ReconciliationPlanStatus
    )

    created_at: str

    updated_at: str

    items: tuple[
        ReconciliationItem,
        ...
    ]

    @property
    def compensate_count(
        self,
    ) -> int:

        return sum(
            1
            for item
            in self.items
            if item.disposition
            == ReconciliationDisposition.COMPENSATE
        )

    @property
    def manual_review_count(
        self,
    ) -> int:

        return sum(
            1
            for item
            in self.items
            if item.disposition
            == ReconciliationDisposition.MANUAL_REVIEW
        )

    @property
    def new_effect_count(
        self,
    ) -> int:

        return sum(
            1
            for item
            in self.items
            if item.disposition
            == ReconciliationDisposition.NEW_EFFECT
        )

    @property
    def keep_count(
        self,
    ) -> int:

        return sum(
            1
            for item
            in self.items
            if item.disposition
            == ReconciliationDisposition.KEEP
        )


# ============================================================
# Helpers
# ============================================================


def utc_now_text(
) -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def checkpoint_id_from_config(
    config: Any,
) -> str | None:

    if not isinstance(
        config,
        dict,
    ):
        return None

    configurable = config.get(
        "configurable",
        {},
    )

    if not isinstance(
        configurable,
        dict,
    ):
        return None

    value = configurable.get(
        "checkpoint_id"
    )

    if not value:
        return None

    return str(
        value
    )


def extract_tool_call_ids(
    messages: Iterable[Any],
) -> set[str]:
    """从累计消息状态中提取所有 Tool Call ID。"""

    result: set[str] = set()

    for message in messages:

        if not isinstance(
            message,
            AIMessage,
        ):
            continue

        for tool_call in list(
            message.tool_calls
            or []
        ):

            if isinstance(
                tool_call,
                dict,
            ):

                value = tool_call.get(
                    "id"
                )

            else:

                value = getattr(
                    tool_call,
                    "id",
                    None,
                )

            if value:

                result.add(
                    str(
                        value
                    )
                )

    return result


# ============================================================
# Repository + Manager
# ============================================================


class BranchReconciliationManager:
    """Replay Branch 外部副作用对账管理器。"""

    def __init__(
        self,
        *,
        agent: Any,
        database_path: (
            str
            | Path
        ) = DEFAULT_CONTROL_DATABASE_PATH,
    ) -> None:

        self.agent = agent

        self.database_path = (
            Path(
                database_path
            ).resolve()
        )

        self.effect_repository = (
            ExternalEffectRepository(
                database_path=(
                    self.database_path
                )
            )
        )

        self.setup()

    # ========================================================
    # SQLite
    # ========================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            str(
                self.database_path
            ),
            timeout=30.0,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        connection.execute(
            "PRAGMA synchronous = NORMAL"
        )

        connection.execute(
            "PRAGMA busy_timeout = 5000"
        )

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    def setup(
        self,
    ) -> None:
        """初始化 Branch Reconciliation 表。"""

        self.effect_repository.setup()

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                branch_reconciliation_plan
                (
                    plan_id TEXT
                        PRIMARY KEY,

                    thread_id TEXT
                        NOT NULL,

                    replay_checkpoint_id TEXT
                        NOT NULL,

                    old_head_checkpoint_id TEXT
                        NOT NULL,

                    new_head_checkpoint_id TEXT
                        NOT NULL,

                    status TEXT
                        NOT NULL,

                    created_at TEXT
                        NOT NULL,

                    updated_at TEXT
                        NOT NULL,

                    CHECK (
                        status IN (
                            'DRAFT',
                            'APPLIED',
                            'PARTIAL',
                            'FAILED'
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                branch_reconciliation_item
                (
                    item_id TEXT
                        PRIMARY KEY,

                    plan_id TEXT
                        NOT NULL,

                    effect_id TEXT
                        NOT NULL,

                    tool_name TEXT
                        NOT NULL,

                    disposition TEXT
                        NOT NULL,

                    status TEXT
                        NOT NULL,

                    reason TEXT
                        NOT NULL,

                    created_at TEXT
                        NOT NULL,

                    updated_at TEXT
                        NOT NULL,

                    FOREIGN KEY (
                        plan_id
                    )
                    REFERENCES
                        branch_reconciliation_plan(
                            plan_id
                        )
                    ON DELETE CASCADE,

                    FOREIGN KEY (
                        effect_id
                    )
                    REFERENCES
                        external_effect_ledger(
                            effect_id
                        ),

                    CHECK (
                        disposition IN (
                            'KEEP',
                            'COMPENSATE',
                            'MANUAL_REVIEW',
                            'NEW_EFFECT',
                            'IGNORE'
                        )
                    ),

                    CHECK (
                        status IN (
                            'PENDING',
                            'NO_ACTION',
                            'COMPENSATED',
                            'MANUAL_REQUIRED',
                            'FAILED'
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_reconciliation_thread
                ON branch_reconciliation_plan(
                    thread_id,
                    created_at
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_reconciliation_item_plan
                ON branch_reconciliation_item(
                    plan_id
                )
                """
            )

    # ========================================================
    # Checkpoint
    # ========================================================

    def _checkpoint_config(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any]:

        return {
            "configurable": {
                "thread_id": str(
                    thread_id
                ),
                "checkpoint_id": str(
                    checkpoint_id
                ),
            }
        }

    def _get_snapshot(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
    ) -> Any:

        return self.agent.graph.get_state(
            self._checkpoint_config(
                thread_id,
                checkpoint_id,
            )
        )

    def _parent_checkpoint_id(
        self,
        snapshot: Any,
    ) -> str | None:

        return checkpoint_id_from_config(
            getattr(
                snapshot,
                "parent_config",
                None,
            )
        )

    def _assert_ancestor(
        self,
        *,
        thread_id: str,
        ancestor_checkpoint_id: str,
        head_checkpoint_id: str,
    ) -> None:
        """确认 ancestor 确实位于 head 的父链中。"""

        current_id: str | None = (
            str(
                head_checkpoint_id
            )
        )

        visited: set[str] = set()

        while current_id:

            if (
                current_id
                == ancestor_checkpoint_id
            ):
                return

            if current_id in visited:

                raise RuntimeError(
                    "Checkpoint 父链出现循环。"
                )

            visited.add(
                current_id
            )

            snapshot = self._get_snapshot(
                thread_id=thread_id,
                checkpoint_id=(
                    current_id
                ),
            )

            current_id = (
                self._parent_checkpoint_id(
                    snapshot
                )
            )

        raise ValueError(
            "指定 Replay Checkpoint "
            "不是该 Branch Head 的祖先："
            f"{ancestor_checkpoint_id} "
            f"-> {head_checkpoint_id}"
        )

    # ========================================================
    # Branch Tool Calls
    # ========================================================

    def _branch_suffix_tool_call_ids(
        self,
        *,
        thread_id: str,
        replay_checkpoint_id: str,
        head_checkpoint_id: str,
    ) -> set[str]:
        """提取从 Replay 点开始到 Branch Head 的 Tool Calls。

        为什么使用 replay checkpoint 的 parent
        作为 baseline？

        因为如果 Replay checkpoint 自身 next=tools，
        那么导致 Tool 调用的 AIMessage 已经存在于
        replay checkpoint 中。

        因此这次 Tool Call 属于需要重新执行的后半段，
        不能被错误排除。
        """

        self._assert_ancestor(
            thread_id=thread_id,
            ancestor_checkpoint_id=(
                replay_checkpoint_id
            ),
            head_checkpoint_id=(
                head_checkpoint_id
            ),
        )

        replay_snapshot = (
            self._get_snapshot(
                thread_id=thread_id,
                checkpoint_id=(
                    replay_checkpoint_id
                ),
            )
        )

        head_snapshot = (
            self._get_snapshot(
                thread_id=thread_id,
                checkpoint_id=(
                    head_checkpoint_id
                ),
            )
        )

        replay_parent_id = (
            self._parent_checkpoint_id(
                replay_snapshot
            )
        )

        baseline_tool_calls: set[
            str
        ] = set()

        if replay_parent_id:

            parent_snapshot = (
                self._get_snapshot(
                    thread_id=thread_id,
                    checkpoint_id=(
                        replay_parent_id
                    ),
                )
            )

            baseline_values = getattr(
                parent_snapshot,
                "values",
                {},
            )

            baseline_messages = (
                baseline_values.get(
                    "messages",
                    [],
                )
                if isinstance(
                    baseline_values,
                    dict,
                )
                else []
            )

            baseline_tool_calls = (
                extract_tool_call_ids(
                    baseline_messages
                )
            )

        head_values = getattr(
            head_snapshot,
            "values",
            {},
        )

        head_messages = (
            head_values.get(
                "messages",
                [],
            )
            if isinstance(
                head_values,
                dict,
            )
            else []
        )

        head_tool_calls = (
            extract_tool_call_ids(
                head_messages
            )
        )

        return (
            head_tool_calls
            - baseline_tool_calls
        )

    # ========================================================
    # Effect Mapping
    # ========================================================

    def _effects_for_tool_call_ids(
        self,
        *,
        thread_id: str,
        tool_call_ids: set[str],
    ) -> dict[
        str,
        ExternalEffectRecord,
    ]:
        """Tool Call ID → External Effect。"""

        if not tool_call_ids:
            return {}

        effect_ids: list[str] = []

        all_ids = sorted(
            tool_call_ids
        )

        # 避免 SQLite 参数数量上限问题。
        chunk_size = 300

        with self._connect() as connection:

            for start in range(
                0,
                len(
                    all_ids
                ),
                chunk_size,
            ):

                current_ids = all_ids[
                    start:
                    start + chunk_size
                ]

                placeholders = ", ".join(
                    "?"
                    for _ in current_ids
                )

                rows = connection.execute(
                    f"""
                    SELECT effect_id
                    FROM external_effect_ledger
                    WHERE
                        thread_id = ?
                        AND tool_call_id
                        IN ({placeholders})
                    """,
                    [
                        str(
                            thread_id
                        ),
                        *current_ids,
                    ],
                ).fetchall()

                effect_ids.extend(
                    str(
                        row[
                            "effect_id"
                        ]
                    )
                    for row
                    in rows
                )

        result: dict[
            str,
            ExternalEffectRecord,
        ] = {}

        for effect_id in effect_ids:

            effect = (
                self.effect_repository.get(
                    effect_id
                )
            )

            if effect is not None:

                result[
                    effect.effect_id
                ] = effect

        return result

    # ========================================================
    # Classification
    # ========================================================

    @staticmethod
    def _classify_old_only_effect(
        effect: ExternalEffectRecord,
    ) -> tuple[
        ReconciliationDisposition,
        ReconciliationItemStatus,
        str,
    ]:
        """旧 Branch 独有 Effect 如何处理。"""

        if (
            effect.status
            == ExternalEffectStatus.COMPENSATED
        ):

            return (
                ReconciliationDisposition.IGNORE,
                ReconciliationItemStatus.NO_ACTION,
                "该 Effect 已经完成补偿。",
            )

        if (
            effect.status
            == ExternalEffectStatus.FAILED
        ):

            return (
                ReconciliationDisposition.IGNORE,
                ReconciliationItemStatus.NO_ACTION,
                "该 Effect 已明确执行失败，"
                "当前没有成功外部副作用需要补偿。",
            )

        if (
            effect.status
            != ExternalEffectStatus.SUCCEEDED
        ):

            return (
                ReconciliationDisposition.MANUAL_REVIEW,
                ReconciliationItemStatus.MANUAL_REQUIRED,
                (
                    "该 Effect 当前状态为 "
                    f"{effect.status.value}，"
                    "无法安全判断外部真实状态。"
                ),
            )

        if (
            effect.effect_type
            == ToolEffectType.COMPENSATABLE_WRITE
            and effect.compensation_tool
        ):

            return (
                ReconciliationDisposition.COMPENSATE,
                ReconciliationItemStatus.PENDING,
                (
                    "旧 Branch 独有外部写操作，"
                    "且存在已配置的补偿处理器。"
                ),
            )

        if (
            effect.effect_type
            == ToolEffectType.IDEMPOTENT_WRITE
        ):

            # ------------------------------------------------
            # 幂等只表示：
            #
            # 重复执行安全。
            #
            # 不代表：
            #
            # 原来的状态修改能够自动撤销。
            # ------------------------------------------------

            return (
                ReconciliationDisposition.MANUAL_REVIEW,
                ReconciliationItemStatus.MANUAL_REQUIRED,
                (
                    "该操作虽然支持幂等重复执行，"
                    "但没有自动撤销语义。"
                ),
            )

        if (
            effect.effect_type
            == ToolEffectType.IRREVERSIBLE_WRITE
        ):

            return (
                ReconciliationDisposition.MANUAL_REVIEW,
                ReconciliationItemStatus.MANUAL_REQUIRED,
                (
                    "旧 Branch 独有操作属于 "
                    "IRREVERSIBLE_WRITE，"
                    "不能自动补偿。"
                ),
            )

        return (
            ReconciliationDisposition.MANUAL_REVIEW,
            ReconciliationItemStatus.MANUAL_REQUIRED,
            "无法确定安全补偿方式。",
        )

    # ========================================================
    # Plan
    # ========================================================

    def create_plan(
        self,
        *,
        thread_id: str,
        replay_checkpoint_id: str,
        old_head_checkpoint_id: str,
        new_head_checkpoint_id: str,
    ) -> ReconciliationPlan:
        """比较 Replay 前后 Branch 并生成补偿计划。"""

        # ----------------------------------------------------
        # 先验证两条 Branch 都从 Replay 点分出。
        # ----------------------------------------------------

        self._assert_ancestor(
            thread_id=thread_id,
            ancestor_checkpoint_id=(
                replay_checkpoint_id
            ),
            head_checkpoint_id=(
                old_head_checkpoint_id
            ),
        )

        self._assert_ancestor(
            thread_id=thread_id,
            ancestor_checkpoint_id=(
                replay_checkpoint_id
            ),
            head_checkpoint_id=(
                new_head_checkpoint_id
            ),
        )

        old_tool_calls = (
            self._branch_suffix_tool_call_ids(
                thread_id=thread_id,
                replay_checkpoint_id=(
                    replay_checkpoint_id
                ),
                head_checkpoint_id=(
                    old_head_checkpoint_id
                ),
            )
        )

        new_tool_calls = (
            self._branch_suffix_tool_call_ids(
                thread_id=thread_id,
                replay_checkpoint_id=(
                    replay_checkpoint_id
                ),
                head_checkpoint_id=(
                    new_head_checkpoint_id
                ),
            )
        )

        old_effects = (
            self._effects_for_tool_call_ids(
                thread_id=thread_id,
                tool_call_ids=(
                    old_tool_calls
                ),
            )
        )

        new_effects = (
            self._effects_for_tool_call_ids(
                thread_id=thread_id,
                tool_call_ids=(
                    new_tool_calls
                ),
            )
        )

        old_ids = set(
            old_effects
        )

        new_ids = set(
            new_effects
        )

        common_ids = (
            old_ids
            & new_ids
        )

        old_only_ids = (
            old_ids
            - new_ids
        )

        new_only_ids = (
            new_ids
            - old_ids
        )

        plan_id = (
            uuid.uuid4().hex
        )

        now = utc_now_text()

        item_specs: list[
            tuple[
                ExternalEffectRecord,
                ReconciliationDisposition,
                ReconciliationItemStatus,
                str,
            ]
        ] = []

        # ----------------------------------------------------
        # 两条 Branch 共享。
        # ----------------------------------------------------

        for effect_id in sorted(
            common_ids
        ):

            effect = old_effects[
                effect_id
            ]

            item_specs.append(
                (
                    effect,
                    ReconciliationDisposition.KEEP,
                    ReconciliationItemStatus.NO_ACTION,
                    (
                        "该 Effect 在旧、新 Branch "
                        "中均被使用，保持不变。"
                    ),
                )
            )

        # ----------------------------------------------------
        # 旧 Branch 独有。
        # ----------------------------------------------------

        for effect_id in sorted(
            old_only_ids
        ):

            effect = old_effects[
                effect_id
            ]

            (
                disposition,
                item_status,
                reason,
            ) = (
                self
                ._classify_old_only_effect(
                    effect
                )
            )

            item_specs.append(
                (
                    effect,
                    disposition,
                    item_status,
                    reason,
                )
            )

        # ----------------------------------------------------
        # 新 Branch 独有。
        # ----------------------------------------------------

        for effect_id in sorted(
            new_only_ids
        ):

            effect = new_effects[
                effect_id
            ]

            item_specs.append(
                (
                    effect,
                    ReconciliationDisposition.NEW_EFFECT,
                    ReconciliationItemStatus.NO_ACTION,
                    (
                        "该 Effect 只存在于 Replay "
                        "产生的新 Branch。"
                    ),
                )
            )

        with self._connect() as connection:

            connection.execute(
                """
                INSERT INTO
                branch_reconciliation_plan
                (
                    plan_id,
                    thread_id,
                    replay_checkpoint_id,
                    old_head_checkpoint_id,
                    new_head_checkpoint_id,
                    status,
                    created_at,
                    updated_at
                )
                VALUES
                (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    'DRAFT',
                    ?,
                    ?
                )
                """,
                (
                    plan_id,
                    str(
                        thread_id
                    ),
                    str(
                        replay_checkpoint_id
                    ),
                    str(
                        old_head_checkpoint_id
                    ),
                    str(
                        new_head_checkpoint_id
                    ),
                    now,
                    now,
                ),
            )

            for (
                effect,
                disposition,
                item_status,
                reason,
            ) in item_specs:

                connection.execute(
                    """
                    INSERT INTO
                    branch_reconciliation_item
                    (
                        item_id,
                        plan_id,
                        effect_id,
                        tool_name,
                        disposition,
                        status,
                        reason,
                        created_at,
                        updated_at
                    )
                    VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        uuid.uuid4().hex,
                        plan_id,
                        effect.effect_id,
                        effect.tool_name,
                        disposition.value,
                        item_status.value,
                        reason,
                        now,
                        now,
                    ),
                )

        result = self.get_plan(
            plan_id
        )

        if result is None:

            raise RuntimeError(
                "Reconciliation Plan "
                "创建后无法重新读取。"
            )

        return result

    # ========================================================
    # Read Plan
    # ========================================================

    def _load_items(
        self,
        plan_id: str,
    ) -> tuple[
        ReconciliationItem,
        ...
    ]:

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT *
                FROM branch_reconciliation_item
                WHERE plan_id = ?
                ORDER BY created_at, item_id
                """,
                (
                    plan_id,
                ),
            ).fetchall()

        return tuple(
            ReconciliationItem(
                item_id=str(
                    row[
                        "item_id"
                    ]
                ),
                plan_id=str(
                    row[
                        "plan_id"
                    ]
                ),
                effect_id=str(
                    row[
                        "effect_id"
                    ]
                ),
                tool_name=str(
                    row[
                        "tool_name"
                    ]
                ),
                disposition=(
                    ReconciliationDisposition(
                        str(
                            row[
                                "disposition"
                            ]
                        )
                    )
                ),
                status=(
                    ReconciliationItemStatus(
                        str(
                            row[
                                "status"
                            ]
                        )
                    )
                ),
                reason=str(
                    row[
                        "reason"
                    ]
                ),
                created_at=str(
                    row[
                        "created_at"
                    ]
                ),
                updated_at=str(
                    row[
                        "updated_at"
                    ]
                ),
            )
            for row
            in rows
        )

    def get_plan(
        self,
        plan_id: str,
    ) -> ReconciliationPlan | None:

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM branch_reconciliation_plan
                WHERE plan_id = ?
                """,
                (
                    str(
                        plan_id
                    ).strip(),
                ),
            ).fetchone()

        if row is None:
            return None

        normalized_plan_id = str(
            row[
                "plan_id"
            ]
        )

        return ReconciliationPlan(
            plan_id=(
                normalized_plan_id
            ),
            thread_id=str(
                row[
                    "thread_id"
                ]
            ),
            replay_checkpoint_id=str(
                row[
                    "replay_checkpoint_id"
                ]
            ),
            old_head_checkpoint_id=str(
                row[
                    "old_head_checkpoint_id"
                ]
            ),
            new_head_checkpoint_id=str(
                row[
                    "new_head_checkpoint_id"
                ]
            ),
            status=(
                ReconciliationPlanStatus(
                    str(
                        row[
                            "status"
                        ]
                    )
                )
            ),
            created_at=str(
                row[
                    "created_at"
                ]
            ),
            updated_at=str(
                row[
                    "updated_at"
                ]
            ),
            items=self._load_items(
                normalized_plan_id
            ),
        )

    def list_plans(
        self,
        *,
        thread_id: (
            str
            | None
        ) = None,
        limit: int = 20,
    ) -> list[
        ReconciliationPlan
    ]:

        effective_limit = max(
            1,
            min(
                int(
                    limit
                ),
                100,
            ),
        )

        if thread_id:

            sql = """
                SELECT plan_id
                FROM branch_reconciliation_plan
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """

            parameters: list[Any] = [
                str(
                    thread_id
                ),
                effective_limit,
            ]

        else:

            sql = """
                SELECT plan_id
                FROM branch_reconciliation_plan
                ORDER BY created_at DESC
                LIMIT ?
            """

            parameters = [
                effective_limit
            ]

        with self._connect() as connection:

            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        result: list[
            ReconciliationPlan
        ] = []

        for row in rows:

            plan = self.get_plan(
                str(
                    row[
                        "plan_id"
                    ]
                )
            )

            if plan is not None:

                result.append(
                    plan
                )

        return result

    # ========================================================
    # Apply
    # ========================================================

    def apply_plan(
        self,
        *,
        plan_id: str,
        compensation_manager: (
            ExternalEffectCompensationManager
        ),
    ) -> ReconciliationPlan:
        """执行计划中的 COMPENSATE 项。

        MANUAL_REVIEW 永远不会在这里自动执行。
        """

        plan = self.get_plan(
            plan_id
        )

        if plan is None:

            raise KeyError(
                f"Plan 不存在：{plan_id}"
            )

        if (
            plan.status
            != ReconciliationPlanStatus.DRAFT
        ):

            raise ValueError(
                "只有 DRAFT Plan "
                "可以执行。\n"
                f"当前状态："
                f"{plan.status.value}"
            )

        had_failure = False

        now = utc_now_text()

        for item in plan.items:

            if (
                item.disposition
                != ReconciliationDisposition.COMPENSATE
            ):

                continue

            if (
                item.status
                != ReconciliationItemStatus.PENDING
            ):

                continue

            try:

                compensation_manager.compensate(
                    item.effect_id
                )

            except Exception as exc:

                had_failure = True

                with self._connect() as connection:

                    connection.execute(
                        """
                        UPDATE
                            branch_reconciliation_item
                        SET
                            status = 'FAILED',
                            reason = ?,
                            updated_at = ?
                        WHERE
                            item_id = ?
                        """,
                        (
                            (
                                item.reason
                                + "\n补偿失败："
                                + f"{type(exc).__name__}: "
                                + str(
                                    exc
                                )
                            ),
                            utc_now_text(),
                            item.item_id,
                        ),
                    )

                continue

            with self._connect() as connection:

                connection.execute(
                    """
                    UPDATE
                        branch_reconciliation_item
                    SET
                        status = 'COMPENSATED',
                        updated_at = ?
                    WHERE
                        item_id = ?
                    """,
                    (
                        utc_now_text(),
                        item.item_id,
                    ),
                )

        refreshed = self.get_plan(
            plan_id
        )

        if refreshed is None:

            raise RuntimeError(
                "执行后无法读取 Plan。"
            )

        has_manual = any(
            item.status
            == ReconciliationItemStatus.MANUAL_REQUIRED
            for item
            in refreshed.items
        )

        has_failed_item = any(
            item.status
            == ReconciliationItemStatus.FAILED
            for item
            in refreshed.items
        )

        if (
            had_failure
            or has_failed_item
        ):

            target_status = (
                ReconciliationPlanStatus.FAILED
            )

        elif has_manual:

            target_status = (
                ReconciliationPlanStatus.PARTIAL
            )

        else:

            target_status = (
                ReconciliationPlanStatus.APPLIED
            )

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE
                    branch_reconciliation_plan
                SET
                    status = ?,
                    updated_at = ?
                WHERE
                    plan_id = ?
                """,
                (
                    target_status.value,
                    now,
                    plan_id,
                ),
            )

        result = self.get_plan(
            plan_id
        )

        if result is None:

            raise RuntimeError(
                "更新 Plan 状态后无法读取。"
            )

        return result