"""Remediation Case SQLite Repository。

用于管理：

    无法自动 Compensation 的外部副作用。

重要原则：

External Effect
=
不可修改的历史事实。

Remediation
=
对该历史事实的后续人工处理记录。

因此人工修复完成后：

    不删除 Effect；
    不修改 Effect 为“没发生”；

而是：

    Effect 保留
    +
    Remediation Case = RESOLVED
"""

from __future__ import annotations

import sqlite3
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any

from raglab.control.external_effect import (
    ExternalEffectRecord,
    ExternalEffectStatus,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
)

from raglab.control.remediation import (
    RemediationActionType,
    RemediationCase,
    RemediationFeedback,
    RemediationFeedbackType,
    RemediationPriority,
    RemediationStatus,
)

from raglab.control.tool_policy import (
    ToolEffectType,
)

from raglab.control.tool_policy_repository import (
    DEFAULT_CONTROL_DATABASE_PATH,
)


# ============================================================
# 时间
# ============================================================


def utc_now_text() -> str:
    """返回 UTC ISO 时间。"""

    return (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


# ============================================================
# Repository
# ============================================================


class RemediationRepository:
    """人工修复工单 Repository。"""

    def __init__(
        self,
        database_path: (
            str
            | Path
        ) = DEFAULT_CONTROL_DATABASE_PATH,
    ) -> None:

        self.database_path = (
            Path(
                database_path
            ).resolve()
        )

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

    # ========================================================
    # Schema
    # ========================================================

    def setup(
        self,
    ) -> None:
        """建立 Remediation 数据表。"""

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                remediation_case
                (
                    case_id TEXT
                        PRIMARY KEY,

                    plan_id TEXT
                        NOT NULL,

                    reconciliation_item_id TEXT
                        NOT NULL
                        UNIQUE,

                    effect_id TEXT
                        NOT NULL,

                    thread_id TEXT
                        NOT NULL,

                    tool_name TEXT
                        NOT NULL,

                    action_type TEXT
                        NOT NULL,

                    priority TEXT
                        NOT NULL,

                    status TEXT
                        NOT NULL,

                    summary TEXT
                        NOT NULL,

                    reason TEXT
                        NOT NULL,

                    owner TEXT,

                    resolution_note TEXT,

                    created_at TEXT
                        NOT NULL,

                    updated_at TEXT
                        NOT NULL,

                    started_at TEXT,

                    resolved_at TEXT,

                    accepted_risk_at TEXT,

                    FOREIGN KEY (
                        plan_id
                    )
                    REFERENCES
                        branch_reconciliation_plan(
                            plan_id
                        ),

                    FOREIGN KEY (
                        reconciliation_item_id
                    )
                    REFERENCES
                        branch_reconciliation_item(
                            item_id
                        ),

                    FOREIGN KEY (
                        effect_id
                    )
                    REFERENCES
                        external_effect_ledger(
                            effect_id
                        ),

                    CHECK (
                        action_type IN (
                            'INVESTIGATE',
                            'CORRECTIVE_ACTION',
                            'MANUAL_FIX'
                        )
                    ),

                    CHECK (
                        priority IN (
                            'LOW',
                            'MEDIUM',
                            'HIGH',
                            'CRITICAL'
                        )
                    ),

                    CHECK (
                        status IN (
                            'OPEN',
                            'IN_PROGRESS',
                            'RESOLVED',
                            'ACCEPTED_RISK'
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                remediation_feedback
                (
                    feedback_id TEXT
                        PRIMARY KEY,

                    case_id TEXT
                        NOT NULL,

                    feedback_type TEXT
                        NOT NULL,

                    actor TEXT
                        NOT NULL,

                    message TEXT
                        NOT NULL,

                    created_at TEXT
                        NOT NULL,

                    FOREIGN KEY (
                        case_id
                    )
                    REFERENCES
                        remediation_case(
                            case_id
                        )
                    ON DELETE CASCADE,

                    CHECK (
                        feedback_type IN (
                            'SYSTEM_CREATED',
                            'NOTE',
                            'STATUS_CHANGE',
                            'RESOLUTION',
                            'RISK_ACCEPTANCE',
                            'REOPEN'
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_remediation_status
                ON remediation_case(
                    status
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_remediation_thread
                ON remediation_case(
                    thread_id,
                    created_at
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_remediation_effect
                ON remediation_case(
                    effect_id
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_remediation_feedback_case
                ON remediation_feedback(
                    case_id,
                    created_at
                )
                """
            )

    # ========================================================
    # Row Convert
    # ========================================================

    @staticmethod
    def _row_to_case(
        row: sqlite3.Row,
    ) -> RemediationCase:

        return RemediationCase(

            case_id=str(
                row[
                    "case_id"
                ]
            ),

            plan_id=str(
                row[
                    "plan_id"
                ]
            ),

            reconciliation_item_id=str(
                row[
                    "reconciliation_item_id"
                ]
            ),

            effect_id=str(
                row[
                    "effect_id"
                ]
            ),

            thread_id=str(
                row[
                    "thread_id"
                ]
            ),

            tool_name=str(
                row[
                    "tool_name"
                ]
            ),

            action_type=(
                RemediationActionType(
                    str(
                        row[
                            "action_type"
                        ]
                    )
                )
            ),

            priority=(
                RemediationPriority(
                    str(
                        row[
                            "priority"
                        ]
                    )
                )
            ),

            status=(
                RemediationStatus(
                    str(
                        row[
                            "status"
                        ]
                    )
                )
            ),

            summary=str(
                row[
                    "summary"
                ]
            ),

            reason=str(
                row[
                    "reason"
                ]
            ),

            owner=(
                str(
                    row[
                        "owner"
                    ]
                )
                if row[
                    "owner"
                ]
                is not None
                else None
            ),

            resolution_note=(
                str(
                    row[
                        "resolution_note"
                    ]
                )
                if row[
                    "resolution_note"
                ]
                is not None
                else None
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

            started_at=(
                str(
                    row[
                        "started_at"
                    ]
                )
                if row[
                    "started_at"
                ]
                is not None
                else None
            ),

            resolved_at=(
                str(
                    row[
                        "resolved_at"
                    ]
                )
                if row[
                    "resolved_at"
                ]
                is not None
                else None
            ),

            accepted_risk_at=(
                str(
                    row[
                        "accepted_risk_at"
                    ]
                )
                if row[
                    "accepted_risk_at"
                ]
                is not None
                else None
            ),
        )

    @staticmethod
    def _row_to_feedback(
        row: sqlite3.Row,
    ) -> RemediationFeedback:

        return RemediationFeedback(

            feedback_id=str(
                row[
                    "feedback_id"
                ]
            ),

            case_id=str(
                row[
                    "case_id"
                ]
            ),

            feedback_type=(
                RemediationFeedbackType(
                    str(
                        row[
                            "feedback_type"
                        ]
                    )
                )
            ),

            actor=str(
                row[
                    "actor"
                ]
            ),

            message=str(
                row[
                    "message"
                ]
            ),

            created_at=str(
                row[
                    "created_at"
                ]
            ),
        )

    # ========================================================
    # Automatic Classification
    # ========================================================

    @staticmethod
    def _classify_effect(
        effect: ExternalEffectRecord,
    ) -> tuple[
        RemediationActionType,
        RemediationPriority,
    ]:
        """根据 Effect 自动确定人工处理类型。

        这里只决定：

            “人工接下来应该先做哪类事情”

        不自动决定具体修复内容。
        """

        # ----------------------------------------------------
        # 外部真实状态不确定。
        #
        # 第一任务必须是调查。
        # ----------------------------------------------------

        if effect.status in {
            ExternalEffectStatus.EXECUTING,
            ExternalEffectStatus.UNKNOWN,
            ExternalEffectStatus.COMPENSATING,
            ExternalEffectStatus.COMPENSATION_UNKNOWN,
        }:

            return (
                RemediationActionType.INVESTIGATE,
                RemediationPriority.HIGH,
            )

        # ----------------------------------------------------
        # 不可逆操作已经成功。
        #
        # 不能撤销，
        # 应设计后续纠正动作。
        # ----------------------------------------------------

        if (
            effect.effect_type
            == ToolEffectType.IRREVERSIBLE_WRITE
        ):

            return (
                RemediationActionType.CORRECTIVE_ACTION,
                RemediationPriority.HIGH,
            )

        # ----------------------------------------------------
        # 幂等写：
        #
        # 重复执行安全，
        # 但旧状态不一定能自动撤销。
        # ----------------------------------------------------

        if (
            effect.effect_type
            == ToolEffectType.IDEMPOTENT_WRITE
        ):

            return (
                RemediationActionType.MANUAL_FIX,
                RemediationPriority.MEDIUM,
            )

        # ----------------------------------------------------
        # 其他情况保守处理。
        # ----------------------------------------------------

        return (
            RemediationActionType.MANUAL_FIX,
            RemediationPriority.MEDIUM,
        )

    # ========================================================
    # 自动创建 Case
    # ========================================================

    def ensure_case(
        self,
        *,
        plan_id: str,
        reconciliation_item_id: str,
        effect: ExternalEffectRecord,
        reason: str,
    ) -> tuple[
        RemediationCase,
        bool,
    ]:
        """为 MANUAL_REVIEW Item 建立人工修复工单。

        同一个 Reconciliation Item
        只能产生一个 Case。

        返回：

            case
            created
        """

        (
            action_type,
            priority,
        ) = self._classify_effect(
            effect
        )

        case_id = (
            uuid.uuid4().hex
        )

        now = utc_now_text()

        summary = (
            "Replay 分支对账发现旧分支独有的"
            "不可自动恢复外部操作："
            f"{effect.tool_name}"
        )

        with self._connect() as connection:

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO
                remediation_case
                (
                    case_id,
                    plan_id,
                    reconciliation_item_id,
                    effect_id,
                    thread_id,
                    tool_name,
                    action_type,
                    priority,
                    status,
                    summary,
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
                    'OPEN',
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    case_id,
                    str(
                        plan_id
                    ),
                    str(
                        reconciliation_item_id
                    ),
                    effect.effect_id,
                    effect.thread_id,
                    effect.tool_name,
                    action_type.value,
                    priority.value,
                    summary,
                    str(
                        reason
                    ),
                    now,
                    now,
                ),
            )

            created = (
                cursor.rowcount
                == 1
            )

            row = connection.execute(
                """
                SELECT *
                FROM remediation_case
                WHERE reconciliation_item_id = ?
                """,
                (
                    str(
                        reconciliation_item_id
                    ),
                ),
            ).fetchone()

        if row is None:

            raise RuntimeError(
                "Remediation Case 创建后"
                "无法重新读取。"
            )

        case = self._row_to_case(
            row
        )

        if created:

            self.add_feedback(
                case_id=case.case_id,
                feedback_type=(
                    RemediationFeedbackType.SYSTEM_CREATED
                ),
                actor="system",
                message=(
                    "系统根据 Replay Branch "
                    "Reconciliation 自动创建了"
                    "人工修复工单。"
                ),
            )

        return (
            case,
            created,
        )

    def create_cases_from_plan(
        self,
        *,
        plan: Any,
        effect_repository: (
            ExternalEffectRepository
        ),
    ) -> list[
        RemediationCase
    ]:
        """为 Reconciliation Plan 中所有
        MANUAL_REVIEW 项创建工单。

        这里故意采用属性访问而不是直接 import
        branch_reconciliation 数据类，
        避免控制模块之间形成循环依赖。
        """

        cases: list[
            RemediationCase
        ] = []

        plan_id = str(
            getattr(
                plan,
                "plan_id",
                "",
            )
        ).strip()

        if not plan_id:

            raise ValueError(
                "Reconciliation Plan "
                "缺少 plan_id。"
            )

        items = list(
            getattr(
                plan,
                "items",
                (),
            )
            or ()
        )

        for item in items:

            disposition = getattr(
                item,
                "disposition",
                None,
            )

            disposition_value = str(
                getattr(
                    disposition,
                    "value",
                    disposition,
                )
            ).strip()

            if (
                disposition_value
                != "MANUAL_REVIEW"
            ):

                continue

            effect_id = str(
                getattr(
                    item,
                    "effect_id",
                    "",
                )
            ).strip()

            item_id = str(
                getattr(
                    item,
                    "item_id",
                    "",
                )
            ).strip()

            reason = str(
                getattr(
                    item,
                    "reason",
                    "",
                )
            ).strip()

            if (
                not effect_id
                or not item_id
            ):

                continue

            effect = (
                effect_repository.get(
                    effect_id
                )
            )

            if effect is None:

                # Effect Ledger 不完整时，
                # 不伪造修复工单。
                continue

            case, _ = (
                self.ensure_case(
                    plan_id=(
                        plan_id
                    ),
                    reconciliation_item_id=(
                        item_id
                    ),
                    effect=effect,
                    reason=reason,
                )
            )

            cases.append(
                case
            )

        return cases

    # ========================================================
    # Query
    # ========================================================

    def get_case(
        self,
        case_id: str,
    ) -> RemediationCase | None:

        normalized = str(
            case_id
        ).strip()

        if not normalized:

            raise ValueError(
                "case_id 不能为空。"
            )

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM remediation_case
                WHERE case_id = ?
                """,
                (
                    normalized,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_case(
            row
        )

    def list_cases(
        self,
        *,
        status: (
            RemediationStatus
            | str
            | None
        ) = None,
        thread_id: (
            str
            | None
        ) = None,
        tool_name: (
            str
            | None
        ) = None,
        limit: int = 50,
    ) -> list[
        RemediationCase
    ]:

        effective_limit = max(
            1,
            min(
                int(
                    limit
                ),
                500,
            ),
        )

        conditions: list[str] = []

        parameters: list[Any] = []

        if status is not None:

            status_value = (
                status.value
                if isinstance(
                    status,
                    RemediationStatus,
                )
                else str(
                    status
                ).strip().upper()
            )

            conditions.append(
                "status = ?"
            )

            parameters.append(
                status_value
            )

        if thread_id:

            conditions.append(
                "thread_id = ?"
            )

            parameters.append(
                str(
                    thread_id
                ).strip()
            )

        if tool_name:

            conditions.append(
                "tool_name = ?"
            )

            parameters.append(
                str(
                    tool_name
                ).strip()
            )

        where_sql = ""

        if conditions:

            where_sql = (
                " WHERE "
                + " AND ".join(
                    conditions
                )
            )

        query = (
            "SELECT * "
            "FROM remediation_case"
            + where_sql
            + " ORDER BY created_at DESC "
            "LIMIT ?"
        )

        parameters.append(
            effective_limit
        )

        with self._connect() as connection:

            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            self._row_to_case(
                row
            )
            for row
            in rows
        ]

    # ========================================================
    # Feedback
    # ========================================================

    def add_feedback(
        self,
        *,
        case_id: str,
        feedback_type: (
            RemediationFeedbackType
            | str
        ),
        actor: str,
        message: str,
    ) -> RemediationFeedback:
        """追加一条人工/系统反馈。

        Feedback 采用 append-only，
        不覆盖旧反馈。
        """

        case = self.get_case(
            case_id
        )

        if case is None:

            raise KeyError(
                f"Remediation Case 不存在："
                f"{case_id}"
            )

        if isinstance(
            feedback_type,
            RemediationFeedbackType,
        ):

            parsed_type = (
                feedback_type
            )

        else:

            parsed_type = (
                RemediationFeedbackType(
                    str(
                        feedback_type
                    ).strip().upper()
                )
            )

        normalized_actor = str(
            actor
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        normalized_message = str(
            message
        ).strip()

        if not normalized_message:

            raise ValueError(
                "feedback message 不能为空。"
            )

        feedback_id = (
            uuid.uuid4().hex
        )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                INSERT INTO
                remediation_feedback
                (
                    feedback_id,
                    case_id,
                    feedback_type,
                    actor,
                    message,
                    created_at
                )
                VALUES
                (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    feedback_id,
                    case.case_id,
                    parsed_type.value,
                    normalized_actor,
                    normalized_message,
                    now,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM remediation_feedback
                WHERE feedback_id = ?
                """,
                (
                    feedback_id,
                ),
            ).fetchone()

        if row is None:

            raise RuntimeError(
                "Feedback 写入后无法读取。"
            )

        return self._row_to_feedback(
            row
        )

    def list_feedback(
        self,
        case_id: str,
    ) -> list[
        RemediationFeedback
    ]:

        case = self.get_case(
            case_id
        )

        if case is None:

            raise KeyError(
                f"Remediation Case 不存在："
                f"{case_id}"
            )

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT *
                FROM remediation_feedback
                WHERE case_id = ?
                ORDER BY
                    created_at ASC,
                    feedback_id ASC
                """,
                (
                    case.case_id,
                ),
            ).fetchall()

        return [
            self._row_to_feedback(
                row
            )
            for row
            in rows
        ]

    def add_note(
        self,
        *,
        case_id: str,
        actor: str,
        message: str,
    ) -> RemediationFeedback:
        """人工追加普通处理备注。"""

        return self.add_feedback(
            case_id=case_id,
            feedback_type=(
                RemediationFeedbackType.NOTE
            ),
            actor=actor,
            message=message,
        )

    # ========================================================
    # Start
    # ========================================================

    def start_case(
        self,
        *,
        case_id: str,
        actor: str,
        note: (
            str
            | None
        ) = None,
    ) -> RemediationCase:
        """开始人工处理。"""

        case = self.get_case(
            case_id
        )

        if case is None:

            raise KeyError(
                f"Remediation Case 不存在："
                f"{case_id}"
            )

        if case.status not in {
            RemediationStatus.OPEN,
            RemediationStatus.IN_PROGRESS,
        }:

            raise ValueError(
                "已关闭的 Remediation Case "
                "不能直接 start；"
                "请先 reopen。"
            )

        normalized_actor = str(
            actor
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE
                    remediation_case
                SET
                    status = 'IN_PROGRESS',
                    owner = ?,
                    started_at =
                        COALESCE(
                            started_at,
                            ?
                        ),
                    updated_at = ?
                WHERE
                    case_id = ?
                """,
                (
                    normalized_actor,
                    now,
                    now,
                    case.case_id,
                ),
            )

        message = (
            f"{normalized_actor} 开始处理该工单。"
        )

        if note:

            message += (
                "\n"
                + str(
                    note
                ).strip()
            )

        self.add_feedback(
            case_id=case.case_id,
            feedback_type=(
                RemediationFeedbackType.STATUS_CHANGE
            ),
            actor=normalized_actor,
            message=message,
        )

        result = self.get_case(
            case.case_id
        )

        if result is None:

            raise RuntimeError(
                "Case 更新后无法读取。"
            )

        return result

    # ========================================================
    # Resolve
    # ========================================================

    def resolve_case(
        self,
        *,
        case_id: str,
        actor: str,
        resolution: str,
    ) -> RemediationCase:
        """人工确认已经完成后续修复。"""

        case = self.get_case(
            case_id
        )

        if case is None:

            raise KeyError(
                f"Remediation Case 不存在："
                f"{case_id}"
            )

        if case.status not in {
            RemediationStatus.OPEN,
            RemediationStatus.IN_PROGRESS,
        }:

            raise ValueError(
                "当前 Case 已关闭："
                f"{case.status.value}"
            )

        normalized_actor = str(
            actor
        ).strip()

        normalized_resolution = str(
            resolution
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        if not normalized_resolution:

            raise ValueError(
                "resolution 不能为空。"
            )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE
                    remediation_case
                SET
                    status = 'RESOLVED',
                    owner = COALESCE(
                        owner,
                        ?
                    ),
                    resolution_note = ?,
                    resolved_at = ?,
                    accepted_risk_at = NULL,
                    updated_at = ?
                WHERE
                    case_id = ?
                """,
                (
                    normalized_actor,
                    normalized_resolution,
                    now,
                    now,
                    case.case_id,
                ),
            )

        self.add_feedback(
            case_id=case.case_id,
            feedback_type=(
                RemediationFeedbackType.RESOLUTION
            ),
            actor=normalized_actor,
            message=(
                normalized_resolution
            ),
        )

        result = self.get_case(
            case.case_id
        )

        if result is None:

            raise RuntimeError(
                "Case RESOLVED 后无法读取。"
            )

        return result

    # ========================================================
    # Accept Risk
    # ========================================================

    def accept_risk(
        self,
        *,
        case_id: str,
        actor: str,
        reason: str,
    ) -> RemediationCase:
        """人工确认不再继续修复，并接受剩余风险。

        这不是“修复成功”。

        它表示：

            外部影响仍然存在，
            但负责人明确决定不再进行进一步处理。
        """

        case = self.get_case(
            case_id
        )

        if case is None:

            raise KeyError(
                f"Remediation Case 不存在："
                f"{case_id}"
            )

        if case.status not in {
            RemediationStatus.OPEN,
            RemediationStatus.IN_PROGRESS,
        }:

            raise ValueError(
                "当前 Case 已关闭："
                f"{case.status.value}"
            )

        normalized_actor = str(
            actor
        ).strip()

        normalized_reason = str(
            reason
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        if not normalized_reason:

            raise ValueError(
                "接受风险必须填写 reason。"
            )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE
                    remediation_case
                SET
                    status = 'ACCEPTED_RISK',
                    owner = COALESCE(
                        owner,
                        ?
                    ),
                    resolution_note = ?,
                    accepted_risk_at = ?,
                    resolved_at = ?,
                    updated_at = ?
                WHERE
                    case_id = ?
                """,
                (
                    normalized_actor,
                    normalized_reason,
                    now,
                    now,
                    now,
                    case.case_id,
                ),
            )

        self.add_feedback(
            case_id=case.case_id,
            feedback_type=(
                RemediationFeedbackType.RISK_ACCEPTANCE
            ),
            actor=normalized_actor,
            message=(
                normalized_reason
            ),
        )

        result = self.get_case(
            case.case_id
        )

        if result is None:

            raise RuntimeError(
                "Case ACCEPTED_RISK "
                "后无法读取。"
            )

        return result

    # ========================================================
    # Reopen
    # ========================================================

    def reopen_case(
        self,
        *,
        case_id: str,
        actor: str,
        reason: str,
    ) -> RemediationCase:
        """重新打开已经关闭的工单。"""

        case = self.get_case(
            case_id
        )

        if case is None:

            raise KeyError(
                f"Remediation Case 不存在："
                f"{case_id}"
            )

        if case.status not in {
            RemediationStatus.RESOLVED,
            RemediationStatus.ACCEPTED_RISK,
        }:

            raise ValueError(
                "只有已关闭 Case "
                "才需要 reopen。"
            )

        normalized_actor = str(
            actor
        ).strip()

        normalized_reason = str(
            reason
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        if not normalized_reason:

            raise ValueError(
                "reopen 必须填写 reason。"
            )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE
                    remediation_case
                SET
                    status = 'OPEN',
                    resolution_note = NULL,
                    resolved_at = NULL,
                    accepted_risk_at = NULL,
                    updated_at = ?
                WHERE
                    case_id = ?
                """,
                (
                    now,
                    case.case_id,
                ),
            )

        self.add_feedback(
            case_id=case.case_id,
            feedback_type=(
                RemediationFeedbackType.REOPEN
            ),
            actor=normalized_actor,
            message=(
                normalized_reason
            ),
        )

        result = self.get_case(
            case.case_id
        )

        if result is None:

            raise RuntimeError(
                "Case REOPEN 后无法读取。"
            )

        return result

    # ========================================================
    # Statistics
    # ========================================================

    def statistics(
        self,
    ) -> dict[str, Any]:

        with self._connect() as connection:

            total = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM remediation_case
                    """
                ).fetchone()[0]
            )

            status_rows = (
                connection.execute(
                    """
                    SELECT
                        status,
                        COUNT(*) AS count
                    FROM remediation_case
                    GROUP BY status
                    ORDER BY status
                    """
                ).fetchall()
            )

            action_rows = (
                connection.execute(
                    """
                    SELECT
                        action_type,
                        COUNT(*) AS count
                    FROM remediation_case
                    GROUP BY action_type
                    ORDER BY action_type
                    """
                ).fetchall()
            )

            priority_rows = (
                connection.execute(
                    """
                    SELECT
                        priority,
                        COUNT(*) AS count
                    FROM remediation_case
                    GROUP BY priority
                    ORDER BY priority
                    """
                ).fetchall()
            )

        return {
            "database_path": str(
                self.database_path
            ),
            "total": total,
            "by_status": {
                str(
                    row[
                        "status"
                    ]
                ): int(
                    row[
                        "count"
                    ]
                )
                for row
                in status_rows
            },
            "by_action_type": {
                str(
                    row[
                        "action_type"
                    ]
                ): int(
                    row[
                        "count"
                    ]
                )
                for row
                in action_rows
            },
            "by_priority": {
                str(
                    row[
                        "priority"
                    ]
                ): int(
                    row[
                        "count"
                    ]
                )
                for row
                in priority_rows
            },
        }