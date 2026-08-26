"""Human Approval 审计记录。"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any

from raglab.control.human_approval import (
    ApprovalAuditEventType,
    ApprovalDecision,
)

from raglab.control.tool_policy_repository import (
    DEFAULT_CONTROL_DATABASE_PATH,
)


def utc_now_text() -> str:
    """UTC ISO 时间。"""

    return (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


def build_event_key(
    *parts: str,
) -> str:
    """生成审计事件幂等键。"""

    raw = "|".join(
        str(
            part
        )
        for part
        in parts
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


class HumanApprovalAuditRepository:
    """高风险 Tool 人工审批审计 Repository。"""

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

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                human_approval_audit
                (
                    event_id TEXT
                        PRIMARY KEY,

                    event_key TEXT
                        NOT NULL
                        UNIQUE,

                    approval_id TEXT
                        NOT NULL,

                    operation_key TEXT
                        NOT NULL,

                    thread_id TEXT
                        NOT NULL,

                    user_id TEXT
                        NOT NULL,

                    tool_name TEXT
                        NOT NULL,

                    tool_call_id TEXT
                        NOT NULL,

                    effect_type TEXT,

                    event_type TEXT
                        NOT NULL,

                    actor TEXT,

                    reason TEXT,

                    args_json TEXT
                        NOT NULL,

                    created_at TEXT
                        NOT NULL,

                    CHECK (
                        event_type IN (
                            'REQUESTED',
                            'APPROVED',
                            'REJECTED'
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_human_approval_thread
                ON human_approval_audit(
                    thread_id,
                    created_at
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_human_approval_id
                ON human_approval_audit(
                    approval_id,
                    created_at
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_human_approval_tool
                ON human_approval_audit(
                    tool_name,
                    created_at
                )
                """
            )

    # ========================================================
    # Request
    # ========================================================

    def record_requested(
        self,
        *,
        approval_id: str,
        operation_key: str,
        thread_id: str,
        user_id: str,
        tool_name: str,
        tool_call_id: str,
        effect_type: (
            str
            | None
        ),
        args_json: str,
    ) -> None:
        """记录审批请求。

        interrupt 节点恢复时会从头重新执行，
        所以这里必须幂等。
        """

        event_key = (
            build_event_key(
                "REQUESTED",
                approval_id,
            )
        )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                INSERT OR IGNORE INTO
                human_approval_audit
                (
                    event_id,
                    event_key,
                    approval_id,
                    operation_key,
                    thread_id,
                    user_id,
                    tool_name,
                    tool_call_id,
                    effect_type,
                    event_type,
                    actor,
                    reason,
                    args_json,
                    created_at
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
                    ?,
                    'REQUESTED',
                    NULL,
                    NULL,
                    ?,
                    ?
                )
                """,
                (
                    uuid.uuid4().hex,
                    event_key,
                    approval_id,
                    operation_key,
                    thread_id,
                    user_id,
                    tool_name,
                    tool_call_id,
                    effect_type,
                    args_json,
                    now,
                ),
            )

    # ========================================================
    # Decision
    # ========================================================

    def record_decision(
        self,
        *,
        approval_id: str,
        operation_key: str,
        thread_id: str,
        user_id: str,
        tool_name: str,
        tool_call_id: str,
        effect_type: (
            str
            | None
        ),
        args_json: str,
        decision: ApprovalDecision,
        actor: str,
        reason: str,
    ) -> None:
        """记录批准或拒绝。

        相同决定重复执行时保持幂等，
        避免 LangGraph 节点恢复造成重复审计。
        """

        event_type = (
            ApprovalAuditEventType.APPROVED
            if decision
            == ApprovalDecision.APPROVE
            else ApprovalAuditEventType.REJECTED
        )

        event_key = (
            build_event_key(
                event_type.value,
                approval_id,
                actor,
                reason,
            )
        )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                INSERT OR IGNORE INTO
                human_approval_audit
                (
                    event_id,
                    event_key,
                    approval_id,
                    operation_key,
                    thread_id,
                    user_id,
                    tool_name,
                    tool_call_id,
                    effect_type,
                    event_type,
                    actor,
                    reason,
                    args_json,
                    created_at
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
                    event_key,
                    approval_id,
                    operation_key,
                    thread_id,
                    user_id,
                    tool_name,
                    tool_call_id,
                    effect_type,
                    event_type.value,
                    actor,
                    reason,
                    args_json,
                    now,
                ),
            )

    # ========================================================
    # Query
    # ========================================================

    def list_events(
        self,
        *,
        thread_id: (
            str
            | None
        ) = None,
        approval_id: (
            str
            | None
        ) = None,
        tool_name: (
            str
            | None
        ) = None,
        limit: int = 100,
    ) -> list[
        dict[str, Any]
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

        if thread_id:

            conditions.append(
                "thread_id = ?"
            )

            parameters.append(
                str(
                    thread_id
                ).strip()
            )

        if approval_id:

            conditions.append(
                "approval_id = ?"
            )

            parameters.append(
                str(
                    approval_id
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
            "FROM human_approval_audit"
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
            dict(
                row
            )
            for row
            in rows
        ]

    # ========================================================
    # Stats
    # ========================================================

    def statistics(
        self,
    ) -> dict[str, Any]:

        with self._connect() as connection:

            total = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM human_approval_audit
                    """
                ).fetchone()[0]
            )

            rows = connection.execute(
                """
                SELECT
                    event_type,
                    COUNT(*) AS count
                FROM human_approval_audit
                GROUP BY event_type
                ORDER BY event_type
                """
            ).fetchall()

        return {
            "database_path": str(
                self.database_path
            ),
            "total": total,
            "by_event_type": {
                str(
                    row[
                        "event_type"
                    ]
                ): int(
                    row[
                        "count"
                    ]
                )
                for row
                in rows
            },
        }