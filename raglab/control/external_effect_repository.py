"""External Effect Ledger SQLite Repository。"""

from __future__ import annotations

import hashlib
import json
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

from raglab.control.tool_policy import (
    ReplayPolicy,
    ToolEffectType,
)

from raglab.control.tool_policy_repository import (
    DEFAULT_CONTROL_DATABASE_PATH,
)


MAXIMUM_RESULT_CHARACTERS = 100_000


def utc_now_text() -> str:
    """UTC ISO 时间。"""

    return (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


def serialize_arguments(
    arguments: Any,
) -> str:
    """稳定序列化 Tool 参数。"""

    try:

        return json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

    except Exception:

        return json.dumps(
            {
                "fallback": str(
                    arguments
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def calculate_text_hash(
    text: str,
) -> str:
    """计算 SHA-256。"""

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def build_operation_key(
    *,
    thread_id: str,
    tool_name: str,
    tool_call_id: str,
    args_hash: str,
) -> str:
    """构造一次具体 Tool Call 的稳定操作键。

    Replay 如果复用的是同一个已经持久化的
    AI Tool Call：

        thread_id 相同
        tool_call_id 相同
        tool_name 相同
        args 相同

    operation_key 就相同。

    因此不会再次执行真实外部写操作。

    如果 Replay 从 Agent 节点之前开始，
    模型重新生成了新的 Tool Call，
    通常会产生新的 tool_call_id。

    这会被视为新的 Branch 操作，
    后续由 Branch Reconciliation 处理。
    """

    raw = "|".join(
        [
            str(
                thread_id
            ),
            str(
                tool_name
            ),
            str(
                tool_call_id
            ),
            str(
                args_hash
            ),
        ]
    )

    return calculate_text_hash(
        raw
    )


def normalize_result_text(
    value: Any,
) -> str:
    """把 Tool 结果转换为 Ledger 文本。"""

    if value is None:
        return ""

    if isinstance(
        value,
        str,
    ):

        text = value

    else:

        try:

            text = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        except Exception:

            text = str(
                value
            )

    if (
        len(
            text
        )
        > MAXIMUM_RESULT_CHARACTERS
    ):

        return (
            text[
                :MAXIMUM_RESULT_CHARACTERS
            ]
            + "\n...[truncated]"
        )

    return text


class ExternalEffectRepository:
    """External Effect Ledger。"""

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
                external_effect_ledger
                (
                    effect_id TEXT
                        PRIMARY KEY,

                    operation_key TEXT
                        NOT NULL
                        UNIQUE,

                    thread_id TEXT
                        NOT NULL,

                    user_id TEXT
                        NOT NULL,

                    checkpoint_id TEXT,

                    replay_from_checkpoint_id TEXT,

                    execution_mode TEXT
                        NOT NULL,

                    tool_name TEXT
                        NOT NULL,

                    tool_call_id TEXT
                        NOT NULL,

                    effect_type TEXT
                        NOT NULL,

                    replay_policy TEXT
                        NOT NULL,

                    args_json TEXT
                        NOT NULL,

                    args_hash TEXT
                        NOT NULL,

                    status TEXT
                        NOT NULL,

                    result_text TEXT,

                    error_text TEXT,

                    compensation_tool TEXT,

                    compensation_result_text TEXT,

                    compensation_error_text TEXT,

                    created_at TEXT
                        NOT NULL,

                    updated_at TEXT
                        NOT NULL,

                    execution_started_at TEXT,

                    succeeded_at TEXT,

                    compensated_at TEXT,

                    CHECK (
                        effect_type IN (
                            'IDEMPOTENT_WRITE',
                            'COMPENSATABLE_WRITE',
                            'IRREVERSIBLE_WRITE'
                        )
                    ),

                    CHECK (
                        replay_policy IN (
                            'ALLOW',
                            'GUARDED',
                            'REQUIRE_APPROVAL',
                            'DENY'
                        )
                    ),

                    CHECK (
                        status IN (
                            'PREPARED',
                            'EXECUTING',
                            'SUCCEEDED',
                            'FAILED',
                            'UNKNOWN',
                            'COMPENSATING',
                            'COMPENSATED',
                            'COMPENSATION_UNKNOWN'
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_external_effect_thread
                ON external_effect_ledger(
                    thread_id,
                    created_at
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_external_effect_status
                ON external_effect_ledger(
                    status
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_external_effect_tool
                ON external_effect_ledger(
                    tool_name
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_external_effect_checkpoint
                ON external_effect_ledger(
                    checkpoint_id
                )
                """
            )

    # ========================================================
    # Row
    # ========================================================

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> ExternalEffectRecord:

        return ExternalEffectRecord(

            effect_id=str(
                row[
                    "effect_id"
                ]
            ),

            operation_key=str(
                row[
                    "operation_key"
                ]
            ),

            thread_id=str(
                row[
                    "thread_id"
                ]
            ),

            user_id=str(
                row[
                    "user_id"
                ]
            ),

            checkpoint_id=(
                str(
                    row[
                        "checkpoint_id"
                    ]
                )
                if row[
                    "checkpoint_id"
                ]
                is not None
                else None
            ),

            replay_from_checkpoint_id=(
                str(
                    row[
                        "replay_from_checkpoint_id"
                    ]
                )
                if row[
                    "replay_from_checkpoint_id"
                ]
                is not None
                else None
            ),

            execution_mode=str(
                row[
                    "execution_mode"
                ]
            ),

            tool_name=str(
                row[
                    "tool_name"
                ]
            ),

            tool_call_id=str(
                row[
                    "tool_call_id"
                ]
            ),

            effect_type=(
                ToolEffectType(
                    str(
                        row[
                            "effect_type"
                        ]
                    )
                )
            ),

            replay_policy=(
                ReplayPolicy(
                    str(
                        row[
                            "replay_policy"
                        ]
                    )
                )
            ),

            args_json=str(
                row[
                    "args_json"
                ]
            ),

            args_hash=str(
                row[
                    "args_hash"
                ]
            ),

            status=(
                ExternalEffectStatus(
                    str(
                        row[
                            "status"
                        ]
                    )
                )
            ),

            result_text=(
                str(
                    row[
                        "result_text"
                    ]
                )
                if row[
                    "result_text"
                ]
                is not None
                else None
            ),

            error_text=(
                str(
                    row[
                        "error_text"
                    ]
                )
                if row[
                    "error_text"
                ]
                is not None
                else None
            ),

            compensation_tool=(
                str(
                    row[
                        "compensation_tool"
                    ]
                )
                if row[
                    "compensation_tool"
                ]
                is not None
                else None
            ),

            compensation_result_text=(
                str(
                    row[
                        "compensation_result_text"
                    ]
                )
                if row[
                    "compensation_result_text"
                ]
                is not None
                else None
            ),

            compensation_error_text=(
                str(
                    row[
                        "compensation_error_text"
                    ]
                )
                if row[
                    "compensation_error_text"
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

            execution_started_at=(
                str(
                    row[
                        "execution_started_at"
                    ]
                )
                if row[
                    "execution_started_at"
                ]
                is not None
                else None
            ),

            succeeded_at=(
                str(
                    row[
                        "succeeded_at"
                    ]
                )
                if row[
                    "succeeded_at"
                ]
                is not None
                else None
            ),

            compensated_at=(
                str(
                    row[
                        "compensated_at"
                    ]
                )
                if row[
                    "compensated_at"
                ]
                is not None
                else None
            ),
        )

    # ========================================================
    # Query
    # ========================================================

    def get(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord | None:

        normalized = str(
            effect_id
        ).strip()

        if not normalized:
            raise ValueError(
                "effect_id 不能为空。"
            )

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM external_effect_ledger
                WHERE effect_id = ?
                """,
                (
                    normalized,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(
            row
        )

    def get_by_operation_key(
        self,
        operation_key: str,
    ) -> ExternalEffectRecord | None:

        normalized = str(
            operation_key
        ).strip()

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM external_effect_ledger
                WHERE operation_key = ?
                """,
                (
                    normalized,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(
            row
        )

    # ========================================================
    # Prepare
    # ========================================================

    def prepare_effect(
        self,
        *,
        operation_key: str,
        thread_id: str,
        user_id: str,
        checkpoint_id: (
            str
            | None
        ),
        replay_from_checkpoint_id: (
            str
            | None
        ),
        execution_mode: str,
        tool_name: str,
        tool_call_id: str,
        effect_type: ToolEffectType,
        replay_policy: ReplayPolicy,
        arguments: Any,
        compensation_tool: (
            str
            | None
        ),
    ) -> tuple[
        ExternalEffectRecord,
        bool,
    ]:
        """在真正调用外部系统之前写入 PREPARED。

        返回：

            record
            created

        created=False 表示这次 Tool Call
        已经在 Ledger 中出现过。
        """

        args_json = (
            serialize_arguments(
                arguments
            )
        )

        args_hash = (
            calculate_text_hash(
                args_json
            )
        )

        effect_id = (
            uuid.uuid4().hex
        )

        now = utc_now_text()

        with self._connect() as connection:

            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO
                external_effect_ledger
                (
                    effect_id,
                    operation_key,
                    thread_id,
                    user_id,
                    checkpoint_id,
                    replay_from_checkpoint_id,
                    execution_mode,
                    tool_name,
                    tool_call_id,
                    effect_type,
                    replay_policy,
                    args_json,
                    args_hash,
                    status,
                    compensation_tool,
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
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    'PREPARED',
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    effect_id,
                    operation_key,
                    str(
                        thread_id
                    ),
                    str(
                        user_id
                    ),
                    checkpoint_id,
                    replay_from_checkpoint_id,
                    str(
                        execution_mode
                    ),
                    str(
                        tool_name
                    ),
                    str(
                        tool_call_id
                    ),
                    effect_type.value,
                    replay_policy.value,
                    args_json,
                    args_hash,
                    compensation_tool,
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
                FROM external_effect_ledger
                WHERE operation_key = ?
                """,
                (
                    operation_key,
                ),
            ).fetchone()

        if row is None:
            raise RuntimeError(
                "Effect PREPARE 后无法重新读取。"
            )

        return (
            self._row_to_record(
                row
            ),
            created,
        )

    # ========================================================
    # Execution lifecycle
    # ========================================================

    def mark_executing(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord:

        return self._transition(
            effect_id=effect_id,
            expected={
                ExternalEffectStatus.PREPARED
            },
            target=(
                ExternalEffectStatus.EXECUTING
            ),
            extra_sql=(
                "execution_started_at = ?"
            ),
            extra_values=[
                utc_now_text()
            ],
        )

    def mark_succeeded(
        self,
        effect_id: str,
        result: Any,
    ) -> ExternalEffectRecord:

        text = normalize_result_text(
            result
        )

        now = utc_now_text()

        return self._transition(
            effect_id=effect_id,
            expected={
                ExternalEffectStatus.EXECUTING
            },
            target=(
                ExternalEffectStatus.SUCCEEDED
            ),
            extra_sql=(
                "result_text = ?, "
                "error_text = NULL, "
                "succeeded_at = ?"
            ),
            extra_values=[
                text,
                now,
            ],
        )

    def mark_failed(
        self,
        effect_id: str,
        error: Any,
    ) -> ExternalEffectRecord:

        return self._transition(
            effect_id=effect_id,
            expected={
                ExternalEffectStatus.EXECUTING
            },
            target=(
                ExternalEffectStatus.FAILED
            ),
            extra_sql=(
                "error_text = ?"
            ),
            extra_values=[
                normalize_result_text(
                    error
                )
            ],
        )

    def mark_unknown(
        self,
        effect_id: str,
        error: Any,
    ) -> ExternalEffectRecord:

        current = self.get(
            effect_id
        )

        if current is None:
            raise KeyError(
                f"Effect 不存在：{effect_id}"
            )

        if (
            current.status
            == ExternalEffectStatus.UNKNOWN
        ):
            return current

        return self._transition(
            effect_id=effect_id,
            expected={
                ExternalEffectStatus.EXECUTING
            },
            target=(
                ExternalEffectStatus.UNKNOWN
            ),
            extra_sql=(
                "error_text = ?"
            ),
            extra_values=[
                normalize_result_text(
                    error
                )
            ],
        )

    # ========================================================
    # Compensation lifecycle
    # ========================================================

    def mark_compensating(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord:

        return self._transition(
            effect_id=effect_id,
            expected={
                ExternalEffectStatus.SUCCEEDED
            },
            target=(
                ExternalEffectStatus.COMPENSATING
            ),
            extra_sql=(
                "compensation_error_text = NULL"
            ),
            extra_values=[],
        )

    def mark_compensated(
        self,
        effect_id: str,
        result: Any,
    ) -> ExternalEffectRecord:

        now = utc_now_text()

        return self._transition(
            effect_id=effect_id,
            expected={
                ExternalEffectStatus.COMPENSATING
            },
            target=(
                ExternalEffectStatus.COMPENSATED
            ),
            extra_sql=(
                "compensation_result_text = ?, "
                "compensation_error_text = NULL, "
                "compensated_at = ?"
            ),
            extra_values=[
                normalize_result_text(
                    result
                ),
                now,
            ],
        )

    def mark_compensation_unknown(
        self,
        effect_id: str,
        error: Any,
    ) -> ExternalEffectRecord:

        return self._transition(
            effect_id=effect_id,
            expected={
                ExternalEffectStatus.COMPENSATING
            },
            target=(
                ExternalEffectStatus.COMPENSATION_UNKNOWN
            ),
            extra_sql=(
                "compensation_error_text = ?"
            ),
            extra_values=[
                normalize_result_text(
                    error
                )
            ],
        )

    # ========================================================
    # Transition
    # ========================================================

    def _transition(
        self,
        *,
        effect_id: str,
        expected: set[
            ExternalEffectStatus
        ],
        target: ExternalEffectStatus,
        extra_sql: str,
        extra_values: list[Any],
    ) -> ExternalEffectRecord:

        normalized = str(
            effect_id
        ).strip()

        if not normalized:
            raise ValueError(
                "effect_id 不能为空。"
            )

        expected_values = [
            item.value
            for item
            in expected
        ]

        placeholders = ", ".join(
            "?"
            for _ in expected_values
        )

        now = utc_now_text()

        sql = (
            "UPDATE external_effect_ledger "
            "SET status = ?, "
            "updated_at = ?"
        )

        if extra_sql:

            sql += (
                ", "
                + extra_sql
            )

        sql += (
            " WHERE effect_id = ? "
            f"AND status IN ({placeholders})"
        )

        parameters: list[Any] = [
            target.value,
            now,
            *extra_values,
            normalized,
            *expected_values,
        ]

        with self._connect() as connection:

            cursor = connection.execute(
                sql,
                parameters,
            )

        if cursor.rowcount != 1:

            current = self.get(
                normalized
            )

            if current is None:
                raise KeyError(
                    f"Effect 不存在：{normalized}"
                )

            raise RuntimeError(
                "非法 Effect 状态迁移："
                f"{current.status.value} "
                f"→ {target.value}"
            )

        result = self.get(
            normalized
        )

        if result is None:
            raise RuntimeError(
                "Effect 状态更新后无法读取。"
            )

        return result

    # ========================================================
    # List
    # ========================================================

    def list_recent(
        self,
        *,
        limit: int = 50,
        thread_id: (
            str
            | None
        ) = None,
        status: (
            ExternalEffectStatus
            | str
            | None
        ) = None,
        tool_name: (
            str
            | None
        ) = None,
    ) -> list[
        ExternalEffectRecord
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

        if status:

            status_value = (
                status.value
                if isinstance(
                    status,
                    ExternalEffectStatus,
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
            "FROM external_effect_ledger"
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
            self._row_to_record(
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
                    FROM external_effect_ledger
                    """
                ).fetchone()[0]
            )

            status_rows = (
                connection.execute(
                    """
                    SELECT
                        status,
                        COUNT(*) AS count
                    FROM
                        external_effect_ledger
                    GROUP BY
                        status
                    ORDER BY
                        status
                    """
                ).fetchall()
            )

            type_rows = (
                connection.execute(
                    """
                    SELECT
                        effect_type,
                        COUNT(*) AS count
                    FROM
                        external_effect_ledger
                    GROUP BY
                        effect_type
                    ORDER BY
                        effect_type
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
            "by_effect_type": {
                str(
                    row[
                        "effect_type"
                    ]
                ): int(
                    row[
                        "count"
                    ]
                )
                for row
                in type_rows
            },
        }