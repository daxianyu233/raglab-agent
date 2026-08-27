"""RAGLab Conversation Event Store.

本模块负责持久保存 Agent 原始对话事件。

它与 LangGraph Checkpoint 的职责不同：

- Checkpoint:
    当前 Graph 可恢复运行状态、节点状态、HITL 等；
- Conversation Event Store:
    长期保存 Human / AI / Tool 等原始事件，
    即使 Checkpoint 后续通过滚动摘要删除旧 messages，
    原始内容仍可从这里恢复。

本阶段只实现存储与读取，不接入主 Agent。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_EVENT_STORE_PATH = (
    PROJECT_ROOT
    / "storage"
    / "agent_state"
    / "raglab_conversation_events.sqlite3"
)


@dataclass(
    frozen=True,
)
class ConversationEvent:
    """Conversation Event Store 中的一条原始事件。"""

    event_id: str
    user_id: str
    thread_id: str
    turn_id: str
    sequence_no: int
    event_type: str
    role: str

    message_id: str | None
    tool_call_id: str | None
    tool_name: str | None

    content_text: str
    payload: dict[str, Any]
    metadata: dict[str, Any]

    created_at: str


@dataclass(
    frozen=True,
)
class ConversationThread:
    """一个可在 Web/API 中恢复的持久化会话。"""

    thread_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message_preview: str = ""


@dataclass(frozen=True)
class AgentExecution:
    """一次可跨越 HITL 中断与恢复的 Agent 业务执行。"""

    execution_id: str
    user_id: str
    thread_id: str
    status: str
    current_step: str
    error_message: str
    created_at: str
    started_at: str
    updated_at: str
    finished_at: str | None


@dataclass(frozen=True)
class AgentExecutionEvent:
    """一次 Agent 执行中可供断线恢复的可观察事件。"""

    execution_id: str
    sequence_no: int
    event_type: str
    payload: dict[str, Any]
    created_at: str


def _utc_now_iso() -> str:
    """返回 UTC ISO-8601 时间。"""

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _normalize_required(
    value: Any,
    *,
    field_name: str,
) -> str:
    text = str(
        value
        or ""
    ).strip()

    if not text:
        raise ValueError(
            f"{field_name} 不能为空。"
        )

    return text


def _normalize_optional(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    return (
        text
        if text
        else None
    )


def _json_dumps(
    value: Any,
) -> str:
    """稳定地序列化 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )


def _json_loads_dict(
    value: str,
) -> dict[str, Any]:
    if not value:
        return {}

    parsed = json.loads(
        value
    )

    if isinstance(
        parsed,
        dict,
    ):
        return parsed

    return {
        "value": parsed
    }


class ConversationEventStore:
    """SQLite 原始对话事件归档。"""

    def __init__(
        self,
        database_path: Path | str = (
            DEFAULT_EVENT_STORE_PATH
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        ).resolve()

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = (
            threading.RLock()
        )

        self._connection = sqlite3.connect(
            str(
                self.database_path
            ),
            timeout=30.0,
            check_same_thread=False,
        )

        self._connection.row_factory = (
            sqlite3.Row
        )

        self._configure_connection()
        self.setup()

    def _configure_connection(
        self,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "PRAGMA journal_mode=WAL"
            )

            self._connection.execute(
                "PRAGMA synchronous=NORMAL"
            )

            self._connection.execute(
                "PRAGMA foreign_keys=ON"
            )

            self._connection.execute(
                "PRAGMA busy_timeout=30000"
            )

    def setup(
        self,
    ) -> None:
        """创建 Event Store Schema。"""

        schema = """
        CREATE TABLE IF NOT EXISTS conversation_threads (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS
        idx_conversation_threads_user_updated
        ON conversation_threads(
            user_id,
            updated_at DESC
        );

        CREATE TABLE IF NOT EXISTS conversation_events (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,

            event_id TEXT NOT NULL UNIQUE,

            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,

            sequence_no INTEGER NOT NULL,

            event_type TEXT NOT NULL,
            role TEXT NOT NULL,

            message_id TEXT,
            tool_call_id TEXT,
            tool_name TEXT,

            content_text TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,

            created_at TEXT NOT NULL,

            UNIQUE(
                thread_id,
                sequence_no
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_conversation_events_message
        ON conversation_events(
            thread_id,
            message_id
        )
        WHERE message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS
        idx_conversation_events_thread
        ON conversation_events(
            thread_id,
            sequence_no
        );

        CREATE INDEX IF NOT EXISTS
        idx_conversation_events_turn
        ON conversation_events(
            thread_id,
            turn_id,
            sequence_no
        );

        CREATE INDEX IF NOT EXISTS
        idx_conversation_events_tool_call
        ON conversation_events(
            thread_id,
            tool_call_id
        );

        CREATE INDEX IF NOT EXISTS
        idx_conversation_events_user
        ON conversation_events(
            user_id,
            created_at
        );

        CREATE TABLE IF NOT EXISTS agent_executions (
            execution_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            status TEXT NOT NULL,
            current_step TEXT NOT NULL,
            error_message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY(thread_id) REFERENCES conversation_threads(thread_id)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_executions_thread_status
        ON agent_executions(thread_id, status, updated_at DESC);

        CREATE TABLE IF NOT EXISTS agent_execution_events (
            execution_id TEXT NOT NULL,
            sequence_no INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(execution_id, sequence_no),
            FOREIGN KEY(execution_id) REFERENCES agent_executions(execution_id)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_execution_events_sequence
        ON agent_execution_events(execution_id, sequence_no);
        """

        with self._lock:
            self._connection.executescript(
                schema
            )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO conversation_threads(
                    thread_id, user_id, title, created_at, updated_at
                )
                SELECT
                    e.thread_id,
                    MIN(e.user_id),
                    COALESCE((
                        SELECT SUBSTR(TRIM(first_human.content_text), 1, 50)
                        FROM conversation_events first_human
                        WHERE first_human.thread_id = e.thread_id
                          AND first_human.role = 'human'
                          AND TRIM(first_human.content_text) <> ''
                        ORDER BY first_human.sequence_no ASC
                        LIMIT 1
                    ), '新会话'),
                    MIN(e.created_at),
                    MAX(e.created_at)
                FROM conversation_events e
                GROUP BY e.thread_id
                """
            )
            self._connection.commit()

    def ensure_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        title: str = "新会话",
    ) -> ConversationThread:
        """幂等创建会话；已存在的会话不会被改名或转移用户。"""

        normalized_user_id = _normalize_required(user_id, field_name="user_id")
        normalized_thread_id = _normalize_required(thread_id, field_name="thread_id")
        normalized_title = str(title or "新会话").strip() or "新会话"
        now = _utc_now_iso()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO conversation_threads(
                    thread_id, user_id, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_thread_id, normalized_user_id, normalized_title, now, now),
            )
            row = self._connection.execute(
                "SELECT * FROM conversation_threads WHERE thread_id = ?",
                (normalized_thread_id,),
            ).fetchone()
        return self._row_to_thread(row)

    def list_threads(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[ConversationThread]:
        normalized_user_id = _normalize_required(user_id, field_name="user_id")
        normalized_limit = int(limit)
        if normalized_limit <= 0:
            raise ValueError("limit 必须大于 0。")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    t.*,
                    COUNT(CASE WHEN e.role IN ('human', 'assistant') THEN 1 END)
                        AS message_count,
                    COALESCE((
                        SELECT content_text
                        FROM conversation_events latest
                        WHERE latest.thread_id = t.thread_id
                          AND latest.role IN ('human', 'assistant')
                        ORDER BY latest.sequence_no DESC
                        LIMIT 1
                    ), '') AS last_message_preview
                FROM conversation_threads t
                LEFT JOIN conversation_events e ON e.thread_id = t.thread_id
                WHERE t.user_id = ?
                GROUP BY t.thread_id
                ORDER BY t.updated_at DESC
                LIMIT ?
                """,
                (normalized_user_id, normalized_limit),
            ).fetchall()
        return [self._row_to_thread(row) for row in rows]

    def get_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> ConversationThread | None:
        normalized_user_id = _normalize_required(user_id, field_name="user_id")
        normalized_thread_id = _normalize_required(thread_id, field_name="thread_id")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM conversation_threads
                WHERE thread_id = ? AND user_id = ?
                """,
                (normalized_thread_id, normalized_user_id),
            ).fetchone()
        return self._row_to_thread(row) if row is not None else None

    def list_user_ids(self, *, limit: int = 100) -> list[str]:
        normalized_limit = int(limit)
        if normalized_limit <= 0:
            raise ValueError("limit 必须大于 0。")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT user_id, MAX(updated_at) AS latest
                FROM conversation_threads
                GROUP BY user_id
                ORDER BY latest DESC
                LIMIT ?
                """,
                (normalized_limit,),
            ).fetchall()
        return [str(row["user_id"]) for row in rows]

    def delete_thread(self, *, user_id: str, thread_id: str) -> bool:
        """删除指定用户会话的元数据和完整事件历史。"""

        normalized_user_id = _normalize_required(user_id, field_name="user_id")
        normalized_thread_id = _normalize_required(thread_id, field_name="thread_id")
        with self._lock, self._connection:
            owned = self._connection.execute(
                """
                SELECT 1 FROM conversation_threads
                WHERE thread_id = ? AND user_id = ?
                """,
                (normalized_thread_id, normalized_user_id),
            ).fetchone()
            if owned is None:
                return False
            self._connection.execute(
                "DELETE FROM conversation_events WHERE thread_id = ?",
                (normalized_thread_id,),
            )
            self._connection.execute(
                "DELETE FROM agent_execution_events WHERE execution_id IN "
                "(SELECT execution_id FROM agent_executions WHERE thread_id = ?)",
                (normalized_thread_id,),
            )
            self._connection.execute(
                "DELETE FROM agent_executions WHERE thread_id = ?",
                (normalized_thread_id,),
            )
            self._connection.execute(
                "DELETE FROM conversation_threads WHERE thread_id = ?",
                (normalized_thread_id,),
            )
        return True

    def start_execution(
        self,
        *,
        execution_id: str,
        user_id: str,
        thread_id: str,
    ) -> AgentExecution:
        """新建执行，或将 HITL 等待中的同一执行恢复为 RUNNING。"""

        execution_id = _normalize_required(execution_id, field_name="execution_id")
        user_id = _normalize_required(user_id, field_name="user_id")
        thread_id = _normalize_required(thread_id, field_name="thread_id")
        now = _utc_now_iso()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT user_id, thread_id, status FROM agent_executions WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if existing is not None and (
                str(existing["user_id"]) != user_id
                or str(existing["thread_id"]) != thread_id
            ):
                raise ValueError("execution_id 不属于当前用户或会话。")
            if existing is not None and str(existing["status"]) != "WAITING_HITL":
                raise ValueError("只有 WAITING_HITL 执行可以恢复。")
            self._connection.execute(
                """
                INSERT INTO agent_executions(
                    execution_id, user_id, thread_id, status, current_step,
                    error_message, created_at, started_at, updated_at, finished_at
                ) VALUES (?, ?, ?, 'RUNNING', 'runtime_started', '', ?, ?, ?, NULL)
                ON CONFLICT(execution_id) DO UPDATE SET
                    status = 'RUNNING',
                    current_step = 'runtime_resumed',
                    error_message = '',
                    updated_at = excluded.updated_at,
                    finished_at = NULL
                """,
                (execution_id, user_id, thread_id, now, now, now),
            )
            row = self._connection.execute(
                "SELECT * FROM agent_executions WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("Agent execution 创建后无法读取。")
        return self._row_to_execution(row)

    def update_execution(
        self,
        execution_id: str,
        *,
        status: str,
        current_step: str,
        error_message: str = "",
    ) -> AgentExecution:
        """更新执行状态；终态同时写入 finished_at。"""

        execution_id = _normalize_required(execution_id, field_name="execution_id")
        status = _normalize_required(status, field_name="status").upper()
        current_step = _normalize_required(current_step, field_name="current_step")
        now = _utc_now_iso()
        finished_at = now if status in {"SUCCEEDED", "FAILED", "CANCELLED"} else None
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE agent_executions
                SET status = ?, current_step = ?, error_message = ?,
                    updated_at = ?, finished_at = ?
                WHERE execution_id = ?
                """,
                (status, current_step, str(error_message or ""), now, finished_at, execution_id),
            )
            row = self._connection.execute(
                "SELECT * FROM agent_executions WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Agent execution 不存在：{execution_id}")
        return self._row_to_execution(row)

    def get_execution(self, execution_id: str) -> AgentExecution | None:
        execution_id = _normalize_required(execution_id, field_name="execution_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agent_executions WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        return self._row_to_execution(row) if row is not None else None

    def get_active_execution(self, *, thread_id: str) -> AgentExecution | None:
        """返回线程最新的运行中或等待审批执行。"""

        thread_id = _normalize_required(thread_id, field_name="thread_id")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM agent_executions
                WHERE thread_id = ? AND status IN (
                    'RUNNING', 'CANCELLING', 'WAITING_HITL'
                )
                ORDER BY updated_at DESC LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        return self._row_to_execution(row) if row is not None else None

    def get_latest_execution(self, *, thread_id: str) -> AgentExecution | None:
        """返回线程最新执行，包括成功、失败和取消等终态。"""

        thread_id = _normalize_required(thread_id, field_name="thread_id")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM agent_executions
                WHERE thread_id = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        return self._row_to_execution(row) if row is not None else None

    def get_last_execution_event_sequence(self, execution_id: str) -> int:
        execution_id = _normalize_required(execution_id, field_name="execution_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) AS value "
                "FROM agent_execution_events WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        return int(row["value"] if row is not None else 0)

    def append_execution_event(
        self,
        *,
        execution_id: str,
        sequence_no: int,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentExecutionEvent:
        """幂等保存 SSE 可观察事件，供刷新后的页面按序号续读。"""

        execution_id = _normalize_required(execution_id, field_name="execution_id")
        event_type = _normalize_required(event_type, field_name="event_type")
        sequence_no = int(sequence_no)
        if sequence_no <= 0:
            raise ValueError("sequence_no 必须大于 0。")
        created_at = _utc_now_iso()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO agent_execution_events(
                    execution_id, sequence_no, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (execution_id, sequence_no, event_type, payload_json, created_at),
            )
            row = self._connection.execute(
                "SELECT * FROM agent_execution_events "
                "WHERE execution_id = ? AND sequence_no = ?",
                (execution_id, sequence_no),
            ).fetchone()
        if row is None:
            raise RuntimeError("Agent execution event 写入后无法读取。")
        return self._row_to_execution_event(row)

    def append_next_execution_event(
        self,
        *,
        execution_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentExecutionEvent:
        """在同一数据库锁内分配下一序号并写入审计事件。"""

        execution_id = _normalize_required(execution_id, field_name="execution_id")
        event_type = _normalize_required(event_type, field_name="event_type")
        created_at = _utc_now_iso()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS value "
                "FROM agent_execution_events WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            sequence_no = int(row["value"] if row is not None else 1)
            self._connection.execute(
                """
                INSERT INTO agent_execution_events(
                    execution_id, sequence_no, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (execution_id, sequence_no, event_type, payload_json, created_at),
            )
            stored = self._connection.execute(
                "SELECT * FROM agent_execution_events "
                "WHERE execution_id = ? AND sequence_no = ?",
                (execution_id, sequence_no),
            ).fetchone()
        if stored is None:
            raise RuntimeError("Agent execution audit event 写入后无法读取。")
        return self._row_to_execution_event(stored)

    def list_execution_events(
        self,
        *,
        execution_id: str,
        after_sequence: int = 0,
        limit: int = 200,
    ) -> list[AgentExecutionEvent]:
        execution_id = _normalize_required(execution_id, field_name="execution_id")
        after_sequence = max(0, int(after_sequence))
        limit = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM agent_execution_events
                WHERE execution_id = ? AND sequence_no > ?
                ORDER BY sequence_no ASC LIMIT ?
                """,
                (execution_id, after_sequence, limit),
            ).fetchall()
        return [self._row_to_execution_event(row) for row in rows]

    @staticmethod
    def _row_to_execution(row: sqlite3.Row) -> AgentExecution:
        return AgentExecution(
            execution_id=str(row["execution_id"]),
            user_id=str(row["user_id"]),
            thread_id=str(row["thread_id"]),
            status=str(row["status"]),
            current_step=str(row["current_step"]),
            error_message=str(row["error_message"]),
            created_at=str(row["created_at"]),
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
            finished_at=(str(row["finished_at"]) if row["finished_at"] is not None else None),
        )

    @staticmethod
    def _row_to_execution_event(row: sqlite3.Row) -> AgentExecutionEvent:
        return AgentExecutionEvent(
            execution_id=str(row["execution_id"]),
            sequence_no=int(row["sequence_no"]),
            event_type=str(row["event_type"]),
            payload=json.loads(str(row["payload_json"] or "{}")),
            created_at=str(row["created_at"]),
        )

    def append_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        turn_id: str,
        event_type: str,
        role: str,
        content_text: str = "",
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> tuple[
        ConversationEvent,
        bool,
    ]:
        """幂等写入一条事件。

        返回：
            (event, inserted)

        inserted=True:
            本次新写入。

        inserted=False:
            同一 event_id 或同一 thread/message_id
            已经存在，返回已有事件。

        设计目的：
            后续 Agent 可以在每轮结束或 HITL resume 后，
            对当前可见 messages 再归档一次，
            不必担心重复写入。
        """

        normalized_user_id = (
            _normalize_required(
                user_id,
                field_name="user_id",
            )
        )

        normalized_thread_id = (
            _normalize_required(
                thread_id,
                field_name="thread_id",
            )
        )

        normalized_turn_id = (
            _normalize_required(
                turn_id,
                field_name="turn_id",
            )
        )

        normalized_event_type = (
            _normalize_required(
                event_type,
                field_name="event_type",
            )
        )

        normalized_role = (
            _normalize_required(
                role,
                field_name="role",
            )
        )

        normalized_message_id = (
            _normalize_optional(
                message_id
            )
        )

        normalized_tool_call_id = (
            _normalize_optional(
                tool_call_id
            )
        )

        normalized_tool_name = (
            _normalize_optional(
                tool_name
            )
        )

        normalized_event_id = (
            _normalize_optional(
                event_id
            )
        )

        if normalized_event_id is None:
            if normalized_message_id:
                normalized_event_id = (
                    "msg:"
                    + normalized_thread_id
                    + ":"
                    + normalized_message_id
                )
            else:
                normalized_event_id = (
                    "evt:"
                    + uuid.uuid4().hex
                )

        normalized_created_at = (
            _normalize_optional(
                created_at
            )
            or _utc_now_iso()
        )

        normalized_payload = dict(
            payload
            or {}
        )

        normalized_metadata = dict(
            metadata
            or {}
        )

        normalized_content_text = str(
            content_text
            or ""
        )

        with self._lock:
            self._connection.execute(
                "BEGIN IMMEDIATE"
            )

            try:
                existing = (
                    self._find_existing_locked(
                        thread_id=(
                            normalized_thread_id
                        ),
                        event_id=(
                            normalized_event_id
                        ),
                        message_id=(
                            normalized_message_id
                        ),
                    )
                )

                if existing is not None:
                    self._connection.commit()

                    return (
                        self._row_to_event(
                            existing
                        ),
                        False,
                    )

                title = "新会话"
                if normalized_role == "human" and normalized_content_text.strip():
                    title = " ".join(normalized_content_text.split())[:50]

                self._connection.execute(
                    """
                    INSERT INTO conversation_threads(
                        thread_id, user_id, title, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        title = CASE
                            WHEN conversation_threads.title = '新会话'
                                 AND excluded.title <> '新会话'
                            THEN excluded.title
                            ELSE conversation_threads.title
                        END
                    """,
                    (
                        normalized_thread_id,
                        normalized_user_id,
                        title,
                        normalized_created_at,
                        normalized_created_at,
                    ),
                )

                sequence_no = int(
                    self._connection.execute(
                        """
                        SELECT COALESCE(
                            MAX(sequence_no),
                            0
                        ) + 1
                        FROM conversation_events
                        WHERE thread_id = ?
                        """,
                        (
                            normalized_thread_id,
                        ),
                    )
                    .fetchone()[0]
                )

                self._connection.execute(
                    """
                    INSERT INTO conversation_events (
                        event_id,
                        user_id,
                        thread_id,
                        turn_id,
                        sequence_no,
                        event_type,
                        role,
                        message_id,
                        tool_call_id,
                        tool_name,
                        content_text,
                        payload_json,
                        metadata_json,
                        created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    """,
                    (
                        normalized_event_id,
                        normalized_user_id,
                        normalized_thread_id,
                        normalized_turn_id,
                        sequence_no,
                        normalized_event_type,
                        normalized_role,
                        normalized_message_id,
                        normalized_tool_call_id,
                        normalized_tool_name,
                        normalized_content_text,
                        _json_dumps(
                            normalized_payload
                        ),
                        _json_dumps(
                            normalized_metadata
                        ),
                        normalized_created_at,
                    ),
                )

                row = self._connection.execute(
                    """
                    SELECT *
                    FROM conversation_events
                    WHERE event_id = ?
                    """,
                    (
                        normalized_event_id,
                    ),
                ).fetchone()

                self._connection.commit()

            except Exception:
                self._connection.rollback()
                raise

        if row is None:
            raise RuntimeError(
                "事件写入成功后无法重新读取。"
            )

        return (
            self._row_to_event(
                row
            ),
            True,
        )

    def _find_existing_locked(
        self,
        *,
        thread_id: str,
        event_id: str,
        message_id: str | None,
    ) -> sqlite3.Row | None:
        row = self._connection.execute(
            """
            SELECT *
            FROM conversation_events
            WHERE event_id = ?
            """,
            (
                event_id,
            ),
        ).fetchone()

        if row is not None:
            return row

        if message_id:
            row = self._connection.execute(
                """
                SELECT *
                FROM conversation_events
                WHERE thread_id = ?
                  AND message_id = ?
                """,
                (
                    thread_id,
                    message_id,
                ),
            ).fetchone()

        return row

    def get_event(
        self,
        event_id: str,
    ) -> ConversationEvent | None:
        normalized_event_id = (
            _normalize_required(
                event_id,
                field_name="event_id",
            )
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM conversation_events
                WHERE event_id = ?
                """,
                (
                    normalized_event_id,
                ),
            ).fetchone()

        return (
            self._row_to_event(
                row
            )
            if row is not None
            else None
        )

    def list_thread_events(
        self,
        thread_id: str,
        *,
        limit: int | None = None,
    ) -> list[
        ConversationEvent
    ]:
        normalized_thread_id = (
            _normalize_required(
                thread_id,
                field_name="thread_id",
            )
        )

        sql = """
        SELECT *
        FROM conversation_events
        WHERE thread_id = ?
        ORDER BY sequence_no ASC
        """

        parameters: list[Any] = [
            normalized_thread_id
        ]

        if limit is not None:
            normalized_limit = int(
                limit
            )

            if normalized_limit <= 0:
                raise ValueError(
                    "limit 必须大于 0。"
                )

            sql += " LIMIT ?"

            parameters.append(
                normalized_limit
            )

        with self._lock:
            rows = self._connection.execute(
                sql,
                tuple(
                    parameters
                ),
            ).fetchall()

        return [
            self._row_to_event(
                row
            )
            for row in rows
        ]

    def list_turn_events(
        self,
        *,
        thread_id: str,
        turn_id: str,
    ) -> list[
        ConversationEvent
    ]:
        normalized_thread_id = (
            _normalize_required(
                thread_id,
                field_name="thread_id",
            )
        )

        normalized_turn_id = (
            _normalize_required(
                turn_id,
                field_name="turn_id",
            )
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM conversation_events
                WHERE thread_id = ?
                  AND turn_id = ?
                ORDER BY sequence_no ASC
                """,
                (
                    normalized_thread_id,
                    normalized_turn_id,
                ),
            ).fetchall()

        return [
            self._row_to_event(
                row
            )
            for row in rows
        ]

    def get_tool_evidence(
        self,
        *,
        thread_id: str,
        turn_id: str | None = None,
    ) -> list[
        ConversationEvent
    ]:
        """读取 Tool 原始结果，不做摘要或截断。"""

        normalized_thread_id = (
            _normalize_required(
                thread_id,
                field_name="thread_id",
            )
        )

        parameters: list[Any] = [
            normalized_thread_id
        ]

        sql = """
        SELECT *
        FROM conversation_events
        WHERE thread_id = ?
          AND role = 'tool'
        """

        if turn_id is not None:
            normalized_turn_id = (
                _normalize_required(
                    turn_id,
                    field_name="turn_id",
                )
            )

            sql += """
              AND turn_id = ?
            """

            parameters.append(
                normalized_turn_id
            )

        sql += """
        ORDER BY sequence_no ASC
        """

        with self._lock:
            rows = self._connection.execute(
                sql,
                tuple(
                    parameters
                ),
            ).fetchall()

        return [
            self._row_to_event(
                row
            )
            for row in rows
        ]

    def search_thread_events(
        self,
        *,
        thread_id: str,
        query: str,
        limit: int = 20,
    ) -> list[
        ConversationEvent
    ]:
        """Phase 3 的最小历史查找能力。

        目前仅使用 SQLite LIKE，
        后续 Conversation Retriever 会替换成更正式的
        lexical / vector / hybrid retrieval。

        这里的目标只是证明：
        原始内容可以被长期保存并再次找到。
        """

        normalized_thread_id = (
            _normalize_required(
                thread_id,
                field_name="thread_id",
            )
        )

        normalized_query = (
            _normalize_required(
                query,
                field_name="query",
            )
        )

        normalized_limit = int(
            limit
        )

        if normalized_limit <= 0:
            raise ValueError(
                "limit 必须大于 0。"
            )

        pattern = (
            "%"
            + normalized_query
            + "%"
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM conversation_events
                WHERE thread_id = ?
                  AND content_text LIKE ?
                ORDER BY sequence_no ASC
                LIMIT ?
                """,
                (
                    normalized_thread_id,
                    pattern,
                    normalized_limit,
                ),
            ).fetchall()

        return [
            self._row_to_event(
                row
            )
            for row in rows
        ]

    def count_events(
        self,
        *,
        thread_id: str | None = None,
    ) -> int:
        if thread_id is None:
            sql = """
            SELECT COUNT(*)
            FROM conversation_events
            """

            parameters: tuple[Any, ...] = ()

        else:
            normalized_thread_id = (
                _normalize_required(
                    thread_id,
                    field_name="thread_id",
                )
            )

            sql = """
            SELECT COUNT(*)
            FROM conversation_events
            WHERE thread_id = ?
            """

            parameters = (
                normalized_thread_id,
            )

        with self._lock:
            row = self._connection.execute(
                sql,
                parameters,
            ).fetchone()

        return int(
            row[0]
        )

    def list_turn_ids(
        self,
        *,
        thread_id: str,
    ) -> list[str]:
        normalized_thread_id = (
            _normalize_required(
                thread_id,
                field_name="thread_id",
            )
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    turn_id,
                    MIN(sequence_no) AS first_sequence
                FROM conversation_events
                WHERE thread_id = ?
                GROUP BY turn_id
                ORDER BY first_sequence ASC
                """,
                (
                    normalized_thread_id,
                ),
            ).fetchall()

        return [
            str(
                row["turn_id"]
            )
            for row in rows
        ]

    def close(
        self,
    ) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(
        self,
    ) -> "ConversationEventStore":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.close()

    @staticmethod
    def _row_to_thread(row: sqlite3.Row) -> ConversationThread:
        keys = set(row.keys())
        return ConversationThread(
            thread_id=str(row["thread_id"]),
            user_id=str(row["user_id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            message_count=int(row["message_count"]) if "message_count" in keys else 0,
            last_message_preview=(
                str(row["last_message_preview"])
                if "last_message_preview" in keys
                else ""
            ),
        )

    @staticmethod
    def _row_to_event(
        row: sqlite3.Row,
    ) -> ConversationEvent:
        return ConversationEvent(
            event_id=str(
                row["event_id"]
            ),
            user_id=str(
                row["user_id"]
            ),
            thread_id=str(
                row["thread_id"]
            ),
            turn_id=str(
                row["turn_id"]
            ),
            sequence_no=int(
                row["sequence_no"]
            ),
            event_type=str(
                row["event_type"]
            ),
            role=str(
                row["role"]
            ),
            message_id=(
                str(
                    row["message_id"]
                )
                if row["message_id"]
                is not None
                else None
            ),
            tool_call_id=(
                str(
                    row["tool_call_id"]
                )
                if row["tool_call_id"]
                is not None
                else None
            ),
            tool_name=(
                str(
                    row["tool_name"]
                )
                if row["tool_name"]
                is not None
                else None
            ),
            content_text=str(
                row["content_text"]
            ),
            payload=(
                _json_loads_dict(
                    str(
                        row[
                            "payload_json"
                        ]
                    )
                )
            ),
            metadata=(
                _json_loads_dict(
                    str(
                        row[
                            "metadata_json"
                        ]
                    )
                )
            ),
            created_at=str(
                row["created_at"]
            ),
        )
