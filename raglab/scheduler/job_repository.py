"""SQLite repository for Scheduler Job / Job Run."""

from __future__ import annotations

import sqlite3
import uuid

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

from raglab.scheduler.job import (
    JobRunStatus,
    JobTriggerType,
    MisfirePolicy,
    ScheduledJob,
    ScheduledJobRun,
    ScheduleType,
)


DEFAULT_DATABASE_PATH = Path(
    "storage/control_plane/"
    "raglab_control.sqlite3"
)


def utc_now_text() -> str:
    """统一使用 UTC ISO 8601 字符串保存时间。"""

    return datetime.now(
        timezone.utc
    ).isoformat()


class ScheduledJobRepository:
    """Scheduler 持久化仓库。"""

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_DATABASE_PATH
        ),
    ) -> None:

        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Connection
    # ========================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:

        connection = sqlite3.connect(
            self.database_path,
            timeout=5.0,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        # 两个进程同时争抢 SQLite 写事务时，
        # 最多等待 5 秒，而不是立即失败。
        connection.execute(
            "PRAGMA busy_timeout = 5000"
        )

        return connection

    # ========================================================
    # Setup
    # ========================================================

    def setup(
        self,
    ) -> None:

        with self._connect() as conn:

            # ------------------------------------------------
            # Job Definition
            # ------------------------------------------------

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS
                scheduled_job (
                    job_id TEXT PRIMARY KEY,

                    job_name TEXT NOT NULL UNIQUE,

                    enabled INTEGER NOT NULL,

                    schedule_type TEXT NOT NULL,

                    schedule_expression TEXT NOT NULL,

                    timezone TEXT NOT NULL,

                    misfire_policy TEXT NOT NULL,

                    requires_start_approval
                        INTEGER NOT NULL,

                    last_scheduled_at TEXT,

                    next_run_at TEXT,

                    created_at TEXT NOT NULL,

                    updated_at TEXT NOT NULL
                )
                """
            )

            # ------------------------------------------------
            # Job Run
            # ------------------------------------------------

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS
                scheduled_job_run (
                    run_id TEXT PRIMARY KEY,

                    job_id TEXT NOT NULL,

                    job_name TEXT NOT NULL,

                    trigger_type TEXT NOT NULL,

                    scheduled_at TEXT,

                    status TEXT NOT NULL,

                    agent_thread_id TEXT,

                    requested_at TEXT NOT NULL,

                    approved_at TEXT,

                    started_at TEXT,

                    finished_at TEXT,

                    approval_actor TEXT,

                    approval_reason TEXT,

                    result_summary TEXT,

                    error_message TEXT,

                    created_at TEXT NOT NULL,

                    updated_at TEXT NOT NULL,

                    FOREIGN KEY(job_id)
                        REFERENCES
                        scheduled_job(job_id)
                )
                """
            )

            # ------------------------------------------------
            # Migration
            #
            # 如果之前已经创建过 scheduled_job_run，
            # CREATE TABLE IF NOT EXISTS 不会自动增加
            # agent_thread_id。
            # ------------------------------------------------

            columns = {

                str(
                    row["name"]
                )

                for row
                in conn.execute(
                    """
                    PRAGMA table_info(
                        scheduled_job_run
                    )
                    """
                ).fetchall()
            }

            if (
                "agent_thread_id"
                not in columns
            ):

                conn.execute(
                    """
                    ALTER TABLE
                    scheduled_job_run

                    ADD COLUMN
                    agent_thread_id TEXT
                    """
                )

            # ------------------------------------------------
            # Indexes
            # ------------------------------------------------

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_scheduled_job_next_run

                ON scheduled_job (
                    enabled,
                    next_run_at
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_scheduled_job_run_status

                ON scheduled_job_run (
                    job_name,
                    status
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_scheduled_job_run_requested

                ON scheduled_job_run (
                    requested_at
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_scheduled_job_run_thread

                ON scheduled_job_run (
                    agent_thread_id
                )
                """
            )

    # ========================================================
    # Job
    # ========================================================

    def upsert_job(
        self,
        *,
        job_name: str,
        schedule_type: ScheduleType,
        schedule_expression: str,
        timezone_name: str,
        misfire_policy: MisfirePolicy,
        requires_start_approval: bool,
        enabled: bool = True,
        next_run_at: str | None = None,
    ) -> ScheduledJob:

        normalized_name = str(
            job_name
        ).strip()

        if not normalized_name:

            raise ValueError(
                "job_name 不能为空。"
            )

        normalized_expression = str(
            schedule_expression
        ).strip()

        if not normalized_expression:

            raise ValueError(
                "schedule_expression 不能为空。"
            )

        normalized_timezone = str(
            timezone_name
        ).strip()

        if not normalized_timezone:

            raise ValueError(
                "timezone 不能为空。"
            )

        now = utc_now_text()

        with self._connect() as conn:

            existing = conn.execute(
                """
                SELECT job_id
                FROM scheduled_job
                WHERE job_name = ?
                """,
                (
                    normalized_name,
                ),
            ).fetchone()

            # ------------------------------------------------
            # Create
            # ------------------------------------------------

            if existing is None:

                job_id = (
                    "job_"
                    + uuid.uuid4().hex
                )

                conn.execute(
                    """
                    INSERT INTO scheduled_job (
                        job_id,
                        job_name,
                        enabled,
                        schedule_type,
                        schedule_expression,
                        timezone,
                        misfire_policy,
                        requires_start_approval,
                        last_scheduled_at,
                        next_run_at,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        NULL,
                        ?, ?, ?
                    )
                    """,
                    (
                        job_id,

                        normalized_name,

                        int(
                            enabled
                        ),

                        schedule_type.value,

                        normalized_expression,

                        normalized_timezone,

                        misfire_policy.value,

                        int(
                            requires_start_approval
                        ),

                        next_run_at,

                        now,

                        now,
                    ),
                )

            # ------------------------------------------------
            # Update
            # ------------------------------------------------

            else:

                job_id = str(
                    existing[
                        "job_id"
                    ]
                )

                conn.execute(
                    """
                    UPDATE scheduled_job

                    SET
                        enabled = ?,
                        schedule_type = ?,
                        schedule_expression = ?,
                        timezone = ?,
                        misfire_policy = ?,
                        requires_start_approval = ?,
                        next_run_at = ?,
                        updated_at = ?

                    WHERE job_id = ?
                    """,
                    (
                        int(
                            enabled
                        ),

                        schedule_type.value,

                        normalized_expression,

                        normalized_timezone,

                        misfire_policy.value,

                        int(
                            requires_start_approval
                        ),

                        next_run_at,

                        now,

                        job_id,
                    ),
                )

        result = self.get_job(
            normalized_name
        )

        if result is None:

            raise RuntimeError(
                "保存 Scheduled Job 后"
                "无法重新读取。"
            )

        return result

    def get_job(
        self,
        job_name: str,
    ) -> ScheduledJob | None:

        normalized_name = str(
            job_name
        ).strip()

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT *
                FROM scheduled_job
                WHERE job_name = ?
                """,
                (
                    normalized_name,
                ),
            ).fetchone()

        if row is None:

            return None

        return self._row_to_job(
            row
        )

    def list_jobs(
        self,
        *,
        enabled_only: bool = False,
    ) -> list[
        ScheduledJob
    ]:

        sql = """
            SELECT *
            FROM scheduled_job
        """

        if enabled_only:

            sql += """
                WHERE enabled = 1
            """

        sql += """
            ORDER BY job_name
        """

        with self._connect() as conn:

            rows = conn.execute(
                sql
            ).fetchall()

        return [

            self._row_to_job(
                row
            )

            for row
            in rows
        ]

    def update_schedule_cursor(
        self,
        *,
        job_name: str,
        last_scheduled_at: str | None,
        next_run_at: str | None,
    ) -> None:

        now = utc_now_text()

        with self._connect() as conn:

            conn.execute(
                """
                UPDATE scheduled_job

                SET
                    last_scheduled_at = ?,
                    next_run_at = ?,
                    updated_at = ?

                WHERE job_name = ?
                """,
                (
                    last_scheduled_at,

                    next_run_at,

                    now,

                    str(
                        job_name
                    ).strip(),
                ),
            )

    # ========================================================
    # Job Run - Create / Read
    # ========================================================

    def create_run(
        self,
        *,
        job: ScheduledJob,
        trigger_type: JobTriggerType,
        scheduled_at: str | None,
        status: JobRunStatus,
    ) -> ScheduledJobRun:

        now = utc_now_text()

        run_id = (
            "run_"
            + uuid.uuid4().hex
        )

        with self._connect() as conn:

            conn.execute(
                """
                INSERT INTO
                scheduled_job_run (
                    run_id,
                    job_id,
                    job_name,
                    trigger_type,
                    scheduled_at,
                    status,
                    agent_thread_id,
                    requested_at,
                    approved_at,
                    started_at,
                    finished_at,
                    approval_actor,
                    approval_reason,
                    result_summary,
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?,
                    NULL,
                    ?,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    ?, ?
                )
                """,
                (
                    run_id,

                    job.job_id,

                    job.job_name,

                    trigger_type.value,

                    scheduled_at,

                    status.value,

                    now,

                    now,

                    now,
                ),
            )

        result = self.get_run(
            run_id
        )

        if result is None:

            raise RuntimeError(
                "创建 Job Run 后"
                "无法重新读取。"
            )

        return result

    def get_run(
        self,
        run_id: str,
    ) -> ScheduledJobRun | None:

        normalized_run_id = str(
            run_id
        ).strip()

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT *
                FROM scheduled_job_run
                WHERE run_id = ?
                """,
                (
                    normalized_run_id,
                ),
            ).fetchone()

        if row is None:

            return None

        return self._row_to_run(
            row
        )

    def list_runs(
        self,
        *,
        job_name: str | None = None,
        status: JobRunStatus | None = None,
        limit: int = 50,
    ) -> list[
        ScheduledJobRun
    ]:

        if int(
            limit
        ) <= 0:

            raise ValueError(
                "limit 必须大于 0。"
            )

        conditions: list[
            str
        ] = []

        parameters: list[
            object
        ] = []

        if job_name is not None:

            conditions.append(
                "job_name = ?"
            )

            parameters.append(
                str(
                    job_name
                ).strip()
            )

        if status is not None:

            conditions.append(
                "status = ?"
            )

            parameters.append(
                status.value
            )

        sql = """
            SELECT *
            FROM scheduled_job_run
        """

        if conditions:

            sql += (
                " WHERE "
                + " AND ".join(
                    conditions
                )
            )

        sql += """
            ORDER BY requested_at DESC
            LIMIT ?
        """

        parameters.append(
            int(
                limit
            )
        )

        with self._connect() as conn:

            rows = conn.execute(
                sql,
                tuple(
                    parameters
                ),
            ).fetchall()

        return [

            self._row_to_run(
                row
            )

            for row
            in rows
        ]

    def list_waiting_approval_runs(
        self,
    ) -> list[
        ScheduledJobRun
    ]:

        return self.list_runs(
            status=(
                JobRunStatus
                .WAITING_APPROVAL
            ),
            limit=100,
        )

    # ========================================================
    # Job Run - Agent Thread Mapping
    # ========================================================

    def bind_agent_thread(
        self,
        *,
        run_id: str,
        agent_thread_id: str,
    ) -> ScheduledJobRun:
        """给 Job Run 绑定固定 LangGraph Thread。

        同一个 run_id 只能绑定一个 agent_thread_id。

        重复绑定相同值属于幂等操作。
        """

        normalized_run_id = str(
            run_id
        ).strip()

        normalized_thread_id = str(
            agent_thread_id
        ).strip()

        if not normalized_run_id:

            raise ValueError(
                "run_id 不能为空。"
            )

        if not normalized_thread_id:

            raise ValueError(
                "agent_thread_id 不能为空。"
            )

        now = utc_now_text()

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT agent_thread_id

                FROM scheduled_job_run

                WHERE run_id = ?
                """,
                (
                    normalized_run_id,
                ),
            ).fetchone()

            if row is None:

                raise KeyError(
                    f"Job Run 不存在："
                    f"{normalized_run_id}"
                )

            existing = row[
                "agent_thread_id"
            ]

            # ------------------------------------------------
            # 已经绑定。
            # ------------------------------------------------

            if (
                existing is not None
                and str(
                    existing
                ).strip()
            ):

                if (
                    str(
                        existing
                    ).strip()
                    != normalized_thread_id
                ):

                    raise RuntimeError(
                        "Job Run 已经绑定另一个 "
                        "Agent Thread："
                        f"{existing}"
                    )

            # ------------------------------------------------
            # 第一次绑定。
            # ------------------------------------------------

            else:

                conn.execute(
                    """
                    UPDATE scheduled_job_run

                    SET
                        agent_thread_id = ?,
                        updated_at = ?

                    WHERE run_id = ?
                    """,
                    (
                        normalized_thread_id,

                        now,

                        normalized_run_id,
                    ),
                )

        result = self.get_run(
            normalized_run_id
        )

        if result is None:

            raise RuntimeError(
                "绑定 Agent Thread 后"
                "无法重新读取 Job Run。"
            )

        return result

    # ========================================================
    # Job Run - Single-Flight
    # ========================================================

    def approve_and_acquire(
        self,
        *,
        run_id: str,
        actor: str,
        reason: str | None = None,
    ) -> tuple[
        ScheduledJobRun,
        str | None,
    ]:
        """批准 Job，并原子地尝试获得 Single-Flight。

        返回：

            (
                当前 Run,
                duplicate_run_id
            )

        duplicate_run_id == None：
            当前任务成功进入 RUNNING。

        duplicate_run_id != None：
            已有同类型 Active Run，
            当前任务变为 SKIPPED_DUPLICATE。
        """

        normalized_run_id = str(
            run_id
        ).strip()

        normalized_actor = str(
            actor
        ).strip()

        if not normalized_run_id:

            raise ValueError(
                "run_id 不能为空。"
            )

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        now = utc_now_text()

        duplicate_run_id: (
            str
            | None
        ) = None

        conn = self._connect()

        try:

            # ------------------------------------------------
            # 短期数据库互斥。
            #
            # “检查有没有 Active Run”
            #
            # 和
            #
            # “把自己设置为 RUNNING”
            #
            # 必须位于同一个 SQLite 写事务。
            # ------------------------------------------------

            conn.execute(
                "BEGIN IMMEDIATE"
            )

            row = conn.execute(
                """
                SELECT *

                FROM scheduled_job_run

                WHERE run_id = ?
                """,
                (
                    normalized_run_id,
                ),
            ).fetchone()

            if row is None:

                raise KeyError(
                    f"Job Run 不存在："
                    f"{normalized_run_id}"
                )

            current_status = (
                JobRunStatus(
                    row[
                        "status"
                    ]
                )
            )

            if (
                current_status
                !=
                JobRunStatus
                .WAITING_APPROVAL
            ):

                raise RuntimeError(
                    "只有 WAITING_APPROVAL "
                    "状态才能批准执行。"
                    f"当前状态="
                    f"{current_status.value}"
                )

            job_name = str(
                row[
                    "job_name"
                ]
            )

            # ------------------------------------------------
            # RUNNING：
            #     正在实际运行。
            #
            # WAITING_TOOL_APPROVAL：
            #     Agent 暂停等待 Tool HITL，
            #     但 Job 尚未结束。
            #
            # 两种状态都长期占有 Single-Flight。
            # ------------------------------------------------

            active_row = conn.execute(
                """
                SELECT run_id

                FROM scheduled_job_run

                WHERE job_name = ?

                  AND run_id <> ?

                  AND status IN (
                      'RUNNING',
                      'WAITING_TOOL_APPROVAL'
                  )

                ORDER BY
                    requested_at ASC

                LIMIT 1
                """,
                (
                    job_name,

                    normalized_run_id,
                ),
            ).fetchone()

            # ------------------------------------------------
            # 已经存在 Active Run。
            # ------------------------------------------------

            if active_row is not None:

                duplicate_run_id = str(
                    active_row[
                        "run_id"
                    ]
                )

                conn.execute(
                    """
                    UPDATE scheduled_job_run

                    SET
                        status = ?,
                        approved_at = ?,
                        approval_actor = ?,
                        approval_reason = ?,
                        finished_at = ?,
                        result_summary = ?,
                        updated_at = ?

                    WHERE run_id = ?
                    """,
                    (
                        JobRunStatus
                        .SKIPPED_DUPLICATE
                        .value,

                        now,

                        normalized_actor,

                        reason,

                        now,

                        (
                            "已有同类型 Active Run："
                            + duplicate_run_id
                        ),

                        now,

                        normalized_run_id,
                    ),
                )

            # ------------------------------------------------
            # 没有 Active Run。
            #
            # 当前任务获得长期执行资格。
            # ------------------------------------------------

            else:

                conn.execute(
                    """
                    UPDATE scheduled_job_run

                    SET
                        status = ?,
                        approved_at = ?,
                        approval_actor = ?,
                        approval_reason = ?,
                        started_at = ?,
                        updated_at = ?

                    WHERE run_id = ?
                    """,
                    (
                        JobRunStatus
                        .RUNNING
                        .value,

                        now,

                        normalized_actor,

                        reason,

                        now,

                        now,

                        normalized_run_id,
                    ),
                )

            conn.commit()

        except Exception:

            conn.rollback()

            raise

        finally:

            conn.close()

        result = self.get_run(
            normalized_run_id
        )

        if result is None:

            raise RuntimeError(
                "批准 Job Run 后"
                "无法重新读取。"
            )

        return (
            result,
            duplicate_run_id,
        )

    # ========================================================
    # Job Run - HITL State
    # ========================================================

    def mark_waiting_tool_approval(
        self,
        run_id: str,
    ) -> ScheduledJobRun:
        """Agent 内部出现 Tool HITL interrupt。"""

        return self._transition_run(

            run_id=run_id,

            expected_statuses={
                JobRunStatus.RUNNING,
            },

            new_status=(
                JobRunStatus
                .WAITING_TOOL_APPROVAL
            ),
        )

    def resume_from_tool_approval(
        self,
        run_id: str,
    ) -> ScheduledJobRun:
        """准备恢复 Tool HITL。

        Job 先从 WAITING_TOOL_APPROVAL
        切回 RUNNING。

        然后 Agent Runtime Controller
        才真正进行 Command(resume=...)。
        """

        return self._transition_run(

            run_id=run_id,

            expected_statuses={
                JobRunStatus
                .WAITING_TOOL_APPROVAL,
            },

            new_status=(
                JobRunStatus.RUNNING
            ),
        )

    # ========================================================
    # Job Run - Finish
    # ========================================================

    def mark_succeeded(
        self,
        *,
        run_id: str,
        result_summary: str | None = None,
    ) -> ScheduledJobRun:

        return self._finish_run(

            run_id=run_id,

            expected_statuses={
                JobRunStatus.RUNNING,
            },

            new_status=(
                JobRunStatus.SUCCEEDED
            ),

            result_summary=(
                result_summary
            ),

            error_message=None,
        )

    def mark_failed(
        self,
        *,
        run_id: str,
        error_message: str,
    ) -> ScheduledJobRun:

        return self._finish_run(

            run_id=run_id,

            expected_statuses={
                JobRunStatus.RUNNING,
            },

            new_status=(
                JobRunStatus.FAILED
            ),

            result_summary=None,

            error_message=str(
                error_message
            ),
        )

    def mark_runtime_canceled(
        self,
        *,
        run_id: str,
        result_summary: str,
    ) -> ScheduledJobRun:
        """Workflow 已经启动，但后续被人工取消。"""

        return self._finish_run(

            run_id=run_id,

            expected_statuses={
                JobRunStatus.RUNNING,
            },

            new_status=(
                JobRunStatus.CANCELED
            ),

            result_summary=(
                result_summary
            ),

            error_message=None,
        )

    def cancel_run(
        self,
        *,
        run_id: str,
        actor: str,
        reason: str,
    ) -> ScheduledJobRun:
        """在 Job Start Approval 阶段拒绝任务。"""

        normalized_run_id = str(
            run_id
        ).strip()

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
                "取消 Job 必须填写原因。"
            )

        now = utc_now_text()

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT status

                FROM scheduled_job_run

                WHERE run_id = ?
                """,
                (
                    normalized_run_id,
                ),
            ).fetchone()

            if row is None:

                raise KeyError(
                    f"Job Run 不存在："
                    f"{normalized_run_id}"
                )

            current_status = (
                JobRunStatus(
                    row[
                        "status"
                    ]
                )
            )

            if (
                current_status
                !=
                JobRunStatus
                .WAITING_APPROVAL
            ):

                raise RuntimeError(
                    "只有 WAITING_APPROVAL "
                    "状态可以取消。"
                    f"当前状态="
                    f"{current_status.value}"
                )

            conn.execute(
                """
                UPDATE scheduled_job_run

                SET
                    status = ?,
                    approval_actor = ?,
                    approval_reason = ?,
                    finished_at = ?,
                    result_summary = ?,
                    updated_at = ?

                WHERE run_id = ?
                """,
                (
                    JobRunStatus
                    .CANCELED
                    .value,

                    normalized_actor,

                    normalized_reason,

                    now,

                    "用户拒绝启动该计划任务。",

                    now,

                    normalized_run_id,
                ),
            )

        result = self.get_run(
            normalized_run_id
        )

        if result is None:

            raise RuntimeError(
                "取消 Run 后"
                "无法重新读取。"
            )

        return result

    # ========================================================
    # Active Run
    # ========================================================

    def find_active_run(
        self,
        job_name: str,
    ) -> ScheduledJobRun | None:
        """查询当前长期占有 Single-Flight 的 Run。"""

        normalized_name = str(
            job_name
        ).strip()

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT *

                FROM scheduled_job_run

                WHERE job_name = ?

                  AND status IN (
                      'RUNNING',
                      'WAITING_TOOL_APPROVAL'
                  )

                ORDER BY
                    requested_at ASC

                LIMIT 1
                """,
                (
                    normalized_name,
                ),
            ).fetchone()

        if row is None:

            return None

        return self._row_to_run(
            row
        )

    # ========================================================
    # Internal State Transition
    # ========================================================

    def _transition_run(
        self,
        *,
        run_id: str,
        expected_statuses: set[
            JobRunStatus
        ],
        new_status: JobRunStatus,
    ) -> ScheduledJobRun:

        normalized_run_id = str(
            run_id
        ).strip()

        now = utc_now_text()

        expected_values = {

            status.value

            for status
            in expected_statuses
        }

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT status

                FROM scheduled_job_run

                WHERE run_id = ?
                """,
                (
                    normalized_run_id,
                ),
            ).fetchone()

            if row is None:

                raise KeyError(
                    f"Job Run 不存在："
                    f"{normalized_run_id}"
                )

            current = str(
                row[
                    "status"
                ]
            )

            if (
                current
                not in expected_values
            ):

                raise RuntimeError(
                    "Job Run 状态转换非法："
                    f"{current}"
                    " -> "
                    f"{new_status.value}"
                )

            conn.execute(
                """
                UPDATE scheduled_job_run

                SET
                    status = ?,
                    updated_at = ?

                WHERE run_id = ?
                """,
                (
                    new_status.value,

                    now,

                    normalized_run_id,
                ),
            )

        result = self.get_run(
            normalized_run_id
        )

        if result is None:

            raise RuntimeError(
                "状态转换后"
                "无法读取 Run。"
            )

        return result

    def _finish_run(
        self,
        *,
        run_id: str,
        expected_statuses: set[
            JobRunStatus
        ],
        new_status: JobRunStatus,
        result_summary: str | None,
        error_message: str | None,
    ) -> ScheduledJobRun:

        normalized_run_id = str(
            run_id
        ).strip()

        now = utc_now_text()

        expected_values = {

            status.value

            for status
            in expected_statuses
        }

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT status

                FROM scheduled_job_run

                WHERE run_id = ?
                """,
                (
                    normalized_run_id,
                ),
            ).fetchone()

            if row is None:

                raise KeyError(
                    f"Job Run 不存在："
                    f"{normalized_run_id}"
                )

            current = str(
                row[
                    "status"
                ]
            )

            if (
                current
                not in expected_values
            ):

                raise RuntimeError(
                    "Job Run 无法结束："
                    f"{current}"
                    " -> "
                    f"{new_status.value}"
                )

            conn.execute(
                """
                UPDATE scheduled_job_run

                SET
                    status = ?,
                    finished_at = ?,
                    result_summary = ?,
                    error_message = ?,
                    updated_at = ?

                WHERE run_id = ?
                """,
                (
                    new_status.value,

                    now,

                    result_summary,

                    error_message,

                    now,

                    normalized_run_id,
                ),
            )

        result = self.get_run(
            normalized_run_id
        )

        if result is None:

            raise RuntimeError(
                "结束 Run 后"
                "无法重新读取。"
            )

        return result

    # ========================================================
    # Mapping
    # ========================================================

    @staticmethod
    def _row_to_job(
        row: sqlite3.Row,
    ) -> ScheduledJob:

        return ScheduledJob(

            job_id=str(
                row[
                    "job_id"
                ]
            ),

            job_name=str(
                row[
                    "job_name"
                ]
            ),

            enabled=bool(
                row[
                    "enabled"
                ]
            ),

            schedule_type=(
                ScheduleType(
                    row[
                        "schedule_type"
                    ]
                )
            ),

            schedule_expression=str(
                row[
                    "schedule_expression"
                ]
            ),

            timezone=str(
                row[
                    "timezone"
                ]
            ),

            misfire_policy=(
                MisfirePolicy(
                    row[
                        "misfire_policy"
                    ]
                )
            ),

            requires_start_approval=bool(
                row[
                    "requires_start_approval"
                ]
            ),

            last_scheduled_at=(
                row[
                    "last_scheduled_at"
                ]
            ),

            next_run_at=(
                row[
                    "next_run_at"
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

    @staticmethod
    def _row_to_run(
        row: sqlite3.Row,
    ) -> ScheduledJobRun:

        return ScheduledJobRun(

            run_id=str(
                row[
                    "run_id"
                ]
            ),

            job_id=str(
                row[
                    "job_id"
                ]
            ),

            job_name=str(
                row[
                    "job_name"
                ]
            ),

            trigger_type=(
                JobTriggerType(
                    row[
                        "trigger_type"
                    ]
                )
            ),

            scheduled_at=(
                row[
                    "scheduled_at"
                ]
            ),

            status=(
                JobRunStatus(
                    row[
                        "status"
                    ]
                )
            ),

            agent_thread_id=(
                row[
                    "agent_thread_id"
                ]
            ),

            requested_at=str(
                row[
                    "requested_at"
                ]
            ),

            approved_at=(
                row[
                    "approved_at"
                ]
            ),

            started_at=(
                row[
                    "started_at"
                ]
            ),

            finished_at=(
                row[
                    "finished_at"
                ]
            ),

            approval_actor=(
                row[
                    "approval_actor"
                ]
            ),

            approval_reason=(
                row[
                    "approval_reason"
                ]
            ),

            result_summary=(
                row[
                    "result_summary"
                ]
            ),

            error_message=(
                row[
                    "error_message"
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