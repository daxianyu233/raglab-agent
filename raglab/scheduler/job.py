"""Scheduler Job domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScheduleType(str, Enum):
    """任务调度类型。"""

    DAILY = "DAILY"
    INTERVAL = "INTERVAL"


class MisfirePolicy(str, Enum):
    """错过计划执行时间后的处理策略。"""

    SKIP = "SKIP"

    # 多次错过的计划任务合并为一次补执行。
    COALESCE_RUN_ONCE = "COALESCE_RUN_ONCE"


class JobTriggerType(str, Enum):
    """某次 Job Run 的触发来源。"""

    SCHEDULED = "SCHEDULED"

    MISFIRE_CATCH_UP = "MISFIRE_CATCH_UP"

    MANUAL = "MANUAL"


class JobRunStatus(str, Enum):
    """Job Run 生命周期。"""

    # Scheduler 已创建本次任务，
    # 但还没有获得用户的 Job Start Approval。
    WAITING_APPROVAL = "WAITING_APPROVAL"

    # Workflow 正在运行。
    RUNNING = "RUNNING"

    # Agent 内部已经触发 LangGraph interrupt，
    # 正在等待具体 Tool 的人工审批。
    #
    # 此时 Python 调用已经可以返回，
    # 但这个 Job 仍然占有 Single-Flight 执行资格。
    WAITING_TOOL_APPROVAL = (
        "WAITING_TOOL_APPROVAL"
    )

    SUCCEEDED = "SUCCEEDED"

    FAILED = "FAILED"

    CANCELED = "CANCELED"

    SKIPPED_DUPLICATE = (
        "SKIPPED_DUPLICATE"
    )


@dataclass(frozen=True)
class ScheduledJob:
    """一个持久化的调度规则。"""

    job_id: str

    job_name: str

    enabled: bool

    schedule_type: ScheduleType

    schedule_expression: str

    timezone: str

    misfire_policy: MisfirePolicy

    requires_start_approval: bool

    last_scheduled_at: str | None

    next_run_at: str | None

    created_at: str

    updated_at: str


@dataclass(frozen=True)
class ScheduledJobRun:
    """某一次具体任务运行。"""

    run_id: str

    job_id: str

    job_name: str

    trigger_type: JobTriggerType

    scheduled_at: str | None

    status: JobRunStatus

    # --------------------------------------------------------
    # Job Run 与 LangGraph Thread 的持久化映射。
    #
    # interrupt 以后原来的 Python 调用栈会释放，
    # 后续恢复依靠：
    #
    # run_id
    #     ↓
    # agent_thread_id
    #     ↓
    # LangGraph Checkpoint
    # --------------------------------------------------------
    agent_thread_id: str | None

    requested_at: str

    approved_at: str | None

    started_at: str | None

    finished_at: str | None

    approval_actor: str | None

    approval_reason: str | None

    result_summary: str | None

    error_message: str | None

    created_at: str

    updated_at: str