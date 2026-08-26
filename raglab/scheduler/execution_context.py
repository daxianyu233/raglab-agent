"""Current Scheduler Job execution context.

这个模块用于把：

    JobExecutionService

中的 Job Run 身份向内部调用链传播。

典型流程：

    JobExecutionService
        ↓
    job_execution_scope(run_001)
        ↓
    Agent Runtime
        ↓
    LangGraph
        ↓
    Tool
        ↓
    get_job_execution_context()
        ↓
    得知当前 Tool 已经属于 run_001

这样 Scheduler Job 内部调用 GitHub Update Tool 时，
就不会再次创建第二个 Job Run。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class JobExecutionContext:
    """当前执行链所属的 Job Run。"""

    run_id: str

    job_name: str

    trigger_type: str


_CURRENT_JOB_EXECUTION_CONTEXT: ContextVar[
    JobExecutionContext | None
] = ContextVar(
    "raglab_current_job_execution_context",
    default=None,
)


def get_job_execution_context(
) -> JobExecutionContext | None:
    """读取当前 Job Execution Context。

    返回 None 表示：

        当前执行链不属于任何已有 Job Run。

    例如普通用户从 Agent 手动触发
    update_github_intelligence 时，
    初始情况下这里就是 None。
    """

    return (
        _CURRENT_JOB_EXECUTION_CONTEXT
        .get()
    )


@contextmanager
def job_execution_scope(
    *,
    run_id: str,
    job_name: str,
    trigger_type: str,
) -> Iterator[
    JobExecutionContext
]:
    """临时建立 Job Execution Context。

    Context 生命周期只覆盖当前执行链。

    一旦当前 runner 返回：

        COMPLETED
        WAITING_TOOL_APPROVAL
        CANCELED
        Exception

    Context 都会自动恢复之前的值。

    HITL 后再次 resume 时，
    JobExecutionService 会重新建立同一个
    run_id 对应的 Context。
    """

    normalized_run_id = str(
        run_id
    ).strip()

    normalized_job_name = str(
        job_name
    ).strip()

    normalized_trigger_type = str(
        trigger_type
    ).strip()

    if not normalized_run_id:

        raise ValueError(
            "run_id 不能为空。"
        )

    if not normalized_job_name:

        raise ValueError(
            "job_name 不能为空。"
        )

    if not normalized_trigger_type:

        raise ValueError(
            "trigger_type 不能为空。"
        )

    context = JobExecutionContext(

        run_id=(
            normalized_run_id
        ),

        job_name=(
            normalized_job_name
        ),

        trigger_type=(
            normalized_trigger_type
        ),
    )

    token = (
        _CURRENT_JOB_EXECUTION_CONTEXT
        .set(
            context
        )
    )

    try:

        yield context

    finally:

        _CURRENT_JOB_EXECUTION_CONTEXT.reset(
            token
        )