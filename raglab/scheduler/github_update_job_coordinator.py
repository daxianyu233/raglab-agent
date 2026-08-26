"""GitHub Intelligence Job coordination layer.

统一以下入口：

1. Scheduler
2. Agent Manual Tool Call
3. Future API

核心目标：

无论从哪个入口触发 GitHub Intelligence Update，
最终都必须进入同一个 Job Single-Flight 边界。

Scheduler 路径：

    Scheduled Job
        ↓
    JobExecutionContext 已存在
        ↓
    Agent
        ↓
    update_github_intelligence
        ↓
    Coordinator
        ↓
    复用已有 Job
        ↓
    execute_github_intelligence_update()

Manual Agent 路径：

    Agent
        ↓
    update_github_intelligence
        ↓
    没有 JobExecutionContext
        ↓
    创建 MANUAL Job Run
        ↓
    Single-Flight
        ↓
    execute_github_intelligence_update()
"""

from __future__ import annotations

from typing import Any, Callable

from raglab.scheduler.execution_context import (
    get_job_execution_context,
    job_execution_scope,
)
from raglab.scheduler.job import (
    JobRunStatus,
    JobTriggerType,
    MisfirePolicy,
    ScheduleType,
)
from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)


GITHUB_UPDATE_JOB_NAME = (
    "github_intelligence_update"
)


UpdateCallable = Callable[
    [],
    dict[str, Any],
]


class GithubUpdateJobCoordinator:
    """统一协调 GitHub Intelligence Update 的 Job 边界。"""

    def __init__(
        self,
        repository: ScheduledJobRepository,
    ) -> None:

        self.repository = repository
        self.repository.setup()

    # ========================================================
    # Public
    # ========================================================

    def execute_tool_call(
        self,
        *,
        execute_update: UpdateCallable,
        manual_actor: str = "agent_manual_request",
    ) -> dict[str, Any]:
        """执行一次 GitHub Update Tool Call。

        如果已经存在 JobExecutionContext：
            当前 Tool 已经属于 Scheduler / Job，
            不再创建第二个 Job。

        如果不存在 JobExecutionContext：
            当前调用来自普通 Agent，
            自动创建 MANUAL Job Run。
        """

        current_context = (
            get_job_execution_context()
        )

        # ----------------------------------------------------
        # 已经属于现有 Job。
        # ----------------------------------------------------

        if current_context is not None:

            if (
                current_context.job_name
                != GITHUB_UPDATE_JOB_NAME
            ):
                raise RuntimeError(
                    "GitHub Update Tool 当前位于"
                    "另一个 JobExecutionContext 中："
                    f"{current_context.job_name}"
                )

            result = execute_update()

            return self._attach_job_metadata(
                result=result,
                run_id=current_context.run_id,
                trigger_type=(
                    current_context.trigger_type
                ),
                reused_existing_job=True,
            )

        # ----------------------------------------------------
        # 普通 Agent 手动调用。
        # ----------------------------------------------------

        return self._execute_manual_job(
            execute_update=execute_update,
            actor=manual_actor,
        )

    # ========================================================
    # Manual Job
    # ========================================================

    def _execute_manual_job(
        self,
        *,
        execute_update: UpdateCallable,
        actor: str,
    ) -> dict[str, Any]:

        job = self._ensure_job_definition()

        # 用户已经明确要求执行 GitHub Update，
        # 所以 MANUAL Job 不再重复询问 Job Start Approval。
        #
        # 但是仍先创建 WAITING_APPROVAL，
        # 再通过 approve_and_acquire() 原子进入 RUNNING，
        # 从而复用统一的 Single-Flight 逻辑。

        run = self.repository.create_run(
            job=job,
            trigger_type=(
                JobTriggerType.MANUAL
            ),
            scheduled_at=None,
            status=(
                JobRunStatus
                .WAITING_APPROVAL
            ),
        )

        run, duplicate_run_id = (
            self.repository
            .approve_and_acquire(
                run_id=run.run_id,
                actor=actor,
                reason=(
                    "用户通过 Agent 明确请求 "
                    "GitHub 技术情报更新。"
                ),
            )
        )

        # ----------------------------------------------------
        # 已经存在 Active Run。
        # ----------------------------------------------------

        if (
            run.status
            ==
            JobRunStatus
            .SKIPPED_DUPLICATE
        ):
            return {
                "status": "busy",
                "tool": (
                    "update_github_intelligence"
                ),
                "message": (
                    "已有 GitHub 技术情报更新任务"
                    "正在运行或等待人工审批，"
                    "本次手动请求不会重复执行。"
                ),
                "job": {
                    "run_id": run.run_id,
                    "status": (
                        run.status.value
                    ),
                    "trigger_type": (
                        run.trigger_type.value
                    ),
                    "duplicate_of_run_id": (
                        duplicate_run_id
                    ),
                },
            }

        # ----------------------------------------------------
        # 成功获得执行资格。
        # ----------------------------------------------------

        try:

            with job_execution_scope(
                run_id=run.run_id,
                job_name=run.job_name,
                trigger_type=(
                    run.trigger_type.value
                ),
            ):
                result = execute_update()

        except Exception as exc:

            failed_run = (
                self.repository
                .mark_failed(
                    run_id=run.run_id,
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            return {
                "status": "failed",
                "tool": (
                    "update_github_intelligence"
                ),
                "message": (
                    "GitHub 技术情报更新"
                    "执行过程中出现异常。"
                ),
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "job": {
                    "run_id": (
                        failed_run.run_id
                    ),
                    "status": (
                        failed_run.status.value
                    ),
                    "trigger_type": (
                        failed_run
                        .trigger_type
                        .value
                    ),
                },
            }

        result_status = str(
            result.get(
                "status",
                "",
            )
        ).strip().lower()

        # ----------------------------------------------------
        # Pipeline success
        # ----------------------------------------------------

        if result_status == "success":

            finished_run = (
                self.repository
                .mark_succeeded(
                    run_id=run.run_id,
                    result_summary=str(
                        result.get(
                            "message",
                            (
                                "GitHub 技术情报"
                                "更新完成。"
                            ),
                        )
                    ),
                )
            )

            return self._attach_job_metadata(
                result=result,
                run_id=(
                    finished_run.run_id
                ),
                trigger_type=(
                    finished_run
                    .trigger_type
                    .value
                ),
                reused_existing_job=False,
                job_status=(
                    finished_run.status.value
                ),
            )

        # ----------------------------------------------------
        # failed / timeout / busy / unknown
        # ----------------------------------------------------

        error_message = str(
            result.get(
                "message",
                (
                    "GitHub 技术情报更新"
                    "没有成功完成。"
                ),
            )
        )

        finished_run = (
            self.repository
            .mark_failed(
                run_id=run.run_id,
                error_message=(
                    "pipeline_status="
                    f"{result_status or 'unknown'}; "
                    f"{error_message}"
                ),
            )
        )

        return self._attach_job_metadata(
            result=result,
            run_id=finished_run.run_id,
            trigger_type=(
                finished_run
                .trigger_type
                .value
            ),
            reused_existing_job=False,
            job_status=(
                finished_run.status.value
            ),
        )

    # ========================================================
    # Job Definition
    # ========================================================

    def _ensure_job_definition(
        self,
    ):
        """确保 GitHub Update Job Definition 存在。

        next_run_at 以及实际调度逻辑，
        后续由 SchedulerService 负责。
        """

        existing = (
            self.repository
            .get_job(
                GITHUB_UPDATE_JOB_NAME
            )
        )

        if existing is not None:
            return existing

        return self.repository.upsert_job(
            job_name=(
                GITHUB_UPDATE_JOB_NAME
            ),
            schedule_type=(
                ScheduleType.DAILY
            ),
            schedule_expression="08:00",
            timezone_name=(
                "Asia/Shanghai"
            ),
            misfire_policy=(
                MisfirePolicy
                .COALESCE_RUN_ONCE
            ),
            requires_start_approval=True,
            enabled=True,
            next_run_at=None,
        )

    # ========================================================
    # Result
    # ========================================================

    @staticmethod
    def _attach_job_metadata(
        *,
        result: dict[str, Any],
        run_id: str,
        trigger_type: str,
        reused_existing_job: bool,
        job_status: str | None = None,
    ) -> dict[str, Any]:

        enriched = dict(
            result
        )

        job_information: dict[
            str,
            Any,
        ] = {
            "run_id": run_id,
            "trigger_type": (
                trigger_type
            ),
            "reused_existing_job": (
                reused_existing_job
            ),
        }

        if job_status is not None:
            job_information[
                "status"
            ] = job_status

        enriched[
            "job"
        ] = job_information

        return enriched