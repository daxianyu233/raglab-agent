"""Scheduled GitHub Update high-level controller.

这一层把：

Scheduler Job
+
JobExecutionService
+
GithubUpdateAgentRunner

组合起来。

CLI 不需要知道内部 Runner 如何工作。
"""

from __future__ import annotations

from typing import Any

from raglab.scheduler.github_update_job_coordinator import (
    GITHUB_UPDATE_JOB_NAME,
)
from raglab.scheduler.github_update_runner import (
    GithubUpdateAgentRunner,
)
from raglab.scheduler.job import (
    JobRunStatus,
)
from raglab.scheduler.job_execution_service import (
    JobExecutionResult,
    JobExecutionService,
)
from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)


class ScheduledGithubUpdateController:
    """GitHub 定时更新任务控制器。"""

    def __init__(
        self,
        *,
        agent: Any,
        user_id: str,
        repository: (
            ScheduledJobRepository
            | None
        ) = None,
    ) -> None:

        self.repository = (
            repository
            or ScheduledJobRepository()
        )

        self.repository.setup()

        self.runner = (
            GithubUpdateAgentRunner(
                agent=agent,
                repository=(
                    self.repository
                ),
                user_id=user_id,
            )
        )

        self.execution_service = (
            JobExecutionService(
                self.repository
            )
        )

        self.execution_service.register_runner(
            job_name=(
                GITHUB_UPDATE_JOB_NAME
            ),
            runner=self.runner,
        )

    # ========================================================
    # Automatic Start
    # ========================================================

    def auto_start_waiting_runs(
        self,
    ) -> list[
        JobExecutionResult
    ]:
        """自动启动等待中的 GitHub Scheduler Job。

        只处理：

            job_name =
            github_intelligence_update

        其他测试 Job 不碰。
        """

        waiting_runs = (
            self.repository
            .list_waiting_approval_runs()
        )

        targets = [
            run
            for run in waiting_runs
            if (
                run.job_name
                ==
                GITHUB_UPDATE_JOB_NAME
            )
        ]

        results: list[
            JobExecutionResult
        ] = []

        for run in targets:

            print()
            print(
                "=" * 80
            )

            print(
                "自动启动到期的 GitHub "
                "技术情报更新任务"
            )

            print(
                "=" * 80
            )

            print(
                "run_id："
                f"{run.run_id}"
            )

            print(
                "trigger_type："
                f"{run.trigger_type.value}"
            )

            print(
                "scheduled_at："
                f"{run.scheduled_at}"
            )

            result = (
                self.execution_service
                .approve_and_execute(
                    run_id=run.run_id,

                    actor=(
                        "scheduler:auto"
                    ),

                    reason=(
                        "计划任务已到期，"
                        "自动进入 Agent Runtime。"
                    ),
                )
            )

            results.append(
                result
            )

        return results

    # ========================================================
    # Pending Tool HITL
    # ========================================================

    def get_waiting_tool_run(
        self,
    ):
        """查找当前正在等待 Tool Approval 的 GitHub Job。"""

        active_run = (
            self.repository
            .find_active_run(
                job_name=(
                    GITHUB_UPDATE_JOB_NAME
                )
            )
        )

        if active_run is None:
            return None

        if (
            active_run.status
            !=
            JobRunStatus
            .WAITING_TOOL_APPROVAL
        ):
            return None

        return active_run

    def resume_waiting_tool(
        self,
        *,
        approved: bool,
        actor: str = "local-user",
        reason: str = "",
    ) -> JobExecutionResult:
        """恢复当前等待 HITL 的 GitHub Job。"""

        run = self.get_waiting_tool_run()

        if run is None:

            raise RuntimeError(
                "当前没有等待 Tool Approval "
                "的 GitHub Update Job。"
            )

        return (
            self.execution_service
            .resume_after_tool_approval(
                run_id=run.run_id,
                approved=approved,
                actor=actor,
                reason=reason,
            )
        )