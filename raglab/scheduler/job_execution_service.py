"""Scheduled Job execution service.

本模块负责：

1. Scheduled Job Run 的执行协调；
2. Job 级 Single-Flight；
3. Runner 注册；
4. Runner 首次执行；
5. Tool HITL 暂停后的恢复；
6. Runner 结果与 Job SQL 状态之间的同步。

职责边界：

SchedulerService
    只负责“什么时候到期、创建哪个 Run”。

JobExecutionService
    负责“这个 Run 能不能执行、怎么执行、执行后是什么状态”。

GithubUpdateAgentRunner
    负责“如何通过 Agent Runtime 执行业务”。

Tool Policy / HITL
    负责“具体 Tool 是否需要人工批准”。

本模块不直接执行 GitHub 更新逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from raglab.scheduler.execution_context import (
    job_execution_scope,
)
from raglab.scheduler.job import (
    JobRunStatus,
    ScheduledJobRun,
)
from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)


# ============================================================
# Workflow Outcome
# ============================================================


class WorkflowOutcomeType(
    str,
    Enum,
):
    """一次 Runner 调用结束后的业务结果。"""

    COMPLETED = "COMPLETED"

    WAITING_TOOL_APPROVAL = (
        "WAITING_TOOL_APPROVAL"
    )

    CANCELED = "CANCELED"


@dataclass(
    frozen=True,
)
class WorkflowOutcome:
    """Runner 返回给 JobExecutionService 的统一结果。"""

    outcome_type: WorkflowOutcomeType

    summary: str

    pending_approval: Any | None = None

    @classmethod
    def completed(
        cls,
        summary: str,
    ) -> "WorkflowOutcome":
        """业务正常完成。"""

        return cls(
            outcome_type=(
                WorkflowOutcomeType.COMPLETED
            ),
            summary=str(
                summary
            ).strip(),
            pending_approval=None,
        )

    @classmethod
    def waiting_tool_approval(
        cls,
        summary: str,
        pending_approval: Any | None = None,
    ) -> "WorkflowOutcome":
        """业务被 Tool HITL interrupt 暂停。"""

        return cls(
            outcome_type=(
                WorkflowOutcomeType
                .WAITING_TOOL_APPROVAL
            ),
            summary=str(
                summary
            ).strip(),
            pending_approval=pending_approval,
        )

    @classmethod
    def canceled(
        cls,
        summary: str,
    ) -> "WorkflowOutcome":
        """业务被取消。"""

        return cls(
            outcome_type=(
                WorkflowOutcomeType.CANCELED
            ),
            summary=str(
                summary
            ).strip(),
            pending_approval=None,
        )


# ============================================================
# Execution Result
# ============================================================


@dataclass(
    frozen=True,
)
class JobExecutionResult:
    """JobExecutionService 对外返回的执行结果。"""

    run: ScheduledJobRun

    outcome: WorkflowOutcome | None

    conflict_reason: str | None = None


# ============================================================
# Runner Types
# ============================================================


StartRunner = Callable[
    [ScheduledJobRun],
    WorkflowOutcome,
]


ResumeRunner = Callable[
    ...,
    WorkflowOutcome,
]


@dataclass(
    frozen=True,
)
class _RegisteredRunner:
    """JobExecutionService 内部使用的 Runner 描述。"""

    start: StartRunner

    resume: ResumeRunner | None = None


# ============================================================
# Job Execution Service
# ============================================================


class JobExecutionService:
    """Scheduled Job 执行协调服务。"""

    def __init__(
        self,
        repository: (
            ScheduledJobRepository
            | None
        ) = None,
    ) -> None:

        self.repository = (
            repository
            if repository is not None
            else ScheduledJobRepository()
        )

        self.repository.setup()

        self._runners: dict[
            str,
            _RegisteredRunner,
        ] = {}

    # ========================================================
    # Runner Registry
    # ========================================================

    def register_runner(
        self,
        job_name: str,
        runner: Any,
    ) -> None:
        """注册某类 Job 对应的 Runner。

        支持两种形式：

        1. 普通 callable：

            register_runner(
                "job_name",
                callable_runner,
            )

        2. Runner 对象：

            runner.start(run)
            runner.resume(
                run,
                approved=True,
            )
        """

        normalized_job_name = str(
            job_name
        ).strip()

        if not normalized_job_name:

            raise ValueError(
                "job_name 不能为空。"
            )

        # ----------------------------------------------------
        # 优先使用对象 start()。
        # ----------------------------------------------------

        start_method = getattr(
            runner,
            "start",
            None,
        )

        if callable(
            start_method
        ):

            start_runner = (
                start_method
            )

        elif callable(
            runner
        ):

            start_runner = runner

        else:

            raise TypeError(
                "runner 必须是 callable，"
                "或者提供 start() 方法。"
            )

        # ----------------------------------------------------
        # resume() 可选。
        # ----------------------------------------------------

        resume_method = getattr(
            runner,
            "resume",
            None,
        )

        if not callable(
            resume_method
        ):

            resume_method = None

        self._runners[
            normalized_job_name
        ] = _RegisteredRunner(
            start=start_runner,
            resume=resume_method,
        )

    def _get_runner(
        self,
        job_name: str,
    ) -> _RegisteredRunner:
        """读取已注册 Runner。"""

        normalized_job_name = str(
            job_name
        ).strip()

        registered = (
            self._runners.get(
                normalized_job_name
            )
        )

        if registered is None:

            raise KeyError(
                "没有为 Job 注册 Runner："
                f"{normalized_job_name}"
            )

        return registered

    # ========================================================
    # Initial Execution
    # ========================================================

    def approve_and_execute(
        self,
        *,
        run_id: str,
        actor: str,
        reason: str | None = None,
    ) -> JobExecutionResult:
        """批准 Job 启动并尝试获取 Single-Flight。

        当前 Scheduler V1 会使用：

            actor="scheduler:auto"

        自动进入此方法。

        Job Start 本身不需要人工批准。

        人工批准只用于真正的高风险 Tool HITL。
        """

        normalized_run_id = str(
            run_id
        ).strip()

        if not normalized_run_id:

            raise ValueError(
                "run_id 不能为空。"
            )

        normalized_actor = str(
            actor
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        # ----------------------------------------------------
        # Repository 内部完成：
        #
        # WAITING_APPROVAL
        #       ↓
        # Single-Flight 检查
        #       ↓
        # RUNNING
        #
        # 如果已有同名 active Job，
        # Repository 返回 conflict_reason。
        # ----------------------------------------------------

        (
            run,
            conflict_reason,
        ) = (
            self.repository
            .approve_and_acquire(
                run_id=(
                    normalized_run_id
                ),
                actor=(
                    normalized_actor
                ),
                reason=(
                    reason
                ),
            )
        )

        if conflict_reason is not None:

            return JobExecutionResult(
                run=run,
                outcome=None,
                conflict_reason=(
                    conflict_reason
                ),
            )

        registered = (
            self._get_runner(
                run.job_name
            )
        )

        return self._execute_runner(
            run=run,
            runner=(
                registered.start
            ),
        )

    # ========================================================
    # Tool HITL Resume
    # ========================================================

    def resume_after_tool_approval(
        self,
        *,
        run_id: str,
        approved: bool,
        actor: str,
        reason: str = "",
    ) -> JobExecutionResult:
        """恢复 WAITING_TOOL_APPROVAL Job。

        正确状态机：

            WAITING_TOOL_APPROVAL
                    ↓
            resume_from_tool_approval()
                    ↓
                 RUNNING
                    ↓
            Runner.resume(...)
                    ↓
            ┌────────┼──────────┐
            ↓        ↓          ↓
        SUCCEEDED  FAILED  WAITING_TOOL_APPROVAL

        approved=False 时同样需要恢复 LangGraph。

        原因是：

            interrupt()
                ↓
            必须获得 REJECT 决定
                ↓
            Graph 才能正确结束

        因此 REJECT 不是简单修改 SQL。
        """

        normalized_run_id = str(
            run_id
        ).strip()

        if not normalized_run_id:

            raise ValueError(
                "run_id 不能为空。"
            )

        normalized_actor = str(
            actor
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        run = (
            self.repository
            .get_run(
                normalized_run_id
            )
        )

        if run is None:

            raise KeyError(
                "Job Run 不存在："
                f"{normalized_run_id}"
            )

        if (
            run.status
            !=
            JobRunStatus
            .WAITING_TOOL_APPROVAL
        ):

            raise RuntimeError(
                "当前 Job 不处于 "
                "WAITING_TOOL_APPROVAL："
                f"{run.run_id} "
                "status="
                f"{run.status.value}"
            )

        registered = (
            self._get_runner(
                run.job_name
            )
        )

        if registered.resume is None:

            raise RuntimeError(
                "当前 Runner "
                "没有 resume()："
                f"{run.job_name}"
            )

        print()

        print(
            "[JobExecutionService] "
            "恢复等待 Tool Approval 的 Job"
        )

        print(
            "  run_id："
            f"{run.run_id}"
        )

        print(
            "  actor："
            f"{normalized_actor}"
        )

        print(
            "  decision："
            f"{'APPROVE' if approved else 'REJECT'}"
        )

        if reason:

            print(
                "  reason："
                f"{reason}"
            )

        # ----------------------------------------------------
        # 关键状态迁移。
        #
        # 之前这里缺失，
        # 导致出现：
        #
        # WAITING_TOOL_APPROVAL
        #       ↓ Agent完成
        # SUCCEEDED
        #
        # Repository 会拒绝这种非法跳转。
        #
        # 正确方式：
        #
        # WAITING_TOOL_APPROVAL
        #       ↓
        # RUNNING
        #       ↓
        # SUCCEEDED / FAILED
        # ----------------------------------------------------

        resumed_run = (
            self.repository
            .resume_from_tool_approval(
                run.run_id
            )
        )

        if (
            resumed_run.status
            !=
            JobRunStatus.RUNNING
        ):

            raise RuntimeError(
                "Tool Approval 后 Job "
                "没有恢复到 RUNNING："
                f"{resumed_run.run_id} "
                "status="
                f"{resumed_run.status.value}"
            )

        print(
            "  Job状态："
            "WAITING_TOOL_APPROVAL "
            "→ RUNNING"
        )

        def resume_callable(
            current_run: ScheduledJobRun,
        ) -> WorkflowOutcome:

            if registered.resume is None:

                raise RuntimeError(
                    "Runner resume() "
                    "意外为空。"
                )

            return registered.resume(
                current_run,
                approved=approved,
            )

        return self._execute_runner(
            run=resumed_run,
            runner=resume_callable,
        )

    # ========================================================
    # Legacy / Administrative Cancel
    # ========================================================

    def reject_run(
        self,
        *,
        run_id: str,
        actor: str,
        reason: str,
    ) -> JobExecutionResult:
        """直接把尚未进入 Tool HITL 的 Job 标记为取消。

        当前 CLI 正常的 Tool REJECT 不走这里。

        Tool REJECT 应走：

            resume_after_tool_approval(
                approved=False
            )

        本方法只保留给管理/兼容路径使用。
        """

        normalized_run_id = str(
            run_id
        ).strip()

        if not normalized_run_id:

            raise ValueError(
                "run_id 不能为空。"
            )

        normalized_actor = str(
            actor
        ).strip()

        if not normalized_actor:

            raise ValueError(
                "actor 不能为空。"
            )

        normalized_reason = str(
            reason
        ).strip()

        summary = (
            f"{normalized_actor}: "
            f"{normalized_reason}"
        )

        run = (
            self.repository
            .mark_runtime_canceled(
                run_id=(
                    normalized_run_id
                ),
                result_summary=(
                    summary
                ),
            )
        )

        return JobExecutionResult(
            run=run,
            outcome=(
                WorkflowOutcome
                .canceled(
                    summary
                )
            ),
            conflict_reason=None,
        )

    # ========================================================
    # Common Runner Execution
    # ========================================================

    def _execute_runner(
        self,
        *,
        run: ScheduledJobRun,
        runner: StartRunner,
    ) -> JobExecutionResult:
        """执行 Runner 并同步 Job SQL 状态。

        所有首次执行和 HITL 恢复都必须经过这里。

        这样状态更新只有一套逻辑。
        """

        if (
            run.status
            !=
            JobRunStatus.RUNNING
        ):

            raise RuntimeError(
                "Runner 只能从 RUNNING "
                "状态开始执行："
                f"{run.run_id} "
                f"status="
                f"{run.status.value}"
            )

        trigger_type = getattr(
            run.trigger_type,
            "value",
            str(
                run.trigger_type
            ),
        )

        try:

            # ------------------------------------------------
            # 将 Job 信息写入 ContextVar。
            #
            # 后续：
            #
            # Agent
            #   ↓
            # update_github_intelligence
            #   ↓
            # GithubUpdateJobCoordinator
            #
            # 可以识别当前已经属于哪个 Job Run，
            # 防止再次创建嵌套 MANUAL Run。
            # ------------------------------------------------

            with job_execution_scope(
                run_id=run.run_id,
                job_name=run.job_name,
                trigger_type=(
                    trigger_type
                ),
            ):

                outcome = runner(
                    run
                )

        except Exception as exc:

            # ------------------------------------------------
            # Runner 真正异常。
            #
            # RUNNING → FAILED
            # ------------------------------------------------

            failed_run = (
                self.repository
                .mark_failed(
                    run_id=(
                        run.run_id
                    ),
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            return JobExecutionResult(
                run=failed_run,
                outcome=None,
                conflict_reason=None,
            )

        if not isinstance(
            outcome,
            WorkflowOutcome,
        ):

            # ------------------------------------------------
            # Runner API 契约错误。
            # ------------------------------------------------

            error_message = (
                "Runner 必须返回 "
                "WorkflowOutcome，"
                "实际返回："
                f"{type(outcome)!r}"
            )

            failed_run = (
                self.repository
                .mark_failed(
                    run_id=(
                        run.run_id
                    ),
                    error_message=(
                        error_message
                    ),
                )
            )

            return JobExecutionResult(
                run=failed_run,
                outcome=None,
                conflict_reason=None,
            )

        # ====================================================
        # Tool HITL
        # ====================================================

        if (
            outcome.outcome_type
            ==
            WorkflowOutcomeType
            .WAITING_TOOL_APPROVAL
        ):

            # RUNNING
            #   ↓
            # WAITING_TOOL_APPROVAL

            waiting_run = (
                self.repository
                .mark_waiting_tool_approval(
                    run.run_id
                )
            )

            return JobExecutionResult(
                run=waiting_run,
                outcome=outcome,
                conflict_reason=None,
            )

        # ====================================================
        # Canceled
        # ====================================================

        if (
            outcome.outcome_type
            ==
            WorkflowOutcomeType.CANCELED
        ):

            # RUNNING
            #   ↓
            # CANCELED

            canceled_run = (
                self.repository
                .mark_runtime_canceled(
                    run_id=(
                        run.run_id
                    ),
                    result_summary=(
                        outcome.summary
                    ),
                )
            )

            return JobExecutionResult(
                run=canceled_run,
                outcome=outcome,
                conflict_reason=None,
            )

        # ====================================================
        # Completed
        # ====================================================

        if (
            outcome.outcome_type
            !=
            WorkflowOutcomeType.COMPLETED
        ):

            error_message = (
                "未知 WorkflowOutcomeType："
                f"{outcome.outcome_type}"
            )

            failed_run = (
                self.repository
                .mark_failed(
                    run_id=(
                        run.run_id
                    ),
                    error_message=(
                        error_message
                    ),
                )
            )

            return JobExecutionResult(
                run=failed_run,
                outcome=outcome,
                conflict_reason=None,
            )

        # RUNNING
        #   ↓
        # SUCCEEDED

        succeeded_run = (
            self.repository
            .mark_succeeded(
                run_id=(
                    run.run_id
                ),
                result_summary=(
                    outcome.summary
                ),
            )
        )

        return JobExecutionResult(
            run=succeeded_run,
            outcome=outcome,
            conflict_reason=None,
        )