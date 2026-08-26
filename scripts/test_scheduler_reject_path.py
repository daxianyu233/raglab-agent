"""Scheduler + HITL Reject 回归测试。

测试目标：

1. 不修改正式 Scheduler 的 next_run_at / last_scheduled_at；
2. 创建一个独立 MANUAL github_intelligence_update Run；
3. 让 Run 正常进入：
       WAITING_APPROVAL
       -> RUNNING
       -> WAITING_TOOL_APPROVAL
4. 自动执行 REJECT；
5. 验证：
       WAITING_TOOL_APPROVAL
       -> RUNNING
       -> CANCELED
6. 验证真实 update_github_intelligence Pipeline 没有执行；
7. 验证 LangGraph HITL interrupt 已经被正确恢复；
8. 验证 Scheduler 时间游标完全没有变化。

安全措施：

真实 execute_github_intelligence_update 会在测试进程中
被替换成一个哨兵函数。

如果 Reject 机制失效并尝试真正执行 GitHub Pipeline，
测试会立即失败，而不会运行真实更新。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback


# ============================================================
# Project Root
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports
# ============================================================


from raglab.application.secure_agent_factory import (
    build_secure_agent,
)

from raglab.settings import (
    CONFIG_DIR,
)

from raglab.scheduler.job import (
    JobRunStatus,
    JobTriggerType,
)

from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)

from raglab.scheduler.scheduled_update_controller import (
    ScheduledGithubUpdateController,
)

import raglab.agent.github_intelligence_skill as github_intelligence_skill


# ============================================================
# Constants
# ============================================================


JOB_NAME = (
    "github_intelligence_update"
)

TEST_USER_ID = (
    "scheduler-reject-regression"
)


# ============================================================
# Helpers
# ============================================================


def section(
    title: str,
) -> None:

    print()

    print(
        "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )


def pass_message(
    message: str,
) -> None:

    print(
        f"[PASS] {message}"
    )


def fail(
    message: str,
) -> None:

    raise AssertionError(
        message
    )


def status_value(
    status,
) -> str:

    return str(
        getattr(
            status,
            "value",
            status,
        )
    )


# ============================================================
# Main
# ============================================================


def main() -> None:

    section(
        "Scheduler + HITL REJECT 回归测试"
    )

    # ========================================================
    # 1. Repository
    # ========================================================

    repository = (
        ScheduledJobRepository()
    )

    repository.setup()

    job = repository.get_job(
        JOB_NAME
    )

    if job is None:

        fail(
            "正式 github_intelligence_update "
            "Job 不存在。"
        )

    print(
        f"job_name：{job.job_name}"
    )

    print(
        f"enabled：{job.enabled}"
    )

    print(
        "last_scheduled_at："
        f"{job.last_scheduled_at}"
    )

    print(
        "next_run_at："
        f"{job.next_run_at}"
    )

    # --------------------------------------------------------
    # 保存正式 Scheduler 游标。
    # --------------------------------------------------------

    before_last_scheduled_at = (
        job.last_scheduled_at
    )

    before_next_run_at = (
        job.next_run_at
    )

    # ========================================================
    # 2. 检查当前是否已有 Active Run
    # ========================================================

    section(
        "1. Active Run 预检查"
    )

    active_runs = []

    for current_status in (
        JobRunStatus.WAITING_APPROVAL,
        JobRunStatus.RUNNING,
        JobRunStatus.WAITING_TOOL_APPROVAL,
    ):

        runs = repository.list_runs(
            job_name=JOB_NAME,
            status=current_status,
            limit=20,
        )

        active_runs.extend(
            runs
        )

    if active_runs:

        print(
            "发现已有 Active Run："
        )

        for run in active_runs:

            print(
                "  "
                f"{run.run_id} "
                f"{status_value(run.status)} "
                f"{run.agent_thread_id}"
            )

        fail(
            "当前已经存在 github_intelligence_update "
            "Active Run。\n"
            "为了避免测试操作错误的 Run，"
            "本次测试主动终止。"
        )

    pass_message(
        "当前没有遗留的 Active "
        "github_intelligence_update Run"
    )

    # ========================================================
    # 3. 安装真实 Pipeline 防执行哨兵
    # ========================================================

    section(
        "2. 安装真实 Pipeline 防执行哨兵"
    )

    if not hasattr(
        github_intelligence_skill,
        "execute_github_intelligence_update",
    ):

        fail(
            "github_intelligence_skill 中不存在 "
            "execute_github_intelligence_update。\n"
            "当前 Skill 实现可能已经变化，"
            "为保证安全，停止测试。"
        )

    original_execute = (
        github_intelligence_skill
        .execute_github_intelligence_update
    )

    forbidden_execute_calls = []

    def forbidden_execute(
        *args,
        **kwargs,
    ):

        forbidden_execute_calls.append(
            {
                "args": args,
                "kwargs": kwargs,
            }
        )

        raise AssertionError(
            "REJECT 测试失败："
            "真实 execute_github_intelligence_update "
            "被调用。\n"
            "为避免真实 GitHub Pipeline 执行，"
            "测试已被安全中止。"
        )

    github_intelligence_skill.execute_github_intelligence_update = (
        forbidden_execute
    )

    pass_message(
        "真实 GitHub Pipeline 已被测试哨兵保护"
    )

    # ========================================================
    # 后续无论成功失败，都恢复原函数。
    # ========================================================

    try:

        # ====================================================
        # 4. 创建 Secure Agent
        # ====================================================

        section(
            "3. 创建 Secure Agent Runtime"
        )

        config_path = (
            CONFIG_DIR
            / "agent.yaml"
        ).resolve()

        agent = build_secure_agent(
            config_path
        )

        pass_message(
            "Secure Agent Runtime 创建成功"
        )

        # ====================================================
        # 5. 创建 Controller
        # ====================================================

        controller = (
            ScheduledGithubUpdateController(
                agent=agent,
                user_id=TEST_USER_ID,
                repository=repository,
            )
        )

        pass_message(
            "ScheduledGithubUpdateController 创建成功"
        )

        # ====================================================
        # 6. 创建独立 MANUAL Run
        # ====================================================

        section(
            "4. 创建独立 MANUAL 测试 Run"
        )

        now_iso = (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )

        test_run = repository.create_run(
            job=job,
            trigger_type=(
                JobTriggerType.MANUAL
            ),
            scheduled_at=now_iso,
            status=(
                JobRunStatus
                .WAITING_APPROVAL
            ),
        )

        print(
            f"run_id：{test_run.run_id}"
        )

        print(
            "trigger_type："
            f"{status_value(test_run.trigger_type)}"
        )

        print(
            "status："
            f"{status_value(test_run.status)}"
        )

        if (
            test_run.status
            !=
            JobRunStatus.WAITING_APPROVAL
        ):

            fail(
                "新建测试 Run "
                "没有进入 WAITING_APPROVAL。"
            )

        pass_message(
            "独立 MANUAL Run 创建成功"
        )

        # ====================================================
        # 7. 自动启动测试 Run
        # ====================================================

        section(
            "5. 启动测试 Run，等待 HITL"
        )

        execution_results = (
            controller
            .auto_start_waiting_runs()
        )

        test_execution_result = None

        for result in execution_results:

            current_run = getattr(
                result,
                "run",
                None,
            )

            if (
                current_run is not None
                and
                current_run.run_id
                ==
                test_run.run_id
            ):

                test_execution_result = (
                    result
                )

                break

        if test_execution_result is None:

            fail(
                "Controller 没有返回刚创建的 "
                "测试 Run 执行结果。"
            )

        hitl_run = (
            test_execution_result.run
        )

        print(
            "status："
            f"{status_value(hitl_run.status)}"
        )

        print(
            "agent_thread_id："
            f"{hitl_run.agent_thread_id}"
        )

        outcome = getattr(
            test_execution_result,
            "outcome",
            None,
        )

        if outcome is not None:

            print(
                "outcome："
                f"{status_value(outcome.outcome_type)}"
            )

            print(
                "summary："
                f"{outcome.summary}"
            )

        # ----------------------------------------------------
        # 如果 Reject 前真实 Pipeline 已经被调用，
        # 测试立即失败。
        # ----------------------------------------------------

        if forbidden_execute_calls:

            fail(
                "在 HITL 阶段之前，"
                "真实 Pipeline 已经被调用。"
            )

        if (
            hitl_run.status
            !=
            JobRunStatus
            .WAITING_TOOL_APPROVAL
        ):

            fail(
                "测试 Run 没有正确停在 "
                "WAITING_TOOL_APPROVAL。\n"
                f"实际状态："
                f"{status_value(hitl_run.status)}"
            )

        if not hitl_run.agent_thread_id:

            fail(
                "WAITING_TOOL_APPROVAL "
                "Run 缺少 agent_thread_id。"
            )

        pass_message(
            "Agent 已正确停在 Tool HITL"
        )

        pass_message(
            "真实 GitHub Pipeline 尚未执行"
        )

        # ====================================================
        # 8. 确认 Controller 找到的 Pending Run
        # ====================================================

        section(
            "6. 检查待审批 Run"
        )

        waiting_run = (
            controller
            .get_waiting_tool_run()
        )

        if waiting_run is None:

            fail(
                "Controller 没有发现 "
                "WAITING_TOOL_APPROVAL Run。"
            )

        print(
            f"waiting run_id："
            f"{waiting_run.run_id}"
        )

        if (
            waiting_run.run_id
            !=
            test_run.run_id
        ):

            fail(
                "Controller 找到的待审批 Run "
                "不是本次测试 Run。\n"
                f"expected={test_run.run_id}\n"
                f"actual={waiting_run.run_id}"
            )

        pass_message(
            "待审批 Run 与本次测试 Run 一致"
        )

        # ====================================================
        # 9. REJECT
        # ====================================================

        section(
            "7. 执行 REJECT"
        )

        reject_result = (
            controller
            .resume_waiting_tool(
                approved=False,
                actor=TEST_USER_ID,
                reason=(
                    "Scheduler HITL "
                    "Reject regression test"
                ),
            )
        )

        final_run = (
            reject_result.run
        )

        print(
            f"final run_id："
            f"{final_run.run_id}"
        )

        print(
            "final status："
            f"{status_value(final_run.status)}"
        )

        final_outcome = getattr(
            reject_result,
            "outcome",
            None,
        )

        if final_outcome is not None:

            print(
                "final outcome："
                f"{status_value(final_outcome.outcome_type)}"
            )

            print(
                "final summary："
                f"{final_outcome.summary}"
            )

        # ====================================================
        # 10. 核心断言：真实 Pipeline 必须完全没执行
        # ====================================================

        section(
            "8. 核心断言"
        )

        if forbidden_execute_calls:

            fail(
                "REJECT 后仍尝试调用了真实 "
                "execute_github_intelligence_update。\n"
                "Reject 安全链路存在问题。"
            )

        pass_message(
            "REJECT 全程没有执行真实 GitHub Pipeline"
        )

        # ====================================================
        # 11. 最终 Job 状态
        # ====================================================

        if (
            final_run.status
            !=
            JobRunStatus.CANCELED
        ):

            fail(
                "REJECT 后最终 Job "
                "没有进入 CANCELED。\n"
                f"实际状态："
                f"{status_value(final_run.status)}\n"
                "如果这里是 SUCCEEDED，"
                "说明 GithubUpdateRunner 对 REJECT "
                "的 WorkflowOutcome 映射仍需修改。"
            )

        pass_message(
            "WAITING_TOOL_APPROVAL "
            "→ RUNNING → CANCELED 正确"
        )

        # ====================================================
        # 12. LangGraph interrupt 必须已经消失
        # ====================================================

        pending_after_reject = (
            agent.get_pending_approval(
                final_run.agent_thread_id
            )
        )

        if (
            pending_after_reject
            is not None
        ):

            fail(
                "REJECT 后 LangGraph "
                "仍然存在 pending approval：\n"
                f"{pending_after_reject}"
            )

        pass_message(
            "LangGraph HITL interrupt "
            "已正确恢复并清除"
        )

        # ====================================================
        # 13. Scheduler Cursor 必须完全没变化
        # ====================================================

        refreshed_job = (
            repository.get_job(
                JOB_NAME
            )
        )

        if refreshed_job is None:

            fail(
                "测试结束后正式 Scheduler Job 丢失。"
            )

        print()

        print(
            "测试前 last_scheduled_at："
            f"{before_last_scheduled_at}"
        )

        print(
            "测试后 last_scheduled_at："
            f"{refreshed_job.last_scheduled_at}"
        )

        print(
            "测试前 next_run_at："
            f"{before_next_run_at}"
        )

        print(
            "测试后 next_run_at："
            f"{refreshed_job.next_run_at}"
        )

        if (
            refreshed_job.last_scheduled_at
            !=
            before_last_scheduled_at
        ):

            fail(
                "Reject 回归测试意外修改了 "
                "last_scheduled_at。"
            )

        if (
            refreshed_job.next_run_at
            !=
            before_next_run_at
        ):

            fail(
                "Reject 回归测试意外修改了 "
                "next_run_at。"
            )

        pass_message(
            "正式 Scheduler 时间游标完全未变化"
        )

        # ====================================================
        # 14. Final
        # ====================================================

        section(
            "REJECT 回归测试通过"
        )

        print(
            "[PASS] MANUAL 测试 Run 创建"
        )

        print(
            "[PASS] 自动进入 RUNNING"
        )

        print(
            "[PASS] Tool Policy 触发 HITL"
        )

        print(
            "[PASS] WAITING_TOOL_APPROVAL"
        )

        print(
            "[PASS] REJECT 恢复原 Checkpoint"
        )

        print(
            "[PASS] 真实 GitHub Pipeline 未执行"
        )

        print(
            "[PASS] 最终 Job = CANCELED"
        )

        print(
            "[PASS] Pending interrupt 已清除"
        )

        print(
            "[PASS] next_run_at 未改变"
        )

        print()

        print(
            "Scheduler / HITL Reject Path："
            "PASSED"
        )

    finally:

        # ====================================================
        # 无论测试成功还是失败，
        # 恢复真实执行函数。
        # ====================================================

        github_intelligence_skill.execute_github_intelligence_update = (
            original_execute
        )

        print()

        print(
            "[Cleanup] "
            "真实 execute_github_intelligence_update "
            "已恢复。"
        )


# ============================================================
# Entry
# ============================================================


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        section(
            "REJECT 回归测试失败"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        traceback.print_exc()

        sys.exit(
            1
        )