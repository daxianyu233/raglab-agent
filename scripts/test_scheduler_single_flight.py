"""Scheduler Single-Flight 回归测试。

验证目标：

1. 创建 Run A；
2. Run A 正常进入 WAITING_TOOL_APPROVAL；
3. WAITING_TOOL_APPROVAL 必须仍被视为 active；
4. 再创建同类型 Run B；
5. Run B 尝试启动时必须被 Single-Flight 拦截；
6. Run B 不能进入 Agent；
7. Run B 不能产生第二个 HITL；
8. 真实 GitHub Pipeline 全程不能执行；
9. 最后 REJECT Run A，使测试环境恢复为无 Active Run；
10. 正式 Scheduler 的时间游标不能被修改。

安全措施：

测试进程会临时替换
execute_github_intelligence_update()。

如果任何路径错误地尝试执行真实 GitHub Pipeline，
测试会立即失败，而不会真的启动采集、LLM 分析和索引重建。
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


JOB_NAME = "github_intelligence_update"

TEST_USER_ID = (
    "scheduler-single-flight-regression"
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


def enum_value(
    value,
) -> str:

    return str(
        getattr(
            value,
            "value",
            value,
        )
    )


def print_run(
    title: str,
    run,
) -> None:

    print()

    print(
        title
    )

    print(
        f"  run_id：{run.run_id}"
    )

    print(
        f"  trigger_type："
        f"{enum_value(run.trigger_type)}"
    )

    print(
        f"  status："
        f"{enum_value(run.status)}"
    )

    print(
        f"  agent_thread_id："
        f"{run.agent_thread_id}"
    )


def find_result_for_run(
    results,
    run_id: str,
):

    for result in results:

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
            run_id
        ):

            return result

    return None


# ============================================================
# Main
# ============================================================


def main() -> None:

    section(
        "Scheduler Single-Flight 回归测试"
    )

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

    # ========================================================
    # 保存 Scheduler Cursor
    # ========================================================

    before_last_scheduled_at = (
        job.last_scheduled_at
    )

    before_next_run_at = (
        job.next_run_at
    )

    print(
        f"job_name：{job.job_name}"
    )

    print(
        f"enabled：{job.enabled}"
    )

    print(
        "last_scheduled_at："
        f"{before_last_scheduled_at}"
    )

    print(
        "next_run_at："
        f"{before_next_run_at}"
    )

    # ========================================================
    # 1. Active Run 预检查
    # ========================================================

    section(
        "1. Active Run 预检查"
    )

    active_runs = []

    for active_status in (
        JobRunStatus.RUNNING,
        JobRunStatus.WAITING_TOOL_APPROVAL,
    ):

        active_runs.extend(
            repository.list_runs(
                job_name=JOB_NAME,
                status=active_status,
                limit=20,
            )
        )

    if active_runs:

        for run in active_runs:

            print_run(
                "发现 Active Run：",
                run,
            )

        fail(
            "测试开始前存在 RUNNING / "
            "WAITING_TOOL_APPROVAL Run。\n"
            "为避免影响真实任务，停止测试。"
        )

    pass_message(
        "测试开始前没有 Active Run"
    )

    # ========================================================
    # 2. 安装 Pipeline 哨兵
    # ========================================================

    section(
        "2. 安装真实 Pipeline 防执行哨兵"
    )

    if not hasattr(
        github_intelligence_skill,
        "execute_github_intelligence_update",
    ):

        fail(
            "当前 Skill 中不存在 "
            "execute_github_intelligence_update。"
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
            "Single-Flight 测试期间 "
            "真实 GitHub Pipeline 被调用。\n"
            "测试已安全中止。"
        )

    github_intelligence_skill.execute_github_intelligence_update = (
        forbidden_execute
    )

    pass_message(
        "真实 GitHub Pipeline 已被哨兵保护"
    )

    try:

        # ====================================================
        # 3. Build Agent
        # ====================================================

        section(
            "3. 创建 Agent / Controller"
        )

        config_path = (
            CONFIG_DIR
            / "agent.yaml"
        ).resolve()

        agent = build_secure_agent(
            config_path
        )

        controller = (
            ScheduledGithubUpdateController(
                agent=agent,
                user_id=TEST_USER_ID,
                repository=repository,
            )
        )

        pass_message(
            "Secure Agent Runtime 创建成功"
        )

        pass_message(
            "Scheduler Controller 创建成功"
        )

        # ====================================================
        # 4. 创建 Run A
        # ====================================================

        section(
            "4. 创建 Run A"
        )

        run_a = repository.create_run(
            job=job,
            trigger_type=(
                JobTriggerType.MANUAL
            ),
            scheduled_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            status=(
                JobRunStatus
                .WAITING_APPROVAL
            ),
        )

        print_run(
            "Run A 初始状态：",
            run_a,
        )

        if (
            run_a.status
            !=
            JobRunStatus.WAITING_APPROVAL
        ):

            fail(
                "Run A 创建后不是 "
                "WAITING_APPROVAL。"
            )

        pass_message(
            "Run A 创建成功"
        )

        # ====================================================
        # 5. 启动 Run A
        # ====================================================

        section(
            "5. Run A 进入 HITL"
        )

        results_a = (
            controller
            .auto_start_waiting_runs()
        )

        result_a = (
            find_result_for_run(
                results_a,
                run_a.run_id,
            )
        )

        if result_a is None:

            fail(
                "Controller 没有返回 "
                "Run A 的执行结果。"
            )

        run_a_waiting = (
            result_a.run
        )

        print_run(
            "Run A HITL 状态：",
            run_a_waiting,
        )

        if (
            run_a_waiting.status
            !=
            JobRunStatus
            .WAITING_TOOL_APPROVAL
        ):

            fail(
                "Run A 没有停在 "
                "WAITING_TOOL_APPROVAL。\n"
                f"实际状态："
                f"{enum_value(run_a_waiting.status)}"
            )

        if not run_a_waiting.agent_thread_id:

            fail(
                "Run A 缺少 agent_thread_id。"
            )

        if forbidden_execute_calls:

            fail(
                "Run A 进入 HITL 前 "
                "真实 Pipeline 已被调用。"
            )

        pass_message(
            "Run A 已占用 Single-Flight 槽位"
        )

        pass_message(
            "Run A 正确停在 WAITING_TOOL_APPROVAL"
        )

        # ====================================================
        # 6. 创建竞争 Run B
        # ====================================================

        section(
            "6. 创建竞争 Run B"
        )

        run_b = repository.create_run(
            job=job,
            trigger_type=(
                JobTriggerType.MANUAL
            ),
            scheduled_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            status=(
                JobRunStatus
                .WAITING_APPROVAL
            ),
        )

        print_run(
            "Run B 初始状态：",
            run_b,
        )

        if (
            run_b.status
            !=
            JobRunStatus.WAITING_APPROVAL
        ):

            fail(
                "Run B 创建后不是 "
                "WAITING_APPROVAL。"
            )

        pass_message(
            "Run B 创建成功"
        )

        # ====================================================
        # 7. 尝试启动 Run B
        # ====================================================

        section(
            "7. Run B 尝试获取 Single-Flight"
        )

        results_b = (
            controller
            .auto_start_waiting_runs()
        )

        result_b = (
            find_result_for_run(
                results_b,
                run_b.run_id,
            )
        )

        if result_b is None:

            fail(
                "Controller 没有返回 "
                "Run B 的执行结果。"
            )

        final_b = (
            result_b.run
        )

        print_run(
            "Run B 竞争结果：",
            final_b,
        )

        conflict_reason = getattr(
            result_b,
            "conflict_reason",
            None,
        )

        print(
            "conflict_reason："
            f"{conflict_reason}"
        )

        # ====================================================
        # 8. Single-Flight 核心断言
        # ====================================================

        section(
            "8. Single-Flight 核心断言"
        )

        # ----------------------------------------------------
        # B 绝不能获得 Agent thread。
        # ----------------------------------------------------

        if final_b.agent_thread_id:

            fail(
                "Run B 被 Single-Flight 拦截后 "
                "却获得了 agent_thread_id：\n"
                f"{final_b.agent_thread_id}"
            )

        pass_message(
            "Run B 没有进入 Agent Runtime"
        )

        # ----------------------------------------------------
        # B 绝不能进入 RUNNING。
        # ----------------------------------------------------

        if (
            final_b.status
            ==
            JobRunStatus.RUNNING
        ):

            fail(
                "Run B 错误进入 RUNNING。"
            )

        # ----------------------------------------------------
        # B 更不能产生第二个 HITL。
        # ----------------------------------------------------

        if (
            final_b.status
            ==
            JobRunStatus
            .WAITING_TOOL_APPROVAL
        ):

            fail(
                "Run B 错误产生第二个 "
                "WAITING_TOOL_APPROVAL。"
            )

        pass_message(
            "Run B 没有产生第二个 Tool HITL"
        )

        # ----------------------------------------------------
        # 理想状态是 SKIPPED_DUPLICATE。
        #
        # 如果 Repository 当前采用 conflict_reason
        # 表达冲突，也接受 conflict_reason 非空，
        # 但 B 仍然不能进入 active 状态。
        # ----------------------------------------------------

        single_flight_blocked = (
            final_b.status
            ==
            JobRunStatus
            .SKIPPED_DUPLICATE
        ) or (
            conflict_reason is not None
        )

        if not single_flight_blocked:

            fail(
                "Run B 虽然没有进入 Agent，"
                "但也没有得到 "
                "SKIPPED_DUPLICATE / conflict_reason。\n"
                f"status="
                f"{enum_value(final_b.status)}"
            )

        pass_message(
            "Run B 被 Single-Flight 正确拦截"
        )

        print(
            "Run B 最终判定："
            f"{enum_value(final_b.status)}"
        )

        # ----------------------------------------------------
        # 全程真实 Pipeline 不能执行。
        # ----------------------------------------------------

        if forbidden_execute_calls:

            fail(
                "竞争测试过程中 "
                "真实 GitHub Pipeline 被调用。"
            )

        pass_message(
            "竞争期间真实 GitHub Pipeline 未执行"
        )

        # ====================================================
        # 9. 数据库中只能存在一个 Tool HITL Run
        # ====================================================

        waiting_tool_runs = (
            repository.list_runs(
                job_name=JOB_NAME,
                status=(
                    JobRunStatus
                    .WAITING_TOOL_APPROVAL
                ),
                limit=20,
            )
        )

        print()

        print(
            "当前 WAITING_TOOL_APPROVAL 数量："
            f"{len(waiting_tool_runs)}"
        )

        for run in waiting_tool_runs:

            print(
                f"  {run.run_id}"
            )

        matching_waiting_runs = [
            run
            for run in waiting_tool_runs
            if run.run_id
            in {
                run_a.run_id,
                run_b.run_id,
            }
        ]

        if (
            len(matching_waiting_runs)
            !=
            1
        ):

            fail(
                "A/B 两条测试 Run 中应该只有 "
                "1 条 WAITING_TOOL_APPROVAL，"
                f"实际为 {len(matching_waiting_runs)}。"
            )

        if (
            matching_waiting_runs[0].run_id
            !=
            run_a.run_id
        ):

            fail(
                "占有 Single-Flight 的不是 Run A。"
            )

        pass_message(
            "数据库中只有 Run A 占有 Tool HITL"
        )

        # ====================================================
        # 10. REJECT A，释放槽位
        # ====================================================

        section(
            "9. REJECT Run A，释放 Single-Flight"
        )

        waiting_run = (
            controller
            .get_waiting_tool_run()
        )

        if waiting_run is None:

            fail(
                "无法找到等待审批的 Run A。"
            )

        if (
            waiting_run.run_id
            !=
            run_a.run_id
        ):

            fail(
                "Controller 找到的等待审批 Run "
                "不是 Run A。\n"
                f"expected={run_a.run_id}\n"
                f"actual={waiting_run.run_id}"
            )

        reject_result = (
            controller
            .resume_waiting_tool(
                approved=False,
                actor=TEST_USER_ID,
                reason=(
                    "Single-Flight regression cleanup"
                ),
            )
        )

        final_a = (
            reject_result.run
        )

        print_run(
            "Run A 最终状态：",
            final_a,
        )

        if (
            final_a.status
            !=
            JobRunStatus.CANCELED
        ):

            fail(
                "REJECT Run A 后 "
                "没有进入 CANCELED。\n"
                f"实际状态："
                f"{enum_value(final_a.status)}"
            )

        pass_message(
            "Run A 已 CANCELED，Single-Flight 槽位释放"
        )

        # ====================================================
        # 11. Pipeline 最终仍未执行
        # ====================================================

        if forbidden_execute_calls:

            fail(
                "REJECT 清理过程中 "
                "真实 Pipeline 被调用。"
            )

        pass_message(
            "整个测试期间真实 GitHub Pipeline 调用次数 = 0"
        )

        # ====================================================
        # 12. Pending interrupt 清除
        # ====================================================

        pending_after = (
            agent.get_pending_approval(
                final_a.agent_thread_id
            )
        )

        if pending_after is not None:

            fail(
                "Run A REJECT 后仍存在 "
                "LangGraph pending approval。"
            )

        pass_message(
            "Run A LangGraph interrupt 已清除"
        )

        # ====================================================
        # 13. Scheduler 时间游标检查
        # ====================================================

        section(
            "10. Scheduler Cursor 检查"
        )

        refreshed_job = (
            repository.get_job(
                JOB_NAME
            )
        )

        if refreshed_job is None:

            fail(
                "正式 Scheduler Job 丢失。"
            )

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
                "Single-Flight 测试修改了 "
                "last_scheduled_at。"
            )

        if (
            refreshed_job.next_run_at
            !=
            before_next_run_at
        ):

            fail(
                "Single-Flight 测试修改了 "
                "next_run_at。"
            )

        pass_message(
            "正式 Scheduler 时间游标未变化"
        )

        # ====================================================
        # Final
        # ====================================================

        section(
            "Single-Flight 回归测试通过"
        )

        print(
            "[PASS] Run A 创建"
        )

        print(
            "[PASS] Run A → WAITING_TOOL_APPROVAL"
        )

        print(
            "[PASS] WAITING_TOOL_APPROVAL "
            "仍属于 active"
        )

        print(
            "[PASS] Run B 被 Single-Flight 拦截"
        )

        print(
            "[PASS] Run B 未进入 Agent"
        )

        print(
            "[PASS] Run B 未产生第二个 HITL"
        )

        print(
            "[PASS] 真实 Pipeline 调用次数 = 0"
        )

        print(
            "[PASS] Run A REJECT → CANCELED"
        )

        print(
            "[PASS] Scheduler Cursor 未变化"
        )

        print()

        print(
            "Scheduler Single-Flight：PASSED"
        )

    finally:

        # ====================================================
        # 恢复真实 Pipeline。
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
            "Single-Flight 回归测试失败"
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