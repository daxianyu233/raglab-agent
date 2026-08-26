"""End-to-end Scheduled GitHub Update test.

测试链路：

已有 WAITING_APPROVAL Job
        ↓
自动启动
        ↓
JobExecutionService
        ↓
GithubUpdateAgentRunner
        ↓
Agent
        ↓
Tool HITL
        ↓
WAITING_TOOL_APPROVAL
        ↓
人工 APPROVE / REJECT
        ↓
恢复 Agent
        ↓
GitHub Pipeline
"""

from __future__ import annotations

from pathlib import Path

from raglab.application.secure_agent_factory import (
    build_secure_agent,
)
from raglab.scheduler.job import (
    JobRunStatus,
)
from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)
from raglab.scheduler.scheduled_update_controller import (
    ScheduledGithubUpdateController,
)
from raglab.settings import (
    CONFIG_DIR,
)


CONFIG_PATH = (
    CONFIG_DIR
    / "agent.yaml"
)

USER_ID = "local-user"


def print_result(
    result,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "Job Execution Result"
    )

    print(
        "=" * 80
    )

    print(
        "run_id："
        f"{result.run.run_id}"
    )

    print(
        "status："
        f"{result.run.status.value}"
    )

    print(
        "agent_thread_id："
        f"{result.run.agent_thread_id}"
    )

    print(
        "duplicate_of_run_id："
        f"{result.duplicate_of_run_id}"
    )

    if result.outcome is not None:

        print(
            "outcome："
            f"{result.outcome.outcome_type.value}"
        )

        print(
            "summary："
            f"{result.outcome.summary}"
        )

    print(
        "=" * 80
    )


def main() -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "Scheduled GitHub Update "
        "端到端测试"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 构建与正常 CLI 完全相同的 Agent。
    # --------------------------------------------------------

    agent = build_secure_agent(
        Path(
            CONFIG_PATH
        ).resolve()
    )

    repository = (
        ScheduledJobRepository()
    )

    repository.setup()

    controller = (
        ScheduledGithubUpdateController(
            agent=agent,
            user_id=USER_ID,
            repository=repository,
        )
    )

    # --------------------------------------------------------
    # 自动启动之前已经由 Scheduler 创建、
    # 但尚未执行的 GitHub Job。
    #
    # 你当前数据库中应该正好存在：
    #
    # run_0a6305...
    # WAITING_APPROVAL
    # --------------------------------------------------------

    results = (
        controller
        .auto_start_waiting_runs()
    )

    if not results:

        waiting_tool_run = (
            controller
            .get_waiting_tool_run()
        )

        if waiting_tool_run is None:

            print()
            print(
                "没有 WAITING_APPROVAL "
                "或 WAITING_TOOL_APPROVAL "
                "的 GitHub Update Job。"
            )

            return

    for result in results:

        print_result(
            result
        )

    # --------------------------------------------------------
    # 查看是否停在 Tool HITL。
    # --------------------------------------------------------

    waiting_run = (
        controller
        .get_waiting_tool_run()
    )

    if waiting_run is None:

        print()
        print(
            "当前没有 Tool HITL 中断。"
        )

        print(
            "如果 Job 已经 SUCCEEDED，"
            "说明 update_github_intelligence "
            "没有触发预期的 HITL；"
            "需要检查 Tool Policy Registry。"
        )

        return

    print()
    print(
        "=" * 80
    )

    print(
        "Tool HITL 已成功触发"
    )

    print(
        "=" * 80
    )

    print(
        "run_id："
        f"{waiting_run.run_id}"
    )

    print(
        "agent_thread_id："
        f"{waiting_run.agent_thread_id}"
    )

    print(
        "status："
        f"{waiting_run.status.value}"
    )

    print()
    print(
        "此时 GitHub Update Pipeline "
        "不应该已经真正执行。"
    )

    print()
    print(
        "输入："
    )

    print(
        "  APPROVE"
    )

    print(
        "或："
    )

    print(
        "  REJECT"
    )

    # --------------------------------------------------------
    # 人工决定。
    # --------------------------------------------------------

    while True:

        decision = input(
            "\nDecision："
        ).strip().upper()

        if decision in {
            "APPROVE",
            "REJECT",
        }:
            break

        print(
            "请输入 APPROVE 或 REJECT。"
        )

    approved = (
        decision
        ==
        "APPROVE"
    )

    # --------------------------------------------------------
    # 这里是一次新的 Python 调用。
    #
    # 不是恢复旧函数栈。
    # --------------------------------------------------------

    resumed_result = (
        controller
        .resume_waiting_tool(
            approved=approved,
            actor=USER_ID,
            reason=(
                "Scheduled GitHub Update "
                "端到端测试。"
            ),
        )
    )

    print_result(
        resumed_result
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    if (
        resumed_result.run.status
        ==
        JobRunStatus.SUCCEEDED
    ):

        print()
        print(
            "✅ Scheduled GitHub Update "
            "端到端执行成功。"
        )

    elif (
        resumed_result.run.status
        ==
        JobRunStatus.CANCELED
    ):

        print()
        print(
            "Job 已因人工拒绝而取消。"
        )

    elif (
        resumed_result.run.status
        ==
        JobRunStatus
        .WAITING_TOOL_APPROVAL
    ):

        print()
        print(
            "Agent 恢复后又遇到了新的 "
            "Tool Approval。"
        )

        print(
            "这说明同一流程中还有第二个"
            "需要审批的高风险 Tool。"
        )

    else:

        print()
        print(
            "最终 Job 状态："
            f"{resumed_result.run.status.value}"
        )


if __name__ == "__main__":
    main()
    