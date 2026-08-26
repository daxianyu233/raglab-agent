"""自动长期记忆 LangGraph 多工具 Agent 控制台。

当前入口统一支持：

1. 普通 PDF 知识库查询；
2. GitHub 技术情报查询；
3. Skill Runtime；
4. 自动长期记忆；
5. CLI 启动时的一次 Scheduler 检查；
6. Scheduler Job 自动启动；
7. Job Single-Flight；
8. Tool Policy Registry；
9. Fail-Closed；
10. HITL 高风险 Tool 审批；
11. /approve 与 /reject 恢复。

当前 Scheduler 不常驻轮询。

运行逻辑：

    CLI 启动
        ↓
    SchedulerService.tick()
        ↓
    判断 now >= next_run_at
        ↓
    必要时创建 Job Run
        ↓
    ScheduledGithubUpdateController
        ↓
    自动启动 WAITING_APPROVAL GitHub Job
        ↓
    JobExecutionService
        ↓
    Single-Flight
        ↓
    Secure Agent Runtime
        ↓
    Tool Policy
        ↓
    高风险 Tool -> HITL interrupt
        ↓
    WAITING_TOOL_APPROVAL
        ↓
    CLI /approve 或 /reject
        ↓
    恢复同一 LangGraph thread/checkpoint
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any


from raglab.agent.long_term_memory_agent import (
    normalize_user_id,
)

from raglab.application.secure_agent_factory import (
    build_secure_agent,
)

from raglab.settings import (
    CONFIG_DIR,
)

from raglab.scheduler.github_update_job_coordinator import (
    GITHUB_UPDATE_JOB_NAME,
)

from raglab.scheduler.job import (
    MisfirePolicy,
    ScheduleType,
)

from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)

from raglab.scheduler.scheduler_service import (
    SchedulerService,
    SchedulerTickResult,
)

from raglab.scheduler.scheduled_update_controller import (
    ScheduledGithubUpdateController,
)

from scripts.chat_long_term_memory_agent import (
    print_long_term_memories,
)

from scripts.chat_persistent_agent import (
    create_thread_id,
    normalize_thread_id,
    print_memory_status,
    print_result,
    print_thread_history,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "agent.yaml"
)

DEFAULT_GITHUB_UPDATE_TIME = (
    "08:00"
)

DEFAULT_GITHUB_UPDATE_TIMEZONE = (
    "Asia/Shanghai"
)


# ============================================================
# CLI Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "运行带 Scheduler / Skill / "
            "Policy / HITL 的 RAG-LAB Agent。"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG_PATH
        ),
    )

    parser.add_argument(
        "--user-id",
        type=str,
        default="local-user",
    )

    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--history-preview",
        type=int,
        default=800,
    )

    return parser.parse_args()


# ============================================================
# Scheduler Definition
# ============================================================

def ensure_github_update_job(
    repository: ScheduledJobRepository,
) -> None:
    """确保 GitHub Update 调度定义存在。

    已经存在时绝不重新覆盖 next_run_at。
    """

    existing = (
        repository.get_job(
            GITHUB_UPDATE_JOB_NAME
        )
    )

    if existing is not None:
        return

    repository.upsert_job(
        job_name=(
            GITHUB_UPDATE_JOB_NAME
        ),

        schedule_type=(
            ScheduleType.DAILY
        ),

        schedule_expression=(
            DEFAULT_GITHUB_UPDATE_TIME
        ),

        timezone_name=(
            DEFAULT_GITHUB_UPDATE_TIMEZONE
        ),

        misfire_policy=(
            MisfirePolicy
            .COALESCE_RUN_ONCE
        ),

        # Repository 当前状态机仍然使用
        # WAITING_APPROVAL 作为“尚未获得执行权”的状态。
        #
        # 这里虽然设置 True，
        # 但 CLI 随后会由 scheduler:auto
        # 自动取得执行权。
        requires_start_approval=True,

        enabled=True,

        next_run_at=None,
    )


# ============================================================
# Scheduler Output
# ============================================================

def print_scheduler_tick_result(
    result: SchedulerTickResult,
) -> None:
    """打印 Scheduler 本次检查产生的事件。"""

    for item in (
        result.initialized_jobs
    ):

        print()
        print(
            "[Scheduler] "
            "已初始化调度游标"
        )

        print(
            "  job_name："
            f"{item.job_name}"
        )

        print(
            "  next_run_at："
            f"{item.next_run_at}"
        )

    for item in (
        result.skipped_misfires
    ):

        print()
        print(
            "[Scheduler] "
            "发现已错过的任务，"
            "当前 Misfire Policy 为 SKIP"
        )

        print(
            "  job_name："
            f"{item.job_name}"
        )

        print(
            "  scheduled_at："
            f"{item.scheduled_at}"
        )

        print(
            "  next_run_at："
            f"{item.next_run_at}"
        )

    for run in (
        result.created_runs
    ):

        print()
        print(
            "=" * 80
        )

        print(
            "发现到期的 GitHub 技术情报更新任务"
        )

        print(
            "=" * 80
        )

        print(
            "run_id："
            f"{run.run_id}"
        )

        print(
            "job_name："
            f"{run.job_name}"
        )

        print(
            "trigger_type："
            f"{run.trigger_type.value}"
        )

        print(
            "scheduled_at："
            f"{run.scheduled_at}"
        )

        print(
            "初始状态："
            f"{run.status.value}"
        )

        print()
        print(
            "Scheduler 已创建 Job Run。"
        )

        print(
            "CLI 随后会自动将该 Run "
            "交给 JobExecutionService。"
        )

        print(
            "不会再要求 Job Start Approval。"
        )

        print(
            "只有真正调用高风险 Tool 时"
            "才触发 HITL。"
        )

        print(
            "=" * 80
        )


# ============================================================
# Scheduler Startup Check
# ============================================================

def run_startup_scheduler_check(
) -> ScheduledJobRepository:
    """CLI 启动时只执行一次 Scheduler Check。"""

    repository = (
        ScheduledJobRepository()
    )

    repository.setup()

    ensure_github_update_job(
        repository
    )

    job_before = (
        repository.get_job(
            GITHUB_UPDATE_JOB_NAME
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "Scheduler 启动检查"
    )

    print(
        "=" * 80
    )

    if job_before is not None:

        print(
            "job_name："
            f"{job_before.job_name}"
        )

        print(
            "enabled："
            f"{job_before.enabled}"
        )

        print(
            "schedule："
            f"{job_before.schedule_expression}"
        )

        print(
            "timezone："
            f"{job_before.timezone}"
        )

        print(
            "last_scheduled_at："
            f"{job_before.last_scheduled_at}"
        )

        print(
            "next_run_at："
            f"{job_before.next_run_at}"
        )

    scheduler = (
        SchedulerService(
            repository=repository,
            misfire_grace_seconds=90,
        )
    )

    try:

        result = (
            scheduler.tick()
        )

    except Exception as exc:

        print()
        print(
            "[Scheduler] 启动检查失败："
            f"{type(exc).__name__}："
            f"{exc}"
        )

        print(
            "=" * 80
        )

        return repository

    print_scheduler_tick_result(
        result
    )

    job_after = (
        repository.get_job(
            GITHUB_UPDATE_JOB_NAME
        )
    )

    print()
    print(
        "[Scheduler] 本次检查完成"
    )

    print(
        "检查的启用任务数："
        f"{result.checked_job_count}"
    )

    print(
        "新建 Job Run 数："
        f"{len(result.created_runs)}"
    )

    print(
        "初始化计划数："
        f"{len(result.initialized_jobs)}"
    )

    print(
        "跳过 Misfire 数："
        f"{len(result.skipped_misfires)}"
    )

    if job_after is not None:

        print(
            "当前 next_run_at："
            f"{job_after.next_run_at}"
        )

    if (
        not result.created_runs
        and
        not result.initialized_jobs
        and
        not result.skipped_misfires
    ):

        print(
            "结果：当前没有新的到期任务。"
        )

    print(
        "=" * 80
    )

    return repository


# ============================================================
# Job Execution Output
# ============================================================

def print_job_execution_result(
    result: Any,
) -> None:
    """打印一次 JobExecutionService 结果。"""

    print()
    print(
        "=" * 80
    )

    print(
        "GitHub Update Job 执行状态"
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
        "trigger_type："
        f"{result.run.trigger_type.value}"
    )

    print(
        "agent_thread_id："
        f"{result.run.agent_thread_id}"
    )

    duplicate_run_id = (
        getattr(
            result,
            "duplicate_of_run_id",
            None,
        )
    )

    if duplicate_run_id:

        print(
            "duplicate_of_run_id："
            f"{duplicate_run_id}"
        )

    outcome = getattr(
        result,
        "outcome",
        None,
    )

    if outcome is not None:

        outcome_type = getattr(
            outcome,
            "outcome_type",
            None,
        )

        if outcome_type is not None:

            outcome_value = getattr(
                outcome_type,
                "value",
                str(
                    outcome_type
                ),
            )

            print(
                "outcome："
                f"{outcome_value}"
            )

        summary = str(
            getattr(
                outcome,
                "summary",
                "",
            )
            or ""
        ).strip()

        if summary:

            print(
                "summary："
                f"{summary}"
            )

    if (
        result.run.status.value
        ==
        "WAITING_TOOL_APPROVAL"
    ):

        print()
        print(
            "⚠ 高风险 Tool 已触发 HITL。"
        )

        print(
            "实际 GitHub Update "
            "尚未获得执行授权。"
        )

        print()
        print(
            "输入："
        )

        print(
            "  /approve"
        )

        print(
            "批准当前 Tool。"
        )

        print()
        print(
            "或："
        )

        print(
            "  /reject"
        )

        print(
            "拒绝当前 Tool。"
        )

    print(
        "=" * 80
    )


# ============================================================
# Scheduled Update Startup
# ============================================================

def auto_start_scheduled_github_jobs(
    *,
    controller,
) -> bool:
    """自动启动等待执行的 GitHub Scheduler Job。

    Returns:
        bool:
            True  - 本次启动过程中产生了 WAITING_TOOL_APPROVAL，
                    HITL 信息已经打印过。
            False - 本次没有新产生 Tool HITL。
    """

    hitl_created = False

    try:

        results = (
            controller
            .auto_start_waiting_runs()
        )

    except Exception as exc:

        print(
            "\n[JobController] "
            "自动启动计划任务失败："
            f"{type(exc).__name__}：{exc}"
        )

        return False

    if results is None:

        return False

    # 兼容单个结果和多个结果。
    if not isinstance(
        results,
        (
            list,
            tuple,
        ),
    ):

        results = [
            results
        ]

    for execution_result in results:

        if execution_result is None:
            continue

        print_job_execution_result(
            execution_result
        )

        run = getattr(
            execution_result,
            "run",
            None,
        )

        if run is None:
            continue

        status = getattr(
            run,
            "status",
            None,
        )

        status_value = getattr(
            status,
            "value",
            str(status),
        )

        if (
            status_value
            ==
            "WAITING_TOOL_APPROVAL"
        ):

            hitl_created = True

    return hitl_created

# ============================================================
# Scheduler Job Status
# ============================================================

def print_scheduler_jobs(
    *,
    repository: ScheduledJobRepository,
    controller: ScheduledGithubUpdateController,
) -> None:
    """打印当前 GitHub Update Job 状态。"""

    print()
    print(
        "=" * 80
    )

    print(
        "GitHub Update Job 状态"
    )

    print(
        "=" * 80
    )

    waiting_start_runs = (
        repository
        .list_waiting_approval_runs()
    )

    github_waiting_runs = [
        run
        for run in waiting_start_runs
        if (
            run.job_name
            ==
            GITHUB_UPDATE_JOB_NAME
        )
    ]

    waiting_tool_run = (
        controller
        .get_waiting_tool_run()
    )

    if (
        not github_waiting_runs
        and
        waiting_tool_run is None
    ):

        print(
            "当前没有等待启动或等待 Tool "
            "审批的 GitHub Update Job。"
        )

        print(
            "=" * 80
        )

        return

    for run in github_waiting_runs:

        print()
        print(
            "run_id："
            f"{run.run_id}"
        )

        print(
            "status："
            f"{run.status.value}"
        )

        print(
            "trigger_type："
            f"{run.trigger_type.value}"
        )

        print(
            "scheduled_at："
            f"{run.scheduled_at}"
        )

    if waiting_tool_run is not None:

        print()
        print(
            "run_id："
            f"{waiting_tool_run.run_id}"
        )

        print(
            "status："
            f"{waiting_tool_run.status.value}"
        )

        print(
            "agent_thread_id："
            f"{waiting_tool_run.agent_thread_id}"
        )

        print(
            "说明：当前正在等待高风险 "
            "Tool 的人工审批。"
        )

    print(
        "=" * 80
    )


# ============================================================
# HITL
# ============================================================

def handle_approval_command(
    *,
    command: str,
    agent: Any,
    controller: ScheduledGithubUpdateController,
    current_thread_id: str,
    user_id: str,
) -> bool:
    """处理 /approve 和 /reject。

    返回 True：
        已经消费该命令。

    路由优先级：

    1. Scheduler Job WAITING_TOOL_APPROVAL
    2. 普通聊天 thread 的 HITL
    """

    if command not in {
        "/approve",
        "/reject",
    }:
        return False

    approved = (
        command
        ==
        "/approve"
    )

    # --------------------------------------------------------
    # 1. Scheduler Job HITL
    # --------------------------------------------------------

    waiting_job = (
        controller
        .get_waiting_tool_run()
    )

    if waiting_job is not None:

        print()
        print(
            "[HITL] "
            "检测到 Scheduler Job "
            "正在等待 Tool Approval。"
        )

        print(
            "run_id："
            f"{waiting_job.run_id}"
        )

        print(
            "agent_thread_id："
            f"{waiting_job.agent_thread_id}"
        )

        print(
            "decision："
            f"{'APPROVE' if approved else 'REJECT'}"
        )

        try:

            execution_result = (
                controller
                .resume_waiting_tool(
                    approved=approved,
                    actor=user_id,
                    reason=(
                        "CLI HITL decision"
                    ),
                )
            )

        except Exception as exc:

            print(
                "HITL 恢复失败："
                f"{type(exc).__name__}："
                f"{exc}"
            )

            return True

        print_job_execution_result(
            execution_result
        )

        return True

    # --------------------------------------------------------
    # 2. 普通 Agent Thread HITL
    #
    # 例如用户手动要求：
    #
    #   “更新 GitHub 技术情报”
    #
    # 此时 interrupt 属于普通聊天 thread，
    # 不属于 Scheduler Job。
    # --------------------------------------------------------

    try:

        result = agent.run(
            command,
            user_id=user_id,
            thread_id=(
                current_thread_id
            ),
        )

    except Exception as exc:

        print(
            "当前没有可恢复的 HITL，"
            "或恢复失败："
            f"{type(exc).__name__}："
            f"{exc}"
        )

        return True

    print_result(
        result
    )

    return True


# ============================================================
# Ordinary Agent HITL Output
# ============================================================


def show_pending_agent_hitl(
    *,
    agent: Any,
    thread_id: str,
) -> bool:
    """展示普通聊天 thread 当前等待中的 Tool Approval。

    Returns:
        True:
            当前 thread 存在 pending HITL，且已打印提示。

        False:
            当前 thread 没有 pending HITL。

    注意：

    Scheduler Job 的 HITL 由 ScheduledGithubUpdateController
    单独维护和展示；本函数只负责普通聊天 thread。
    """

    pending = (
        agent.get_pending_approval(
            thread_id
        )
    )

    if pending is None:
        return False

    interrupts = list(
        pending.get(
            "interrupts",
            [],
        )
        or []
    )

    print()
    print(
        "=" * 80
    )

    print(
        "普通聊天正在等待高风险 Tool 审批"
    )

    print(
        "=" * 80
    )

    print(
        "thread_id："
        f"{pending.get('thread_id', thread_id)}"
    )

    if not interrupts:

        print(
            "当前存在 LangGraph interrupt，"
            "但没有可展示的详细审批信息。"
        )

    for index, current in enumerate(
        interrupts,
        start=1,
    ):

        if not isinstance(
            current,
            dict,
        ):

            print()
            print(
                f"审批项 {index}："
                f"{current}"
            )

            continue

        print()
        print(
            f"审批项 {index}"
        )

        print(
            "  type："
            f"{current.get('type', '')}"
        )

        print(
            "  tool_name："
            f"{current.get('tool_name', '')}"
        )

        print(
            "  tool_call_id："
            f"{current.get('tool_call_id', '')}"
        )

        print(
            "  effect_type："
            f"{current.get('effect_type', '')}"
        )

        print(
            "  has_external_side_effect："
            f"{current.get('has_external_side_effect', '')}"
        )

        description = str(
            current.get(
                "description",
                "",
            )
            or ""
        ).strip()

        if description:

            print(
                "  description："
                f"{description}"
            )

        args = current.get(
            "args",
            {},
        )

        print(
            "  args："
            + json.dumps(
                args,
                ensure_ascii=False,
                default=str,
            )
        )

        message = str(
            current.get(
                "message",
                "",
            )
            or ""
        ).strip()

        if message:

            print(
                "  message："
                f"{message}"
            )

    print()
    print(
        "当前 Tool 尚未执行。"
    )

    print(
        "请输入："
    )

    print(
        "  /approve"
    )

    print(
        "批准当前 Tool。"
    )

    print()
    print(
        "或："
    )

    print(
        "  /reject"
    )

    print(
        "拒绝当前 Tool。"
    )

    print(
        "在完成审批前，新的普通聊天消息不会进入该 thread。"
    )

    print(
        "=" * 80
    )

    return True


# ============================================================
# Memory Output
# ============================================================

def print_flush_report(
    report: dict,
) -> None:
    """打印长期记忆整理报告。"""

    print()

    print(
        "=" * 80
    )

    print(
        "长期记忆整理结果"
    )

    print(
        "=" * 80
    )

    for key, value in (
        report.items()
    ):

        print(
            f"{key}：{value}"
        )


# ============================================================
# Skill Output
# ============================================================

def print_skill_status(
    agent: Any,
) -> None:
    """打印当前 Skill Runtime 状态。"""

    runtime = getattr(
        agent,
        "skill_runtime",
        None,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "Skill Runtime 状态"
    )

    print(
        "=" * 80
    )

    if runtime is None:

        print(
            "当前 Agent 没有配置 SkillRuntime。"
        )

        return

    status = runtime.status()

    available_skills = (
        status.get(
            "available_skills",
            [],
        )
    )

    if available_skills:

        print(
            "可用 Skills："
        )

        for skill in available_skills:

            loaded_text = (
                "已加载"
                if skill.get(
                    "loaded",
                    False,
                )
                else
                "未加载"
            )

            print(
                "  - "
                f"{skill.get('id', 'unknown')} "
                f"[{loaded_text}]"
            )

            print(
                "    "
                f"{skill.get('description', '')}"
            )

    else:

        print(
            "可用 Skills：无"
        )

    loaded_skill_ids = (
        status.get(
            "loaded_skill_ids",
            [],
        )
    )

    print(
        "已加载 Skills："
        + (
            ", ".join(
                loaded_skill_ids
            )
            if loaded_skill_ids
            else "无"
        )
    )

    active_tool_names = (
        agent.get_active_tool_names()
    )

    print(
        "当前 Active Tools："
        + (
            ", ".join(
                active_tool_names
            )
            if active_tool_names
            else "无"
        )
    )


# ============================================================
# Help
# ============================================================

def print_help() -> None:
    """打印命令帮助。"""

    print()

    print(
        "/skills               "
        "查看 Skill Catalog"
    )

    print(
        "/tools                "
        "查看 Active Tools"
    )

    print(
        "/jobs                 "
        "查看 GitHub Update Job 状态"
    )

    print(
        "/approve              "
        "批准当前 HITL Tool"
    )

    print(
        "/reject               "
        "拒绝当前 HITL Tool"
    )

    print(
        "/history              "
        "查看当前 thread 历史"
    )

    print(
        "/summary              "
        "查看滚动摘要"
    )

    print(
        "/memories             "
        "查看长期记忆"
    )

    print(
        "/flush-memory         "
        "立即整理长期记忆"
    )

    print(
        "/memory-report        "
        "查看最近记忆整理报告"
    )

    print(
        "/remember key=value   "
        "显式写入长期记忆"
    )

    print(
        "/forget key           "
        "删除长期记忆"
    )

    print(
        "/new                  "
        "创建新 thread"
    )

    print(
        "/exit                 "
        "退出"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """程序入口。"""

    args = parse_args()

    user_id = (
        normalize_user_id(
            args.user_id
        )
    )

    thread_id = (
        normalize_thread_id(
            args.thread_id
            if args.thread_id
            else create_thread_id()
        )
    )

    # ========================================================
    # 1. Secure Agent Runtime
    # ========================================================

    agent = (
        build_secure_agent(
            Path(
                args.config
            ).resolve()
        )
    )

    # ========================================================
    # 2. Scheduler
    #
    # 只检查一次。
    # 不启动 while/sleep Scheduler。
    # ========================================================

    scheduler_repository = (
        run_startup_scheduler_check()
    )

    # ========================================================
    # 3. Job Controller
    #
    # 这里就是之前缺失的连接。
    # ========================================================

    scheduled_update_controller = (
        ScheduledGithubUpdateController(
            agent=agent,
            user_id=user_id,
            repository=(
                scheduler_repository
            ),
        )
    )

    # ========================================================
    # 4. 自动启动所有尚未获得执行权的
    #    GitHub Scheduler Job
    #
    # WAITING_APPROVAL
    #       ↓
    # Single-Flight
    #       ↓
    # RUNNING
    #       ↓
    # Secure Agent
    #       ↓
    # HITL
    # ========================================================

    hitl_created_now = (
        auto_start_scheduled_github_jobs(
            controller=(
                scheduled_update_controller
            ),
        )
    )

    if not hitl_created_now:

        show_pending_scheduled_hitl(
            controller=(
                scheduled_update_controller
            ),
        )
    # ========================================================
    # 5. CLI
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "自动长期记忆多工具 Agent"
    )

    print(
        "=" * 80
    )

    print(
        f"user_id：{user_id}"
    )

    print(
        f"thread_id：{thread_id}"
    )

    print(
        "输入 /help 查看命令。"
    )

    print_skill_status(
        agent
    )

    # ========================================================
    # CLI 用户交互循环
    #
    # 注意：
    #
    # 这个 while True 与 Scheduler 无关。
    # Scheduler 已经在启动阶段检查结束。
    # ========================================================

    while True:

        try:

            user_input = input(
                "\n你："
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            report = (
                agent
                .flush_long_term_memory(
                    thread_id=(
                        thread_id
                    ),
                    user_id=user_id,
                    trigger=(
                        "console_interrupted"
                    ),
                )
            )

            print_flush_report(
                report
            )

            print(
                "程序已结束。"
            )

            break

        if not user_input:
            continue

        command = (
            user_input.lower()
        )

        # ----------------------------------------------------
        # HITL
        #
        # 必须放得比较靠前，
        # 避免 /approve 被当作普通聊天内容。
        # ----------------------------------------------------

        if handle_approval_command(
            command=command,
            agent=agent,
            controller=(
                scheduled_update_controller
            ),
            current_thread_id=(
                thread_id
            ),
            user_id=user_id,
        ):

            continue

        # ----------------------------------------------------
        # Help
        # ----------------------------------------------------

        if command == "/help":

            print_help()

            continue

        # ----------------------------------------------------
        # Job
        # ----------------------------------------------------

        if command == "/jobs":

            print_scheduler_jobs(
                repository=(
                    scheduler_repository
                ),
                controller=(
                    scheduled_update_controller
                ),
            )

            continue

        # ----------------------------------------------------
        # Skill
        # ----------------------------------------------------

        if command == "/skills":

            print_skill_status(
                agent
            )

            continue

        if command == "/tools":

            print()

            print(
                "当前 Active Tools："
                + ", ".join(
                    agent
                    .get_active_tool_names()
                )
            )

            continue

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if command in {
            "/exit",
            "/quit",
        }:

            report = (
                agent
                .flush_long_term_memory(
                    thread_id=(
                        thread_id
                    ),
                    user_id=user_id,
                    trigger=(
                        "session_exit"
                    ),
                )
            )

            print_flush_report(
                report
            )

            print(
                "程序已结束。"
            )

            break

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        if command == "/history":

            print_thread_history(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )

            continue

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        if command == "/summary":

            print_memory_status(
                agent,
                thread_id,
                maximum_characters=(
                    args.history_preview
                ),
            )

            continue

        # ----------------------------------------------------
        # Long-term Memory
        # ----------------------------------------------------

        if command == "/memories":

            print_long_term_memories(
                agent,
                user_id,
            )

            continue

        if command == "/flush-memory":

            report = (
                agent
                .flush_long_term_memory(
                    thread_id=(
                        thread_id
                    ),
                    user_id=user_id,
                    trigger=(
                        "manual_flush"
                    ),
                )
            )

            print_flush_report(
                report
            )

            continue

        if command == "/memory-report":

            report = (
                agent
                .get_last_auto_memory_report(
                    thread_id=(
                        thread_id
                    )
                )
            )

            print_flush_report(
                report
                if report
                else {
                    "status":
                        "尚未执行自动整理"
                }
            )

            continue

        # ----------------------------------------------------
        # New Thread
        # ----------------------------------------------------

        if command == "/new":

            report = (
                agent
                .flush_long_term_memory(
                    thread_id=thread_id,
                    user_id=user_id,
                    trigger=(
                        "before_new_thread"
                    ),
                )
            )

            print_flush_report(
                report
            )

            thread_id = (
                create_thread_id()
            )

            print()

            print(
                "已切换到新 thread："
                f"{thread_id}"
            )

            continue

        # ----------------------------------------------------
        # Remember
        # ----------------------------------------------------

        if command.startswith(
            "/remember "
        ):

            expression = (
                user_input[
                    len(
                        "/remember "
                    ):
                ].strip()
            )

            if "=" not in expression:

                print(
                    "格式："
                    "/remember key=value"
                )

                continue

            key, content = (
                expression.split(
                    "=",
                    maxsplit=1,
                )
            )

            result = (
                agent.remember(
                    user_id=user_id,
                    key=key,
                    content=content,
                )
            )

            print(
                "长期记忆已保存："
                f"{result['key']}"
            )

            continue

        # ----------------------------------------------------
        # Forget
        # ----------------------------------------------------

        if command.startswith(
            "/forget "
        ):

            key = (
                user_input[
                    len(
                        "/forget "
                    ):
                ].strip()
            )

            deleted = (
                agent.forget(
                    user_id=user_id,
                    key=key,
                )
            )

            print(
                "删除成功。"
                if deleted
                else
                "未找到该记忆。"
            )

            continue

        # ----------------------------------------------------
        # Normal Agent Request
        # ----------------------------------------------------
        #
        # 第一道保护：
        #
        # 如果这个普通聊天 thread 已经停在 HITL interrupt，
        # 不再把新的自然语言消息交给 Agent。
        #
        # SecureAgentRuntime.run() 内部还有第二道 Guard，
        # 因此即使未来 FastAPI / Worker 忘了做 CLI 层检查，
        # Runtime 仍不会破坏 ToolCall / ToolMessage 配对。

        if show_pending_agent_hitl(
            agent=agent,
            thread_id=thread_id,
        ):

            print(
                "本次普通输入未进入 Agent。"
            )

            continue

        try:

            result = agent.run(
                user_input,
                user_id=user_id,
                thread_id=thread_id,
            )

        except Exception as error:

            print(
                "执行失败："
                f"{type(error).__name__}："
                f"{error}"
            )

            continue

        # ----------------------------------------------------
        # 第二次检查非常重要。
        #
        # 当前这一次 agent.run() 本身可能刚刚执行到：
        #
        #     SecureToolNode -> interrupt()
        #
        # 此时 Graph 并没有正常完成业务 Tool，
        # 因此不能把 interrupt 前最后一条 AIMessage
        # 当成正常 Agent Answer 打印。
        # ----------------------------------------------------

        if show_pending_agent_hitl(
            agent=agent,
            thread_id=thread_id,
        ):

            continue

        print_result(
            result
        )

def show_pending_scheduled_hitl(
    *,
    controller: ScheduledGithubUpdateController,
) -> None:
    """CLI 重启时恢复展示尚未处理的 Scheduler HITL。"""

    waiting_run = (
        controller
        .get_waiting_tool_run()
    )

    if waiting_run is None:
        return

    print()
    print(
        "=" * 80
    )

    print(
        "发现尚未处理的高风险 Tool 审批"
    )

    print(
        "=" * 80
    )

    print(
        "run_id："
        f"{waiting_run.run_id}"
    )

    print(
        "job_name："
        f"{waiting_run.job_name}"
    )

    print(
        "status："
        f"{waiting_run.status.value}"
    )

    print(
        "agent_thread_id："
        f"{waiting_run.agent_thread_id}"
    )

    print()
    print(
        "该 Job 已经在之前的执行中"
        "触发 LangGraph HITL interrupt。"
    )

    print(
        "Checkpoint 已持久化，"
        "实际高风险 Tool 尚未继续执行。"
    )

    print()
    print(
        "输入："
    )

    print(
        "  /approve    批准并恢复执行"
    )

    print(
        "  /reject     拒绝并恢复执行"
    )

    print()
    print(
        "如果程序曾重启，"
        "恢复时会自动重新加载所需 Skill。"
    )

    print(
        "=" * 80
    )

if __name__ == "__main__":
    main()