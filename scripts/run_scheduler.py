"""Run the persistent RAG-LAB Scheduler.

这个进程只负责：

- 周期扫描 scheduled_job；
- 判断任务是否到期；
- 识别 Misfire；
- 创建 WAITING_APPROVAL Job Run。

它不会自动批准任务，
也不会直接执行 GitHub Update。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from raglab.scheduler.github_update_job_coordinator import (
    GITHUB_UPDATE_JOB_NAME,
)
from raglab.scheduler.job import (
    MisfirePolicy,
    ScheduleType,
)
from raglab.scheduler.job_repository import (
    DEFAULT_DATABASE_PATH,
    ScheduledJobRepository,
)
from raglab.scheduler.scheduler_service import (
    SchedulerService,
    SchedulerTickResult,
)


DEFAULT_POLL_SECONDS = 30.0

DEFAULT_GITHUB_UPDATE_TIME = (
    "08:00"
)

DEFAULT_GITHUB_TIMEZONE = (
    "Asia/Shanghai"
)


def parse_args(
) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "运行 RAG-LAB Persistent Scheduler。"
        )
    )

    parser.add_argument(
        "--database",
        type=str,
        default=str(
            DEFAULT_DATABASE_PATH
        ),
        help=(
            "Scheduler Control Plane SQLite 路径。"
        ),
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=(
            DEFAULT_POLL_SECONDS
        ),
        help=(
            "每隔多少秒扫描一次到期任务。"
        ),
    )

    parser.add_argument(
        "--misfire-grace-seconds",
        type=int,
        default=90,
        help=(
            "超过计划时间多少秒以后"
            "开始判定为 Misfire。"
        ),
    )

    parser.add_argument(
        "--github-time",
        type=str,
        default=(
            DEFAULT_GITHUB_UPDATE_TIME
        ),
        help=(
            "首次创建 GitHub Update Job 时"
            "使用的每日执行时间，例如 08:00。"
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "只执行一次 tick 后退出，"
            "用于调试。"
        ),
    )

    return parser.parse_args()


def ensure_default_github_job(
    repository: ScheduledJobRepository,
    *,
    schedule_expression: str,
) -> None:
    """确保默认 GitHub Update Job Definition 存在。

    注意：

    如果任务已经存在，
    这里绝对不会重新 upsert，

    因为重新 upsert 并传 next_run_at=None
    会破坏已经持久化的 Scheduler Cursor。
    """

    existing = (
        repository.get_job(
            GITHUB_UPDATE_JOB_NAME
        )
    )

    if existing is not None:

        print(
            "[Scheduler] GitHub Job 已存在："
        )

        print(
            "  job_name     = "
            f"{existing.job_name}"
        )

        print(
            "  schedule     = "
            f"{existing.schedule_expression}"
        )

        print(
            "  timezone     = "
            f"{existing.timezone}"
        )

        print(
            "  next_run_at  = "
            f"{existing.next_run_at}"
        )

        return

    created = (
        repository.upsert_job(
            job_name=(
                GITHUB_UPDATE_JOB_NAME
            ),

            schedule_type=(
                ScheduleType.DAILY
            ),

            schedule_expression=(
                schedule_expression
            ),

            timezone_name=(
                DEFAULT_GITHUB_TIMEZONE
            ),

            misfire_policy=(
                MisfirePolicy
                .COALESCE_RUN_ONCE
            ),

            requires_start_approval=True,

            enabled=True,

            next_run_at=None,
        )
    )

    print(
        "[Scheduler] 创建默认 GitHub Job："
    )

    print(
        "  job_name = "
        f"{created.job_name}"
    )

    print(
        "  schedule = "
        f"{created.schedule_expression}"
    )

    print(
        "  timezone = "
        f"{created.timezone}"
    )


def print_tick_result(
    result: SchedulerTickResult,
) -> None:
    """只输出有意义的 Scheduler 事件。"""

    for initialized in (
        result.initialized_jobs
    ):

        print(
            "[Scheduler] 初始化调度游标："
        )

        print(
            "  job      = "
            f"{initialized.job_name}"
        )

        print(
            "  next_run = "
            f"{initialized.next_run_at}"
        )

    for skipped in (
        result.skipped_misfires
    ):

        print(
            "[Scheduler] 跳过 Misfire："
        )

        print(
            "  job          = "
            f"{skipped.job_name}"
        )

        print(
            "  scheduled_at = "
            f"{skipped.scheduled_at}"
        )

        print(
            "  next_run_at  = "
            f"{skipped.next_run_at}"
        )

    for run in (
        result.created_runs
    ):

        print()
        print(
            "=" * 72
        )

        print(
            "[Scheduler] 创建待审批任务"
        )

        print(
            "=" * 72
        )

        print(
            "run_id       = "
            f"{run.run_id}"
        )

        print(
            "job_name     = "
            f"{run.job_name}"
        )

        print(
            "trigger_type = "
            f"{run.trigger_type.value}"
        )

        print(
            "scheduled_at = "
            f"{run.scheduled_at}"
        )

        print(
            "status        = "
            f"{run.status.value}"
        )

        print()
        print(
            "当前任务不会自动执行。"
        )

        print(
            "后续使用："
        )

        print(
            f"  /job-approve {run.run_id}"
        )

        print(
            "或："
        )

        print(
            f"  /job-reject {run.run_id} <原因>"
        )

        print(
            "=" * 72
        )


def main() -> None:

    args = parse_args()

    poll_seconds = float(
        args.poll_seconds
    )

    if poll_seconds <= 0:

        raise ValueError(
            "poll-seconds 必须大于 0。"
        )

    database_path = Path(
        args.database
    )

    repository = (
        ScheduledJobRepository(
            database_path=(
                database_path
            )
        )
    )

    repository.setup()

    ensure_default_github_job(
        repository,

        schedule_expression=(
            str(
                args.github_time
            ).strip()
        ),
    )

    scheduler = (
        SchedulerService(
            repository=repository,

            misfire_grace_seconds=(
                int(
                    args
                    .misfire_grace_seconds
                )
            ),
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        "RAG-LAB Scheduler 已启动"
    )

    print(
        "=" * 72
    )

    print(
        "数据库："
        f"{database_path}"
    )

    print(
        "扫描间隔："
        f"{poll_seconds} 秒"
    )

    print(
        "注意：Scheduler 只创建待审批任务，"
        "不会静默执行 GitHub Update。"
    )

    print(
        "=" * 72
    )

    try:

        while True:

            try:

                result = (
                    scheduler.tick()
                )

                print_tick_result(
                    result
                )

            except Exception as exc:

                print(
                    "[Scheduler ERROR] "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            if args.once:
                break

            time.sleep(
                poll_seconds
            )

    except KeyboardInterrupt:

        print()
        print(
            "[Scheduler] 已停止。"
        )


if __name__ == "__main__":
    main()