"""Persistent Scheduler service.

职责：

- 读取 scheduled_job；
- 判断 next_run_at 是否到期；
- 识别正常触发和 Misfire；
- 按 Misfire Policy 创建 Job Run；
- 推进 last_scheduled_at / next_run_at。

本模块不负责：

- 执行 Agent；
- 处理 Tool HITL；
- 执行 Job Start Approval；
- 调用具体 GitHub Pipeline。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    time as datetime_time,
    timedelta,
    timezone,
)
from zoneinfo import ZoneInfo

from raglab.scheduler.job import (
    JobRunStatus,
    JobTriggerType,
    MisfirePolicy,
    ScheduledJob,
    ScheduledJobRun,
    ScheduleType,
)
from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)


def utc_now() -> datetime:
    """获取当前 UTC 时间。"""

    return datetime.now(
        timezone.utc
    )


def datetime_to_utc_text(
    value: datetime,
) -> str:
    """统一转换为带时区的 UTC ISO 字符串。"""

    if value.tzinfo is None:
        raise ValueError(
            "datetime 必须包含时区。"
        )

    return (
        value
        .astimezone(
            timezone.utc
        )
        .isoformat()
    )


def parse_utc_text(
    value: str,
) -> datetime:
    """读取数据库中的 UTC ISO 时间。"""

    parsed = datetime.fromisoformat(
        str(
            value
        ).strip()
    )

    if parsed.tzinfo is None:

        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


@dataclass(frozen=True)
class InitializedJob:
    """第一次初始化 next_run_at 的 Job。"""

    job_name: str

    next_run_at: str


@dataclass(frozen=True)
class SkippedMisfire:
    """因 MisfirePolicy.SKIP 被跳过的计划。"""

    job_name: str

    scheduled_at: str

    next_run_at: str


@dataclass(frozen=True)
class SchedulerTickResult:
    """一次 Scheduler Tick 的结果。"""

    checked_job_count: int

    initialized_jobs: tuple[
        InitializedJob,
        ...,
    ]

    created_runs: tuple[
        ScheduledJobRun,
        ...,
    ]

    skipped_misfires: tuple[
        SkippedMisfire,
        ...,
    ]


class SchedulerService:
    """持久化 Scheduler 的时间判定层。"""

    def __init__(
        self,
        repository: ScheduledJobRepository,
        *,
        misfire_grace_seconds: int = 90,
    ) -> None:

        self.repository = repository

        self.repository.setup()

        self.misfire_grace_seconds = int(
            misfire_grace_seconds
        )

        if (
            self.misfire_grace_seconds
            < 0
        ):
            raise ValueError(
                "misfire_grace_seconds "
                "不能小于 0。"
            )

    # ========================================================
    # Tick
    # ========================================================

    def tick(
        self,
        *,
        now_utc: (
            datetime
            | None
        ) = None,
    ) -> SchedulerTickResult:
        """执行一次调度扫描。

        tick() 本身只执行一次检查。

        常驻 Scheduler 进程会：

            while True:
                tick()
                sleep(...)

        从而周期性扫描任务。
        """

        now = (
            now_utc
            or utc_now()
        )

        if now.tzinfo is None:

            raise ValueError(
                "now_utc 必须包含时区。"
            )

        now = now.astimezone(
            timezone.utc
        )

        jobs = (
            self.repository
            .list_jobs(
                enabled_only=True
            )
        )

        initialized_jobs: list[
            InitializedJob
        ] = []

        created_runs: list[
            ScheduledJobRun
        ] = []

        skipped_misfires: list[
            SkippedMisfire
        ] = []

        for job in jobs:

            # ------------------------------------------------
            # Scheduler V1：
            #
            # 当前只正式接入需要 Job Start Approval
            # 的 GitHub Update。
            #
            # 暂时不实现“完全无人值守自动执行”。
            # ------------------------------------------------

            if (
                not job
                .requires_start_approval
            ):

                raise RuntimeError(
                    "Scheduler V1 当前只支持 "
                    "requires_start_approval=True "
                    "的任务；"
                    f"当前任务：{job.job_name}"
                )

            # ------------------------------------------------
            # 第一次初始化。
            #
            # next_run_at 为空说明 Scheduler
            # 从未为这个 Job 建立调度游标。
            # ------------------------------------------------

            if not job.next_run_at:

                next_run = (
                    self._initial_next_run(
                        job=job,
                        now_utc=now,
                    )
                )

                next_run_text = (
                    datetime_to_utc_text(
                        next_run
                    )
                )

                self.repository.update_schedule_cursor(
                    job_name=(
                        job.job_name
                    ),

                    last_scheduled_at=(
                        job.last_scheduled_at
                    ),

                    next_run_at=(
                        next_run_text
                    ),
                )

                initialized_jobs.append(
                    InitializedJob(
                        job_name=(
                            job.job_name
                        ),
                        next_run_at=(
                            next_run_text
                        ),
                    )
                )

                continue

            # ------------------------------------------------
            # 已经有 next_run_at。
            # ------------------------------------------------

            scheduled_at = (
                parse_utc_text(
                    job.next_run_at
                )
            )

            # 还没到时间。
            if scheduled_at > now:
                continue

            # ------------------------------------------------
            # 当前时间已经 >= scheduled_at。
            # ------------------------------------------------

            lateness_seconds = max(
                0.0,
                (
                    now
                    - scheduled_at
                ).total_seconds(),
            )

            # 不论是正常触发还是 Misfire，
            # 本次处理完以后都直接推进到
            # “now 之后的下一次执行时间”。
            #
            # 这样机器关闭三天也不会：
            #
            # 11号补一次
            # 12号再补一次
            # 13号再补一次
            #
            # 而是合并处理。
            next_run = (
                self._next_future_run(
                    job=job,

                    scheduled_at=(
                        scheduled_at
                    ),

                    now_utc=now,
                )
            )

            next_run_text = (
                datetime_to_utc_text(
                    next_run
                )
            )

            scheduled_at_text = (
                datetime_to_utc_text(
                    scheduled_at
                )
            )

            # ------------------------------------------------
            # MISFIRE + SKIP
            # ------------------------------------------------

            if (
                lateness_seconds
                >
                self.misfire_grace_seconds

                and

                job.misfire_policy
                ==
                MisfirePolicy.SKIP
            ):

                self.repository.update_schedule_cursor(
                    job_name=(
                        job.job_name
                    ),

                    last_scheduled_at=(
                        scheduled_at_text
                    ),

                    next_run_at=(
                        next_run_text
                    ),
                )

                skipped_misfires.append(
                    SkippedMisfire(
                        job_name=(
                            job.job_name
                        ),

                        scheduled_at=(
                            scheduled_at_text
                        ),

                        next_run_at=(
                            next_run_text
                        ),
                    )
                )

                continue

            # ------------------------------------------------
            # 判断触发来源。
            #
            # Scheduler 每隔几十秒扫描一次，
            # 所以不可能刚好精确到 08:00:00。
            #
            # 因此允许一个小的 Grace Window。
            # ------------------------------------------------

            if (
                lateness_seconds
                <=
                self.misfire_grace_seconds
            ):

                trigger_type = (
                    JobTriggerType
                    .SCHEDULED
                )

            else:

                trigger_type = (
                    JobTriggerType
                    .MISFIRE_CATCH_UP
                )

            # ------------------------------------------------
            # Scheduler 这里只创建 Run。
            #
            # 不执行 Agent。
            #
            # 下一步由用户：
            #
            # /job-approve <run_id>
            #
            # 才真正进入 JobExecutionService。
            # ------------------------------------------------

            run = (
                self.repository
                .create_run(
                    job=job,

                    trigger_type=(
                        trigger_type
                    ),

                    scheduled_at=(
                        scheduled_at_text
                    ),

                    status=(
                        JobRunStatus
                        .WAITING_APPROVAL
                    ),
                )
            )

            # ------------------------------------------------
            # 创建成功后再推进游标。
            # ------------------------------------------------

            self.repository.update_schedule_cursor(
                job_name=(
                    job.job_name
                ),

                last_scheduled_at=(
                    scheduled_at_text
                ),

                next_run_at=(
                    next_run_text
                ),
            )

            created_runs.append(
                run
            )

        return SchedulerTickResult(
            checked_job_count=(
                len(
                    jobs
                )
            ),

            initialized_jobs=tuple(
                initialized_jobs
            ),

            created_runs=tuple(
                created_runs
            ),

            skipped_misfires=tuple(
                skipped_misfires
            ),
        )

    # ========================================================
    # Initial Cursor
    # ========================================================

    def _initial_next_run(
        self,
        *,
        job: ScheduledJob,
        now_utc: datetime,
    ) -> datetime:
        """第一次启用 Scheduler 时初始化未来执行时间。

        一个非常重要的规则：

        如果 Scheduler 从来没有运行过，
        不追溯 Scheduler 存在之前的计划时间。

        例如：

            今天18:00第一次启用
            schedule=08:00

        不会立刻认为：

            “今天08:00错过了！”

        而是：

            next_run_at = 明天08:00

        只有以后 next_run_at 已经持久化，
        机器离线错过它，
        才算真正 Misfire。
        """

        if (
            job.schedule_type
            ==
            ScheduleType.DAILY
        ):

            return (
                self._next_daily_after(
                    job=job,
                    after_utc=(
                        now_utc
                    ),
                )
            )

        if (
            job.schedule_type
            ==
            ScheduleType.INTERVAL
        ):

            interval_seconds = (
                self._parse_interval_seconds(
                    job.schedule_expression
                )
            )

            return (
                now_utc
                +
                timedelta(
                    seconds=(
                        interval_seconds
                    )
                )
            )

        raise ValueError(
            "不支持的 schedule_type："
            f"{job.schedule_type}"
        )

    # ========================================================
    # Cursor Advance
    # ========================================================

    def _next_future_run(
        self,
        *,
        job: ScheduledJob,
        scheduled_at: datetime,
        now_utc: datetime,
    ) -> datetime:
        """将 Cursor 一次推进到当前时间之后。

        例如：

            原 next_run_at
            = 8月11日 08:00

            当前
            = 8月14日 14:00

        COALESCE_RUN_ONCE：

            只创建一次 MISFIRE_CATCH_UP

        然后直接：

            next_run_at
            = 8月15日 08:00
        """

        if (
            job.schedule_type
            ==
            ScheduleType.DAILY
        ):

            return (
                self._next_daily_after(
                    job=job,
                    after_utc=(
                        now_utc
                    ),
                )
            )

        if (
            job.schedule_type
            ==
            ScheduleType.INTERVAL
        ):

            interval_seconds = (
                self._parse_interval_seconds(
                    job.schedule_expression
                )
            )

            if (
                scheduled_at
                >
                now_utc
            ):
                return (
                    scheduled_at
                )

            elapsed_seconds = (
                now_utc
                - scheduled_at
            ).total_seconds()

            steps = (
                int(
                    elapsed_seconds
                    //
                    interval_seconds
                )
                + 1
            )

            return (
                scheduled_at
                +
                timedelta(
                    seconds=(
                        steps
                        *
                        interval_seconds
                    )
                )
            )

        raise ValueError(
            "不支持的 schedule_type："
            f"{job.schedule_type}"
        )

    # ========================================================
    # Daily
    # ========================================================

    def _next_daily_after(
        self,
        *,
        job: ScheduledJob,
        after_utc: datetime,
    ) -> datetime:

        hour, minute = (
            self._parse_daily_expression(
                job.schedule_expression
            )
        )

        job_timezone = (
            ZoneInfo(
                job.timezone
            )
        )

        local_after = (
            after_utc
            .astimezone(
                job_timezone
            )
        )

        local_candidate = (
            datetime.combine(
                local_after.date(),

                datetime_time(
                    hour=hour,
                    minute=minute,
                ),

                tzinfo=(
                    job_timezone
                ),
            )
        )

        if (
            local_candidate
            <=
            local_after
        ):

            local_candidate = (
                local_candidate
                +
                timedelta(
                    days=1
                )
            )

        return (
            local_candidate
            .astimezone(
                timezone.utc
            )
        )

    @staticmethod
    def _parse_daily_expression(
        expression: str,
    ) -> tuple[
        int,
        int,
    ]:

        text = str(
            expression
        ).strip()

        parts = text.split(
            ":"
        )

        if len(parts) != 2:

            raise ValueError(
                "DAILY schedule_expression "
                "必须是 HH:MM，"
                f"实际值：{text!r}"
            )

        try:

            hour = int(
                parts[0]
            )

            minute = int(
                parts[1]
            )

        except ValueError as exc:

            raise ValueError(
                "DAILY schedule_expression "
                "必须是 HH:MM，"
                f"实际值：{text!r}"
            ) from exc

        if not (
            0 <= hour <= 23
        ):

            raise ValueError(
                f"小时范围错误：{hour}"
            )

        if not (
            0 <= minute <= 59
        ):

            raise ValueError(
                f"分钟范围错误：{minute}"
            )

        return (
            hour,
            minute,
        )

    # ========================================================
    # Interval
    # ========================================================

    @staticmethod
    def _parse_interval_seconds(
        expression: str,
    ) -> int:

        text = str(
            expression
        ).strip()

        try:

            seconds = int(
                text
            )

        except ValueError as exc:

            raise ValueError(
                "INTERVAL schedule_expression "
                "当前使用秒数表示，"
                "例如 3600。"
            ) from exc

        if seconds <= 0:

            raise ValueError(
                "INTERVAL 秒数必须大于 0。"
            )

        return seconds