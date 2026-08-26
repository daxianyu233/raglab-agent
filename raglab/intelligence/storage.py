from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import NormalizedRepository


# 第一阶段及热点筛选阶段使用的 SQLite 数据库结构。
#
# 当前包含六张主要数据表：
#
# 1. repositories
#    保存仓库相对稳定的基础信息。
#
# 2. repository_snapshots
#    保存仓库每天的指标快照。
#
# 3. repository_discoveries
#    保存仓库当天通过什么来源和搜索词被发现。
#
# 4. collection_runs
#    保存每次采集任务的执行状态。
#
# 5. daily_repository_selections
#    保存每天进入后续深度处理名单的仓库。
#
# 6. repository_processing_state
#    保存仓库上一次真正完成深度分析时的状态。
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;


CREATE TABLE IF NOT EXISTS repositories (
    full_name TEXT PRIMARY KEY,
    github_id INTEGER,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    html_url TEXT NOT NULL,
    description TEXT,
    language TEXT,
    topics_json TEXT NOT NULL DEFAULT '[]',
    license_spdx TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    is_fork INTEGER NOT NULL DEFAULT 0,
    default_branch TEXT,
    created_at TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_repositories_github_id
ON repositories(github_id)
WHERE github_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_repositories_language
ON repositories(language);

CREATE INDEX IF NOT EXISTS idx_repositories_last_seen_at
ON repositories(last_seen_at);


CREATE TABLE IF NOT EXISTS repository_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    collected_at TEXT NOT NULL,

    stars INTEGER NOT NULL DEFAULT 0,
    forks INTEGER NOT NULL DEFAULT 0,
    open_issues INTEGER NOT NULL DEFAULT 0,
    subscribers INTEGER NOT NULL DEFAULT 0,

    pushed_at TEXT,
    updated_at TEXT,

    period_stars INTEGER,
    trending_rank INTEGER,
    trending_period TEXT,

    FOREIGN KEY(full_name)
        REFERENCES repositories(full_name)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    UNIQUE(full_name, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date
ON repository_snapshots(snapshot_date);

CREATE INDEX IF NOT EXISTS idx_snapshots_stars
ON repository_snapshots(stars);

CREATE INDEX IF NOT EXISTS idx_snapshots_period_stars
ON repository_snapshots(period_stars);

CREATE INDEX IF NOT EXISTS idx_snapshots_trending_rank
ON repository_snapshots(trending_rank);


CREATE TABLE IF NOT EXISTS repository_discoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,

    source TEXT NOT NULL,
    query_name TEXT NOT NULL DEFAULT '',
    result_rank INTEGER,

    FOREIGN KEY(full_name)
        REFERENCES repositories(full_name)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    UNIQUE(
        full_name,
        snapshot_date,
        source,
        query_name
    )
);

CREATE INDEX IF NOT EXISTS idx_discoveries_date
ON repository_discoveries(snapshot_date);

CREATE INDEX IF NOT EXISTS idx_discoveries_source
ON repository_discoveries(source);

CREATE INDEX IF NOT EXISTS idx_discoveries_query_name
ON repository_discoveries(query_name);


CREATE TABLE IF NOT EXISTS collection_runs (
    run_id TEXT PRIMARY KEY,

    started_at TEXT NOT NULL,
    finished_at TEXT,

    status TEXT NOT NULL,

    trending_count INTEGER NOT NULL DEFAULT 0,
    search_result_count INTEGER NOT NULL DEFAULT 0,
    deduped_count INTEGER NOT NULL DEFAULT 0,
    stored_count INTEGER NOT NULL DEFAULT 0,
    selected_count INTEGER NOT NULL DEFAULT 0,

    raw_directory TEXT,
    errors_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_collection_runs_started_at
ON collection_runs(started_at);

CREATE INDEX IF NOT EXISTS idx_collection_runs_status
ON collection_runs(status);


CREATE TABLE IF NOT EXISTS daily_repository_selections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    selected_at TEXT NOT NULL,

    full_name TEXT NOT NULL,

    selection_group TEXT NOT NULL,
    selection_score REAL NOT NULL DEFAULT 0,

    reasons_json TEXT NOT NULL DEFAULT '[]',

    is_new_repository INTEGER NOT NULL DEFAULT 0,
    is_new_trending INTEGER NOT NULL DEFAULT 0,

    new_queries_json TEXT NOT NULL DEFAULT '[]',

    star_growth INTEGER NOT NULL DEFAULT 0,
    star_growth_rate REAL NOT NULL DEFAULT 0,

    cooldown_passed INTEGER NOT NULL DEFAULT 0,

    FOREIGN KEY(run_id)
        REFERENCES collection_runs(run_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    FOREIGN KEY(full_name)
        REFERENCES repositories(full_name)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    UNIQUE(run_id, full_name)
);

CREATE INDEX IF NOT EXISTS idx_daily_selections_date
ON daily_repository_selections(snapshot_date);

CREATE INDEX IF NOT EXISTS idx_daily_selections_full_name
ON daily_repository_selections(full_name);

CREATE INDEX IF NOT EXISTS idx_daily_selections_group
ON daily_repository_selections(selection_group);

CREATE INDEX IF NOT EXISTS idx_daily_selections_score
ON daily_repository_selections(selection_score);


CREATE TABLE IF NOT EXISTS repository_processing_state (
    full_name TEXT PRIMARY KEY,

    last_processed_at TEXT NOT NULL,
    last_processed_date TEXT NOT NULL,

    last_processed_stars INTEGER NOT NULL DEFAULT 0,

    last_processed_description TEXT,

    last_processed_topics_json TEXT NOT NULL DEFAULT '[]',
    last_processed_queries_json TEXT NOT NULL DEFAULT '[]',

    processed_count INTEGER NOT NULL DEFAULT 1,

    FOREIGN KEY(full_name)
        REFERENCES repositories(full_name)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_processing_state_date
ON repository_processing_state(last_processed_date);
"""


class IntelligenceStore:
    """
    GitHub 热点情报的 SQLite 存储类。

    该类负责：

    1. 初始化和升级数据库；
    2. 保存仓库基础信息；
    3. 保存每日指标快照；
    4. 保存仓库发现来源；
    5. 保存每日筛选结果；
    6. 查询仓库历史状态；
    7. 记录仓库上次深度处理状态。
    """

    def __init__(
        self,
        database_path: Path,
    ) -> None:
        """
        初始化 SQLite 数据库。

        参数：
        database_path：
            SQLite 数据库文件路径。
        """
        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.database_path,
        )

        # 查询结果支持通过字段名访问。
        self.connection.row_factory = sqlite3.Row

        # 启用外键约束。
        self.connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        # 遇到数据库锁时最多等待 30 秒。
        self.connection.execute(
            "PRAGMA busy_timeout = 30000;"
        )

        self.connection.executescript(
            SCHEMA_SQL
        )

        # 兼容已经由旧代码创建的数据库。
        self._run_schema_migrations()

        self.connection.commit()

    def _run_schema_migrations(self) -> None:
        """
        对旧版数据库执行轻量迁移。

        旧版 collection_runs 表没有 selected_count 字段，
        因此这里检查并自动补充。
        """
        collection_run_columns = (
            self._get_table_columns(
                "collection_runs"
            )
        )

        if (
            "selected_count"
            not in collection_run_columns
        ):
            self.connection.execute(
                """
                ALTER TABLE collection_runs
                ADD COLUMN selected_count
                INTEGER NOT NULL DEFAULT 0
                """
            )

    def _get_table_columns(
        self,
        table_name: str,
    ) -> set[str]:
        """
        获取指定数据表的字段名称。
        """
        rows = self.connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()

        return {
            str(row["name"])
            for row in rows
        }

    @staticmethod
    def _loads_string_list(
        value: str | None,
    ) -> list[str]:
        """
        将数据库中的 JSON 字符串恢复为字符串列表。
        """
        if not value:
            return []

        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return []

        if not isinstance(
            data,
            list,
        ):
            return []

        return [
            str(item)
            for item in data
            if str(item).strip()
        ]

    def close(self) -> None:
        """
        关闭数据库连接。
        """
        self.connection.close()

    def __enter__(
        self,
    ) -> "IntelligenceStore":
        """
        支持使用 with 语句。
        """
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        """
        离开 with 语句时自动关闭数据库。
        """
        self.close()

    def begin_run(
        self,
        run_id: str,
        started_at: str,
        raw_directory: str,
    ) -> None:
        """
        记录一次采集任务开始执行。
        """
        self.connection.execute(
            """
            INSERT INTO collection_runs (
                run_id,
                started_at,
                status,
                raw_directory
            )
            VALUES (?, ?, 'running', ?)

            ON CONFLICT(run_id) DO UPDATE SET
                started_at = excluded.started_at,
                finished_at = NULL,
                status = 'running',
                trending_count = 0,
                search_result_count = 0,
                deduped_count = 0,
                stored_count = 0,
                selected_count = 0,
                raw_directory = excluded.raw_directory,
                errors_json = '[]'
            """,
            (
                run_id,
                started_at,
                raw_directory,
            ),
        )

        self.connection.commit()

    def finish_run(
        self,
        *,
        run_id: str,
        finished_at: str,
        status: str,
        trending_count: int,
        search_result_count: int,
        deduped_count: int,
        stored_count: int,
        errors: list[str],
        selected_count: int = 0,
    ) -> None:
        """
        更新一次采集任务的最终状态。

        status 通常为：

        success
        partial_success
        failed
        """
        errors_json = json.dumps(
            errors,
            ensure_ascii=False,
        )

        cursor = self.connection.execute(
            """
            UPDATE collection_runs
            SET
                finished_at = ?,
                status = ?,
                trending_count = ?,
                search_result_count = ?,
                deduped_count = ?,
                stored_count = ?,
                selected_count = ?,
                errors_json = ?
            WHERE run_id = ?
            """,
            (
                finished_at,
                status,
                trending_count,
                search_result_count,
                deduped_count,
                stored_count,
                selected_count,
                errors_json,
                run_id,
            ),
        )

        if cursor.rowcount == 0:
            raise RuntimeError(
                "没有找到需要更新的采集任务："
                f"{run_id}"
            )

        self.connection.commit()

    def upsert_repositories(
        self,
        repositories: Iterable[
            NormalizedRepository
        ],
    ) -> int:
        """
        保存规范化后的仓库信息。

        对每个仓库执行：

        1. 保存仓库基础信息；
        2. 保存当天指标快照；
        3. 保存当天发现来源。

        如果同一仓库在同一天重复采集，
        则更新当天快照，不重复创建多条记录。
        """
        stored_count = 0

        with self.connection:
            for repository in repositories:
                self._upsert_repository(
                    repository
                )

                self._upsert_snapshot(
                    repository
                )

                self._insert_discoveries(
                    repository
                )

                stored_count += 1

        return stored_count

    def _upsert_repository(
        self,
        repository: NormalizedRepository,
    ) -> None:
        """
        插入或更新仓库基础信息。
        """
        topics_json = json.dumps(
            repository.topics,
            ensure_ascii=False,
        )

        self.connection.execute(
            """
            INSERT INTO repositories (
                full_name,
                github_id,
                owner,
                name,
                html_url,
                description,
                language,
                topics_json,
                license_spdx,
                archived,
                is_fork,
                default_branch,
                created_at,
                first_seen_at,
                last_seen_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )

            ON CONFLICT(full_name) DO UPDATE SET
                github_id = COALESCE(
                    excluded.github_id,
                    repositories.github_id
                ),
                owner = excluded.owner,
                name = excluded.name,
                html_url = excluded.html_url,
                description = excluded.description,
                language = excluded.language,
                topics_json = excluded.topics_json,
                license_spdx = excluded.license_spdx,
                archived = excluded.archived,
                is_fork = excluded.is_fork,
                default_branch = excluded.default_branch,
                created_at = COALESCE(
                    excluded.created_at,
                    repositories.created_at
                ),
                last_seen_at = excluded.last_seen_at
            """,
            (
                repository.full_name,
                repository.github_id,
                repository.owner,
                repository.name,
                repository.html_url,
                repository.description,
                repository.language,
                topics_json,
                repository.license_spdx,
                int(repository.archived),
                int(repository.is_fork),
                repository.default_branch,
                repository.created_at,
                repository.collected_at,
                repository.collected_at,
            ),
        )

    def _upsert_snapshot(
        self,
        repository: NormalizedRepository,
    ) -> None:
        """
        插入或更新仓库当天的指标快照。
        """
        self.connection.execute(
            """
            INSERT INTO repository_snapshots (
                full_name,
                snapshot_date,
                collected_at,
                stars,
                forks,
                open_issues,
                subscribers,
                pushed_at,
                updated_at,
                period_stars,
                trending_rank,
                trending_period
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )

            ON CONFLICT(
                full_name,
                snapshot_date
            )
            DO UPDATE SET
                collected_at = excluded.collected_at,
                stars = excluded.stars,
                forks = excluded.forks,
                open_issues = excluded.open_issues,
                subscribers = excluded.subscribers,
                pushed_at = excluded.pushed_at,
                updated_at = excluded.updated_at,

                period_stars = COALESCE(
                    excluded.period_stars,
                    repository_snapshots.period_stars
                ),

                trending_rank = COALESCE(
                    excluded.trending_rank,
                    repository_snapshots.trending_rank
                ),

                trending_period = COALESCE(
                    excluded.trending_period,
                    repository_snapshots.trending_period
                )
            """,
            (
                repository.full_name,
                repository.snapshot_date,
                repository.collected_at,
                repository.stars,
                repository.forks,
                repository.open_issues,
                repository.subscribers,
                repository.pushed_at,
                repository.updated_at,
                repository.period_stars,
                repository.trending_rank,
                repository.trending_period,
            ),
        )

    def _insert_discoveries(
        self,
        repository: NormalizedRepository,
    ) -> None:
        """
        保存仓库当天的发现来源。
        """
        for source in sorted(
            set(repository.sources)
        ):
            # Search 的具体查询名称在下面单独保存。
            if source == "github_search":
                continue

            result_rank = None

            if source == "github_trending":
                result_rank = (
                    repository.trending_rank
                )

            self.connection.execute(
                """
                INSERT OR IGNORE INTO repository_discoveries (
                    full_name,
                    snapshot_date,
                    source,
                    query_name,
                    result_rank
                )
                VALUES (?, ?, ?, '', ?)
                """,
                (
                    repository.full_name,
                    repository.snapshot_date,
                    source,
                    result_rank,
                ),
            )

        for query_name in sorted(
            set(repository.search_queries)
        ):
            self.connection.execute(
                """
                INSERT OR IGNORE INTO repository_discoveries (
                    full_name,
                    snapshot_date,
                    source,
                    query_name,
                    result_rank
                )
                VALUES (
                    ?,
                    ?,
                    'github_search',
                    ?,
                    NULL
                )
                """,
                (
                    repository.full_name,
                    repository.snapshot_date,
                    query_name,
                ),
            )

    def get_selection_context(
        self,
        full_name: str,
        snapshot_date: str,
    ) -> dict[str, Any]:
        """
        获取一个候选仓库进入每日筛选时需要的历史信息。

        该方法必须在写入当天快照之前调用，
        否则当天数据会影响“是否首次出现”的判断。
        """

        # 判断此前日期是否已经存在该仓库快照。
        existed_before_row = self.connection.execute(
            """
            SELECT 1
            FROM repository_snapshots
            WHERE full_name = ?
              AND snapshot_date < ?
            LIMIT 1
            """,
            (
                full_name,
                snapshot_date,
            ),
        ).fetchone()

        # 获取最近一次、但早于今天的快照。
        previous_snapshot_row = (
            self.connection.execute(
                """
                SELECT
                    snapshot_date,
                    stars,
                    forks,
                    open_issues,
                    subscribers,
                    pushed_at,
                    updated_at,
                    period_stars,
                    trending_rank,
                    trending_period
                FROM repository_snapshots
                WHERE full_name = ?
                  AND snapshot_date < ?
                ORDER BY snapshot_date DESC
                LIMIT 1
                """,
                (
                    full_name,
                    snapshot_date,
                ),
            ).fetchone()
        )

        # 获取仓库上次真正完成深度分析时的状态。
        processing_state_row = (
            self.connection.execute(
                """
                SELECT
                    last_processed_at,
                    last_processed_date,
                    last_processed_stars,
                    last_processed_description,
                    last_processed_topics_json,
                    last_processed_queries_json,
                    processed_count
                FROM repository_processing_state
                WHERE full_name = ?
                """,
                (
                    full_name,
                ),
            ).fetchone()
        )

        # 查询该仓库最近一次进入每日处理名单的日期。
        last_selection_row = (
            self.connection.execute(
                """
                SELECT
                    snapshot_date,
                    selected_at,
                    selection_group,
                    selection_score
                FROM daily_repository_selections
                WHERE full_name = ?
                ORDER BY snapshot_date DESC, id DESC
                LIMIT 1
                """,
                (
                    full_name,
                ),
            ).fetchone()
        )

        # 查询今天之前曾经命中过哪些搜索规则。
        previous_query_rows = (
            self.connection.execute(
                """
                SELECT DISTINCT query_name
                FROM repository_discoveries
                WHERE full_name = ?
                  AND snapshot_date < ?
                  AND source = 'github_search'
                  AND query_name <> ''
                ORDER BY query_name
                """,
                (
                    full_name,
                    snapshot_date,
                ),
            ).fetchall()
        )

        previous_queries = [
            str(row["query_name"])
            for row in previous_query_rows
        ]

        # 判断今天之前是否进入过 Trending。
        ever_trending_row = (
            self.connection.execute(
                """
                SELECT 1
                FROM repository_discoveries
                WHERE full_name = ?
                  AND snapshot_date < ?
                  AND source = 'github_trending'
                LIMIT 1
                """,
                (
                    full_name,
                    snapshot_date,
                ),
            ).fetchone()
        )

        previous_snapshot = (
            dict(previous_snapshot_row)
            if previous_snapshot_row
            is not None
            else None
        )

        processing_state: dict[
            str,
            Any,
        ] | None = None

        if processing_state_row is not None:
            processing_state = dict(
                processing_state_row
            )

            processing_state[
                "last_processed_topics"
            ] = self._loads_string_list(
                processing_state.pop(
                    "last_processed_topics_json",
                    None,
                )
            )

            processing_state[
                "last_processed_queries"
            ] = self._loads_string_list(
                processing_state.pop(
                    "last_processed_queries_json",
                    None,
                )
            )

        last_selection = (
            dict(last_selection_row)
            if last_selection_row
            is not None
            else None
        )

        return {
            "full_name": full_name,
            "existed_before": (
                existed_before_row
                is not None
            ),
            "previous_snapshot": (
                previous_snapshot
            ),
            "processing_state": (
                processing_state
            ),
            "last_selection": (
                last_selection
            ),
            "previous_queries": (
                previous_queries
            ),
            "ever_trending_before": (
                ever_trending_row
                is not None
            ),
        }

    def save_daily_selections(
        self,
        *,
        run_id: str,
        snapshot_date: str,
        selected_at: str,
        selections: Iterable[
            dict[str, Any]
        ],
    ) -> int:
        """
        保存当天进入后续深度分析名单的仓库。

        selection 字典应包含：

        full_name
        selection_group
        selection_score
        reasons
        is_new_repository
        is_new_trending
        new_queries
        star_growth
        star_growth_rate
        cooldown_passed
        """
        saved_count = 0

        with self.connection:
            for selection in selections:
                reasons = selection.get(
                    "reasons"
                )

                if not isinstance(
                    reasons,
                    list,
                ):
                    reasons = []

                new_queries = selection.get(
                    "new_queries"
                )

                if not isinstance(
                    new_queries,
                    list,
                ):
                    new_queries = []

                self.connection.execute(
                    """
                    INSERT INTO daily_repository_selections (
                        run_id,
                        snapshot_date,
                        selected_at,
                        full_name,
                        selection_group,
                        selection_score,
                        reasons_json,
                        is_new_repository,
                        is_new_trending,
                        new_queries_json,
                        star_growth,
                        star_growth_rate,
                        cooldown_passed
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )

                    ON CONFLICT(run_id, full_name)
                    DO UPDATE SET
                        snapshot_date = excluded.snapshot_date,
                        selected_at = excluded.selected_at,
                        selection_group = excluded.selection_group,
                        selection_score = excluded.selection_score,
                        reasons_json = excluded.reasons_json,
                        is_new_repository = excluded.is_new_repository,
                        is_new_trending = excluded.is_new_trending,
                        new_queries_json = excluded.new_queries_json,
                        star_growth = excluded.star_growth,
                        star_growth_rate = excluded.star_growth_rate,
                        cooldown_passed = excluded.cooldown_passed
                    """,
                    (
                        run_id,
                        snapshot_date,
                        selected_at,
                        str(
                            selection["full_name"]
                        ),
                        str(
                            selection.get(
                                "selection_group"
                            )
                            or "unknown"
                        ),
                        float(
                            selection.get(
                                "selection_score"
                            )
                            or 0
                        ),
                        json.dumps(
                            reasons,
                            ensure_ascii=False,
                        ),
                        int(
                            bool(
                                selection.get(
                                    "is_new_repository"
                                )
                            )
                        ),
                        int(
                            bool(
                                selection.get(
                                    "is_new_trending"
                                )
                            )
                        ),
                        json.dumps(
                            new_queries,
                            ensure_ascii=False,
                        ),
                        int(
                            selection.get(
                                "star_growth"
                            )
                            or 0
                        ),
                        float(
                            selection.get(
                                "star_growth_rate"
                            )
                            or 0
                        ),
                        int(
                            bool(
                                selection.get(
                                    "cooldown_passed"
                                )
                            )
                        ),
                    ),
                )

                saved_count += 1

        return saved_count

    def mark_repositories_processed(
        self,
        *,
        repositories: Iterable[
            NormalizedRepository
        ],
        processed_at: str,
    ) -> int:
        """
        标记仓库已经真正完成深度分析。

        当前 GitHub 采集阶段暂时不会调用该方法。

        等后续完成 README、Discussion、Issue 和 LLM 摘要后，
        深度分析流程才调用该方法。
        """
        processed_count = 0

        with self.connection:
            for repository in repositories:
                self.connection.execute(
                    """
                    INSERT INTO repository_processing_state (
                        full_name,
                        last_processed_at,
                        last_processed_date,
                        last_processed_stars,
                        last_processed_description,
                        last_processed_topics_json,
                        last_processed_queries_json,
                        processed_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)

                    ON CONFLICT(full_name)
                    DO UPDATE SET
                        last_processed_at =
                            excluded.last_processed_at,
                        last_processed_date =
                            excluded.last_processed_date,
                        last_processed_stars =
                            excluded.last_processed_stars,
                        last_processed_description =
                            excluded.last_processed_description,
                        last_processed_topics_json =
                            excluded.last_processed_topics_json,
                        last_processed_queries_json =
                            excluded.last_processed_queries_json,
                        processed_count =
                            repository_processing_state.processed_count
                            + 1
                    """,
                    (
                        repository.full_name,
                        processed_at,
                        repository.snapshot_date,
                        repository.stars,
                        repository.description,
                        json.dumps(
                            repository.topics,
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            repository.search_queries,
                            ensure_ascii=False,
                        ),
                    ),
                )

                processed_count += 1

        return processed_count

    def get_table_names(
        self,
    ) -> list[str]:
        """
        返回当前数据库中的用户数据表名称。
        """
        rows = self.connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        return [
            str(row["name"])
            for row in rows
        ]

    def count_repositories(
        self,
    ) -> int:
        """
        返回数据库中的仓库总数。
        """
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM repositories
            """
        ).fetchone()

        if row is None:
            return 0

        return int(
            row["count"]
        )

    def count_snapshots(
        self,
    ) -> int:
        """
        返回数据库中的仓库快照总数。
        """
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM repository_snapshots
            """
        ).fetchone()

        if row is None:
            return 0

        return int(
            row["count"]
        )

    def count_daily_selections(
        self,
        snapshot_date: str,
    ) -> int:
        """
        返回指定日期的入选记录数量。

        同一天多次运行可能具有不同 run_id，
        因此这里统计的是记录总数，不是去重仓库数。
        """
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM daily_repository_selections
            WHERE snapshot_date = ?
            """,
            (
                snapshot_date,
            ),
        ).fetchone()

        if row is None:
            return 0

        return int(
            row["count"]
        )