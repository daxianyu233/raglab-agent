from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from datetime import date, datetime, time as datetime_time
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from .github_client import GitHubApiError, GitHubClient


def _load_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    读取 GitHub 情报系统 YAML 配置。
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{config_path}"
        )

    try:
        config_text = config_path.read_text(
            encoding="utf-8",
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            "配置文件不是有效的 UTF-8 编码："
            f"{config_path}"
        ) from exc

    try:
        config = yaml.safe_load(
            config_text
        )
    except yaml.YAMLError as exc:
        raise ValueError(
            "YAML 配置文件格式错误："
            f"{config_path}\n{exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            "配置文件根节点必须是 YAML 对象。"
        )

    return config


def _write_json(
    path: Path,
    data: Any,
) -> None:
    """
    将数据保存成 UTF-8 JSON 文件。
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_json(
    path: Path,
) -> Any:
    """
    读取 UTF-8 JSON 文件。
    """
    path = Path(path)

    try:
        text = path.read_text(
            encoding="utf-8",
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            "JSON 文件不是有效的 UTF-8 编码："
            f"{path}"
        ) from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "JSON 文件格式错误："
            f"{path}\n{exc}"
        ) from exc


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    将任意值安全转换为整数。
    """
    if value is None:
        return default

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    将任意值安全转换为浮点数。
    """
    if value is None:
        return default

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _loads_json_list(
    value: str | None,
) -> list[Any]:
    """
    将数据库中的 JSON 字符串恢复为列表。
    """
    if not value:
        return []

    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return data


def _normalize_snapshot_date(
    snapshot_date: str | None,
    local_timezone: ZoneInfo,
) -> date:
    """
    解析采集日期。

    没有传入日期时，使用配置时区中的当前日期。
    """
    if snapshot_date is None:
        return datetime.now(
            local_timezone
        ).date()

    try:
        return date.fromisoformat(
            snapshot_date
        )
    except ValueError as exc:
        raise ValueError(
            "snapshot_date 格式错误，"
            "必须使用 YYYY-MM-DD："
            f"{snapshot_date}"
        ) from exc


def _normalize_text(
    text: Any,
) -> str:
    """
    统一文本换行并清除行尾多余空格。

    不会折叠普通段落，也不会删除代码块。
    """
    if text is None:
        return ""

    normalized = str(text).replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = [
        line.rstrip()
        for line in normalized.split("\n")
    ]

    return "\n".join(lines).strip()


def _clean_markdown_for_analysis(
    text: Any,
) -> str:
    """
    对 Markdown 进行轻量清洗。

    原始文件仍然保存完整内容。
    这里只清除通常没有分析价值的内容：

    1. HTML 注释；
    2. 常见徽章行；
    3. 单独存在的图片标签；
    4. 过多连续空行。
    """
    cleaned = _normalize_text(
        text
    )

    if not cleaned:
        return ""

    cleaned = re.sub(
        r"<!--.*?-->",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    retained_lines: list[str] = []

    for line in cleaned.splitlines():
        stripped_line = line.strip()
        lowered_line = stripped_line.lower()

        is_badge_line = (
            (
                "shields.io" in lowered_line
                or "badge.svg" in lowered_line
                or "badgen.net" in lowered_line
            )
            and len(stripped_line) < 1000
        )

        is_single_image_line = (
            (
                stripped_line.startswith("![")
                and stripped_line.endswith(")")
            )
            or (
                stripped_line.startswith("<img")
                and stripped_line.endswith(">")
            )
        )

        if is_badge_line:
            continue

        if is_single_image_line:
            continue

        retained_lines.append(
            line
        )

    cleaned = "\n".join(
        retained_lines
    )

    # 最多保留两个连续换行。
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


def _truncate_text(
    text: Any,
    *,
    maximum_characters: int,
    head_ratio: float = 0.8,
) -> dict[str, Any]:
    """
    按字符数量裁剪文本。

    文本未超过限制时完整保留。

    文本超过限制时：
    - 保留前部内容；
    - 保留少量结尾内容；
    - 中间插入裁剪说明。
    """
    normalized_text = _clean_markdown_for_analysis(
        text
    )

    original_characters = len(
        normalized_text
    )

    maximum_characters = max(
        0,
        int(maximum_characters),
    )

    if maximum_characters == 0:
        return {
            "content": "",
            "original_characters": (
                original_characters
            ),
            "retained_characters": 0,
            "truncated": (
                original_characters > 0
            ),
        }

    if original_characters <= maximum_characters:
        return {
            "content": normalized_text,
            "original_characters": (
                original_characters
            ),
            "retained_characters": (
                original_characters
            ),
            "truncated": False,
        }

    normalized_head_ratio = min(
        max(
            float(head_ratio),
            0.5,
        ),
        1.0,
    )

    separator = (
        "\n\n"
        "[中间内容因长度限制已裁剪]"
        "\n\n"
    )

    available_characters = max(
        0,
        maximum_characters
        - len(separator),
    )

    head_characters = int(
        available_characters
        * normalized_head_ratio
    )

    tail_characters = (
        available_characters
        - head_characters
    )

    if tail_characters > 0:
        content = (
            normalized_text[
                :head_characters
            ]
            + separator
            + normalized_text[
                -tail_characters:
            ]
        )
    else:
        content = normalized_text[
            :maximum_characters
        ]

    return {
        "content": content,
        "original_characters": (
            original_characters
        ),
        "retained_characters": len(
            content
        ),
        "truncated": True,
    }


def _safe_repository_filename(
    full_name: str,
) -> str:
    """
    将 owner/repository 转换为 Windows 安全文件名。
    """
    normalized_name = full_name.replace(
        "/",
        "__",
    )

    normalized_name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        normalized_name,
    )

    normalized_name = normalized_name.strip(
        ". "
    )

    if not normalized_name:
        normalized_name = "unknown_repository"

    return normalized_name


def _deduplicate_selected_items(
    selected_items: list[Any],
) -> list[dict[str, Any]]:
    """
    按仓库 full_name 对入选结果去重。

    保留第一次出现的顺序。
    """
    deduplicated: list[
        dict[str, Any]
    ] = []

    seen_full_names: set[str] = set()

    for item in selected_items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        repository = item.get(
            "repository"
        )

        selection = item.get(
            "selection"
        )

        if not isinstance(
            repository,
            dict,
        ):
            continue

        if not isinstance(
            selection,
            dict,
        ):
            selection = {}

        full_name = str(
            repository.get(
                "full_name"
            )
            or ""
        ).strip()

        if full_name.count("/") != 1:
            continue

        if full_name in seen_full_names:
            continue

        seen_full_names.add(
            full_name
        )

        deduplicated.append(
            {
                "selection": selection,
                "repository": repository,
            }
        )

    return deduplicated


def _load_selected_items_from_json(
    selected_path: Path,
) -> list[dict[str, Any]]:
    """
    从第一阶段生成的 JSON 文件读取入选仓库。
    """
    if not selected_path.exists():
        return []

    data = _read_json(
        selected_path
    )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "入选仓库 JSON 的根节点必须是列表："
            f"{selected_path}"
        )

    return _deduplicate_selected_items(
        data
    )


def _load_discovery_context(
    connection: sqlite3.Connection,
    *,
    full_name: str,
    snapshot_date: str,
) -> tuple[list[str], list[str]]:
    """
    从 SQLite 读取仓库当天的发现来源和搜索主题。
    """
    rows = connection.execute(
        """
        SELECT
            source,
            query_name
        FROM repository_discoveries
        WHERE full_name = ?
          AND snapshot_date = ?
        ORDER BY source, query_name
        """,
        (
            full_name,
            snapshot_date,
        ),
    ).fetchall()

    sources: set[str] = set()
    search_queries: set[str] = set()

    for row in rows:
        source = str(
            row["source"]
            or ""
        ).strip()

        query_name = str(
            row["query_name"]
            or ""
        ).strip()

        if source:
            sources.add(
                source
            )

        if (
            source == "github_search"
            and query_name
        ):
            search_queries.add(
                query_name
            )

    return (
        sorted(sources),
        sorted(search_queries),
    )


def _load_selected_items_from_sqlite(
    database_path: Path,
    *,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    """
    从 SQLite 恢复指定日期最近一次非空入选名单。

    该逻辑用于处理：
    同一天重新运行第一阶段时，
    JSON 文件可能被空列表覆盖的情况。
    """
    database_path = Path(
        database_path
    )

    if not database_path.exists():
        return []

    connection = sqlite3.connect(
        database_path
    )

    connection.row_factory = (
        sqlite3.Row
    )

    try:
        latest_run_row = connection.execute(
            """
            SELECT
                run_id,
                MAX(selected_at) AS latest_selected_at
            FROM daily_repository_selections
            WHERE snapshot_date = ?
            GROUP BY run_id
            ORDER BY latest_selected_at DESC
            LIMIT 1
            """,
            (
                snapshot_date,
            ),
        ).fetchone()

        if latest_run_row is None:
            return []

        run_id = str(
            latest_run_row["run_id"]
        )

        rows = connection.execute(
            """
            SELECT
                ds.full_name AS full_name,
                ds.selection_group AS selection_group,
                ds.selection_score AS selection_score,
                ds.reasons_json AS reasons_json,
                ds.is_new_repository AS is_new_repository,
                ds.is_new_trending AS is_new_trending,
                ds.new_queries_json AS new_queries_json,
                ds.star_growth AS star_growth,
                ds.star_growth_rate AS star_growth_rate,
                ds.cooldown_passed AS cooldown_passed,

                r.github_id AS github_id,
                r.owner AS owner,
                r.name AS repository_name,
                r.html_url AS html_url,
                r.description AS description,
                r.language AS language,
                r.topics_json AS topics_json,
                r.license_spdx AS license_spdx,
                r.archived AS archived,
                r.is_fork AS is_fork,
                r.default_branch AS default_branch,
                r.created_at AS created_at,

                s.collected_at AS collected_at,
                s.stars AS stars,
                s.forks AS forks,
                s.open_issues AS open_issues,
                s.subscribers AS subscribers,
                s.pushed_at AS pushed_at,
                s.updated_at AS updated_at,
                s.period_stars AS period_stars,
                s.trending_rank AS trending_rank,
                s.trending_period AS trending_period,
                s.snapshot_date AS snapshot_date

            FROM daily_repository_selections AS ds

            INNER JOIN repositories AS r
                ON r.full_name = ds.full_name

            INNER JOIN repository_snapshots AS s
                ON s.full_name = ds.full_name
               AND s.snapshot_date = ds.snapshot_date

            WHERE ds.run_id = ?
            ORDER BY ds.selection_score DESC, ds.id ASC
            """,
            (
                run_id,
            ),
        ).fetchall()

        selected_items: list[
            dict[str, Any]
        ] = []

        for row in rows:
            full_name = str(
                row["full_name"]
            )

            (
                sources,
                search_queries,
            ) = _load_discovery_context(
                connection,
                full_name=full_name,
                snapshot_date=snapshot_date,
            )

            reasons = [
                str(value)
                for value
                in _loads_json_list(
                    row["reasons_json"]
                )
            ]

            new_queries = [
                str(value)
                for value
                in _loads_json_list(
                    row["new_queries_json"]
                )
            ]

            topics = [
                str(value)
                for value
                in _loads_json_list(
                    row["topics_json"]
                )
            ]

            selection = {
                "group": row[
                    "selection_group"
                ],
                "score": float(
                    row["selection_score"]
                    or 0
                ),
                "reasons": reasons,
                "is_new_repository": bool(
                    row["is_new_repository"]
                ),
                "is_new_trending": bool(
                    row["is_new_trending"]
                ),
                "new_queries": new_queries,
                "star_growth": int(
                    row["star_growth"]
                    or 0
                ),
                "star_growth_rate": float(
                    row["star_growth_rate"]
                    or 0
                ),
                "cooldown_passed": bool(
                    row["cooldown_passed"]
                ),
            }

            repository = {
                "full_name": full_name,
                "owner": row["owner"],
                "name": row[
                    "repository_name"
                ],
                "html_url": row[
                    "html_url"
                ],
                "description": row[
                    "description"
                ],
                "github_id": row[
                    "github_id"
                ],
                "language": row[
                    "language"
                ],
                "topics": topics,
                "license_spdx": row[
                    "license_spdx"
                ],
                "archived": bool(
                    row["archived"]
                ),
                "is_fork": bool(
                    row["is_fork"]
                ),
                "default_branch": row[
                    "default_branch"
                ],
                "created_at": row[
                    "created_at"
                ],
                "updated_at": row[
                    "updated_at"
                ],
                "pushed_at": row[
                    "pushed_at"
                ],
                "stars": int(
                    row["stars"]
                    or 0
                ),
                "forks": int(
                    row["forks"]
                    or 0
                ),
                "open_issues": int(
                    row["open_issues"]
                    or 0
                ),
                "subscribers": int(
                    row["subscribers"]
                    or 0
                ),
                "sources": sources,
                "search_queries": (
                    search_queries
                ),
                "trending_rank": row[
                    "trending_rank"
                ],
                "trending_period": row[
                    "trending_period"
                ],
                "period_stars": row[
                    "period_stars"
                ],
                "collected_at": row[
                    "collected_at"
                ],
                "snapshot_date": row[
                    "snapshot_date"
                ],
            }

            selected_items.append(
                {
                    "selection": selection,
                    "repository": repository,
                }
            )

        return _deduplicate_selected_items(
            selected_items
        )

    finally:
        connection.close()


def _resolve_selected_items(
    *,
    selected_json_path: Path,
    database_path: Path,
    snapshot_date: str,
) -> tuple[list[dict[str, Any]], str]:
    """
    确定本次深度采集使用的入选仓库来源。

    优先级：

    1. 当天非空 JSON 文件；
    2. SQLite 中当天最近一次非空入选名单。
    """
    selected_items = (
        _load_selected_items_from_json(
            selected_json_path
        )
    )

    if selected_items:
        return (
            selected_items,
            "json",
        )

    selected_items = (
        _load_selected_items_from_sqlite(
            database_path,
            snapshot_date=snapshot_date,
        )
    )

    if selected_items:
        return (
            selected_items,
            "sqlite",
        )

    return (
        [],
        "none",
    )


def _build_issue_since_time(
    *,
    current_date: date,
    lookback_days: int,
    local_timezone: ZoneInfo,
) -> str:
    """
    生成 GitHub Issues API 使用的 since 时间。

    GitHub REST API 使用 ISO 8601 UTC 时间。
    """
    start_date = current_date - timedelta(
        days=max(
            1,
            lookback_days,
        )
    )

    local_datetime = datetime.combine(
        start_date,
        datetime_time.min,
        tzinfo=local_timezone,
    )

    utc_datetime = local_datetime.astimezone(
        timezone.utc
    )

    return utc_datetime.isoformat().replace(
        "+00:00",
        "Z",
    )


def _prepare_readme_analysis(
    readme: dict[str, Any] | None,
    *,
    maximum_characters: int,
    head_ratio: float,
) -> dict[str, Any] | None:
    """
    生成供后续规则和 LLM 使用的 README 材料。
    """
    if readme is None:
        return None

    truncated = _truncate_text(
        readme.get(
            "content"
        ),
        maximum_characters=(
            maximum_characters
        ),
        head_ratio=head_ratio,
    )

    return {
        "name": readme.get(
            "name"
        ),
        "path": readme.get(
            "path"
        ),
        "sha": readme.get(
            "sha"
        ),
        "html_url": readme.get(
            "html_url"
        ),
        "original_characters": (
            truncated[
                "original_characters"
            ]
        ),
        "retained_characters": (
            truncated[
                "retained_characters"
            ]
        ),
        "truncated": truncated[
            "truncated"
        ],
        "content": truncated[
            "content"
        ],
    }


def _prepare_release_analysis(
    releases: list[dict[str, Any]],
    *,
    maximum_body_characters: int,
) -> list[dict[str, Any]]:
    """
    裁剪 Release 正文并保留主要元数据。
    """
    prepared_releases: list[
        dict[str, Any]
    ] = []

    for release in releases:
        truncated = _truncate_text(
            release.get(
                "body"
            ),
            maximum_characters=(
                maximum_body_characters
            ),
            head_ratio=0.85,
        )

        prepared_releases.append(
            {
                "id": release.get(
                    "id"
                ),
                "tag_name": release.get(
                    "tag_name"
                ),
                "name": release.get(
                    "name"
                ),
                "draft": release.get(
                    "draft"
                ),
                "prerelease": release.get(
                    "prerelease"
                ),
                "created_at": release.get(
                    "created_at"
                ),
                "published_at": (
                    release.get(
                        "published_at"
                    )
                ),
                "html_url": release.get(
                    "html_url"
                ),
                "body_original_characters": (
                    truncated[
                        "original_characters"
                    ]
                ),
                "body_retained_characters": (
                    truncated[
                        "retained_characters"
                    ]
                ),
                "body_truncated": (
                    truncated[
                        "truncated"
                    ]
                ),
                "body": truncated[
                    "content"
                ],
            }
        )

    return prepared_releases


def _prepare_issue_analysis(
    issues: list[dict[str, Any]],
    *,
    maximum_body_characters: int,
) -> list[dict[str, Any]]:
    """
    裁剪 Issue 正文并保留热度指标。
    """
    prepared_issues: list[
        dict[str, Any]
    ] = []

    for issue in issues:
        truncated = _truncate_text(
            issue.get(
                "body"
            ),
            maximum_characters=(
                maximum_body_characters
            ),
            head_ratio=0.9,
        )

        prepared_issues.append(
            {
                "number": issue.get(
                    "number"
                ),
                "title": issue.get(
                    "title"
                ),
                "state": issue.get(
                    "state"
                ),
                "labels": issue.get(
                    "labels"
                ),
                "comments": issue.get(
                    "comments"
                ),
                "reactions": issue.get(
                    "reactions"
                ),
                "created_at": issue.get(
                    "created_at"
                ),
                "updated_at": issue.get(
                    "updated_at"
                ),
                "closed_at": issue.get(
                    "closed_at"
                ),
                "html_url": issue.get(
                    "html_url"
                ),
                "body_original_characters": (
                    truncated[
                        "original_characters"
                    ]
                ),
                "body_retained_characters": (
                    truncated[
                        "retained_characters"
                    ]
                ),
                "body_truncated": (
                    truncated[
                        "truncated"
                    ]
                ),
                "body": truncated[
                    "content"
                ],
            }
        )

    return prepared_issues


def _calculate_analysis_character_count(
    analysis_record: dict[str, Any],
) -> int:
    """
    估算单个仓库分析材料中的文本字符总量。
    """
    total_characters = 0

    readme = analysis_record.get(
        "readme"
    )

    if isinstance(
        readme,
        dict,
    ):
        total_characters += len(
            str(
                readme.get(
                    "content"
                )
                or ""
            )
        )

    releases = analysis_record.get(
        "releases"
    )

    if isinstance(
        releases,
        list,
    ):
        for release in releases:
            if isinstance(
                release,
                dict,
            ):
                total_characters += len(
                    str(
                        release.get(
                            "body"
                        )
                        or ""
                    )
                )

    issues = analysis_record.get(
        "issues"
    )

    if isinstance(
        issues,
        list,
    ):
        for issue in issues:
            if isinstance(
                issue,
                dict,
            ):
                total_characters += len(
                    str(
                        issue.get(
                            "title"
                        )
                        or ""
                    )
                )

                total_characters += len(
                    str(
                        issue.get(
                            "body"
                        )
                        or ""
                    )
                )

    return total_characters


def collect_selected_repository_details(
    *,
    project_root: Path,
    config_path: Path,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    """
    对每日入选仓库执行深度信息采集。

    当前采集：

    1. README；
    2. Release；
    3. Issue。

    当前不执行：

    1. GitHub Discussion；
    2. LLM 摘要；
    3. ArXiv 检索；
    4. 动态关键词生成。
    """
    project_root = Path(
        project_root
    ).resolve()

    config_path = Path(
        config_path
    ).resolve()

    config = _load_config(
        config_path
    )

    timezone_name = str(
        config.get(
            "timezone"
        )
        or "Asia/Shanghai"
    )

    try:
        local_timezone = ZoneInfo(
            timezone_name
        )
    except Exception as exc:
        raise ValueError(
            "无效的时区配置："
            f"{timezone_name}"
        ) from exc

    current_date = _normalize_snapshot_date(
        snapshot_date,
        local_timezone,
    )

    snapshot_date_text = (
        current_date.isoformat()
    )

    started_datetime = datetime.now(
        local_timezone
    )

    started_at = (
        started_datetime.isoformat()
    )

    collection_id = (
        f"{snapshot_date_text}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    paths_config = config.get(
        "paths"
    )

    if not isinstance(
        paths_config,
        dict,
    ):
        paths_config = {}

    raw_root = project_root / str(
        paths_config.get(
            "raw_root"
        )
        or "data/intelligence/raw"
    )

    deep_raw_root = project_root / str(
        paths_config.get(
            "deep_raw_root"
        )
        or "data/intelligence/deep"
    )

    database_path = project_root / str(
        paths_config.get(
            "database"
        )
        or (
            "storage/intelligence/"
            "github_intelligence.sqlite3"
        )
    )

    selected_json_path = (
        raw_root
        / snapshot_date_text
        / "github_repositories_selected.json"
    )

    date_directory = (
        deep_raw_root
        / snapshot_date_text
    )

    collection_directory = (
        date_directory
        / collection_id
    )

    repositories_directory = (
        collection_directory
        / "repositories"
    )

    repositories_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    deep_config = config.get(
        "deep_collection"
    )

    if not isinstance(
        deep_config,
        dict,
    ):
        deep_config = {}

    deep_enabled = bool(
        deep_config.get(
            "enabled",
            True,
        )
    )

    if not deep_enabled:
        return {
            "collection_id": collection_id,
            "status": "disabled",
            "snapshot_date": (
                snapshot_date_text
            ),
            "selected_count": 0,
            "processed_count": 0,
            "errors": [
                "配置中的 deep_collection.enabled "
                "为 false。"
            ],
        }

    (
        selected_items,
        selected_source,
    ) = _resolve_selected_items(
        selected_json_path=selected_json_path,
        database_path=database_path,
        snapshot_date=snapshot_date_text,
    )

    maximum_repositories = max(
        0,
        _safe_int(
            deep_config.get(
                "maximum_repositories"
            ),
            15,
        ),
    )

    if maximum_repositories > 0:
        selected_items = selected_items[
            :maximum_repositories
        ]
    else:
        selected_items = []

    if not selected_items:
        raise RuntimeError(
            "没有找到可供深度采集的入选仓库。"
            "请先运行第一阶段采集，并检查："
            f"{selected_json_path}"
        )

    token = os.getenv(
        "GITHUB_TOKEN",
        "",
    ).strip()

    if not token:
        raise ValueError(
            "当前终端未设置 GITHUB_TOKEN。"
        )

    github_config = config.get(
        "github"
    )

    if not isinstance(
        github_config,
        dict,
    ):
        raise ValueError(
            "配置文件缺少 github 配置。"
        )

    readme_config = deep_config.get(
        "readme"
    )

    if not isinstance(
        readme_config,
        dict,
    ):
        readme_config = {}

    release_config = deep_config.get(
        "releases"
    )

    if not isinstance(
        release_config,
        dict,
    ):
        release_config = {}

    issue_config = deep_config.get(
        "issues"
    )

    if not isinstance(
        issue_config,
        dict,
    ):
        issue_config = {}

    readme_enabled = bool(
        readme_config.get(
            "enabled",
            True,
        )
    )

    readme_maximum_characters = max(
        0,
        _safe_int(
            readme_config.get(
                "maximum_characters"
            ),
            18000,
        ),
    )

    readme_head_ratio = min(
        max(
            _safe_float(
                readme_config.get(
                    "head_ratio"
                ),
                0.8,
            ),
            0.5,
        ),
        1.0,
    )

    releases_enabled = bool(
        release_config.get(
            "enabled",
            True,
        )
    )

    release_maximum_items = max(
        0,
        _safe_int(
            release_config.get(
                "maximum_items"
            ),
            3,
        ),
    )

    release_body_maximum_characters = max(
        0,
        _safe_int(
            release_config.get(
                "maximum_body_characters"
            ),
            4000,
        ),
    )

    issues_enabled = bool(
        issue_config.get(
            "enabled",
            True,
        )
    )

    issue_maximum_items = max(
        0,
        _safe_int(
            issue_config.get(
                "maximum_items"
            ),
            5,
        ),
    )

    issue_lookback_days = max(
        1,
        _safe_int(
            issue_config.get(
                "lookback_days"
            ),
            30,
        ),
    )

    issue_body_maximum_characters = max(
        0,
        _safe_int(
            issue_config.get(
                "maximum_body_characters"
            ),
            3000,
        ),
    )

    issue_state = str(
        issue_config.get(
            "state"
        )
        or "all"
    )

    issue_sort = str(
        issue_config.get(
            "sort"
        )
        or "comments"
    )

    issue_direction = str(
        issue_config.get(
            "direction"
        )
        or "desc"
    )

    repository_delay_seconds = max(
        0.0,
        _safe_float(
            deep_config.get(
                "repository_delay_seconds"
            ),
            0.3,
        ),
    )

    request_delay_seconds = max(
        0.0,
        _safe_float(
            deep_config.get(
                "request_delay_seconds"
            ),
            0.15,
        ),
    )

    issue_since = _build_issue_since_time(
        current_date=current_date,
        lookback_days=issue_lookback_days,
        local_timezone=local_timezone,
    )

    global_errors: list[str] = []

    raw_index: list[
        dict[str, Any]
    ] = []

    analysis_materials: list[
        dict[str, Any]
    ] = []

    readme_success_count = 0
    readme_missing_count = 0
    release_success_count = 0
    issue_success_count = 0
    repository_error_count = 0

    with GitHubClient(
        token=token,
        api_base_url=str(
            github_config.get(
                "api_base_url"
            )
            or "https://api.github.com"
        ),
        api_version=str(
            github_config.get(
                "api_version"
            )
            or "2022-11-28"
        ),
        timeout_seconds=_safe_int(
            github_config.get(
                "request_timeout_seconds"
            ),
            20,
        ),
        max_retries=_safe_int(
            github_config.get(
                "max_retries"
            ),
            3,
        ),
        retry_base_seconds=_safe_float(
            github_config.get(
                "retry_base_seconds"
            ),
            2.0,
        ),
    ) as github_client:

        for repository_index, selected_item in enumerate(
            selected_items,
            start=1,
        ):
            repository = selected_item[
                "repository"
            ]

            selection = selected_item[
                "selection"
            ]

            full_name = str(
                repository.get(
                    "full_name"
                )
                or ""
            ).strip()

            repository_errors: list[str] = []

            readme_data: dict[
                str,
                Any,
            ] | None = None

            release_data: list[
                dict[str, Any]
            ] = []

            issue_data: list[
                dict[str, Any]
            ] = []

            fetched_at = datetime.now(
                local_timezone
            ).isoformat()

            if readme_enabled:
                try:
                    readme_data = (
                        github_client.get_repository_readme(
                            full_name
                        )
                    )

                    if readme_data is None:
                        readme_missing_count += 1
                    else:
                        readme_success_count += 1

                except GitHubApiError as exc:
                    repository_errors.append(
                        "README 获取失败："
                        f"{exc}"
                    )

                if request_delay_seconds > 0:
                    time.sleep(
                        request_delay_seconds
                    )

            if (
                releases_enabled
                and release_maximum_items > 0
            ):
                try:
                    release_data = (
                        github_client.list_repository_releases(
                            full_name,
                            per_page=(
                                release_maximum_items
                            ),
                        )
                    )

                    release_success_count += 1

                except GitHubApiError as exc:
                    repository_errors.append(
                        "Release 获取失败："
                        f"{exc}"
                    )

                if request_delay_seconds > 0:
                    time.sleep(
                        request_delay_seconds
                    )

            if (
                issues_enabled
                and issue_maximum_items > 0
            ):
                try:
                    issue_data = (
                        github_client.list_repository_issues(
                            full_name,
                            per_page=(
                                issue_maximum_items
                            ),
                            state=issue_state,
                            sort=issue_sort,
                            direction=(
                                issue_direction
                            ),
                            since=issue_since,
                        )
                    )

                    issue_success_count += 1

                except GitHubApiError as exc:
                    repository_errors.append(
                        "Issue 获取失败："
                        f"{exc}"
                    )

            if repository_errors:
                repository_error_count += 1

            raw_record = {
                "collection_id": (
                    collection_id
                ),
                "snapshot_date": (
                    snapshot_date_text
                ),
                "fetched_at": fetched_at,
                "selection": selection,
                "repository": repository,
                "readme": readme_data,
                "releases": release_data,
                "issues": issue_data,
                "errors": repository_errors,
            }

            safe_filename = (
                _safe_repository_filename(
                    full_name
                )
                + ".json"
            )

            raw_repository_path = (
                repositories_directory
                / safe_filename
            )

            _write_json(
                raw_repository_path,
                raw_record,
            )

            readme_analysis = (
                _prepare_readme_analysis(
                    readme_data,
                    maximum_characters=(
                        readme_maximum_characters
                    ),
                    head_ratio=(
                        readme_head_ratio
                    ),
                )
            )

            release_analysis = (
                _prepare_release_analysis(
                    release_data,
                    maximum_body_characters=(
                        release_body_maximum_characters
                    ),
                )
            )

            issue_analysis = (
                _prepare_issue_analysis(
                    issue_data,
                    maximum_body_characters=(
                        issue_body_maximum_characters
                    ),
                )
            )

            analysis_record = {
                "collection_id": (
                    collection_id
                ),
                "snapshot_date": (
                    snapshot_date_text
                ),
                "fetched_at": fetched_at,
                "selection": selection,
                "repository": repository,
                "readme": readme_analysis,
                "releases": (
                    release_analysis
                ),
                "issues": issue_analysis,
                "collection_errors": (
                    repository_errors
                ),
            }

            analysis_record[
                "analysis_text_characters"
            ] = (
                _calculate_analysis_character_count(
                    analysis_record
                )
            )

            analysis_materials.append(
                analysis_record
            )

            raw_index.append(
                {
                    "full_name": full_name,
                    "raw_file": str(
                        raw_repository_path.relative_to(
                            collection_directory
                        )
                    ),
                    "readme_available": (
                        readme_data is not None
                    ),
                    "release_count": len(
                        release_data
                    ),
                    "issue_count": len(
                        issue_data
                    ),
                    "error_count": len(
                        repository_errors
                    ),
                }
            )

            if (
                repository_delay_seconds > 0
                and repository_index
                < len(selected_items)
            ):
                time.sleep(
                    repository_delay_seconds
                )

        rate_limit_data: dict[
            str,
            Any,
        ] = {}

        try:
            rate_limit_data = (
                github_client.get_rate_limit_status()
            )
        except Exception as exc:
            global_errors.append(
                "GitHub API 限额信息获取失败："
                f"{exc}"
            )

        captured_rate_limits = (
            github_client.get_captured_rate_limits()
        )

    processed_count = len(
        analysis_materials
    )

    total_analysis_characters = sum(
        int(
            item.get(
                "analysis_text_characters"
            )
            or 0
        )
        for item in analysis_materials
    )

    if (
        processed_count == len(selected_items)
        and repository_error_count == 0
        and not global_errors
    ):
        status = "success"
    elif processed_count > 0:
        status = "partial_success"
    else:
        status = "failed"

    finished_at = datetime.now(
        local_timezone
    ).isoformat()

    index_path = (
        collection_directory
        / "github_repository_deep_index.json"
    )

    analysis_material_path = (
        collection_directory
        / "github_repository_analysis_material.json"
    )

    summary_path = (
        collection_directory
        / "deep_collection_summary.json"
    )

    rate_limit_path = (
        collection_directory
        / "github_rate_limit.json"
    )

    _write_json(
        index_path,
        raw_index,
    )

    _write_json(
        analysis_material_path,
        analysis_materials,
    )

    _write_json(
        rate_limit_path,
        {
            "api_response": rate_limit_data,
            "captured_headers": (
                captured_rate_limits
            ),
        },
    )

    summary = {
        "collection_id": collection_id,
        "status": status,
        "snapshot_date": (
            snapshot_date_text
        ),
        "started_at": started_at,
        "finished_at": finished_at,
        "selected_source": (
            selected_source
        ),
        "selected_json_path": str(
            selected_json_path
        ),
        "selected_count": len(
            selected_items
        ),
        "processed_count": (
            processed_count
        ),
        "repository_error_count": (
            repository_error_count
        ),
        "readme_success_count": (
            readme_success_count
        ),
        "readme_missing_count": (
            readme_missing_count
        ),
        "release_request_success_count": (
            release_success_count
        ),
        "issue_request_success_count": (
            issue_success_count
        ),
        "total_release_count": sum(
            len(
                item.get(
                    "releases"
                )
                or []
            )
            for item in analysis_materials
        ),
        "total_issue_count": sum(
            len(
                item.get(
                    "issues"
                )
                or []
            )
            for item in analysis_materials
        ),
        "total_analysis_characters": (
            total_analysis_characters
        ),
        "collection_directory": str(
            collection_directory
        ),
        "raw_index_path": str(
            index_path
        ),
        "analysis_material_path": str(
            analysis_material_path
        ),
        "errors": global_errors,
    }

    _write_json(
        summary_path,
        summary,
    )

    # 保存一个轻量指针，方便后续阶段找到当天最近一次深度采集。
    latest_pointer_path = (
        date_directory
        / "latest_collection.json"
    )

    _write_json(
        latest_pointer_path,
        {
            "collection_id": collection_id,
            "status": status,
            "snapshot_date": (
                snapshot_date_text
            ),
            "collection_directory": str(
                collection_directory
            ),
            "summary_path": str(
                summary_path
            ),
            "analysis_material_path": str(
                analysis_material_path
            ),
        },
    )

    return {
        **summary,
        "summary_path": str(
            summary_path
        ),
        "latest_pointer_path": str(
            latest_pointer_path
        ),
        "repositories": raw_index,
        "rate_limit": rate_limit_data,
    }