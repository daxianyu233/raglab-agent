from __future__ import annotations

import json
import math
import os
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from .github_client import GitHubApiError, GitHubClient
from .github_trending import fetch_github_trending
from .models import (
    NormalizedRepository,
    SearchResult,
    TrendingRepository,
)
from .storage import IntelligenceStore


def _load_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    读取 YAML 配置文件。
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
            "配置文件 YAML 格式错误："
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
    将数据保存为 UTF-8 JSON 文件。
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


def _write_text(
    path: Path,
    text: str,
) -> None:
    """
    保存 UTF-8 文本文件。
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    将值安全转换为整数。
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
    将值安全转换为浮点数。
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


def _normalize_topics(
    value: Any,
) -> list[str]:
    """
    将 GitHub Topics 转换为去重后的字符串列表。
    """
    if not isinstance(value, list):
        return []

    topics = {
        str(topic).strip()
        for topic in value
        if str(topic).strip()
    }

    return sorted(topics)


def _extract_license_spdx(
    detail: dict[str, Any],
) -> str | None:
    """
    从仓库详情中提取许可证 SPDX 编号。
    """
    license_data = detail.get(
        "license"
    )

    if not isinstance(
        license_data,
        dict,
    ):
        return None

    spdx_id = license_data.get(
        "spdx_id"
    )

    if not spdx_id:
        return None

    normalized_spdx = str(
        spdx_id
    ).strip()

    if normalized_spdx in {
        "",
        "NOASSERTION",
        "OTHER",
    }:
        return None

    return normalized_spdx


def _parse_iso_date(
    value: str | None,
) -> date | None:
    """
    将 ISO 日期或日期时间转换成 date。

    支持：

    2026-08-01
    2026-08-01T12:30:00+08:00
    2026-08-01T04:30:00Z
    """
    if not value:
        return None

    normalized_value = str(
        value
    ).strip()

    if not normalized_value:
        return None

    try:
        return date.fromisoformat(
            normalized_value[:10]
        )
    except ValueError:
        pass

    try:
        return datetime.fromisoformat(
            normalized_value.replace(
                "Z",
                "+00:00",
            )
        ).date()
    except ValueError:
        return None


def _calculate_days_since(
    earlier_date_text: str | None,
    current_date: date,
) -> int | None:
    """
    计算某个日期距离当前日期多少天。
    """
    earlier_date = _parse_iso_date(
        earlier_date_text
    )

    if earlier_date is None:
        return None

    return max(
        0,
        (
            current_date
            - earlier_date
        ).days,
    )


def _build_repository_from_api(
    *,
    detail: dict[str, Any],
    sources: list[str],
    search_queries: list[str],
    trending: TrendingRepository | None,
    collected_at: str,
    snapshot_date: str,
    trending_period: str,
) -> NormalizedRepository:
    """
    使用仓库详情 API 结果构建统一仓库对象。
    """
    full_name = str(
        detail.get("full_name")
        or ""
    ).strip()

    if full_name.count("/") != 1:
        raise ValueError(
            "仓库详情缺少有效的 full_name："
            f"{full_name}"
        )

    default_owner, default_name = (
        full_name.split(
            "/",
            1,
        )
    )

    owner_data = detail.get(
        "owner"
    )

    if isinstance(
        owner_data,
        dict,
    ):
        owner = str(
            owner_data.get("login")
            or default_owner
        ).strip()
    else:
        owner = default_owner

    repository_name = str(
        detail.get("name")
        or default_name
    ).strip()

    html_url = str(
        detail.get("html_url")
        or f"https://github.com/{full_name}"
    ).strip()

    return NormalizedRepository(
        full_name=full_name,
        owner=owner,
        name=repository_name,
        html_url=html_url,
        description=detail.get(
            "description"
        ),
        github_id=(
            _safe_int(
                detail.get("id")
            )
            if detail.get("id")
            is not None
            else None
        ),
        language=detail.get(
            "language"
        ),
        topics=_normalize_topics(
            detail.get("topics")
        ),
        license_spdx=_extract_license_spdx(
            detail
        ),
        archived=bool(
            detail.get(
                "archived",
                False,
            )
        ),
        is_fork=bool(
            detail.get(
                "fork",
                False,
            )
        ),
        default_branch=detail.get(
            "default_branch"
        ),
        created_at=detail.get(
            "created_at"
        ),
        updated_at=detail.get(
            "updated_at"
        ),
        pushed_at=detail.get(
            "pushed_at"
        ),
        stars=_safe_int(
            detail.get(
                "stargazers_count"
            )
        ),
        forks=_safe_int(
            detail.get(
                "forks_count"
            )
        ),
        open_issues=_safe_int(
            detail.get(
                "open_issues_count"
            )
        ),
        subscribers=_safe_int(
            detail.get(
                "subscribers_count"
            )
        ),
        sources=sorted(
            set(sources)
        ),
        search_queries=sorted(
            set(search_queries)
        ),
        trending_rank=(
            trending.rank
            if trending is not None
            else None
        ),
        trending_period=(
            trending_period
            if trending is not None
            else None
        ),
        period_stars=(
            trending.period_stars
            if trending is not None
            else None
        ),
        collected_at=collected_at,
        snapshot_date=snapshot_date,
    )


def _build_repository_from_fallback(
    *,
    full_name: str,
    trending: TrendingRepository | None,
    search_item: dict[str, Any] | None,
    sources: list[str],
    search_queries: list[str],
    collected_at: str,
    snapshot_date: str,
    trending_period: str,
) -> NormalizedRepository:
    """
    仓库详情 API 请求失败时，
    使用 Trending 或 Search 数据构建统一对象。
    """
    owner, repository_name = (
        full_name.split(
            "/",
            1,
        )
    )

    item = search_item or {}

    description = item.get(
        "description"
    )

    if (
        not description
        and trending is not None
    ):
        description = (
            trending.description
        )

    language = item.get(
        "language"
    )

    if (
        not language
        and trending is not None
    ):
        language = (
            trending.language
        )

    html_url = str(
        item.get("html_url")
        or (
            trending.html_url
            if trending is not None
            else (
                "https://github.com/"
                f"{full_name}"
            )
        )
    )

    if (
        item.get(
            "stargazers_count"
        )
        is not None
    ):
        stars = _safe_int(
            item.get(
                "stargazers_count"
            )
        )
    elif trending is not None:
        stars = (
            trending.total_stars
        )
    else:
        stars = 0

    if (
        item.get(
            "forks_count"
        )
        is not None
    ):
        forks = _safe_int(
            item.get(
                "forks_count"
            )
        )
    elif trending is not None:
        forks = trending.forks
    else:
        forks = 0

    return NormalizedRepository(
        full_name=full_name,
        owner=owner,
        name=repository_name,
        html_url=html_url,
        description=description,
        github_id=(
            _safe_int(
                item.get("id")
            )
            if item.get("id")
            is not None
            else None
        ),
        language=language,
        topics=_normalize_topics(
            item.get("topics")
        ),
        archived=bool(
            item.get(
                "archived",
                False,
            )
        ),
        is_fork=bool(
            item.get(
                "fork",
                False,
            )
        ),
        default_branch=item.get(
            "default_branch"
        ),
        created_at=item.get(
            "created_at"
        ),
        updated_at=item.get(
            "updated_at"
        ),
        pushed_at=item.get(
            "pushed_at"
        ),
        stars=stars,
        forks=forks,
        open_issues=_safe_int(
            item.get(
                "open_issues_count"
            )
        ),
        subscribers=0,
        sources=sorted(
            set(sources)
        ),
        search_queries=sorted(
            set(search_queries)
        ),
        trending_rank=(
            trending.rank
            if trending is not None
            else None
        ),
        trending_period=(
            trending_period
            if trending is not None
            else None
        ),
        period_stars=(
            trending.period_stars
            if trending is not None
            else None
        ),
        collected_at=collected_at,
        snapshot_date=snapshot_date,
    )


def _build_search_query(
    *,
    query_template: str,
    since_date: date,
) -> str:
    """
    将配置中的 {since} 替换成实际日期。
    """
    return query_template.format(
        since=since_date.isoformat()
    )


def _candidate_sort_key(
    item: tuple[str, dict[str, Any]],
) -> tuple[int, int, int]:
    """
    决定仓库详情补全顺序。

    优先级：

    1. 来自 Trending；
    2. 当前周期新增 Star；
    3. 仓库总 Star。
    """
    _, candidate = item

    trending = candidate.get(
        "trending"
    )

    search_item = candidate.get(
        "search_item"
    )

    is_trending = int(
        isinstance(
            trending,
            TrendingRepository,
        )
    )

    period_stars = (
        trending.period_stars
        if isinstance(
            trending,
            TrendingRepository,
        )
        else 0
    )

    search_stars = 0

    if isinstance(
        search_item,
        dict,
    ):
        search_stars = _safe_int(
            search_item.get(
                "stargazers_count"
            )
        )

    return (
        is_trending,
        period_stars,
        search_stars,
    )


def _calculate_selection_information(
    *,
    repository: NormalizedRepository,
    context: dict[str, Any],
    selection_config: dict[str, Any],
    current_date: date,
) -> dict[str, Any]:
    """
    根据仓库当前状态和历史状态，
    计算是否值得进入今日深度处理名单。
    """
    is_new_repository = not bool(
        context.get(
            "existed_before"
        )
    )

    is_trending = (
        "github_trending"
        in repository.sources
    )

    ever_trending_before = bool(
        context.get(
            "ever_trending_before"
        )
    )

    is_new_trending = (
        is_trending
        and not ever_trending_before
    )

    previous_queries_value = (
        context.get(
            "previous_queries"
        )
    )

    if not isinstance(
        previous_queries_value,
        list,
    ):
        previous_queries_value = []

    previous_queries = {
        str(query)
        for query
        in previous_queries_value
    }

    current_queries = {
        str(query)
        for query
        in repository.search_queries
    }

    new_queries = sorted(
        current_queries
        - previous_queries
    )

    previous_snapshot = context.get(
        "previous_snapshot"
    )

    processing_state = context.get(
        "processing_state"
    )

    last_selection = context.get(
        "last_selection"
    )

    baseline_stars = 0

    if isinstance(
        processing_state,
        dict,
    ):
        baseline_stars = _safe_int(
            processing_state.get(
                "last_processed_stars"
            )
        )
    elif isinstance(
        previous_snapshot,
        dict,
    ):
        baseline_stars = _safe_int(
            previous_snapshot.get(
                "stars"
            )
        )

    star_growth = max(
        0,
        repository.stars
        - baseline_stars,
    )

    if baseline_stars > 0:
        star_growth_rate = (
            star_growth
            / baseline_stars
        )
    else:
        star_growth_rate = 0.0

    minimum_star_growth = _safe_int(
        selection_config.get(
            "minimum_star_growth"
        ),
        50,
    )

    minimum_star_growth_rate = (
        _safe_float(
            selection_config.get(
                "minimum_star_growth_rate"
            ),
            0.05,
        )
    )

    significant_star_growth = (
        baseline_stars > 0
        and (
            star_growth
            >= minimum_star_growth
            or star_growth_rate
            >= minimum_star_growth_rate
        )
    )

    cooldown_days = max(
        0,
        _safe_int(
            selection_config.get(
                "cooldown_days"
            ),
            7,
        ),
    )

    last_reference_date: str | None = None

    if isinstance(
        processing_state,
        dict,
    ):
        last_reference_date = str(
            processing_state.get(
                "last_processed_date"
            )
            or ""
        )

    if (
        not last_reference_date
        and isinstance(
            last_selection,
            dict,
        )
    ):
        last_reference_date = str(
            last_selection.get(
                "snapshot_date"
            )
            or ""
        )

    days_since_reference = (
        _calculate_days_since(
            last_reference_date,
            current_date,
        )
    )

    if days_since_reference is None:
        cooldown_passed = True
    else:
        cooldown_passed = (
            days_since_reference
            >= cooldown_days
        )

    selected_today_before = False

    if isinstance(
        last_selection,
        dict,
    ):
        selected_today_before = (
            str(
                last_selection.get(
                    "snapshot_date"
                )
                or ""
            )
            == repository.snapshot_date
        )

    process_on_new_trending = bool(
        selection_config.get(
            "process_on_new_trending",
            True,
        )
    )

    process_on_new_query = bool(
        selection_config.get(
            "process_on_new_query",
            True,
        )
    )

    prioritize_new_repositories = bool(
        selection_config.get(
            "prioritize_new_repositories",
            True,
        )
    )

    reasons: list[str] = []

    if is_new_repository:
        reasons.append(
            "首次发现该仓库"
        )

    if is_new_trending:
        reasons.append(
            "首次进入 GitHub Trending"
        )

    if new_queries:
        reasons.append(
            "首次命中搜索主题："
            + "、".join(new_queries)
        )

    if significant_star_growth:
        reasons.append(
            "Star 明显增长："
            f"+{star_growth:,}，"
            f"增长率 {star_growth_rate:.2%}"
        )

    if (
        cooldown_passed
        and not is_new_repository
    ):
        reasons.append(
            "距离上次处理已超过冷却期"
        )

    should_process = False

    if (
        is_new_repository
        and prioritize_new_repositories
    ):
        should_process = True

    if (
        is_new_trending
        and process_on_new_trending
    ):
        should_process = True

    if (
        new_queries
        and process_on_new_query
    ):
        should_process = True

    if significant_star_growth:
        should_process = True

    if (
        cooldown_passed
        and not is_new_repository
    ):
        should_process = True

    # 同一天已经进入过筛选名单时，
    # 不再重复占用当天名额。
    if selected_today_before:
        should_process = False
        reasons.append(
            "今天已经进入过处理名单"
        )

    selection_score = 0.0

    if is_trending:
        selection_score += 100.0

        selection_score += min(
            math.log1p(
                max(
                    repository.period_stars
                    or 0,
                    0,
                )
            )
            * 12.0,
            100.0,
        )

        if repository.trending_rank:
            selection_score += max(
                0.0,
                30.0
                - float(
                    repository.trending_rank
                ),
            )

    if is_new_repository:
        selection_score += 50.0

    if is_new_trending:
        selection_score += 50.0

    selection_score += (
        len(new_queries)
        * 15.0
    )

    if len(
        repository.search_queries
    ) > 1:
        selection_score += (
            len(
                repository.search_queries
            )
            * 8.0
        )

    if significant_star_growth:
        selection_score += 40.0

        selection_score += min(
            math.log1p(
                star_growth
            )
            * 5.0,
            40.0,
        )

    if cooldown_passed:
        selection_score += 10.0

    selection_score += min(
        math.log1p(
            max(
                repository.stars,
                0,
            )
        )
        * 2.0,
        30.0,
    )

    if selected_today_before:
        selection_score = -1.0

    revisit_triggered = (
        not is_new_repository
        and (
            is_new_trending
            or bool(new_queries)
            or significant_star_growth
            or cooldown_passed
        )
    )

    return {
        "full_name": repository.full_name,
        "repository": repository,
        "should_process": should_process,
        "selection_score": round(
            selection_score,
            4,
        ),
        "reasons": reasons,
        "is_new_repository": (
            is_new_repository
        ),
        "is_trending": is_trending,
        "is_new_trending": (
            is_new_trending
        ),
        "new_queries": new_queries,
        "star_growth": star_growth,
        "star_growth_rate": round(
            star_growth_rate,
            6,
        ),
        "cooldown_passed": (
            cooldown_passed
        ),
        "selected_today_before": (
            selected_today_before
        ),
        "revisit_triggered": (
            revisit_triggered
        ),
    }


def _selection_score_key(
    item: dict[str, Any],
) -> tuple[float, int, int]:
    """
    默认筛选排序规则。
    """
    repository = item[
        "repository"
    ]

    return (
        float(
            item.get(
                "selection_score"
            )
            or 0
        ),
        _safe_int(
            repository.period_stars
        ),
        repository.stars,
    )


def _exploration_sort_key(
    item: dict[str, Any],
) -> tuple[str, str, int]:
    """
    探索项目排序规则。

    探索位置优先考虑：

    1. 创建时间较新的项目；
    2. 最近推送时间较新的项目；
    3. 已经获得一定关注但不完全由总 Star 主导。
    """
    repository = item[
        "repository"
    ]

    return (
        str(
            repository.created_at
            or ""
        ),
        str(
            repository.pushed_at
            or ""
        ),
        min(
            repository.stars,
            10_000,
        ),
    )


def _select_daily_repositories(
    *,
    repositories: list[
        NormalizedRepository
    ],
    contexts: dict[
        str,
        dict[str, Any]
    ],
    selection_config: dict[str, Any],
    current_date: date,
) -> list[dict[str, Any]]:
    """
    根据每日上限和分类配额，
    选择真正进入后续深度分析的仓库。
    """
    enabled = bool(
        selection_config.get(
            "enabled",
            True,
        )
    )

    daily_limit = max(
        0,
        _safe_int(
            selection_config.get(
                "daily_processing_limit"
            ),
            15,
        ),
    )

    if (
        not enabled
        or daily_limit == 0
    ):
        return []

    trending_limit = max(
        0,
        _safe_int(
            selection_config.get(
                "trending_limit"
            ),
            8,
        ),
    )

    search_limit = max(
        0,
        _safe_int(
            selection_config.get(
                "search_limit"
            ),
            5,
        ),
    )

    revisit_limit = max(
        0,
        _safe_int(
            selection_config.get(
                "revisit_limit"
            ),
            2,
        ),
    )

    exploration_slots = max(
        0,
        _safe_int(
            selection_config.get(
                "exploration_slots"
            ),
            2,
        ),
    )

    # 探索位置属于搜索项目配额的一部分。
    exploration_slots = min(
        exploration_slots,
        search_limit,
    )

    regular_search_limit = max(
        0,
        search_limit
        - exploration_slots,
    )

    evaluated: list[
        dict[str, Any]
    ] = []

    for repository in repositories:
        context = contexts.get(
            repository.full_name,
            {},
        )

        evaluated_item = (
            _calculate_selection_information(
                repository=repository,
                context=context,
                selection_config=selection_config,
                current_date=current_date,
            )
        )

        if evaluated_item[
            "should_process"
        ]:
            evaluated.append(
                evaluated_item
            )

    selected: list[
        dict[str, Any]
    ] = []

    selected_names: set[str] = set()

    def append_selected(
        item: dict[str, Any],
        group: str,
    ) -> bool:
        """
        添加一条入选记录，并执行全局去重和总量控制。
        """
        full_name = str(
            item["full_name"]
        )

        if full_name in selected_names:
            return False

        if len(selected) >= daily_limit:
            return False

        selected_item = dict(
            item
        )

        selected_item[
            "selection_group"
        ] = group

        selected.append(
            selected_item
        )

        selected_names.add(
            full_name
        )

        return True

    # -------------------------------------------------
    # 一、Trending 项目
    # -------------------------------------------------
    trending_candidates = [
        item
        for item in evaluated
        if item["is_trending"]
    ]

    trending_candidates.sort(
        key=_selection_score_key,
        reverse=True,
    )

    for item in trending_candidates[
        :trending_limit
    ]:
        append_selected(
            item,
            "trending",
        )

    # -------------------------------------------------
    # 二、普通搜索发现的新项目
    # -------------------------------------------------
    new_search_candidates = [
        item
        for item in evaluated
        if (
            not item["is_trending"]
            and item[
                "is_new_repository"
            ]
            and bool(
                item[
                    "repository"
                ].search_queries
            )
        )
    ]

    new_search_candidates.sort(
        key=_selection_score_key,
        reverse=True,
    )

    regular_search_selected = 0

    for item in new_search_candidates:
        if (
            regular_search_selected
            >= regular_search_limit
        ):
            break

        if append_selected(
            item,
            "search_new",
        ):
            regular_search_selected += 1

    # -------------------------------------------------
    # 三、探索项目
    # -------------------------------------------------
    exploration_candidates = [
        item
        for item in new_search_candidates
        if item["full_name"]
        not in selected_names
    ]

    exploration_candidates.sort(
        key=_exploration_sort_key,
        reverse=True,
    )

    exploration_selected = 0

    for item in exploration_candidates:
        if (
            exploration_selected
            >= exploration_slots
        ):
            break

        if append_selected(
            item,
            "exploration",
        ):
            exploration_selected += 1

    # -------------------------------------------------
    # 四、需要重新处理的旧项目
    # -------------------------------------------------
    revisit_candidates = [
        item
        for item in evaluated
        if (
            item[
                "revisit_triggered"
            ]
            and not item[
                "is_new_repository"
            ]
            and item["full_name"]
            not in selected_names
        )
    ]

    revisit_candidates.sort(
        key=_selection_score_key,
        reverse=True,
    )

    revisit_selected = 0

    for item in revisit_candidates:
        if (
            revisit_selected
            >= revisit_limit
        ):
            break

        if append_selected(
            item,
            "revisit",
        ):
            revisit_selected += 1

    return selected[
        :daily_limit
    ]


def _selection_to_storage_dict(
    selection: dict[str, Any],
) -> dict[str, Any]:
    """
    将内部筛选结果转换成 SQLite 所需结构。
    """
    return {
        "full_name": (
            selection["full_name"]
        ),
        "selection_group": (
            selection.get(
                "selection_group"
            )
            or "unknown"
        ),
        "selection_score": (
            selection.get(
                "selection_score"
            )
            or 0
        ),
        "reasons": (
            selection.get(
                "reasons"
            )
            or []
        ),
        "is_new_repository": bool(
            selection.get(
                "is_new_repository"
            )
        ),
        "is_new_trending": bool(
            selection.get(
                "is_new_trending"
            )
        ),
        "new_queries": (
            selection.get(
                "new_queries"
            )
            or []
        ),
        "star_growth": (
            selection.get(
                "star_growth"
            )
            or 0
        ),
        "star_growth_rate": (
            selection.get(
                "star_growth_rate"
            )
            or 0
        ),
        "cooldown_passed": bool(
            selection.get(
                "cooldown_passed"
            )
        ),
    }


def _selection_to_json_dict(
    selection: dict[str, Any],
) -> dict[str, Any]:
    """
    将筛选结果转换成适合保存到 JSON 的结构。
    """
    repository = selection[
        "repository"
    ]

    return {
        "selection": {
            "group": selection.get(
                "selection_group"
            ),
            "score": selection.get(
                "selection_score"
            ),
            "reasons": selection.get(
                "reasons"
            ),
            "is_new_repository": (
                selection.get(
                    "is_new_repository"
                )
            ),
            "is_new_trending": (
                selection.get(
                    "is_new_trending"
                )
            ),
            "new_queries": (
                selection.get(
                    "new_queries"
                )
            ),
            "star_growth": (
                selection.get(
                    "star_growth"
                )
            ),
            "star_growth_rate": (
                selection.get(
                    "star_growth_rate"
                )
            ),
            "cooldown_passed": (
                selection.get(
                    "cooldown_passed"
                )
            ),
        },
        "repository": (
            repository.to_dict()
        ),
    }


def collect_github_intelligence(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """
    执行 GitHub 热点情报采集。

    完整流程：

    1. 获取 GitHub Trending；
    2. 执行 Repository Search；
    3. 合并并去重候选仓库；
    4. 限制仓库详情补全数量；
    5. 查询仓库历史状态；
    6. 计算每日处理名单；
    7. 保存原始 JSON；
    8. 写入 SQLite；
    9. 返回采集摘要。
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
        config.get("timezone")
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

    started_datetime = datetime.now(
        local_timezone
    )

    current_date = (
        started_datetime.date()
    )

    collected_at = (
        started_datetime.isoformat()
    )

    snapshot_date = (
        current_date.isoformat()
    )

    run_id = (
        f"{snapshot_date}-"
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
        paths_config.get("raw_root")
        or "data/intelligence/raw"
    )

    database_path = project_root / str(
        paths_config.get("database")
        or (
            "storage/intelligence/"
            "github_intelligence.sqlite3"
        )
    )

    raw_directory = (
        raw_root
        / snapshot_date
    )

    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
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

    selection_config = config.get(
        "selection"
    )

    if not isinstance(
        selection_config,
        dict,
    ):
        selection_config = {}

    token = os.getenv(
        "GITHUB_TOKEN",
        "",
    ).strip()

    if not token:
        raise ValueError(
            "当前终端未设置 GITHUB_TOKEN。"
        )

    errors: list[str] = []

    trending_repositories: list[
        TrendingRepository
    ] = []

    search_results: list[
        SearchResult
    ] = []

    raw_search_responses: list[
        dict[str, Any]
    ] = []

    raw_repository_details: list[
        dict[str, Any]
    ] = []

    normalized_repositories: list[
        NormalizedRepository
    ] = []

    selected_repositories: list[
        dict[str, Any]
    ] = []

    rate_limit_data: dict[
        str,
        Any,
    ] = {}

    stored_count = 0
    selected_count = 0
    deduped_count = 0

    with IntelligenceStore(
        database_path
    ) as store:
        store.begin_run(
            run_id=run_id,
            started_at=collected_at,
            raw_directory=str(
                raw_directory
            ),
        )

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
            timeout_seconds=int(
                github_config.get(
                    "request_timeout_seconds"
                )
                or 20
            ),
            max_retries=int(
                github_config.get(
                    "max_retries"
                )
                or 3
            ),
            retry_base_seconds=float(
                github_config.get(
                    "retry_base_seconds"
                )
                or 2
            ),
        ) as github_client:

            # ---------------------------------------------
            # 一、采集 GitHub Trending
            # ---------------------------------------------
            trending_config = (
                github_config.get(
                    "trending"
                )
            )

            if not isinstance(
                trending_config,
                dict,
            ):
                trending_config = {}

            trending_period = str(
                trending_config.get(
                    "since"
                )
                or "daily"
            )

            if bool(
                trending_config.get(
                    "enabled",
                    True,
                )
            ):
                try:
                    (
                        trending_repositories,
                        trending_html,
                    ) = fetch_github_trending(
                        since=trending_period,
                        spoken_language_code=str(
                            trending_config.get(
                                "spoken_language_code"
                            )
                            or ""
                        ),
                        timeout_seconds=int(
                            github_config.get(
                                "request_timeout_seconds"
                            )
                            or 20
                        ),
                    )

                    _write_text(
                        raw_directory
                        / "github_trending.html",
                        trending_html,
                    )

                except Exception as exc:
                    errors.append(
                        "GitHub Trending 采集失败："
                        f"{exc}"
                    )

            _write_json(
                raw_directory
                / "github_trending.json",
                [
                    repository.to_dict()
                    for repository
                    in trending_repositories
                ],
            )

            # ---------------------------------------------
            # 二、执行 GitHub Repository Search
            # ---------------------------------------------
            search_config = (
                github_config.get(
                    "search"
                )
            )

            if not isinstance(
                search_config,
                dict,
            ):
                search_config = {}

            if bool(
                search_config.get(
                    "enabled",
                    True,
                )
            ):
                lookback_days = max(
                    1,
                    _safe_int(
                        search_config.get(
                            "lookback_days"
                        ),
                        7,
                    ),
                )

                since_date = (
                    current_date
                    - timedelta(
                        days=lookback_days
                    )
                )

                per_query = max(
                    1,
                    min(
                        _safe_int(
                            search_config.get(
                                "per_query"
                            ),
                            8,
                        ),
                        100,
                    ),
                )

                sort = str(
                    search_config.get(
                        "sort"
                    )
                    or "stars"
                )

                order = str(
                    search_config.get(
                        "order"
                    )
                    or "desc"
                )

                search_delay = max(
                    0.0,
                    _safe_float(
                        search_config.get(
                            "delay_seconds"
                        ),
                        0.0,
                    ),
                )

                query_definitions = (
                    search_config.get(
                        "queries"
                    )
                )

                if not isinstance(
                    query_definitions,
                    list,
                ):
                    query_definitions = []

                for index, query_definition in enumerate(
                    query_definitions
                ):
                    if not isinstance(
                        query_definition,
                        dict,
                    ):
                        continue

                    query_name = str(
                        query_definition.get(
                            "name"
                        )
                        or f"query-{index + 1}"
                    ).strip()

                    query_template = str(
                        query_definition.get(
                            "query"
                        )
                        or ""
                    ).strip()

                    if not query_template:
                        continue

                    query = _build_search_query(
                        query_template=query_template,
                        since_date=since_date,
                    )

                    try:
                        response_data = (
                            github_client.search_repositories(
                                query,
                                per_page=per_query,
                                sort=sort,
                                order=order,
                            )
                        )

                        items = response_data.get(
                            "items"
                        )

                        if not isinstance(
                            items,
                            list,
                        ):
                            items = []

                        raw_search_responses.append(
                            {
                                "query_name": query_name,
                                "query": query,
                                "total_count": (
                                    response_data.get(
                                        "total_count"
                                    )
                                ),
                                "incomplete_results": (
                                    response_data.get(
                                        "incomplete_results"
                                    )
                                ),
                                "items": items,
                            }
                        )

                        for rank, item in enumerate(
                            items,
                            start=1,
                        ):
                            if not isinstance(
                                item,
                                dict,
                            ):
                                continue

                            search_results.append(
                                SearchResult(
                                    query_name=query_name,
                                    query=query,
                                    rank=rank,
                                    item=item,
                                )
                            )

                    except Exception as exc:
                        errors.append(
                            "GitHub Search 失败，"
                            f"查询名称：{query_name}，"
                            f"原因：{exc}"
                        )

                    if (
                        search_delay > 0
                        and index
                        < len(query_definitions)
                        - 1
                    ):
                        time.sleep(
                            search_delay
                        )

            _write_json(
                raw_directory
                / "github_search.json",
                raw_search_responses,
            )

            # ---------------------------------------------
            # 三、合并并去重候选仓库
            # ---------------------------------------------
            candidates: dict[
                str,
                dict[str, Any],
            ] = {}

            for trending_repository in (
                trending_repositories
            ):
                full_name = (
                    trending_repository.full_name
                )

                candidate = candidates.setdefault(
                    full_name,
                    {
                        "trending": None,
                        "search_item": None,
                        "sources": [],
                        "search_queries": [],
                    },
                )

                candidate["trending"] = (
                    trending_repository
                )

                candidate["sources"].append(
                    "github_trending"
                )

            for search_result in search_results:
                full_name = str(
                    search_result.item.get(
                        "full_name"
                    )
                    or ""
                ).strip()

                if full_name.count("/") != 1:
                    continue

                candidate = candidates.setdefault(
                    full_name,
                    {
                        "trending": None,
                        "search_item": None,
                        "sources": [],
                        "search_queries": [],
                    },
                )

                if (
                    candidate["search_item"]
                    is None
                ):
                    candidate["search_item"] = (
                        search_result.item
                    )

                candidate["sources"].append(
                    "github_search"
                )

                candidate[
                    "search_queries"
                ].append(
                    search_result.query_name
                )

            deduped_count = len(
                candidates
            )

            ordered_candidates = sorted(
                candidates.items(),
                key=_candidate_sort_key,
                reverse=True,
            )

            # ---------------------------------------------
            # 四、限制仓库详情补全数量
            # ---------------------------------------------
            details_config = (
                github_config.get(
                    "repository_details"
                )
            )

            if not isinstance(
                details_config,
                dict,
            ):
                details_config = {}

            details_enabled = bool(
                details_config.get(
                    "enabled",
                    True,
                )
            )

            max_repositories = max(
                0,
                _safe_int(
                    details_config.get(
                        "max_repositories"
                    ),
                    30,
                ),
            )

            if max_repositories > 0:
                ordered_candidates = (
                    ordered_candidates[
                        :max_repositories
                    ]
                )

            detail_delay = max(
                0.0,
                _safe_float(
                    details_config.get(
                        "delay_seconds"
                    ),
                    0.0,
                ),
            )

            # ---------------------------------------------
            # 五、获取详情并建立统一仓库对象
            # ---------------------------------------------
            for index, (
                full_name,
                candidate,
            ) in enumerate(
                ordered_candidates
            ):
                detail: dict[
                    str,
                    Any,
                ] | None = None

                if details_enabled:
                    try:
                        detail = (
                            github_client.get_repository(
                                full_name
                            )
                        )

                        raw_repository_details.append(
                            detail
                        )

                    except GitHubApiError as exc:
                        errors.append(
                            "仓库详情获取失败，"
                            f"仓库：{full_name}，"
                            f"原因：{exc}"
                        )

                if detail is not None:
                    repository = (
                        _build_repository_from_api(
                            detail=detail,
                            sources=candidate[
                                "sources"
                            ],
                            search_queries=candidate[
                                "search_queries"
                            ],
                            trending=candidate[
                                "trending"
                            ],
                            collected_at=collected_at,
                            snapshot_date=snapshot_date,
                            trending_period=trending_period,
                        )
                    )
                else:
                    repository = (
                        _build_repository_from_fallback(
                            full_name=full_name,
                            trending=candidate[
                                "trending"
                            ],
                            search_item=candidate[
                                "search_item"
                            ],
                            sources=candidate[
                                "sources"
                            ],
                            search_queries=candidate[
                                "search_queries"
                            ],
                            collected_at=collected_at,
                            snapshot_date=snapshot_date,
                            trending_period=trending_period,
                        )
                    )

                normalized_repositories.append(
                    repository
                )

                if (
                    detail_delay > 0
                    and index
                    < len(
                        ordered_candidates
                    )
                    - 1
                ):
                    time.sleep(
                        detail_delay
                    )

            _write_json(
                raw_directory
                / "github_repository_details.json",
                raw_repository_details,
            )

            _write_json(
                raw_directory
                / "github_repositories_normalized.json",
                [
                    repository.to_dict()
                    for repository
                    in normalized_repositories
                ],
            )

            # ---------------------------------------------
            # 六、读取写入当天数据之前的历史状态
            # ---------------------------------------------
            selection_contexts: dict[
                str,
                dict[str, Any],
            ] = {}

            for repository in (
                normalized_repositories
            ):
                selection_contexts[
                    repository.full_name
                ] = store.get_selection_context(
                    full_name=repository.full_name,
                    snapshot_date=snapshot_date,
                )

            # ---------------------------------------------
            # 七、计算每日深度处理名单
            # ---------------------------------------------
            selected_repositories = (
                _select_daily_repositories(
                    repositories=normalized_repositories,
                    contexts=selection_contexts,
                    selection_config=selection_config,
                    current_date=current_date,
                )
            )

            selected_count = len(
                selected_repositories
            )

            _write_json(
                raw_directory
                / "github_repositories_selected.json",
                [
                    _selection_to_json_dict(
                        selection
                    )
                    for selection
                    in selected_repositories
                ],
            )

            # ---------------------------------------------
            # 八、写入仓库、快照和发现来源
            # ---------------------------------------------
            stored_count = (
                store.upsert_repositories(
                    normalized_repositories
                )
            )

            # 仓库基础信息写入后，
            # 再保存筛选结果以满足外键要求。
            store.save_daily_selections(
                run_id=run_id,
                snapshot_date=snapshot_date,
                selected_at=collected_at,
                selections=[
                    _selection_to_storage_dict(
                        selection
                    )
                    for selection
                    in selected_repositories
                ],
            )

            # ---------------------------------------------
            # 九、获取 API 限额
            # ---------------------------------------------
            try:
                rate_limit_data = (
                    github_client.get_rate_limit_status()
                )
            except Exception as exc:
                errors.append(
                    "GitHub API 限额获取失败："
                    f"{exc}"
                )

            _write_json(
                raw_directory
                / "github_rate_limit.json",
                rate_limit_data,
            )

            _write_json(
                raw_directory
                / "collection_summary.json",
                {
                    "run_id": run_id,
                    "snapshot_date": snapshot_date,
                    "collected_at": collected_at,
                    "trending_count": len(
                        trending_repositories
                    ),
                    "search_result_count": len(
                        search_results
                    ),
                    "deduped_count": (
                        deduped_count
                    ),
                    "detail_processed_count": len(
                        normalized_repositories
                    ),
                    "stored_count": (
                        stored_count
                    ),
                    "selected_count": (
                        selected_count
                    ),
                    "errors": errors,
                },
            )

        finished_at = datetime.now(
            local_timezone
        ).isoformat()

        if not errors:
            final_status = "success"
        elif stored_count > 0:
            final_status = (
                "partial_success"
            )
        else:
            final_status = "failed"

        store.finish_run(
            run_id=run_id,
            finished_at=finished_at,
            status=final_status,
            trending_count=len(
                trending_repositories
            ),
            search_result_count=len(
                search_results
            ),
            deduped_count=deduped_count,
            stored_count=stored_count,
            selected_count=selected_count,
            errors=errors,
        )

    return {
        "run_id": run_id,
        "status": final_status,
        "snapshot_date": snapshot_date,
        "collected_at": collected_at,
        "trending_count": len(
            trending_repositories
        ),
        "search_result_count": len(
            search_results
        ),
        "deduped_count": deduped_count,
        "detail_processed_count": len(
            normalized_repositories
        ),
        "stored_count": stored_count,
        "selected_count": selected_count,
        "raw_directory": str(
            raw_directory
        ),
        "database_path": str(
            database_path
        ),
        "repositories": [
            repository.to_dict()
            for repository
            in normalized_repositories
        ],
        "selected_repositories": [
            _selection_to_json_dict(
                selection
            )
            for selection
            in selected_repositories
        ],
        "errors": errors,
        "rate_limit": rate_limit_data,
    }