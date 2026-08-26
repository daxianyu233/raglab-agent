from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from .models import TrendingRepository


CandidateItem = tuple[
    str,
    dict[str, Any],
]


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


def _parse_datetime_timestamp(
    value: Any,
) -> float:
    """
    将 GitHub ISO 时间转换为时间戳。

    无效或缺失时间返回 0。
    """
    if not value:
        return 0.0

    text = str(value).strip()

    if not text:
        return 0.0

    try:
        parsed_datetime = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )

        return parsed_datetime.timestamp()

    except ValueError:
        return 0.0


def _get_trending_repository(
    candidate: dict[str, Any],
) -> TrendingRepository | None:
    """
    获取候选仓库中的 Trending 对象。
    """
    trending = candidate.get(
        "trending"
    )

    if isinstance(
        trending,
        TrendingRepository,
    ):
        return trending

    return None


def _get_search_item(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    获取候选仓库中的 Search API 结果。
    """
    search_item = candidate.get(
        "search_item"
    )

    if isinstance(
        search_item,
        dict,
    ):
        return search_item

    return {}


def _get_search_queries(
    candidate: dict[str, Any],
) -> list[str]:
    """
    获取候选仓库命中的搜索主题，并进行去重。
    """
    query_values = candidate.get(
        "search_queries"
    )

    if not isinstance(
        query_values,
        list,
    ):
        return []

    normalized_queries = {
        str(query).strip()
        for query in query_values
        if str(query).strip()
    }

    return sorted(
        normalized_queries
    )


def _candidate_total_stars(
    candidate: dict[str, Any],
) -> int:
    """
    获取候选仓库的总 Star 数。

    Search API 数据优先；
    没有 Search 数据时使用 Trending 页面数据。
    """
    search_item = _get_search_item(
        candidate
    )

    if (
        search_item.get(
            "stargazers_count"
        )
        is not None
    ):
        return _safe_int(
            search_item.get(
                "stargazers_count"
            )
        )

    trending = _get_trending_repository(
        candidate
    )

    if trending is not None:
        return _safe_int(
            trending.total_stars
        )

    return 0


def _candidate_period_stars(
    candidate: dict[str, Any],
) -> int:
    """
    获取 Trending 周期新增 Star 数。
    """
    trending = _get_trending_repository(
        candidate
    )

    if trending is None:
        return 0

    return max(
        0,
        _safe_int(
            trending.period_stars
        ),
    )


def _candidate_trending_rank(
    candidate: dict[str, Any],
) -> int:
    """
    获取 Trending 排名。

    非 Trending 项目返回一个较大的默认排名。
    """
    trending = _get_trending_repository(
        candidate
    )

    if trending is None:
        return 999999

    rank = _safe_int(
        trending.rank,
        999999,
    )

    if rank <= 0:
        return 999999

    return rank


def _candidate_created_timestamp(
    candidate: dict[str, Any],
) -> float:
    """
    获取仓库创建时间戳。
    """
    search_item = _get_search_item(
        candidate
    )

    return _parse_datetime_timestamp(
        search_item.get(
            "created_at"
        )
    )


def _candidate_pushed_timestamp(
    candidate: dict[str, Any],
) -> float:
    """
    获取仓库最近推送时间戳。
    """
    search_item = _get_search_item(
        candidate
    )

    return _parse_datetime_timestamp(
        search_item.get(
            "pushed_at"
        )
    )


def _candidate_updated_timestamp(
    candidate: dict[str, Any],
) -> float:
    """
    获取仓库最近更新时间戳。
    """
    search_item = _get_search_item(
        candidate
    )

    return _parse_datetime_timestamp(
        search_item.get(
            "updated_at"
        )
    )


def _candidate_is_trending(
    candidate: dict[str, Any],
) -> bool:
    """
    判断候选项目是否来自 GitHub Trending。
    """
    return (
        _get_trending_repository(
            candidate
        )
        is not None
    )


def _calculate_trending_score(
    candidate: dict[str, Any],
) -> float:
    """
    计算 Trending 项目的详情补全优先级。

    主要考虑：

    1. 周期新增 Star；
    2. Trending 排名；
    3. 是否同时命中搜索主题；
    4. 总 Star 数。
    """
    period_stars = (
        _candidate_period_stars(
            candidate
        )
    )

    trending_rank = (
        _candidate_trending_rank(
            candidate
        )
    )

    total_stars = (
        _candidate_total_stars(
            candidate
        )
    )

    query_count = len(
        _get_search_queries(
            candidate
        )
    )

    score = 0.0

    score += 200.0

    score += min(
        math.log1p(
            period_stars
        )
        * 25.0,
        220.0,
    )

    score += max(
        0.0,
        50.0
        - float(
            trending_rank
        )
        * 2.0,
    )

    score += (
        query_count
        * 25.0
    )

    score += min(
        math.log1p(
            total_stars
        )
        * 3.0,
        40.0,
    )

    return round(
        score,
        4,
    )


def _calculate_search_score(
    candidate: dict[str, Any],
) -> float:
    """
    计算关键词搜索项目的详情补全优先级。

    主要考虑：

    1. 同时命中的搜索主题数量；
    2. 最近是否仍有代码推送；
    3. 总 Star 数；
    4. 仓库更新时间。
    """
    query_count = len(
        _get_search_queries(
            candidate
        )
    )

    total_stars = (
        _candidate_total_stars(
            candidate
        )
    )

    pushed_timestamp = (
        _candidate_pushed_timestamp(
            candidate
        )
    )

    updated_timestamp = (
        _candidate_updated_timestamp(
            candidate
        )
    )

    score = 0.0

    score += (
        query_count
        * 120.0
    )

    score += min(
        math.log1p(
            total_stars
        )
        * 8.0,
        100.0,
    )

    if pushed_timestamp > 0:
        score += 30.0

    if updated_timestamp > 0:
        score += 10.0

    return round(
        score,
        4,
    )


def _calculate_exploration_score(
    candidate: dict[str, Any],
) -> float:
    """
    计算探索项目优先级。

    探索项目不应只由总 Star 决定，
    因此更强调：

    1. 创建时间较新；
    2. 最近仍有推送；
    3. 至少命中一个搜索主题；
    4. 已经获得一定初始关注。
    """
    query_count = len(
        _get_search_queries(
            candidate
        )
    )

    total_stars = (
        _candidate_total_stars(
            candidate
        )
    )

    created_timestamp = (
        _candidate_created_timestamp(
            candidate
        )
    )

    pushed_timestamp = (
        _candidate_pushed_timestamp(
            candidate
        )
    )

    score = 0.0

    score += (
        query_count
        * 40.0
    )

    # 创建时间越新，时间戳越大。
    # 为避免直接加入超大数值，这里进行缩放。
    if created_timestamp > 0:
        score += (
            created_timestamp
            / 100_000_000
        )

    if pushed_timestamp > 0:
        score += (
            pushed_timestamp
            / 200_000_000
        )

    # 探索项目需要一定关注度，
    # 但不让超高 Star 老项目完全支配排序。
    score += min(
        math.log1p(
            total_stars
        )
        * 5.0,
        45.0,
    )

    return round(
        score,
        4,
    )


def _calculate_backfill_score(
    candidate: dict[str, Any],
) -> float:
    """
    计算空缺名额补足时的综合优先级。
    """
    if _candidate_is_trending(
        candidate
    ):
        return _calculate_trending_score(
            candidate
        )

    search_score = (
        _calculate_search_score(
            candidate
        )
    )

    exploration_score = (
        _calculate_exploration_score(
            candidate
        )
    )

    return round(
        max(
            search_score,
            exploration_score,
        ),
        4,
    )


def _clone_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    创建候选仓库的浅拷贝。

    同时复制列表字段，避免修改原始候选对象。
    """
    cloned_candidate = dict(
        candidate
    )

    sources = candidate.get(
        "sources"
    )

    search_queries = candidate.get(
        "search_queries"
    )

    cloned_candidate["sources"] = (
        list(sources)
        if isinstance(
            sources,
            list,
        )
        else []
    )

    cloned_candidate[
        "search_queries"
    ] = (
        list(search_queries)
        if isinstance(
            search_queries,
            list,
        )
        else []
    )

    return cloned_candidate


def _build_selected_candidate(
    *,
    full_name: str,
    candidate: dict[str, Any],
    group: str,
    score: float,
    reasons: list[str],
) -> CandidateItem:
    """
    为入选基础详情补全的候选仓库增加筛选信息。
    """
    selected_candidate = (
        _clone_candidate(
            candidate
        )
    )

    selected_candidate[
        "detail_selection"
    ] = {
        "group": group,
        "score": round(
            score,
            4,
        ),
        "reasons": reasons,
    }

    return (
        full_name,
        selected_candidate,
    )


def _normalize_detail_quotas(
    *,
    maximum_repositories: int,
    trending_limit: int,
    search_limit: int,
    exploration_limit: int,
) -> tuple[int, int, int]:
    """
    确保三个分类配额之和不超过总上限。

    当配置总和超过最大数量时，
    按以下顺序保留名额：

    1. Trending；
    2. 搜索高相关项目；
    3. 探索项目。
    """
    remaining = max(
        0,
        maximum_repositories,
    )

    normalized_trending_limit = min(
        max(
            0,
            trending_limit,
        ),
        remaining,
    )

    remaining -= (
        normalized_trending_limit
    )

    normalized_search_limit = min(
        max(
            0,
            search_limit,
        ),
        remaining,
    )

    remaining -= (
        normalized_search_limit
    )

    normalized_exploration_limit = min(
        max(
            0,
            exploration_limit,
        ),
        remaining,
    )

    return (
        normalized_trending_limit,
        normalized_search_limit,
        normalized_exploration_limit,
    )


def select_candidates_for_details(
    *,
    candidates: dict[
        str,
        dict[str, Any],
    ],
    details_config: dict[str, Any],
) -> tuple[
    list[CandidateItem],
    dict[str, Any],
]:
    """
    从全部候选仓库中选择需要补全基础详情的项目。

    选择过程完全不使用 LLM，也不读取 README。

    默认配额：

    1. Trending 项目最多 15 个；
    2. 关键词高相关项目最多 10 个；
    3. 新项目探索最多 5 个；
    4. 分类没有用满的名额由剩余候选自动补足。
    """
    maximum_repositories = max(
        0,
        _safe_int(
            details_config.get(
                "max_repositories"
            ),
            30,
        ),
    )

    if maximum_repositories == 0:
        all_candidates = [
            _build_selected_candidate(
                full_name=full_name,
                candidate=candidate,
                group="unlimited",
                score=(
                    _calculate_backfill_score(
                        candidate
                    )
                ),
                reasons=[
                    "基础详情补全未设置总量上限"
                ],
            )
            for full_name, candidate
            in candidates.items()
        ]

        all_candidates.sort(
            key=lambda item: (
                float(
                    item[1]
                    .get(
                        "detail_selection",
                        {},
                    )
                    .get(
                        "score",
                        0,
                    )
                ),
                item[0],
            ),
            reverse=True,
        )

        return (
            all_candidates,
            {
                "maximum_repositories": 0,
                "candidate_count": len(
                    candidates
                ),
                "selected_count": len(
                    all_candidates
                ),
                "group_counts": {
                    "unlimited": len(
                        all_candidates
                    ),
                },
            },
        )

    trending_limit = max(
        0,
        _safe_int(
            details_config.get(
                "trending_limit"
            ),
            15,
        ),
    )

    search_limit = max(
        0,
        _safe_int(
            details_config.get(
                "search_limit"
            ),
            10,
        ),
    )

    exploration_limit = max(
        0,
        _safe_int(
            details_config.get(
                "exploration_limit"
            ),
            5,
        ),
    )

    (
        trending_limit,
        search_limit,
        exploration_limit,
    ) = _normalize_detail_quotas(
        maximum_repositories=(
            maximum_repositories
        ),
        trending_limit=(
            trending_limit
        ),
        search_limit=search_limit,
        exploration_limit=(
            exploration_limit
        ),
    )

    trending_candidates: list[
        CandidateItem
    ] = []

    search_candidates: list[
        CandidateItem
    ] = []

    for full_name, candidate in candidates.items():
        if _candidate_is_trending(
            candidate
        ):
            trending_candidates.append(
                (
                    full_name,
                    candidate,
                )
            )
        else:
            search_candidates.append(
                (
                    full_name,
                    candidate,
                )
            )

    trending_candidates.sort(
        key=lambda item: (
            _calculate_trending_score(
                item[1]
            ),
            _candidate_period_stars(
                item[1]
            ),
            -_candidate_trending_rank(
                item[1]
            ),
            item[0],
        ),
        reverse=True,
    )

    search_candidates.sort(
        key=lambda item: (
            _calculate_search_score(
                item[1]
            ),
            len(
                _get_search_queries(
                    item[1]
                )
            ),
            _candidate_total_stars(
                item[1]
            ),
            item[0],
        ),
        reverse=True,
    )

    selected: list[
        CandidateItem
    ] = []

    selected_names: set[str] = set()

    group_counts: dict[str, int] = {
        "trending": 0,
        "search_relevant": 0,
        "exploration": 0,
        "backfill": 0,
    }

    def append_candidate(
        *,
        full_name: str,
        candidate: dict[str, Any],
        group: str,
        score: float,
        reasons: list[str],
    ) -> bool:
        """
        将仓库加入详情补全名单。

        同时执行去重和总量限制。
        """
        if full_name in selected_names:
            return False

        if len(selected) >= maximum_repositories:
            return False

        selected.append(
            _build_selected_candidate(
                full_name=full_name,
                candidate=candidate,
                group=group,
                score=score,
                reasons=reasons,
            )
        )

        selected_names.add(
            full_name
        )

        group_counts[group] = (
            group_counts.get(
                group,
                0,
            )
            + 1
        )

        return True

    # -------------------------------------------------
    # 一、选择 Trending 项目
    # -------------------------------------------------
    for (
        full_name,
        candidate,
    ) in trending_candidates:
        if (
            group_counts["trending"]
            >= trending_limit
        ):
            break

        trending = (
            _get_trending_repository(
                candidate
            )
        )

        reasons = [
            "进入 GitHub Trending"
        ]

        if trending is not None:
            reasons.append(
                "Trending 排名："
                f"{trending.rank}"
            )

            reasons.append(
                "周期新增 Star："
                f"{trending.period_stars:,}"
            )

        query_names = (
            _get_search_queries(
                candidate
            )
        )

        if query_names:
            reasons.append(
                "同时命中搜索主题："
                + "、".join(
                    query_names
                )
            )

        append_candidate(
            full_name=full_name,
            candidate=candidate,
            group="trending",
            score=(
                _calculate_trending_score(
                    candidate
                )
            ),
            reasons=reasons,
        )

    # -------------------------------------------------
    # 二、选择关键词高相关项目
    # -------------------------------------------------
    for (
        full_name,
        candidate,
    ) in search_candidates:
        if (
            group_counts[
                "search_relevant"
            ]
            >= search_limit
        ):
            break

        query_names = (
            _get_search_queries(
                candidate
            )
        )

        reasons = [
            "通过关键词搜索发现"
        ]

        if query_names:
            reasons.append(
                "命中搜索主题："
                + "、".join(
                    query_names
                )
            )

        if len(query_names) > 1:
            reasons.append(
                "同时命中多个关注方向"
            )

        reasons.append(
            "当前总 Star："
            f"{_candidate_total_stars(candidate):,}"
        )

        append_candidate(
            full_name=full_name,
            candidate=candidate,
            group="search_relevant",
            score=(
                _calculate_search_score(
                    candidate
                )
            ),
            reasons=reasons,
        )

    # -------------------------------------------------
    # 三、从剩余搜索项目中选择探索项目
    # -------------------------------------------------
    exploration_candidates = [
        (
            full_name,
            candidate,
        )
        for full_name, candidate
        in search_candidates
        if full_name not in selected_names
    ]

    exploration_candidates.sort(
        key=lambda item: (
            _calculate_exploration_score(
                item[1]
            ),
            _candidate_created_timestamp(
                item[1]
            ),
            _candidate_pushed_timestamp(
                item[1]
            ),
            item[0],
        ),
        reverse=True,
    )

    for (
        full_name,
        candidate,
    ) in exploration_candidates:
        if (
            group_counts["exploration"]
            >= exploration_limit
        ):
            break

        query_names = (
            _get_search_queries(
                candidate
            )
        )

        reasons = [
            "作为新方向探索项目保留"
        ]

        if query_names:
            reasons.append(
                "命中搜索主题："
                + "、".join(
                    query_names
                )
            )

        search_item = _get_search_item(
            candidate
        )

        created_at = search_item.get(
            "created_at"
        )

        pushed_at = search_item.get(
            "pushed_at"
        )

        if created_at:
            reasons.append(
                "仓库创建时间："
                f"{created_at}"
            )

        if pushed_at:
            reasons.append(
                "最近推送时间："
                f"{pushed_at}"
            )

        append_candidate(
            full_name=full_name,
            candidate=candidate,
            group="exploration",
            score=(
                _calculate_exploration_score(
                    candidate
                )
            ),
            reasons=reasons,
        )

    # -------------------------------------------------
    # 四、分类配额没有用满时自动补足
    # -------------------------------------------------
    remaining_candidates = [
        (
            full_name,
            candidate,
        )
        for full_name, candidate
        in candidates.items()
        if full_name not in selected_names
    ]

    remaining_candidates.sort(
        key=lambda item: (
            _calculate_backfill_score(
                item[1]
            ),
            int(
                _candidate_is_trending(
                    item[1]
                )
            ),
            _candidate_total_stars(
                item[1]
            ),
            item[0],
        ),
        reverse=True,
    )

    for (
        full_name,
        candidate,
    ) in remaining_candidates:
        if (
            len(selected)
            >= maximum_repositories
        ):
            break

        reasons = [
            "其他分类存在空缺，"
            "按综合优先级补足详情名额"
        ]

        if _candidate_is_trending(
            candidate
        ):
            reasons.append(
                "来源包含 GitHub Trending"
            )

        query_names = (
            _get_search_queries(
                candidate
            )
        )

        if query_names:
            reasons.append(
                "命中搜索主题："
                + "、".join(
                    query_names
                )
            )

        append_candidate(
            full_name=full_name,
            candidate=candidate,
            group="backfill",
            score=(
                _calculate_backfill_score(
                    candidate
                )
            ),
            reasons=reasons,
        )

    selection_summary = {
        "maximum_repositories": (
            maximum_repositories
        ),
        "candidate_count": len(
            candidates
        ),
        "selected_count": len(
            selected
        ),
        "configured_quotas": {
            "trending": trending_limit,
            "search_relevant": (
                search_limit
            ),
            "exploration": (
                exploration_limit
            ),
        },
        "group_counts": (
            group_counts
        ),
        "unselected_count": max(
            0,
            len(candidates)
            - len(selected),
        ),
    }

    return (
        selected,
        selection_summary,
    )