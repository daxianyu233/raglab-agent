from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


# 当前脚本位于项目的 scripts 目录。
# parents[1] 对应项目根目录。
PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# 保证直接运行脚本时可以导入 raglab 包。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from raglab.intelligence.collector import (
    collect_github_intelligence,
)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "采集 GitHub 热点项目，"
            "生成候选仓库和每日处理名单。"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "config"
            / "github_intelligence.yaml"
        ),
        help=(
            "配置文件路径。默认使用 "
            "config\\github_intelligence.yaml"
        ),
    )

    return parser.parse_args()


def load_output_top_n(
    config_path: Path,
) -> int:
    """
    从配置文件读取终端展示数量。
    """
    try:
        config = yaml.safe_load(
            config_path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(
            config,
            dict,
        ):
            return 15

        output_config = config.get(
            "output"
        )

        if not isinstance(
            output_config,
            dict,
        ):
            return 15

        return max(
            1,
            int(
                output_config.get(
                    "top_n"
                )
                or 15
            ),
        )

    except (
        OSError,
        UnicodeDecodeError,
        yaml.YAMLError,
        TypeError,
        ValueError,
    ):
        return 15


def repository_sort_key(
    repository: dict[str, Any],
) -> tuple[int, int, int]:
    """
    候选仓库展示顺序。

    优先显示：
    1. Trending 仓库；
    2. 周期新增 Star 高的仓库；
    3. 总 Star 高的仓库。
    """
    return (
        int(
            repository.get(
                "trending_rank"
            )
            is not None
        ),
        int(
            repository.get(
                "period_stars"
            )
            or 0
        ),
        int(
            repository.get(
                "stars"
            )
            or 0
        ),
    )


def format_sources(
    repository: dict[str, Any],
) -> str:
    """
    格式化仓库发现来源。
    """
    sources = repository.get(
        "sources"
    )

    if not isinstance(
        sources,
        list,
    ):
        sources = []

    search_queries = repository.get(
        "search_queries"
    )

    if not isinstance(
        search_queries,
        list,
    ):
        search_queries = []

    labels: list[str] = []

    if "github_trending" in sources:
        labels.append(
            "Trending"
        )

    if "github_search" in sources:
        if search_queries:
            labels.append(
                "Search:"
                + ",".join(
                    str(query)
                    for query in search_queries
                )
            )
        else:
            labels.append(
                "Search"
            )

    return (
        " | ".join(labels)
        if labels
        else "未知来源"
    )


def format_repository_metric(
    repository: dict[str, Any],
) -> str:
    """
    格式化仓库主要热度指标。
    """
    period_stars = repository.get(
        "period_stars"
    )

    if period_stars is not None:
        return (
            "周期新增 Star："
            f"{int(period_stars):,}"
        )

    return (
        "总 Star："
        f"{int(repository.get('stars') or 0):,}"
    )


def format_rate_limit_lines(
    rate_limit_data: dict[str, Any],
) -> list[str]:
    """
    格式化 GitHub API 限额信息。
    """
    resources = rate_limit_data.get(
        "resources"
    )

    if not isinstance(
        resources,
        dict,
    ):
        return [
            "没有获得有效的 API 限额信息。"
        ]

    lines: list[str] = []

    for resource_name in (
        "core",
        "search",
    ):
        resource = resources.get(
            resource_name
        )

        if not isinstance(
            resource,
            dict,
        ):
            continue

        lines.append(
            (
                f"{resource_name}: "
                f"remaining="
                f"{resource.get('remaining')}/"
                f"{resource.get('limit')}, "
                f"used={resource.get('used')}, "
                f"reset_epoch="
                f"{resource.get('reset')}"
            )
        )

    return (
        lines
        if lines
        else [
            "没有获得 core 或 search 限额信息。"
        ]
    )


def normalize_description(
    description: Any,
    maximum_length: int = 160,
) -> str:
    """
    将仓库描述压缩成一行。
    """
    if not description:
        return ""

    text = " ".join(
        str(description).split()
    )

    if len(text) <= maximum_length:
        return text

    return (
        text[
            : maximum_length - 3
        ]
        + "..."
    )


def print_candidate_repositories(
    repositories: list[dict[str, Any]],
    top_n: int,
) -> None:
    """
    显示候选仓库中的前若干个。
    """
    sorted_repositories = sorted(
        repositories,
        key=repository_sort_key,
        reverse=True,
    )

    display_count = min(
        top_n,
        len(sorted_repositories),
    )

    print()
    print(
        f"候选仓库前 {display_count} 个："
    )

    if not sorted_repositories:
        print(
            "  本次没有获得候选仓库。"
        )
        return

    for index, repository in enumerate(
        sorted_repositories[:top_n],
        start=1,
    ):
        print()
        print(
            f"{index:>2}. "
            f"{repository.get('full_name')}"
        )

        print(
            "    热度："
            + format_repository_metric(
                repository
            )
        )

        print(
            "    来源："
            + format_sources(
                repository
            )
        )

        description = normalize_description(
            repository.get(
                "description"
            )
        )

        if description:
            print(
                f"    描述：{description}"
            )


def print_selected_repositories(
    selected_items: list[dict[str, Any]],
) -> None:
    """
    显示当天真正进入后续深度分析名单的仓库。
    """
    print()
    print(
        "=" * 78
    )

    print(
        "今日进入后续深度分析的仓库"
    )

    print(
        "=" * 78
    )

    if not selected_items:
        print(
            "本次没有仓库进入深度分析名单。"
        )
        return

    group_counter: Counter[str] = Counter()

    for item in selected_items:
        selection = item.get(
            "selection"
        )

        if not isinstance(
            selection,
            dict,
        ):
            selection = {}

        group = str(
            selection.get(
                "group"
            )
            or "unknown"
        )

        group_counter[group] += 1

    print(
        "分组统计："
    )

    group_labels = {
        "trending": "Trending 热点",
        "search_new": "搜索发现的新项目",
        "exploration": "探索项目",
        "revisit": "需要重新分析的旧项目",
        "unknown": "其他",
    }

    for group, count in group_counter.items():
        print(
            "  "
            + group_labels.get(
                group,
                group,
            )
            + f"：{count}"
        )

    for index, item in enumerate(
        selected_items,
        start=1,
    ):
        selection = item.get(
            "selection"
        )

        repository = item.get(
            "repository"
        )

        if not isinstance(
            selection,
            dict,
        ):
            selection = {}

        if not isinstance(
            repository,
            dict,
        ):
            repository = {}

        group = str(
            selection.get(
                "group"
            )
            or "unknown"
        )

        group_text = group_labels.get(
            group,
            group,
        )

        print()
        print(
            f"{index:>2}. "
            f"{repository.get('full_name')}"
        )

        print(
            f"    入选分组：{group_text}"
        )

        print(
            "    筛选得分："
            f"{selection.get('score')}"
        )

        print(
            "    当前 Star："
            f"{int(repository.get('stars') or 0):,}"
        )

        period_stars = repository.get(
            "period_stars"
        )

        if period_stars is not None:
            print(
                "    周期新增 Star："
                f"{int(period_stars):,}"
            )

        reasons = selection.get(
            "reasons"
        )

        if isinstance(
            reasons,
            list,
        ) and reasons:
            print(
                "    入选原因："
            )

            for reason in reasons:
                print(
                    f"      - {reason}"
                )

        source_text = format_sources(
            repository
        )

        print(
            f"    发现来源：{source_text}"
        )

        description = normalize_description(
            repository.get(
                "description"
            )
        )

        if description:
            print(
                f"    描述：{description}"
            )


def print_collection_summary(
    result: dict[str, Any],
    top_n: int,
) -> None:
    """
    输出本次采集的整体摘要。
    """
    print(
        "=" * 78
    )

    print(
        "GitHub 热点情报采集完成"
    )

    print(
        "=" * 78
    )

    print(
        f"运行编号：{result.get('run_id')}"
    )

    print(
        f"运行状态：{result.get('status')}"
    )

    print(
        f"采集日期：{result.get('snapshot_date')}"
    )

    print(
        f"采集时间：{result.get('collected_at')}"
    )

    print()
    print(
        "发现阶段："
    )

    print(
        "  Trending 仓库数："
        f"{result.get('trending_count', 0)}"
    )

    print(
        "  Search 原始结果数："
        f"{result.get('search_result_count', 0)}"
    )

    print(
        "  去重后的候选仓库数："
        f"{result.get('deduped_count', 0)}"
    )

    print()
    print(
        "成本控制："
    )

    print(
        "  完成详情补全的仓库数："
        f"{result.get('detail_processed_count', 0)}"
    )

    print(
        "  写入 SQLite 的仓库数："
        f"{result.get('stored_count', 0)}"
    )

    print(
        "  进入后续深度分析的仓库数："
        f"{result.get('selected_count', 0)}"
    )

    print()
    print(
        "保存位置："
    )

    print(
        "  原始数据目录："
        f"{result.get('raw_directory')}"
    )

    print(
        "  SQLite 数据库："
        f"{result.get('database_path')}"
    )

    print()
    print(
        "GitHub API 剩余额度："
    )

    rate_limit_data = result.get(
        "rate_limit"
    )

    if not isinstance(
        rate_limit_data,
        dict,
    ):
        rate_limit_data = {}

    for line in format_rate_limit_lines(
        rate_limit_data
    ):
        print(
            f"  {line}"
        )

    repositories = result.get(
        "repositories"
    )

    if not isinstance(
        repositories,
        list,
    ):
        repositories = []

    print_candidate_repositories(
        repositories=repositories,
        top_n=top_n,
    )

    selected_repositories = result.get(
        "selected_repositories"
    )

    if not isinstance(
        selected_repositories,
        list,
    ):
        selected_repositories = []

    print_selected_repositories(
        selected_repositories
    )

    errors = result.get(
        "errors"
    )

    if not isinstance(
        errors,
        list,
    ):
        errors = []

    if errors:
        print()
        print(
            "=" * 78
        )

        print(
            "本次采集中的非致命错误"
        )

        print(
            "=" * 78
        )

        for error in errors:
            print(
                f"  - {error}"
            )


def main() -> int:
    """
    命令行入口。
    """
    arguments = parse_arguments()

    config_path = (
        arguments.config.resolve()
    )

    if not config_path.exists():
        print(
            "错误：配置文件不存在："
            f"{config_path}",
            file=sys.stderr,
        )

        return 2

    github_token = os.getenv(
        "GITHUB_TOKEN",
        "",
    ).strip()

    if not github_token:
        print(
            "错误：当前终端没有设置 "
            "GITHUB_TOKEN。",
            file=sys.stderr,
        )

        print(
            "PowerShell 设置方法：",
            file=sys.stderr,
        )

        print(
            '$env:GITHUB_TOKEN="你的GitHubToken"',
            file=sys.stderr,
        )

        return 2

    top_n = load_output_top_n(
        config_path
    )

    try:
        result = collect_github_intelligence(
            project_root=PROJECT_ROOT,
            config_path=config_path,
        )

    except KeyboardInterrupt:
        print(
            "\n用户中止了采集任务。",
            file=sys.stderr,
        )

        return 130

    except Exception as exc:
        print(
            "GitHub 热点情报采集失败："
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    print_collection_summary(
        result=result,
        top_n=top_n,
    )

    if result.get(
        "status"
    ) == "failed":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )