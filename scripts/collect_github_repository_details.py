from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


# 当前脚本位于项目根目录下的 scripts 文件夹。
# parents[1] 即为 rag-lab 项目根目录。
PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# 直接运行 scripts 中的脚本时，
# Python 默认不一定能够找到项目根目录中的 raglab 包。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from raglab.intelligence.deep_collector import (
    collect_selected_repository_details,
)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "对每日入选的 GitHub 仓库执行深度信息采集，"
            "获取 README、Release 和 Issue。"
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

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "需要处理的采集日期，格式为 YYYY-MM-DD。"
            "不填写时使用配置时区中的当前日期。"
        ),
    )

    return parser.parse_args()


def format_number(
    value: Any,
) -> str:
    """
    将数字格式化为带千位分隔符的字符串。
    """
    try:
        return f"{int(value):,}"
    except (
        TypeError,
        ValueError,
    ):
        return "0"


def format_status(
    status: Any,
) -> str:
    """
    将内部运行状态转换为中文。
    """
    status_text = str(
        status
        or "unknown"
    )

    status_mapping = {
        "success": "成功",
        "partial_success": "部分成功",
        "failed": "失败",
        "disabled": "已禁用",
        "unknown": "未知",
    }

    return status_mapping.get(
        status_text,
        status_text,
    )


def format_selected_source(
    source: Any,
) -> str:
    """
    格式化入选名单的读取来源。
    """
    source_text = str(
        source
        or "none"
    )

    source_mapping = {
        "json": (
            "第一阶段生成的 "
            "github_repositories_selected.json"
        ),
        "sqlite": (
            "SQLite 中当天最近一次非空入选名单"
        ),
        "none": "未找到入选名单",
    }

    return source_mapping.get(
        source_text,
        source_text,
    )


def shorten_text(
    text: Any,
    maximum_length: int = 140,
) -> str:
    """
    将文本压缩为一行，并限制展示长度。
    """
    if text is None:
        return ""

    normalized_text = " ".join(
        str(text).split()
    )

    if len(
        normalized_text
    ) <= maximum_length:
        return normalized_text

    return (
        normalized_text[
            : maximum_length - 3
        ]
        + "..."
    )


def get_api_limit_lines(
    rate_limit_data: dict[str, Any],
) -> list[str]:
    """
    从 GitHub /rate_limit API 结果中提取主要限额。
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
        "graphql",
    ):
        resource_data = resources.get(
            resource_name
        )

        if not isinstance(
            resource_data,
            dict,
        ):
            continue

        remaining = resource_data.get(
            "remaining"
        )

        limit = resource_data.get(
            "limit"
        )

        used = resource_data.get(
            "used"
        )

        reset = resource_data.get(
            "reset"
        )

        lines.append(
            (
                f"{resource_name}: "
                f"remaining={remaining}/{limit}, "
                f"used={used}, "
                f"reset_epoch={reset}"
            )
        )

    if not lines:
        return [
            "没有获得 core、search 或 graphql 限额信息。"
        ]

    return lines


def print_repository_result(
    repository_result: dict[str, Any],
    index: int,
) -> None:
    """
    输出单个仓库的深度采集结果。
    """
    full_name = str(
        repository_result.get(
            "full_name"
        )
        or "unknown/unknown"
    )

    readme_available = bool(
        repository_result.get(
            "readme_available"
        )
    )

    release_count = int(
        repository_result.get(
            "release_count"
        )
        or 0
    )

    issue_count = int(
        repository_result.get(
            "issue_count"
        )
        or 0
    )

    error_count = int(
        repository_result.get(
            "error_count"
        )
        or 0
    )

    raw_file = str(
        repository_result.get(
            "raw_file"
        )
        or ""
    )

    print()
    print(
        f"{index:>2}. {full_name}"
    )

    print(
        "    README："
        + (
            "已获取"
            if readme_available
            else "未获取或不存在"
        )
    )

    print(
        f"    Release 数量：{release_count}"
    )

    print(
        f"    Issue 数量：{issue_count}"
    )

    print(
        f"    错误数量：{error_count}"
    )

    if raw_file:
        print(
            f"    原始文件：{raw_file}"
        )


def print_collection_summary(
    result: dict[str, Any],
) -> None:
    """
    输出深度采集任务摘要。
    """
    print(
        "=" * 78
    )

    print(
        "GitHub 入选仓库深度信息采集完成"
    )

    print(
        "=" * 78
    )

    print(
        "运行编号："
        f"{result.get('collection_id')}"
    )

    print(
        "运行状态："
        f"{format_status(result.get('status'))}"
    )

    print(
        "采集日期："
        f"{result.get('snapshot_date')}"
    )

    print(
        "开始时间："
        f"{result.get('started_at')}"
    )

    print(
        "结束时间："
        f"{result.get('finished_at')}"
    )

    print()
    print(
        "入选名单来源："
    )

    print(
        "  "
        + format_selected_source(
            result.get(
                "selected_source"
            )
        )
    )

    print()
    print(
        "仓库处理统计："
    )

    print(
        "  入选仓库数："
        f"{format_number(result.get('selected_count'))}"
    )

    print(
        "  实际处理仓库数："
        f"{format_number(result.get('processed_count'))}"
    )

    print(
        "  存在采集错误的仓库数："
        f"{format_number(result.get('repository_error_count'))}"
    )

    print()
    print(
        "README 统计："
    )

    print(
        "  成功获取："
        f"{format_number(result.get('readme_success_count'))}"
    )

    print(
        "  README 不存在："
        f"{format_number(result.get('readme_missing_count'))}"
    )

    print()
    print(
        "Release 和 Issue 统计："
    )

    print(
        "  Release 请求成功仓库数："
        f"{format_number(result.get('release_request_success_count'))}"
    )

    print(
        "  实际获取 Release 总数："
        f"{format_number(result.get('total_release_count'))}"
    )

    print(
        "  Issue 请求成功仓库数："
        f"{format_number(result.get('issue_request_success_count'))}"
    )

    print(
        "  实际获取 Issue 总数："
        f"{format_number(result.get('total_issue_count'))}"
    )

    print()
    print(
        "分析材料规模："
    )

    total_characters = int(
        result.get(
            "total_analysis_characters"
        )
        or 0
    )

    print(
        "  裁剪后文本字符总数："
        f"{total_characters:,}"
    )

    processed_count = int(
        result.get(
            "processed_count"
        )
        or 0
    )

    if processed_count > 0:
        average_characters = (
            total_characters
            / processed_count
        )

        print(
            "  平均每个仓库字符数："
            f"{average_characters:,.0f}"
        )

    print()
    print(
        "数据保存位置："
    )

    print(
        "  本次运行目录："
        f"{result.get('collection_directory')}"
    )

    print(
        "  原始文件索引："
        f"{result.get('raw_index_path')}"
    )

    print(
        "  裁剪后的分析材料："
        f"{result.get('analysis_material_path')}"
    )

    print(
        "  运行摘要："
        f"{result.get('summary_path')}"
    )

    print(
        "  当天最新运行指针："
        f"{result.get('latest_pointer_path')}"
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

    for line in get_api_limit_lines(
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

    print()
    print(
        "=" * 78
    )

    print(
        "各仓库深度采集结果"
    )

    print(
        "=" * 78
    )

    if not repositories:
        print(
            "本次没有处理任何仓库。"
        )
    else:
        for index, repository_result in enumerate(
            repositories,
            start=1,
        ):
            if not isinstance(
                repository_result,
                dict,
            ):
                continue

            print_repository_result(
                repository_result,
                index,
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
            "本次运行中的全局错误"
        )

        print(
            "=" * 78
        )

        for error in errors:
            print(
                "  - "
                + shorten_text(
                    error,
                    maximum_length=500,
                )
            )


def main() -> int:
    """
    深度采集命令行入口。
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
            "错误：当前 PowerShell "
            "没有设置 GITHUB_TOKEN。",
            file=sys.stderr,
        )

        print(
            "请在当前 PowerShell 中执行：",
            file=sys.stderr,
        )

        print(
            '$env:GITHUB_TOKEN="你的GitHubToken"',
            file=sys.stderr,
        )

        return 2

    try:
        result = (
            collect_selected_repository_details(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                snapshot_date=arguments.date,
            )
        )

    except KeyboardInterrupt:
        print(
            "\n用户中止了深度采集任务。",
            file=sys.stderr,
        )

        return 130

    except Exception as exc:
        print(
            "GitHub 深度信息采集失败："
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    print_collection_summary(
        result
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