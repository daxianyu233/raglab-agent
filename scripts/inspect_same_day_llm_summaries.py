from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# 默认配置
# ============================================================

DEFAULT_TIMEZONE = "Asia/Shanghai"

DEEP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "intelligence"
    / "deep"
)

LATEST_COLLECTION_NAME = (
    "latest_collection.json"
)

ANALYSIS_MATERIAL_NAME = (
    "github_repository_analysis_material.json"
)

LLM_SUMMARIES_NAME = (
    "repository_llm_summaries.json"
)


# ============================================================
# 命令行参数
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "检查指定日期各个 GitHub 深度采集 collection 中，"
            "项目级 LLM 摘要文件是否存在，以及其中有哪些成功摘要。"
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "检查日期，格式为 YYYY-MM-DD；"
            "不填写时使用北京时间当天。"
        ),
    )

    return parser.parse_args()


# ============================================================
# 基础工具
# ============================================================

def resolve_date(
    value: str | None,
) -> str:
    """
    解析检查日期。
    """
    if value is None:
        return datetime.now(
            ZoneInfo(
                DEFAULT_TIMEZONE
            )
        ).date().isoformat()

    try:
        return date.fromisoformat(
            value
        ).isoformat()

    except ValueError as exc:
        raise ValueError(
            "日期格式错误，应为 YYYY-MM-DD："
            f"{value}"
        ) from exc


def read_json(
    path: Path,
) -> Any:
    """
    读取 UTF-8 JSON 文件。
    """
    text = path.read_text(
        encoding="utf-8",
    )

    return json.loads(
        text
    )


def compact_text(
    value: Any,
) -> str:
    """
    将值转换为去除多余空白的字符串。
    """
    if value is None:
        return ""

    return " ".join(
        str(value).split()
    ).strip()


def resolve_pointer_path(
    raw_value: Any,
    pointer_path: Path,
) -> Path | None:
    """
    解析 latest_collection.json 中记录的文件路径。
    """
    if not raw_value:
        return None

    candidate = Path(
        str(raw_value)
    )

    if candidate.is_absolute():
        return candidate.resolve()

    relative_to_pointer = (
        pointer_path.parent
        / candidate
    ).resolve()

    if relative_to_pointer.exists():
        return relative_to_pointer

    return (
        PROJECT_ROOT
        / candidate
    ).resolve()


# ============================================================
# 摘要文件解析
# ============================================================

def extract_summary_records(
    payload: Any,
) -> list[dict[str, Any]]:
    """
    兼容不同的摘要 JSON 根结构。

    正常结构应当是列表，但这里也兼容：
    {
        "repositories": [...]
    }
    {
        "summaries": [...]
    }
    {
        "items": [...]
    }
    """
    if isinstance(
        payload,
        list,
    ):
        return [
            item
            for item in payload
            if isinstance(
                item,
                dict,
            )
        ]

    if isinstance(
        payload,
        dict,
    ):
        for key in (
            "repositories",
            "summaries",
            "items",
            "records",
            "results",
        ):
            value = payload.get(
                key
            )

            if isinstance(
                value,
                list,
            ):
                return [
                    item
                    for item in value
                    if isinstance(
                        item,
                        dict,
                    )
                ]

    return []


def inspect_summary_file(
    summary_path: Path,
) -> dict[str, Any]:
    """
    检查单个项目摘要文件。
    """
    result: dict[str, Any] = {
        "path": str(
            summary_path
        ),
        "exists": (
            summary_path.is_file()
        ),
        "root_type": None,
        "record_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "missing_status_count": 0,
        "successful_repositories": [],
        "failed_repositories": [],
        "other_repositories": [],
        "error": None,
    }

    if not summary_path.is_file():
        return result

    try:
        payload = read_json(
            summary_path
        )

    except Exception as exc:
        result["error"] = str(
            exc
        )

        return result

    result["root_type"] = type(
        payload
    ).__name__

    records = extract_summary_records(
        payload
    )

    result["record_count"] = len(
        records
    )

    successful_repositories: list[
        str
    ] = []

    failed_repositories: list[
        str
    ] = []

    other_repositories: list[
        str
    ] = []

    for record in records:
        full_name = compact_text(
            record.get(
                "full_name"
            )
        )

        if not full_name:
            full_name = (
                "<缺少 full_name>"
            )

        status = compact_text(
            record.get(
                "status"
            )
        ).lower()

        if status == "success":
            successful_repositories.append(
                full_name
            )

        elif status == "failed":
            failed_repositories.append(
                full_name
            )

        else:
            other_repositories.append(
                (
                    f"{full_name} "
                    f"(status={status or '缺失'})"
                )
            )

    successful_repositories.sort(
        key=str.casefold
    )

    failed_repositories.sort(
        key=str.casefold
    )

    other_repositories.sort(
        key=str.casefold
    )

    result["success_count"] = len(
        successful_repositories
    )

    result["failed_count"] = len(
        failed_repositories
    )

    result["missing_status_count"] = len(
        other_repositories
    )

    result[
        "successful_repositories"
    ] = successful_repositories

    result[
        "failed_repositories"
    ] = failed_repositories

    result[
        "other_repositories"
    ] = other_repositories

    return result


# ============================================================
# 主程序
# ============================================================

def main() -> int:
    """
    检查指定日期的所有深度采集 collection。
    """
    arguments = parse_arguments()

    try:
        snapshot_date = resolve_date(
            arguments.date
        )

    except ValueError as exc:
        print(
            str(exc),
            file=sys.stderr,
        )

        return 2

    date_directory = (
        DEEP_ROOT
        / snapshot_date
    )

    print(
        "=" * 80
    )

    print(
        "同日 GitHub 项目摘要检查"
    )

    print(
        "=" * 80
    )

    print(
        f"日期：{snapshot_date}"
    )

    print(
        f"目录：{date_directory}"
    )

    if not date_directory.is_dir():
        print(
            "错误：当天深度采集目录不存在。",
            file=sys.stderr,
        )

        return 1

    latest_pointer_path = (
        date_directory
        / LATEST_COLLECTION_NAME
    )

    latest_collection_directory: (
        Path | None
    ) = None

    latest_analysis_material_path: (
        Path | None
    ) = None

    latest_summary_path: (
        Path | None
    ) = None

    if latest_pointer_path.is_file():
        try:
            latest_pointer = read_json(
                latest_pointer_path
            )

            if isinstance(
                latest_pointer,
                dict,
            ):
                collection_value = (
                    latest_pointer.get(
                        "collection_directory"
                    )
                )

                analysis_value = (
                    latest_pointer.get(
                        "analysis_material_path"
                    )
                )

                summary_value = (
                    latest_pointer.get(
                        "repository_llm_summaries_path"
                    )
                )

                latest_collection_directory = (
                    resolve_pointer_path(
                        collection_value,
                        latest_pointer_path,
                    )
                )

                latest_analysis_material_path = (
                    resolve_pointer_path(
                        analysis_value,
                        latest_pointer_path,
                    )
                )

                latest_summary_path = (
                    resolve_pointer_path(
                        summary_value,
                        latest_pointer_path,
                    )
                )

        except Exception as exc:
            print(
                "警告：读取 latest_collection.json 失败："
                f"{exc}"
            )

    print()

    print(
        "latest_collection.json 指向："
    )

    print(
        "  collection_directory："
        f"{latest_collection_directory}"
    )

    print(
        "  analysis_material_path："
        f"{latest_analysis_material_path}"
    )

    print(
        "  repository_llm_summaries_path："
        f"{latest_summary_path}"
    )

    collection_directories = sorted(
        (
            child
            for child in date_directory.iterdir()
            if child.is_dir()
        ),
        key=lambda item: item.name,
    )

    print()

    print(
        "Collection 数量："
        f"{len(collection_directories)}"
    )

    print()

    all_successful_locations: dict[
        str,
        list[str],
    ] = {}

    for index, collection_directory in enumerate(
        collection_directories,
        start=1,
    ):
        analysis_material_path = (
            collection_directory
            / ANALYSIS_MATERIAL_NAME
        )

        summary_path = (
            collection_directory
            / LLM_SUMMARIES_NAME
        )

        inspection = inspect_summary_file(
            summary_path
        )

        is_latest_collection = (
            latest_collection_directory
            is not None
            and collection_directory.resolve()
            == latest_collection_directory.resolve()
        )

        print(
            "-" * 80
        )

        print(
            f"[{index}/"
            f"{len(collection_directories)}] "
            f"{collection_directory.name}"
        )

        print(
            "  是否为 latest collection："
            f"{is_latest_collection}"
        )

        print(
            "  粗缩减材料："
            f"{'存在' if analysis_material_path.is_file() else '不存在'}"
        )

        print(
            "  LLM 摘要："
            f"{'存在' if inspection['exists'] else '不存在'}"
        )

        if inspection["error"]:
            print(
                "  摘要读取错误："
                f"{inspection['error']}"
            )

            continue

        if not inspection["exists"]:
            continue

        print(
            "  JSON 根类型："
            f"{inspection['root_type']}"
        )

        print(
            "  摘要记录数："
            f"{inspection['record_count']}"
        )

        print(
            "  success："
            f"{inspection['success_count']}"
        )

        print(
            "  failed："
            f"{inspection['failed_count']}"
        )

        print(
            "  其他或缺失状态："
            f"{inspection['missing_status_count']}"
        )

        successful_repositories = (
            inspection[
                "successful_repositories"
            ]
        )

        if successful_repositories:
            print(
                "  成功项目："
            )

            for full_name in (
                successful_repositories
            ):
                print(
                    f"    - {full_name}"
                )

                all_successful_locations.setdefault(
                    full_name.casefold(),
                    [],
                ).append(
                    str(summary_path)
                )

        failed_repositories = (
            inspection[
                "failed_repositories"
            ]
        )

        if failed_repositories:
            print(
                "  失败项目："
            )

            for full_name in (
                failed_repositories
            ):
                print(
                    f"    - {full_name}"
                )

        other_repositories = (
            inspection[
                "other_repositories"
            ]
        )

        if other_repositories:
            print(
                "  其他状态项目："
            )

            for full_name in (
                other_repositories
            ):
                print(
                    f"    - {full_name}"
                )

    print()

    print(
        "=" * 80
    )

    print(
        "汇总"
    )

    print(
        "=" * 80
    )

    print(
        "当天至少存在一条成功摘要的项目数："
        f"{len(all_successful_locations)}"
    )

    if all_successful_locations:
        for (
            repository_key,
            locations,
        ) in sorted(
            all_successful_locations.items(),
            key=lambda item: item[0],
        ):
            print(
                f"- {repository_key}"
            )

            for location in locations:
                print(
                    f"    {location}"
                )

    print()

    print(
        "检查完成。"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )