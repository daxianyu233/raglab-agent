from __future__ import annotations

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


from raglab.intelligence.persistence import (
    save_daily_brief_assets,
)


# ============================================================
# 配置区
# ============================================================

TIMEZONE = "Asia/Shanghai"

# None 表示使用北京时间当天。
# 处理历史日期时可修改为：
# SNAPSHOT_DATE = "2026-08-02"
SNAPSHOT_DATE: str | None = None

# 精简日报最多保留的内容数量。
MAX_HOTSPOT_TOPICS = 3
MAX_NOTABLE_PROJECTS = 3
MAX_WATCH_ITEMS = 3

# 各字段最大字符数。
EXECUTIVE_SUMMARY_MAX_CHARACTERS = 180
TOPIC_SUMMARY_MAX_CHARACTERS = 110
TOPIC_REASON_MAX_CHARACTERS = 80
PROJECT_SUMMARY_MAX_CHARACTERS = 90
PROJECT_REASON_MAX_CHARACTERS = 70
WATCH_ITEM_MAX_CHARACTERS = 70

# 每个热点最多展示几个相关项目。
MAX_RELATED_PROJECTS = 2


# ============================================================
# 文件工具
# ============================================================


def read_json(
    path: Path,
) -> Any:
    """
    读取 UTF-8 JSON 文件。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"文件不存在：{path}"
        )

    try:
        text = path.read_text(
            encoding="utf-8",
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            "文件不是有效的 UTF-8 编码："
            f"{path}"
        ) from exc

    try:
        return json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "JSON 文件格式错误："
            f"{path}\n{exc}"
        ) from exc


def write_json(
    path: Path,
    data: Any,
) -> None:
    """
    原子化保存 UTF-8 JSON。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def write_text(
    path: Path,
    text: str,
) -> None:
    """
    原子化保存 UTF-8 文本。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temporary_path.write_text(
        text,
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def resolve_snapshot_date() -> str:
    """
    确定需要处理的日期。
    """
    if SNAPSHOT_DATE is None:
        return datetime.now(
            ZoneInfo(
                TIMEZONE
            )
        ).date().isoformat()

    try:
        return date.fromisoformat(
            SNAPSHOT_DATE
        ).isoformat()

    except ValueError as exc:
        raise ValueError(
            "SNAPSHOT_DATE 格式错误，"
            "必须使用 YYYY-MM-DD："
            f"{SNAPSHOT_DATE}"
        ) from exc


def resolve_path(
    value: Any,
) -> Path:
    """
    将指针文件中的路径转换为绝对路径。
    """
    path = Path(
        str(
            value
        )
    )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


# ============================================================
# 文本工具
# ============================================================


def compact_text(
    value: Any,
) -> str:
    """
    将文本中的连续空白压缩为一个空格。
    """
    if value is None:
        return ""

    return " ".join(
        str(
            value
        ).split()
    ).strip()


def truncate_text(
    value: Any,
    maximum_characters: int,
) -> str:
    """
    将文本限制在指定字符数以内。
    """
    text = compact_text(
        value
    )

    if len(
        text
    ) <= maximum_characters:
        return text

    if maximum_characters <= 3:
        return text[
            :maximum_characters
        ]

    return (
        text[
            : maximum_characters - 3
        ]
        + "..."
    )


def compact_string_list(
    value: Any,
) -> list[str]:
    """
    将任意列表转换为非空字符串列表。
    """
    if not isinstance(
        value,
        list,
    ):
        return []

    results: list[str] = []

    for item in value:
        text = compact_text(
            item
        )

        if text:
            results.append(
                text
            )

    return results


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    将任意值安全转换为整数。
    """
    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# 输入文件定位
# ============================================================


def locate_daily_hotspots(
    snapshot_date: str,
) -> tuple[
    Path,
    Path,
    dict[str, Any],
]:
    """
    根据 latest_collection.json
    定位 daily_hotspots.json。

    返回：

    1. daily_hotspots.json 路径；
    2. latest_collection.json 路径；
    3. latest_collection.json 内容。
    """
    latest_pointer_path = (
        PROJECT_ROOT
        / "data"
        / "intelligence"
        / "deep"
        / snapshot_date
        / "latest_collection.json"
    )

    latest_pointer = read_json(
        latest_pointer_path
    )

    if not isinstance(
        latest_pointer,
        dict,
    ):
        raise ValueError(
            "latest_collection.json "
            "根节点必须是对象。"
        )

    hotspots_value = (
        latest_pointer.get(
            "daily_hotspots_path"
        )
    )

    if hotspots_value:
        hotspots_path = resolve_path(
            hotspots_value
        )

        return (
            hotspots_path,
            latest_pointer_path,
            latest_pointer,
        )

    collection_directory_value = (
        latest_pointer.get(
            "collection_directory"
        )
    )

    if not collection_directory_value:
        raise ValueError(
            "latest_collection.json 中既没有 "
            "daily_hotspots_path，也没有 "
            "collection_directory。"
        )

    collection_directory = resolve_path(
        collection_directory_value
    )

    hotspots_path = (
        collection_directory
        / "daily_hotspots.json"
    )

    return (
        hotspots_path,
        latest_pointer_path,
        latest_pointer,
    )


# ============================================================
# 精简日报数据构建
# ============================================================


def build_brief_data(
    report: dict[str, Any],
) -> dict[str, Any]:
    """
    从完整热点报告中提取精简日报数据。
    """
    snapshot_date = compact_text(
        report.get(
            "snapshot_date"
        )
    )

    source_project_count = safe_int(
        report.get(
            "source_project_count"
        )
    )

    original_hotspot_topics = (
        report.get(
            "hotspot_topics"
        )
    )

    if not isinstance(
        original_hotspot_topics,
        list,
    ):
        original_hotspot_topics = []

    source_hotspot_count = len(
        original_hotspot_topics
    )

    executive_summary = truncate_text(
        report.get(
            "executive_summary"
        ),
        EXECUTIVE_SUMMARY_MAX_CHARACTERS,
    )

    # --------------------------------------------------------
    # 主要热点
    # --------------------------------------------------------

    valid_topics = [
        topic
        for topic
        in original_hotspot_topics
        if isinstance(
            topic,
            dict,
        )
    ]

    valid_topics.sort(
        key=lambda topic: safe_int(
            topic.get(
                "importance_score"
            )
        ),
        reverse=True,
    )

    hotspot_topics: list[
        dict[str, Any]
    ] = []

    for topic in valid_topics[
        :MAX_HOTSPOT_TOPICS
    ]:
        topic_name = compact_text(
            topic.get(
                "topic_name"
            )
        )

        if not topic_name:
            continue

        related_projects = (
            compact_string_list(
                topic.get(
                    "related_projects"
                )
            )[
                :MAX_RELATED_PROJECTS
            ]
        )

        hotspot_topics.append(
            {
                "topic_name": (
                    topic_name
                ),

                "importance_score": (
                    min(
                        5,
                        max(
                            1,
                            safe_int(
                                topic.get(
                                    "importance_score"
                                ),
                                1,
                            ),
                        ),
                    )
                ),

                "summary": (
                    truncate_text(
                        topic.get(
                            "summary"
                        ),
                        TOPIC_SUMMARY_MAX_CHARACTERS,
                    )
                ),

                "why_it_matters": (
                    truncate_text(
                        topic.get(
                            "why_it_matters"
                        ),
                        TOPIC_REASON_MAX_CHARACTERS,
                    )
                ),

                "related_projects": (
                    related_projects
                ),
            }
        )

    # --------------------------------------------------------
    # 重点项目
    # --------------------------------------------------------

    notable_projects_value = (
        report.get(
            "notable_projects"
        )
    )

    if not isinstance(
        notable_projects_value,
        list,
    ):
        notable_projects_value = []

    valid_projects = [
        project
        for project
        in notable_projects_value
        if isinstance(
            project,
            dict,
        )
    ]

    valid_projects.sort(
        key=lambda project: safe_int(
            project.get(
                "importance_score"
            )
        ),
        reverse=True,
    )

    notable_projects: list[
        dict[str, Any]
    ] = []

    for project in valid_projects[
        :MAX_NOTABLE_PROJECTS
    ]:
        full_name = compact_text(
            project.get(
                "full_name"
            )
        )

        if not full_name:
            continue

        notable_projects.append(
            {
                "full_name": (
                    full_name
                ),

                "importance_score": (
                    min(
                        5,
                        max(
                            1,
                            safe_int(
                                project.get(
                                    "importance_score"
                                ),
                                1,
                            ),
                        ),
                    )
                ),

                "summary": (
                    truncate_text(
                        project.get(
                            "summary"
                        ),
                        PROJECT_SUMMARY_MAX_CHARACTERS,
                    )
                ),

                "why_watch": (
                    truncate_text(
                        project.get(
                            "why_watch"
                        ),
                        PROJECT_REASON_MAX_CHARACTERS,
                    )
                ),
            }
        )

    # --------------------------------------------------------
    # 后续观察
    # --------------------------------------------------------

    watch_next = [
        truncate_text(
            item,
            WATCH_ITEM_MAX_CHARACTERS,
        )
        for item in compact_string_list(
            report.get(
                "watch_next"
            )
        )[
            :MAX_WATCH_ITEMS
        ]
    ]

    return {
        "snapshot_date": (
            snapshot_date
        ),

        "source_project_count": (
            source_project_count
        ),

        "source_hotspot_count": (
            source_hotspot_count
        ),

        "executive_summary": (
            executive_summary
        ),

        "hotspot_topics": (
            hotspot_topics
        ),

        "notable_projects": (
            notable_projects
        ),

        "watch_next": (
            watch_next
        ),

        "generated_at": datetime.now(
            ZoneInfo(
                TIMEZONE
            )
        ).isoformat(),
    }


# ============================================================
# Markdown 渲染
# ============================================================


def render_markdown(
    brief: dict[str, Any],
) -> str:
    """
    将精简日报数据渲染为 Markdown。
    """
    snapshot_date = compact_text(
        brief.get(
            "snapshot_date"
        )
    )

    lines: list[str] = [
        (
            "# AI 技术热点速报｜"
            f"{snapshot_date}"
        ),
        "",
        "## 今日一句话",
        "",
        str(
            brief.get(
                "executive_summary"
            )
            or "今日暂无有效热点概览。"
        ),
    ]

    # --------------------------------------------------------
    # 主要热点
    # --------------------------------------------------------

    hotspot_topics = brief.get(
        "hotspot_topics"
    )

    if isinstance(
        hotspot_topics,
        list,
    ) and hotspot_topics:
        lines.extend(
            [
                "",
                "## 主要热点",
            ]
        )

        for index, topic in enumerate(
            hotspot_topics,
            start=1,
        ):
            if not isinstance(
                topic,
                dict,
            ):
                continue

            topic_name = compact_text(
                topic.get(
                    "topic_name"
                )
            )

            importance_score = safe_int(
                topic.get(
                    "importance_score"
                ),
                1,
            )

            lines.extend(
                [
                    "",
                    (
                        f"### {index}. "
                        f"{topic_name}"
                        f"（{importance_score}/5）"
                    ),
                    "",
                ]
            )

            summary = compact_text(
                topic.get(
                    "summary"
                )
            )

            if summary:
                lines.append(
                    summary
                )

            why_it_matters = compact_text(
                topic.get(
                    "why_it_matters"
                )
            )

            if why_it_matters:
                lines.extend(
                    [
                        "",
                        (
                            "**关注原因：** "
                            f"{why_it_matters}"
                        ),
                    ]
                )

            related_projects = (
                compact_string_list(
                    topic.get(
                        "related_projects"
                    )
                )
            )

            if related_projects:
                lines.extend(
                    [
                        "",
                        (
                            "**相关项目：** "
                            + "、".join(
                                related_projects
                            )
                        ),
                    ]
                )

    # --------------------------------------------------------
    # 重点项目
    # --------------------------------------------------------

    notable_projects = brief.get(
        "notable_projects"
    )

    if isinstance(
        notable_projects,
        list,
    ) and notable_projects:
        lines.extend(
            [
                "",
                "## 今日重点项目",
                "",
            ]
        )

        for project in notable_projects:
            if not isinstance(
                project,
                dict,
            ):
                continue

            full_name = compact_text(
                project.get(
                    "full_name"
                )
            )

            summary = compact_text(
                project.get(
                    "summary"
                )
            )

            why_watch = compact_text(
                project.get(
                    "why_watch"
                )
            )

            description_parts = [
                text
                for text in (
                    summary,
                    why_watch,
                )
                if text
            ]

            description = "；".join(
                description_parts
            )

            if description:
                lines.append(
                    f"- **{full_name}**："
                    f"{description}"
                )
            else:
                lines.append(
                    f"- **{full_name}**"
                )

    # --------------------------------------------------------
    # 后续观察
    # --------------------------------------------------------

    watch_next = compact_string_list(
        brief.get(
            "watch_next"
        )
    )

    if watch_next:
        lines.extend(
            [
                "",
                "## 后续观察",
                "",
            ]
        )

        for item in watch_next:
            lines.append(
                f"- {item}"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            (
                "分析项目："
                f"{brief.get('source_project_count', 0)} 个"
                "｜热点主题："
                f"{brief.get('source_hotspot_count', 0)} 个"
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# 持久化
# ============================================================


def persist_daily_brief(
    *,
    brief: dict[str, Any],
    brief_json_path: Path,
    markdown_path: Path,
) -> dict[str, str]:
    """
    将精简日报同步保存到：

    1. SQLite；
    2. RAG JSONL 文档源。
    """
    return save_daily_brief_assets(
        project_root=PROJECT_ROOT,
        brief=brief,
        brief_json_path=(
            brief_json_path
        ),
        markdown_path=(
            markdown_path
        ),
    )


# ============================================================
# 主程序
# ============================================================


def main() -> int:
    """
    程序入口。
    """
    try:
        snapshot_date = (
            resolve_snapshot_date()
        )

        (
            hotspots_path,
            latest_pointer_path,
            latest_pointer,
        ) = locate_daily_hotspots(
            snapshot_date
        )

        report = read_json(
            hotspots_path
        )

        if not isinstance(
            report,
            dict,
        ):
            raise ValueError(
                "daily_hotspots.json "
                "根节点必须是对象。"
            )

    except Exception as exc:
        print(
            "读取完整热点日报失败："
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    brief = build_brief_data(
        report
    )

    if not brief.get(
        "snapshot_date"
    ):
        brief[
            "snapshot_date"
        ] = snapshot_date

    collection_directory = (
        hotspots_path.parent
    )

    brief_markdown_path = (
        collection_directory
        / "github_daily_brief.md"
    )

    brief_json_path = (
        collection_directory
        / "github_daily_brief.json"
    )

    markdown_content = (
        render_markdown(
            brief
        )
    )

    try:
        write_text(
            brief_markdown_path,
            markdown_content,
        )

        write_json(
            brief_json_path,
            brief,
        )

    except Exception as exc:
        print(
            "精简日报文件保存失败："
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------
    # 写入 SQLite 和 RAG 文档源
    # --------------------------------------------------------

    persistence_status = "success"
    persistence_error: str | None = None
    persistence_paths: dict[
        str,
        str,
    ] = {}

    try:
        persistence_paths = (
            persist_daily_brief(
                brief=brief,
                brief_json_path=(
                    brief_json_path
                ),
                markdown_path=(
                    brief_markdown_path
                ),
            )
        )

    except Exception as exc:
        persistence_status = "failed"
        persistence_error = str(
            exc
        )

    # --------------------------------------------------------
    # 更新 latest_collection.json
    # --------------------------------------------------------

    latest_pointer[
        "github_daily_brief_path"
    ] = str(
        brief_markdown_path
    )

    latest_pointer[
        "github_daily_brief_json_path"
    ] = str(
        brief_json_path
    )

    latest_pointer[
        "github_daily_brief_generated_at"
    ] = brief[
        "generated_at"
    ]

    latest_pointer[
        "github_daily_brief_persistence_status"
    ] = persistence_status

    if persistence_error:
        latest_pointer[
            "github_daily_brief_persistence_error"
        ] = persistence_error

    else:
        latest_pointer.pop(
            "github_daily_brief_persistence_error",
            None,
        )

    if persistence_paths.get(
        "database_path"
    ):
        latest_pointer[
            "intelligence_database_path"
        ] = persistence_paths[
            "database_path"
        ]

    if persistence_paths.get(
        "rag_documents_path"
    ):
        latest_pointer[
            "daily_brief_rag_documents_path"
        ] = persistence_paths[
            "rag_documents_path"
        ]

    try:
        write_json(
            latest_pointer_path,
            latest_pointer,
        )

    except Exception as exc:
        print(
            "latest_collection.json 更新失败："
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------
    # 输出结果
    # --------------------------------------------------------

    print(
        "=" * 78
    )

    print(
        "GitHub 精简日报生成与持久化完成"
    )

    print(
        "=" * 78
    )

    print(
        f"日期：{snapshot_date}"
    )

    print(
        "未调用 LLM。"
    )

    print(
        "分析项目数："
        f"{brief['source_project_count']}"
    )

    print(
        "原始热点数："
        f"{brief['source_hotspot_count']}"
    )

    print(
        "精简热点数："
        f"{len(brief['hotspot_topics'])}"
    )

    print(
        "重点项目数："
        f"{len(brief['notable_projects'])}"
    )

    print(
        "后续观察数："
        f"{len(brief['watch_next'])}"
    )

    print(
        "日报字符数："
        f"{len(markdown_content):,}"
    )

    print(
        f"持久化状态：{persistence_status}"
    )

    print(
        f"Markdown：{brief_markdown_path}"
    )

    print(
        f"JSON：{brief_json_path}"
    )

    if persistence_paths:
        print(
            "SQLite："
            f"{persistence_paths.get('database_path')}"
        )

        print(
            "RAG 文档源："
            f"{persistence_paths.get('rag_documents_path')}"
        )

    if persistence_error:
        print(
            "持久化错误："
            f"{persistence_error}"
        )

    if persistence_status == "success":
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )