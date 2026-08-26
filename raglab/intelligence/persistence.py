from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATABASE_RELATIVE_PATH = Path(
    "storage/intelligence/github_intelligence.sqlite3"
)

DEFAULT_RAG_DOCUMENTS_RELATIVE_ROOT = Path(
    "data/intelligence/rag_documents"
)


# ============================================================
# 通用工具
# ============================================================


def _json_dumps(
    value: Any,
) -> str:
    """
    将 Python 对象转换为 UTF-8 JSON 字符串。
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


def _safe_int(
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


def _compact_text(
    value: Any,
) -> str:
    """
    将多行或带有多余空格的文本压缩成单行。
    """
    if value is None:
        return ""

    return " ".join(
        str(
            value
        ).split()
    ).strip()


def _string_list(
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
        text = _compact_text(
            item
        )

        if text:
            results.append(
                text
            )

    return results


def _list_section(
    title: str,
    values: Any,
) -> list[str]:
    """
    将列表字段转换为适合 RAG 文档的文本段落。
    """
    items = _string_list(
        values
    )

    if not items:
        return []

    lines = [
        f"{title}："
    ]

    for item in items:
        lines.append(
            f"- {item}"
        )

    return lines


def _relative_path_text(
    project_root: Path,
    path_value: str | Path,
) -> str:
    """
    尽量将绝对路径转换为项目内相对路径。

    如果路径不属于项目目录，则保留绝对路径。
    """
    project_root = Path(
        project_root
    ).resolve()

    path = Path(
        path_value
    ).resolve()

    try:
        return str(
            path.relative_to(
                project_root
            )
        )
    except ValueError:
        return str(
            path
        )


def resolve_intelligence_database_path(
    project_root: Path,
) -> Path:
    """
    返回技术情报 SQLite 数据库路径。
    """
    return (
        Path(
            project_root
        ).resolve()
        / DEFAULT_DATABASE_RELATIVE_PATH
    )


def resolve_rag_documents_directory(
    project_root: Path,
    snapshot_date: str,
) -> Path:
    """
    返回某一天的 RAG 文档源目录。
    """
    return (
        Path(
            project_root
        ).resolve()
        / DEFAULT_RAG_DOCUMENTS_RELATIVE_ROOT
        / snapshot_date
    )


# ============================================================
# SQLite 初始化
# ============================================================


def initialize_intelligence_tables(
    database_path: Path,
) -> None:
    """
    创建项目摘要、热点报告、热点主题和精简日报表。

    使用 IF NOT EXISTS，
    不会破坏 storage.py 已经创建的原有表。
    """
    database_path = Path(
        database_path
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS repository_llm_summaries (
                snapshot_date TEXT NOT NULL,
                full_name TEXT NOT NULL,

                project_type TEXT,
                is_relevant INTEGER NOT NULL DEFAULT 0,
                relevance_level TEXT,
                relevance_reason TEXT,

                hotspot_value INTEGER NOT NULL DEFAULT 1,
                one_sentence_summary TEXT,

                keywords_json TEXT NOT NULL DEFAULT '[]',
                summary_json TEXT NOT NULL,

                source_file TEXT,
                analyzed_at TEXT,
                model TEXT,

                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,

                PRIMARY KEY (
                    snapshot_date,
                    full_name
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_repository_llm_summaries_full_name
            ON repository_llm_summaries (
                full_name
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_repository_llm_summaries_date_score
            ON repository_llm_summaries (
                snapshot_date,
                hotspot_value DESC
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_hotspot_reports (
                snapshot_date TEXT PRIMARY KEY,

                report_title TEXT,
                executive_summary TEXT,
                source_project_count INTEGER NOT NULL DEFAULT 0,

                report_json TEXT NOT NULL,

                source_summaries_path TEXT,
                full_markdown_path TEXT,
                compact_markdown_path TEXT,

                generated_at TEXT,
                model TEXT,

                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_hotspot_topics (
                snapshot_date TEXT NOT NULL,
                topic_name TEXT NOT NULL,

                importance_score INTEGER NOT NULL DEFAULT 1,
                summary TEXT,
                why_it_matters TEXT,

                related_projects_json TEXT NOT NULL DEFAULT '[]',
                engineering_signals_json TEXT NOT NULL DEFAULT '[]',

                PRIMARY KEY (
                    snapshot_date,
                    topic_name
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_daily_hotspot_topics_score
            ON daily_hotspot_topics (
                snapshot_date,
                importance_score DESC
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_intelligence_briefs (
                snapshot_date TEXT PRIMARY KEY,

                executive_summary TEXT,
                source_project_count INTEGER NOT NULL DEFAULT 0,
                source_hotspot_count INTEGER NOT NULL DEFAULT 0,

                brief_json TEXT NOT NULL,
                markdown_path TEXT,

                generated_at TEXT
            )
            """
        )

        connection.commit()


# ============================================================
# JSONL RAG 文档写入
# ============================================================


def _read_jsonl_documents(
    path: Path,
) -> list[dict[str, Any]]:
    """
    读取现有 JSONL 文档。

    个别损坏行会被忽略，不影响其他文档。
    """
    path = Path(
        path
    )

    if not path.exists():
        return []

    documents: list[
        dict[str, Any]
    ] = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            data = json.loads(
                line
            )
        except json.JSONDecodeError:
            continue

        if isinstance(
            data,
            dict,
        ):
            documents.append(
                data
            )

    return documents


def upsert_jsonl_documents(
    path: Path,
    documents: Iterable[
        dict[str, Any]
    ],
) -> int:
    """
    按 document_id 更新 JSONL 文档。

    同一个项目或热点重复生成时，
    会覆盖旧版本，而不是重复追加。
    """
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_documents = (
        _read_jsonl_documents(
            path
        )
    )

    document_map: dict[
        str,
        dict[str, Any],
    ] = {}

    document_order: list[str] = []

    for document in existing_documents:
        document_id = _compact_text(
            document.get(
                "document_id"
            )
        )

        if not document_id:
            continue

        if document_id not in document_map:
            document_order.append(
                document_id
            )

        document_map[
            document_id
        ] = document

    changed_count = 0

    for document in documents:
        if not isinstance(
            document,
            dict,
        ):
            continue

        document_id = _compact_text(
            document.get(
                "document_id"
            )
        )

        if not document_id:
            continue

        if document_id not in document_map:
            document_order.append(
                document_id
            )

        document_map[
            document_id
        ] = document

        changed_count += 1

    output_lines = [
        json.dumps(
            document_map[
                document_id
            ],
            ensure_ascii=False,
        )
        for document_id in document_order
        if document_id in document_map
    ]

    temporary_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temporary_path.write_text(
        (
            "\n".join(
                output_lines
            )
            + (
                "\n"
                if output_lines
                else ""
            )
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )

    return changed_count


# ============================================================
# 项目摘要保存
# ============================================================


def _repository_hotspot_value(
    summary: dict[str, Any],
) -> int:
    """
    兼容当前 hotspot_value 和未来的
    intrinsic_hotspot_value 字段。
    """
    value = summary.get(
        "intrinsic_hotspot_value"
    )

    if value is None:
        value = summary.get(
            "hotspot_value"
        )

    return min(
        5,
        max(
            1,
            _safe_int(
                value,
                1,
            ),
        ),
    )


def build_repository_summary_rag_document(
    *,
    summary: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    """
    将单项目 LLM 摘要转换为 RAG 文档。
    """
    snapshot_date = _compact_text(
        summary.get(
            "source_snapshot_date"
        )
        or summary.get(
            "snapshot_date"
        )
    )

    full_name = _compact_text(
        summary.get(
            "full_name"
        )
    )

    project_type = _compact_text(
        summary.get(
            "project_type"
        )
    )

    one_sentence_summary = (
        _compact_text(
            summary.get(
                "one_sentence_summary"
            )
        )
    )

    relevance_reason = _compact_text(
        summary.get(
            "relevance_reason"
        )
    )

    hotspot_reason = _compact_text(
        summary.get(
            "hotspot_reason"
        )
    )

    lines: list[str] = [
        f"项目：{full_name}",
        f"日期：{snapshot_date}",
    ]

    if project_type:
        lines.append(
            f"项目类型：{project_type}"
        )

    if one_sentence_summary:
        lines.extend(
            [
                "",
                "项目摘要：",
                one_sentence_summary,
            ]
        )

    if relevance_reason:
        lines.extend(
            [
                "",
                "相关性：",
                relevance_reason,
            ]
        )

    section_definitions = (
        (
            "解决的问题",
            "problem_solved",
        ),
        (
            "核心能力",
            "core_capabilities",
        ),
        (
            "技术特点",
            "technical_features",
        ),
        (
            "使用场景",
            "use_cases",
        ),
        (
            "近期变化",
            "recent_changes",
        ),
        (
            "社区信号",
            "community_signals",
        ),
        (
            "局限性和不确定事项",
            "limitations_or_uncertainties",
        ),
    )

    for title, field in section_definitions:
        section_lines = _list_section(
            title,
            summary.get(
                field
            ),
        )

        if section_lines:
            lines.append("")
            lines.extend(
                section_lines
            )

    if hotspot_reason:
        lines.extend(
            [
                "",
                "热点价值判断：",
                hotspot_reason,
            ]
        )

    keywords = _string_list(
        summary.get(
            "keywords"
        )
    )

    if keywords:
        lines.extend(
            [
                "",
                "关键词："
                + "、".join(
                    keywords
                ),
            ]
        )

    return {
        "document_id": (
            "repository_summary:"
            f"{snapshot_date}:"
            f"{full_name}"
        ),

        "doc_type": (
            "repository_summary"
        ),

        "title": (
            f"{full_name} 项目摘要"
        ),

        "text": "\n".join(
            lines
        ).strip(),

        "metadata": {
            "snapshot_date": (
                snapshot_date
            ),

            "repository": (
                full_name
            ),

            "project_type": (
                project_type
            ),

            "is_relevant": bool(
                summary.get(
                    "is_relevant"
                )
            ),

            "relevance_level": (
                summary.get(
                    "relevance_level"
                )
            ),

            "hotspot_value": (
                _repository_hotspot_value(
                    summary
                )
            ),

            "keywords": keywords,

            "source_path": (
                source_path
            ),
        },
    }


def save_repository_summary_assets(
    *,
    project_root: Path,
    summary: dict[str, Any],
    source_file: Path,
) -> dict[str, str]:
    """
    保存一条项目摘要：

    1. 写入 SQLite；
    2. 写入 repository_summaries.jsonl。
    """
    project_root = Path(
        project_root
    ).resolve()

    source_file = Path(
        source_file
    ).resolve()

    database_path = (
        resolve_intelligence_database_path(
            project_root
        )
    )

    initialize_intelligence_tables(
        database_path
    )

    snapshot_date = _compact_text(
        summary.get(
            "source_snapshot_date"
        )
        or summary.get(
            "snapshot_date"
        )
    )

    full_name = _compact_text(
        summary.get(
            "full_name"
        )
    )

    if not snapshot_date:
        raise ValueError(
            "项目摘要缺少 source_snapshot_date。"
        )

    if not full_name:
        raise ValueError(
            "项目摘要缺少 full_name。"
        )

    api_data = summary.get(
        "api"
    )

    if not isinstance(
        api_data,
        dict,
    ):
        api_data = {}

    usage = api_data.get(
        "usage"
    )

    if not isinstance(
        usage,
        dict,
    ):
        usage = {}

    source_path_text = (
        _relative_path_text(
            project_root,
            source_file,
        )
    )

    keywords = _string_list(
        summary.get(
            "keywords"
        )
    )

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            INSERT INTO repository_llm_summaries (
                snapshot_date,
                full_name,

                project_type,
                is_relevant,
                relevance_level,
                relevance_reason,

                hotspot_value,
                one_sentence_summary,

                keywords_json,
                summary_json,

                source_file,
                analyzed_at,
                model,

                prompt_tokens,
                completion_tokens,
                total_tokens
            )
            VALUES (
                ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT (
                snapshot_date,
                full_name
            )
            DO UPDATE SET
                project_type = excluded.project_type,
                is_relevant = excluded.is_relevant,
                relevance_level = excluded.relevance_level,
                relevance_reason = excluded.relevance_reason,

                hotspot_value = excluded.hotspot_value,
                one_sentence_summary = excluded.one_sentence_summary,

                keywords_json = excluded.keywords_json,
                summary_json = excluded.summary_json,

                source_file = excluded.source_file,
                analyzed_at = excluded.analyzed_at,
                model = excluded.model,

                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                total_tokens = excluded.total_tokens
            """,
            (
                snapshot_date,
                full_name,

                _compact_text(
                    summary.get(
                        "project_type"
                    )
                ),

                int(
                    bool(
                        summary.get(
                            "is_relevant"
                        )
                    )
                ),

                _compact_text(
                    summary.get(
                        "relevance_level"
                    )
                )
                or None,

                _compact_text(
                    summary.get(
                        "relevance_reason"
                    )
                ),

                _repository_hotspot_value(
                    summary
                ),

                _compact_text(
                    summary.get(
                        "one_sentence_summary"
                    )
                ),

                _json_dumps(
                    keywords
                ),

                _json_dumps(
                    summary
                ),

                source_path_text,

                _compact_text(
                    summary.get(
                        "analyzed_at"
                    )
                ),

                _compact_text(
                    api_data.get(
                        "model"
                    )
                ),

                _safe_int(
                    usage.get(
                        "prompt_tokens"
                    )
                ),

                _safe_int(
                    usage.get(
                        "completion_tokens"
                    )
                ),

                _safe_int(
                    usage.get(
                        "total_tokens"
                    )
                ),
            ),
        )

        connection.commit()

    rag_directory = (
        resolve_rag_documents_directory(
            project_root,
            snapshot_date,
        )
    )

    rag_path = (
        rag_directory
        / "repository_summaries.jsonl"
    )

    rag_document = (
        build_repository_summary_rag_document(
            summary=summary,
            source_path=(
                source_path_text
            ),
        )
    )

    upsert_jsonl_documents(
        rag_path,
        [
            rag_document
        ],
    )

    return {
        "database_path": str(
            database_path
        ),

        "rag_documents_path": str(
            rag_path
        ),
    }


# ============================================================
# 每日热点报告保存
# ============================================================


def build_hotspot_topic_rag_documents(
    *,
    report: dict[str, Any],
    source_path: str,
) -> list[dict[str, Any]]:
    """
    将每个热点主题转换为一份独立 RAG 文档。
    """
    snapshot_date = _compact_text(
        report.get(
            "snapshot_date"
        )
    )

    topics = report.get(
        "hotspot_topics"
    )

    if not isinstance(
        topics,
        list,
    ):
        return []

    documents: list[
        dict[str, Any]
    ] = []

    for topic in topics:
        if not isinstance(
            topic,
            dict,
        ):
            continue

        topic_name = _compact_text(
            topic.get(
                "topic_name"
            )
        )

        if not topic_name:
            continue

        summary = _compact_text(
            topic.get(
                "summary"
            )
        )

        why_it_matters = _compact_text(
            topic.get(
                "why_it_matters"
            )
        )

        related_projects = _string_list(
            topic.get(
                "related_projects"
            )
        )

        engineering_signals = _string_list(
            topic.get(
                "engineering_signals"
            )
        )

        lines = [
            f"热点主题：{topic_name}",
            f"日期：{snapshot_date}",
        ]

        if summary:
            lines.extend(
                [
                    "",
                    "热点摘要：",
                    summary,
                ]
            )

        if why_it_matters:
            lines.extend(
                [
                    "",
                    "关注原因：",
                    why_it_matters,
                ]
            )

        if related_projects:
            lines.extend(
                [
                    "",
                    "相关项目："
                    + "、".join(
                        related_projects
                    ),
                ]
            )

        if engineering_signals:
            lines.extend(
                [
                    "",
                    "工程信号：",
                ]
            )

            for signal in engineering_signals:
                lines.append(
                    f"- {signal}"
                )

        documents.append(
            {
                "document_id": (
                    "daily_hotspot:"
                    f"{snapshot_date}:"
                    f"{topic_name}"
                ),

                "doc_type": (
                    "daily_hotspot"
                ),

                "title": (
                    f"{snapshot_date} "
                    f"{topic_name}"
                ),

                "text": "\n".join(
                    lines
                ).strip(),

                "metadata": {
                    "snapshot_date": (
                        snapshot_date
                    ),

                    "topic_name": (
                        topic_name
                    ),

                    "importance_score": (
                        min(
                            5,
                            max(
                                1,
                                _safe_int(
                                    topic.get(
                                        "importance_score"
                                    ),
                                    1,
                                ),
                            ),
                        )
                    ),

                    "related_projects": (
                        related_projects
                    ),

                    "source_path": (
                        source_path
                    ),
                },
            }
        )

    return documents


def save_daily_hotspot_assets(
    *,
    project_root: Path,
    report: dict[str, Any],
    hotspots_json_path: Path,
    full_markdown_path: Path,
) -> dict[str, str]:
    """
    保存每日热点分析：

    1. 写入 daily_hotspot_reports；
    2. 替换当天 daily_hotspot_topics；
    3. 写入 daily_hotspots.jsonl。
    """
    project_root = Path(
        project_root
    ).resolve()

    hotspots_json_path = Path(
        hotspots_json_path
    ).resolve()

    full_markdown_path = Path(
        full_markdown_path
    ).resolve()

    database_path = (
        resolve_intelligence_database_path(
            project_root
        )
    )

    initialize_intelligence_tables(
        database_path
    )

    snapshot_date = _compact_text(
        report.get(
            "snapshot_date"
        )
    )

    if not snapshot_date:
        raise ValueError(
            "热点报告缺少 snapshot_date。"
        )

    api_data = report.get(
        "api"
    )

    if not isinstance(
        api_data,
        dict,
    ):
        api_data = {}

    usage = api_data.get(
        "usage"
    )

    if not isinstance(
        usage,
        dict,
    ):
        usage = {}

    source_summaries_path = _compact_text(
        report.get(
            "source_summaries_path"
        )
    )

    hotspots_path_text = (
        _relative_path_text(
            project_root,
            hotspots_json_path,
        )
    )

    markdown_path_text = (
        _relative_path_text(
            project_root,
            full_markdown_path,
        )
    )

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            INSERT INTO daily_hotspot_reports (
                snapshot_date,

                report_title,
                executive_summary,
                source_project_count,

                report_json,

                source_summaries_path,
                full_markdown_path,
                compact_markdown_path,

                generated_at,
                model,

                prompt_tokens,
                completion_tokens,
                total_tokens
            )
            VALUES (
                ?,
                ?, ?, ?,
                ?,
                ?, ?, NULL,
                ?, ?,
                ?, ?, ?
            )
            ON CONFLICT (
                snapshot_date
            )
            DO UPDATE SET
                report_title = excluded.report_title,
                executive_summary = excluded.executive_summary,
                source_project_count = excluded.source_project_count,

                report_json = excluded.report_json,

                source_summaries_path = excluded.source_summaries_path,
                full_markdown_path = excluded.full_markdown_path,

                generated_at = excluded.generated_at,
                model = excluded.model,

                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                total_tokens = excluded.total_tokens
            """,
            (
                snapshot_date,

                _compact_text(
                    report.get(
                        "report_title"
                    )
                ),

                _compact_text(
                    report.get(
                        "executive_summary"
                    )
                ),

                _safe_int(
                    report.get(
                        "source_project_count"
                    )
                ),

                _json_dumps(
                    report
                ),

                source_summaries_path,

                markdown_path_text,

                _compact_text(
                    report.get(
                        "generated_at"
                    )
                ),

                _compact_text(
                    api_data.get(
                        "model"
                    )
                ),

                _safe_int(
                    usage.get(
                        "prompt_tokens"
                    )
                ),

                _safe_int(
                    usage.get(
                        "completion_tokens"
                    )
                ),

                _safe_int(
                    usage.get(
                        "total_tokens"
                    )
                ),
            ),
        )

        connection.execute(
            """
            DELETE FROM daily_hotspot_topics
            WHERE snapshot_date = ?
            """,
            (
                snapshot_date,
            ),
        )

        topics = report.get(
            "hotspot_topics"
        )

        if not isinstance(
            topics,
            list,
        ):
            topics = []

        for topic in topics:
            if not isinstance(
                topic,
                dict,
            ):
                continue

            topic_name = _compact_text(
                topic.get(
                    "topic_name"
                )
            )

            if not topic_name:
                continue

            connection.execute(
                """
                INSERT INTO daily_hotspot_topics (
                    snapshot_date,
                    topic_name,

                    importance_score,
                    summary,
                    why_it_matters,

                    related_projects_json,
                    engineering_signals_json
                )
                VALUES (
                    ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    snapshot_date,
                    topic_name,

                    min(
                        5,
                        max(
                            1,
                            _safe_int(
                                topic.get(
                                    "importance_score"
                                ),
                                1,
                            ),
                        ),
                    ),

                    _compact_text(
                        topic.get(
                            "summary"
                        )
                    ),

                    _compact_text(
                        topic.get(
                            "why_it_matters"
                        )
                    ),

                    _json_dumps(
                        _string_list(
                            topic.get(
                                "related_projects"
                            )
                        )
                    ),

                    _json_dumps(
                        _string_list(
                            topic.get(
                                "engineering_signals"
                            )
                        )
                    ),
                ),
            )

        connection.commit()

    rag_directory = (
        resolve_rag_documents_directory(
            project_root,
            snapshot_date,
        )
    )

    rag_path = (
        rag_directory
        / "daily_hotspots.jsonl"
    )

    rag_documents = (
        build_hotspot_topic_rag_documents(
            report=report,
            source_path=(
                hotspots_path_text
            ),
        )
    )

    upsert_jsonl_documents(
        rag_path,
        rag_documents,
    )

    return {
        "database_path": str(
            database_path
        ),

        "rag_documents_path": str(
            rag_path
        ),
    }


# ============================================================
# 精简日报保存
# ============================================================


def build_daily_brief_rag_document(
    *,
    brief: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    """
    将精简日报转换为单份 RAG 文档。
    """
    snapshot_date = _compact_text(
        brief.get(
            "snapshot_date"
        )
    )

    executive_summary = _compact_text(
        brief.get(
            "executive_summary"
        )
    )

    lines: list[str] = [
        f"AI 技术热点速报：{snapshot_date}",
    ]

    if executive_summary:
        lines.extend(
            [
                "",
                "今日概览：",
                executive_summary,
            ]
        )

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
                "主要热点：",
            ]
        )

        for topic in hotspot_topics:
            if not isinstance(
                topic,
                dict,
            ):
                continue

            topic_name = _compact_text(
                topic.get(
                    "topic_name"
                )
            )

            summary = _compact_text(
                topic.get(
                    "summary"
                )
            )

            if topic_name and summary:
                lines.append(
                    f"- {topic_name}：{summary}"
                )
            elif topic_name:
                lines.append(
                    f"- {topic_name}"
                )

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
                "重点项目：",
            ]
        )

        for project in notable_projects:
            if not isinstance(
                project,
                dict,
            ):
                continue

            full_name = _compact_text(
                project.get(
                    "full_name"
                )
            )

            summary = _compact_text(
                project.get(
                    "summary"
                )
            )

            if full_name and summary:
                lines.append(
                    f"- {full_name}：{summary}"
                )
            elif full_name:
                lines.append(
                    f"- {full_name}"
                )

    watch_next = _string_list(
        brief.get(
            "watch_next"
        )
    )

    if watch_next:
        lines.extend(
            [
                "",
                "后续观察：",
            ]
        )

        for item in watch_next:
            lines.append(
                f"- {item}"
            )

    return {
        "document_id": (
            "daily_brief:"
            f"{snapshot_date}"
        ),

        "doc_type": (
            "daily_brief"
        ),

        "title": (
            f"AI 技术热点速报 "
            f"{snapshot_date}"
        ),

        "text": "\n".join(
            lines
        ).strip(),

        "metadata": {
            "snapshot_date": (
                snapshot_date
            ),

            "source_project_count": (
                _safe_int(
                    brief.get(
                        "source_project_count"
                    )
                )
            ),

            "source_hotspot_count": (
                _safe_int(
                    brief.get(
                        "source_hotspot_count"
                    )
                )
            ),

            "source_path": (
                source_path
            ),
        },
    }


def save_daily_brief_assets(
    *,
    project_root: Path,
    brief: dict[str, Any],
    brief_json_path: Path,
    markdown_path: Path,
) -> dict[str, str]:
    """
    保存精简日报：

    1. 写入 daily_intelligence_briefs；
    2. 更新 daily_hotspot_reports 的精简日报路径；
    3. 写入 daily_brief.jsonl。
    """
    project_root = Path(
        project_root
    ).resolve()

    brief_json_path = Path(
        brief_json_path
    ).resolve()

    markdown_path = Path(
        markdown_path
    ).resolve()

    database_path = (
        resolve_intelligence_database_path(
            project_root
        )
    )

    initialize_intelligence_tables(
        database_path
    )

    snapshot_date = _compact_text(
        brief.get(
            "snapshot_date"
        )
    )

    if not snapshot_date:
        raise ValueError(
            "精简日报缺少 snapshot_date。"
        )

    brief_json_path_text = (
        _relative_path_text(
            project_root,
            brief_json_path,
        )
    )

    markdown_path_text = (
        _relative_path_text(
            project_root,
            markdown_path,
        )
    )

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            INSERT INTO daily_intelligence_briefs (
                snapshot_date,

                executive_summary,
                source_project_count,
                source_hotspot_count,

                brief_json,
                markdown_path,
                generated_at
            )
            VALUES (
                ?,
                ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT (
                snapshot_date
            )
            DO UPDATE SET
                executive_summary = excluded.executive_summary,
                source_project_count = excluded.source_project_count,
                source_hotspot_count = excluded.source_hotspot_count,

                brief_json = excluded.brief_json,
                markdown_path = excluded.markdown_path,
                generated_at = excluded.generated_at
            """,
            (
                snapshot_date,

                _compact_text(
                    brief.get(
                        "executive_summary"
                    )
                ),

                _safe_int(
                    brief.get(
                        "source_project_count"
                    )
                ),

                _safe_int(
                    brief.get(
                        "source_hotspot_count"
                    )
                ),

                _json_dumps(
                    brief
                ),

                markdown_path_text,

                _compact_text(
                    brief.get(
                        "generated_at"
                    )
                ),
            ),
        )

        connection.execute(
            """
            UPDATE daily_hotspot_reports
            SET compact_markdown_path = ?
            WHERE snapshot_date = ?
            """,
            (
                markdown_path_text,
                snapshot_date,
            ),
        )

        connection.commit()

    rag_directory = (
        resolve_rag_documents_directory(
            project_root,
            snapshot_date,
        )
    )

    rag_path = (
        rag_directory
        / "daily_brief.jsonl"
    )

    rag_document = (
        build_daily_brief_rag_document(
            brief=brief,
            source_path=(
                brief_json_path_text
            ),
        )
    )

    upsert_jsonl_documents(
        rag_path,
        [
            rag_document
        ],
    )

    return {
        "database_path": str(
            database_path
        ),

        "rag_documents_path": str(
            rag_path
        ),
    }