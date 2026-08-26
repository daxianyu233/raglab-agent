from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests


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
    save_daily_hotspot_assets,
)


# ============================================================
# 配置
# ============================================================

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 3
MAX_OUTPUT_TOKENS = 4_000


SYSTEM_PROMPT = """
你是一名 AI 技术情报分析员。

你会收到同一天采集的多个 GitHub 项目的结构化摘要。
这些项目可能属于 AI Agent、RAG、MCP、语音 AI、3D 生成、
开发工具、自动化工具、教程或资源列表等不同类别。

你的任务是比较这些项目，而不是再次逐项目复述。

分析要求：

1. 只能依据输入的项目摘要，不得编造项目功能或外部事实。

2. 识别真正的跨项目技术趋势。
   如果某个方向只有一个项目支持，可以将其称为“值得关注的信号”，
   但不要伪装成多个项目共同形成的趋势。

3. 区分：
   - 核心技术热点；
   - 单个高价值项目；
   - 教程或资源聚合项目；
   - 成熟项目的普通更新；
   - 与当天主题关联较弱的项目。

4. 项目摘要中的 hotspot_value 只代表项目自身价值，
   不能直接当作当天项目间的最终排名。

5. GitHub 检索词应适合发现工程项目。
   ArXiv 检索词应更偏学术概念，而不是直接照搬仓库名称。

6. 输出中文分析。
   技术名称和搜索关键词可以保留英文。

7. 必须只输出一个合法 JSON 对象。
   不要输出 Markdown 代码块，不要输出 JSON 以外的文字。

输出结构：

{
  "report_title": "当天日报标题",

  "executive_summary": "用一段话概括当天最重要的技术信号",

  "hotspot_topics": [
    {
      "topic_name": "热点名称",

      "summary": "该热点的主要内容",

      "why_it_matters": "为什么值得关注",

      "related_projects": [
        "owner/repository"
      ],

      "engineering_signals": [
        "从项目、版本或社区问题中观察到的工程信号"
      ],

      "importance_score": 1
    }
  ],

  "notable_projects": [
    {
      "full_name": "owner/repository",

      "summary": "项目是什么",

      "why_watch": "为什么值得继续关注",

      "importance_score": 1
    }
  ],

  "low_priority_projects": [
    {
      "full_name": "owner/repository",

      "reason": "为什么不应作为当天核心热点"
    }
  ],

  "cross_project_patterns": [
    "多个项目共同体现出的工程趋势或社区需求"
  ],

  "github_search_queries": [
    {
      "query": "适合 GitHub Repository Search 的关键词",

      "purpose": "该检索词希望发现什么项目"
    }
  ],

  "arxiv_search_queries": [
    {
      "query": "适合 ArXiv 检索的英文研究词组",

      "purpose": "该检索词对应的研究问题"
    }
  ],

  "watch_next": [
    "接下来应该持续观察的技术、项目或问题"
  ]
}

importance_score 使用 1 到 5：

1 = 信息价值低；
2 = 次要信号；
3 = 值得观察；
4 = 明显热点；
5 = 当天最重要的技术热点。

热点数量建议为 2 到 5 个。
GitHub 和 ArXiv 检索词分别生成 5 到 10 个。
""".strip()


# ============================================================
# 参数和文件工具
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "读取逐项目 LLM 摘要，生成当天 GitHub "
            "技术热点、动态检索词和 Markdown 日报，"
            "并同步保存到 SQLite 和 RAG 文档源。"
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "处理日期，格式为 YYYY-MM-DD；"
            "不填写时使用北京时间当天。"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "重新调用模型生成当天热点日报。"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv(
            "DEEPSEEK_MODEL",
            DEFAULT_MODEL,
        ),
        help=(
            "DeepSeek 模型名称，"
            f"默认使用 {DEFAULT_MODEL}。"
        ),
    )

    return parser.parse_args()


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


def resolve_date(
    value: str | None,
) -> str:
    """
    确定需要处理的日期。
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


def resolve_path(
    value: Any,
) -> Path:
    """
    将指针中的路径转换为绝对路径。
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

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    安全转换整数。
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


def strip_json_fence(
    content: str,
) -> str:
    """
    去除模型可能返回的 Markdown 代码块。
    """
    text = content.strip()

    if text.startswith(
        "```json"
    ):
        text = text[7:]

    elif text.startswith(
        "```"
    ):
        text = text[3:]

    if text.endswith(
        "```"
    ):
        text = text[:-3]

    return text.strip()


# ============================================================
# 跨项目输入构建
# ============================================================


def build_project_payload(
    summaries: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """
    删除 API 元数据，只保留热点分析所需字段。
    """
    payload: list[
        dict[str, Any]
    ] = []

    for summary in summaries:
        hotspot_value = summary.get(
            "intrinsic_hotspot_value"
        )

        if hotspot_value is None:
            hotspot_value = summary.get(
                "hotspot_value"
            )

        payload.append(
            {
                "full_name": (
                    summary.get(
                        "full_name"
                    )
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

                "relevance_reason": (
                    summary.get(
                        "relevance_reason"
                    )
                ),

                "project_type": (
                    summary.get(
                        "project_type"
                    )
                ),

                "one_sentence_summary": (
                    summary.get(
                        "one_sentence_summary"
                    )
                ),

                "problem_solved": (
                    compact_string_list(
                        summary.get(
                            "problem_solved"
                        )
                    )
                ),

                "core_capabilities": (
                    compact_string_list(
                        summary.get(
                            "core_capabilities"
                        )
                    )
                ),

                "technical_features": (
                    compact_string_list(
                        summary.get(
                            "technical_features"
                        )
                    )
                ),

                "use_cases": (
                    compact_string_list(
                        summary.get(
                            "use_cases"
                        )
                    )
                ),

                "recent_changes": (
                    compact_string_list(
                        summary.get(
                            "recent_changes"
                        )
                    )
                ),

                "community_signals": (
                    compact_string_list(
                        summary.get(
                            "community_signals"
                        )
                    )
                ),

                "limitations_or_uncertainties": (
                    compact_string_list(
                        summary.get(
                            "limitations_or_uncertainties"
                        )
                    )
                ),

                "keywords": (
                    compact_string_list(
                        summary.get(
                            "keywords"
                        )
                    )
                ),

                "hotspot_value": (
                    safe_int(
                        hotspot_value,
                        1,
                    )
                ),

                "hotspot_reason": (
                    summary.get(
                        "hotspot_reason"
                    )
                ),
            }
        )

    return payload


def build_user_prompt(
    snapshot_date: str,
    summaries: list[
        dict[str, Any]
    ],
) -> str:
    """
    构造跨项目分析提示词。
    """
    payload = {
        "snapshot_date": (
            snapshot_date
        ),

        "project_count": len(
            summaries
        ),

        "projects": (
            build_project_payload(
                summaries
            )
        ),
    }

    return (
        "请比较以下当天 GitHub 项目摘要，"
        "生成规定结构的技术热点 JSON。\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# 模型输出标准化
# ============================================================


def normalize_query_items(
    value: Any,
) -> list[dict[str, str]]:
    """
    统一动态检索词结构。
    """
    if not isinstance(
        value,
        list,
    ):
        return []

    normalized: list[
        dict[str, str]
    ] = []

    for item in value:
        if isinstance(
            item,
            dict,
        ):
            query = str(
                item.get(
                    "query"
                )
                or ""
            ).strip()

            purpose = str(
                item.get(
                    "purpose"
                )
                or ""
            ).strip()

        else:
            query = str(
                item
            ).strip()

            purpose = ""

        if query:
            normalized.append(
                {
                    "query": query,
                    "purpose": purpose,
                }
            )

    return normalized


def normalize_score(
    value: Any,
) -> int:
    """
    将重要程度限制在 1 到 5。
    """
    return min(
        5,
        max(
            1,
            safe_int(
                value,
                1,
            ),
        ),
    )


def normalize_daily_result(
    data: dict[str, Any],
    *,
    snapshot_date: str,
    project_count: int,
) -> dict[str, Any]:
    """
    校正跨项目模型输出。
    """
    result = dict(
        data
    )

    result[
        "snapshot_date"
    ] = snapshot_date

    result[
        "source_project_count"
    ] = project_count

    result[
        "report_title"
    ] = str(
        result.get(
            "report_title"
        )
        or (
            "GitHub AI 技术热点日报 "
            f"{snapshot_date}"
        )
    ).strip()

    result[
        "executive_summary"
    ] = str(
        result.get(
            "executive_summary"
        )
        or ""
    ).strip()

    # --------------------------------------------------------
    # 热点主题
    # --------------------------------------------------------

    hotspot_topics: list[
        dict[str, Any]
    ] = []

    hotspot_topics_value = (
        result.get(
            "hotspot_topics"
        )
    )

    if isinstance(
        hotspot_topics_value,
        list,
    ):
        for topic in hotspot_topics_value:
            if not isinstance(
                topic,
                dict,
            ):
                continue

            topic_name = str(
                topic.get(
                    "topic_name"
                )
                or ""
            ).strip()

            if not topic_name:
                continue

            hotspot_topics.append(
                {
                    "topic_name": (
                        topic_name
                    ),

                    "summary": str(
                        topic.get(
                            "summary"
                        )
                        or ""
                    ).strip(),

                    "why_it_matters": str(
                        topic.get(
                            "why_it_matters"
                        )
                        or ""
                    ).strip(),

                    "related_projects": (
                        compact_string_list(
                            topic.get(
                                "related_projects"
                            )
                        )
                    ),

                    "engineering_signals": (
                        compact_string_list(
                            topic.get(
                                "engineering_signals"
                            )
                        )
                    ),

                    "importance_score": (
                        normalize_score(
                            topic.get(
                                "importance_score"
                            )
                        )
                    ),
                }
            )

    hotspot_topics.sort(
        key=lambda item: int(
            item.get(
                "importance_score"
            )
            or 0
        ),
        reverse=True,
    )

    result[
        "hotspot_topics"
    ] = hotspot_topics

    # --------------------------------------------------------
    # 重点项目
    # --------------------------------------------------------

    notable_projects: list[
        dict[str, Any]
    ] = []

    notable_projects_value = (
        result.get(
            "notable_projects"
        )
    )

    if isinstance(
        notable_projects_value,
        list,
    ):
        for project in notable_projects_value:
            if not isinstance(
                project,
                dict,
            ):
                continue

            full_name = str(
                project.get(
                    "full_name"
                )
                or ""
            ).strip()

            if not full_name:
                continue

            notable_projects.append(
                {
                    "full_name": (
                        full_name
                    ),

                    "summary": str(
                        project.get(
                            "summary"
                        )
                        or ""
                    ).strip(),

                    "why_watch": str(
                        project.get(
                            "why_watch"
                        )
                        or ""
                    ).strip(),

                    "importance_score": (
                        normalize_score(
                            project.get(
                                "importance_score"
                            )
                        )
                    ),
                }
            )

    notable_projects.sort(
        key=lambda item: int(
            item.get(
                "importance_score"
            )
            or 0
        ),
        reverse=True,
    )

    result[
        "notable_projects"
    ] = notable_projects

    # --------------------------------------------------------
    # 低优先级项目
    # --------------------------------------------------------

    low_priority_projects: list[
        dict[str, str]
    ] = []

    low_priority_value = (
        result.get(
            "low_priority_projects"
        )
    )

    if isinstance(
        low_priority_value,
        list,
    ):
        for project in low_priority_value:
            if not isinstance(
                project,
                dict,
            ):
                continue

            full_name = str(
                project.get(
                    "full_name"
                )
                or ""
            ).strip()

            if not full_name:
                continue

            low_priority_projects.append(
                {
                    "full_name": (
                        full_name
                    ),

                    "reason": str(
                        project.get(
                            "reason"
                        )
                        or ""
                    ).strip(),
                }
            )

    result[
        "low_priority_projects"
    ] = low_priority_projects

    result[
        "cross_project_patterns"
    ] = compact_string_list(
        result.get(
            "cross_project_patterns"
        )
    )

    result[
        "github_search_queries"
    ] = normalize_query_items(
        result.get(
            "github_search_queries"
        )
    )

    result[
        "arxiv_search_queries"
    ] = normalize_query_items(
        result.get(
            "arxiv_search_queries"
        )
    )

    result[
        "watch_next"
    ] = compact_string_list(
        result.get(
            "watch_next"
        )
    )

    return result


# ============================================================
# Markdown 渲染
# ============================================================


def render_markdown(
    report: dict[str, Any],
) -> str:
    """
    将结构化热点结果渲染为完整 Markdown 日报。
    """
    lines: list[str] = [
        (
            "# "
            + str(
                report.get(
                    "report_title"
                )
                or "GitHub AI 技术热点日报"
            )
        ),
        "",
        (
            "- 日期："
            f"{report.get('snapshot_date')}"
        ),
        (
            "- 分析项目数："
            f"{report.get('source_project_count', 0)}"
        ),
        "",
        "## 今日概览",
        "",
        str(
            report.get(
                "executive_summary"
            )
            or "暂无概览。"
        ),
    ]

    # --------------------------------------------------------
    # 主要技术热点
    # --------------------------------------------------------

    hotspot_topics = report.get(
        "hotspot_topics"
    )

    if isinstance(
        hotspot_topics,
        list,
    ) and hotspot_topics:
        lines.extend(
            [
                "",
                "## 主要技术热点",
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

            lines.extend(
                [
                    "",
                    (
                        f"### {index}. "
                        f"{topic.get('topic_name')} "
                        f"（{topic.get('importance_score')}/5）"
                    ),
                    "",
                ]
            )

            summary = str(
                topic.get(
                    "summary"
                )
                or ""
            ).strip()

            if summary:
                lines.append(
                    summary
                )

            why_it_matters = str(
                topic.get(
                    "why_it_matters"
                )
                or ""
            ).strip()

            if why_it_matters:
                lines.extend(
                    [
                        "",
                        (
                            "**关注原因：** "
                            + why_it_matters
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

            signals = compact_string_list(
                topic.get(
                    "engineering_signals"
                )
            )

            if signals:
                lines.extend(
                    [
                        "",
                        "**工程信号：**",
                    ]
                )

                for signal in signals:
                    lines.append(
                        f"- {signal}"
                    )

    # --------------------------------------------------------
    # 重点项目
    # --------------------------------------------------------

    notable_projects = report.get(
        "notable_projects"
    )

    if isinstance(
        notable_projects,
        list,
    ) and notable_projects:
        lines.extend(
            [
                "",
                "## 值得关注的项目",
            ]
        )

        for project in notable_projects:
            if not isinstance(
                project,
                dict,
            ):
                continue

            lines.extend(
                [
                    "",
                    (
                        "### "
                        f"{project.get('full_name')} "
                        f"（{project.get('importance_score')}/5）"
                    ),
                ]
            )

            summary = str(
                project.get(
                    "summary"
                )
                or ""
            ).strip()

            if summary:
                lines.extend(
                    [
                        "",
                        summary,
                    ]
                )

            why_watch = str(
                project.get(
                    "why_watch"
                )
                or ""
            ).strip()

            if why_watch:
                lines.extend(
                    [
                        "",
                        (
                            "**持续关注：** "
                            + why_watch
                        ),
                    ]
                )

    # --------------------------------------------------------
    # 跨项目趋势
    # --------------------------------------------------------

    patterns = compact_string_list(
        report.get(
            "cross_project_patterns"
        )
    )

    if patterns:
        lines.extend(
            [
                "",
                "## 跨项目工程趋势",
                "",
            ]
        )

        for pattern in patterns:
            lines.append(
                f"- {pattern}"
            )

    # --------------------------------------------------------
    # 非核心热点项目
    # --------------------------------------------------------

    low_priority_projects = (
        report.get(
            "low_priority_projects"
        )
    )

    if isinstance(
        low_priority_projects,
        list,
    ) and low_priority_projects:
        lines.extend(
            [
                "",
                "## 非核心热点项目",
                "",
            ]
        )

        for project in low_priority_projects:
            if not isinstance(
                project,
                dict,
            ):
                continue

            lines.append(
                "- "
                f"**{project.get('full_name')}**："
                f"{project.get('reason')}"
            )

    # --------------------------------------------------------
    # 动态检索词
    # --------------------------------------------------------

    github_queries = report.get(
        "github_search_queries"
    )

    arxiv_queries = report.get(
        "arxiv_search_queries"
    )

    if (
        isinstance(
            github_queries,
            list,
        )
        and github_queries
    ) or (
        isinstance(
            arxiv_queries,
            list,
        )
        and arxiv_queries
    ):
        lines.extend(
            [
                "",
                "## 下一轮动态检索词",
            ]
        )

    if isinstance(
        github_queries,
        list,
    ) and github_queries:
        lines.extend(
            [
                "",
                "### GitHub",
                "",
            ]
        )

        for item in github_queries:
            if not isinstance(
                item,
                dict,
            ):
                continue

            query = str(
                item.get(
                    "query"
                )
                or ""
            )

            purpose = str(
                item.get(
                    "purpose"
                )
                or ""
            )

            if purpose:
                lines.append(
                    f"- `{query}`：{purpose}"
                )
            else:
                lines.append(
                    f"- `{query}`"
                )

    if isinstance(
        arxiv_queries,
        list,
    ) and arxiv_queries:
        lines.extend(
            [
                "",
                "### ArXiv",
                "",
            ]
        )

        for item in arxiv_queries:
            if not isinstance(
                item,
                dict,
            ):
                continue

            query = str(
                item.get(
                    "query"
                )
                or ""
            )

            purpose = str(
                item.get(
                    "purpose"
                )
                or ""
            )

            if purpose:
                lines.append(
                    f"- `{query}`：{purpose}"
                )
            else:
                lines.append(
                    f"- `{query}`"
                )

    # --------------------------------------------------------
    # 后续观察
    # --------------------------------------------------------

    watch_next = compact_string_list(
        report.get(
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

    lines.append(
        ""
    )

    return "\n".join(
        lines
    )


# ============================================================
# DeepSeek 客户端
# ============================================================


class DeepSeekClient:
    """
    DeepSeek Chat Completions 客户端。
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
    ) -> None:
        self.model = model

        self.base_url = os.getenv(
            "DEEPSEEK_BASE_URL",
            DEFAULT_BASE_URL,
        ).rstrip(
            "/"
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Authorization": (
                    f"Bearer {api_key}"
                ),

                "Content-Type": (
                    "application/json"
                ),
            }
        )

    def summarize(
        self,
        user_prompt: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        """
        调用 DeepSeek 进行跨项目热点归纳。
        """
        request_body = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                    ),
                },

                {
                    "role": "user",
                    "content": (
                        user_prompt
                    ),
                },
            ],

            "stream": False,

            "temperature": 0.2,

            "max_tokens": (
                MAX_OUTPUT_TOKENS
            ),

            "response_format": {
                "type": "json_object",
            },

            "thinking": {
                "type": "disabled",
            },
        }

        last_error: Exception | None = None

        for attempt in range(
            MAX_RETRIES + 1
        ):
            try:
                response = self.session.post(
                    (
                        f"{self.base_url}"
                        "/chat/completions"
                    ),
                    json=request_body,
                    timeout=(
                        REQUEST_TIMEOUT_SECONDS
                    ),
                )

                if (
                    response.status_code
                    == 429
                    or response.status_code
                    >= 500
                ):
                    if attempt < MAX_RETRIES:
                        time.sleep(
                            min(
                                2**attempt * 2,
                                20,
                            )
                        )

                        continue

                if not response.ok:
                    raise RuntimeError(
                        "DeepSeek API 请求失败："
                        f"status={response.status_code}，"
                        f"response={response.text[:1000]}"
                    )

                response_data = (
                    response.json()
                )

                choices = response_data.get(
                    "choices"
                )

                if (
                    not isinstance(
                        choices,
                        list,
                    )
                    or not choices
                ):
                    raise RuntimeError(
                        "DeepSeek 响应缺少 choices。"
                    )

                first_choice = choices[0]

                message = first_choice.get(
                    "message"
                )

                if not isinstance(
                    message,
                    dict,
                ):
                    raise RuntimeError(
                        "DeepSeek 响应缺少 message。"
                    )

                content = str(
                    message.get(
                        "content"
                    )
                    or ""
                ).strip()

                if not content:
                    raise RuntimeError(
                        "DeepSeek 返回了空内容。"
                    )

                parsed = json.loads(
                    strip_json_fence(
                        content
                    )
                )

                if not isinstance(
                    parsed,
                    dict,
                ):
                    raise RuntimeError(
                        "模型输出 JSON 根节点不是对象。"
                    )

                usage = response_data.get(
                    "usage"
                )

                if not isinstance(
                    usage,
                    dict,
                ):
                    usage = {}

                metadata = {
                    "model": (
                        response_data.get(
                            "model"
                        )
                        or self.model
                    ),

                    "finish_reason": (
                        first_choice.get(
                            "finish_reason"
                        )
                    ),

                    "response_id": (
                        response_data.get(
                            "id"
                        )
                    ),

                    "usage": (
                        usage
                    ),
                }

                return (
                    parsed,
                    metadata,
                )

            except (
                requests.RequestException,
                json.JSONDecodeError,
                RuntimeError,
                ValueError,
            ) as exc:
                last_error = exc

                if attempt < MAX_RETRIES:
                    time.sleep(
                        min(
                            2**attempt * 2,
                            20,
                        )
                    )

                    continue

                break

        raise RuntimeError(
            "跨项目热点分析失败："
            f"{last_error}"
        )

    def close(
        self,
    ) -> None:
        """
        关闭 HTTP Session。
        """
        self.session.close()


# ============================================================
# 持久化
# ============================================================


def persist_daily_report(
    *,
    report: dict[str, Any],
    hotspots_path: Path,
    report_path: Path,
) -> dict[str, str]:
    """
    将热点报告写入：

    1. SQLite；
    2. 每个热点主题对应的 RAG JSONL 文档源。
    """
    return save_daily_hotspot_assets(
        project_root=PROJECT_ROOT,
        report=report,
        hotspots_json_path=(
            hotspots_path
        ),
        full_markdown_path=(
            report_path
        ),
    )


# ============================================================
# 主程序
# ============================================================


def main() -> int:
    """
    程序入口。
    """
    arguments = parse_arguments()

    api_key = os.getenv(
        "DEEPSEEK_API_KEY",
        "",
    ).strip()

    if not api_key:
        print(
            "错误：当前 PowerShell "
            "没有设置 DEEPSEEK_API_KEY。",
            file=sys.stderr,
        )

        return 2

    if not api_key.isascii():
        print(
            "错误：DEEPSEEK_API_KEY "
            "包含非 ASCII 字符。",
            file=sys.stderr,
        )

        return 2

    if any(
        character.isspace()
        for character in api_key
    ):
        print(
            "错误：DEEPSEEK_API_KEY "
            "包含空格或换行。",
            file=sys.stderr,
        )

        return 2

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

    latest_pointer_path = (
        PROJECT_ROOT
        / "data"
        / "intelligence"
        / "deep"
        / snapshot_date
        / "latest_collection.json"
    )

    try:
        latest_pointer = read_json(
            latest_pointer_path
        )

        if not isinstance(
            latest_pointer,
            dict,
        ):
            raise ValueError(
                "latest_collection.json "
                "根节点不是对象。"
            )

        summaries_value = (
            latest_pointer.get(
                "repository_llm_summaries_path"
            )
        )

        if not summaries_value:
            raise ValueError(
                "latest_collection.json 中缺少 "
                "repository_llm_summaries_path。"
            )

        summaries_path = resolve_path(
            summaries_value
        )

        summaries_data = read_json(
            summaries_path
        )

        if not isinstance(
            summaries_data,
            list,
        ):
            raise ValueError(
                "repository_llm_summaries.json "
                "根节点不是列表。"
            )

    except Exception as exc:
        print(
            "读取项目摘要失败："
            f"{exc}",
            file=sys.stderr,
        )

        return 1

    successful_summaries = [
        item
        for item in summaries_data
        if (
            isinstance(
                item,
                dict,
            )
            and item.get(
                "status"
            )
            == "success"
        )
    ]

    if not successful_summaries:
        print(
            "没有找到成功的项目摘要。",
            file=sys.stderr,
        )

        return 1

    collection_directory = (
        summaries_path.parent
    )

    hotspots_path = (
        collection_directory
        / "daily_hotspots.json"
    )

    report_path = (
        collection_directory
        / "github_daily_report.md"
    )

    run_summary_path = (
        collection_directory
        / "daily_hotspot_analysis_run.json"
    )

    # ========================================================
    # 已有日报时：
    # 不调用 LLM，只补写 SQLite 和 RAG 文档源。
    # ========================================================

    if (
        hotspots_path.exists()
        and report_path.exists()
        and not arguments.force
    ):
        try:
            existing_report = read_json(
                hotspots_path
            )

            if not isinstance(
                existing_report,
                dict,
            ):
                raise ValueError(
                    "daily_hotspots.json "
                    "根节点不是对象。"
                )

            persistence_paths = (
                persist_daily_report(
                    report=existing_report,
                    hotspots_path=(
                        hotspots_path
                    ),
                    report_path=(
                        report_path
                    ),
                )
            )

            persisted_at = datetime.now(
                ZoneInfo(
                    DEFAULT_TIMEZONE
                )
            ).isoformat()

            latest_pointer[
                "daily_hotspots_path"
            ] = str(
                hotspots_path
            )

            latest_pointer[
                "github_daily_report_path"
            ] = str(
                report_path
            )

            latest_pointer[
                "intelligence_database_path"
            ] = persistence_paths[
                "database_path"
            ]

            latest_pointer[
                "daily_hotspot_rag_documents_path"
            ] = persistence_paths[
                "rag_documents_path"
            ]

            latest_pointer[
                "daily_hotspots_persisted_at"
            ] = persisted_at

            write_json(
                latest_pointer_path,
                latest_pointer,
            )

            print(
                "=" * 78
            )

            print(
                "已有 GitHub 热点日报持久化完成"
            )

            print(
                "=" * 78
            )

            print(
                f"日期：{snapshot_date}"
            )

            print(
                "未重新调用 LLM。"
            )

            print(
                "热点主题数："
                f"{len(existing_report.get('hotspot_topics', []))}"
            )

            print(
                f"热点 JSON：{hotspots_path}"
            )

            print(
                f"Markdown 日报：{report_path}"
            )

            print(
                "SQLite："
                f"{persistence_paths['database_path']}"
            )

            print(
                "RAG 文档源："
                f"{persistence_paths['rag_documents_path']}"
            )

            return 0

        except Exception as exc:
            print(
                "已有热点日报持久化失败："
                f"{exc}",
                file=sys.stderr,
            )

            return 1

    # ========================================================
    # 不存在日报，或者使用 --force 时重新生成。
    # ========================================================

    user_prompt = build_user_prompt(
        snapshot_date,
        successful_summaries,
    )

    print(
        "=" * 78
    )

    print(
        "GitHub 跨项目热点分析"
    )

    print(
        "=" * 78
    )

    print(
        f"日期：{snapshot_date}"
    )

    print(
        "项目摘要数："
        f"{len(successful_summaries)}"
    )

    print(
        f"模型：{arguments.model}"
    )

    print(
        f"输入字符数：{len(user_prompt):,}"
    )

    print(
        "正在进行跨项目归纳……"
    )

    started_at = datetime.now(
        ZoneInfo(
            DEFAULT_TIMEZONE
        )
    ).isoformat()

    client = DeepSeekClient(
        api_key=api_key,
        model=arguments.model,
    )

    try:
        (
            raw_result,
            api_metadata,
        ) = client.summarize(
            user_prompt
        )

    except Exception as exc:
        print(
            f"热点分析失败：{exc}",
            file=sys.stderr,
        )

        return 1

    finally:
        client.close()

    normalized_result = (
        normalize_daily_result(
            raw_result,
            snapshot_date=(
                snapshot_date
            ),
            project_count=len(
                successful_summaries
            ),
        )
    )

    generated_at = datetime.now(
        ZoneInfo(
            DEFAULT_TIMEZONE
        )
    ).isoformat()

    normalized_result[
        "generated_at"
    ] = generated_at

    normalized_result[
        "source_summaries_path"
    ] = str(
        summaries_path
    )

    normalized_result[
        "source_projects"
    ] = [
        str(
            summary.get(
                "full_name"
            )
        )
        for summary
        in successful_summaries
        if summary.get(
            "full_name"
        )
    ]

    normalized_result[
        "api"
    ] = api_metadata

    markdown_report = render_markdown(
        normalized_result
    )

    write_json(
        hotspots_path,
        normalized_result,
    )

    write_text(
        report_path,
        markdown_report,
    )

    persistence_status = "success"
    persistence_error: str | None = None
    persistence_paths: dict[
        str,
        str,
    ] = {}

    try:
        persistence_paths = (
            persist_daily_report(
                report=normalized_result,
                hotspots_path=(
                    hotspots_path
                ),
                report_path=(
                    report_path
                ),
            )
        )

    except Exception as exc:
        persistence_status = "failed"
        persistence_error = str(
            exc
        )

    run_summary = {
        "status": (
            "success"
            if persistence_status
            == "success"
            else "partial_success"
        ),

        "snapshot_date": (
            snapshot_date
        ),

        "started_at": (
            started_at
        ),

        "finished_at": (
            generated_at
        ),

        "model": (
            arguments.model
        ),

        "source_project_count": len(
            successful_summaries
        ),

        "prompt_characters": len(
            user_prompt
        ),

        "usage": (
            api_metadata.get(
                "usage",
                {},
            )
        ),

        "source_summaries_path": str(
            summaries_path
        ),

        "hotspots_path": str(
            hotspots_path
        ),

        "report_path": str(
            report_path
        ),

        "persistence_status": (
            persistence_status
        ),

        "persistence_error": (
            persistence_error
        ),

        "database_path": (
            persistence_paths.get(
                "database_path"
            )
        ),

        "rag_documents_path": (
            persistence_paths.get(
                "rag_documents_path"
            )
        ),
    }

    write_json(
        run_summary_path,
        run_summary,
    )

    latest_pointer[
        "daily_hotspots_path"
    ] = str(
        hotspots_path
    )

    latest_pointer[
        "github_daily_report_path"
    ] = str(
        report_path
    )

    latest_pointer[
        "daily_hotspot_analysis_run_path"
    ] = str(
        run_summary_path
    )

    latest_pointer[
        "daily_hotspots_generated_at"
    ] = generated_at

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
            "daily_hotspot_rag_documents_path"
        ] = persistence_paths[
            "rag_documents_path"
        ]

    write_json(
        latest_pointer_path,
        latest_pointer,
    )

    print()

    print(
        "=" * 78
    )

    print(
        "跨项目热点分析与持久化完成"
    )

    print(
        "=" * 78
    )

    print(
        "热点主题数："
        f"{len(normalized_result.get('hotspot_topics', []))}"
    )

    print(
        "重点项目数："
        f"{len(normalized_result.get('notable_projects', []))}"
    )

    print(
        "GitHub 动态检索词："
        f"{len(normalized_result.get('github_search_queries', []))}"
    )

    print(
        "ArXiv 动态检索词："
        f"{len(normalized_result.get('arxiv_search_queries', []))}"
    )

    print(
        "Token 使用："
        f"{api_metadata.get('usage', {})}"
    )

    print(
        f"持久化状态：{persistence_status}"
    )

    print(
        f"热点 JSON：{hotspots_path}"
    )

    print(
        f"Markdown 日报：{report_path}"
    )

    print(
        f"运行摘要：{run_summary_path}"
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