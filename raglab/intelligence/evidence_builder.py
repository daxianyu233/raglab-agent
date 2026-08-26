from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


# README 中优先保留的章节关键词。
#
# 这些章节通常能够说明：
# - 项目是什么；
# - 项目解决什么问题；
# - 项目提供哪些能力；
# - 项目采用什么架构；
# - 项目适合哪些使用场景。
POSITIVE_HEADING_KEYWORDS: dict[str, float] = {
    "overview": 12.0,
    "introduction": 12.0,
    "about": 11.0,
    "what is": 11.0,
    "why": 10.0,
    "motivation": 10.0,
    "background": 7.0,
    "features": 12.0,
    "feature": 11.0,
    "capabilities": 12.0,
    "capability": 11.0,
    "highlights": 11.0,
    "key concepts": 10.0,
    "use cases": 11.0,
    "use case": 10.0,
    "applications": 10.0,
    "application": 8.0,
    "examples": 7.0,
    "architecture": 12.0,
    "design": 9.0,
    "how it works": 12.0,
    "workflow": 10.0,
    "components": 9.0,
    "system": 7.0,
    "framework": 8.0,
    "agent": 8.0,
    "agents": 8.0,
    "memory": 8.0,
    "rag": 8.0,
    "retrieval": 8.0,
    "mcp": 8.0,
    "model context protocol": 9.0,
    "tool": 6.0,
    "tools": 6.0,
    "integration": 8.0,
    "integrations": 8.0,
    "benchmark": 9.0,
    "benchmarks": 9.0,
    "performance": 8.0,
    "evaluation": 8.0,
    "comparison": 7.0,
    "roadmap": 7.0,
    "limitations": 7.0,
    "security": 6.0,
}


# README 中通常不适合进入热点分析材料的章节。
NEGATIVE_HEADING_KEYWORDS: dict[str, float] = {
    "installation": 10.0,
    "install": 9.0,
    "setup": 8.0,
    "quick start": 5.0,
    "quickstart": 5.0,
    "getting started": 4.0,
    "usage": 2.0,
    "configuration": 5.0,
    "requirements": 8.0,
    "prerequisites": 8.0,
    "contributing": 15.0,
    "contribution": 15.0,
    "contributors": 15.0,
    "license": 20.0,
    "citation": 15.0,
    "acknowledgement": 15.0,
    "acknowledgements": 15.0,
    "sponsor": 15.0,
    "sponsors": 15.0,
    "support": 7.0,
    "community": 6.0,
    "contact": 10.0,
    "faq": 5.0,
    "table of contents": 20.0,
    "contents": 10.0,
}


# 规则关键词提取使用的英文停用词。
STOPWORDS: set[str] = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "among",
    "another",
    "because",
    "before",
    "being",
    "below",
    "between",
    "both",
    "build",
    "building",
    "built",
    "can",
    "could",
    "data",
    "does",
    "doing",
    "each",
    "easy",
    "example",
    "examples",
    "first",
    "from",
    "github",
    "have",
    "having",
    "help",
    "here",
    "into",
    "just",
    "latest",
    "like",
    "more",
    "most",
    "new",
    "only",
    "open",
    "other",
    "our",
    "over",
    "project",
    "projects",
    "repository",
    "same",
    "simple",
    "source",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "using",
    "very",
    "want",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
    "would",
    "your",
}


# 对当前 AI Agent、RAG 和应用工程方向有较高意义的固定短语。
#
# 这里只做规则检测，不直接作为最终动态搜索词。
IMPORTANT_PHRASES: tuple[str, ...] = (
    "ai agent",
    "ai agents",
    "autonomous agent",
    "browser agent",
    "coding agent",
    "voice agent",
    "multi agent",
    "multi-agent",
    "agent memory",
    "long term memory",
    "long-term memory",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "model context protocol",
    "tool calling",
    "function calling",
    "context engineering",
    "prompt engineering",
    "knowledge graph",
    "vector database",
    "semantic search",
    "deep research",
    "workflow automation",
    "human in the loop",
    "human-in-the-loop",
    "large language model",
    "multimodal model",
    "reasoning model",
    "local llm",
    "voice cloning",
    "text to speech",
    "speech to speech",
)


def _load_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    读取 YAML 配置文件。
    """
    config_path = Path(
        config_path
    )

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

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "配置文件根节点必须是 YAML 对象。"
        )

    return config


def _read_json(
    path: Path,
) -> Any:
    """
    读取 UTF-8 JSON 文件。
    """
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"JSON 文件不存在：{path}"
        )

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
        return json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "JSON 文件格式错误："
            f"{path}\n{exc}"
        ) from exc


def _write_json(
    path: Path,
    data: Any,
) -> None:
    """
    将数据保存为 UTF-8 JSON。
    """
    path = Path(
        path
    )

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
        return int(
            value
        )
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
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _normalize_text(
    value: Any,
) -> str:
    """
    统一文本换行和行尾空格。
    """
    if value is None:
        return ""

    text = str(
        value
    ).replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    text = "\n".join(
        lines
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def _compact_single_line(
    value: Any,
) -> str:
    """
    将文本压缩成一行。
    """
    if value is None:
        return ""

    return " ".join(
        str(value).split()
    ).strip()


def _truncate_text(
    value: Any,
    maximum_characters: int,
) -> str:
    """
    从开头截取指定字符数。
    """
    text = _normalize_text(
        value
    )

    maximum_characters = max(
        0,
        int(
            maximum_characters
        ),
    )

    if len(text) <= maximum_characters:
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


def _resolve_path(
    project_root: Path,
    path_value: Any,
) -> Path:
    """
    将配置或指针中的路径转换成绝对路径。
    """
    path = Path(
        str(
            path_value
        )
    )

    if path.is_absolute():
        return path.resolve()

    return (
        project_root
        / path
    ).resolve()


def _resolve_snapshot_date(
    snapshot_date: str | None,
    local_timezone: ZoneInfo,
) -> date:
    """
    确定需要构建证据包的日期。
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
            "日期格式错误，必须使用 YYYY-MM-DD："
            f"{snapshot_date}"
        ) from exc


def _split_markdown_sections(
    markdown_text: Any,
) -> list[dict[str, Any]]:
    """
    按 Markdown 标题拆分 README。

    返回结构：

    {
        "heading": "Features",
        "level": 2,
        "content": "...",
        "order": 3
    }
    """
    text = _normalize_text(
        markdown_text
    )

    if not text:
        return []

    heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$"
    )

    sections: list[
        dict[str, Any]
    ] = []

    current_heading = "README 概览"
    current_level = 0
    current_lines: list[str] = []
    section_order = 0

    def append_current_section() -> None:
        """
        将当前累计内容加入章节列表。
        """
        nonlocal section_order

        content = _normalize_text(
            "\n".join(
                current_lines
            )
        )

        if not content:
            return

        sections.append(
            {
                "heading": (
                    current_heading
                ),
                "level": current_level,
                "content": content,
                "order": section_order,
            }
        )

        section_order += 1

    for line in text.splitlines():
        heading_match = heading_pattern.match(
            line
        )

        if heading_match:
            append_current_section()

            current_heading = _compact_single_line(
                heading_match.group(
                    2
                )
            )

            current_level = len(
                heading_match.group(
                    1
                )
            )

            current_lines = []

            continue

        current_lines.append(
            line
        )

    append_current_section()

    return sections


def _heading_score(
    heading: Any,
    section_order: int,
) -> float:
    """
    计算 README 章节的重要程度。

    项目开头内容具有天然优先级。
    """
    normalized_heading = _compact_single_line(
        heading
    ).lower()

    score = 0.0

    # README 开头的项目简介通常价值较高。
    if section_order == 0:
        score += 15.0

    if section_order <= 2:
        score += 5.0

    for keyword, weight in (
        POSITIVE_HEADING_KEYWORDS.items()
    ):
        if keyword in normalized_heading:
            score += weight

    for keyword, weight in (
        NEGATIVE_HEADING_KEYWORDS.items()
    ):
        if keyword in normalized_heading:
            score -= weight

    return round(
        score,
        4,
    )


def _select_readme_sections(
    readme: dict[str, Any] | None,
    *,
    maximum_sections: int,
    maximum_section_characters: int,
) -> list[dict[str, Any]]:
    """
    从 README 中选择适合热点分析的主要章节。
    """
    if not isinstance(
        readme,
        dict,
    ):
        return []

    readme_content = readme.get(
        "content"
    )

    sections = _split_markdown_sections(
        readme_content
    )

    if not sections:
        return []

    scored_sections: list[
        dict[str, Any]
    ] = []

    for section in sections:
        content = _normalize_text(
            section.get(
                "content"
            )
        )

        if len(content) < 30:
            continue

        heading = str(
            section.get(
                "heading"
            )
            or "未命名章节"
        )

        order = _safe_int(
            section.get(
                "order"
            )
        )

        score = _heading_score(
            heading,
            order,
        )

        scored_sections.append(
            {
                "heading": heading,
                "level": _safe_int(
                    section.get(
                        "level"
                    )
                ),
                "order": order,
                "score": score,
                "content": _truncate_text(
                    content,
                    maximum_section_characters,
                ),
            }
        )

    # 优先按照价值分数筛选。
    scored_sections.sort(
        key=lambda item: (
            float(
                item.get(
                    "score"
                )
                or 0
            ),
            -int(
                item.get(
                    "order"
                )
                or 0
            ),
        ),
        reverse=True,
    )

    selected_sections = scored_sections[
        : max(
            0,
            maximum_sections,
        )
    ]

    # 最终按照 README 中的原始顺序展示。
    selected_sections.sort(
        key=lambda item: int(
            item.get(
                "order"
            )
            or 0
        )
    )

    return selected_sections


def _prepare_release_signals(
    releases: Any,
    *,
    maximum_items: int,
    maximum_body_characters: int,
) -> list[dict[str, Any]]:
    """
    选择近期 Release 信号。
    """
    if not isinstance(
        releases,
        list,
    ):
        return []

    prepared: list[
        dict[str, Any]
    ] = []

    for release in releases[
        : max(
            0,
            maximum_items,
        )
    ]:
        if not isinstance(
            release,
            dict,
        ):
            continue

        release_name = (
            release.get(
                "name"
            )
            or release.get(
                "tag_name"
            )
            or "未命名版本"
        )

        prepared.append(
            {
                "name": _compact_single_line(
                    release_name
                ),
                "tag_name": (
                    release.get(
                        "tag_name"
                    )
                ),
                "published_at": (
                    release.get(
                        "published_at"
                    )
                ),
                "prerelease": bool(
                    release.get(
                        "prerelease"
                    )
                ),
                "body": _truncate_text(
                    release.get(
                        "body"
                    ),
                    maximum_body_characters,
                ),
                "html_url": release.get(
                    "html_url"
                ),
            }
        )

    return prepared


def _issue_priority_score(
    issue: dict[str, Any],
) -> float:
    """
    根据评论、回应和更新时间计算 Issue 优先级。
    """
    comments = max(
        0,
        _safe_int(
            issue.get(
                "comments"
            )
        ),
    )

    reactions = max(
        0,
        _safe_int(
            issue.get(
                "reactions"
            )
        ),
    )

    score = (
        comments * 3.0
        + reactions * 2.0
    )

    if issue.get(
        "state"
    ) == "open":
        score += 3.0

    if issue.get(
        "updated_at"
    ):
        score += 1.0

    return round(
        score,
        4,
    )


def _prepare_issue_signals(
    issues: Any,
    *,
    maximum_items: int,
    maximum_body_characters: int,
) -> list[dict[str, Any]]:
    """
    从 Issue 中选择高关注的问题信号。
    """
    if not isinstance(
        issues,
        list,
    ):
        return []

    valid_issues = [
        issue
        for issue in issues
        if isinstance(
            issue,
            dict,
        )
    ]

    valid_issues.sort(
        key=lambda issue: (
            _issue_priority_score(
                issue
            ),
            str(
                issue.get(
                    "updated_at"
                )
                or ""
            ),
        ),
        reverse=True,
    )

    prepared: list[
        dict[str, Any]
    ] = []

    for issue in valid_issues[
        : max(
            0,
            maximum_items,
        )
    ]:
        labels = issue.get(
            "labels"
        )

        if not isinstance(
            labels,
            list,
        ):
            labels = []

        prepared.append(
            {
                "number": issue.get(
                    "number"
                ),
                "title": _compact_single_line(
                    issue.get(
                        "title"
                    )
                ),
                "state": issue.get(
                    "state"
                ),
                "labels": [
                    str(label)
                    for label in labels
                    if str(label).strip()
                ],
                "comments": _safe_int(
                    issue.get(
                        "comments"
                    )
                ),
                "reactions": _safe_int(
                    issue.get(
                        "reactions"
                    )
                ),
                "updated_at": issue.get(
                    "updated_at"
                ),
                "priority_score": (
                    _issue_priority_score(
                        issue
                    )
                ),
                "body": _truncate_text(
                    issue.get(
                        "body"
                    ),
                    maximum_body_characters,
                ),
                "html_url": issue.get(
                    "html_url"
                ),
            }
        )

    return prepared


def _extract_candidate_keywords(
    *,
    repository: dict[str, Any],
    readme_sections: list[dict[str, Any]],
    releases: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    maximum_keywords: int,
) -> list[str]:
    """
    使用确定性规则提取候选关键词。

    优先级：

    1. GitHub Topics；
    2. 预定义的重要技术短语；
    3. README 标题；
    4. 描述、Release 和 Issue 中的高频词。
    """
    keyword_scores: Counter[str] = Counter()

    topics = repository.get(
        "topics"
    )

    if isinstance(
        topics,
        list,
    ):
        for topic in topics:
            normalized_topic = str(
                topic
            ).strip().lower()

            if normalized_topic:
                keyword_scores[
                    normalized_topic
                ] += 20

    text_parts: list[str] = [
        str(
            repository.get(
                "description"
            )
            or ""
        )
    ]

    for section in readme_sections:
        heading = str(
            section.get(
                "heading"
            )
            or ""
        )

        content = str(
            section.get(
                "content"
            )
            or ""
        )

        text_parts.append(
            heading
        )

        text_parts.append(
            content
        )

        normalized_heading = (
            heading.strip().lower()
        )

        if (
            normalized_heading
            and len(
                normalized_heading
            ) <= 60
        ):
            keyword_scores[
                normalized_heading
            ] += 4

    for release in releases:
        text_parts.append(
            str(
                release.get(
                    "name"
                )
                or ""
            )
        )

        text_parts.append(
            str(
                release.get(
                    "body"
                )
                or ""
            )
        )

    for issue in issues:
        text_parts.append(
            str(
                issue.get(
                    "title"
                )
                or ""
            )
        )

        text_parts.append(
            str(
                issue.get(
                    "body"
                )
                or ""
            )
        )

        labels = issue.get(
            "labels"
        )

        if isinstance(
            labels,
            list,
        ):
            for label in labels:
                normalized_label = str(
                    label
                ).strip().lower()

                if normalized_label:
                    keyword_scores[
                        normalized_label
                    ] += 3

    combined_text = " ".join(
        text_parts
    ).lower()

    # 先识别具有明确语义的多词短语。
    for phrase in IMPORTANT_PHRASES:
        occurrence_count = (
            combined_text.count(
                phrase
            )
        )

        if occurrence_count > 0:
            keyword_scores[
                phrase
            ] += (
                8
                + occurrence_count
            )

    # 再提取普通英文技术词。
    tokens = re.findall(
        r"\b[a-z][a-z0-9_.+\-]{2,}\b",
        combined_text,
    )

    token_counter = Counter(
        token
        for token in tokens
        if (
            token not in STOPWORDS
            and not token.isdigit()
            and len(token) <= 40
        )
    )

    for token, count in (
        token_counter.items()
    ):
        keyword_scores[
            token
        ] += min(
            count,
            8,
        )

    sorted_keywords = sorted(
        keyword_scores.items(),
        key=lambda item: (
            item[1],
            len(
                item[0]
            ),
            item[0],
        ),
        reverse=True,
    )

    selected_keywords: list[str] = []
    seen_keywords: set[str] = set()

    for keyword, _ in sorted_keywords:
        normalized_keyword = (
            keyword.strip().lower()
        )

        if not normalized_keyword:
            continue

        if normalized_keyword in seen_keywords:
            continue

        # 过滤过于宽泛的章节名称。
        if normalized_keyword in {
            "readme 概览",
            "overview",
            "introduction",
            "features",
            "usage",
            "examples",
            "architecture",
        }:
            continue

        selected_keywords.append(
            normalized_keyword
        )

        seen_keywords.add(
            normalized_keyword
        )

        if (
            len(
                selected_keywords
            )
            >= maximum_keywords
        ):
            break

    return selected_keywords


def _render_evidence_text(
    *,
    repository: dict[str, Any],
    selection: dict[str, Any],
    readme_sections: list[dict[str, Any]],
    releases: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    keywords: list[str],
    maximum_characters: int,
) -> str:
    """
    将结构化证据渲染成供 LLM 使用的紧凑文本。
    """
    lines: list[str] = []

    full_name = str(
        repository.get(
            "full_name"
        )
        or "unknown/unknown"
    )

    lines.append(
        f"项目：{full_name}"
    )

    description = _compact_single_line(
        repository.get(
            "description"
        )
    )

    if description:
        lines.append(
            f"项目描述：{description}"
        )

    language = repository.get(
        "language"
    )

    stars = _safe_int(
        repository.get(
            "stars"
        )
    )

    period_stars = repository.get(
        "period_stars"
    )

    metadata_parts = [
        f"主要语言={language or '未知'}",
        f"总 Star={stars:,}",
    ]

    if period_stars is not None:
        metadata_parts.append(
            "周期新增 Star="
            f"{_safe_int(period_stars):,}"
        )

    selection_group = selection.get(
        "group"
    )

    if selection_group:
        metadata_parts.append(
            "入选分组="
            f"{selection_group}"
        )

    lines.append(
        "基础信息："
        + "；".join(
            metadata_parts
        )
    )

    topics = repository.get(
        "topics"
    )

    if isinstance(
        topics,
        list,
    ) and topics:
        lines.append(
            "Topics："
            + "、".join(
                str(topic)
                for topic in topics
            )
        )

    if readme_sections:
        lines.append("")
        lines.append(
            "README 核心内容："
        )

        for section in readme_sections:
            heading = _compact_single_line(
                section.get(
                    "heading"
                )
            )

            content = _normalize_text(
                section.get(
                    "content"
                )
            )

            lines.append(
                f"[{heading}]"
            )

            lines.append(
                content
            )

    if releases:
        lines.append("")
        lines.append(
            "近期版本变化："
        )

        for release in releases:
            release_name = (
                release.get(
                    "name"
                )
                or release.get(
                    "tag_name"
                )
                or "未命名版本"
            )

            published_at = (
                release.get(
                    "published_at"
                )
                or "未知日期"
            )

            lines.append(
                "- "
                f"{release_name}"
                f"（{published_at}）"
            )

            release_body = _normalize_text(
                release.get(
                    "body"
                )
            )

            if release_body:
                lines.append(
                    release_body
                )

    if issues:
        lines.append("")
        lines.append(
            "高关注 Issue 信号："
        )

        for issue in issues:
            issue_number = issue.get(
                "number"
            )

            issue_title = _compact_single_line(
                issue.get(
                    "title"
                )
            )

            comments = _safe_int(
                issue.get(
                    "comments"
                )
            )

            reactions = _safe_int(
                issue.get(
                    "reactions"
                )
            )

            lines.append(
                "- "
                f"#{issue_number} "
                f"{issue_title}"
                f"（评论 {comments}，"
                f"回应 {reactions}）"
            )

            issue_body = _normalize_text(
                issue.get(
                    "body"
                )
            )

            if issue_body:
                lines.append(
                    issue_body
                )

    if keywords:
        lines.append("")
        lines.append(
            "规则提取候选关键词："
            + "、".join(
                keywords
            )
        )

    evidence_text = _normalize_text(
        "\n".join(
            lines
        )
    )

    return _truncate_text(
        evidence_text,
        maximum_characters,
    )


def _build_repository_evidence_pack(
    *,
    material: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    为单个仓库构建规则证据包。
    """
    repository = material.get(
        "repository"
    )

    if not isinstance(
        repository,
        dict,
    ):
        repository = {}

    selection = material.get(
        "selection"
    )

    if not isinstance(
        selection,
        dict,
    ):
        selection = {}

    readme_config = config.get(
        "readme"
    )

    if not isinstance(
        readme_config,
        dict,
    ):
        readme_config = {}

    release_config = config.get(
        "releases"
    )

    if not isinstance(
        release_config,
        dict,
    ):
        release_config = {}

    issue_config = config.get(
        "issues"
    )

    if not isinstance(
        issue_config,
        dict,
    ):
        issue_config = {}

    keyword_config = config.get(
        "keywords"
    )

    if not isinstance(
        keyword_config,
        dict,
    ):
        keyword_config = {}

    maximum_evidence_characters = max(
        1000,
        _safe_int(
            config.get(
                "maximum_evidence_characters"
            ),
            4000,
        ),
    )

    readme_sections = _select_readme_sections(
        material.get(
            "readme"
        ),
        maximum_sections=max(
            0,
            _safe_int(
                readme_config.get(
                    "maximum_sections"
                ),
                5,
            ),
        ),
        maximum_section_characters=max(
            100,
            _safe_int(
                readme_config.get(
                    "maximum_section_characters"
                ),
                650,
            ),
        ),
    )

    releases = _prepare_release_signals(
        material.get(
            "releases"
        ),
        maximum_items=max(
            0,
            _safe_int(
                release_config.get(
                    "maximum_items"
                ),
                2,
            ),
        ),
        maximum_body_characters=max(
            0,
            _safe_int(
                release_config.get(
                    "maximum_body_characters"
                ),
                450,
            ),
        ),
    )

    issues = _prepare_issue_signals(
        material.get(
            "issues"
        ),
        maximum_items=max(
            0,
            _safe_int(
                issue_config.get(
                    "maximum_items"
                ),
                4,
            ),
        ),
        maximum_body_characters=max(
            0,
            _safe_int(
                issue_config.get(
                    "maximum_body_characters"
                ),
                300,
            ),
        ),
    )

    keywords = _extract_candidate_keywords(
        repository=repository,
        readme_sections=readme_sections,
        releases=releases,
        issues=issues,
        maximum_keywords=max(
            1,
            _safe_int(
                keyword_config.get(
                    "maximum_items"
                ),
                15,
            ),
        ),
    )

    evidence_text = _render_evidence_text(
        repository=repository,
        selection=selection,
        readme_sections=readme_sections,
        releases=releases,
        issues=issues,
        keywords=keywords,
        maximum_characters=(
            maximum_evidence_characters
        ),
    )

    return {
        "collection_id": material.get(
            "collection_id"
        ),
        "snapshot_date": material.get(
            "snapshot_date"
        ),
        "built_at": None,
        "repository": {
            "full_name": repository.get(
                "full_name"
            ),
            "html_url": repository.get(
                "html_url"
            ),
            "description": repository.get(
                "description"
            ),
            "language": repository.get(
                "language"
            ),
            "topics": repository.get(
                "topics"
            )
            if isinstance(
                repository.get(
                    "topics"
                ),
                list,
            )
            else [],
            "stars": _safe_int(
                repository.get(
                    "stars"
                )
            ),
            "forks": _safe_int(
                repository.get(
                    "forks"
                )
            ),
            "period_stars": (
                repository.get(
                    "period_stars"
                )
            ),
            "trending_rank": (
                repository.get(
                    "trending_rank"
                )
            ),
            "created_at": repository.get(
                "created_at"
            ),
            "pushed_at": repository.get(
                "pushed_at"
            ),
            "updated_at": repository.get(
                "updated_at"
            ),
            "search_queries": (
                repository.get(
                    "search_queries"
                )
                if isinstance(
                    repository.get(
                        "search_queries"
                    ),
                    list,
                )
                else []
            ),
        },
        "selection": selection,
        "evidence": {
            "readme_sections": (
                readme_sections
            ),
            "releases": releases,
            "issues": issues,
            "candidate_keywords": (
                keywords
            ),
        },
        "evidence_text": evidence_text,
        "evidence_characters": len(
            evidence_text
        ),
        "source_analysis_characters": (
            _safe_int(
                material.get(
                    "analysis_text_characters"
                )
            )
        ),
        "collection_errors": (
            material.get(
                "collection_errors"
            )
            if isinstance(
                material.get(
                    "collection_errors"
                ),
                list,
            )
            else []
        ),
    }


def _safe_repository_filename(
    full_name: Any,
) -> str:
    """
    将仓库名称转换成 Windows 安全文件名。
    """
    filename = str(
        full_name
        or "unknown_repository"
    ).replace(
        "/",
        "__",
    )

    filename = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        filename,
    )

    filename = filename.strip(
        ". "
    )

    return (
        filename
        or "unknown_repository"
    )


def build_repository_evidence_packs(
    *,
    project_root: Path,
    config_path: Path,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    """
    从最近一次深度采集结果中构建仓库证据包。

    本函数不调用任何 LLM。
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

    target_date = _resolve_snapshot_date(
        snapshot_date,
        local_timezone,
    )

    snapshot_date_text = (
        target_date.isoformat()
    )

    paths_config = config.get(
        "paths"
    )

    if not isinstance(
        paths_config,
        dict,
    ):
        paths_config = {}

    deep_raw_root = project_root / str(
        paths_config.get(
            "deep_raw_root"
        )
        or "data/intelligence/deep"
    )

    latest_pointer_path = (
        deep_raw_root
        / snapshot_date_text
        / "latest_collection.json"
    )

    latest_pointer = _read_json(
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

    analysis_material_value = (
        latest_pointer.get(
            "analysis_material_path"
        )
    )

    if not analysis_material_value:
        raise ValueError(
            "latest_collection.json 中缺少 "
            "analysis_material_path。"
        )

    analysis_material_path = _resolve_path(
        project_root,
        analysis_material_value,
    )

    analysis_materials = _read_json(
        analysis_material_path
    )

    if not isinstance(
        analysis_materials,
        list,
    ):
        raise ValueError(
            "github_repository_analysis_material.json "
            "根节点必须是列表。"
        )

    evidence_config = config.get(
        "evidence_builder"
    )

    if not isinstance(
        evidence_config,
        dict,
    ):
        evidence_config = {}

    enabled = bool(
        evidence_config.get(
            "enabled",
            True,
        )
    )

    if not enabled:
        return {
            "status": "disabled",
            "snapshot_date": (
                snapshot_date_text
            ),
            "processed_count": 0,
            "errors": [
                "配置中的 evidence_builder.enabled "
                "为 false。"
            ],
        }

    maximum_repositories = max(
        0,
        _safe_int(
            evidence_config.get(
                "maximum_repositories"
            ),
            15,
        ),
    )

    if maximum_repositories > 0:
        analysis_materials = (
            analysis_materials[
                :maximum_repositories
            ]
        )
    else:
        analysis_materials = []

    if not analysis_materials:
        raise RuntimeError(
            "深度采集结果中没有可供构建证据包的仓库。"
        )

    collection_directory_value = (
        latest_pointer.get(
            "collection_directory"
        )
    )

    if not collection_directory_value:
        collection_directory = (
            analysis_material_path.parent
        )
    else:
        collection_directory = _resolve_path(
            project_root,
            collection_directory_value,
        )

    evidence_directory = (
        collection_directory
        / "evidence"
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    built_at = datetime.now(
        local_timezone
    ).isoformat()

    evidence_packs: list[
        dict[str, Any]
    ] = []

    errors: list[str] = []

    for material in analysis_materials:
        if not isinstance(
            material,
            dict,
        ):
            continue

        try:
            evidence_pack = (
                _build_repository_evidence_pack(
                    material=material,
                    config=evidence_config,
                )
            )

            evidence_pack[
                "built_at"
            ] = built_at

            repository = evidence_pack.get(
                "repository"
            )

            if not isinstance(
                repository,
                dict,
            ):
                repository = {}

            full_name = str(
                repository.get(
                    "full_name"
                )
                or "unknown/unknown"
            )

            evidence_file_path = (
                evidence_directory
                / (
                    _safe_repository_filename(
                        full_name
                    )
                    + ".json"
                )
            )

            _write_json(
                evidence_file_path,
                evidence_pack,
            )

            evidence_pack[
                "evidence_file"
            ] = str(
                evidence_file_path.relative_to(
                    collection_directory
                )
            )

            evidence_packs.append(
                evidence_pack
            )

        except Exception as exc:
            full_name = "unknown/unknown"

            repository_value = material.get(
                "repository"
            )

            if isinstance(
                repository_value,
                dict,
            ):
                full_name = str(
                    repository_value.get(
                        "full_name"
                    )
                    or full_name
                )

            errors.append(
                "证据包构建失败，"
                f"仓库：{full_name}，"
                f"原因：{exc}"
            )

    aggregate_path = (
        collection_directory
        / "repository_evidence_packs.json"
    )

    summary_path = (
        collection_directory
        / "evidence_build_summary.json"
    )

    _write_json(
        aggregate_path,
        evidence_packs,
    )

    source_characters = sum(
        _safe_int(
            pack.get(
                "source_analysis_characters"
            )
        )
        for pack in evidence_packs
    )

    evidence_characters = sum(
        _safe_int(
            pack.get(
                "evidence_characters"
            )
        )
        for pack in evidence_packs
    )

    if evidence_packs and not errors:
        status = "success"
    elif evidence_packs:
        status = "partial_success"
    else:
        status = "failed"

    if source_characters > 0:
        compression_rate = (
            evidence_characters
            / source_characters
        )
    else:
        compression_rate = 0.0

    summary = {
        "status": status,
        "snapshot_date": (
            snapshot_date_text
        ),
        "built_at": built_at,
        "source_analysis_material_path": str(
            analysis_material_path
        ),
        "collection_directory": str(
            collection_directory
        ),
        "processed_count": len(
            evidence_packs
        ),
        "source_characters": (
            source_characters
        ),
        "evidence_characters": (
            evidence_characters
        ),
        "compression_rate": round(
            compression_rate,
            6,
        ),
        "average_evidence_characters": (
            round(
                evidence_characters
                / len(
                    evidence_packs
                ),
                2,
            )
            if evidence_packs
            else 0
        ),
        "aggregate_path": str(
            aggregate_path
        ),
        "evidence_directory": str(
            evidence_directory
        ),
        "errors": errors,
    }

    _write_json(
        summary_path,
        summary,
    )

    # 更新当天指针，但不覆盖深度采集原有字段。
    latest_pointer[
        "evidence_status"
    ] = status

    latest_pointer[
        "evidence_built_at"
    ] = built_at

    latest_pointer[
        "evidence_packs_path"
    ] = str(
        aggregate_path
    )

    latest_pointer[
        "evidence_summary_path"
    ] = str(
        summary_path
    )

    _write_json(
        latest_pointer_path,
        latest_pointer,
    )

    return {
        **summary,
        "summary_path": str(
            summary_path
        ),
        "latest_pointer_path": str(
            latest_pointer_path
        ),
        "repositories": [
            {
                "full_name": (
                    pack.get(
                        "repository",
                        {},
                    ).get(
                        "full_name"
                    )
                    if isinstance(
                        pack.get(
                            "repository"
                        ),
                        dict,
                    )
                    else None
                ),
                "evidence_characters": (
                    pack.get(
                        "evidence_characters"
                    )
                ),
                "source_analysis_characters": (
                    pack.get(
                        "source_analysis_characters"
                    )
                ),
                "candidate_keywords": (
                    pack.get(
                        "evidence",
                        {},
                    ).get(
                        "candidate_keywords"
                    )
                    if isinstance(
                        pack.get(
                            "evidence"
                        ),
                        dict,
                    )
                    else []
                ),
                "evidence_file": (
                    pack.get(
                        "evidence_file"
                    )
                ),
            }
            for pack in evidence_packs
        ],
    }