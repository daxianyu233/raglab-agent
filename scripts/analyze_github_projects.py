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
    save_repository_summary_assets,
)


# ============================================================
# DeepSeek 和输入长度配置
# ============================================================

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

README_MAX_CHARACTERS = 10_000

RELEASE_MAX_ITEMS = 2
RELEASE_BODY_MAX_CHARACTERS = 1_200

ISSUE_MAX_ITEMS = 3
ISSUE_BODY_MAX_CHARACTERS = 800

MAX_OUTPUT_TOKENS = 1_800

REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3


SYSTEM_PROMPT = """
你是一名 GitHub 技术情报分析员。

你会收到一个 GitHub 项目的结构化元数据、README、
近期 Release 和 Issue 证据。

任务要求：

1. 只能依据输入证据判断，不得补充材料没有支持的事实。

2. 识别项目真实类型，区分：
   - 技术框架；
   - SDK；
   - 应用；
   - 工具；
   - 库；
   - 教程；
   - 资源列表；
   - 数据集；
   - 论文或研究代码；
   - 其他项目。

3. 输出中文分析，技术关键词尽量使用常见英文术语。

4. 不复述安装命令、许可证、贡献指南和营销性语言。

5. 当证据不足时，明确写入 limitations_or_uncertainties。

6. 必须输出一个合法 JSON 对象。

7. 不要输出 Markdown 代码块，不要输出 JSON 以外的文字。

JSON 输出结构：

{
  "full_name": "owner/repository",

  "is_relevant": true,

  "relevance_reason":
    "与 AI Agent、RAG、MCP、LLM 应用工程或邻近技术趋势的关系",

  "project_type":
    "framework|sdk|application|tool|library|tutorial|resource-list|dataset|research-code|other",

  "one_sentence_summary":
    "一句话说明项目是什么以及解决什么问题",

  "problem_solved": [
    "项目试图解决的问题"
  ],

  "core_capabilities": [
    "核心能力"
  ],

  "technical_features": [
    "能够从证据确认的技术特点"
  ],

  "use_cases": [
    "典型使用场景"
  ],

  "recent_changes": [
    "近期 Release 反映的变化，没有证据则为空数组"
  ],

  "community_signals": [
    "Issue 反映的需求、痛点或关注点，没有证据则为空数组"
  ],

  "limitations_or_uncertainties": [
    "材料不足、功能边界或不确定事项"
  ],

  "keywords": [
    "3 到 10 个适合后续检索的英文技术关键词"
  ],

  "hotspot_value": 1,

  "hotspot_reason":
    "仅根据项目自身的技术价值、近期变化和社区信号，
    说明为什么值得或不值得进入热点分析"
}

hotspot_value 使用 1 到 5：

1 = 基本无关或信息价值很低；

2 = 相关，但主要是教程、资源聚合，
    或成熟项目的常规更新；

3 = 具有一定工程趋势价值；

4 = 具有较强的新技术或工程趋势信号；

5 = 项目自身表现出非常突出的技术价值。

注意：

hotspot_value 仅评价项目自身，
不要判断它是否是“当天最突出”的项目，
因为当前输入中没有其他项目可供比较。
""".strip()


# ============================================================
# 命令行参数
# ============================================================


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "使用 DeepSeek 对当天入选的 GitHub 项目"
            "逐个生成结构化摘要，并同步保存到 SQLite "
            "和 RAG 文档源。"
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "处理日期，格式为 YYYY-MM-DD。"
            "不填写时使用北京时间当天。"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "本次最多检查多少个项目。"
            "0 表示检查当天全部项目。"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "重新调用 LLM 分析已经成功完成的项目。"
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
            "DeepSeek 模型名称。"
            f"默认使用 {DEFAULT_MODEL}。"
        ),
    )

    return parser.parse_args()


# ============================================================
# 文件和文本工具
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
    将数据原子化保存为 UTF-8 JSON。
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


def resolve_date(
    date_text: str | None,
) -> str:
    """
    确定需要处理的日期。
    """
    if date_text is None:
        return datetime.now(
            ZoneInfo(
                DEFAULT_TIMEZONE
            )
        ).date().isoformat()

    try:
        parsed_date = date.fromisoformat(
            date_text
        )
    except ValueError as exc:
        raise ValueError(
            "日期格式错误，应为 YYYY-MM-DD："
            f"{date_text}"
        ) from exc

    return parsed_date.isoformat()


def resolve_path(
    path_value: Any,
) -> Path:
    """
    将指针文件中的路径转换成绝对路径。
    """
    path = Path(
        str(
            path_value
        )
    )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def truncate_text(
    value: Any,
    maximum_characters: int,
) -> str:
    """
    对文本进行简单字符截断。
    """
    text = str(
        value
        or ""
    ).replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    ).strip()

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


def compact_list(
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


def save_results_file(
    output_path: Path,
    results_by_name: dict[
        str,
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    """
    按仓库名称排序并保存全部分析结果。
    """
    ordered_results = sorted(
        results_by_name.values(),
        key=lambda item: str(
            item.get(
                "full_name"
            )
            or ""
        ),
    )

    write_json(
        output_path,
        ordered_results,
    )

    return ordered_results


# ============================================================
# LLM 输入构建
# ============================================================


def build_project_input(
    material: dict[str, Any],
) -> tuple[str, str, int]:
    """
    构造单个项目的 LLM 输入。
    """
    repository = material.get(
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

    selection = material.get(
        "selection"
    )

    if not isinstance(
        selection,
        dict,
    ):
        selection = {}

    payload: dict[str, Any] = {
        "repository": {
            "full_name": full_name,

            "description": (
                repository.get(
                    "description"
                )
            ),

            "language": (
                repository.get(
                    "language"
                )
            ),

            "topics": compact_list(
                repository.get(
                    "topics"
                )
            ),

            "stars": repository.get(
                "stars"
            ),

            "forks": repository.get(
                "forks"
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

            "created_at": (
                repository.get(
                    "created_at"
                )
            ),

            "pushed_at": (
                repository.get(
                    "pushed_at"
                )
            ),

            "updated_at": (
                repository.get(
                    "updated_at"
                )
            ),

            "search_queries": compact_list(
                repository.get(
                    "search_queries"
                )
            ),
        },

        "selection": selection,

        "readme": "",

        "recent_releases": [],

        "selected_issues": [],

        "collection_errors": compact_list(
            material.get(
                "collection_errors"
            )
        ),
    }

    readme = material.get(
        "readme"
    )

    if isinstance(
        readme,
        dict,
    ):
        payload["readme"] = truncate_text(
            readme.get(
                "content"
            ),
            README_MAX_CHARACTERS,
        )

    releases = material.get(
        "releases"
    )

    if isinstance(
        releases,
        list,
    ):
        for release in releases[
            :RELEASE_MAX_ITEMS
        ]:
            if not isinstance(
                release,
                dict,
            ):
                continue

            payload[
                "recent_releases"
            ].append(
                {
                    "name": (
                        release.get(
                            "name"
                        )
                        or release.get(
                            "tag_name"
                        )
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

                    "body": truncate_text(
                        release.get(
                            "body"
                        ),
                        RELEASE_BODY_MAX_CHARACTERS,
                    ),
                }
            )

    issues = material.get(
        "issues"
    )

    if isinstance(
        issues,
        list,
    ):
        for issue in issues[
            :ISSUE_MAX_ITEMS
        ]:
            if not isinstance(
                issue,
                dict,
            ):
                continue

            payload[
                "selected_issues"
            ].append(
                {
                    "number": (
                        issue.get(
                            "number"
                        )
                    ),

                    "title": (
                        issue.get(
                            "title"
                        )
                    ),

                    "state": (
                        issue.get(
                            "state"
                        )
                    ),

                    "labels": compact_list(
                        issue.get(
                            "labels"
                        )
                    ),

                    "comments": (
                        issue.get(
                            "comments"
                        )
                    ),

                    "reactions": (
                        issue.get(
                            "reactions"
                        )
                    ),

                    "updated_at": (
                        issue.get(
                            "updated_at"
                        )
                    ),

                    "body": truncate_text(
                        issue.get(
                            "body"
                        ),
                        ISSUE_BODY_MAX_CHARACTERS,
                    ),
                }
            )

    user_prompt = (
        "请根据以下项目证据生成规定结构的 JSON 项目摘要。\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    return (
        full_name,
        user_prompt,
        len(
            user_prompt
        ),
    )


# ============================================================
# 模型结果处理
# ============================================================


def strip_json_fence(
    content: str,
) -> str:
    """
    去除模型可能返回的 Markdown JSON 代码块。
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


def normalize_summary(
    data: dict[str, Any],
    full_name: str,
) -> dict[str, Any]:
    """
    校正模型输出字段。
    """
    normalized = dict(
        data
    )

    normalized[
        "full_name"
    ] = full_name

    normalized[
        "is_relevant"
    ] = bool(
        normalized.get(
            "is_relevant",
            False,
        )
    )

    project_type = str(
        normalized.get(
            "project_type"
        )
        or "other"
    ).strip().lower()

    allowed_types = {
        "framework",
        "sdk",
        "application",
        "tool",
        "library",
        "tutorial",
        "resource-list",
        "dataset",
        "research-code",
        "other",
    }

    if project_type not in allowed_types:
        project_type = "other"

    normalized[
        "project_type"
    ] = project_type

    list_fields = (
        "problem_solved",
        "core_capabilities",
        "technical_features",
        "use_cases",
        "recent_changes",
        "community_signals",
        "limitations_or_uncertainties",
        "keywords",
    )

    for field in list_fields:
        normalized[
            field
        ] = compact_list(
            normalized.get(
                field
            )
        )

    try:
        hotspot_value = int(
            normalized.get(
                "hotspot_value",
                1,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        hotspot_value = 1

    normalized[
        "hotspot_value"
    ] = min(
        5,
        max(
            1,
            hotspot_value,
        ),
    )

    text_fields = (
        "relevance_reason",
        "one_sentence_summary",
        "hotspot_reason",
    )

    for field in text_fields:
        normalized[
            field
        ] = str(
            normalized.get(
                field
            )
            or ""
        ).strip()

    return normalized


# ============================================================
# DeepSeek 客户端
# ============================================================


class DeepSeekClient:
    """
    DeepSeek Chat Completions API 客户端。
    """

    def __init__(
        self,
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

    def analyze(
        self,
        user_prompt: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        """
        调用 DeepSeek 分析单个项目。
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

            "max_tokens": (
                MAX_OUTPUT_TOKENS
            ),

            "temperature": 0.2,

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
                        wait_seconds = min(
                            2**attempt * 2,
                            20,
                        )

                        time.sleep(
                            wait_seconds
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
                        "DeepSeek API 响应缺少 choices。"
                    )

                first_choice = choices[0]

                if not isinstance(
                    first_choice,
                    dict,
                ):
                    raise RuntimeError(
                        "DeepSeek API 的 choice 格式错误。"
                    )

                message = first_choice.get(
                    "message"
                )

                if not isinstance(
                    message,
                    dict,
                ):
                    raise RuntimeError(
                        "DeepSeek API 响应缺少 message。"
                    )

                content = str(
                    message.get(
                        "content"
                    )
                    or ""
                ).strip()

                if not content:
                    raise RuntimeError(
                        "DeepSeek API 返回了空内容。"
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

                    "usage": usage,

                    "response_id": (
                        response_data.get(
                            "id"
                        )
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
                    wait_seconds = min(
                        2**attempt * 2,
                        20,
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                break

        raise RuntimeError(
            "项目分析失败："
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
# 已有结果和持久化
# ============================================================


def load_existing_results(
    output_path: Path,
) -> list[dict[str, Any]]:
    """
    读取已经生成的项目摘要。
    """
    if not output_path.exists():
        return []

    data = read_json(
        output_path
    )

    if not isinstance(
        data,
        list,
    ):
        return []

    return [
        item
        for item in data
        if isinstance(
            item,
            dict,
        )
    ]


def persist_summary(
    *,
    summary: dict[str, Any],
    output_path: Path,
) -> dict[str, str]:
    """
    将单条成功摘要保存到：

    1. SQLite；
    2. RAG JSONL 文档源。
    """
    return save_repository_summary_assets(
        project_root=PROJECT_ROOT,
        summary=summary,
        source_file=output_path,
    )


def sync_existing_successful_results(
    *,
    results_by_name: dict[
        str,
        dict[str, Any],
    ],
    output_path: Path,
) -> tuple[
    int,
    int,
    list[str],
    dict[str, str],
]:
    """
    将已经存在的成功摘要补写进 SQLite 和 RAG 文档源。

    不调用 LLM。
    """
    success_count = 0
    failure_count = 0
    errors: list[str] = []
    last_paths: dict[str, str] = {}

    for full_name, result in (
        results_by_name.items()
    ):
        if result.get(
            "status"
        ) != "success":
            continue

        try:
            saved_paths = persist_summary(
                summary=result,
                output_path=output_path,
            )

            result[
                "persistence"
            ] = {
                "status": "success",
                **saved_paths,
            }

            last_paths = saved_paths
            success_count += 1

        except Exception as exc:
            failure_count += 1

            error_text = (
                "已有项目摘要持久化失败，"
                f"仓库：{full_name}，"
                f"原因：{exc}"
            )

            errors.append(
                error_text
            )

            result[
                "persistence"
            ] = {
                "status": "failed",
                "error": str(
                    exc
                ),
            }

    return (
        success_count,
        failure_count,
        errors,
        last_paths,
    )


# ============================================================
# 重复检测决策与历史摘要复用
# ============================================================


UPDATE_DECISIONS_FILE_NAME = (
    "repository_update_decisions.json"
)

REPOSITORY_SUMMARIES_FILE_NAME = (
    "repository_llm_summaries.json"
)


def material_repository_full_name(
    material: dict[str, Any],
) -> str:
    """
    从粗缩减分析材料中读取 owner/repository。
    """
    repository = material.get(
        "repository"
    )

    if not isinstance(
        repository,
        dict,
    ):
        return ""

    return str(
        repository.get(
            "full_name"
        )
        or ""
    ).strip()


def load_update_decisions(
    decisions_path: Path,
    snapshot_date: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """
    读取重复检测输出，并按 full_name.casefold() 建立索引。

    analyze_github_projects.py 不再自己判断项目是否需要调用
    LLM，而是以 detect_github_repository_updates.py 的输出为准。
    """
    payload = read_json(
        decisions_path
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "repository_update_decisions.json "
            "根节点不是对象。"
        )

    decision_date = str(
        payload.get(
            "snapshot_date"
        )
        or ""
    ).strip()

    if (
        decision_date
        and decision_date != snapshot_date
    ):
        raise ValueError(
            "重复检测结果日期与当前处理日期不一致："
            f"decision={decision_date}，"
            f"current={snapshot_date}"
        )

    repositories = payload.get(
        "repositories"
    )

    if not isinstance(
        repositories,
        list,
    ):
        raise ValueError(
            "repository_update_decisions.json "
            "缺少 repositories 列表。"
        )

    index: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in repositories:
        if not isinstance(
            item,
            dict,
        ):
            continue

        full_name = str(
            item.get(
                "repository"
            )
            or ""
        ).strip()

        if not full_name:
            continue

        key = full_name.casefold()

        if key in index:
            raise ValueError(
                "重复检测结果中存在重复仓库："
                f"{full_name}"
            )

        index[key] = item

    return index, payload


def load_summary_index(
    summary_path: Path,
    cache: dict[
        str,
        dict[str, dict[str, Any]]
        | None,
    ],
) -> dict[str, dict[str, Any]] | None:
    """
    读取一个 repository_llm_summaries.json，并缓存结果。
    """
    cache_key = str(
        summary_path.resolve()
    )

    if cache_key in cache:
        return cache[cache_key]

    if not summary_path.is_file():
        cache[cache_key] = None
        return None

    try:
        payload = read_json(
            summary_path
        )
    except Exception:
        cache[cache_key] = None
        return None

    if not isinstance(
        payload,
        list,
    ):
        cache[cache_key] = None
        return None

    index: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in payload:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if item.get(
            "status"
        ) != "success":
            continue

        full_name = str(
            item.get(
                "full_name"
            )
            or ""
        ).strip()

        if not full_name:
            continue

        index[
            full_name.casefold()
        ] = item

    cache[cache_key] = index
    return index


def append_unique_path(
    paths: list[Path],
    candidate: Path | None,
) -> None:
    """
    向候选列表追加尚未出现的路径。
    """
    if candidate is None:
        return

    resolved = candidate.resolve()

    if resolved not in paths:
        paths.append(
            resolved
        )


def historical_summary_candidates(
    *,
    decision: dict[str, Any],
    deep_root: Path,
) -> list[Path]:
    """
    根据重复检测结果定位同一历史日期的项目摘要文件。

    优先使用 history_lookup.analysis_material_path 的父目录，
    因为它与本次差异比较使用的是同一次历史 collection。
    只有该位置不可用时，才查看同一日期的 latest 指针和其他
    collection。不会向更早日期寻找摘要，避免复用过时结论。
    """
    candidates: list[Path] = []

    history_lookup = decision.get(
        "history_lookup"
    )

    if not isinstance(
        history_lookup,
        dict,
    ):
        history_lookup = {}

    analysis_material_value = (
        history_lookup.get(
            "analysis_material_path"
        )
    )

    if analysis_material_value:
        analysis_material_path = (
            resolve_path(
                analysis_material_value
            )
        )

        append_unique_path(
            candidates,
            analysis_material_path.parent
            / REPOSITORY_SUMMARIES_FILE_NAME,
        )

    source_value = history_lookup.get(
        "source_pointer_or_file"
    )

    if source_value:
        source_path = resolve_path(
            source_value
        )

        if source_path.name == (
            "latest_collection.json"
        ) and source_path.is_file():
            try:
                pointer = read_json(
                    source_path
                )
            except Exception:
                pointer = None

            if isinstance(
                pointer,
                dict,
            ):
                summary_value = pointer.get(
                    "repository_llm_summaries_path"
                )

                if summary_value:
                    append_unique_path(
                        candidates,
                        resolve_path(
                            summary_value
                        ),
                    )

                collection_value = pointer.get(
                    "collection_directory"
                )

                if collection_value:
                    append_unique_path(
                        candidates,
                        resolve_path(
                            collection_value
                        )
                        / REPOSITORY_SUMMARIES_FILE_NAME,
                    )

    previous_snapshot_date = str(
        decision.get(
            "previous_snapshot_date"
        )
        or history_lookup.get(
            "selected_snapshot_date"
        )
        or ""
    ).strip()

    if not previous_snapshot_date:
        return candidates

    historical_date_directory = (
        deep_root
        / previous_snapshot_date
    )

    latest_pointer_path = (
        historical_date_directory
        / "latest_collection.json"
    )

    if latest_pointer_path.is_file():
        try:
            latest_pointer = read_json(
                latest_pointer_path
            )
        except Exception:
            latest_pointer = None

        if isinstance(
            latest_pointer,
            dict,
        ):
            summary_value = latest_pointer.get(
                "repository_llm_summaries_path"
            )

            if summary_value:
                append_unique_path(
                    candidates,
                    resolve_path(
                        summary_value
                    ),
                )

            collection_value = latest_pointer.get(
                "collection_directory"
            )

            if collection_value:
                append_unique_path(
                    candidates,
                    resolve_path(
                        collection_value
                    )
                    / REPOSITORY_SUMMARIES_FILE_NAME,
                )

    if historical_date_directory.is_dir():
        collection_directories = sorted(
            (
                child
                for child in historical_date_directory.iterdir()
                if child.is_dir()
            ),
            key=lambda path: path.name,
            reverse=True,
        )

        for collection_directory in (
            collection_directories
        ):
            append_unique_path(
                candidates,
                collection_directory
                / REPOSITORY_SUMMARIES_FILE_NAME,
            )

    return candidates


def load_historical_summary(
    *,
    full_name: str,
    decision: dict[str, Any],
    deep_root: Path,
    cache: dict[
        str,
        dict[str, dict[str, Any]]
        | None,
    ],
) -> tuple[
    dict[str, Any] | None,
    Path | None,
    list[str],
]:
    """
    读取可复用的历史成功摘要。
    """
    notes: list[str] = []

    for summary_path in (
        historical_summary_candidates(
            decision=decision,
            deep_root=deep_root,
        )
    ):
        index = load_summary_index(
            summary_path,
            cache,
        )

        if index is None:
            notes.append(
                f"历史摘要不可用：{summary_path}"
            )
            continue

        summary = index.get(
            full_name.casefold()
        )

        if summary is None:
            notes.append(
                "历史摘要文件不包含目标项目："
                f"{summary_path}"
            )
            continue

        return (
            dict(summary),
            summary_path,
            notes,
        )

    return None, None, notes


def build_reused_record(
    *,
    historical_summary: dict[str, Any],
    historical_summary_path: Path,
    decision: dict[str, Any],
    full_name: str,
    snapshot_date: str,
    reused_at: str,
) -> dict[str, Any]:
    """
    将历史摘要包装成当天的摘要记录。

    文本分析结论来自历史记录，但 source_snapshot_date 更新为
    当天，以便 SQLite 和 RAG JSONL 将其作为当天情报保存；原始
    分析日期、历史日期和来源文件保存在 reuse 字段中。
    """
    record = dict(
        historical_summary
    )

    historical_source_snapshot_date = str(
        historical_summary.get(
            "source_snapshot_date"
        )
        or decision.get(
            "previous_snapshot_date"
        )
        or ""
    ).strip()

    historical_analyzed_at = (
        historical_summary.get(
            "analyzed_at"
        )
    )

    record[
        "full_name"
    ] = full_name

    record[
        "status"
    ] = "success"

    record[
        "source_snapshot_date"
    ] = snapshot_date

    record[
        "analysis_origin"
    ] = "reused_history"

    record[
        "reused_at"
    ] = reused_at

    record[
        "reuse"
    ] = {
        "reused": True,
        "source_snapshot_date": (
            historical_source_snapshot_date
        ),
        "source_analyzed_at": (
            historical_analyzed_at
        ),
        "source_summary_path": str(
            historical_summary_path
        ),
        "change_status": decision.get(
            "change_status"
        ),
        "llm_action": decision.get(
            "llm_action"
        ),
        "previous_snapshot_date": (
            decision.get(
                "previous_snapshot_date"
            )
        ),
    }

    # 当前 collection 的持久化状态必须重新写入，不能沿用历史值。
    record.pop(
        "persistence",
        None,
    )

    return record


def validate_deepseek_api_key() -> str:
    """
    仅在确实需要调用 LLM 时校验 DeepSeek API Key。
    """
    api_key = os.getenv(
        "DEEPSEEK_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "当前 PowerShell 没有设置 DEEPSEEK_API_KEY。"
        )

    if not api_key.isascii():
        raise RuntimeError(
            "DEEPSEEK_API_KEY 包含非 ASCII 字符。"
        )

    if any(
        character.isspace()
        for character in api_key
    ):
        raise RuntimeError(
            "DEEPSEEK_API_KEY 包含空格或换行。"
        )

    return api_key


def add_usage(
    total_usage: dict[str, int],
    api_metadata: dict[str, Any],
) -> None:
    """
    累加本次真正发生的 DeepSeek Token 使用量。
    """
    usage = api_metadata.get(
        "usage"
    )

    if not isinstance(
        usage,
        dict,
    ):
        return

    for usage_key in total_usage:
        try:
            total_usage[
                usage_key
            ] += int(
                usage.get(
                    usage_key
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            continue


# ============================================================
# 主程序
# ============================================================


def main() -> int:
    """
    命令行入口。
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

    deep_root = (
        PROJECT_ROOT
        / "data"
        / "intelligence"
        / "deep"
    )

    latest_pointer_path = (
        deep_root
        / snapshot_date
        / "latest_collection.json"
    )

    decisions_path = (
        deep_root
        / snapshot_date
        / UPDATE_DECISIONS_FILE_NAME
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

        analysis_material_value = (
            latest_pointer.get(
                "analysis_material_path"
            )
        )

        if not analysis_material_value:
            raise ValueError(
                "latest_collection.json "
                "缺少 analysis_material_path。"
            )

        analysis_material_path = (
            resolve_path(
                analysis_material_value
            )
        )

        materials = read_json(
            analysis_material_path
        )

        if not isinstance(
            materials,
            list,
        ):
            raise ValueError(
                "分析材料文件根节点不是列表。"
            )

        collection_directory_value = (
            latest_pointer.get(
                "collection_directory"
            )
        )

        if collection_directory_value:
            collection_directory = (
                resolve_path(
                    collection_directory_value
                )
            )
        else:
            collection_directory = (
                analysis_material_path.parent
            )

        if arguments.force:
            decision_index: dict[
                str,
                dict[str, Any],
            ] = {}

            decision_payload: dict[
                str,
                Any,
            ] = {}
        else:
            (
                decision_index,
                decision_payload,
            ) = load_update_decisions(
                decisions_path,
                snapshot_date,
            )

    except Exception as exc:
        print(
            "读取深度采集或重复检测结果失败："
            f"{exc}",
            file=sys.stderr,
        )

        if (
            not arguments.force
            and not decisions_path.exists()
        ):
            print(
                "请先运行："
                "python scripts/"
                "detect_github_repository_updates.py "
                f"--date {snapshot_date}",
                file=sys.stderr,
            )

        return 1

    all_valid_materials = [
        material
        for material in materials
        if isinstance(
            material,
            dict,
        )
    ]

    if arguments.limit > 0:
        valid_materials = (
            all_valid_materials[
                : arguments.limit
            ]
        )
    else:
        valid_materials = (
            all_valid_materials
        )

    if not valid_materials:
        print(
            "没有找到可供分析的项目材料。",
            file=sys.stderr,
        )

        return 1

    output_path = (
        collection_directory
        / REPOSITORY_SUMMARIES_FILE_NAME
    )

    run_summary_path = (
        collection_directory
        / "repository_llm_analysis_run.json"
    )

    existing_results = (
        load_existing_results(
            output_path
        )
    )

    results_by_name = {
        str(
            item.get(
                "full_name"
            )
        ): item
        for item in existing_results
        if item.get(
            "full_name"
        )
    }

    # 先补写当前 collection 中已有成功结果，不重新调用模型。
    (
        existing_persisted_count,
        existing_persistence_failure_count,
        persistence_errors,
        persistence_paths,
    ) = sync_existing_successful_results(
        results_by_name=(
            results_by_name
        ),
        output_path=output_path,
    )

    if results_by_name:
        save_results_file(
            output_path,
            results_by_name,
        )

    success_count = 0
    reused_count = 0
    skipped_count = 0
    failure_count = 0

    reuse_fallback_llm_count = 0
    decision_missing_count = 0

    new_persisted_count = 0
    reused_persisted_count = 0
    new_persistence_failure_count = 0

    total_prompt_characters = 0

    total_usage_this_run: dict[
        str,
        int,
    ] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    planned_llm_count = 0
    planned_reuse_count = 0

    if arguments.force:
        planned_llm_count = len(
            valid_materials
        )
    else:
        for material in valid_materials:
            full_name = (
                material_repository_full_name(
                    material
                )
            )

            decision = decision_index.get(
                full_name.casefold()
            )

            if (
                isinstance(
                    decision,
                    dict,
                )
                and not bool(
                    decision.get(
                        "llm_should_run",
                        True,
                    )
                )
            ):
                planned_reuse_count += 1
            else:
                planned_llm_count += 1

    started_at = datetime.now(
        ZoneInfo(
            DEFAULT_TIMEZONE
        )
    ).isoformat()

    print(
        "=" * 78
    )

    print(
        "GitHub 项目 LLM 结构化分析"
    )

    print(
        "=" * 78
    )

    print(
        f"日期：{snapshot_date}"
    )

    print(
        f"模型：{arguments.model}"
    )

    print(
        "待处理项目数："
        f"{len(valid_materials)}"
    )

    print(
        "计划调用 LLM："
        f"{planned_llm_count}"
    )

    print(
        "计划复用历史摘要："
        f"{planned_reuse_count}"
    )

    print(
        "已有成功摘要补写："
        f"{existing_persisted_count}"
    )

    print(
        "已有摘要补写失败："
        f"{existing_persistence_failure_count}"
    )

    print(
        f"输入文件：{analysis_material_path}"
    )

    if not arguments.force:
        print(
            f"变化决策：{decisions_path}"
        )

    client: DeepSeekClient | None = None

    historical_summary_cache: dict[
        str,
        dict[str, dict[str, Any]]
        | None,
    ] = {}

    try:
        for index, material in enumerate(
            valid_materials,
            start=1,
        ):
            (
                full_name,
                user_prompt,
                prompt_characters,
            ) = build_project_input(
                material
            )

            existing_result = (
                results_by_name.get(
                    full_name
                )
            )

            if (
                not arguments.force
                and isinstance(
                    existing_result,
                    dict,
                )
                and existing_result.get(
                    "status"
                )
                == "success"
            ):
                skipped_count += 1

                print(
                    f"[{index}/"
                    f"{len(valid_materials)}] "
                    f"跳过当前 collection 已完成："
                    f"{full_name}"
                )

                continue

            decision = None

            if not arguments.force:
                decision = decision_index.get(
                    full_name.casefold()
                )

            should_call_llm = True
            fallback_reason = ""

            if isinstance(
                decision,
                dict,
            ):
                should_call_llm = bool(
                    decision.get(
                        "llm_should_run",
                        True,
                    )
                )

            elif not arguments.force:
                decision_missing_count += 1
                fallback_reason = (
                    "重复检测结果缺少该项目，"
                    "为避免漏分析，降级调用 LLM。"
                )

            if (
                not arguments.force
                and isinstance(
                    decision,
                    dict,
                )
                and not should_call_llm
            ):
                (
                    historical_summary,
                    historical_summary_path,
                    reuse_notes,
                ) = load_historical_summary(
                    full_name=full_name,
                    decision=decision,
                    deep_root=deep_root,
                    cache=(
                        historical_summary_cache
                    ),
                )

                if (
                    historical_summary is not None
                    and historical_summary_path
                    is not None
                ):
                    reused_at = datetime.now(
                        ZoneInfo(
                            DEFAULT_TIMEZONE
                        )
                    ).isoformat()

                    record = build_reused_record(
                        historical_summary=(
                            historical_summary
                        ),
                        historical_summary_path=(
                            historical_summary_path
                        ),
                        decision=decision,
                        full_name=full_name,
                        snapshot_date=(
                            snapshot_date
                        ),
                        reused_at=reused_at,
                    )

                    results_by_name[
                        full_name
                    ] = record

                    save_results_file(
                        output_path,
                        results_by_name,
                    )

                    try:
                        saved_paths = persist_summary(
                            summary=record,
                            output_path=output_path,
                        )

                        record[
                            "persistence"
                        ] = {
                            "status": "success",
                            **saved_paths,
                        }

                        persistence_paths = (
                            saved_paths
                        )

                        reused_persisted_count += 1

                    except Exception as exc:
                        new_persistence_failure_count += 1

                        persistence_error = (
                            "复用项目摘要持久化失败，"
                            f"仓库：{full_name}，"
                            f"原因：{exc}"
                        )

                        persistence_errors.append(
                            persistence_error
                        )

                        record[
                            "persistence"
                        ] = {
                            "status": "failed",
                            "error": str(
                                exc
                            ),
                        }

                    save_results_file(
                        output_path,
                        results_by_name,
                    )

                    reused_count += 1

                    print(
                        f"[{index}/"
                        f"{len(valid_materials)}] "
                        f"复用历史摘要：{full_name}"
                    )

                    print(
                        "    历史日期="
                        f"{decision.get('previous_snapshot_date')}，"
                        "状态="
                        f"{decision.get('change_status')}，"
                        "保存="
                        f"{record['persistence']['status']}"
                    )

                    continue

                reuse_fallback_llm_count += 1

                fallback_reason = (
                    "决策要求复用，但没有找到可用的"
                    "历史成功摘要，降级调用 LLM。"
                )

                if reuse_notes:
                    fallback_reason += (
                        " "
                        + " | ".join(
                            reuse_notes
                        )
                    )

            print(
                f"[{index}/"
                f"{len(valid_materials)}] "
                f"正在分析：{full_name}"
            )

            if fallback_reason:
                print(
                    f"    降级原因：{fallback_reason}"
                )

            analyzed_at = datetime.now(
                ZoneInfo(
                    DEFAULT_TIMEZONE
                )
            ).isoformat()

            total_prompt_characters += (
                prompt_characters
            )

            try:
                if client is None:
                    api_key = (
                        validate_deepseek_api_key()
                    )

                    client = DeepSeekClient(
                        api_key=api_key,
                        model=arguments.model,
                    )

                (
                    parsed_result,
                    api_metadata,
                ) = client.analyze(
                    user_prompt
                )

                add_usage(
                    total_usage_this_run,
                    api_metadata,
                )

                normalized_result = (
                    normalize_summary(
                        parsed_result,
                        full_name,
                    )
                )

                record: dict[str, Any] = {
                    **normalized_result,

                    "status": "success",

                    "analyzed_at": (
                        analyzed_at
                    ),

                    "source_snapshot_date": (
                        snapshot_date
                    ),

                    "analysis_origin": (
                        "llm_current"
                    ),

                    "prompt_characters": (
                        prompt_characters
                    ),

                    "api": api_metadata,
                }

                if isinstance(
                    decision,
                    dict,
                ):
                    record[
                        "update_decision"
                    ] = {
                        "change_status": (
                            decision.get(
                                "change_status"
                            )
                        ),
                        "llm_action": (
                            decision.get(
                                "llm_action"
                            )
                        ),
                        "previous_snapshot_date": (
                            decision.get(
                                "previous_snapshot_date"
                            )
                        ),
                    }

                if fallback_reason:
                    record[
                        "decision_fallback_reason"
                    ] = fallback_reason

                results_by_name[
                    full_name
                ] = record

                # 先保存项目摘要 JSON，再以该文件作为持久化来源。
                save_results_file(
                    output_path,
                    results_by_name,
                )

                try:
                    saved_paths = persist_summary(
                        summary=record,
                        output_path=output_path,
                    )

                    record[
                        "persistence"
                    ] = {
                        "status": "success",
                        **saved_paths,
                    }

                    persistence_paths = (
                        saved_paths
                    )

                    new_persisted_count += 1

                except Exception as exc:
                    new_persistence_failure_count += 1

                    persistence_error = (
                        "新项目摘要持久化失败，"
                        f"仓库：{full_name}，"
                        f"原因：{exc}"
                    )

                    persistence_errors.append(
                        persistence_error
                    )

                    record[
                        "persistence"
                    ] = {
                        "status": "failed",
                        "error": str(
                            exc
                        ),
                    }

                save_results_file(
                    output_path,
                    results_by_name,
                )

                success_count += 1

                print(
                    "    完成："
                    f"类型="
                    f"{record['project_type']}，"
                    f"热点价值="
                    f"{record['hotspot_value']}，"
                    f"相关="
                    f"{record['is_relevant']}，"
                    "保存="
                    f"{record['persistence']['status']}"
                )

            except Exception as exc:
                failure_count += 1

                results_by_name[
                    full_name
                ] = {
                    "full_name": (
                        full_name
                    ),

                    "status": "failed",

                    "analyzed_at": (
                        analyzed_at
                    ),

                    "source_snapshot_date": (
                        snapshot_date
                    ),

                    "analysis_origin": (
                        "llm_current"
                    ),

                    "prompt_characters": (
                        prompt_characters
                    ),

                    "error": str(
                        exc
                    ),
                }

                save_results_file(
                    output_path,
                    results_by_name,
                )

                print(
                    f"    失败：{exc}"
                )

    finally:
        if client is not None:
            client.close()

    finished_at = datetime.now(
        ZoneInfo(
            DEFAULT_TIMEZONE
        )
    ).isoformat()

    all_results = save_results_file(
        output_path,
        results_by_name,
    )

    successful_results = [
        item
        for item in all_results
        if item.get(
            "status"
        )
        == "success"
    ]

    total_usage_all_stored_results: dict[
        str,
        int,
    ] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for result in successful_results:
        api_data = result.get(
            "api"
        )

        if not isinstance(
            api_data,
            dict,
        ):
            continue

        add_usage(
            total_usage_all_stored_results,
            api_data,
        )

    total_persistence_failure_count = (
        existing_persistence_failure_count
        + new_persistence_failure_count
    )

    if (
        failure_count == 0
        and total_persistence_failure_count == 0
    ):
        final_status = "success"

    elif successful_results:
        final_status = (
            "partial_success"
        )

    else:
        final_status = "failed"

    run_summary = {
        "status": final_status,

        "snapshot_date": (
            snapshot_date
        ),

        "started_at": started_at,

        "finished_at": (
            finished_at
        ),

        "model": arguments.model,

        "source_material_count": len(
            all_valid_materials
        ),

        "requested_count": len(
            valid_materials
        ),

        "planned_llm_count": (
            planned_llm_count
        ),

        "planned_reuse_count": (
            planned_reuse_count
        ),

        "new_success_count": (
            success_count
        ),

        "reused_success_count": (
            reused_count
        ),

        "reuse_fallback_llm_count": (
            reuse_fallback_llm_count
        ),

        "decision_missing_count": (
            decision_missing_count
        ),

        "skipped_count": (
            skipped_count
        ),

        "llm_failure_count": (
            failure_count
        ),

        "stored_success_count": len(
            successful_results
        ),

        "existing_persisted_count": (
            existing_persisted_count
        ),

        "new_persisted_count": (
            new_persisted_count
        ),

        "reused_persisted_count": (
            reused_persisted_count
        ),

        "persistence_failure_count": (
            total_persistence_failure_count
        ),

        "persistence_errors": (
            persistence_errors
        ),

        "total_prompt_characters_this_run": (
            total_prompt_characters
        ),

        "total_usage_this_run": (
            total_usage_this_run
        ),

        "total_usage_all_stored_results": (
            total_usage_all_stored_results
        ),

        "input_path": str(
            analysis_material_path
        ),

        "decision_path": (
            None
            if arguments.force
            else str(
                decisions_path
            )
        ),

        "output_path": str(
            output_path
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
        "repository_llm_summaries_path"
    ] = str(
        output_path
    )

    latest_pointer[
        "repository_llm_analysis_run_path"
    ] = str(
        run_summary_path
    )

    latest_pointer[
        "repository_llm_analyzed_at"
    ] = finished_at

    latest_pointer[
        "repository_update_decisions_path"
    ] = (
        None
        if arguments.force
        else str(
            decisions_path
        )
    )

    latest_pointer[
        "repository_llm_new_count"
    ] = success_count

    latest_pointer[
        "repository_llm_reused_count"
    ] = reused_count

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
            "repository_summary_rag_documents_path"
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
        "分析、复用与持久化完成"
    )

    print(
        "=" * 78
    )

    print(
        f"本次新分析成功：{success_count}"
    )

    print(
        f"历史摘要复用成功：{reused_count}"
    )

    print(
        "复用失败后降级分析："
        f"{reuse_fallback_llm_count}"
    )

    print(
        f"跳过当前已完成：{skipped_count}"
    )

    print(
        f"LLM 失败：{failure_count}"
    )

    print(
        "累计成功摘要："
        f"{len(successful_results)}"
    )

    print(
        "已有摘要补写成功："
        f"{existing_persisted_count}"
    )

    print(
        "新摘要持久化成功："
        f"{new_persisted_count}"
    )

    print(
        "复用摘要持久化成功："
        f"{reused_persisted_count}"
    )

    print(
        "持久化失败："
        f"{total_persistence_failure_count}"
    )

    print(
        "本次实际 Token 使用："
        f"{total_usage_this_run}"
    )

    print(
        f"项目摘要：{output_path}"
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

    if persistence_errors:
        print()
        print(
            "持久化错误："
        )

        for error in persistence_errors:
            print(
                f"  - {error}"
            )

    if final_status == "failed":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )