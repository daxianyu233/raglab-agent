from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TIMEZONE_NAME = "Asia/Shanghai"

DEEP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "intelligence"
    / "deep"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "storage"
    / "intelligence"
    / "github_intelligence.sqlite3"
)

CURRENT_MANIFEST_NAME = "latest_collection.json"

ANALYSIS_MATERIAL_NAME = (
    "github_repository_analysis_material.json"
)

REPOSITORY_SUMMARIES_NAME = (
    "repository_llm_summaries.json"
)

OUTPUT_FILE_NAME = (
    "repository_update_decisions.json"
)

README_MINOR_SIMILARITY_THRESHOLD = 0.92
README_MAJOR_SIMILARITY_THRESHOLD = 0.72
README_MAJOR_LENGTH_RATIO = 0.60

MAX_SIMILARITY_CHARACTERS = 80_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检测当天 GitHub 项目是否已经在同日完成 LLM 分析；"
            "未完成时再查询跨日历史并比较粗缩减材料。"
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "处理日期，格式为 YYYY-MM-DD；"
            "默认使用北京时间当天。"
        ),
    )

    parser.add_argument(
        "--deep-root",
        type=Path,
        default=DEEP_ROOT,
        help="GitHub 深度采集数据根目录。",
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="GitHub 技术情报 SQLite 数据库路径。",
    )

    return parser.parse_args()


def compact_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value).split()
    ).strip()


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def save_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def resolve_path(
    raw_path: Any,
    reference_path: Path,
) -> Path:
    candidate = Path(
        str(raw_path)
    )

    if candidate.is_absolute():
        return candidate.resolve()

    reference_candidate = (
        reference_path.parent
        / candidate
    ).resolve()

    if reference_candidate.exists():
        return reference_candidate

    return (
        PROJECT_ROOT
        / candidate
    ).resolve()


def sha256_text(
    text: str,
) -> str:
    if not text:
        return ""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def stable_json_text(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def safe_int(
    value: Any,
) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def first_nonempty(
    *values: Any,
) -> Any:
    for value in values:
        if value not in (
            None,
            "",
            [],
            {},
        ):
            return value

    return None


def repository_full_name(
    record: dict[str, Any],
) -> str:
    repository = record.get(
        "repository"
    )

    if isinstance(
        repository,
        dict,
    ):
        full_name = compact_text(
            first_nonempty(
                repository.get(
                    "full_name"
                ),
                repository.get(
                    "name_with_owner"
                ),
            )
        )

        if "/" in full_name:
            return full_name

        owner = repository.get(
            "owner"
        )

        name = repository.get(
            "name"
        )

        if isinstance(
            owner,
            dict,
        ):
            owner = first_nonempty(
                owner.get("login"),
                owner.get("name"),
            )

        if owner and name:
            return f"{owner}/{name}"

    for key in (
        "full_name",
        "repository_full_name",
        "repo_full_name",
        "name_with_owner",
    ):
        value = compact_text(
            record.get(key)
        )

        if "/" in value:
            return value

    html_url = compact_text(
        first_nonempty(
            record.get("html_url"),
            record.get("url"),
        )
    )

    match = re.search(
        r"github\.com/([^/\s]+)/([^/\s#?]+)",
        html_url,
        flags=re.IGNORECASE,
    )

    if match:
        owner = match.group(1)
        name = match.group(2)

        if name.endswith(".git"):
            name = name[:-4]

        return f"{owner}/{name}"

    return ""


def load_analysis_material(
    source_path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    Path,
]:
    """
    source_path 可以是：

    1. latest_collection.json；
    2. github_repository_analysis_material.json。
    """
    payload = load_json(
        source_path
    )

    actual_path = source_path

    if (
        source_path.name
        == CURRENT_MANIFEST_NAME
        and isinstance(
            payload,
            dict,
        )
    ):
        material_value = payload.get(
            "analysis_material_path"
        )

        if not material_value:
            raise ValueError(
                f"{source_path} "
                "缺少 analysis_material_path。"
            )

        actual_path = resolve_path(
            material_value,
            source_path,
        )

        payload = load_json(
            actual_path
        )

    if isinstance(
        payload,
        dict,
    ):
        for key in (
            "repositories",
            "materials",
            "items",
            "records",
        ):
            value = payload.get(key)

            if isinstance(
                value,
                list,
            ):
                payload = value
                break

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(
            "分析材料根节点不是列表："
            f"{actual_path}"
        )

    records: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in payload:
        if not isinstance(
            item,
            dict,
        ):
            continue

        full_name = (
            repository_full_name(
                item
            )
        )

        if not full_name:
            continue

        records[
            full_name
        ] = item

    if not records:
        raise RuntimeError(
            "分析材料中没有识别出仓库记录："
            f"{actual_path}"
        )

    return (
        records,
        actual_path,
    )


def find_record_case_insensitive(
    records: dict[
        str,
        dict[str, Any],
    ],
    full_name: str,
) -> dict[str, Any] | None:
    direct = records.get(
        full_name
    )

    if isinstance(
        direct,
        dict,
    ):
        return direct

    target = full_name.casefold()

    for (
        historical_name,
        record,
    ) in records.items():
        if (
            historical_name.casefold()
            == target
            and isinstance(
                record,
                dict,
            )
        ):
            return record

    return None


def value_to_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        str,
    ):
        return value.strip()

    if isinstance(
        value,
        dict,
    ):
        for key in (
            "content",
            "body",
            "text",
            "markdown",
            "decoded_content",
        ):
            content = value.get(key)

            if isinstance(
                content,
                str,
            ):
                return content.strip()

    return stable_json_text(
        value
    )


def normalize_readme(
    value: Any,
) -> str:
    text = value_to_text(
        value
    )

    if not text:
        return ""

    text = re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL,
    )

    retained_lines: list[str] = []

    for line in text.splitlines():
        lowered = line.lower()

        if (
            "shields.io" in lowered
            or "badge.svg" in lowered
            or "actions/workflows"
            in lowered
        ):
            continue

        retained_lines.append(
            line
        )

    text = "\n".join(
        retained_lines
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def normalize_labels(
    value: Any,
) -> list[str]:
    if not isinstance(
        value,
        list,
    ):
        return []

    labels: list[str] = []

    for item in value:
        if isinstance(
            item,
            str,
        ):
            name = compact_text(
                item
            )

        elif isinstance(
            item,
            dict,
        ):
            name = compact_text(
                first_nonempty(
                    item.get("name"),
                    item.get("title"),
                )
            )

        else:
            name = ""

        if name:
            labels.append(name)

    return sorted(
        set(labels)
    )


def normalize_releases(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        dict,
    ):
        items = value.get(
            "items"
        )

        value = (
            items
            if isinstance(
                items,
                list,
            )
            else [value]
        )

    if not isinstance(
        value,
        list,
    ):
        return value_to_text(
            value
        )

    normalized: list[
        dict[str, Any]
    ] = []

    for item in value:
        if not isinstance(
            item,
            dict,
        ):
            continue

        normalized.append(
            {
                "id": item.get("id"),
                "tag_name": item.get(
                    "tag_name"
                ),
                "name": item.get(
                    "name"
                ),
                "published_at": item.get(
                    "published_at"
                ),
                "body": compact_text(
                    first_nonempty(
                        item.get("body"),
                        item.get("content"),
                    )
                ),
                "draft": item.get(
                    "draft"
                ),
                "prerelease": item.get(
                    "prerelease"
                ),
            }
        )

    normalized.sort(
        key=lambda item: (
            compact_text(
                item.get(
                    "published_at"
                )
            ),
            compact_text(
                item.get(
                    "tag_name"
                )
            ),
        )
    )

    return stable_json_text(
        normalized
    )


def normalize_issues(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        dict,
    ):
        items = value.get(
            "items"
        )

        value = (
            items
            if isinstance(
                items,
                list,
            )
            else [value]
        )

    if not isinstance(
        value,
        list,
    ):
        return value_to_text(
            value
        )

    normalized: list[
        dict[str, Any]
    ] = []

    for item in value:
        if not isinstance(
            item,
            dict,
        ):
            continue

        normalized.append(
            {
                "id": item.get("id"),
                "number": item.get(
                    "number"
                ),
                "title": compact_text(
                    item.get("title")
                ),
                "body": compact_text(
                    item.get("body")
                ),
                "state": item.get(
                    "state"
                ),
                "labels": normalize_labels(
                    item.get("labels")
                ),
            }
        )

    normalized.sort(
        key=lambda item: (
            item.get("number") or 0,
            compact_text(
                item.get("title")
            ),
        )
    )

    return stable_json_text(
        normalized
    )


def normalize_topics(
    value: Any,
) -> list[str]:
    if not isinstance(
        value,
        list,
    ):
        return []

    return sorted(
        {
            compact_text(
                item
            ).lower()
            for item in value
            if compact_text(item)
        }
    )


def normalize_license(
    value: Any,
) -> str:
    if isinstance(
        value,
        dict,
    ):
        return compact_text(
            first_nonempty(
                value.get("spdx_id"),
                value.get("key"),
                value.get("name"),
            )
        )

    return compact_text(
        value
    )


def build_repository_snapshot(
    full_name: str,
    material: dict[str, Any],
) -> dict[str, Any]:
    repository = material.get(
        "repository"
    )

    if not isinstance(
        repository,
        dict,
    ):
        repository = material

    readme = normalize_readme(
        first_nonempty(
            material.get("readme"),
            material.get(
                "readme_text"
            ),
            material.get(
                "readme_content"
            ),
            repository.get(
                "readme"
            ),
        )
    )

    releases = normalize_releases(
        first_nonempty(
            material.get(
                "releases"
            ),
            material.get(
                "recent_releases"
            ),
            repository.get(
                "releases"
            ),
        )
    )

    issues = normalize_issues(
        first_nonempty(
            material.get(
                "issues"
            ),
            material.get(
                "recent_issues"
            ),
            repository.get(
                "issues"
            ),
        )
    )

    description = compact_text(
        first_nonempty(
            repository.get(
                "description"
            ),
            material.get(
                "description"
            ),
        )
    )

    language = compact_text(
        first_nonempty(
            repository.get(
                "language"
            ),
            repository.get(
                "primary_language"
            ),
            material.get(
                "language"
            ),
        )
    )

    topics = normalize_topics(
        first_nonempty(
            repository.get(
                "topics"
            ),
            material.get(
                "topics"
            ),
        )
    )

    license_name = normalize_license(
        first_nonempty(
            repository.get(
                "license"
            ),
            material.get(
                "license"
            ),
        )
    )

    default_branch = compact_text(
        first_nonempty(
            repository.get(
                "default_branch"
            ),
            material.get(
                "default_branch"
            ),
        )
    )

    return {
        "repository": full_name,

        "content": {
            "readme": readme,
            "releases": releases,
            "issues": issues,
        },

        "hashes": {
            "readme": sha256_text(
                readme
            ),
            "releases": sha256_text(
                releases
            ),
            "issues": sha256_text(
                issues
            ),
        },

        "stable_metadata": {
            "description": description,
            "language": language,
            "topics": topics,
            "license": license_name,
            "default_branch": (
                default_branch
            ),
            "archived": first_nonempty(
                repository.get(
                    "archived"
                ),
                material.get(
                    "archived"
                ),
            ),
            "disabled": first_nonempty(
                repository.get(
                    "disabled"
                ),
                material.get(
                    "disabled"
                ),
            ),
        },

        "activity": {
            "updated_at": first_nonempty(
                repository.get(
                    "updated_at"
                ),
                material.get(
                    "updated_at"
                ),
            ),
            "pushed_at": first_nonempty(
                repository.get(
                    "pushed_at"
                ),
                material.get(
                    "pushed_at"
                ),
            ),
            "default_branch_sha": (
                first_nonempty(
                    repository.get(
                        "default_branch_sha"
                    ),
                    repository.get(
                        "head_sha"
                    ),
                    repository.get(
                        "commit_sha"
                    ),
                    material.get(
                        "default_branch_sha"
                    ),
                )
            ),
        },

        "metrics": {
            "stars": safe_int(
                first_nonempty(
                    repository.get(
                        "stargazers_count"
                    ),
                    repository.get(
                        "stars"
                    ),
                    material.get(
                        "stars"
                    ),
                )
            ),
            "forks": safe_int(
                first_nonempty(
                    repository.get(
                        "forks_count"
                    ),
                    repository.get(
                        "forks"
                    ),
                    material.get(
                        "forks"
                    ),
                )
            ),
            "watchers": safe_int(
                first_nonempty(
                    repository.get(
                        "watchers_count"
                    ),
                    repository.get(
                        "watchers"
                    ),
                    material.get(
                        "watchers"
                    ),
                )
            ),
            "open_issues": safe_int(
                first_nonempty(
                    repository.get(
                        "open_issues_count"
                    ),
                    repository.get(
                        "open_issues"
                    ),
                    material.get(
                        "open_issues"
                    ),
                )
            ),
        },
    }


def text_similarity(
    previous: str,
    current: str,
) -> float | None:
    if not previous or not current:
        return None

    return SequenceMatcher(
        None,
        previous[
            :MAX_SIMILARITY_CHARACTERS
        ],
        current[
            :MAX_SIMILARITY_CHARACTERS
        ],
        autojunk=False,
    ).ratio()


def text_length_ratio(
    previous: str,
    current: str,
) -> float | None:
    if not previous or not current:
        return None

    longer = max(
        len(previous),
        len(current),
    )

    if longer == 0:
        return None

    return (
        min(
            len(previous),
            len(current),
        )
        / longer
    )


def dictionary_changes(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[
    str,
    dict[str, Any],
]:
    changes: dict[
        str,
        dict[str, Any],
    ] = {}

    for key in sorted(
        set(previous)
        | set(current)
    ):
        old_value = previous.get(
            key
        )

        new_value = current.get(
            key
        )

        if old_value != new_value:
            changes[key] = {
                "previous": old_value,
                "current": new_value,
            }

    return changes


def metric_changes(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[
    str,
    dict[str, Any],
]:
    changes: dict[
        str,
        dict[str, Any],
    ] = {}

    for key in sorted(
        set(previous)
        | set(current)
    ):
        old_value = previous.get(
            key
        )

        new_value = current.get(
            key
        )

        if old_value == new_value:
            continue

        delta = None

        if (
            isinstance(
                old_value,
                int,
            )
            and isinstance(
                new_value,
                int,
            )
        ):
            delta = (
                new_value
                - old_value
            )

        changes[key] = {
            "previous": old_value,
            "current": new_value,
            "delta": delta,
        }

    return changes


def new_or_baseline_decision(
    current: dict[str, Any],
    *,
    history_exists_in_sql: bool,
) -> dict[str, Any]:
    if history_exists_in_sql:
        status = (
            "BASELINE_REQUIRED"
        )

        reason = (
            "SQLite 中存在历史入选记录，"
            "但没有找到可用的历史粗缩减材料；"
            "需要重新建立项目分析基线。"
        )

    else:
        status = "NEW"

        reason = (
            "SQLite 历史入选记录中"
            "未找到该项目。"
        )

    return {
        "repository": current[
            "repository"
        ],
        "change_status": status,
        "llm_action": (
            "FULL_ANALYSIS"
        ),
        "llm_should_run": True,
        "previous_snapshot_date": (
            None
        ),
        "reasons": [reason],
        "content_changes": {},
        "stable_metadata_changes": {},
        "activity_changes": {},
        "metric_changes": {},
        "similarities": {},
        "current_hashes": current[
            "hashes"
        ],
    }


def compare_repository_snapshots(
    current: dict[str, Any],
    previous: (
        dict[str, Any]
        | None
    ),
    previous_snapshot_date: (
        str
        | None
    ),
    *,
    history_exists_in_sql: bool,
) -> dict[str, Any]:
    if previous is None:
        return (
            new_or_baseline_decision(
                current,
                history_exists_in_sql=(
                    history_exists_in_sql
                ),
            )
        )

    content_changes = {
        key: (
            previous[
                "hashes"
            ].get(key)
            != current[
                "hashes"
            ].get(key)
        )
        for key in (
            "readme",
            "releases",
            "issues",
        )
    }

    stable_metadata_changes = (
        dictionary_changes(
            previous[
                "stable_metadata"
            ],
            current[
                "stable_metadata"
            ],
        )
    )

    activity_changes = (
        dictionary_changes(
            previous[
                "activity"
            ],
            current[
                "activity"
            ],
        )
    )

    metrics_changed = (
        metric_changes(
            previous[
                "metrics"
            ],
            current[
                "metrics"
            ],
        )
    )

    previous_readme = (
        previous[
            "content"
        ].get(
            "readme",
            "",
        )
    )

    current_readme = (
        current[
            "content"
        ].get(
            "readme",
            "",
        )
    )

    readme_similarity = (
        text_similarity(
            previous_readme,
            current_readme,
        )
    )

    readme_length_ratio = (
        text_length_ratio(
            previous_readme,
            current_readme,
        )
    )

    if (
        not previous_readme
        and current_readme
    ):
        return {
            "repository": current[
                "repository"
            ],
            "change_status": (
                "BASELINE_REQUIRED"
            ),
            "llm_action": (
                "FULL_ANALYSIS"
            ),
            "llm_should_run": True,
            "previous_snapshot_date": (
                previous_snapshot_date
            ),
            "reasons": [
                (
                    "历史材料缺少可比较的 README，"
                    "需要建立完整分析基线。"
                )
            ],
            "content_changes": (
                content_changes
            ),
            "stable_metadata_changes": (
                stable_metadata_changes
            ),
            "activity_changes": (
                activity_changes
            ),
            "metric_changes": (
                metrics_changed
            ),
            "similarities": {
                "readme": (
                    readme_similarity
                ),
                "readme_length_ratio": (
                    readme_length_ratio
                ),
            },
            "current_hashes": current[
                "hashes"
            ],
        }

    major_readme_change = (
        content_changes[
            "readme"
        ]
        and readme_similarity
        is not None
        and (
            readme_similarity
            < README_MAJOR_SIMILARITY_THRESHOLD

            or (
                readme_length_ratio
                is not None
                and readme_length_ratio
                < README_MAJOR_LENGTH_RATIO
            )
        )
    )

    important_metadata_changed = (
        bool(
            {
                "default_branch",
                "archived",
                "disabled",
            }
            & set(
                stable_metadata_changes
            )
        )
    )

    reasons: list[str] = []

    if major_readme_change:
        status = (
            "MAJOR_UPDATED"
        )

        llm_action = (
            "FULL_REANALYSIS"
        )

        llm_should_run = True

        reasons.append(
            "README 发生大规模内容变化。"
        )

    elif important_metadata_changed:
        status = (
            "MAJOR_UPDATED"
        )

        llm_action = (
            "FULL_REANALYSIS"
        )

        llm_should_run = True

        reasons.append(
            "项目关键状态或默认分支发生变化。"
        )

    elif content_changes[
        "releases"
    ]:
        status = "UPDATED"

        llm_action = (
            "DELTA_ANALYSIS"
        )

        llm_should_run = True

        reasons.append(
            "Release 内容发生变化。"
        )

    elif content_changes[
        "issues"
    ]:
        status = "UPDATED"

        llm_action = (
            "DELTA_ANALYSIS"
        )

        llm_should_run = True

        reasons.append(
            "重点 Issue 内容发生变化。"
        )

    elif (
        content_changes[
            "readme"
        ]
        and readme_similarity
        is not None
        and readme_similarity
        >= README_MINOR_SIMILARITY_THRESHOLD
    ):
        status = (
            "MINOR_UPDATED"
        )

        llm_action = (
            "REUSE_SUMMARY"
        )

        llm_should_run = False

        reasons.append(
            "README 仅有小规模修改，"
            "暂不重新调用 LLM。"
        )

    elif content_changes[
        "readme"
    ]:
        status = "UPDATED"

        llm_action = (
            "DELTA_ANALYSIS"
        )

        llm_should_run = True

        reasons.append(
            "README 出现明显但非整体重写的变化。"
        )

    elif stable_metadata_changes:
        status = "UPDATED"

        llm_action = (
            "DELTA_ANALYSIS"
        )

        llm_should_run = True

        reasons.append(
            "项目稳定元数据发生变化。"
        )

    elif (
        activity_changes
        or metrics_changed
    ):
        status = (
            "METADATA_ONLY"
        )

        llm_action = (
            "REUSE_SUMMARY"
        )

        llm_should_run = False

        reasons.append(
            "只有活跃度或热度指标变化，"
            "复用历史项目摘要。"
        )

    else:
        status = "UNCHANGED"

        llm_action = (
            "REUSE_SUMMARY"
        )

        llm_should_run = False

        reasons.append(
            "未检测到有效变化。"
        )

    return {
        "repository": current[
            "repository"
        ],
        "change_status": status,
        "llm_action": llm_action,
        "llm_should_run": (
            llm_should_run
        ),
        "previous_snapshot_date": (
            previous_snapshot_date
        ),
        "reasons": reasons,
        "content_changes": (
            content_changes
        ),
        "stable_metadata_changes": (
            stable_metadata_changes
        ),
        "activity_changes": (
            activity_changes
        ),
        "metric_changes": (
            metrics_changed
        ),
        "similarities": {
            "readme": (
                readme_similarity
            ),
            "readme_length_ratio": (
                readme_length_ratio
            ),
        },
        "current_hashes": current[
            "hashes"
        ],
    }


def load_successful_summary_index(
    summary_path: Path,
    cache: dict[
        str,
        (
            dict[
                str,
                dict[str, Any],
            ]
            | None
        ),
    ],
) -> (
    dict[
        str,
        dict[str, Any],
    ]
    | None
):
    cache_key = str(
        summary_path.resolve()
    )

    if cache_key in cache:
        return cache[
            cache_key
        ]

    if not summary_path.is_file():
        cache[
            cache_key
        ] = None

        return None

    try:
        payload = load_json(
            summary_path
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        cache[
            cache_key
        ] = None

        return None

    if not isinstance(
        payload,
        list,
    ):
        cache[
            cache_key
        ] = None

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

        status = compact_text(
            item.get("status")
        ).lower()

        if status != "success":
            continue

        full_name = compact_text(
            item.get(
                "full_name"
            )
        )

        if full_name:
            index[
                full_name.casefold()
            ] = item

    cache[
        cache_key
    ] = index

    return index


def load_material_cached(
    material_path: Path,
    cache: dict[
        str,
        (
            tuple[
                dict[
                    str,
                    dict[str, Any],
                ],
                Path,
            ]
            | None
        ),
    ],
) -> (
    tuple[
        dict[
            str,
            dict[str, Any],
        ],
        Path,
    ]
    | None
):
    cache_key = str(
        material_path.resolve()
    )

    if cache_key in cache:
        return cache[
            cache_key
        ]

    if not material_path.is_file():
        cache[
            cache_key
        ] = None

        return None

    try:
        result = (
            load_analysis_material(
                material_path
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ):
        cache[
            cache_key
        ] = None

        return None

    cache[
        cache_key
    ] = result

    return result


def find_same_day_processing_state(
    *,
    deep_root: Path,
    full_name: str,
    current_snapshot_date: str,
    current_analysis_material_path: Path,
    material_cache: dict[
        str,
        (
            tuple[
                dict[
                    str,
                    dict[str, Any],
                ],
                Path,
            ]
            | None
        ),
    ],
    summary_cache: dict[
        str,
        (
            dict[
                str,
                dict[str, Any],
            ]
            | None
        ),
    ],
) -> dict[str, Any]:
    """
    同日判断以成功的项目级 LLM 摘要为准，
    而不是以粗缩减材料是否存在为准。

    processed：
        存在 full_name 匹配且 status=success 的摘要。

    analysis_required：
        当天采集过该项目，但没有成功摘要。

    not_found：
        当天此前没有该项目。
    """
    date_directory = (
        deep_root
        / current_snapshot_date
    )

    base_result = {
        "state": "not_found",
        "found": False,
        "method": (
            "same_day_llm_summary"
        ),
        "history_exists_in_sql": (
            False
        ),
        "candidate_snapshot_dates": [],
        "snapshot_date": None,
        "record": None,
        "source_pointer_or_file": (
            None
        ),
        "analysis_material_path": (
            None
        ),
        "repository_llm_summaries_path": (
            None
        ),
        "lookup_notes": [],
    }

    if not date_directory.is_dir():
        return base_result

    current_collection_directory = (
        current_analysis_material_path
        .parent
        .resolve()
    )

    collection_directories = [
        child
        for child
        in date_directory.iterdir()
        if (
            child.is_dir()
            and child.resolve()
            != current_collection_directory
        )
    ]

    def modified_time(
        directory: Path,
    ) -> float:
        for candidate in (
            directory
            / REPOSITORY_SUMMARIES_NAME,

            directory
            / ANALYSIS_MATERIAL_NAME,

            directory,
        ):
            try:
                return (
                    candidate
                    .stat()
                    .st_mtime
                )
            except OSError:
                continue

        return 0.0

    collection_directories.sort(
        key=lambda directory: (
            modified_time(
                directory
            ),
            directory.name,
        ),
        reverse=True,
    )

    target_key = (
        full_name.casefold()
    )

    lookup_notes: list[
        str
    ] = []

    latest_material_attempt: (
        dict[str, Any]
        | None
    ) = None

    for collection_directory in (
        collection_directories
    ):
        summary_path = (
            collection_directory
            / REPOSITORY_SUMMARIES_NAME
        )

        material_path = (
            collection_directory
            / ANALYSIS_MATERIAL_NAME
        )

        summary_index = (
            load_successful_summary_index(
                summary_path,
                summary_cache,
            )
        )

        if (
            isinstance(
                summary_index,
                dict,
            )
            and target_key
            in summary_index
        ):
            return {
                "state": "processed",
                "found": True,
                "method": (
                    "same_day_llm_summary"
                ),
                "history_exists_in_sql": (
                    False
                ),
                "candidate_snapshot_dates": [
                    current_snapshot_date
                ],
                "snapshot_date": (
                    current_snapshot_date
                ),
                "record": None,
                "source_pointer_or_file": (
                    str(summary_path)
                ),
                "analysis_material_path": (
                    str(material_path)
                ),
                "repository_llm_summaries_path": (
                    str(summary_path)
                ),
                "lookup_notes": (
                    lookup_notes
                ),
            }

        if summary_path.is_file():
            lookup_notes.append(
                "同日摘要文件没有目标项目的成功记录："
                f"{summary_path}。"
            )

        loaded = (
            load_material_cached(
                material_path,
                material_cache,
            )
        )

        if loaded is None:
            continue

        (
            records,
            actual_material_path,
        ) = loaded

        historical_record = (
            find_record_case_insensitive(
                records,
                full_name,
            )
        )

        if historical_record is None:
            continue

        if (
            latest_material_attempt
            is None
        ):
            latest_material_attempt = {
                "state": (
                    "analysis_required"
                ),
                "found": True,
                "method": (
                    "same_day_material_without_"
                    "successful_summary"
                ),
                "history_exists_in_sql": (
                    False
                ),
                "candidate_snapshot_dates": [
                    current_snapshot_date
                ],
                "snapshot_date": (
                    current_snapshot_date
                ),
                "record": (
                    historical_record
                ),
                "source_pointer_or_file": (
                    str(material_path)
                ),
                "analysis_material_path": (
                    str(
                        actual_material_path
                    )
                ),
                "repository_llm_summaries_path": (
                    str(summary_path)
                    if summary_path.is_file()
                    else None
                ),
                "lookup_notes": (
                    lookup_notes
                ),
            }

    if (
        latest_material_attempt
        is not None
    ):
        return (
            latest_material_attempt
        )

    base_result[
        "lookup_notes"
    ] = lookup_notes

    return base_result


def build_same_day_reuse_decision(
    current_snapshot: dict[
        str,
        Any,
    ],
    same_day_result: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    summary_path = (
        same_day_result.get(
            "repository_llm_summaries_path"
        )
    )

    return {
        "repository": (
            current_snapshot[
                "repository"
            ]
        ),
        "change_status": (
            "SAME_DAY_REUSED"
        ),
        "llm_action": (
            "REUSE_SUMMARY"
        ),
        "llm_should_run": False,
        "previous_snapshot_date": (
            same_day_result.get(
                "snapshot_date"
            )
        ),
        "reasons": [
            (
                "当天此前的项目级 LLM 摘要中"
                "已经存在该项目，且 status=success；"
                "直接复用，不进行相似度比较。"
            ),
            (
                "同日成功摘要："
                f"{summary_path}"
            ),
        ],
        "content_changes": {},
        "stable_metadata_changes": {},
        "activity_changes": {},
        "metric_changes": {},
        "similarities": {
            "skipped": True,
            "reason": (
                "same_day_successful_"
                "llm_summary"
            ),
        },
        "current_hashes": (
            current_snapshot[
                "hashes"
            ]
        ),
    }


def build_same_day_analysis_required_decision(
    current_snapshot: dict[
        str,
        Any,
    ],
    same_day_result: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    material_path = (
        same_day_result.get(
            "analysis_material_path"
        )
    )

    return {
        "repository": (
            current_snapshot[
                "repository"
            ]
        ),
        "change_status": (
            "SAME_DAY_ANALYSIS_REQUIRED"
        ),
        "llm_action": (
            "FULL_ANALYSIS"
        ),
        "llm_should_run": True,
        "previous_snapshot_date": (
            same_day_result.get(
                "snapshot_date"
            )
        ),
        "reasons": [
            (
                "当天此前的粗缩减材料中存在该项目，"
                "但没有找到 full_name 匹配且 "
                "status=success 的项目级 LLM 摘要。"
            ),
            (
                "这只能证明已经采集，"
                "不能证明已经成功分析；"
                "本次继续调用 LLM。"
            ),
            (
                "同日粗缩减材料："
                f"{material_path}"
            ),
        ],
        "content_changes": {},
        "stable_metadata_changes": {},
        "activity_changes": {},
        "metric_changes": {},
        "similarities": {
            "skipped": True,
            "reason": (
                "same_day_material_without_"
                "successful_summary"
            ),
        },
        "current_hashes": (
            current_snapshot[
                "hashes"
            ]
        ),
    }


def validate_history_database(
    connection: sqlite3.Connection,
) -> None:
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'daily_repository_selections'
        """
    ).fetchone()

    if row is None:
        raise RuntimeError(
            "SQLite 中不存在 "
            "daily_repository_selections 表。"
        )


def query_previous_snapshot_dates(
    connection: sqlite3.Connection,
    *,
    full_name: str,
    current_snapshot_date: str,
) -> list[str]:
    rows = connection.execute(
        """
        SELECT DISTINCT snapshot_date
        FROM daily_repository_selections
        WHERE full_name = ? COLLATE NOCASE
          AND snapshot_date < ?
          AND snapshot_date IS NOT NULL
          AND TRIM(snapshot_date) <> ''
        ORDER BY snapshot_date DESC
        """,
        (
            full_name,
            current_snapshot_date,
        ),
    ).fetchall()

    return [
        str(
            row[0]
        ).strip()
        for row in rows
        if row[0]
    ]


def candidate_material_files_for_date(
    deep_root: Path,
    snapshot_date: str,
) -> list[Path]:
    date_directory = (
        deep_root
        / snapshot_date
    )

    candidates: list[
        Path
    ] = []

    latest_pointer = (
        date_directory
        / CURRENT_MANIFEST_NAME
    )

    if latest_pointer.is_file():
        candidates.append(
            latest_pointer
        )

    if date_directory.is_dir():
        collection_directories = sorted(
            (
                child
                for child
                in date_directory.iterdir()
                if child.is_dir()
            ),
            key=lambda child: (
                child.name
            ),
            reverse=True,
        )

        for collection_directory in (
            collection_directories
        ):
            material_path = (
                collection_directory
                / ANALYSIS_MATERIAL_NAME
            )

            if (
                material_path.is_file()
                and material_path
                not in candidates
            ):
                candidates.append(
                    material_path
                )

    return candidates


def find_previous_repository_from_sql(
    connection: sqlite3.Connection,
    *,
    deep_root: Path,
    full_name: str,
    current_snapshot_date: str,
    material_cache: dict[
        str,
        (
            tuple[
                dict[
                    str,
                    dict[str, Any],
                ],
                Path,
            ]
            | None
        ),
    ],
) -> dict[str, Any]:
    candidate_dates = (
        query_previous_snapshot_dates(
            connection,
            full_name=full_name,
            current_snapshot_date=(
                current_snapshot_date
            ),
        )
    )

    lookup_notes: list[
        str
    ] = []

    for historical_date in (
        candidate_dates
    ):
        source_candidates = (
            candidate_material_files_for_date(
                deep_root,
                historical_date,
            )
        )

        if not source_candidates:
            lookup_notes.append(
                f"{historical_date}: "
                "未找到历史粗缩减材料。"
            )

            continue

        for source_path in (
            source_candidates
        ):
            cache_key = str(
                source_path.resolve()
            )

            if cache_key in material_cache:
                loaded = (
                    material_cache[
                        cache_key
                    ]
                )

            else:
                try:
                    loaded = (
                        load_analysis_material(
                            source_path
                        )
                    )
                except (
                    OSError,
                    json.JSONDecodeError,
                    ValueError,
                    RuntimeError,
                ):
                    loaded = None

                material_cache[
                    cache_key
                ] = loaded

            if loaded is None:
                lookup_notes.append(
                    f"{historical_date}: "
                    f"无法读取 {source_path}。"
                )

                continue

            (
                records,
                actual_material_path,
            ) = loaded

            previous_record = (
                find_record_case_insensitive(
                    records,
                    full_name,
                )
            )

            if previous_record is None:
                continue

            return {
                "state": (
                    "cross_day_found"
                ),
                "found": True,
                "method": (
                    "sqlite_daily_"
                    "repository_selections"
                ),
                "history_exists_in_sql": (
                    True
                ),
                "candidate_snapshot_dates": (
                    candidate_dates
                ),
                "snapshot_date": (
                    historical_date
                ),
                "record": (
                    previous_record
                ),
                "source_pointer_or_file": (
                    str(source_path)
                ),
                "analysis_material_path": (
                    str(
                        actual_material_path
                    )
                ),
                "repository_llm_summaries_path": (
                    str(
                        actual_material_path
                        .parent
                        / REPOSITORY_SUMMARIES_NAME
                    )
                ),
                "lookup_notes": (
                    lookup_notes
                ),
            }

    return {
        "state": (
            "cross_day_not_found"
        ),
        "found": False,
        "method": (
            "sqlite_daily_"
            "repository_selections"
        ),
        "history_exists_in_sql": (
            bool(candidate_dates)
        ),
        "candidate_snapshot_dates": (
            candidate_dates
        ),
        "snapshot_date": None,
        "record": None,
        "source_pointer_or_file": (
            None
        ),
        "analysis_material_path": (
            None
        ),
        "repository_llm_summaries_path": (
            None
        ),
        "lookup_notes": (
            lookup_notes
        ),
    }


def main() -> None:
    args = parse_arguments()

    deep_root = Path(
        args.deep_root
    ).resolve()

    database_path = Path(
        args.database
    ).resolve()

    if args.date:
        snapshot_date = (
            date.fromisoformat(
                args.date
            ).isoformat()
        )

    else:
        snapshot_date = (
            datetime.now(
                ZoneInfo(
                    TIMEZONE_NAME
                )
            )
            .date()
            .isoformat()
        )

    current_manifest_path = (
        deep_root
        / snapshot_date
        / CURRENT_MANIFEST_NAME
    )

    if not current_manifest_path.is_file():
        raise FileNotFoundError(
            "没有找到当天深度采集清单：\n"
            f"{current_manifest_path}"
        )

    if not database_path.is_file():
        raise FileNotFoundError(
            "没有找到 GitHub 技术情报数据库：\n"
            f"{database_path}"
        )

    print(
        "正在读取当天仓库粗缩减材料：",
        current_manifest_path,
    )

    (
        current_records,
        current_analysis_material_path,
    ) = load_analysis_material(
        current_manifest_path
    )

    print(
        "当天粗缩减材料文件：",
        current_analysis_material_path,
    )

    print(
        "当天仓库数量：",
        len(current_records),
    )

    print(
        "历史定位数据库：",
        database_path,
    )

    material_cache: dict[
        str,
        (
            tuple[
                dict[
                    str,
                    dict[str, Any],
                ],
                Path,
            ]
            | None
        ),
    ] = {}

    summary_cache: dict[
        str,
        (
            dict[
                str,
                dict[str, Any],
            ]
            | None
        ),
    ] = {}

    decisions: list[
        dict[str, Any]
    ] = []

    connection = sqlite3.connect(
        database_path
    )

    try:
        validate_history_database(
            connection
        )

        for full_name in sorted(
            current_records,
            key=str.casefold,
        ):
            current_snapshot = (
                build_repository_snapshot(
                    full_name,
                    current_records[
                        full_name
                    ],
                )
            )

            same_day_result = (
                find_same_day_processing_state(
                    deep_root=deep_root,
                    full_name=full_name,
                    current_snapshot_date=(
                        snapshot_date
                    ),
                    current_analysis_material_path=(
                        current_analysis_material_path
                    ),
                    material_cache=(
                        material_cache
                    ),
                    summary_cache=(
                        summary_cache
                    ),
                )
            )

            if (
                same_day_result[
                    "state"
                ]
                == "processed"
            ):
                history_result = (
                    same_day_result
                )

                previous_snapshot_date = (
                    snapshot_date
                )

                decision = (
                    build_same_day_reuse_decision(
                        current_snapshot,
                        same_day_result,
                    )
                )

            elif (
                same_day_result[
                    "state"
                ]
                == "analysis_required"
            ):
                history_result = (
                    same_day_result
                )

                previous_snapshot_date = (
                    snapshot_date
                )

                decision = (
                    build_same_day_analysis_required_decision(
                        current_snapshot,
                        same_day_result,
                    )
                )

            else:
                history_result = (
                    find_previous_repository_from_sql(
                        connection,
                        deep_root=deep_root,
                        full_name=full_name,
                        current_snapshot_date=(
                            snapshot_date
                        ),
                        material_cache=(
                            material_cache
                        ),
                    )
                )

                previous_snapshot_date = (
                    history_result.get(
                        "snapshot_date"
                    )
                )

                previous_record = (
                    history_result.get(
                        "record"
                    )
                )

                previous_snapshot = None

                if isinstance(
                    previous_record,
                    dict,
                ):
                    previous_snapshot = (
                        build_repository_snapshot(
                            full_name,
                            previous_record,
                        )
                    )

                decision = (
                    compare_repository_snapshots(
                        current_snapshot,
                        previous_snapshot,
                        previous_snapshot_date,
                        history_exists_in_sql=(
                            bool(
                                history_result[
                                    "history_exists_in_sql"
                                ]
                            )
                        ),
                    )
                )

            decision[
                "history_lookup"
            ] = {
                "method": (
                    history_result.get(
                        "method"
                    )
                ),
                "database_path": (
                    str(database_path)
                ),
                "candidate_snapshot_dates": (
                    history_result.get(
                        "candidate_snapshot_dates",
                        [],
                    )
                ),
                "selected_snapshot_date": (
                    previous_snapshot_date
                ),
                "source_pointer_or_file": (
                    history_result.get(
                        "source_pointer_or_file"
                    )
                ),
                "analysis_material_path": (
                    history_result.get(
                        "analysis_material_path"
                    )
                ),
                "repository_llm_summaries_path": (
                    history_result.get(
                        "repository_llm_summaries_path"
                    )
                ),
                "lookup_notes": (
                    history_result.get(
                        "lookup_notes",
                        [],
                    )
                ),
            }

            decisions.append(
                decision
            )

            print(
                f"[{decision['change_status']}] "
                f"{full_name} "
                f"→ {decision['llm_action']}"
            )

            if previous_snapshot_date:
                print(
                    "  历史日期：",
                    previous_snapshot_date,
                )

            if history_result.get(
                "analysis_material_path"
            ):
                print(
                    "  历史材料：",
                    history_result[
                        "analysis_material_path"
                    ],
                )

            if history_result.get(
                "repository_llm_summaries_path"
            ):
                print(
                    "  历史摘要：",
                    history_result[
                        "repository_llm_summaries_path"
                    ],
                )

    finally:
        connection.close()

    status_counts: dict[
        str,
        int,
    ] = {}

    llm_required_count = 0

    for decision in decisions:
        status = decision[
            "change_status"
        ]

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

        if decision[
            "llm_should_run"
        ]:
            llm_required_count += 1

    output_data = {
        "snapshot_date": (
            snapshot_date
        ),

        "generated_at": (
            datetime.now(
                ZoneInfo(
                    TIMEZONE_NAME
                )
            ).isoformat()
        ),

        "source_manifest": (
            str(
                current_manifest_path
            )
        ),

        "source_analysis_material": (
            str(
                current_analysis_material_path
            )
        ),

        "history_database": (
            str(database_path)
        ),

        "history_lookup_policy": {
            "same_day_rule": (
                "先检查当天其他 collection 的 "
                "repository_llm_summaries.json；"
                "仅当 full_name 匹配且 "
                "status=success 时，"
                "判定 SAME_DAY_REUSED。"
                "若当天只有粗缩减材料而无成功摘要，"
                "判定 SAME_DAY_ANALYSIS_REQUIRED。"
            ),

            "cross_day_sql_table": (
                "daily_repository_selections"
            ),

            "identity_key": (
                "full_name"
            ),

            "cross_day_date_rule": (
                "snapshot_date < current_date, "
                "ORDER BY snapshot_date DESC"
            ),

            "comparison_source": (
                ANALYSIS_MATERIAL_NAME
            ),
        },

        "comparison_policy": {
            "readme_minor_similarity_threshold": (
                README_MINOR_SIMILARITY_THRESHOLD
            ),

            "readme_major_similarity_threshold": (
                README_MAJOR_SIMILARITY_THRESHOLD
            ),

            "readme_major_length_ratio": (
                README_MAJOR_LENGTH_RATIO
            ),

            "comparison_level": (
                "规则粗缩减后的 README、Release、Issue "
                "及保留元数据"
            ),
        },

        "summary": {
            "repository_count": (
                len(decisions)
            ),

            "llm_required_count": (
                llm_required_count
            ),

            "llm_reuse_count": (
                len(decisions)
                - llm_required_count
            ),

            "status_counts": (
                status_counts
            ),
        },

        "repositories": (
            decisions
        ),
    }

    output_path = (
        deep_root
        / snapshot_date
        / OUTPUT_FILE_NAME
    )

    save_json(
        output_path,
        output_data,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "仓库变化检测完成"
    )

    print(
        "日期：",
        snapshot_date,
    )

    print(
        "项目总数：",
        len(decisions),
    )

    print(
        "需要调用 LLM：",
        llm_required_count,
    )

    print(
        "复用历史摘要：",
        (
            len(decisions)
            - llm_required_count
        ),
    )

    print(
        "状态统计：",
        status_counts,
    )

    print(
        "输出文件：",
        output_path,
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()