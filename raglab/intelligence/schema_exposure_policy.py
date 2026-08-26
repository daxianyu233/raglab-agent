"""
GitHub Intelligence 数据库 Schema 暴露策略。

这个模块不负责 LLM 查询，也不属于 Agent Tool。

职责：

1. 创建 schema_exposure_policy 元数据表；
2. 为业务 Table / View 登记暴露策略；
3. 区分：
   - queryable：正常业务查询；
   - observable：运行状态类查询；
   - internal：内部实现数据；
   - sensitive：敏感数据；
4. 支持字段级隐藏；
5. 支持查询优先级，用于以后给 LLM 提供选表提示；
6. 未登记的数据对象默认不暴露。

核心原则：

    数据库真实 Schema 决定“有什么”。

    schema_exposure_policy 决定
    “哪些可以给 Agent 看”。

    Agent / LLM 永远不能修改本策略。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# 项目路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT
    / "storage"
    / "intelligence"
    / "github_intelligence.sqlite3"
)


POLICY_TABLE_NAME = (
    "schema_exposure_policy"
)


# ============================================================
# 策略常量
# ============================================================


VISIBILITY_QUERYABLE = "queryable"
VISIBILITY_OBSERVABLE = "observable"
VISIBILITY_INTERNAL = "internal"
VISIBILITY_SENSITIVE = "sensitive"


VALID_VISIBILITIES = {
    VISIBILITY_QUERYABLE,
    VISIBILITY_OBSERVABLE,
    VISIBILITY_INTERNAL,
    VISIBILITY_SENSITIVE,
}


PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"
PRIORITY_NONE = "none"


VALID_QUERY_PRIORITIES = {
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    PRIORITY_NONE,
}


EXPOSED_VISIBILITIES = {
    VISIBILITY_QUERYABLE,
    VISIBILITY_OBSERVABLE,
}


# ============================================================
# 基础工具
# ============================================================


def utc_now_text() -> str:
    """
    返回 UTC ISO 时间。
    """

    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )


def normalize_database_path(
    database_path: Path,
) -> Path:
    """
    规范化数据库路径。
    """

    path = Path(
        database_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            "GitHub Intelligence SQLite "
            f"数据库不存在：{path}"
        )

    return path


def normalize_object_name(
    object_name: str,
) -> str:
    """
    规范化 SQLite 对象名。
    """

    normalized = str(
        object_name
    ).strip()

    if not normalized:
        raise ValueError(
            "object_name 不能为空。"
        )

    return normalized


def normalize_object_type(
    object_type: str,
) -> str:
    """
    规范化对象类型。
    """

    normalized = str(
        object_type
    ).strip().lower()

    if normalized not in {
        "table",
        "view",
    }:
        raise ValueError(
            "object_type 只能是 "
            "'table' 或 'view'。"
        )

    return normalized


def normalize_visibility(
    visibility: str,
) -> str:
    """
    校验访问级别。
    """

    normalized = str(
        visibility
    ).strip().lower()

    if (
        normalized
        not in VALID_VISIBILITIES
    ):
        raise ValueError(
            "visibility 无效："
            f"{visibility!r}；"
            "允许值："
            + ", ".join(
                sorted(
                    VALID_VISIBILITIES
                )
            )
        )

    return normalized


def normalize_query_priority(
    query_priority: str,
) -> str:
    """
    校验查询优先级。
    """

    normalized = str(
        query_priority
    ).strip().lower()

    if (
        normalized
        not in VALID_QUERY_PRIORITIES
    ):
        raise ValueError(
            "query_priority 无效："
            f"{query_priority!r}；"
            "允许值："
            + ", ".join(
                sorted(
                    VALID_QUERY_PRIORITIES
                )
            )
        )

    return normalized


def normalize_hidden_columns(
    hidden_columns: Iterable[str] | None,
) -> list[str]:
    """
    规范化隐藏字段列表。
    """

    if hidden_columns is None:
        return []

    result: list[str] = []

    seen: set[str] = set()

    for item in hidden_columns:
        column_name = str(
            item
        ).strip()

        if not column_name:
            continue

        key = column_name.casefold()

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            column_name
        )

    return result


# ============================================================
# Policy Table
# ============================================================


def ensure_schema_exposure_policy_table(
    connection: sqlite3.Connection,
) -> None:
    """
    创建 Schema Exposure Policy 表。

    注意：

    这是确定性业务代码执行的数据库修改，
    不对 LLM 暴露。
    """

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {POLICY_TABLE_NAME}
        (
            object_name TEXT PRIMARY KEY,

            object_type TEXT NOT NULL
                CHECK (
                    object_type
                    IN ('table', 'view')
                ),

            visibility TEXT NOT NULL
                CHECK (
                    visibility IN (
                        'queryable',
                        'observable',
                        'internal',
                        'sensitive'
                    )
                ),

            query_priority TEXT NOT NULL
                DEFAULT 'medium'
                CHECK (
                    query_priority IN (
                        'high',
                        'medium',
                        'low',
                        'none'
                    )
                ),

            description TEXT,

            hidden_columns_json TEXT
                NOT NULL
                DEFAULT '[]',

            created_at TEXT NOT NULL,

            updated_at TEXT NOT NULL
        )
        """
    )

    # Policy 表自身永远属于 internal。
    #
    # 这样后续 Resolver 即使读取这张表，
    # 也绝不能把它暴露给 LLM。
    register_schema_object(
        connection,
        object_name=(
            POLICY_TABLE_NAME
        ),
        object_type="table",
        visibility=(
            VISIBILITY_INTERNAL
        ),
        query_priority=(
            PRIORITY_NONE
        ),
        description=(
            "Agent Schema 暴露策略内部元数据。"
            "不得向 LLM 暴露或允许 Agent 查询。"
        ),
        hidden_columns=[],
    )


# ============================================================
# 策略登记
# ============================================================


def register_schema_object(
    connection: sqlite3.Connection,
    *,
    object_name: str,
    object_type: str = "table",
    visibility: str,
    query_priority: str = PRIORITY_MEDIUM,
    description: str = "",
    hidden_columns: Iterable[str] | None = None,
) -> None:
    """
    登记或更新一张 Table / View 的暴露策略。

    使用 UPSERT：

    已存在：
        更新策略。

    不存在：
        新增策略。
    """

    normalized_name = (
        normalize_object_name(
            object_name
        )
    )

    normalized_type = (
        normalize_object_type(
            object_type
        )
    )

    normalized_visibility = (
        normalize_visibility(
            visibility
        )
    )

    normalized_priority = (
        normalize_query_priority(
            query_priority
        )
    )

    normalized_hidden_columns = (
        normalize_hidden_columns(
            hidden_columns
        )
    )

    now = utc_now_text()

    hidden_columns_json = (
        json.dumps(
            normalized_hidden_columns,
            ensure_ascii=False,
        )
    )

    connection.execute(
        f"""
        INSERT INTO {POLICY_TABLE_NAME}
        (
            object_name,
            object_type,
            visibility,
            query_priority,
            description,
            hidden_columns_json,
            created_at,
            updated_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(object_name)
        DO UPDATE SET

            object_type =
                excluded.object_type,

            visibility =
                excluded.visibility,

            query_priority =
                excluded.query_priority,

            description =
                excluded.description,

            hidden_columns_json =
                excluded.hidden_columns_json,

            updated_at =
                excluded.updated_at
        """,
        (
            normalized_name,
            normalized_type,
            normalized_visibility,
            normalized_priority,
            str(
                description
                or ""
            ).strip(),
            hidden_columns_json,
            now,
            now,
        ),
    )


# ============================================================
# 策略读取
# ============================================================


def load_schema_exposure_policies(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    """
    读取当前全部暴露策略。

    返回：

        {
            "repositories": {
                ...
            }
        }

    key 使用 casefold，
    便于 SQLite 名称大小写兼容。
    """

    rows = connection.execute(
        f"""
        SELECT
            object_name,
            object_type,
            visibility,
            query_priority,
            description,
            hidden_columns_json,
            created_at,
            updated_at

        FROM {POLICY_TABLE_NAME}

        ORDER BY object_name
        """
    ).fetchall()

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in rows:

        hidden_columns_text = str(
            row["hidden_columns_json"]
            or "[]"
        )

        try:
            hidden_columns_raw = (
                json.loads(
                    hidden_columns_text
                )
            )

        except json.JSONDecodeError:
            hidden_columns_raw = []

        if not isinstance(
            hidden_columns_raw,
            list,
        ):
            hidden_columns_raw = []

        hidden_columns = (
            normalize_hidden_columns(
                hidden_columns_raw
            )
        )

        object_name = str(
            row["object_name"]
        )

        result[
            object_name.casefold()
        ] = {
            "object_name": (
                object_name
            ),

            "object_type": str(
                row["object_type"]
            ),

            "visibility": str(
                row["visibility"]
            ),

            "query_priority": str(
                row["query_priority"]
            ),

            "description": str(
                row["description"]
                or ""
            ),

            "hidden_columns": (
                hidden_columns
            ),

            "created_at": (
                row["created_at"]
            ),

            "updated_at": (
                row["updated_at"]
            ),
        }

    return result


def policy_is_exposed(
    policy: dict[str, Any],
) -> bool:
    """
    判断策略是否允许 Agent 查询。
    """

    visibility = str(
        policy.get(
            "visibility",
            "",
        )
    ).strip().lower()

    return (
        visibility
        in EXPOSED_VISIBILITIES
    )


# ============================================================
# 当前 GitHub Intelligence 初始策略
# ============================================================


def register_existing_github_schema_policy(
    connection: sqlite3.Connection,
) -> None:
    """
    一次性为当前已经存在的 GitHub Intelligence 表
    建立初始暴露策略。

    以后新增业务表时，不需要修改 SQL Tool。

    应在创建新表的确定性业务代码中直接调用：

        register_schema_object(...)

    完成策略登记。
    """

    # --------------------------------------------------------
    # 核心 GitHub 项目数据
    # --------------------------------------------------------

    register_schema_object(
        connection,
        object_name="repositories",
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_HIGH
        ),
        description=(
            "GitHub 仓库稳定身份及基础元数据。"
        ),
    )

    register_schema_object(
        connection,
        object_name=(
            "repository_snapshots"
        ),
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_HIGH
        ),
        description=(
            "GitHub 仓库每日指标和活跃度快照，"
            "包括 stars、forks、issues 等。"
        ),
    )

    register_schema_object(
        connection,
        object_name=(
            "repository_discoveries"
        ),
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_MEDIUM
        ),
        description=(
            "GitHub 仓库被发现时的来源、"
            "搜索查询和结果排名。"
        ),
    )

    register_schema_object(
        connection,
        object_name=(
            "daily_repository_selections"
        ),
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_HIGH
        ),
        description=(
            "每日 GitHub 技术情报入选项目。"
        ),
    )

    # --------------------------------------------------------
    # LLM 项目分析
    # --------------------------------------------------------

    register_schema_object(
        connection,
        object_name=(
            "repository_llm_summaries"
        ),
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_HIGH
        ),
        description=(
            "GitHub 项目 LLM 结构化分析结果。"
        ),

        # 本地文件路径属于内部实现细节。
        hidden_columns=[
            "source_file",
        ],
    )

    # --------------------------------------------------------
    # 跨项目热点
    # --------------------------------------------------------

    register_schema_object(
        connection,
        object_name=(
            "daily_hotspot_topics"
        ),
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_HIGH
        ),
        description=(
            "每日跨项目技术热点主题。"
        ),
    )

    register_schema_object(
        connection,
        object_name=(
            "daily_hotspot_reports"
        ),
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_MEDIUM
        ),
        description=(
            "每日完整 GitHub 热点报告。"
        ),

        # 不需要向 LLM 暴露本地文件系统路径。
        hidden_columns=[
            "source_summaries_path",
            "full_markdown_path",
            "compact_markdown_path",
        ],
    )

    register_schema_object(
        connection,
        object_name=(
            "daily_intelligence_briefs"
        ),
        visibility=(
            VISIBILITY_QUERYABLE
        ),
        query_priority=(
            PRIORITY_MEDIUM
        ),
        description=(
            "每日 GitHub 技术情报精简摘要。"
        ),
        hidden_columns=[
            "markdown_path",
        ],
    )

    # --------------------------------------------------------
    # Pipeline 内部状态
    # --------------------------------------------------------

    register_schema_object(
        connection,
        object_name=(
            "repository_processing_state"
        ),
        visibility=(
            VISIBILITY_INTERNAL
        ),
        query_priority=(
            PRIORITY_NONE
        ),
        description=(
            "仓库增量处理内部状态。"
            "仅供 Pipeline 使用。"
        ),
    )

    register_schema_object(
        connection,
        object_name=(
            "collection_runs"
        ),
        visibility=(
            VISIBILITY_INTERNAL
        ),
        query_priority=(
            PRIORITY_NONE
        ),
        description=(
            "GitHub 采集 Pipeline 内部运行记录。"
            "包含运行目录和错误信息等内部实现细节。"
        ),
    )


# ============================================================
# 初始化入口
# ============================================================


def initialize_schema_exposure_policy(
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
) -> None:
    """
    初始化 Schema Exposure Policy。

    这是确定性数据库初始化操作。

    只需执行一次；
    后续重复执行也是幂等的。
    """

    path = normalize_database_path(
        database_path
    )

    connection = sqlite3.connect(
        path
    )

    connection.row_factory = (
        sqlite3.Row
    )

    try:
        ensure_schema_exposure_policy_table(
            connection
        )

        register_existing_github_schema_policy(
            connection
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# 调试输出
# ============================================================


def render_schema_exposure_policy(
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
) -> str:
    """
    打印当前暴露策略，方便人工检查。
    """

    path = normalize_database_path(
        database_path
    )

    connection = sqlite3.connect(
        path
    )

    connection.row_factory = (
        sqlite3.Row
    )

    try:
        policies = (
            load_schema_exposure_policies(
                connection
            )
        )

    finally:
        connection.close()

    lines = [
        "GitHub Intelligence Schema Exposure Policy",
        "",
    ]

    for policy in sorted(
        policies.values(),
        key=lambda item: (
            str(
                item[
                    "object_name"
                ]
            ).casefold()
        ),
    ):

        hidden_columns = (
            policy.get(
                "hidden_columns",
                [],
            )
        )

        lines.append(
            f"{policy['object_name']}"
        )

        lines.append(
            "  visibility: "
            f"{policy['visibility']}"
        )

        lines.append(
            "  priority: "
            f"{policy['query_priority']}"
        )

        if policy.get(
            "description"
        ):
            lines.append(
                "  description: "
                f"{policy['description']}"
            )

        if hidden_columns:
            lines.append(
                "  hidden_columns: "
                + ", ".join(
                    hidden_columns
                )
            )

        lines.append("")

    return "\n".join(
        lines
    ).strip()