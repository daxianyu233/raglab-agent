"""
GitHub Intelligence 只读 Text-to-SQL Tool。

职责：

1. 动态读取真实 SQLite Schema；
2. 读取 schema_exposure_policy；
3. 根据 Exposure Policy 生成 Filtered Schema；
4. 只把允许暴露的 Table / View / Column 提供给 LLM；
5. 接收 LLM 生成的只读 SQL；
6. 使用 SQLite Authorizer 按同一份 Filtered Schema
   再次限制真实数据库访问权限；
7. 使用 SQLite Read-Only Connection、
   query_only、超时、行数和字符数限制
   构成多层安全边界。

核心原则：

    Real Schema
        决定数据库实际上有什么。

    Exposure Policy
        决定 Agent 可以看到什么。

    Filtered Schema
        同时决定：
        1. LLM 能看到什么；
        2. SQL 实际能读取什么。

    LLM 永远没有数据库写权限。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import (
    BaseTool,
    tool,
)

from pydantic import (
    BaseModel,
    Field,
)

from raglab.intelligence.schema_exposure_policy import (
    DEFAULT_DATABASE_PATH as DEFAULT_SCHEMA_DATABASE_PATH,
    POLICY_TABLE_NAME,
    load_schema_exposure_policies,
    policy_is_exposed,
)


# ============================================================
# 数据库路径
# ============================================================


# 保留原来的常量名称，
# 避免 tools.py 等现有模块需要修改。
DEFAULT_DATABASE_PATH = (
    DEFAULT_SCHEMA_DATABASE_PATH
)


# ============================================================
# 安全与输出限制
# ============================================================


MAX_SQL_CHARACTERS = 12_000


DEFAULT_MAX_ROWS = 100


DEFAULT_MAX_OUTPUT_CHARACTERS = (
    20_000
)


DEFAULT_QUERY_TIMEOUT_SECONDS = (
    3.0
)


PROGRESS_HANDLER_STEPS = 1_000


# ============================================================
# Tool Input
# ============================================================


class GitHubIntelligenceSQLInput(
    BaseModel
):
    """
    GitHub Intelligence SQL Tool 输入。
    """

    sql: str = Field(
        min_length=1,
        max_length=MAX_SQL_CHARACTERS,
        description=(
            "针对 GitHub Intelligence SQLite "
            "生成的只读 SQL 查询。"
            "只能使用 Agent 当前可见 Schema "
            "中的 Table、View 和 Column。"
            "只允许 SELECT 或 WITH ... SELECT。"
            "不要使用 SELECT *，应明确列出"
            "需要查询的可见字段。"
            "禁止 INSERT、UPDATE、DELETE、DROP、"
            "ALTER、CREATE、ATTACH、PRAGMA 等"
            "任何写入或数据库管理操作。"
        ),
    )


# ============================================================
# 基础辅助函数
# ============================================================


def normalize_database_path(
    database_path: Path,
) -> Path:
    """
    规范化数据库路径并检查存在性。
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


def quote_identifier(
    identifier: str,
) -> str:
    """
    安全引用 SQLite Identifier。
    """

    escaped = str(
        identifier
    ).replace(
        '"',
        '""',
    )

    return (
        f'"{escaped}"'
    )


def open_read_only_connection(
    database_path: Path,
) -> sqlite3.Connection:
    """
    以 SQLite 真正的只读模式打开数据库。
    """

    path = normalize_database_path(
        database_path
    )

    database_uri = (
        path.as_uri()
        + "?mode=ro"
    )

    connection = sqlite3.connect(
        database_uri,
        uri=True,
        timeout=1.0,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    # 第二层只读保护。
    connection.execute(
        "PRAGMA query_only = ON"
    )

    return connection


# ============================================================
# 真实 Schema Discovery
# ============================================================


def inspect_real_database_objects(
    connection: sqlite3.Connection,
) -> list[dict[str, str]]:
    """
    动态发现 SQLite 当前实际存在的
   业务 Table / View。

    这里只回答：

        “数据库里实际上有什么？”

    不处理 Agent 权限。
    """

    rows = connection.execute(
        """
        SELECT
            name,
            type

        FROM sqlite_schema

        WHERE type IN (
            'table',
            'view'
        )

          AND name NOT LIKE 'sqlite_%'

        ORDER BY
            type,
            name
        """
    ).fetchall()

    objects: list[
        dict[str, str]
    ] = []

    for row in rows:

        object_name = str(
            row["name"]
        ).strip()

        object_type = str(
            row["type"]
        ).strip().lower()

        if not object_name:
            continue

        objects.append(
            {
                "name": object_name,
                "type": object_type,
            }
        )

    return objects


def inspect_object_columns(
    connection: sqlite3.Connection,
    *,
    object_name: str,
) -> list[dict[str, Any]]:
    """
    读取一张 Table / View 的真实字段。
    """

    quoted_name = quote_identifier(
        object_name
    )

    rows = connection.execute(
        f"PRAGMA table_info({quoted_name})"
    ).fetchall()

    columns: list[
        dict[str, Any]
    ] = []

    for row in rows:

        columns.append(
            {
                "name": (
                    str(
                        row["name"]
                    )
                ),

                "type": (
                    str(
                        row["type"]
                        or "ANY"
                    )
                ),

                "not_null": bool(
                    row["notnull"]
                ),

                "default": (
                    row["dflt_value"]
                ),

                "primary_key_position": (
                    int(
                        row["pk"]
                        or 0
                    )
                ),
            }
        )

    return columns


def inspect_object_foreign_keys(
    connection: sqlite3.Connection,
    *,
    object_name: str,
) -> list[dict[str, str]]:
    """
    读取一张表的真实 Foreign Key 信息。
    """

    quoted_name = quote_identifier(
        object_name
    )

    rows = connection.execute(
        "PRAGMA foreign_key_list"
        f"({quoted_name})"
    ).fetchall()

    foreign_keys: list[
        dict[str, str]
    ] = []

    for row in rows:

        foreign_keys.append(
            {
                "from": str(
                    row["from"]
                    or ""
                ),

                "to_table": str(
                    row["table"]
                    or ""
                ),

                "to_column": str(
                    row["to"]
                    or ""
                ),
            }
        )

    return foreign_keys


# ============================================================
# Exposure Policy + Schema Resolver
# ============================================================


def resolve_exposed_database_schema(
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
) -> dict[str, Any]:
    """
    生成 Agent 真正可见、可查询的 Filtered Schema。

    流程：

        Real SQLite Schema
                +
        schema_exposure_policy
                ↓
         Filtered Schema

    规则：

    1. queryable / observable：
       可以暴露；

    2. internal / sensitive：
       不暴露；

    3. Policy 中没有登记：
       默认拒绝；

    4. hidden_columns：
       从暴露 Schema 中删除；

    5. Foreign Key 如果涉及不可见对象
       或不可见字段，也不会暴露。
    """

    connection = (
        open_read_only_connection(
            database_path
        )
    )

    try:

        # ----------------------------------------------------
        # 真实数据库对象
        # ----------------------------------------------------

        real_objects = (
            inspect_real_database_objects(
                connection
            )
        )

        # ----------------------------------------------------
        # 检查 Policy Table
        # ----------------------------------------------------

        real_object_names = {
            str(
                item["name"]
            ).casefold()
            for item in real_objects
        }

        if (
            POLICY_TABLE_NAME.casefold()
            not in real_object_names
        ):
            raise RuntimeError(
                "数据库尚未初始化 "
                "schema_exposure_policy。\n"
                "请先执行 "
                "initialize_schema_exposure_policy()。"
            )

        # ----------------------------------------------------
        # 读取 Exposure Policy
        # ----------------------------------------------------

        policies = (
            load_schema_exposure_policies(
                connection
            )
        )

        # ----------------------------------------------------
        # 第一轮：
        # 找到真正允许暴露的对象
        # ----------------------------------------------------

        exposed_object_names: set[
            str
        ] = set()

        exposed_candidates: list[
            tuple[
                dict[str, str],
                dict[str, Any],
            ]
        ] = []

        for real_object in real_objects:

            object_name = str(
                real_object["name"]
            )

            object_type = str(
                real_object["type"]
            ).lower()

            policy = policies.get(
                object_name.casefold()
            )

            # Default Deny：
            #
            # 数据库中新出现但尚未登记 Policy 的表，
            # 不会自动暴露。
            if policy is None:
                continue

            if not policy_is_exposed(
                policy
            ):
                continue

            policy_object_type = str(
                policy.get(
                    "object_type",
                    "",
                )
            ).strip().lower()

            # Policy 与真实对象类型不一致时，
            # 为安全起见拒绝暴露。
            if (
                policy_object_type
                != object_type
            ):
                continue

            exposed_object_names.add(
                object_name.casefold()
            )

            exposed_candidates.append(
                (
                    real_object,
                    policy,
                )
            )

        # ----------------------------------------------------
        # 第二轮：
        # 字段过滤 + Foreign Key 过滤
        # ----------------------------------------------------

        resolved_objects: list[
            dict[str, Any]
        ] = []

        # 先缓存各暴露对象允许的字段，
        # 后面 Foreign Key 过滤会使用。
        allowed_columns_by_object: dict[
            str,
            set[str],
        ] = {}

        raw_columns_by_object: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for (
            real_object,
            policy,
        ) in exposed_candidates:

            object_name = str(
                real_object["name"]
            )

            hidden_columns = {
                str(
                    column_name
                ).casefold()

                for column_name in (
                    policy.get(
                        "hidden_columns",
                        [],
                    )
                    or []
                )
            }

            raw_columns = (
                inspect_object_columns(
                    connection,
                    object_name=(
                        object_name
                    ),
                )
            )

            visible_columns = [
                column
                for column in raw_columns
                if str(
                    column["name"]
                ).casefold()
                not in hidden_columns
            ]

            # 如果一张表没有任何可见字段，
            # 就没有向 Agent 暴露的价值。
            if not visible_columns:
                continue

            object_key = (
                object_name.casefold()
            )

            raw_columns_by_object[
                object_key
            ] = visible_columns

            allowed_columns_by_object[
                object_key
            ] = {
                str(
                    column["name"]
                ).casefold()

                for column
                in visible_columns
            }

        # ----------------------------------------------------
        # 第三轮：
        # 构造最终对象信息
        # ----------------------------------------------------

        for (
            real_object,
            policy,
        ) in exposed_candidates:

            object_name = str(
                real_object["name"]
            )

            object_key = (
                object_name.casefold()
            )

            visible_columns = (
                raw_columns_by_object.get(
                    object_key
                )
            )

            if not visible_columns:
                continue

            raw_foreign_keys = (
                inspect_object_foreign_keys(
                    connection,
                    object_name=(
                        object_name
                    ),
                )
            )

            visible_foreign_keys: list[
                dict[str, str]
            ] = []

            for foreign_key in (
                raw_foreign_keys
            ):

                from_column = str(
                    foreign_key.get(
                        "from",
                        "",
                    )
                )

                target_table = str(
                    foreign_key.get(
                        "to_table",
                        "",
                    )
                )

                target_column = str(
                    foreign_key.get(
                        "to_column",
                        "",
                    )
                )

                if (
                    from_column.casefold()
                    not in
                    allowed_columns_by_object.get(
                        object_key,
                        set(),
                    )
                ):
                    continue

                target_key = (
                    target_table.casefold()
                )

                if (
                    target_key
                    not in
                    allowed_columns_by_object
                ):
                    continue

                if (
                    target_column
                    and target_column.casefold()
                    not in
                    allowed_columns_by_object.get(
                        target_key,
                        set(),
                    )
                ):
                    continue

                visible_foreign_keys.append(
                    foreign_key
                )

            resolved_objects.append(
                {
                    "name": object_name,

                    "type": str(
                        real_object[
                            "type"
                        ]
                    ),

                    "visibility": str(
                        policy.get(
                            "visibility",
                            "",
                        )
                    ),

                    "query_priority": str(
                        policy.get(
                            "query_priority",
                            "medium",
                        )
                    ),

                    "description": str(
                        policy.get(
                            "description",
                            "",
                        )
                        or ""
                    ),

                    "columns": (
                        visible_columns
                    ),

                    "foreign_keys": (
                        visible_foreign_keys
                    ),
                }
            )

        # ----------------------------------------------------
        # 优先级排序
        # ----------------------------------------------------

        priority_order = {
            "high": 0,
            "medium": 1,
            "low": 2,
            "none": 3,
        }

        resolved_objects.sort(
            key=lambda item: (
                priority_order.get(
                    str(
                        item.get(
                            "query_priority",
                            "medium",
                        )
                    ),
                    99,
                ),

                str(
                    item.get(
                        "name",
                        "",
                    )
                ).casefold(),
            )
        )

        return {
            "database_path": str(
                Path(
                    database_path
                ).resolve()
            ),

            "real_object_count": len(
                real_objects
            ),

            "policy_count": len(
                policies
            ),

            "exposed_object_count": len(
                resolved_objects
            ),

            "objects": (
                resolved_objects
            ),
        }

    finally:
        connection.close()


# ============================================================
# 兼容旧接口
# ============================================================


def inspect_database_schema(
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
) -> dict[str, Any]:
    """
    返回 Agent 可见的 Filtered Schema。

    保留原函数名，
    避免其他代码需要同步修改。
    """

    return (
        resolve_exposed_database_schema(
            database_path
        )
    )


# ============================================================
# Schema Render
# ============================================================


def render_database_schema(
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
) -> str:
    """
    将 Filtered Schema 渲染成适合 LLM 阅读的文本。

    注意：

    此处绝不会显示：

    - internal 对象；
    - sensitive 对象；
    - 未登记对象；
    - hidden_columns。
    """

    schema = (
        resolve_exposed_database_schema(
            database_path
        )
    )

    objects = schema.get(
        "objects",
        [],
    )

    lines: list[str] = [
        "GitHub Intelligence SQLite Schema",
        "",
        (
            "Only the following database objects "
            "and columns are available to the Agent."
        ),
        "",
    ]

    if not objects:
        lines.append(
            "当前没有允许 Agent 查询的数据库对象。"
        )

        return "\n".join(
            lines
        ).strip()

    for database_object in objects:

        object_name = str(
            database_object.get(
                "name",
                "",
            )
        )

        object_type = str(
            database_object.get(
                "type",
                "table",
            )
        )

        visibility = str(
            database_object.get(
                "visibility",
                "",
            )
        )

        priority = str(
            database_object.get(
                "query_priority",
                "medium",
            )
        )

        description = str(
            database_object.get(
                "description",
                "",
            )
            or ""
        )

        lines.append(
            f"{object_type.upper()} "
            f"{object_name}"
        )

        lines.append(
            "  visibility: "
            f"{visibility}"
        )

        lines.append(
            "  query_priority: "
            f"{priority}"
        )

        if description:
            lines.append(
                "  description: "
                f"{description}"
            )

        columns = (
            database_object.get(
                "columns",
                [],
            )
        )

        for column in columns:

            column_name = str(
                column.get(
                    "name",
                    "",
                )
            )

            column_type = str(
                column.get(
                    "type",
                    "ANY",
                )
            )

            suffixes: list[str] = []

            if column.get(
                "primary_key_position"
            ):
                suffixes.append(
                    "PRIMARY KEY"
                )

            if column.get(
                "not_null"
            ):
                suffixes.append(
                    "NOT NULL"
                )

            suffix = (
                " "
                + " ".join(
                    suffixes
                )
                if suffixes
                else ""
            )

            lines.append(
                f"  - {column_name}: "
                f"{column_type}"
                f"{suffix}"
            )

        foreign_keys = (
            database_object.get(
                "foreign_keys",
                [],
            )
        )

        if foreign_keys:

            lines.append(
                "  Foreign Keys:"
            )

            for foreign_key in (
                foreign_keys
            ):
                lines.append(
                    "    - "
                    f"{foreign_key['from']} "
                    "-> "
                    f"{foreign_key['to_table']}."
                    f"{foreign_key['to_column']}"
                )

        lines.append("")

    lines.extend(
        [
            "SQL Rules:",
            "",
            "- 只能查询上面明确展示的 Table / View。",
            "- 只能读取上面明确展示的 Column。",
            "- 未显示的数据库对象默认不可访问。",
            "- 未显示的字段默认不可访问。",
            "- 不要使用 SELECT *；应明确列出需要的字段。",
            "- 仅允许 SELECT 或 WITH ... SELECT。",
            "- 不允许修改数据库。",
            "- 数量、日期、排序、聚合、历史记录等精确问题优先使用 SQL。",
            "- 项目技术内容、技术方案、摘要解释和趋势语义优先使用 GitHub RAG。",
        ]
    )

    return "\n".join(
        lines
    ).strip()


# ============================================================
# SQL 文本基础校验
# ============================================================


def remove_leading_sql_comments(
    sql: str,
) -> str:
    """
    删除 SQL 开头的注释。
    """

    remaining = str(
        sql
    ).lstrip()

    while remaining:

        # ----------------------------------------------------
        # -- comment
        # ----------------------------------------------------

        if remaining.startswith(
            "--"
        ):

            newline_index = (
                remaining.find(
                    "\n"
                )
            )

            if newline_index < 0:
                return ""

            remaining = (
                remaining[
                    newline_index + 1:
                ]
                .lstrip()
            )

            continue

        # ----------------------------------------------------
        # /* comment */
        # ----------------------------------------------------

        if remaining.startswith(
            "/*"
        ):

            end_index = (
                remaining.find(
                    "*/",
                    2,
                )
            )

            if end_index < 0:
                return ""

            remaining = (
                remaining[
                    end_index + 2:
                ]
                .lstrip()
            )

            continue

        break

    return remaining


def validate_read_only_sql(
    sql: str,
) -> str:
    """
    第一层 SQL 文本校验。

    注意：

    这里只是快速拒绝明显写操作。

    真正的数据库权限边界仍然是：

        Filtered Schema
        +
        SQLite Authorizer
        +
        mode=ro
        +
        query_only
    """

    normalized = str(
        sql
    ).strip()

    if not normalized:
        raise ValueError(
            "SQL 不能为空。"
        )

    if (
        len(normalized)
        > MAX_SQL_CHARACTERS
    ):
        raise ValueError(
            "SQL 长度超过限制："
            f"{MAX_SQL_CHARACTERS}"
        )

    executable_sql = (
        remove_leading_sql_comments(
            normalized
        )
    )

    if not executable_sql:
        raise ValueError(
            "SQL 不包含可执行语句。"
        )

    first_word = (
        executable_sql
        .split(
            None,
            1,
        )[0]
        .upper()
    )

    if first_word not in {
        "SELECT",
        "WITH",
    }:
        raise ValueError(
            "只允许 SELECT 或 "
            "WITH ... SELECT 查询。"
        )

    return normalized


# ============================================================
# Authorizer 权限模型
# ============================================================


def build_authorizer(
    filtered_schema: dict[str, Any],
):
    """
    根据 Filtered Schema 创建 SQLite Authorizer。

    这是整个 Exposure Policy 真正的执行层。

    LLM 能看到什么：
        Filtered Schema

    SQLite 能读取什么：
        同一份 Filtered Schema
    """

    # --------------------------------------------------------
    # 构造允许访问的：
    #
    # table -> columns
    # --------------------------------------------------------

    allowed_columns_by_object: dict[
        str,
        set[str],
    ] = {}

    for database_object in (
        filtered_schema.get(
            "objects",
            [],
        )
    ):

        object_name = str(
            database_object.get(
                "name",
                "",
            )
        ).strip()

        if not object_name:
            continue

        columns = {
            str(
                column.get(
                    "name",
                    "",
                )
            ).casefold()

            for column in (
                database_object.get(
                    "columns",
                    [],
                )
            )

            if column.get(
                "name"
            )
        }

        allowed_columns_by_object[
            object_name.casefold()
        ] = columns

    # --------------------------------------------------------
    # 禁止数据库修改 / 管理行为
    # --------------------------------------------------------

    denied_action_names = (
        "SQLITE_INSERT",
        "SQLITE_UPDATE",
        "SQLITE_DELETE",

        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",

        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",

        "SQLITE_ALTER_TABLE",
        "SQLITE_REINDEX",
        "SQLITE_ANALYZE",

        "SQLITE_ATTACH",
        "SQLITE_DETACH",

        "SQLITE_TRANSACTION",
        "SQLITE_SAVEPOINT",

        "SQLITE_PRAGMA",
    )

    denied_actions = {
        getattr(
            sqlite3,
            action_name,
        )

        for action_name
        in denied_action_names

        if hasattr(
            sqlite3,
            action_name,
        )
    }

    sqlite_read_action = getattr(
        sqlite3,
        "SQLITE_READ",
        None,
    )

    sqlite_function_action = getattr(
        sqlite3,
        "SQLITE_FUNCTION",
        None,
    )

    # --------------------------------------------------------
    # Authorizer callback
    # --------------------------------------------------------

    def authorizer(
        action_code: int,
        parameter_1: str | None,
        parameter_2: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        """
        SQLite 准备执行每一个底层动作时调用。
        """

        del database_name
        del trigger_name

        # ----------------------------------------------------
        # 写操作全部拒绝
        # ----------------------------------------------------

        if action_code in (
            denied_actions
        ):
            return sqlite3.SQLITE_DENY

        # ----------------------------------------------------
        # Table + Column 权限
        # ----------------------------------------------------

        if (
            sqlite_read_action
            is not None
            and action_code
            == sqlite_read_action
        ):

            table_name = str(
                parameter_1
                or ""
            )

            column_name = str(
                parameter_2
                or ""
            )

            table_key = (
                table_name.casefold()
            )

            # sqlite_* 系统对象永不暴露。
            if table_key.startswith(
                "sqlite_"
            ):
                return sqlite3.SQLITE_DENY

            # 表没有出现在 Filtered Schema。
            if (
                table_key
                not in
                allowed_columns_by_object
            ):
                return sqlite3.SQLITE_DENY

            # COUNT(*) 等情况，
            # SQLite 可能没有具体 column name。
            #
            # 只要表允许访问即可。
            if not column_name:
                return sqlite3.SQLITE_OK

            column_key = (
                column_name.casefold()
            )

            # 字段必须同样出现在 Filtered Schema。
            if (
                column_key
                not in
                allowed_columns_by_object[
                    table_key
                ]
            ):
                return sqlite3.SQLITE_DENY

        # ----------------------------------------------------
        # 危险 SQLite Function
        # ----------------------------------------------------

        if (
            sqlite_function_action
            is not None
            and action_code
            == sqlite_function_action
        ):

            function_name = str(
                parameter_2
                or parameter_1
                or ""
            ).casefold()

            if function_name in {
                "load_extension",
                "writefile",
            }:
                return sqlite3.SQLITE_DENY

        return sqlite3.SQLITE_OK

    return authorizer


# ============================================================
# JSON 安全转换
# ============================================================


def make_json_safe(
    value: Any,
) -> Any:
    """
    转换 SQLite 值为 JSON 可序列化数据。
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        bytes,
    ):
        return (
            "0x"
            + value.hex()
        )

    return str(
        value
    )


# ============================================================
# SQL Execution
# ============================================================


def execute_read_only_sql(
    sql: str,
    *,
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
    max_rows: int = (
        DEFAULT_MAX_ROWS
    ),
    max_output_characters: int = (
        DEFAULT_MAX_OUTPUT_CHARACTERS
    ),
    timeout_seconds: float = (
        DEFAULT_QUERY_TIMEOUT_SECONDS
    ),
) -> dict[str, Any]:
    """
    执行受 Exposure Policy 控制的只读 SQL。
    """

    normalized_sql = (
        validate_read_only_sql(
            sql
        )
    )

    if max_rows <= 0:
        raise ValueError(
            "max_rows 必须大于 0。"
        )

    if max_output_characters <= 0:
        raise ValueError(
            "max_output_characters "
            "必须大于 0。"
        )

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds 必须大于 0。"
        )

    # --------------------------------------------------------
    # 每次 Tool Query 都重新 Resolve Policy。
    #
    # 这里没有调用 LLM。
    #
    # 优点：
    # Policy 改动后不需要修改 SQL Tool。
    # --------------------------------------------------------

    filtered_schema = (
        resolve_exposed_database_schema(
            database_path
        )
    )

    if not filtered_schema.get(
        "objects"
    ):
        raise RuntimeError(
            "当前没有允许 Agent 查询的 "
            "GitHub Intelligence 数据表。"
        )

    connection = (
        open_read_only_connection(
            database_path
        )
    )

    started_at = (
        time.perf_counter()
    )

    deadline = (
        started_at
        + timeout_seconds
    )

    def progress_handler() -> int:
        """
        SQL 执行超时中断。
        """

        if (
            time.perf_counter()
            > deadline
        ):
            return 1

        return 0

    try:

        # ----------------------------------------------------
        # Exposure Policy 执行层
        # ----------------------------------------------------

        connection.set_authorizer(
            build_authorizer(
                filtered_schema
            )
        )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        connection.set_progress_handler(
            progress_handler,
            PROGRESS_HANDLER_STEPS,
        )

        cursor = connection.execute(
            normalized_sql
        )

        # ----------------------------------------------------
        # Column Names
        # ----------------------------------------------------

        description = (
            cursor.description
            or []
        )

        columns = [
            str(
                item[0]
            )
            for item in description
        ]

        # ----------------------------------------------------
        # Row Limit
        # ----------------------------------------------------

        raw_rows = cursor.fetchmany(
            max_rows + 1
        )

        truncated = (
            len(raw_rows)
            > max_rows
        )

        if truncated:
            raw_rows = (
                raw_rows[
                    :max_rows
                ]
            )

        rows: list[
            dict[str, Any]
        ] = []

        for raw_row in raw_rows:

            row_data: dict[
                str,
                Any,
            ] = {}

            for column_name in columns:

                row_data[
                    column_name
                ] = make_json_safe(
                    raw_row[
                        column_name
                    ]
                )

            rows.append(
                row_data
            )

        elapsed_ms = (
            (
                time.perf_counter()
                - started_at
            )
            * 1000.0
        )

        result: dict[
            str,
            Any,
        ] = {
            "status": "success",

            "sql": normalized_sql,

            "columns": columns,

            "returned_row_count": (
                len(rows)
            ),

            "truncated": truncated,

            "max_rows": max_rows,

            "latency_ms": (
                elapsed_ms
            ),

            "rows": rows,
        }

        # ----------------------------------------------------
        # 最终字符数限制
        # ----------------------------------------------------

        serialized = json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

        if (
            len(serialized)
            > max_output_characters
        ):

            limited_rows = list(
                rows
            )

            while (
                limited_rows
                and len(
                    json.dumps(
                        {
                            **result,
                            "rows": (
                                limited_rows
                            ),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )
                > max_output_characters
            ):
                limited_rows.pop()

            result[
                "rows"
            ] = limited_rows

            result[
                "returned_row_count"
            ] = len(
                limited_rows
            )

            result[
                "truncated"
            ] = True

            result[
                "output_truncated_by_char_limit"
            ] = True

        else:

            result[
                "output_truncated_by_char_limit"
            ] = False

        return result

    except sqlite3.OperationalError as exc:

        error_text = str(
            exc
        )

        if (
            "interrupted"
            in error_text.lower()
        ):
            raise TimeoutError(
                "SQL 查询超过允许时间："
                f"{timeout_seconds:.2f} 秒。"
            ) from exc

        raise

    except sqlite3.DatabaseError as exc:

        error_text = str(
            exc
        )

        if (
            "not authorized"
            in error_text.lower()
        ):
            raise PermissionError(
                "SQL 访问了 Agent 不可见的"
                "数据库对象或字段，"
                "或尝试执行禁止的数据库操作。"
            ) from exc

        raise

    finally:

        connection.set_progress_handler(
            None,
            0,
        )

        connection.close()


# ============================================================
# LangChain Tool
# ============================================================


def create_github_intelligence_sql_tool(
    *,
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
    max_rows: int = (
        DEFAULT_MAX_ROWS
    ),
    max_output_characters: int = (
        DEFAULT_MAX_OUTPUT_CHARACTERS
    ),
    timeout_seconds: float = (
        DEFAULT_QUERY_TIMEOUT_SECONDS
    ),
) -> BaseTool:
    """
    创建 GitHub Intelligence
    Exposure-Policy-aware SQL Tool。
    """

    normalized_database_path = (
        Path(
            database_path
        ).resolve()
    )

    @tool(
        "query_github_intelligence_sql",
        args_schema=(
            GitHubIntelligenceSQLInput
        ),
    )
    def query_github_intelligence_sql(
        sql: str,
    ) -> str:
        """
        使用只读 SQL 查询本地 GitHub Intelligence SQLite。

        适用于精确结构化问题，例如：

        - 本地共有多少 GitHub 项目；
        - 某天入选多少项目；
        - 某项目出现过多少次；
        - 第一次 / 最近一次出现日期；
        - Star、Fork、Issue 等历史变化；
        - 项目语言、类型和数量统计；
        - GROUP BY、COUNT、AVG、MIN、MAX；
        - 多个可见业务表之间的 JOIN。

        必须严格根据当前 Agent 获得的
        GitHub Intelligence SQLite Schema
        生成 SQL。

        只能使用 Schema 中明确出现的：

        - Table / View；
        - Column。

        未显示的对象或字段都不可访问。

        不要使用 SELECT *。
        应明确列出需要读取的字段。

        项目技术内容、技术方案、详细摘要、
        热点解释和趋势语义等问题，
        优先使用 search_github_intelligence。

        本 Tool 永远只读。
        数据库修改必须由确定性业务 Pipeline 完成。
        """

        try:

            result = (
                execute_read_only_sql(
                    sql,

                    database_path=(
                        normalized_database_path
                    ),

                    max_rows=max_rows,

                    max_output_characters=(
                        max_output_characters
                    ),

                    timeout_seconds=(
                        timeout_seconds
                    ),
                )
            )

        except Exception as exc:

            return (
                "GitHub Intelligence SQL "
                "查询失败："
                f"{type(exc).__name__}："
                f"{exc}"
            )

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return (
        query_github_intelligence_sql
    )

# ============================================================
# LangChain Schema Tool
# ============================================================


def create_github_intelligence_schema_tool(
    *,
    database_path: Path = (
        DEFAULT_DATABASE_PATH
    ),
) -> BaseTool:
    """
    创建 GitHub Intelligence Schema 查询 Tool。

    本 Tool 不查询 GitHub 业务数据，
    只返回当前 Agent 被允许访问的数据库结构。

    返回内容来自：

        Real SQLite Schema
                +
        schema_exposure_policy
                ↓
        Filtered Schema

    因此：

    - internal 对象不会显示；
    - sensitive 对象不会显示；
    - 未登记对象不会显示；
    - hidden_columns 不会显示。

    该 Tool 主要供 LLM 在生成 SQL 之前调用。
    """

    normalized_database_path = (
        Path(
            database_path
        ).resolve()
    )

    @tool(
        "get_github_intelligence_schema"
    )
    def get_github_intelligence_schema() -> str:
        """
        获取当前 GitHub Intelligence SQLite
        对 Agent 可见的数据库 Schema。

        当用户的问题需要精确结构化查询时，
        应在生成 SQL 之前调用本工具。

        典型问题包括：

        - 数据库中共有多少项目；
        - 某一天入选多少项目；
        - 最近几天哪些项目出现次数最多；
        - Star、Fork、Issue 等数值统计；
        - 首次 / 最近出现日期；
        - 按语言、项目类型等进行统计；
        - 排名、聚合、分组；
        - 需要 JOIN 多张数据库表。

        本工具返回：

        - 当前允许访问的 Table / View；
        - 可访问 Column；
        - Column 类型；
        - Primary Key；
        - Foreign Key；
        - 表的业务说明；
        - query_priority；
        - SQLite 查询规则。

        本工具不会返回数据库中的实际业务记录。

        获取 Schema 后，
        应根据返回的真实 Schema 生成 SQLite
        SELECT 或 WITH ... SELECT 查询，
        再调用 query_github_intelligence_sql。

        对项目技术内容、技术方案、摘要解释、
        热点含义和趋势语义等问题，
        不应使用本工具，
        应优先使用 search_github_intelligence。
        """

        try:
            return render_database_schema(
                database_path=(
                    normalized_database_path
                )
            )

        except Exception as exc:
            return (
                "GitHub Intelligence Schema "
                "获取失败："
                f"{type(exc).__name__}："
                f"{exc}"
            )

    return get_github_intelligence_schema