"""SQLite Tool Policy Registry。

职责：

1. 创建 tool_policy_registry；
2. 自动登记新发现的 Tool；
3. 未分类 Tool 默认 PENDING + disabled；
4. 查询 Tool Policy；
5. 修改 Tool Policy；
6. 统计 Tool 的安全类型；
7. 保存 Tool 来源、Replay 策略和审批要求。

重要原则：

    新 Tool 默认不可信。

即：

    UNKNOWN TOOL
        ↓
    自动注册
        ↓
    PENDING
        ↓
    enabled = False
        ↓
    不暴露给 LLM

只有完成安全分类以后才能 ACTIVE。
"""

from __future__ import annotations

import sqlite3

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path
from typing import Any, Sequence

from raglab.control.tool_policy import (
    ReplayPolicy,
    ToolEffectType,
    ToolPolicyRecord,
    ToolPolicyStatus,
    normalize_tool_name,
    normalize_tool_source,
    parse_effect_type,
    parse_policy_status,
    parse_replay_policy,
)

from raglab.settings import (
    PROJECT_ROOT,
)


# ============================================================
# 默认数据库
# ============================================================


DEFAULT_CONTROL_DATABASE_PATH = (
    PROJECT_ROOT
    / "storage"
    / "control_plane"
    / "raglab_control.sqlite3"
)


# ============================================================
# 时间
# ============================================================


def utc_now_text() -> str:
    """生成 UTC ISO 时间。"""

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


# ============================================================
# Repository
# ============================================================


class ToolPolicyRepository:
    """SQLite Tool Policy Registry。"""

    def __init__(
        self,
        database_path: (
            str
            | Path
        ) = DEFAULT_CONTROL_DATABASE_PATH,
    ) -> None:

        self.database_path = (
            Path(
                database_path
            )
            .resolve()
        )

    # --------------------------------------------------------
    # SQLite
    # --------------------------------------------------------

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """创建 SQLite 连接。"""

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            str(
                self.database_path
            ),
            timeout=30.0,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        connection.execute(
            "PRAGMA synchronous = NORMAL"
        )

        connection.execute(
            "PRAGMA busy_timeout = 5000"
        )

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    def setup(
        self,
    ) -> None:
        """初始化 Control Plane Schema。"""

        with self._connect() as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                tool_policy_registry
                (
                    tool_name TEXT
                        PRIMARY KEY,

                    tool_source TEXT
                        NOT NULL,

                    source_id TEXT,

                    effect_type TEXT,

                    has_external_side_effect
                        INTEGER,

                    replay_policy TEXT,

                    requires_approval INTEGER
                        NOT NULL
                        DEFAULT 0,

                    idempotency_strategy TEXT,

                    compensation_tool TEXT,

                    enabled INTEGER
                        NOT NULL
                        DEFAULT 0,

                    status TEXT
                        NOT NULL
                        DEFAULT 'PENDING',

                    description TEXT
                        NOT NULL
                        DEFAULT '',

                    discovered_at TEXT
                        NOT NULL,

                    last_seen_at TEXT
                        NOT NULL,

                    updated_at TEXT
                        NOT NULL,

                    CHECK (
                        effect_type IS NULL
                        OR effect_type IN (
                            'READ_ONLY',
                            'IDEMPOTENT_WRITE',
                            'COMPENSATABLE_WRITE',
                            'IRREVERSIBLE_WRITE'
                        )
                    ),

                    CHECK (
                        replay_policy IS NULL
                        OR replay_policy IN (
                            'ALLOW',
                            'GUARDED',
                            'REQUIRE_APPROVAL',
                            'DENY'
                        )
                    ),

                    CHECK (
                        status IN (
                            'PENDING',
                            'ACTIVE',
                            'BLOCKED'
                        )
                    ),

                    CHECK (
                        enabled IN (
                            0,
                            1
                        )
                    ),

                    CHECK (
                        requires_approval IN (
                            0,
                            1
                        )
                    ),

                    CHECK (
                        has_external_side_effect
                        IS NULL
                        OR
                        has_external_side_effect
                        IN (
                            0,
                            1
                        )
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_tool_policy_status
                ON tool_policy_registry(
                    status
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_tool_policy_effect_type
                ON tool_policy_registry(
                    effect_type
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_tool_policy_source
                ON tool_policy_registry(
                    tool_source,
                    source_id
                )
                """
            )

    # --------------------------------------------------------
    # Row → Model
    # --------------------------------------------------------

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> ToolPolicyRecord:
        """SQLite Row 转 ToolPolicyRecord。"""

        external_value = (
            row[
                "has_external_side_effect"
            ]
        )

        return ToolPolicyRecord(

            tool_name=str(
                row[
                    "tool_name"
                ]
            ),

            tool_source=str(
                row[
                    "tool_source"
                ]
            ),

            source_id=(
                str(
                    row[
                        "source_id"
                    ]
                )
                if row[
                    "source_id"
                ]
                is not None
                else None
            ),

            effect_type=(
                parse_effect_type(
                    row[
                        "effect_type"
                    ]
                )
            ),

            has_external_side_effect=(
                bool(
                    external_value
                )
                if external_value
                is not None
                else None
            ),

            replay_policy=(
                parse_replay_policy(
                    row[
                        "replay_policy"
                    ]
                )
            ),

            requires_approval=bool(
                row[
                    "requires_approval"
                ]
            ),

            idempotency_strategy=(
                str(
                    row[
                        "idempotency_strategy"
                    ]
                )
                if row[
                    "idempotency_strategy"
                ]
                is not None
                else None
            ),

            compensation_tool=(
                str(
                    row[
                        "compensation_tool"
                    ]
                )
                if row[
                    "compensation_tool"
                ]
                is not None
                else None
            ),

            enabled=bool(
                row[
                    "enabled"
                ]
            ),

            status=(
                parse_policy_status(
                    row[
                        "status"
                    ]
                )
            ),

            description=str(
                row[
                    "description"
                ]
                or ""
            ),

            discovered_at=str(
                row[
                    "discovered_at"
                ]
            ),

            last_seen_at=str(
                row[
                    "last_seen_at"
                ]
            ),

            updated_at=str(
                row[
                    "updated_at"
                ]
            ),
        )

    # --------------------------------------------------------
    # 新 Tool 自动发现
    # --------------------------------------------------------

    def ensure_discovered(
        self,
        *,
        tool_name: str,
        tool_source: str = "unknown",
        source_id: str | None = None,
        description: str = "",
    ) -> ToolPolicyRecord:
        """确保 Tool 已经存在于 Registry。

        如果是第一次发现：

            status = PENDING
            enabled = False
            effect_type = NULL

        即默认 Fail Closed。
        """

        normalized_name = (
            normalize_tool_name(
                tool_name
            )
        )

        normalized_source = (
            normalize_tool_source(
                tool_source
            )
        )

        normalized_source_id = (
            str(
                source_id
            ).strip()
            if source_id
            is not None
            and str(
                source_id
            ).strip()
            else None
        )

        normalized_description = (
            str(
                description
            ).strip()
        )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                INSERT INTO
                tool_policy_registry
                (
                    tool_name,
                    tool_source,
                    source_id,
                    effect_type,
                    has_external_side_effect,
                    replay_policy,
                    requires_approval,
                    idempotency_strategy,
                    compensation_tool,
                    enabled,
                    status,
                    description,
                    discovered_at,
                    last_seen_at,
                    updated_at
                )
                VALUES
                (
                    ?,
                    ?,
                    ?,
                    NULL,
                    NULL,
                    NULL,
                    0,
                    NULL,
                    NULL,
                    0,
                    'PENDING',
                    ?,
                    ?,
                    ?,
                    ?
                )
                ON CONFLICT(
                    tool_name
                )
                DO UPDATE SET
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    normalized_name,
                    normalized_source,
                    normalized_source_id,
                    normalized_description,
                    now,
                    now,
                    now,
                ),
            )

        record = self.get(
            normalized_name
        )

        if record is None:
            raise RuntimeError(
                "Tool 自动注册后无法重新读取："
                f"{normalized_name}"
            )

        return record

    # --------------------------------------------------------
    # 查询
    # --------------------------------------------------------

    def get(
        self,
        tool_name: str,
    ) -> ToolPolicyRecord | None:
        """读取一个 Tool Policy。"""

        normalized_name = (
            normalize_tool_name(
                tool_name
            )
        )

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT *
                FROM tool_policy_registry
                WHERE tool_name = ?
                """,
                (
                    normalized_name,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(
            row
        )

    def list_all(
        self,
    ) -> list[ToolPolicyRecord]:
        """读取全部 Tool Policy。"""

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT *
                FROM tool_policy_registry
                ORDER BY
                    tool_source,
                    tool_name
                """
            ).fetchall()

        return [
            self._row_to_record(
                row
            )
            for row in rows
        ]

    def list_pending(
        self,
    ) -> list[ToolPolicyRecord]:
        """读取尚未完成分类的 Tool。"""

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT *
                FROM tool_policy_registry
                WHERE status = 'PENDING'
                ORDER BY tool_name
                """
            ).fetchall()

        return [
            self._row_to_record(
                row
            )
            for row in rows
        ]

    # --------------------------------------------------------
    # 完整 Policy 设置
    # --------------------------------------------------------

    def set_policy(
        self,
        *,
        tool_name: str,
        effect_type: (
            ToolEffectType
            | str
        ),
        has_external_side_effect: bool,
        replay_policy: (
            ReplayPolicy
            | str
        ),
        requires_approval: bool = False,
        enabled: bool = True,
        status: (
            ToolPolicyStatus
            | str
        ) = ToolPolicyStatus.ACTIVE,
        tool_source: str | None = None,
        source_id: str | None = None,
        idempotency_strategy: str | None = None,
        compensation_tool: str | None = None,
        description: str | None = None,
    ) -> ToolPolicyRecord:
        """设置一个 Tool 的完整安全策略。"""

        normalized_name = (
            normalize_tool_name(
                tool_name
            )
        )

        parsed_effect = (
            parse_effect_type(
                effect_type
            )
        )

        parsed_replay = (
            parse_replay_policy(
                replay_policy
            )
        )

        parsed_status = (
            parse_policy_status(
                status
            )
        )

        if parsed_effect is None:
            raise ValueError(
                "ACTIVE Policy 必须提供 effect_type。"
            )

        if parsed_replay is None:
            raise ValueError(
                "ACTIVE Policy 必须提供 replay_policy。"
            )

        current = self.get(
            normalized_name
        )

        if current is None:

            current = (
                self.ensure_discovered(
                    tool_name=normalized_name,
                    tool_source=(
                        tool_source
                        or "manual"
                    ),
                    source_id=source_id,
                    description=(
                        description
                        or ""
                    ),
                )
            )

        effective_source = (
            normalize_tool_source(
                tool_source
            )
            if tool_source
            is not None
            else current.tool_source
        )

        effective_source_id = (
            str(
                source_id
            ).strip()
            if source_id
            is not None
            else current.source_id
        )

        effective_description = (
            str(
                description
            ).strip()
            if description
            is not None
            else current.description
        )

        normalized_idempotency = (
            str(
                idempotency_strategy
            ).strip()
            if idempotency_strategy
            else None
        )

        normalized_compensation = (
            normalize_tool_name(
                compensation_tool
            )
            if compensation_tool
            else None
        )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE
                    tool_policy_registry
                SET
                    tool_source = ?,
                    source_id = ?,
                    effect_type = ?,
                    has_external_side_effect = ?,
                    replay_policy = ?,
                    requires_approval = ?,
                    idempotency_strategy = ?,
                    compensation_tool = ?,
                    enabled = ?,
                    status = ?,
                    description = ?,
                    updated_at = ?
                WHERE
                    tool_name = ?
                """,
                (
                    effective_source,
                    effective_source_id,
                    parsed_effect.value,
                    (
                        1
                        if has_external_side_effect
                        else 0
                    ),
                    parsed_replay.value,
                    (
                        1
                        if requires_approval
                        else 0
                    ),
                    normalized_idempotency,
                    normalized_compensation,
                    (
                        1
                        if enabled
                        else 0
                    ),
                    parsed_status.value,
                    effective_description,
                    now,
                    normalized_name,
                ),
            )

        result = self.get(
            normalized_name
        )

        if result is None:
            raise RuntimeError(
                "更新 Tool Policy 后无法重新读取："
                f"{normalized_name}"
            )

        return result

    # --------------------------------------------------------
    # 状态控制
    # --------------------------------------------------------

    def block(
        self,
        tool_name: str,
    ) -> ToolPolicyRecord:
        """阻止 Tool 执行。"""

        return self._set_runtime_status(
            tool_name=tool_name,
            status=(
                ToolPolicyStatus.BLOCKED
            ),
            enabled=False,
        )

    def disable(
        self,
        tool_name: str,
    ) -> ToolPolicyRecord:
        """暂时禁用 Tool。"""

        current = self.get(
            tool_name
        )

        if current is None:
            raise KeyError(
                f"Tool 尚未注册：{tool_name}"
            )

        return self._set_runtime_status(
            tool_name=tool_name,
            status=current.status,
            enabled=False,
        )

    def enable(
        self,
        tool_name: str,
    ) -> ToolPolicyRecord:
        """重新启用已经完成分类的 Tool。"""

        current = self.get(
            tool_name
        )

        if current is None:
            raise KeyError(
                f"Tool 尚未注册：{tool_name}"
            )

        if (
            not current.is_classified
        ):
            raise ValueError(
                "PENDING Tool 尚未完成安全分类，"
                "不能直接 enable："
                f"{tool_name}"
            )

        if (
            current.status
            == ToolPolicyStatus.BLOCKED
        ):
            raise ValueError(
                "BLOCKED Tool 必须先重新 set-policy，"
                "不能直接 enable："
                f"{tool_name}"
            )

        return self._set_runtime_status(
            tool_name=tool_name,
            status=(
                ToolPolicyStatus.ACTIVE
            ),
            enabled=True,
        )

    def _set_runtime_status(
        self,
        *,
        tool_name: str,
        status: ToolPolicyStatus,
        enabled: bool,
    ) -> ToolPolicyRecord:
        """修改 Tool Runtime 状态。"""

        normalized_name = (
            normalize_tool_name(
                tool_name
            )
        )

        current = self.get(
            normalized_name
        )

        if current is None:
            raise KeyError(
                f"Tool 尚未注册：{normalized_name}"
            )

        now = utc_now_text()

        with self._connect() as connection:

            connection.execute(
                """
                UPDATE
                    tool_policy_registry
                SET
                    status = ?,
                    enabled = ?,
                    updated_at = ?
                WHERE
                    tool_name = ?
                """,
                (
                    status.value,
                    (
                        1
                        if enabled
                        else 0
                    ),
                    now,
                    normalized_name,
                ),
            )

        result = self.get(
            normalized_name
        )

        if result is None:
            raise RuntimeError(
                "修改 Tool 状态后无法读取："
                f"{normalized_name}"
            )

        return result

    # --------------------------------------------------------
    # Tool 集合同步
    # --------------------------------------------------------

    def discover_tool_objects(
        self,
        tools: Sequence[Any],
        *,
        base_tool_names: (
            set[str]
            | None
        ) = None,
    ) -> list[ToolPolicyRecord]:
        """同步当前 Agent 发现的全部 Tool。

        base_tool_names 中的 Tool：

            source = base

        其他后续动态出现的 Tool：

            source = dynamic

        后面接 MCP 时可以继续扩展来源识别。
        """

        base_names = (
            base_tool_names
            or set()
        )

        records: list[
            ToolPolicyRecord
        ] = []

        for current_tool in tools:

            tool_name = str(
                getattr(
                    current_tool,
                    "name",
                    "",
                )
            ).strip()

            if not tool_name:
                raise ValueError(
                    "发现了没有 name 的 Tool："
                    f"{current_tool!r}"
                )

            tool_source = (
                "base"
                if tool_name
                in base_names
                else "dynamic"
            )

            tool_description = str(
                getattr(
                    current_tool,
                    "description",
                    "",
                )
                or ""
            ).strip()

            record = (
                self.ensure_discovered(
                    tool_name=tool_name,
                    tool_source=tool_source,
                    description=(
                        tool_description
                    ),
                )
            )

            records.append(
                record
            )

        return records

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        """统计 Tool Policy Registry。"""

        with self._connect() as connection:

            total = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tool_policy_registry
                    """
                ).fetchone()[0]
            )

            pending = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tool_policy_registry
                    WHERE status = 'PENDING'
                    """
                ).fetchone()[0]
            )

            active = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tool_policy_registry
                    WHERE status = 'ACTIVE'
                    """
                ).fetchone()[0]
            )

            blocked = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tool_policy_registry
                    WHERE status = 'BLOCKED'
                    """
                ).fetchone()[0]
            )

            enabled = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tool_policy_registry
                    WHERE enabled = 1
                    """
                ).fetchone()[0]
            )

            external = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM tool_policy_registry
                    WHERE
                        has_external_side_effect = 1
                    """
                ).fetchone()[0]
            )

            rows = connection.execute(
                """
                SELECT
                    COALESCE(
                        effect_type,
                        'UNCLASSIFIED'
                    ) AS effect_type,
                    COUNT(*) AS count
                FROM
                    tool_policy_registry
                GROUP BY
                    COALESCE(
                        effect_type,
                        'UNCLASSIFIED'
                    )
                ORDER BY
                    effect_type
                """
            ).fetchall()

        by_effect_type = {
            str(
                row[
                    "effect_type"
                ]
            ): int(
                row[
                    "count"
                ]
            )
            for row in rows
        }

        return {
            "database_path": str(
                self.database_path
            ),
            "total": total,
            "active": active,
            "pending": pending,
            "blocked": blocked,
            "enabled": enabled,
            "external_side_effect_tools": (
                external
            ),
            "by_effect_type": (
                by_effect_type
            ),
        }

    # --------------------------------------------------------
    # 初始系统 Tool Bootstrap
    # --------------------------------------------------------

    def bootstrap_known_tools(
        self,
    ) -> None:
        """写入当前项目已经确认过的 Tool Policy。

        这里仅用于第一次建立数据库。

        使用的是 INSERT / set-if-missing 逻辑，
        不会覆盖管理员后续在 SQLite 中修改的策略。

        以后新增 Tool 不需要修改这里：
        新 Tool 会自动进入 PENDING，
        再通过 manage_tool_policies CLI 完成分类。
        """

        seeds = [
            {
                "tool_name": (
                    "search_knowledge_base"
                ),
                "tool_source": "base",
                "effect_type": (
                    ToolEffectType.READ_ONLY
                ),
                "external": False,
                "replay": (
                    ReplayPolicy.ALLOW
                ),
                "approval": False,
                "description": (
                    "只读取 PDF 知识库索引。"
                ),
            },
            {
                "tool_name": (
                    "search_github_intelligence"
                ),
                "tool_source": "base",
                "effect_type": (
                    ToolEffectType.READ_ONLY
                ),
                "external": False,
                "replay": (
                    ReplayPolicy.ALLOW
                ),
                "approval": False,
                "description": (
                    "只读取已有 GitHub 技术情报索引。"
                ),
            },
            {
                "tool_name": (
                    "get_github_intelligence_schema"
                ),
                "tool_source": "base",
                "effect_type": (
                    ToolEffectType.READ_ONLY
                ),
                "external": False,
                "replay": (
                    ReplayPolicy.ALLOW
                ),
                "approval": False,
                "description": (
                    "只读取允许暴露的 GitHub "
                    "情报数据库 Schema。"
                ),
            },
            {
                "tool_name": (
                    "query_github_intelligence_sql"
                ),
                "tool_source": "base",
                "effect_type": (
                    ToolEffectType.READ_ONLY
                ),
                "external": False,
                "replay": (
                    ReplayPolicy.ALLOW
                ),
                "approval": False,
                "description": (
                    "只执行受限制的 SQLite "
                    "SELECT / WITH 查询。"
                ),
            },
            {
                "tool_name": (
                    "list_skills"
                ),
                "tool_source": "base",
                "effect_type": (
                    ToolEffectType.READ_ONLY
                ),
                "external": False,
                "replay": (
                    ReplayPolicy.ALLOW
                ),
                "approval": False,
                "description": (
                    "读取 Skill Catalog 和 Runtime 状态。"
                ),
            },
            {
                "tool_name": (
                    "load_skill"
                ),
                "tool_source": "base",
                "effect_type": (
                    ToolEffectType.IDEMPOTENT_WRITE
                ),
                "external": False,
                "replay": (
                    ReplayPolicy.ALLOW
                ),
                "approval": False,
                "description": (
                    "修改当前进程的 Skill Runtime，"
                    "不修改外部业务系统。"
                ),
            },
            {
                "tool_name": (
                    "update_github_intelligence"
                ),
                "tool_source": "skill",
                "source_id": (
                    "github-intelligence-update"
                ),
                "effect_type": (
                    ToolEffectType.IRREVERSIBLE_WRITE
                ),
                "external": True,
                "replay": (
                    ReplayPolicy.REQUIRE_APPROVAL
                ),
                "approval": True,
                "description": (
                    "会更新 GitHub 技术情报文件、"
                    "数据库和检索索引；"
                    "当前尚未接入 Effect Ledger "
                    "和 Compensation。"
                ),
            },
        ]

        for seed in seeds:

            tool_name = str(
                seed[
                    "tool_name"
                ]
            )

            if (
                self.get(
                    tool_name
                )
                is not None
            ):
                # 已存在时绝不覆盖。
                continue

            self.ensure_discovered(
                tool_name=tool_name,
                tool_source=str(
                    seed.get(
                        "tool_source",
                        "base",
                    )
                ),
                source_id=seed.get(
                    "source_id"
                ),
                description=str(
                    seed.get(
                        "description",
                        "",
                    )
                ),
            )

            self.set_policy(
                tool_name=tool_name,
                tool_source=str(
                    seed.get(
                        "tool_source",
                        "base",
                    )
                ),
                source_id=seed.get(
                    "source_id"
                ),
                effect_type=seed[
                    "effect_type"
                ],
                has_external_side_effect=bool(
                    seed[
                        "external"
                    ]
                ),
                replay_policy=seed[
                    "replay"
                ],
                requires_approval=bool(
                    seed[
                        "approval"
                    ]
                ),
                enabled=True,
                status=(
                    ToolPolicyStatus.ACTIVE
                ),
                description=str(
                    seed.get(
                        "description",
                        "",
                    )
                ),
            )