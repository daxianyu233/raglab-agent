"""RAGLab Agent SQLite persistence backend.

负责为 LangGraph Agent 创建两类持久化能力：

1. SqliteSaver
   - thread 级 Graph Checkpoint；
   - 会话状态；
   - 消息历史；
   - Graph State；
   - pending writes；
   - interrupt / resume；
   - checkpoint history。

2. SqliteStore
   - user 级长期记忆；
   - 跨 thread 共享。

注意：

SqliteSaver 与 SqliteStore 使用同一个 SQLite 文件，
但使用两个独立的 sqlite3.Connection。

两者的事务管理方式不同：

- SqliteSaver 使用普通 sqlite3 事务模式；
- SqliteStore 自己显式执行 BEGIN / COMMIT，
  因此它的 Connection 必须使用
  isolation_level=None，也就是 autocommit 模式。
"""

from __future__ import annotations

import sqlite3

from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)
from langgraph.store.sqlite import (
    SqliteStore,
)


# ============================================================
# 默认路径
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DEFAULT_AGENT_STATE_DIRECTORY = (
    PROJECT_ROOT
    / "storage"
    / "agent_state"
)


DEFAULT_AGENT_STATE_DATABASE_PATH = (
    DEFAULT_AGENT_STATE_DIRECTORY
    / "raglab_agent_state.sqlite3"
)


# ============================================================
# Persistence Bundle
# ============================================================


@dataclass
class SQLiteAgentPersistence:
    """Agent SQLite 持久化资源集合。

    Checkpointer 与 Store：

    - 使用同一个 SQLite 文件；
    - 使用不同 Connection；
    - 采用各自需要的事务模式。

    这样既保持数据文件统一，
    又不会让两套事务管理机制互相干扰。
    """

    database_path: Path

    checkpoint_connection: sqlite3.Connection

    store_connection: sqlite3.Connection

    checkpointer: SqliteSaver

    store: SqliteStore

    def close(self) -> None:
        """关闭持久化资源。"""

        # SqliteStore 当前可能没有启动 TTL Sweeper，
        # 但如果未来启用 TTL，这里仍尝试安全停止。
        stop_ttl_sweeper = getattr(
            self.store,
            "stop_ttl_sweeper",
            None,
        )

        if callable(
            stop_ttl_sweeper
        ):
            try:
                stop_ttl_sweeper(
                    timeout=1.0
                )
            except Exception:
                # 关闭阶段不因为 TTL Sweeper
                # 阻止 SQLite Connection 释放。
                pass

        try:
            self.checkpoint_connection.close()

        finally:
            self.store_connection.close()


# ============================================================
# SQLite 公共配置
# ============================================================


def configure_sqlite_connection(
    connection: sqlite3.Connection,
) -> None:
    """配置本地 Agent State SQLite Connection。

    WAL：
        提高同一数据库文件上读写并存时的可用性。

    NORMAL：
        本地 Agent 场景下在性能与持久性之间取得平衡。

    busy_timeout：
        短暂写锁冲突时等待，而不是立即失败。

    foreign_keys：
        启用 SQLite 外键约束。
    """

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

    # 明确确保连接配置完成后不存在
    # 遗留的未提交事务。
    connection.commit()


# ============================================================
# Checkpointer Connection
# ============================================================


def create_checkpoint_connection(
    database_path: Path,
) -> sqlite3.Connection:
    """创建 SqliteSaver 使用的 Connection。

    SqliteSaver 官方实现自身通过 cursor
    管理 checkpoint 写入和 commit。

    因此这里保持 sqlite3 默认事务模式。
    """

    path = Path(
        database_path
    ).resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(path),

        check_same_thread=False,

        timeout=30.0,
    )

    try:

        configure_sqlite_connection(
            connection
        )

        return connection

    except Exception:

        connection.close()

        raise


# ============================================================
# Store Connection
# ============================================================


def create_store_connection(
    database_path: Path,
) -> sqlite3.Connection:
    """创建 SqliteStore 使用的 Connection。

    关键：

        isolation_level=None

    SqliteStore 自己会显式执行：

        BEGIN
        ...
        COMMIT

    因此必须关闭 Python sqlite3
    默认的隐式事务管理。

    否则可能出现：

        OperationalError:
        cannot start a transaction
        within a transaction
    """

    path = Path(
        database_path
    ).resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(path),

        check_same_thread=False,

        timeout=30.0,

        # ----------------------------------------------------
        # 非常重要
        # ----------------------------------------------------
        #
        # SqliteStore 内部自己执行 BEGIN / COMMIT。
        #
        # 如果保留 sqlite3 默认 isolation_level，
        # Python 可能提前创建隐式事务，
        # 后续 SqliteStore 再执行 BEGIN 就会报：
        #
        # cannot start a transaction
        # within a transaction
        #
        isolation_level=None,
    )

    try:

        configure_sqlite_connection(
            connection
        )

        return connection

    except Exception:

        connection.close()

        raise


# ============================================================
# Persistence Factory
# ============================================================


def create_sqlite_agent_persistence(
    database_path: Path = (
        DEFAULT_AGENT_STATE_DATABASE_PATH
    ),
) -> SQLiteAgentPersistence:
    """创建 Agent SQLite Persistence Backend。

    数据文件：

        storage/
        └─ agent_state/
           └─ raglab_agent_state.sqlite3

    逻辑结构：

        SQLite File
        │
        ├─ SqliteSaver
        │    ↓
        │  thread-level checkpoints
        │
        └─ SqliteStore
             ↓
           user-level long-term memory
    """

    path = Path(
        database_path
    ).resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Checkpointer Connection
    # --------------------------------------------------------

    checkpoint_connection = (
        create_checkpoint_connection(
            path
        )
    )

    try:

        # ----------------------------------------------------
        # Store Connection
        # ----------------------------------------------------

        store_connection = (
            create_store_connection(
                path
            )
        )

    except Exception:

        checkpoint_connection.close()

        raise

    try:

        # ----------------------------------------------------
        # Thread-level Checkpointer
        # ----------------------------------------------------

        checkpointer = SqliteSaver(
            checkpoint_connection
        )

        # SqliteSaver 自身也会在首次使用时自动 setup。
        #
        # 这里主动执行一次的目的，
        # 是让数据库问题在 Agent 启动阶段暴露，
        # 而不是等到第一次用户请求时才发现。
        checkpointer.setup()

        # ----------------------------------------------------
        # User-level Long-Term Store
        # ----------------------------------------------------

        store = SqliteStore(
            store_connection
        )

        # SqliteStore 使用前需要完成 migration。
        store.setup()

        return SQLiteAgentPersistence(
            database_path=path,

            checkpoint_connection=(
                checkpoint_connection
            ),

            store_connection=(
                store_connection
            ),

            checkpointer=(
                checkpointer
            ),

            store=(
                store
            ),
        )

    except Exception:

        checkpoint_connection.close()

        store_connection.close()

        raise


# ============================================================
# Debug / Inspection
# ============================================================


def list_persistence_tables(
    database_path: Path = (
        DEFAULT_AGENT_STATE_DATABASE_PATH
    ),
) -> list[str]:
    """列出 Agent State SQLite 中的内部表。

    仅用于开发和调试。

    不应作为 Agent Tool 暴露给 LLM。
    """

    path = Path(
        database_path
    ).resolve()

    if not path.exists():
        return []

    connection = sqlite3.connect(
        str(path)
    )

    try:

        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        return [
            str(
                row[0]
            )
            for row in rows
        ]

    finally:

        connection.close()