"""Runtime security adapter for RAG-LAB Agent.

把现有 Agent 的普通 ToolNode 替换为安全 ToolNode。

当前真正接入：

1. Tool Policy Registry
2. Fail-closed
3. HITL interrupt / resume
4. 与现有 Job Single-Flight 协同
5. 动态 Skill Tool 刷新后的安全包装
6. Human / AI / Tool 原始消息写入 Conversation Event Store

注意：

Single-Flight 不在本文件重新实现。

它仍然由：

    ScheduledJobRepository
    JobExecutionService
    GithubUpdateJobCoordinator

负责。

本文件负责的是：

    ToolCall
        ↓
    Policy
        ↓
    HITL
        ↓
    真正 ToolNode
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

from contextlib import closing

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import (
    Command,
    interrupt,
)

from raglab.agent.conversation_event_adapter import (
    ConversationArchiveReport,
    archive_messages_to_event_store,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.conversation_archive_reconciler import (
    ArchiveReconciliationResult,
    reconcile_checkpoint_archive,
)
from raglab.agent.long_term_memory_agent import (
    LongTermMemoryContext,
    normalize_user_id,
)

from raglab.agent.persistent_langgraph_agent import (
    PersistentLangGraphResult,
    count_human_turns,
    normalize_thread_id,
)

from raglab.generation.rag_chain import (
    extract_answer_text,
)
from raglab.observability.runtime_events import emit_runtime_event


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DEFAULT_CONTROL_DB = (
    PROJECT_ROOT
    / "storage"
    / "control_plane"
    / "raglab_control.sqlite3"
)


# ============================================================
# Policy Model
# ============================================================

@dataclass(frozen=True)
class ToolPolicySnapshot:
    tool_name: str

    effect_type: str | None

    # 与 has_external_side_effect 完全不同：
    #
    # RETRIEVAL:
    #   读取当前 Conversation Context 之外的数据源，
    #   例如 RAG / SQL / GitHub Intelligence。
    #
    # CONTROL:
    #   Runtime / Skill 控制面操作。
    #
    # ACTION:
    #   会执行真实业务动作或写操作。
    context_access_type: str

    has_external_side_effect: bool

    requires_approval: bool

    enabled: bool

    status: str

    description: str


# ============================================================
# SQLite Policy Store
# ============================================================

class SQLiteToolPolicyStore:
    """只负责读取 Tool Policy Registry。

    Policy 的语义仍然由数据库保存，
    SecureToolNode 不把每个 Tool 的分类硬编码在执行逻辑里。
    """

    def __init__(
        self,
        database_path: Path = DEFAULT_CONTROL_DB,
    ) -> None:

        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.setup()

    def _connect(
        self,
    ) -> sqlite3.Connection:

        connection = sqlite3.connect(
            str(
                self.database_path
            ),
            timeout=30.0,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def setup(
        self,
    ) -> None:
        """兼容数据库尚未初始化的情况。"""

        with closing(self._connect()) as connection:

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS
                tool_policy_registry (
                    tool_name TEXT PRIMARY KEY,
                    tool_source TEXT,
                    source_id TEXT,
                    effect_type TEXT,
                    context_access_type TEXT
                        NOT NULL DEFAULT 'UNSPECIFIED',
                    has_external_side_effect INTEGER
                        NOT NULL DEFAULT 0,
                    replay_policy TEXT,
                    requires_approval INTEGER
                        NOT NULL DEFAULT 0,
                    idempotency_strategy TEXT,
                    compensation_tool TEXT,
                    enabled INTEGER
                        NOT NULL DEFAULT 0,
                    status TEXT
                        NOT NULL DEFAULT 'PENDING',
                    description TEXT,
                    discovered_at TEXT,
                    last_seen_at TEXT,
                    updated_at TEXT
                )
                """
            )

            # Phase 7B schema migration：
            # 已存在的 control DB 也必须补上 Context Access 维度。
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(tool_policy_registry)"
                ).fetchall()
            }

            if "context_access_type" not in columns:
                connection.execute(
                    """
                    ALTER TABLE tool_policy_registry
                    ADD COLUMN context_access_type TEXT
                    NOT NULL DEFAULT 'UNSPECIFIED'
                    """
                )

            connection.commit()

        self.ensure_default_policies()

    # ========================================================
    # Bootstrap
    # ========================================================

    def ensure_default_policies(
        self,
    ) -> None:
        """确保当前核心 Tool 有明确 Policy。

        特别重要：

        update_github_intelligence
            =
        IRREVERSIBLE_WRITE
        + requires_approval=True
        """

        definitions = [
            {
                "tool_name":
                    "search_knowledge_base",
                "effect_type":
                    "READ_ONLY",
                "context_access_type":
                    "RETRIEVAL",
                "external": False,
                "approval": False,
                "description":
                    "PDF 知识库只读检索。",
            },
            {
                "tool_name":
                    "get_github_intelligence_schema",
                "effect_type":
                    "READ_ONLY",
                "context_access_type":
                    "RETRIEVAL",
                "external": False,
                "approval": False,
                "description":
                    "读取 GitHub 情报数据库 Schema。",
            },
            {
                "tool_name":
                    "get_github_daily_report",
                "effect_type":
                    "READ_ONLY",
                "context_access_type":
                    "RETRIEVAL",
                "external": False,
                "approval": False,
                "description":
                    "按目标日期精确读取 GitHub 日报和当日分析数据。",
            },
            {
                "tool_name":
                    "search_github_intelligence",
                "effect_type":
                    "READ_ONLY",
                "context_access_type":
                    "RETRIEVAL",
                "external": False,
                "approval": False,
                "description":
                    "GitHub 技术情报只读检索。",
            },
            {
                "tool_name":
                    "query_github_intelligence_sql",
                "effect_type":
                    "READ_ONLY",
                "context_access_type":
                    "RETRIEVAL",
                "external": False,
                "approval": False,
                "description":
                    "受限 SQLite SELECT 查询。",
            },
            {
                "tool_name":
                    "list_skills",
                "effect_type":
                    "READ_ONLY",
                "context_access_type":
                    "CONTROL",
                "external": False,
                "approval": False,
                "description":
                    "读取 Skill Catalog。",
            },
            {
                "tool_name":
                    "load_skill",
                "effect_type":
                    "IDEMPOTENT_WRITE",
                "context_access_type":
                    "CONTROL",
                "external": False,
                "approval": False,
                "description":
                    "加载 Skill Runtime 状态。",
            },
            {
                "tool_name":
                    "update_github_intelligence",
                "effect_type":
                    "IRREVERSIBLE_WRITE",
                "context_access_type":
                    "ACTION",
                "external": True,
                "approval": True,
                "description":
                    "重新采集、分析并写入 "
                    "GitHub 技术情报。"
                    "执行前必须 HITL 审批。",
            },
        ]

        for definition in definitions:

            self._ensure_policy(
                **definition
            )

    def _ensure_policy(
        self,
        *,
        tool_name: str,
        effect_type: str,
        context_access_type: str,
        external: bool,
        approval: bool,
        description: str,
    ) -> None:

        now = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        with closing(self._connect()) as connection:

            cursor = connection.execute(
                """
                UPDATE tool_policy_registry
                SET
                    effect_type = ?,
                    context_access_type = ?,
                    has_external_side_effect = ?,
                    requires_approval = ?,
                    enabled = 1,
                    status = 'ACTIVE',
                    description = ?,
                    last_seen_at = ?,
                    updated_at = ?
                WHERE tool_name = ?
                """,
                (
                    effect_type,
                    str(
                        context_access_type
                    ).strip().upper()
                    or "UNSPECIFIED",
                    int(
                        external
                    ),
                    int(
                        approval
                    ),
                    description,
                    now,
                    now,
                    tool_name,
                ),
            )

            if cursor.rowcount == 0:

                connection.execute(
                    """
                    INSERT INTO
                    tool_policy_registry (
                        tool_name,
                        tool_source,
                        source_id,
                        effect_type,
                        context_access_type,
                        has_external_side_effect,
                        requires_approval,
                        enabled,
                        status,
                        description,
                        discovered_at,
                        last_seen_at,
                        updated_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        1, 'ACTIVE', ?, ?, ?, ?
                    )
                    """,
                    (
                        tool_name,
                        "RUNTIME",
                        None,
                        effect_type,
                        str(
                            context_access_type
                        ).strip().upper()
                        or "UNSPECIFIED",
                        int(
                            external
                        ),
                        int(
                            approval
                        ),
                        description,
                        now,
                        now,
                        now,
                    ),
                )

            connection.commit()

    # ========================================================
    # Read
    # ========================================================

    def get_policy(
        self,
        tool_name: str,
    ) -> ToolPolicySnapshot | None:

        normalized_name = str(
            tool_name
        ).strip()

        with closing(self._connect()) as connection:

            row = connection.execute(
                """
                SELECT
                    tool_name,
                    effect_type,
                    context_access_type,
                    has_external_side_effect,
                    requires_approval,
                    enabled,
                    status,
                    description
                FROM tool_policy_registry
                WHERE tool_name = ?
                LIMIT 1
                """,
                (
                    normalized_name,
                ),
            ).fetchone()

        if row is None:
            return None

        return ToolPolicySnapshot(
            tool_name=str(
                row["tool_name"]
            ),
            effect_type=(
                None
                if row["effect_type"] is None
                else str(
                    row["effect_type"]
                )
            ),
            context_access_type=str(
                row[
                    "context_access_type"
                ]
                or "UNSPECIFIED"
            ).strip().upper(),
            has_external_side_effect=bool(
                row[
                    "has_external_side_effect"
                ]
            ),
            requires_approval=bool(
                row[
                    "requires_approval"
                ]
            ),
            enabled=bool(
                row[
                    "enabled"
                ]
            ),
            status=str(
                row[
                    "status"
                ]
                or ""
            ),
            description=str(
                row[
                    "description"
                ]
                or ""
            ),
        )


# ============================================================
# Tool Call Helpers
# ============================================================

def _latest_ai_message(
    messages: Sequence[Any],
) -> AIMessage:

    for message in reversed(
        list(
            messages
        )
    ):

        if isinstance(
            message,
            AIMessage,
        ):
            return message

    raise RuntimeError(
        "没有找到最新 AIMessage。"
    )


def _tool_call_name(
    tool_call: Any,
) -> str:

    if isinstance(
        tool_call,
        dict,
    ):
        return str(
            tool_call.get(
                "name",
                "",
            )
        ).strip()

    return str(
        getattr(
            tool_call,
            "name",
            "",
        )
    ).strip()


def _tool_call_id(
    tool_call: Any,
) -> str:

    if isinstance(
        tool_call,
        dict,
    ):
        return str(
            tool_call.get(
                "id",
                "",
            )
        ).strip()

    return str(
        getattr(
            tool_call,
            "id",
            "",
        )
    ).strip()


def _tool_call_args(
    tool_call: Any,
) -> Any:

    if isinstance(
        tool_call,
        dict,
    ):
        return tool_call.get(
            "args",
            {},
        )

    return getattr(
        tool_call,
        "args",
        {},
    )


def _json_safe(
    value: Any,
) -> Any:

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    )


def _approval_is_granted(
    value: Any,
) -> bool:
    """Fail closed。

    只有明确 APPROVE 才允许执行。
    """

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        dict,
    ):

        if isinstance(
            value.get(
                "approved"
            ),
            bool,
        ):
            return bool(
                value[
                    "approved"
                ]
            )

        decision = str(
            value.get(
                "decision",
                "",
            )
        ).strip().upper()

        return decision in {
            "APPROVE",
            "APPROVED",
            "ALLOW",
            "YES",
            "Y",
        }

    decision = str(
        value
    ).strip().upper()

    return decision in {
        "APPROVE",
        "APPROVED",
        "ALLOW",
        "YES",
        "Y",
    }



def _context_retrieval_allowed(
    input_data: dict[str, Any],
) -> bool | None:
    """读取本轮 ContextPlan 的 Retrieval Permission。

    Returns
    -------
    True
        明确允许新的 Retrieval Tool。
    False
        明确禁止新的 Retrieval Tool。
    None
        没有启用 Context Pipeline / 缺少 ContextPlan，
        保持 legacy 行为，不额外施加 ContextPlan 约束。
    """

    if not bool(
        input_data.get(
            "context_pipeline_enabled",
            False,
        )
    ):
        return None

    raw_plan = input_data.get(
        "context_plan"
    )

    if not isinstance(
        raw_plan,
        dict,
    ):
        return None

    value = raw_plan.get(
        "external_retrieval_allowed"
    )

    if isinstance(
        value,
        bool,
    ):
        return value

    return None


def _batch_error_messages(
    *,
    tool_calls: Sequence[Any],
    reason_by_call_id: dict[str, str],
    default_reason: str,
) -> list[ToolMessage]:
    """对一个 Tool batch 中的每个 call 都返回 ToolMessage。

    这样即使其中一个调用被 Policy/HITL/ContextPlan 阻止，
    也不会留下兄弟 tool_call_id 未解析。
    """

    output: list[
        ToolMessage
    ] = []

    for call in tool_calls:
        tool_name = (
            _tool_call_name(
                call
            )
        )

        call_id = (
            _tool_call_id(
                call
            )
        )

        reason = str(
            reason_by_call_id.get(
                call_id,
                default_reason,
            )
        ).strip()

        output.append(
            ToolMessage(
                content=reason,
                tool_call_id=call_id,
                name=tool_name,
                status="error",
            )
        )

    return output


# ============================================================
# Secure ToolNode
# ============================================================

class SecureToolNode:
    """Policy + HITL 安全执行节点。"""

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        policy_store: SQLiteToolPolicyStore,
    ) -> None:

        self.tools = list(
            tools
        )

        self.policy_store = (
            policy_store
        )

        self.delegate = ToolNode(
            self.tools
        )

    def invoke(
        self,
        input_data: dict[str, Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:

        event_thread_id = str(
            input_data.get("working_memory_thread_id", "") or ""
        ).strip()

        messages = list(
            input_data.get(
                "messages",
                [],
            )
        )

        latest = (
            _latest_ai_message(
                messages
            )
        )

        tool_calls = list(
            latest.tool_calls
            or []
        )

        if not tool_calls:

            raise RuntimeError(
                "SecureToolNode 被调用，"
                "但当前 AIMessage "
                "没有 ToolCall。"
            )

        # ====================================================
        # Phase 7B：Batch Preflight
        # ====================================================
        #
        # 一批 Tool Call 在任何真实 Tool 执行之前完成：
        #
        # 1. Registry Policy
        # 2. ContextPlan Retrieval Permission
        # 3. HITL
        #
        # 只要一个 call 不允许执行，这一批全部不执行，
        # 并为每个 tool_call_id 返回 ToolMessage，
        # 保证 Tool Calling 协议完整。
        # ====================================================

        retrieval_allowed = (
            _context_retrieval_allowed(
                input_data
            )
        )

        policies: list[
            tuple[
                Any,
                ToolPolicySnapshot,
            ]
        ] = []

        preflight_errors: dict[
            str,
            str,
        ] = {}

        for call in tool_calls:

            tool_name = (
                _tool_call_name(
                    call
                )
            )

            call_id = (
                _tool_call_id(
                    call
                )
            )

            policy = (
                self.policy_store
                .get_policy(
                    tool_name
                )
            )

            if policy is None:

                preflight_errors[
                    call_id
                ] = (
                    "安全策略阻止了 Tool："
                    f"{tool_name}。\n"
                    "原因：Tool Policy Registry "
                    "中不存在该 Tool。\n"
                    "系统采用 Fail-Closed，"
                    "不会执行未知 Tool。"
                )

                continue

            if (
                not policy.enabled
                or
                policy.status.upper()
                != "ACTIVE"
            ):

                preflight_errors[
                    call_id
                ] = (
                    "安全策略阻止了 Tool："
                    f"{tool_name}。\n"
                    "Policy 状态："
                    f"{policy.status}\n"
                    "enabled="
                    f"{policy.enabled}"
                )

                continue

            # ------------------------------------------------
            # ContextPlan -> Tool Permission
            # ------------------------------------------------
            #
            # RETRIEVAL 不是“网络访问”或“外部副作用”；
            # 它表示读取当前 Conversation Context 之外的数据源。
            #
            # 因而本地 PDF RAG / SQLite SQL 也属于 RETRIEVAL。
            # ------------------------------------------------

            if (
                retrieval_allowed
                is False
                and
                policy
                .context_access_type
                .upper()
                == "RETRIEVAL"
            ):

                preflight_errors[
                    call_id
                ] = (
                    "ContextPlan 阻止了新的 Retrieval Tool："
                    f"{tool_name}。\n"
                    "本轮 external_retrieval_allowed=false。\n"
                    "请使用已经恢复的 Conversation History、"
                    "历史 Tool Evidence 或当前上下文回答，"
                    "不要重新检索外部数据源。"
                )

                continue

            policies.append(
                (
                    call,
                    policy,
                )
            )

        if preflight_errors:

            return {
                "messages":
                    _batch_error_messages(
                        tool_calls=(
                            tool_calls
                        ),
                        reason_by_call_id=(
                            preflight_errors
                        ),
                        default_reason=(
                            "本批 Tool Call 因同批次另一个调用"
                            "未通过安全/ContextPlan Preflight，"
                            "因此未执行。"
                        ),
                    )
            }

        # ====================================================
        # HITL Preflight
        # ====================================================

        for call, policy in policies:

            if not (
                policy.requires_approval
            ):
                continue

            tool_name = (
                _tool_call_name(
                    call
                )
            )

            call_id = (
                _tool_call_id(
                    call
                )
            )

            args = (
                _json_safe(
                    _tool_call_args(
                        call
                    )
                )
            )

            emit_runtime_event(
                "hitl_requested",
                {
                    "tool_name": tool_name,
                    "thread_id": event_thread_id,
                    "tool_call_id": call_id,
                    "message": f"工具 {tool_name} 正在等待人工审批。",
                },
            )

            decision = interrupt(
                {
                    "type":
                        "TOOL_APPROVAL_REQUIRED",

                    "tool_name":
                        tool_name,

                    "tool_call_id":
                        call_id,

                    "effect_type":
                        policy.effect_type,

                    "context_access_type":
                        policy.context_access_type,

                    "has_external_side_effect":
                        (
                            policy
                            .has_external_side_effect
                        ),

                    "requires_approval":
                        True,

                    "args":
                        args,

                    "description":
                        policy.description,

                    "message":
                        (
                            "该 Tool 属于高风险写操作，"
                            "必须人工批准后才能执行。"
                        ),
                }
            )

            if not (
                _approval_is_granted(
                    decision
                )
            ):

                reason_by_call_id = {
                    call_id: (
                        "Tool 调用已被人工拒绝："
                        f"{tool_name}。\n"
                        "实际 Tool 未执行。"
                    )
                }

                return {
                    "messages":
                        _batch_error_messages(
                            tool_calls=(
                                tool_calls
                            ),
                            reason_by_call_id=(
                                reason_by_call_id
                            ),
                            default_reason=(
                                "本批 Tool Call 因同批次审批未通过，"
                                "因此全部取消且未执行。"
                            ),
                        )
                }

        # ----------------------------------------------------
        # 所有 Policy / ContextPlan / HITL 均通过，
        # 到这里才真正执行 Tool。
        # ----------------------------------------------------

        # ----------------------------------------------------
        # LangGraph 1.x programmatic ToolNode invocation:
        #
        # SecureToolNode 本身不是直接注册到 StateGraph 的
        # 原生 ToolNode，而是在自定义 _tools_node 中再次
        # programmatically invoke 一个 ToolNode。
        #
        # 因此 Graph 不会自动为 delegate 注入 ToolRuntime。
        # 当前安装版本在缺少 Runtime 时会报：
        #
        #   Missing required config key 'N/A' for 'tools'
        #
        # 优先使用调用方显式传入的 runtime；
        # 没有时创建一个空 Runtime，保持当前无 Runtime Tool
        # 与普通 StructuredTool 的兼容。
        # ----------------------------------------------------

        delegate_runtime = (
            kwargs.get(
                "runtime"
            )
            or Runtime()
        )

        tool_names = [_tool_call_name(call) for call in tool_calls]
        emit_runtime_event(
            "tools_started",
            {
                "tool_names": tool_names,
                "thread_id": event_thread_id,
                "tool_count": len(tool_names),
                "message": "正在执行工具：" + "、".join(tool_names),
            },
        )
        try:
            if config is None:
                result = self.delegate.invoke(
                    input_data,
                    runtime=delegate_runtime,
                )
            else:
                result = self.delegate.invoke(
                    input_data,
                    config=config,
                    runtime=delegate_runtime,
                )
        except Exception as error:
            emit_runtime_event(
                "tools_failed",
                {
                    "tool_names": tool_names,
                    "thread_id": event_thread_id,
                    "message": "工具执行失败。",
                    "error_type": type(error).__name__,
                },
            )
            raise

        emit_runtime_event(
            "tools_completed",
            {
                "tool_names": tool_names,
                "thread_id": event_thread_id,
                "tool_count": len(tool_names),
                "message": "工具执行完成：" + "、".join(tool_names),
            },
        )
        return result


# ============================================================
# Secure Agent Runtime
# ============================================================

class SecureAgentRuntime:
    """给现有 Agent 增加 Policy/HITL Runtime。

    通过代理而不是重写原 Agent，
    保留：

    - Skill Runtime
    - Memory
    - Checkpointer
    - Replay
    - 原来的 graph
    """

    def __init__(
        self,
        base_agent: Any,
        *,
        policy_store: (
            SQLiteToolPolicyStore
            | None
        ) = None,
        conversation_event_store: (
            ConversationEventStore
            | None
        ) = None,
    ) -> None:

        self.base_agent = (
            base_agent
        )

        self.policy_store = (
            policy_store
            or SQLiteToolPolicyStore()
        )

        # ----------------------------------------------------
        # Context Tool Exposure Phase 7C
        # ----------------------------------------------------
        #
        # Agent 的 Context 层只依赖一个 resolver callable，
        # 不直接依赖 SQLiteToolPolicyStore 类型。
        #
        # 这样：
        #   Context Layer -> generic policy resolver
        #   Security Layer -> concrete Policy Store
        #
        # 避免双向 import。
        if hasattr(
            base_agent,
            "context_tool_policy_resolver",
        ):
            base_agent.context_tool_policy_resolver = (
                self.policy_store.get_policy
            )

        # ----------------------------------------------------
        # Raw Conversation Archive
        # ----------------------------------------------------
        #
        # Checkpoint 负责“当前 Graph 怎么恢复”；
        # Conversation Event Store 负责
        # “过去真实发生过哪些 Human / AI / Tool 消息”。
        #
        # 默认独立保存到：
        #
        # storage/agent_state/
        # raglab_conversation_events.sqlite3
        #
        # 测试时也可以显式注入临时 Event Store。
        inherited_event_store = getattr(
            base_agent,
            "conversation_event_store",
            None,
        )

        self.conversation_event_store = (
            conversation_event_store
            or inherited_event_store
            or ConversationEventStore()
        )

        # 让读取历史的 Base Agent 与负责归档的 Secure Runtime
        # 指向同一个 Event Store 实例。
        if hasattr(
            base_agent,
            "conversation_event_store",
        ):
            base_agent.conversation_event_store = (
                self.conversation_event_store
            )

            context_pipeline = getattr(
                base_agent,
                "context_pipeline",
                None,
            )

            if context_pipeline is not None:
                context_pipeline.event_store = (
                    self.conversation_event_store
                )

                retriever = getattr(
                    context_pipeline,
                    "retriever",
                    None,
                )

                if retriever is not None:
                    retriever.store = (
                        self.conversation_event_store
                    )

        self.last_conversation_archive_report: (
            ConversationArchiveReport
            | None
        ) = None

        # Phase 7E-3:
        # 每次新 Human Turn 前的 archive repair 结果。
        self.last_conversation_reconciliation_result: (
            ArchiveReconciliationResult
            | None
        ) = None

        self._install_secure_tool_refresh()

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        """其余能力全部代理给原 Agent。"""

        return getattr(
            self.base_agent,
            name,
        )

    # ========================================================
    # Conversation Event Archive
    # ========================================================

    def _reconcile_checkpoint_archive(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> ArchiveReconciliationResult:
        """Repair missing Event Store records from existing Checkpoint.

        This path is intentionally fail-open for conversation availability.
        The returned result records failure details.

        Working Memory deletion stays fail-closed because Phase 7E-2/7E-3
        still verifies each Turn before removing it.
        """

        result = (
            reconcile_checkpoint_archive(
                base_agent=(
                    self.base_agent
                ),
                store=(
                    self
                    .conversation_event_store
                ),
                user_id=(
                    user_id
                ),
                thread_id=(
                    thread_id
                ),
            )
        )

        self.last_conversation_reconciliation_result = (
            result
        )

        return result

    def _archive_result_messages(
        self,
        *,
        result: Any,
        user_id: str,
        thread_id: str,
    ) -> ConversationArchiveReport:
        """把一次 Agent Result 中当前可见 messages 幂等归档。

        这里故意放在 SecureAgentRuntime 边界，而不是各个 Tool 中：

        - 普通 RAG / SQL Tool 不需要知道归档逻辑；
        - Dynamic Skill Tool 不需要重复接入；
        - HITL resume 后仍走同一归档逻辑；
        - Event Store 与主 Agent Graph State 保持解耦。
        """

        normalized_user_id = (
            normalize_user_id(
                user_id
            )
        )

        normalized_thread_id = (
            normalize_thread_id(
                thread_id
            )
        )

        messages = list(
            getattr(
                result,
                "messages",
                [],
            )
            or []
        )

        report = (
            archive_messages_to_event_store(
                store=(
                    self.conversation_event_store
                ),
                user_id=(
                    normalized_user_id
                ),
                thread_id=(
                    normalized_thread_id
                ),
                messages=messages,
            )
        )

        self.last_conversation_archive_report = (
            report
        )

        return report

    # ========================================================
    # Tool Runtime Hook
    # ========================================================

    def _install_secure_tool_refresh(
        self,
    ) -> None:
        """保证动态 Skill Tool 也经过 SecureToolNode。

        当前 Agent 每次模型决策前会刷新
        Active Tools。

        所以不能只启动时替换一次 ToolNode，
        否则 load_skill 后新的 Tool 会重新生成
        普通 ToolNode，从而绕过安全层。
        """

        original_refresh = getattr(
            self.base_agent,
            "_refresh_tool_bindings",
            None,
        )

        if callable(
            original_refresh
        ):

            def secure_refresh():

                active_tools = (
                    original_refresh()
                )

                self.base_agent.tool_node = (
                    SecureToolNode(
                        active_tools,
                        policy_store=(
                            self.policy_store
                        ),
                    )
                )

                return active_tools

            # 实例级替换。
            self.base_agent._refresh_tool_bindings = (
                secure_refresh
            )

            # 立即包装一次当前 Tool。
            secure_refresh()

            return

        # ----------------------------------------------------
        # 兼容没有动态 Skill Runtime 的老版本。
        # ----------------------------------------------------

        current_tools = list(
            getattr(
                self.base_agent,
                "tools",
                [],
            )
        )

        if not current_tools:

            raise RuntimeError(
                "无法安装 SecureToolNode："
                "Agent 没有 tools。"
            )

        self.base_agent.tool_node = (
            SecureToolNode(
                current_tools,
                policy_store=(
                    self.policy_store
                ),
            )
        )

    # ========================================================
    # Pending Interrupt
    # ========================================================

    def get_pending_approval(
        self,
        thread_id: str,
    ) -> dict[str, Any] | None:

        normalized_thread_id = (
            normalize_thread_id(
                thread_id
            )
        )

        config = (
            self.base_agent
            ._build_config(
                normalized_thread_id
            )
        )

        snapshot = (
            self.base_agent
            .graph
            .get_state(
                config
            )
        )

        tasks = list(
            getattr(
                snapshot,
                "tasks",
                (),
            )
            or ()
        )

        interrupts: list[
            Any
        ] = []

        for task in tasks:

            task_interrupts = list(
                getattr(
                    task,
                    "interrupts",
                    (),
                )
                or ()
            )

            for current in (
                task_interrupts
            ):

                value = getattr(
                    current,
                    "value",
                    current,
                )

                interrupts.append(
                    value
                )

        if not interrupts:
            return None

        return {
            "thread_id":
                normalized_thread_id,

            "interrupts":
                _json_safe(
                    interrupts
                ),
        }

    # ========================================================
    # Run / Resume
    # ========================================================

    def run(
        self,
        question: str,
        *,
        thread_id: str,
        user_id: str,
    ) -> PersistentLangGraphResult:

        emit_runtime_event(
            "runtime_started",
            {
                "thread_id": thread_id,
                "message": "Secure Agent Runtime 已开始执行。",
            },
        )

        normalized_question = str(
            question
        ).strip()

        command = (
            normalized_question
            .lower()
        )

        if command == "/approve":

            return self._resume(
                thread_id=thread_id,
                user_id=user_id,
                approved=True,
                question=(
                    normalized_question
                ),
            )

        if command == "/reject":

            return self._resume(
                thread_id=thread_id,
                user_id=user_id,
                approved=False,
                question=(
                    normalized_question
                ),
            )

        # ----------------------------------------------------
        # Pending HITL Guard
        # ----------------------------------------------------
        #
        # 当同一个 thread 已经停在 interrupt() 上等待
        # Tool Approval 时，不能继续追加新的 HumanMessage。
        #
        # 否则消息会从：
        #
        #     AIMessage(tool_calls=[call_x])
        #
        # 直接变成：
        #
        #     AIMessage(tool_calls=[call_x])
        #     HumanMessage(...)
        #
        # 中间缺少与 call_x 对应的 ToolMessage，下一次模型
        # 调用会违反 Tool Calling 协议并触发 400。
        #
        # /approve 与 /reject 已经在上方被优先处理，因此
        # 这里仅拦截“等待审批时的新普通消息”。

        normalized_thread_id = (
            normalize_thread_id(
                thread_id
            )
        )

        pending = (
            self.get_pending_approval(
                normalized_thread_id
            )
        )

        if pending is not None:

            emit_runtime_event(
                "runtime_blocked",
                {
                    "thread_id": normalized_thread_id,
                    "message": "当前会话仍有等待处理的 HITL 中断。",
                },
            )

            pending_tool_names: list[str] = []

            for current in (
                pending.get(
                    "interrupts",
                    [],
                )
                or []
            ):

                if not isinstance(
                    current,
                    dict,
                ):
                    continue

                tool_name = str(
                    current.get(
                        "tool_name",
                        "",
                    )
                    or ""
                ).strip()

                if (
                    tool_name
                    and tool_name
                    not in pending_tool_names
                ):
                    pending_tool_names.append(
                        tool_name
                    )

            pending_tools_text = (
                ", ".join(
                    pending_tool_names
                )
                if pending_tool_names
                else "未知 Tool"
            )

            raise RuntimeError(
                "当前 thread 存在等待中的 "
                "Tool Approval，不能继续追加新的普通消息。"
                f"thread_id={normalized_thread_id}；"
                "等待审批 Tool="
                f"{pending_tools_text}。"
                "请先输入 /approve 或 /reject "
                "完成当前审批。"
            )

        # ----------------------------------------------------
        # Phase 7E-3 Archive Reconciliation
        # ----------------------------------------------------
        #
        # 在新 Human Turn 真正进入 Graph 之前，
        # 用“上一份 Checkpoint Working Messages”
        # 对 Event Store 做一次幂等补归档。
        #
        # - 已存在 Event：自动跳过；
        # - 缺失 Event：补写；
        # - Repair 失败：不阻断正常对话，
        #   但后面的 Working Memory Compaction
        #   会继续 Pin 未验证 Turn。
        self._reconcile_checkpoint_archive(
            user_id=user_id,
            thread_id=(
                normalized_thread_id
            ),
        )

        emit_runtime_event(
            "graph_started",
            {
                "thread_id": normalized_thread_id,
                "message": "LangGraph 开始执行 Agent 循环。",
            },
        )

        result = self.base_agent.run(
            normalized_question,
            thread_id=(
                normalized_thread_id
            ),
            user_id=user_id,
        )

        self._archive_result_messages(
            result=result,
            user_id=user_id,
            thread_id=(
                normalized_thread_id
            ),
        )

        emit_runtime_event(
            "graph_completed",
            {
                "thread_id": normalized_thread_id,
                "tool_calls": int(getattr(result, "turn_tool_call_count", 0)),
                "llm_calls": int(getattr(result, "turn_llm_call_count", 0)),
                "message": "LangGraph 本轮执行完成。",
            },
        )

        return result

    def _resume(
        self,
        *,
        thread_id: str,
        user_id: str,
        approved: bool,
        question: str,
    ) -> PersistentLangGraphResult:

        normalized_thread_id = (
            normalize_thread_id(
                thread_id
            )
        )

        normalized_user_id = (
            normalize_user_id(
                user_id
            )
        )

        pending = (
            self.get_pending_approval(
                normalized_thread_id
            )
        )

        if pending is None:

            raise RuntimeError(
                "当前 thread 没有等待中的 "
                "HITL interrupt。"
            )

        emit_runtime_event(
            "hitl_resume_started",
            {
                "thread_id": normalized_thread_id,
                "approved": approved,
                "message": (
                    "审批已通过，正在从 LangGraph Checkpoint 恢复执行。"
                    if approved
                    else "审批已拒绝，正在从 LangGraph Checkpoint 恢复执行。"
                ),
            },
        )

        # Phase 7E-3:
        # HITL resume 前也补一次当前 Checkpoint。
        self._reconcile_checkpoint_archive(
            user_id=(
                normalized_user_id
            ),
            thread_id=(
                normalized_thread_id
            ),
        )

        config = (
            self.base_agent
            ._build_config(
                normalized_thread_id
            )
        )

        resume_payload = {
            "decision": (
                "APPROVE"
                if approved
                else "REJECT"
            ),
            "approved": (
                approved
            ),
            "actor": (
                normalized_user_id
            ),
        }

        start_time = (
            time.perf_counter()
        )

        # ====================================================
        # 真正恢复 LangGraph Checkpoint
        # ====================================================

        final_state = (
            self.base_agent
            .graph
            .invoke(
                Command(
                    resume=(
                        resume_payload
                    )
                ),
                config=config,
                context=(
                    LongTermMemoryContext(
                        user_id=(
                            normalized_user_id
                        )
                    )
                ),
            )
        )

        total_latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        result = self._build_result(
            final_state=(
                final_state
            ),
            thread_id=(
                normalized_thread_id
            ),
            question=(
                question
            ),
            latency_ms=(
                total_latency_ms
            ),
        )

        self._archive_result_messages(
            result=result,
            user_id=(
                normalized_user_id
            ),
            thread_id=(
                normalized_thread_id
            ),
        )

        emit_runtime_event(
            "hitl_resume_completed",
            {
                "thread_id": normalized_thread_id,
                "approved": approved,
                "message": "HITL 中断恢复流程已完成。",
            },
        )

        return result

    # ========================================================
    # Result
    # ========================================================

    @staticmethod
    def _build_result(
        *,
        final_state: dict[str, Any],
        thread_id: str,
        question: str,
        latency_ms: float,
    ) -> PersistentLangGraphResult:

        messages = list(
            final_state.get(
                "messages",
                [],
            )
        )

        answer = ""

        if messages:

            try:

                answer = (
                    extract_answer_text(
                        messages[-1]
                    ).strip()
                )

            except Exception:
                answer = ""

        stopped_by_max_steps = bool(
            final_state.get(
                "stopped_by_max_steps",
                False,
            )
        )

        return PersistentLangGraphResult(
            thread_id=thread_id,

            question=question,

            answer=answer,

            messages=messages,

            summary=str(
                final_state.get(
                    "summary",
                    "",
                )
            ).strip(),

            turn_llm_call_count=int(
                final_state.get(
                    "turn_llm_calls",
                    0,
                )
            ),

            turn_tool_call_count=int(
                final_state.get(
                    "turn_tool_calls",
                    0,
                )
            ),

            turn_summary_call_count=int(
                final_state.get(
                    "turn_summary_calls",
                    0,
                )
            ),

            summary_updated=bool(
                final_state.get(
                    "summary_updated",
                    False,
                )
            ),

            summarized_turns_this_run=int(
                final_state.get(
                    "summarized_turns_this_run",
                    0,
                )
            ),

            total_summarized_turns=int(
                final_state.get(
                    "total_summarized_turns",
                    0,
                )
            ),

            recent_turn_count=(
                count_human_turns(
                    messages
                )
            ),

            stopped_by_max_steps=(
                stopped_by_max_steps
            ),

            completed_normally=(
                not stopped_by_max_steps
            ),

            model_trace=list(
                final_state.get(
                    "model_trace",
                    [],
                )
            ),

            tool_trace=list(
                final_state.get(
                    "tool_trace",
                    [],
                )
            ),

            total_message_count=len(
                messages
            ),

            total_latency_ms=float(
                latency_ms
            ),

            final_state=dict(
                final_state
            ),
        )
