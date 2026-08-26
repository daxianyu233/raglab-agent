"""Tool Policy 核心数据模型。

本模块只定义：

    系统认识哪些安全类型，
    每条 Tool Policy 应包含哪些字段。

它不负责：

    某个具体 Tool 属于什么等级。

例如：

    READ_ONLY
    IDEMPOTENT_WRITE
    COMPENSATABLE_WRITE
    IRREVERSIBLE_WRITE

这些是系统级语义，因此应该由代码定义。

但是：

    search_knowledge_base
    update_github_intelligence
    send_email

分别属于哪一种，则由 SQLite
tool_policy_registry 管理。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ============================================================
# Tool 副作用类型
# ============================================================


class ToolEffectType(
    str,
    Enum,
):
    """Tool 对系统状态产生的副作用类型。"""

    READ_ONLY = (
        "READ_ONLY"
    )

    IDEMPOTENT_WRITE = (
        "IDEMPOTENT_WRITE"
    )

    COMPENSATABLE_WRITE = (
        "COMPENSATABLE_WRITE"
    )

    IRREVERSIBLE_WRITE = (
        "IRREVERSIBLE_WRITE"
    )


# ============================================================
# Replay 策略
# ============================================================


class ReplayPolicy(
    str,
    Enum,
):
    """Checkpoint Replay 时的 Tool 策略。"""

    # 可以重新执行。
    ALLOW = "ALLOW"

    # 需要先进行幂等性 /
    # Effect Ledger 等检查。
    GUARDED = "GUARDED"

    # 重新执行前必须人工审批。
    REQUIRE_APPROVAL = (
        "REQUIRE_APPROVAL"
    )

    # Replay 时禁止重新执行。
    DENY = "DENY"


# ============================================================
# Tool Policy 状态
# ============================================================


class ToolPolicyStatus(
    str,
    Enum,
):
    """Tool 在安全控制面中的状态。"""

    # Tool 已被发现，
    # 但尚未完成安全分类。
    PENDING = "PENDING"

    # 已完成安全分类，
    # 可以根据 enabled 决定是否暴露。
    ACTIVE = "ACTIVE"

    # 明确禁止 Agent 使用。
    BLOCKED = "BLOCKED"


# ============================================================
# Tool Policy Record
# ============================================================


@dataclass(
    frozen=True,
)
class ToolPolicyRecord:
    """tool_policy_registry 中的一条记录。"""

    tool_name: str

    tool_source: str

    source_id: str | None

    effect_type: (
        ToolEffectType
        | None
    )

    has_external_side_effect: (
        bool
        | None
    )

    replay_policy: (
        ReplayPolicy
        | None
    )

    requires_approval: bool

    idempotency_strategy: (
        str
        | None
    )

    compensation_tool: (
        str
        | None
    )

    enabled: bool

    status: ToolPolicyStatus

    description: str

    discovered_at: str

    last_seen_at: str

    updated_at: str

    @property
    def is_classified(
        self,
    ) -> bool:
        """是否已经完成安全分类。"""

        return (
            self.effect_type
            is not None
            and self.replay_policy
            is not None
            and self.has_external_side_effect
            is not None
        )

    @property
    def is_executable(
        self,
    ) -> bool:
        """当前是否允许进入 Agent Tool 集合。"""

        return (
            self.status
            == ToolPolicyStatus.ACTIVE
            and self.enabled
            and self.is_classified
        )

    @property
    def is_read_only(
        self,
    ) -> bool:
        """是否为纯只读 Tool。"""

        return (
            self.effect_type
            == ToolEffectType.READ_ONLY
        )

    @property
    def is_write(
        self,
    ) -> bool:
        """是否属于写操作。"""

        return (
            self.effect_type
            in {
                ToolEffectType.IDEMPOTENT_WRITE,
                ToolEffectType.COMPENSATABLE_WRITE,
                ToolEffectType.IRREVERSIBLE_WRITE,
            }
        )

    @property
    def requires_effect_record(
        self,
    ) -> bool:
        """是否应该进入 External Effect Ledger。

        Effect Ledger 会在下一阶段实现。
        """

        return bool(
            self.has_external_side_effect
            and self.is_write
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """转换成普通字典。"""

        return {
            "tool_name": (
                self.tool_name
            ),
            "tool_source": (
                self.tool_source
            ),
            "source_id": (
                self.source_id
            ),
            "effect_type": (
                self.effect_type.value
                if self.effect_type
                is not None
                else None
            ),
            "has_external_side_effect": (
                self.has_external_side_effect
            ),
            "replay_policy": (
                self.replay_policy.value
                if self.replay_policy
                is not None
                else None
            ),
            "requires_approval": (
                self.requires_approval
            ),
            "idempotency_strategy": (
                self.idempotency_strategy
            ),
            "compensation_tool": (
                self.compensation_tool
            ),
            "enabled": (
                self.enabled
            ),
            "status": (
                self.status.value
            ),
            "description": (
                self.description
            ),
            "discovered_at": (
                self.discovered_at
            ),
            "last_seen_at": (
                self.last_seen_at
            ),
            "updated_at": (
                self.updated_at
            ),
        }


# ============================================================
# Normalize Helpers
# ============================================================


def normalize_tool_name(
    tool_name: str,
) -> str:
    """规范化 Tool name。"""

    normalized = str(
        tool_name
    ).strip()

    if not normalized:
        raise ValueError(
            "tool_name 不能为空。"
        )

    return normalized


def normalize_tool_source(
    tool_source: str,
) -> str:
    """规范化 Tool 来源。"""

    normalized = str(
        tool_source
    ).strip()

    if not normalized:
        return "unknown"

    return normalized


def parse_effect_type(
    value: str | ToolEffectType | None,
) -> ToolEffectType | None:
    """解析 ToolEffectType。"""

    if value is None:
        return None

    if isinstance(
        value,
        ToolEffectType,
    ):
        return value

    normalized = str(
        value
    ).strip().upper()

    if not normalized:
        return None

    return ToolEffectType(
        normalized
    )


def parse_replay_policy(
    value: str | ReplayPolicy | None,
) -> ReplayPolicy | None:
    """解析 ReplayPolicy。"""

    if value is None:
        return None

    if isinstance(
        value,
        ReplayPolicy,
    ):
        return value

    normalized = str(
        value
    ).strip().upper()

    if not normalized:
        return None

    return ReplayPolicy(
        normalized
    )


def parse_policy_status(
    value: (
        str
        | ToolPolicyStatus
    ),
) -> ToolPolicyStatus:
    """解析 ToolPolicyStatus。"""

    if isinstance(
        value,
        ToolPolicyStatus,
    ):
        return value

    normalized = str(
        value
    ).strip().upper()

    return ToolPolicyStatus(
        normalized
    )