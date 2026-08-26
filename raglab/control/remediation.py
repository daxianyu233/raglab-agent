"""不可自动补偿外部操作的人工修复数据模型。

External Effect Ledger 负责保存：

    “真实发生过什么”

Remediation Case 负责保存：

    “后来人工怎么处理”

两者必须分开。

Effect 是审计事实，不应该因为人工处理而被覆盖或删除。

Remediation 可以持续追加：

- 开始处理；
- 调查结果；
- 修复说明；
- 风险接受；
- 最终关闭。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ============================================================
# 修复动作类型
# ============================================================


class RemediationActionType(
    str,
    Enum,
):
    """人工后续处理类型。"""

    # --------------------------------------------------------
    # 首先需要调查外部真实状态。
    #
    # 常见于：
    #
    # UNKNOWN
    # EXECUTING 后进程异常退出
    #
    # 当前甚至无法确定原操作是否真正生效。
    # --------------------------------------------------------

    INVESTIGATE = "INVESTIGATE"

    # --------------------------------------------------------
    # 原操作已经发生且无法撤销，
    # 但可以进行后续纠正。
    #
    # 例如：
    #
    # 错误邮件已经发送
    # → 再发送更正邮件
    #
    # 错误数据已经发布
    # → 发布修正版
    # --------------------------------------------------------

    CORRECTIVE_ACTION = (
        "CORRECTIVE_ACTION"
    )

    # --------------------------------------------------------
    # 需要人工根据业务情况处理，
    # 没有统一自动操作。
    # --------------------------------------------------------

    MANUAL_FIX = "MANUAL_FIX"


# ============================================================
# 优先级
# ============================================================


class RemediationPriority(
    str,
    Enum,
):

    LOW = "LOW"

    MEDIUM = "MEDIUM"

    HIGH = "HIGH"

    CRITICAL = "CRITICAL"


# ============================================================
# Case 状态
# ============================================================


class RemediationStatus(
    str,
    Enum,
):
    """人工修复工单生命周期。"""

    # 已发现问题，尚未处理。
    OPEN = "OPEN"

    # 人工正在处理。
    IN_PROGRESS = "IN_PROGRESS"

    # 已经完成修复。
    RESOLVED = "RESOLVED"

    # 无法/无需进一步处理，
    # 人工明确接受剩余风险。
    ACCEPTED_RISK = (
        "ACCEPTED_RISK"
    )


# ============================================================
# Feedback 类型
# ============================================================


class RemediationFeedbackType(
    str,
    Enum,
):
    """人工反馈历史类型。"""

    SYSTEM_CREATED = (
        "SYSTEM_CREATED"
    )

    NOTE = "NOTE"

    STATUS_CHANGE = (
        "STATUS_CHANGE"
    )

    RESOLUTION = "RESOLUTION"

    RISK_ACCEPTANCE = (
        "RISK_ACCEPTANCE"
    )

    REOPEN = "REOPEN"


# ============================================================
# Case
# ============================================================


@dataclass(
    frozen=True,
)
class RemediationCase:
    """一个需要人工处理的外部副作用问题。"""

    case_id: str

    plan_id: str

    reconciliation_item_id: str

    effect_id: str

    thread_id: str

    tool_name: str

    action_type: (
        RemediationActionType
    )

    priority: (
        RemediationPriority
    )

    status: (
        RemediationStatus
    )

    summary: str

    reason: str

    owner: str | None

    resolution_note: str | None

    created_at: str

    updated_at: str

    started_at: str | None

    resolved_at: str | None

    accepted_risk_at: (
        str
        | None
    )

    @property
    def is_open(
        self,
    ) -> bool:

        return self.status in {
            RemediationStatus.OPEN,
            RemediationStatus.IN_PROGRESS,
        }

    @property
    def is_closed(
        self,
    ) -> bool:

        return self.status in {
            RemediationStatus.RESOLVED,
            RemediationStatus.ACCEPTED_RISK,
        }

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "case_id": (
                self.case_id
            ),
            "plan_id": (
                self.plan_id
            ),
            "reconciliation_item_id": (
                self.reconciliation_item_id
            ),
            "effect_id": (
                self.effect_id
            ),
            "thread_id": (
                self.thread_id
            ),
            "tool_name": (
                self.tool_name
            ),
            "action_type": (
                self.action_type.value
            ),
            "priority": (
                self.priority.value
            ),
            "status": (
                self.status.value
            ),
            "summary": (
                self.summary
            ),
            "reason": (
                self.reason
            ),
            "owner": (
                self.owner
            ),
            "resolution_note": (
                self.resolution_note
            ),
            "created_at": (
                self.created_at
            ),
            "updated_at": (
                self.updated_at
            ),
            "started_at": (
                self.started_at
            ),
            "resolved_at": (
                self.resolved_at
            ),
            "accepted_risk_at": (
                self.accepted_risk_at
            ),
        }


# ============================================================
# Feedback
# ============================================================


@dataclass(
    frozen=True,
)
class RemediationFeedback:
    """人工处理过程中的一条追加记录。"""

    feedback_id: str

    case_id: str

    feedback_type: (
        RemediationFeedbackType
    )

    actor: str

    message: str

    created_at: str

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "feedback_id": (
                self.feedback_id
            ),
            "case_id": (
                self.case_id
            ),
            "feedback_type": (
                self.feedback_type.value
            ),
            "actor": (
                self.actor
            ),
            "message": (
                self.message
            ),
            "created_at": (
                self.created_at
            ),
        }