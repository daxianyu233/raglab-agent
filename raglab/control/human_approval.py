"""Human-in-the-Loop Tool Approval 数据模型。"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ApprovalDecision(
    str,
    Enum,
):
    """人工审批决定。"""

    APPROVE = "APPROVE"
    REJECT = "REJECT"


class ApprovalAuditEventType(
    str,
    Enum,
):
    """审批审计事件。"""

    REQUESTED = "REQUESTED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(
    frozen=True,
)
class ApprovalDecisionResult:
    """解析后的人工审批结果。"""

    decision: ApprovalDecision
    actor: str
    reason: str


def build_approval_id(
    operation_key: str,
) -> str:
    """根据具体 Tool Operation 生成稳定 approval_id。"""

    normalized = str(
        operation_key
    ).strip()

    if not normalized:
        raise ValueError(
            "operation_key 不能为空。"
        )

    raw = (
        "approval|"
        + normalized
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:32]


def parse_approval_resume(
    value: Any,
) -> ApprovalDecisionResult:
    """解析 Command(resume=...) 返回值。

    推荐格式：

    {
        "decision": "APPROVE",
        "actor": "huangwu",
        "reason": "确认执行"
    }
    """

    if isinstance(
        value,
        bool,
    ):

        return ApprovalDecisionResult(
            decision=(
                ApprovalDecision.APPROVE
                if value
                else ApprovalDecision.REJECT
            ),
            actor="human",
            reason="",
        )

    if isinstance(
        value,
        str,
    ):

        normalized = (
            value.strip().upper()
        )

        if normalized in {
            "APPROVE",
            "APPROVED",
            "YES",
            "Y",
        }:

            return ApprovalDecisionResult(
                decision=(
                    ApprovalDecision.APPROVE
                ),
                actor="human",
                reason="",
            )

        if normalized in {
            "REJECT",
            "REJECTED",
            "NO",
            "N",
        }:

            return ApprovalDecisionResult(
                decision=(
                    ApprovalDecision.REJECT
                ),
                actor="human",
                reason="",
            )

        raise ValueError(
            "无法识别审批结果："
            f"{value}"
        )

    if not isinstance(
        value,
        dict,
    ):

        raise TypeError(
            "审批恢复值必须是 dict、bool 或 str。"
        )

    decision_text = str(
        value.get(
            "decision",
            "",
        )
    ).strip().upper()

    if decision_text in {
        "APPROVE",
        "APPROVED",
        "YES",
    }:

        decision = (
            ApprovalDecision.APPROVE
        )

    elif decision_text in {
        "REJECT",
        "REJECTED",
        "NO",
    }:

        decision = (
            ApprovalDecision.REJECT
        )

    else:

        raise ValueError(
            "审批 decision 必须为 "
            "APPROVE 或 REJECT。"
        )

    actor = str(
        value.get(
            "actor",
            "human",
        )
    ).strip()

    if not actor:

        actor = "human"

    reason = str(
        value.get(
            "reason",
            "",
        )
    ).strip()

    return ApprovalDecisionResult(
        decision=decision,
        actor=actor,
        reason=reason,
    )