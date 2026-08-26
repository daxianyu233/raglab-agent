"""External Effect Ledger 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from raglab.control.tool_policy import (
    ReplayPolicy,
    ToolEffectType,
)


class ExternalEffectStatus(
    str,
    Enum,
):
    """一次外部副作用的生命周期。"""

    # 已经写 Ledger，
    # 真实外部调用尚未开始。
    PREPARED = "PREPARED"

    # 已经准备调用外部系统。
    #
    # 如果程序在这里崩溃，
    # 下一次不能假设外部操作没发生。
    EXECUTING = "EXECUTING"

    # 外部操作成功。
    SUCCEEDED = "SUCCEEDED"

    # 可以确认没有成功产生目标效果。
    FAILED = "FAILED"

    # 无法确认外部世界是否已经改变。
    UNKNOWN = "UNKNOWN"

    # 正在执行补偿。
    COMPENSATING = "COMPENSATING"

    # 补偿完成。
    COMPENSATED = "COMPENSATED"

    # 补偿调用过程中断，
    # 无法判断补偿到底执行到了什么程度。
    COMPENSATION_UNKNOWN = (
        "COMPENSATION_UNKNOWN"
    )


@dataclass(
    frozen=True,
)
class ExternalEffectRecord:
    """一次实际外部副作用记录。"""

    effect_id: str

    operation_key: str

    thread_id: str

    user_id: str

    checkpoint_id: (
        str
        | None
    )

    replay_from_checkpoint_id: (
        str
        | None
    )

    execution_mode: str

    tool_name: str

    tool_call_id: str

    effect_type: ToolEffectType

    replay_policy: ReplayPolicy

    args_json: str

    args_hash: str

    status: ExternalEffectStatus

    result_text: (
        str
        | None
    )

    error_text: (
        str
        | None
    )

    compensation_tool: (
        str
        | None
    )

    compensation_result_text: (
        str
        | None
    )

    compensation_error_text: (
        str
        | None
    )

    created_at: str

    updated_at: str

    execution_started_at: (
        str
        | None
    )

    succeeded_at: (
        str
        | None
    )

    compensated_at: (
        str
        | None
    )

    @property
    def is_uncertain(
        self,
    ) -> bool:
        """是否处于外部状态不确定状态。"""

        return self.status in {
            ExternalEffectStatus.EXECUTING,
            ExternalEffectStatus.UNKNOWN,
            ExternalEffectStatus.COMPENSATING,
            ExternalEffectStatus.COMPENSATION_UNKNOWN,
        }

    @property
    def can_compensate(
        self,
    ) -> bool:
        """当前是否具备执行补偿的基本条件。"""

        return (
            self.status
            == ExternalEffectStatus.SUCCEEDED
            and self.effect_type
            == ToolEffectType.COMPENSATABLE_WRITE
            and bool(
                self.compensation_tool
            )
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """转换为普通字典。"""

        return {
            "effect_id": self.effect_id,
            "operation_key": (
                self.operation_key
            ),
            "thread_id": (
                self.thread_id
            ),
            "user_id": self.user_id,
            "checkpoint_id": (
                self.checkpoint_id
            ),
            "replay_from_checkpoint_id": (
                self.replay_from_checkpoint_id
            ),
            "execution_mode": (
                self.execution_mode
            ),
            "tool_name": (
                self.tool_name
            ),
            "tool_call_id": (
                self.tool_call_id
            ),
            "effect_type": (
                self.effect_type.value
            ),
            "replay_policy": (
                self.replay_policy.value
            ),
            "args_json": (
                self.args_json
            ),
            "args_hash": (
                self.args_hash
            ),
            "status": (
                self.status.value
            ),
            "result_text": (
                self.result_text
            ),
            "error_text": (
                self.error_text
            ),
            "compensation_tool": (
                self.compensation_tool
            ),
            "compensation_result_text": (
                self.compensation_result_text
            ),
            "compensation_error_text": (
                self.compensation_error_text
            ),
            "created_at": (
                self.created_at
            ),
            "updated_at": (
                self.updated_at
            ),
            "execution_started_at": (
                self.execution_started_at
            ),
            "succeeded_at": (
                self.succeeded_at
            ),
            "compensated_at": (
                self.compensated_at
            ),
        }