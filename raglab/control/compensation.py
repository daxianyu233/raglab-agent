"""External Effect Compensation。"""

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any

from raglab.control.external_effect import (
    ExternalEffectRecord,
    ExternalEffectStatus,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
)

from raglab.control.tool_policy import (
    ToolEffectType,
)

from raglab.control.tool_policy_repository import (
    ToolPolicyRepository,
)


class CompensationExecutionUncertainError(
    RuntimeError
):
    """补偿调用结果无法确认。"""


@dataclass(
    frozen=True,
)
class CompensationResult:
    """一次补偿结果。"""

    effect_id: str

    original_tool: str

    compensation_tool: str

    status: str

    result_text: str


def result_to_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    if isinstance(
        value,
        str,
    ):
        return value

    try:

        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    except Exception:

        return str(
            value
        )


class ExternalEffectCompensationManager:
    """External Effect 补偿管理器。

    补偿 Tool 统一约定：

        compensation_tool(
            effect_id: str
        )

    Tool 自己根据 effect_id
    读取 External Effect Ledger，
    获取：

        原 Tool
        原参数
        原结果
        thread_id
        checkpoint_id

    再执行具体业务补偿。

    因此中央控制层不需要知道：

        GitHub 怎么撤销
        文件怎么恢复
        数据库怎么还原
        订单怎么取消
    """

    def __init__(
        self,
        *,
        agent: Any,
        policy_repository: (
            ToolPolicyRepository
        ),
        effect_repository: (
            ExternalEffectRepository
        ),
    ) -> None:

        self.agent = agent

        self.policy_repository = (
            policy_repository
        )

        self.effect_repository = (
            effect_repository
        )

    def compensate(
        self,
        effect_id: str,
    ) -> CompensationResult:

        effect = (
            self.effect_repository.get(
                effect_id
            )
        )

        if effect is None:

            raise KeyError(
                f"Effect 不存在：{effect_id}"
            )

        if (
            effect.status
            != ExternalEffectStatus.SUCCEEDED
        ):

            raise ValueError(
                "只有 SUCCEEDED Effect "
                "可以开始补偿。\n"
                f"当前状态："
                f"{effect.status.value}"
            )

        if (
            effect.effect_type
            != ToolEffectType.COMPENSATABLE_WRITE
        ):

            raise ValueError(
                "该 Effect 不是 "
                "COMPENSATABLE_WRITE："
                f"{effect.effect_type.value}"
            )

        compensation_tool_name = (
            str(
                effect.compensation_tool
                or ""
            ).strip()
        )

        if not compensation_tool_name:

            raise ValueError(
                "该 Effect 没有配置 "
                "compensation_tool。"
            )

        compensation_policy = (
            self.policy_repository.get(
                compensation_tool_name
            )
        )

        if compensation_policy is None:

            raise ValueError(
                "补偿 Tool 尚未进入 "
                "Tool Policy Registry："
                f"{compensation_tool_name}"
            )

        if (
            not compensation_policy
            .is_executable
        ):

            raise ValueError(
                "补偿 Tool 当前不可执行："
                f"{compensation_tool_name}"
            )

        if (
            not compensation_policy
            .is_write
        ):

            raise ValueError(
                "补偿 Tool 必须被标记为写操作："
                f"{compensation_tool_name}"
            )

        # ----------------------------------------------------
        # 刷新当前动态 Tool。
        # ----------------------------------------------------

        active_tools = (
            self.agent
            ._refresh_tool_bindings()
        )

        tool_map = {
            str(
                tool.name
            ): tool
            for tool
            in active_tools
        }

        compensation_tool = (
            tool_map.get(
                compensation_tool_name
            )
        )

        if compensation_tool is None:

            raise RuntimeError(
                "补偿 Tool 已有 Policy，"
                "但当前 Agent Runtime 中没有加载："
                f"{compensation_tool_name}\n"
                "如果它属于某个 Skill，"
                "请先加载对应 Skill。"
            )

        self.effect_repository.mark_compensating(
            effect.effect_id
        )

        try:

            # ------------------------------------------------
            # 补偿 Tool 的统一协议：
            #
            # 只给 effect_id。
            #
            # 具体业务 Tool 自己读取 Ledger。
            # ------------------------------------------------

            result = (
                compensation_tool.invoke(
                    {
                        "effect_id": (
                            effect.effect_id
                        )
                    }
                )
            )

        except Exception as exc:

            self.effect_repository.mark_compensation_unknown(
                effect.effect_id,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            raise CompensationExecutionUncertainError(
                "补偿 Tool 调用出现异常，"
                "无法确定补偿是否已经作用于外部系统。\n"
                f"effect_id={effect.effect_id}\n"
                f"compensation_tool="
                f"{compensation_tool_name}"
            ) from exc

        text = result_to_text(
            result
        )

        compensated = (
            self.effect_repository
            .mark_compensated(
                effect.effect_id,
                text,
            )
        )

        return CompensationResult(
            effect_id=(
                compensated.effect_id
            ),
            original_tool=(
                effect.tool_name
            ),
            compensation_tool=(
                compensation_tool_name
            ),
            status=(
                compensated.status.value
            ),
            result_text=text,
        )