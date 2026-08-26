"""Effect-aware ToolNode。

职责：

    READ_ONLY / 内部 Runtime Tool
        → 正常执行

    外部 WRITE Tool
        → 先写 Effect Ledger
        → 再调用真实 Tool
        → 保存结果

如果发现相同 operation_key
已经 SUCCEEDED：

    不再次调用真实外部系统
    直接复用第一次结果。

如果之前状态 UNKNOWN：

    不自动重试，
    防止重复副作用。
"""

from __future__ import annotations

import json

from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from langchain_core.tools import (
    BaseTool,
)

from langgraph.prebuilt import (
    ToolNode,
)

from raglab.control.execution_context import (
    get_effect_execution_context,
)

from raglab.control.external_effect import (
    ExternalEffectStatus,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
    build_operation_key,
    calculate_text_hash,
    serialize_arguments,
)

from raglab.control.tool_policy_repository import (
    ToolPolicyRepository,
)


class ExternalEffectUncertainError(
    RuntimeError
):
    """外部操作可能已经执行，但结果无法确认。"""


def message_content_to_text(
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


def tool_call_name(
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


def tool_call_id(
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


def tool_call_args(
    tool_call: Any,
) -> dict[str, Any]:

    if isinstance(
        tool_call,
        dict,
    ):

        value = tool_call.get(
            "args",
            {},
        )

    else:

        value = getattr(
            tool_call,
            "args",
            {},
        )

    if isinstance(
        value,
        dict,
    ):
        return dict(
            value
        )

    return {
        "value": value
    }


class EffectAwareToolNode:
    """兼容当前 Agent self.tool_node.invoke() 接口。"""

    def __init__(
        self,
        *,
        agent: Any,
        tools: Sequence[
            BaseTool
        ],
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

        self.tools = list(
            tools
        )

        self.tool_map = {
            str(
                tool.name
            ): tool
            for tool
            in self.tools
        }

        self.single_tool_nodes = {
            str(
                tool.name
            ): ToolNode(
                [
                    tool
                ]
            )
            for tool
            in self.tools
        }

    # ========================================================
    # Public
    # ========================================================

    def invoke(
        self,
        input_state: dict[str, Any],
    ) -> dict[str, Any]:

        messages = list(
            input_state.get(
                "messages",
                [],
            )
        )

        if not messages:

            raise RuntimeError(
                "EffectAwareToolNode "
                "没有收到 messages。"
            )

        latest_ai_index = (
            self._find_latest_ai_index(
                messages
            )
        )

        if latest_ai_index is None:

            raise RuntimeError(
                "ToolNode 输入中没有 AIMessage。"
            )

        latest_ai = messages[
            latest_ai_index
        ]

        tool_calls = list(
            latest_ai.tool_calls
            or []
        )

        if not tool_calls:

            raise RuntimeError(
                "AIMessage 没有 Tool Call。"
            )

        output_messages: list[
            ToolMessage
        ] = []

        # ----------------------------------------------------
        # 当前阶段故意按顺序执行。
        #
        # 原 ToolNode 支持并行 Tool；
        # 但写操作加入 Ledger 后，
        # 顺序执行更容易保证：
        #
        # PREPARED
        # EXECUTING
        # SUCCEEDED
        #
        # 与实际调用严格对应。
        #
        # 后面只读 Tool 可以再恢复并行。
        # ----------------------------------------------------

        for call in tool_calls:

            message = (
                self._execute_one(
                    messages=messages,
                    latest_ai_index=(
                        latest_ai_index
                    ),
                    latest_ai=latest_ai,
                    tool_call=call,
                )
            )

            output_messages.append(
                message
            )

        return {
            "messages": (
                output_messages
            )
        }

    # ========================================================
    # Execute one
    # ========================================================

    def _execute_one(
        self,
        *,
        messages: list[
            BaseMessage
        ],
        latest_ai_index: int,
        latest_ai: AIMessage,
        tool_call: Any,
    ) -> ToolMessage:

        name = tool_call_name(
            tool_call
        )

        call_id = tool_call_id(
            tool_call
        )

        arguments = tool_call_args(
            tool_call
        )

        if not name:

            raise ValueError(
                "Tool Call 缺少 name。"
            )

        if not call_id:

            raise ValueError(
                "Tool Call 缺少 id。"
            )

        tool = self.tool_map.get(
            name
        )

        if tool is None:

            return ToolMessage(
                content=(
                    "Tool Policy Guard 拒绝执行："
                    f"当前没有可执行 Tool：{name}"
                ),
                tool_call_id=(
                    call_id
                ),
                name=name,
                status="error",
            )

        policy = (
            self.policy_repository.get(
                name
            )
        )

        if (
            policy is None
            or not policy.is_executable
        ):

            return ToolMessage(
                content=(
                    "Tool Policy Guard 拒绝执行："
                    f"{name} 尚未获得有效 Policy。"
                ),
                tool_call_id=(
                    call_id
                ),
                name=name,
                status="error",
            )

        # ----------------------------------------------------
        # READ_ONLY 或仅修改 Agent Runtime 的 Tool
        #
        # 不进入 External Effect Ledger。
        # ----------------------------------------------------

        if (
            not policy.requires_effect_record
        ):

            return self._invoke_real_tool(
                messages=messages,
                latest_ai_index=(
                    latest_ai_index
                ),
                latest_ai=latest_ai,
                tool_call=tool_call,
                tool=tool,
            )

        # ----------------------------------------------------
        # 外部写操作必须有执行上下文。
        # ----------------------------------------------------

        context = (
            get_effect_execution_context()
        )

        if context is None:

            return ToolMessage(
                content=(
                    "外部写操作被安全阻止："
                    "当前没有 Effect Execution Context。"
                    "请通过 Agent 的 run/replay "
                    "标准入口执行。"
                ),
                tool_call_id=(
                    call_id
                ),
                name=name,
                status="error",
            )

        args_json = (
            serialize_arguments(
                arguments
            )
        )

        args_hash = (
            calculate_text_hash(
                args_json
            )
        )

        operation_key = (
            build_operation_key(
                thread_id=(
                    context.thread_id
                ),
                tool_name=name,
                tool_call_id=(
                    call_id
                ),
                args_hash=(
                    args_hash
                ),
            )
        )

        checkpoint_id = (
            self._best_effort_checkpoint_id(
                context.thread_id
            )
        )

        effect, created = (
            self.effect_repository
            .prepare_effect(
                operation_key=(
                    operation_key
                ),
                thread_id=(
                    context.thread_id
                ),
                user_id=(
                    context.user_id
                ),
                checkpoint_id=(
                    checkpoint_id
                ),
                replay_from_checkpoint_id=(
                    context
                    .replay_from_checkpoint_id
                ),
                execution_mode=(
                    context.mode
                ),
                tool_name=name,
                tool_call_id=(
                    call_id
                ),
                effect_type=(
                    policy.effect_type
                ),
                replay_policy=(
                    policy.replay_policy
                ),
                arguments=(
                    arguments
                ),
                compensation_tool=(
                    policy
                    .compensation_tool
                ),
            )
        )

        # ----------------------------------------------------
        # 已经成功执行过同一个 Tool Call。
        #
        # Replay 不再触碰真实外部系统。
        # ----------------------------------------------------

        if (
            not created
            and effect.status
            == ExternalEffectStatus.SUCCEEDED
        ):

            return ToolMessage(
                content=(
                    effect.result_text
                    or (
                        "外部操作此前已经执行成功。"
                        f"effect_id={effect.effect_id}"
                    )
                ),
                tool_call_id=(
                    call_id
                ),
                name=name,
                status="success",
                additional_kwargs={
                    "effect_id": (
                        effect.effect_id
                    ),
                    "effect_reused": True,
                },
            )

        # ----------------------------------------------------
        # FAILED：
        #
        # 已知此次调用没有成功。
        # 不机械重复同一个 Tool Call，
        # 交回 Agent 重新决策。
        # ----------------------------------------------------

        if (
            not created
            and effect.status
            == ExternalEffectStatus.FAILED
        ):

            return ToolMessage(
                content=(
                    "此前相同外部 Tool Call "
                    "已经失败，因此未自动重复执行。\n"
                    f"effect_id={effect.effect_id}\n"
                    f"error={effect.error_text or ''}"
                ),
                tool_call_id=(
                    call_id
                ),
                name=name,
                status="error",
            )

        # ----------------------------------------------------
        # EXECUTING：
        #
        # 多半表示上一次进程在外部调用阶段中断。
        # 不能认为没执行。
        # 转换为 UNKNOWN。
        # ----------------------------------------------------

        if (
            not created
            and effect.status
            == ExternalEffectStatus.EXECUTING
        ):

            effect = (
                self.effect_repository
                .mark_unknown(
                    effect.effect_id,
                    (
                        "检测到上一次执行停留在 "
                        "EXECUTING 状态。"
                        "无法确认外部调用是否已经生效。"
                    ),
                )
            )

        # ----------------------------------------------------
        # UNKNOWN / Compensation 状态：
        #
        # 均禁止自动重新执行原外部写操作。
        # ----------------------------------------------------

        if effect.status in {
            ExternalEffectStatus.UNKNOWN,
            ExternalEffectStatus.COMPENSATING,
            ExternalEffectStatus.COMPENSATED,
            ExternalEffectStatus.COMPENSATION_UNKNOWN,
        }:

            return ToolMessage(
                content=(
                    "外部操作没有被再次执行。\n"
                    f"effect_id={effect.effect_id}\n"
                    f"status={effect.status.value}\n"
                    "当前外部真实状态需要先确认或补偿。"
                ),
                tool_call_id=(
                    call_id
                ),
                name=name,
                status="error",
            )

        if (
            effect.status
            != ExternalEffectStatus.PREPARED
        ):

            return ToolMessage(
                content=(
                    "Effect 状态不允许执行："
                    f"{effect.status.value}"
                ),
                tool_call_id=(
                    call_id
                ),
                name=name,
                status="error",
            )

        # ----------------------------------------------------
        # 真正开始调用外部系统。
        # ----------------------------------------------------

        effect = (
            self.effect_repository
            .mark_executing(
                effect.effect_id
            )
        )

        try:

            result_message = (
                self._invoke_real_tool(
                    messages=messages,
                    latest_ai_index=(
                        latest_ai_index
                    ),
                    latest_ai=latest_ai,
                    tool_call=tool_call,
                    tool=tool,
                )
            )

        except Exception as exc:

            # ------------------------------------------------
            # 这里不能写 FAILED。
            #
            # Tool 内部异常并不能证明：
            # 外部服务一定没有完成操作。
            #
            # 例如：
            #
            # POST 成功
            # ↓
            # 网络在返回响应时中断
            #
            # 所以必须 UNKNOWN。
            # ------------------------------------------------

            self.effect_repository.mark_unknown(
                effect.effect_id,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            raise ExternalEffectUncertainError(
                "外部 Tool 调用出现异常，"
                "无法确认真实外部状态。\n"
                f"effect_id={effect.effect_id}\n"
                f"tool={name}"
            ) from exc

        result_text = (
            message_content_to_text(
                result_message.content
            )
        )

        status = str(
            getattr(
                result_message,
                "status",
                "success",
            )
        ).lower()

        # ----------------------------------------------------
        # Tool 参数验证等错误通常会被
        # ToolNode 转成 status=error。
        #
        # 此时 Tool 函数没有成功完成。
        # ----------------------------------------------------

        if status == "error":

            self.effect_repository.mark_failed(
                effect.effect_id,
                result_text,
            )

            return result_message

        succeeded_effect = (
            self.effect_repository
            .mark_succeeded(
                effect.effect_id,
                result_text,
            )
        )

        # ----------------------------------------------------
        # 不改变 Tool 原始内容，
        # 只追加内部 metadata。
        # ----------------------------------------------------

        try:

            additional_kwargs = dict(
                getattr(
                    result_message,
                    "additional_kwargs",
                    {},
                )
                or {}
            )

            additional_kwargs.update(
                {
                    "effect_id": (
                        succeeded_effect
                        .effect_id
                    ),
                    "effect_reused": False,
                }
            )

            result_message.additional_kwargs = (
                additional_kwargs
            )

        except Exception:

            # metadata 添加失败
            # 不影响真实 Tool 结果。
            pass

        return result_message

    # ========================================================
    # Real Tool
    # ========================================================

    def _invoke_real_tool(
        self,
        *,
        messages: list[
            BaseMessage
        ],
        latest_ai_index: int,
        latest_ai: AIMessage,
        tool_call: Any,
        tool: BaseTool,
    ) -> ToolMessage:

        # ----------------------------------------------------
        # 将多个 Tool Call 拆成单个 Tool Call，
        # 让 External Effect Ledger 能逐一确认执行状态。
        # ----------------------------------------------------

        try:

            isolated_ai = (
                latest_ai.model_copy(
                    update={
                        "tool_calls": [
                            tool_call
                        ]
                    }
                )
            )

        except Exception:

            isolated_ai = AIMessage(
                content="",
                tool_calls=[
                    tool_call
                ],
            )

        isolated_messages = [
            *messages[
                :latest_ai_index
            ],
            isolated_ai,
        ]

        node = self.single_tool_nodes[
            str(
                tool.name
            )
        ]

        output = node.invoke(
            {
                "messages": (
                    isolated_messages
                )
            }
        )

        if not isinstance(
            output,
            dict,
        ):

            raise TypeError(
                "当前 EffectAwareToolNode "
                "要求 ToolNode 返回 dict。"
            )

        output_messages = list(
            output.get(
                "messages",
                [],
            )
        )

        current_call_id = (
            tool_call_id(
                tool_call
            )
        )

        for message in output_messages:

            if (
                isinstance(
                    message,
                    ToolMessage,
                )
                and str(
                    message.tool_call_id
                )
                == current_call_id
            ):

                return message

        for message in output_messages:

            if isinstance(
                message,
                ToolMessage,
            ):
                return message

        raise RuntimeError(
            "ToolNode 没有返回 ToolMessage。"
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _find_latest_ai_index(
        messages: list[
            BaseMessage
        ],
    ) -> int | None:

        for index in range(
            len(
                messages
            )
            - 1,
            -1,
            -1,
        ):

            if isinstance(
                messages[
                    index
                ],
                AIMessage,
            ):

                return index

        return None

    def _best_effort_checkpoint_id(
        self,
        thread_id: str,
    ) -> str | None:
        """尽量记录 Tool 执行前的 Checkpoint。

        这个字段只用于审计，
        operation_key 并不依赖它。

        因此读取失败不会阻断 Tool。
        """

        try:

            config = (
                self.agent
                ._build_config(
                    thread_id
                )
            )

            snapshot = (
                self.agent
                .graph
                .get_state(
                    config
                )
            )

            snapshot_config = (
                getattr(
                    snapshot,
                    "config",
                    {},
                )
                or {}
            )

            configurable = (
                snapshot_config.get(
                    "configurable",
                    {},
                )
            )

            value = configurable.get(
                "checkpoint_id"
            )

            return (
                str(
                    value
                )
                if value
                else None
            )

        except Exception:

            return None