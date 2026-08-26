"""Human Approval Gate ToolNode。

执行顺序：

LLM Tool Call
    ↓
Tool Policy
    ↓
Approval Gate
    ↓
EffectAwareToolNode
    ↓
External Effect Ledger
    ↓
Real Tool
"""

from __future__ import annotations

import json

from typing import (
    Any,
    Sequence,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from langchain_core.tools import (
    BaseTool,
)

from langgraph.types import (
    interrupt,
)

from raglab.control.effect_tool_node import (
    EffectAwareToolNode,
)

from raglab.control.execution_context import (
    get_effect_execution_context,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
    build_operation_key,
    calculate_text_hash,
    serialize_arguments,
)

from raglab.control.human_approval import (
    ApprovalDecision,
    build_approval_id,
    parse_approval_resume,
)

from raglab.control.human_approval_repository import (
    HumanApprovalAuditRepository,
)

from raglab.control.tool_policy_repository import (
    ToolPolicyRepository,
)


# ============================================================
# Tool Call Helpers
# ============================================================


def _tool_call_name(
    tool_call: dict[
        str,
        Any,
    ],
) -> str:

    return str(
        tool_call.get(
            "name",
            "",
        )
    ).strip()


def _tool_call_id(
    tool_call: dict[
        str,
        Any,
    ],
) -> str:

    return str(
        tool_call.get(
            "id",
            "",
        )
    ).strip()


def _tool_call_args(
    tool_call: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    arguments = tool_call.get(
        "args",
        {},
    )

    if isinstance(
        arguments,
        dict,
    ):

        return dict(
            arguments
        )

    return {
        "value": arguments
    }


def _enum_value(
    value: Any,
) -> str:

    return str(
        getattr(
            value,
            "value",
            value,
        )
    ).strip()


# ============================================================
# Approval Gate
# ============================================================


class ApprovalGateToolNode:
    """在真实 Tool 执行前增加 HITL Gate。"""

    def __init__(
        self,
        *,
        delegate: (
            EffectAwareToolNode
        ),
        tools: Sequence[
            BaseTool
        ],
        policy_repository: (
            ToolPolicyRepository
        ),
        effect_repository: (
            ExternalEffectRepository
        ),
        approval_repository: (
            HumanApprovalAuditRepository
        ),
    ) -> None:

        self.delegate = delegate

        self.tools = list(
            tools
        )

        self.policy_repository = (
            policy_repository
        )

        self.effect_repository = (
            effect_repository
        )

        self.approval_repository = (
            approval_repository
        )

    # ========================================================
    # Invoke
    # ========================================================

    def invoke(
        self,
        input_state: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:

        messages = list(
            input_state.get(
                "messages",
                [],
            )
            or []
        )

        if not messages:

            raise RuntimeError(
                "ApprovalGateToolNode "
                "没有收到 messages。"
            )

        latest_ai_index = (
            self._find_latest_ai_index(
                messages
            )
        )

        if latest_ai_index is None:

            raise RuntimeError(
                "没有找到 AIMessage。"
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
                "AIMessage 中没有 Tool Call。"
            )

        context = (
            get_effect_execution_context()
        )

        rejected_messages: dict[
            str,
            ToolMessage,
        ] = {}

        executable_call_ids: set[
            str
        ] = set()

        # ----------------------------------------------------
        # 第一阶段：
        #
        # 只做 Policy + Approval。
        #
        # 此阶段禁止真实 Tool 执行。
        # ----------------------------------------------------

        for tool_call in tool_calls:

            name = _tool_call_name(
                tool_call
            )

            call_id = _tool_call_id(
                tool_call
            )

            arguments = _tool_call_args(
                tool_call
            )

            if not name:

                raise RuntimeError(
                    "Tool Call 缺少 name。"
                )

            if not call_id:

                raise RuntimeError(
                    "Tool Call 缺少 id。"
                )

            policy = (
                self.policy_repository.get(
                    name
                )
            )

            # ------------------------------------------------
            # 无 Policy / 已禁用。
            #
            # 交给 EffectAwareToolNode
            # 做最终安全拒绝。
            # ------------------------------------------------

            if (
                policy is None
                or not policy.is_executable
            ):

                executable_call_ids.add(
                    call_id
                )

                continue

            replay_policy = (
                _enum_value(
                    getattr(
                        policy,
                        "replay_policy",
                        "ALLOW",
                    )
                ).upper()
            )

            is_replay = (
                context is not None
                and context.mode
                == "replay"
            )

            # ------------------------------------------------
            # Replay Policy = DENY
            #
            # 无论人工是否愿意，
            # 当前 Policy 就是不允许 Replay 执行。
            # ------------------------------------------------

            if (
                is_replay
                and replay_policy
                == "DENY"
            ):

                rejected_messages[
                    call_id
                ] = ToolMessage(

                    content=(
                        "Replay Policy 阻止了"
                        "该 Tool 的重新执行。\n"
                        f"tool={name}\n"
                        "replay_policy=DENY"
                    ),

                    tool_call_id=(
                        call_id
                    ),

                    name=name,

                    status="error",

                    additional_kwargs={
                        "replay_blocked": True,
                        "replay_policy": (
                            "DENY"
                        ),
                    },
                )

                continue

            # ------------------------------------------------
            # 是否要求审批。
            #
            # 两种来源：
            #
            # 1. Tool 本身 requires_approval=True
            # 2. Replay Policy=REQUIRE_APPROVAL
            # ------------------------------------------------

            approval_required = bool(
                getattr(
                    policy,
                    "requires_approval",
                    False,
                )
            )

            if (
                is_replay
                and replay_policy
                == "REQUIRE_APPROVAL"
            ):

                approval_required = True

            if not approval_required:

                executable_call_ids.add(
                    call_id
                )

                continue

            # ------------------------------------------------
            # Approval 必须存在运行上下文。
            # ------------------------------------------------

            if context is None:

                rejected_messages[
                    call_id
                ] = ToolMessage(

                    content=(
                        "高风险 Tool 被阻止："
                        "当前不存在 "
                        "Effect Execution Context。"
                    ),

                    tool_call_id=(
                        call_id
                    ),

                    name=name,

                    status="error",
                )

                continue

            # ------------------------------------------------
            # Operation Key
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 如果相同 Effect 已经有确定结果，
            # EffectAwareToolNode 本身不会再次真实执行。
            #
            # 因此没有必要重新审批。
            # ------------------------------------------------

            requires_effect_record = bool(
                getattr(
                    policy,
                    "requires_effect_record",
                    getattr(
                        policy,
                        "has_external_side_effect",
                        False,
                    ),
                )
            )

            existing_effect = None

            if requires_effect_record:

                existing_effect = (
                    self.effect_repository
                    .get_by_operation_key(
                        operation_key
                    )
                )

            if existing_effect is not None:

                existing_status = (
                    _enum_value(
                        existing_effect.status
                    ).upper()
                )

                if existing_status in {
                    "SUCCEEDED",
                    "FAILED",
                    "UNKNOWN",
                    "EXECUTING",
                    "COMPENSATING",
                    "COMPENSATED",
                    "COMPENSATION_UNKNOWN",
                }:

                    executable_call_ids.add(
                        call_id
                    )

                    continue

            # ------------------------------------------------
            # 建立 Approval ID。
            # ------------------------------------------------

            approval_id = (
                build_approval_id(
                    operation_key
                )
            )

            effect_type_value = (
                _enum_value(
                    getattr(
                        policy,
                        "effect_type",
                        "",
                    )
                )
            )

            # ------------------------------------------------
            # REQUESTED 是内部幂等审计操作。
            # ------------------------------------------------

            self.approval_repository.record_requested(

                approval_id=(
                    approval_id
                ),

                operation_key=(
                    operation_key
                ),

                thread_id=(
                    context.thread_id
                ),

                user_id=(
                    context.user_id
                ),

                tool_name=name,

                tool_call_id=(
                    call_id
                ),

                effect_type=(
                    effect_type_value
                    or None
                ),

                args_json=(
                    args_json
                ),
            )

            try:

                safe_arguments = (
                    json.loads(
                        args_json
                    )
                )

            except Exception:

                safe_arguments = {
                    "raw": args_json
                }

            # ------------------------------------------------
            # 真正 LangGraph interrupt。
            #
            # Replay 元数据也写进 payload。
            #
            # Checkpointer 会一起持久化。
            # ------------------------------------------------

            resume_value = interrupt(
                {
                    "type": (
                        "tool_approval"
                    ),

                    "approval_id": (
                        approval_id
                    ),

                    "tool_name": name,

                    "tool_call_id": (
                        call_id
                    ),

                    "effect_type": (
                        effect_type_value
                        or None
                    ),

                    "arguments": (
                        safe_arguments
                    ),

                    "description": (
                        getattr(
                            policy,
                            "description",
                            "",
                        )
                    ),

                    "execution_mode": (
                        context.mode
                    ),

                    # ----------------------------------------
                    # Replay 恢复所需要的关键元数据。
                    # ----------------------------------------

                    "replay_from_checkpoint_id": (
                        context
                        .replay_from_checkpoint_id
                    ),

                    "replay_old_head_checkpoint_id": (
                        context
                        .replay_old_head_checkpoint_id
                    ),

                    "replay_policy": (
                        replay_policy
                    ),

                    "message": (
                        "该 Tool 需要"
                        "人工授权后才能执行。"
                    ),
                }
            )

            decision = (
                parse_approval_resume(
                    resume_value
                )
            )

            # ------------------------------------------------
            # 记录人工决定。
            # ------------------------------------------------

            self.approval_repository.record_decision(

                approval_id=(
                    approval_id
                ),

                operation_key=(
                    operation_key
                ),

                thread_id=(
                    context.thread_id
                ),

                user_id=(
                    context.user_id
                ),

                tool_name=name,

                tool_call_id=(
                    call_id
                ),

                effect_type=(
                    effect_type_value
                    or None
                ),

                args_json=(
                    args_json
                ),

                decision=(
                    decision.decision
                ),

                actor=(
                    decision.actor
                ),

                reason=(
                    decision.reason
                ),
            )

            # ------------------------------------------------
            # REJECT
            # ------------------------------------------------

            if (
                decision.decision
                == ApprovalDecision.REJECT
            ):

                rejected_messages[
                    call_id
                ] = ToolMessage(

                    content=(
                        "人工拒绝执行该 Tool。\n"
                        f"tool={name}\n"
                        f"approval_id="
                        f"{approval_id}\n"
                        f"reason="
                        f"{decision.reason or '未填写'}"
                    ),

                    tool_call_id=(
                        call_id
                    ),

                    name=name,

                    status="error",

                    additional_kwargs={
                        "approval_id": (
                            approval_id
                        ),
                        "approval_decision": (
                            "REJECT"
                        ),
                    },
                )

                continue

            # ------------------------------------------------
            # APPROVE
            # ------------------------------------------------

            executable_call_ids.add(
                call_id
            )

        # ----------------------------------------------------
        # 第二阶段：
        #
        # 所有 Approval 已经处理后，
        # 才允许进入 EffectAwareToolNode。
        # ----------------------------------------------------

        executable_calls = [

            current_call

            for current_call
            in tool_calls

            if _tool_call_id(
                current_call
            )
            in executable_call_ids
        ]

        delegate_messages: list[
            ToolMessage
        ] = []

        if executable_calls:

            try:

                modified_ai = (
                    latest_ai.model_copy(
                        update={
                            "tool_calls": (
                                executable_calls
                            )
                        }
                    )
                )

            except Exception:

                modified_ai = AIMessage(

                    content=(
                        latest_ai.content
                    ),

                    tool_calls=(
                        executable_calls
                    ),
                )

            delegated_messages = [

                *messages[
                    :latest_ai_index
                ],

                modified_ai,
            ]

            delegate_output = (
                self.delegate.invoke(
                    {
                        "messages": (
                            delegated_messages
                        )
                    }
                )
            )

            delegate_messages = [

                current_message

                for current_message
                in list(
                    delegate_output.get(
                        "messages",
                        [],
                    )
                    or []
                )

                if isinstance(
                    current_message,
                    ToolMessage,
                )
            ]

        delegate_map = {

            str(
                current_message
                .tool_call_id
            ): current_message

            for current_message
            in delegate_messages
        }

        # ----------------------------------------------------
        # 恢复原 Tool Call 顺序。
        # ----------------------------------------------------

        final_messages: list[
            ToolMessage
        ] = []

        for tool_call in tool_calls:

            call_id = _tool_call_id(
                tool_call
            )

            if (
                call_id
                in rejected_messages
            ):

                final_messages.append(
                    rejected_messages[
                        call_id
                    ]
                )

                continue

            delegated = (
                delegate_map.get(
                    call_id
                )
            )

            if delegated is None:

                final_messages.append(
                    ToolMessage(

                        content=(
                            "Tool 执行没有返回"
                            "对应 ToolMessage。"
                        ),

                        tool_call_id=(
                            call_id
                        ),

                        name=(
                            _tool_call_name(
                                tool_call
                            )
                        ),

                        status="error",
                    )
                )

                continue

            final_messages.append(
                delegated
            )

        return {
            "messages": (
                final_messages
            )
        }

    # ========================================================
    # Helper
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