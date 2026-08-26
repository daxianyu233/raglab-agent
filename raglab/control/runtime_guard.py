"""Agent Runtime Safety Controller。

负责：

1. Tool Policy
2. Human Approval
3. External Effect
4. Replay
5. Replay + HITL
6. Branch Reconciliation
7. Remediation
"""

from __future__ import annotations

import threading
import time
import types

from dataclasses import (
    dataclass,
    replace,
)

from typing import Any

from langchain_core.messages import (
    AIMessage,
)

from langgraph.types import (
    Command,
)

from raglab.agent.long_term_memory_agent import (
    LongTermMemoryContext,
)

from raglab.agent.persistent_langgraph_agent import (
    PersistentLangGraphResult,
    count_human_turns,
)

from raglab.control.approval_gate_tool_node import (
    ApprovalGateToolNode,
)

from raglab.control.branch_reconciliation import (
    BranchReconciliationManager,
)

from raglab.control.effect_tool_node import (
    EffectAwareToolNode,
)

from raglab.control.execution_context import (
    effect_execution_scope,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
)

from raglab.control.human_approval_repository import (
    HumanApprovalAuditRepository,
)

from raglab.control.remediation_repository import (
    RemediationRepository,
)

from raglab.control.tool_policy import (
    ToolPolicyRecord,
)

from raglab.control.tool_policy_repository import (
    ToolPolicyRepository,
)

from raglab.generation.rag_chain import (
    extract_answer_text,
)


# ============================================================
# Tool Guard Result
# ============================================================


@dataclass(
    frozen=True,
)
class ToolPolicyGuardResult:

    discovered_tool_names: tuple[
        str,
        ...
    ]

    allowed_tool_names: tuple[
        str,
        ...
    ]

    blocked_tool_names: tuple[
        str,
        ...
    ]


# ============================================================
# Checkpoint Helper
# ============================================================


def snapshot_checkpoint_id(
    snapshot: Any,
) -> str | None:

    config = getattr(
        snapshot,
        "config",
        {},
    )

    if not isinstance(
        config,
        dict,
    ):

        return None

    configurable = config.get(
        "configurable",
        {},
    )

    if not isinstance(
        configurable,
        dict,
    ):

        return None

    value = configurable.get(
        "checkpoint_id"
    )

    if not value:

        return None

    return str(
        value
    )


# ============================================================
# Runtime Controller
# ============================================================


class ToolPolicyRuntimeGuard:

    def __init__(
        self,
        *,
        agent: Any,
        repository: (
            ToolPolicyRepository
        ),
        effect_repository: (
            ExternalEffectRepository
        ),
        approval_repository: (
            HumanApprovalAuditRepository
        ),
    ) -> None:

        self.agent = agent

        self.repository = (
            repository
        )

        self.effect_repository = (
            effect_repository
        )

        self.approval_repository = (
            approval_repository
        )

        self._lock = (
            threading.RLock()
        )

        self._installed = False

        self._original_refresh = None

        self._original_run = None

        self._original_replay = None

        self._base_tool_names = {

            str(
                getattr(
                    tool,
                    "name",
                    "",
                )
            ).strip()

            for tool
            in list(
                getattr(
                    agent,
                    "tools",
                    [],
                )
                or []
            )

            if str(
                getattr(
                    tool,
                    "name",
                    "",
                )
            ).strip()
        }

        self._last_result = (
            ToolPolicyGuardResult(
                discovered_tool_names=(),
                allowed_tool_names=(),
                blocked_tool_names=(),
            )
        )

        self.reconciliation_manager = (
            BranchReconciliationManager(

                agent=agent,

                database_path=(
                    repository.database_path
                ),
            )
        )

        self.remediation_repository = (
            RemediationRepository(

                database_path=(
                    repository.database_path
                )
            )
        )

        self.remediation_repository.setup()

    # ========================================================
    # Install
    # ========================================================

    def install(
        self,
    ) -> None:

        if self._installed:

            return

        refresh_method = getattr(
            self.agent,
            "_refresh_tool_bindings",
            None,
        )

        if not callable(
            refresh_method
        ):

            raise TypeError(
                "Agent 没有 "
                "_refresh_tool_bindings()。"
            )

        self._original_refresh = (
            refresh_method
        )

        run_method = getattr(
            self.agent,
            "run",
            None,
        )

        if not callable(
            run_method
        ):

            raise TypeError(
                "Agent 没有 run()。"
            )

        self._original_run = (
            run_method
        )

        guard = self

        # ----------------------------------------------------
        # refresh
        # ----------------------------------------------------

        def guarded_refresh(
            agent_self: Any,
        ) -> list[Any]:

            return guard.refresh()

        self.agent._refresh_tool_bindings = (
            types.MethodType(
                guarded_refresh,
                self.agent,
            )
        )

        # ----------------------------------------------------
        # run
        # ----------------------------------------------------

        def guarded_run(
            agent_self: Any,
            question: str,
            *,
            thread_id: str,
            user_id: str,
        ) -> PersistentLangGraphResult:

            normalized_question = str(
                question
            ).strip()

            if not normalized_question:

                raise ValueError(
                    "question 不能为空。"
                )

            lower_question = (
                normalized_question.lower()
            )

            # --------------------------------------------
            # APPROVE
            # --------------------------------------------

            if (
                lower_question
                == "/approve"
                or lower_question.startswith(
                    "/approve "
                )
            ):

                reason = (
                    normalized_question[
                        len(
                            "/approve"
                        ):
                    ]
                    .strip()
                )

                return (
                    guard.resume_approval(

                        thread_id=(
                            thread_id
                        ),

                        user_id=(
                            user_id
                        ),

                        decision=(
                            "APPROVE"
                        ),

                        reason=(
                            reason
                        ),
                    )
                )

            # --------------------------------------------
            # REJECT
            # --------------------------------------------

            if (
                lower_question
                == "/reject"
                or lower_question.startswith(
                    "/reject "
                )
            ):

                reason = (
                    normalized_question[
                        len(
                            "/reject"
                        ):
                    ]
                    .strip()
                )

                if not reason:

                    raise ValueError(
                        "/reject 必须填写原因。\n"
                        "格式："
                        "/reject <原因>"
                    )

                return (
                    guard.resume_approval(

                        thread_id=(
                            thread_id
                        ),

                        user_id=(
                            user_id
                        ),

                        decision=(
                            "REJECT"
                        ),

                        reason=(
                            reason
                        ),
                    )
                )

            # --------------------------------------------
            # 当前已有 Pending Interrupt，
            # 不接受新的自然语言 Turn。
            # --------------------------------------------

            pending = (
                guard.get_pending_approval(
                    thread_id
                )
            )

            if pending is not None:

                raise RuntimeError(
                    "当前 thread 正在等待"
                    " Tool 人工审批。\n\n"
                    "请输入：\n"
                    "/approve\n"
                    "或：\n"
                    "/reject <原因>"
                )

            # --------------------------------------------
            # 普通 Agent Run。
            # --------------------------------------------

            with effect_execution_scope(

                thread_id=(
                    thread_id
                ),

                user_id=(
                    user_id
                ),

                mode="normal",
            ):

                result = (
                    guard._original_run(

                        normalized_question,

                        thread_id=(
                            thread_id
                        ),

                        user_id=(
                            user_id
                        ),
                    )
                )

            return (
                guard
                .decorate_result_if_pending(
                    result,
                    thread_id=(
                        thread_id
                    ),
                )
            )

        self.agent.run = (
            types.MethodType(
                guarded_run,
                self.agent,
            )
        )

        # ----------------------------------------------------
        # Replay
        # ----------------------------------------------------

        replay_method = getattr(
            self.agent,
            "replay_checkpoint",
            None,
        )

        if callable(
            replay_method
        ):

            self._original_replay = (
                replay_method
            )

            def guarded_replay(
                agent_self: Any,
                *,
                thread_id: str,
                user_id: str,
                checkpoint_id: str,
            ) -> Any:
                """开始一次 Replay。

                如果中途产生 interrupt：

                    暂停 Replay
                    不做 Reconciliation

                等人工 resume 后 Replay 真正结束：

                    Reconciliation
                    Remediation
                """

                # ----------------------------------------
                # 当前 thread 已经有中断，
                # 不能同时再开启另一次 Replay。
                # ----------------------------------------

                existing_pending = (
                    guard
                    .get_pending_approval(
                        thread_id
                    )
                )

                if (
                    existing_pending
                    is not None
                ):

                    raise RuntimeError(
                        "当前 thread 已经"
                        "存在待审批操作，"
                        "请先处理当前 interrupt。"
                    )

                # ----------------------------------------
                # Replay 开始之前记录 Old Head。
                # ----------------------------------------

                old_snapshot = (
                    agent_self.graph.get_state(
                        agent_self._build_config(
                            thread_id
                        )
                    )
                )

                old_head_id = (
                    snapshot_checkpoint_id(
                        old_snapshot
                    )
                )

                if not old_head_id:

                    raise RuntimeError(
                        "Replay 前无法读取 "
                        "Old Branch Head。"
                    )

                # ----------------------------------------
                # 真正 Replay。
                #
                # Replay 元数据进入 Execution Context。
                # ----------------------------------------

                with effect_execution_scope(

                    thread_id=(
                        thread_id
                    ),

                    user_id=(
                        user_id
                    ),

                    mode="replay",

                    replay_from_checkpoint_id=(
                        checkpoint_id
                    ),

                    replay_old_head_checkpoint_id=(
                        old_head_id
                    ),
                ):

                    replay_result = (
                        guard
                        ._original_replay(

                            thread_id=(
                                thread_id
                            ),

                            user_id=(
                                user_id
                            ),

                            checkpoint_id=(
                                checkpoint_id
                            ),
                        )
                    )

                # ----------------------------------------
                # Replay 是否在中途 interrupt？
                # ----------------------------------------

                pending = (
                    guard.get_pending_approval(
                        thread_id
                    )
                )

                if pending is not None:

                    agent_self.last_replay_waiting_approval = (
                        True
                    )

                    agent_self.last_reconciliation_plan = (
                        None
                    )

                    agent_self.last_reconciliation_error = (
                        None
                    )

                    # ------------------------------------
                    # CLI 的 Replay Result 本身已有
                    # answer 字段时，
                    # 将其替换为审批提示。
                    # ------------------------------------

                    try:

                        replay_result = replace(
                            replay_result,
                            answer=(
                                guard
                                .format_pending_approval(
                                    pending
                                )
                            ),
                        )

                    except Exception:

                        pass

                    return replay_result

                # ----------------------------------------
                # 没有 interrupt：
                # Replay 已完整执行。
                # ----------------------------------------

                agent_self.last_replay_waiting_approval = (
                    False
                )

                guard._finalize_replay(

                    thread_id=(
                        thread_id
                    ),

                    replay_checkpoint_id=(
                        checkpoint_id
                    ),

                    old_head_checkpoint_id=(
                        old_head_id
                    ),
                )

                return replay_result

            self.agent.replay_checkpoint = (
                types.MethodType(
                    guarded_replay,
                    self.agent,
                )
            )

        # ----------------------------------------------------
        # Expose
        # ----------------------------------------------------

        self.agent.tool_policy_repository = (
            self.repository
        )

        self.agent.external_effect_repository = (
            self.effect_repository
        )

        self.agent.human_approval_repository = (
            self.approval_repository
        )

        self.agent.tool_policy_guard = (
            self
        )

        self.agent.branch_reconciliation_manager = (
            self.reconciliation_manager
        )

        self.agent.remediation_repository = (
            self.remediation_repository
        )

        self.agent.last_reconciliation_plan = (
            None
        )

        self.agent.last_reconciliation_error = (
            None
        )

        self.agent.last_remediation_cases = (
            []
        )

        self.agent.last_remediation_error = (
            None
        )

        self.agent.last_replay_waiting_approval = (
            False
        )

        self._installed = True

        self.refresh()

    # ========================================================
    # Refresh Tools
    # ========================================================

    def refresh(
        self,
    ) -> list[Any]:

        with self._lock:

            if (
                self._original_refresh
                is None
            ):

                raise RuntimeError(
                    "Runtime Guard 尚未安装。"
                )

            discovered_tools = list(
                self._original_refresh()
                or []
            )

            records = (
                self.repository
                .discover_tool_objects(

                    discovered_tools,

                    base_tool_names=(
                        self._base_tool_names
                    ),
                )
            )

            policy_map: dict[
                str,
                ToolPolicyRecord,
            ] = {

                record.tool_name: (
                    record
                )

                for record
                in records
            }

            allowed_tools: list[
                Any
            ] = []

            blocked_tool_names: list[
                str
            ] = []

            for tool in discovered_tools:

                name = str(
                    getattr(
                        tool,
                        "name",
                        "",
                    )
                ).strip()

                policy = (
                    policy_map.get(
                        name
                    )
                )

                if (
                    policy is None
                    or not policy.is_executable
                ):

                    blocked_tool_names.append(
                        name
                    )

                    continue

                # --------------------------------------------
                # requires_approval 的 Tool
                # 仍然允许暴露给 LLM。
                #
                # 只是执行时必须经过 Approval Gate。
                # --------------------------------------------

                allowed_tools.append(
                    tool
                )

            if not allowed_tools:

                raise RuntimeError(
                    "Tool Policy 过滤后"
                    "没有可执行 Tool。"
                )

            bind_tools = getattr(
                self.agent.chat_model,
                "bind_tools",
                None,
            )

            if not callable(
                bind_tools
            ):

                raise TypeError(
                    "chat_model 不支持 "
                    "bind_tools()。"
                )

            self.agent.tool_enabled_model = (
                bind_tools(
                    allowed_tools
                )
            )

            # --------------------------------------------
            # Effect Layer
            # --------------------------------------------

            effect_node = (
                EffectAwareToolNode(

                    agent=(
                        self.agent
                    ),

                    tools=(
                        allowed_tools
                    ),

                    policy_repository=(
                        self.repository
                    ),

                    effect_repository=(
                        self.effect_repository
                    ),
                )
            )

            # --------------------------------------------
            # Approval Layer
            #
            # 必须位于 Effect Layer 前。
            # --------------------------------------------

            self.agent.tool_node = (
                ApprovalGateToolNode(

                    delegate=(
                        effect_node
                    ),

                    tools=(
                        allowed_tools
                    ),

                    policy_repository=(
                        self.repository
                    ),

                    effect_repository=(
                        self.effect_repository
                    ),

                    approval_repository=(
                        self.approval_repository
                    ),
                )
            )

            discovered_names = tuple(

                str(
                    getattr(
                        tool,
                        "name",
                        "",
                    )
                )

                for tool
                in discovered_tools
            )

            allowed_names = tuple(

                str(
                    getattr(
                        tool,
                        "name",
                        "",
                    )
                )

                for tool
                in allowed_tools
            )

            self._last_result = (
                ToolPolicyGuardResult(

                    discovered_tool_names=(
                        discovered_names
                    ),

                    allowed_tool_names=(
                        allowed_names
                    ),

                    blocked_tool_names=tuple(
                        blocked_tool_names
                    ),
                )
            )

            return allowed_tools

    # ========================================================
    # Pending Approval
    # ========================================================

    def get_pending_approval(
        self,
        thread_id: str,
    ) -> dict[
        str,
        Any,
    ] | None:
        """读取 Checkpoint 中的 pending interrupt。"""

        snapshot = (
            self.agent.graph.get_state(
                self.agent._build_config(
                    thread_id
                )
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

        for task in tasks:

            interrupts = list(
                getattr(
                    task,
                    "interrupts",
                    (),
                )
                or ()
            )

            for current_interrupt in interrupts:

                value = getattr(
                    current_interrupt,
                    "value",
                    None,
                )

                if (
                    isinstance(
                        value,
                        dict,
                    )
                    and value.get(
                        "type"
                    )
                    == "tool_approval"
                ):

                    return dict(
                        value
                    )

        return None

    # ========================================================
    # Resume Approval
    # ========================================================

    def resume_approval(
        self,
        *,
        thread_id: str,
        user_id: str,
        decision: str,
        reason: str,
    ) -> PersistentLangGraphResult:
        """恢复普通运行或 Replay 的 interrupt。"""

        pending = (
            self.get_pending_approval(
                thread_id
            )
        )

        if pending is None:

            raise RuntimeError(
                "当前 thread 没有"
                "待处理 Approval。"
            )

        # ----------------------------------------------------
        # 关键：
        #
        # 从 interrupt payload 中恢复
        # execution_mode。
        # ----------------------------------------------------

        execution_mode = str(
            pending.get(
                "execution_mode",
                "normal",
            )
        ).strip().lower()

        if execution_mode not in {
            "normal",
            "replay",
        }:

            execution_mode = "normal"

        replay_from_checkpoint_id = (
            pending.get(
                "replay_from_checkpoint_id"
            )
        )

        replay_old_head_checkpoint_id = (
            pending.get(
                "replay_old_head_checkpoint_id"
            )
        )

        resume_value = {

            "decision": (
                decision
            ),

            "actor": (
                user_id
            ),

            "reason": (
                reason
            ),
        }

        total_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # 恢复对应 Execution Context。
        # ----------------------------------------------------

        with effect_execution_scope(

            thread_id=(
                thread_id
            ),

            user_id=(
                user_id
            ),

            mode=(
                execution_mode
            ),

            replay_from_checkpoint_id=(
                replay_from_checkpoint_id
            ),

            replay_old_head_checkpoint_id=(
                replay_old_head_checkpoint_id
            ),
        ):

            final_state = (
                self.agent.graph.invoke(

                    Command(
                        resume=(
                            resume_value
                        )
                    ),

                    config=(
                        self.agent
                        ._build_config(
                            thread_id
                        )
                    ),

                    context=(
                        LongTermMemoryContext(
                            user_id=(
                                user_id
                            )
                        )
                    ),
                )
            )

        latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        # ----------------------------------------------------
        # 可能存在第二个 Approval。
        #
        # 此时 Replay 仍然没有结束。
        # ----------------------------------------------------

        pending_after = (
            self.get_pending_approval(
                thread_id
            )
        )

        # ----------------------------------------------------
        # 如果这是 Replay：
        #
        # 只有整个 Replay 完成后
        # 才允许进行 Branch Reconciliation。
        # ----------------------------------------------------

        if execution_mode == "replay":

            if (
                pending_after
                is not None
            ):

                self.agent.last_replay_waiting_approval = (
                    True
                )

            else:

                self.agent.last_replay_waiting_approval = (
                    False
                )

                if (
                    replay_from_checkpoint_id
                    and
                    replay_old_head_checkpoint_id
                ):

                    self._finalize_replay(

                        thread_id=(
                            thread_id
                        ),

                        replay_checkpoint_id=str(
                            replay_from_checkpoint_id
                        ),

                        old_head_checkpoint_id=str(
                            replay_old_head_checkpoint_id
                        ),
                    )

                else:

                    self.agent.last_reconciliation_plan = (
                        None
                    )

                    self.agent.last_reconciliation_error = (
                        "Replay HITL 恢复后"
                        "缺少 Replay 元数据，"
                        "无法执行 Branch Reconciliation。"
                    )

        # ----------------------------------------------------
        # 构造统一 Agent Result。
        # ----------------------------------------------------

        return self._build_result(

            final_state=(
                final_state
            ),

            thread_id=(
                thread_id
            ),

            question=(
                "/approve"
                if decision
                == "APPROVE"
                else (
                    "/reject "
                    + reason
                )
            ),

            latency_ms=(
                latency_ms
            ),
        )

    # ========================================================
    # Finalize Replay
    # ========================================================

    def _finalize_replay(
        self,
        *,
        thread_id: str,
        replay_checkpoint_id: str,
        old_head_checkpoint_id: str,
    ) -> None:
        """Replay 真正完成后执行：

        New Head
            ↓
        Reconciliation
            ↓
        Remediation
        """

        # ----------------------------------------------------
        # New Branch Head
        # ----------------------------------------------------

        new_snapshot = (
            self.agent.graph.get_state(
                self.agent._build_config(
                    thread_id
                )
            )
        )

        new_head_checkpoint_id = (
            snapshot_checkpoint_id(
                new_snapshot
            )
        )

        if not new_head_checkpoint_id:

            self.agent.last_reconciliation_plan = (
                None
            )

            self.agent.last_reconciliation_error = (
                "Replay 完成后无法读取 "
                "New Branch Head。"
            )

            return

        # ----------------------------------------------------
        # Reconciliation
        # ----------------------------------------------------

        try:

            plan = (
                self.reconciliation_manager
                .create_plan(

                    thread_id=(
                        thread_id
                    ),

                    replay_checkpoint_id=(
                        replay_checkpoint_id
                    ),

                    old_head_checkpoint_id=(
                        old_head_checkpoint_id
                    ),

                    new_head_checkpoint_id=(
                        new_head_checkpoint_id
                    ),
                )
            )

        except Exception as exc:

            self.agent.last_reconciliation_plan = (
                None
            )

            self.agent.last_reconciliation_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            self.agent.last_remediation_cases = (
                []
            )

            return

        self.agent.last_reconciliation_plan = (
            plan
        )

        self.agent.last_reconciliation_error = (
            None
        )

        # ----------------------------------------------------
        # Remediation
        # ----------------------------------------------------

        try:

            cases = (
                self.remediation_repository
                .create_cases_from_plan(

                    plan=(
                        plan
                    ),

                    effect_repository=(
                        self.effect_repository
                    ),
                )
            )

        except Exception as exc:

            self.agent.last_remediation_cases = (
                []
            )

            self.agent.last_remediation_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return

        self.agent.last_remediation_cases = (
            cases
        )

        self.agent.last_remediation_error = (
            None
        )

    # ========================================================
    # Result Helpers
    # ========================================================

    def decorate_result_if_pending(
        self,
        result: (
            PersistentLangGraphResult
        ),
        *,
        thread_id: str,
    ) -> PersistentLangGraphResult:

        pending = (
            self.get_pending_approval(
                thread_id
            )
        )

        if pending is None:

            return result

        return replace(

            result,

            answer=(
                self.format_pending_approval(
                    pending
                )
            ),

            completed_normally=False,
        )

    def _build_result(
        self,
        *,
        final_state: dict[
            str,
            Any,
        ],
        thread_id: str,
        question: str,
        latency_ms: float,
    ) -> PersistentLangGraphResult:

        messages = list(
            final_state.get(
                "messages",
                [],
            )
            or []
        )

        pending = (
            self.get_pending_approval(
                thread_id
            )
        )

        if pending is not None:

            answer = (
                self.format_pending_approval(
                    pending
                )
            )

        else:

            answer = ""

            for message in reversed(
                messages
            ):

                if isinstance(
                    message,
                    AIMessage,
                ):

                    answer = (
                        extract_answer_text(
                            message
                        )
                        .strip()
                    )

                    break

        stopped_by_max_steps = bool(
            final_state.get(
                "stopped_by_max_steps",
                False,
            )
        )

        return PersistentLangGraphResult(

            thread_id=(
                str(
                    thread_id
                )
            ),

            question=(
                question
            ),

            answer=(
                answer
            ),

            messages=(
                messages
            ),

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
                and pending is None
            ),

            model_trace=list(
                final_state.get(
                    "model_trace",
                    [],
                )
                or []
            ),

            tool_trace=list(
                final_state.get(
                    "tool_trace",
                    [],
                )
                or []
            ),

            total_message_count=len(
                messages
            ),

            total_latency_ms=(
                latency_ms
            ),

            final_state=dict(
                final_state
            ),
        )

    # ========================================================
    # Approval Message
    # ========================================================

    @staticmethod
    def format_pending_approval(
        pending: dict[
            str,
            Any,
        ],
    ) -> str:

        arguments = pending.get(
            "arguments",
            {},
        )

        try:

            import json

            arguments_text = (
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        except Exception:

            arguments_text = str(
                arguments
            )

        execution_mode = str(
            pending.get(
                "execution_mode",
                "normal",
            )
        )

        mode_text = (
            "Replay 重新执行"
            if execution_mode
            == "replay"
            else "普通执行"
        )

        return (
            "当前 Agent 已暂停，"
            "等待高风险 Tool 人工授权。\n\n"

            f"执行场景："
            f"{mode_text}\n"

            f"Tool："
            f"{pending.get('tool_name')}\n"

            f"安全类型："
            f"{pending.get('effect_type')}\n"

            f"Replay Policy："
            f"{pending.get('replay_policy')}\n"

            f"approval_id："
            f"{pending.get('approval_id')}\n\n"

            "Tool 参数：\n"
            f"{arguments_text}\n\n"

            "批准请输入：\n"
            "/approve\n\n"

            "拒绝请输入：\n"
            "/reject <原因>"
        )

    # ========================================================
    # Status
    # ========================================================

    def last_result(
        self,
    ) -> ToolPolicyGuardResult:

        return self._last_result