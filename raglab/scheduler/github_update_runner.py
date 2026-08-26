"""GitHub Update Job -> Agent Runtime adapter.

负责：

    Scheduled Job Run
        ↓
    Agent Runtime
        ↓
    LangGraph / Tool Policy / HITL
        ↓
    WorkflowOutcome

同时负责跨进程恢复时重新构建
github-intelligence-update Skill Runtime。

不负责：

- Scheduler 时间计算；
- Job Single-Flight；
- Job SQL 状态机；
- GitHub Pipeline 具体实现。
"""

from __future__ import annotations

from typing import Any

from raglab.scheduler.job import (
    ScheduledJobRun,
)

from raglab.scheduler.job_execution_service import (
    WorkflowOutcome,
)


# ============================================================
# Constants
# ============================================================

GITHUB_UPDATE_SKILL_ID = (
    "github-intelligence-update"
)

GITHUB_UPDATE_TOOL_NAME = (
    "update_github_intelligence"
)


GITHUB_UPDATE_JOB_PROMPT = """
这是由系统计划任务触发的 GitHub 技术情报更新任务。

请真正执行 GitHub 技术情报更新，不要只解释操作步骤。

如果 github-intelligence-update Skill 尚未加载，
请先调用 load_skill 加载该 Skill。

随后调用 update_github_intelligence
执行完整 GitHub 技术情报更新流程。

如果 update_github_intelligence
需要人工审批，请按照现有 Tool Policy / HITL
机制暂停并等待人工决定。
""".strip()


# ============================================================
# Runner
# ============================================================

class GithubUpdateAgentRunner:
    """GitHub Update Job 与 Agent Runtime 的适配器。"""

    def __init__(
        self,
        *,
        agent: Any,
        repository: Any,
        user_id: str,
    ) -> None:

        self.agent = agent

        self.repository = repository

        self.user_id = str(
            user_id
        ).strip()

        if not self.user_id:

            raise ValueError(
                "user_id 不能为空。"
            )

    # ========================================================
    # Start
    # ========================================================

    def start(
        self,
        run: ScheduledJobRun,
    ) -> WorkflowOutcome:
        """第一次启动 Scheduler Job。

        调用链：

            Job
            ↓
            Agent
            ↓
            load_skill
            ↓
            update_github_intelligence
            ↓
            HITL
        """

        thread_id = str(
            run.agent_thread_id
            or
            self._build_thread_id(
                run
            )
        ).strip()

        # ----------------------------------------------------
        # Job -> Agent Thread 映射。
        # ----------------------------------------------------

        if not run.agent_thread_id:

            self.repository.bind_agent_thread(
                run_id=run.run_id,
                agent_thread_id=thread_id,
            )

        print()

        print(
            "[GithubUpdateRunner] "
            "启动 Agent Runtime"
        )

        print(
            "  run_id："
            f"{run.run_id}"
        )

        print(
            "  agent_thread_id："
            f"{thread_id}"
        )

        # ----------------------------------------------------
        # 正常首次运行。
        # ----------------------------------------------------

        result = self.agent.run(
            GITHUB_UPDATE_JOB_PROMPT,
            thread_id=thread_id,
            user_id=self.user_id,
        )

        return self._translate_result(
            result=result,
            thread_id=thread_id,
            rejection_requested=False,
        )

    # ========================================================
    # Resume
    # ========================================================

    def resume(
        self,
        run: ScheduledJobRun,
        *,
        approved: bool,
    ) -> WorkflowOutcome:
        """恢复 WAITING_TOOL_APPROVAL Job。

        特别注意：

        Python 进程可能已经重启。

        持久化的是：

            Job Run
            LangGraph Checkpoint

        但 SkillRuntime 是进程内对象。

        因此恢复 LangGraph 前必须先重新加载
        github-intelligence-update Skill。
        """

        thread_id = str(
            run.agent_thread_id
            or ""
        ).strip()

        if not thread_id:

            raise RuntimeError(
                "当前 Job 没有 agent_thread_id，"
                "无法恢复 Agent。"
            )

        command = (
            "/approve"
            if approved
            else "/reject"
        )

        print()

        print(
            "[GithubUpdateRunner] "
            "恢复 Agent Runtime"
        )

        print(
            "  run_id："
            f"{run.run_id}"
        )

        print(
            "  agent_thread_id："
            f"{thread_id}"
        )

        print(
            "  decision："
            f"{'APPROVE' if approved else 'REJECT'}"
        )

        # ====================================================
        # 恢复进程内 Skill Runtime
        # ====================================================

        self._ensure_github_update_skill_loaded()

        # ====================================================
        # 恢复 LangGraph Checkpoint
        # ====================================================

        result = self.agent.run(
            command,
            thread_id=thread_id,
            user_id=self.user_id,
        )

        return self._translate_result(
            result=result,
            thread_id=thread_id,
            rejection_requested=(
                not approved
            ),
        )

    # ========================================================
    # Runtime Hydration
    # ========================================================

    def _ensure_github_update_skill_loaded(
        self,
    ) -> None:
        """确保恢复 HITL 时动态 Skill 已重新加载。

        SkillRuntime 不属于持久 Checkpoint。

        所以程序重启以后：

            SQL Job             还在
            LangGraph checkpoint 还在
            SkillRuntime         丢失

        必须重新加载 Skill，
        再恢复 checkpoint。
        """

        runtime = getattr(
            self.agent,
            "skill_runtime",
            None,
        )

        if runtime is None:

            raise RuntimeError(
                "Agent 没有 SkillRuntime，"
                "无法恢复 GitHub Update Job。"
            )

        # ----------------------------------------------------
        # 先检查是否已经加载。
        # ----------------------------------------------------

        status = runtime.status()

        loaded_skill_ids = set(
            status.get(
                "loaded_skill_ids",
                [],
            )
            or []
        )

        if (
            GITHUB_UPDATE_SKILL_ID
            in loaded_skill_ids
        ):

            self._refresh_agent_tools()

            self._assert_update_tool_active()

            return

        print()

        print(
            "[GithubUpdateRunner] "
            "检测到 Python Runtime 已重建。"
        )

        print(
            "[GithubUpdateRunner] "
            "重新加载 Skill："
            f"{GITHUB_UPDATE_SKILL_ID}"
        )

        # ----------------------------------------------------
        # 使用 SkillRuntime 自己提供的 load_skill Tool。
        #
        # 这样不依赖 SkillRuntime 私有方法。
        # ----------------------------------------------------

        control_tools = list(
            runtime.get_control_tools()
        )

        load_tool = None

        for current_tool in (
            control_tools
        ):

            current_name = str(
                getattr(
                    current_tool,
                    "name",
                    "",
                )
            ).strip()

            if (
                current_name
                ==
                "load_skill"
            ):

                load_tool = (
                    current_tool
                )

                break

        if load_tool is None:

            raise RuntimeError(
                "SkillRuntime 没有提供 "
                "load_skill Tool。"
            )

        # ----------------------------------------------------
        # 第一种：
        #
        # load_skill.invoke(
        #     {"skill_id": "..."}
        # )
        #
        # 当前项目设计就是使用完整 skill id。
        # ----------------------------------------------------

        first_error: (
            Exception
            | None
        ) = None

        try:

            load_tool.invoke(
                {
                    "skill_id":
                        GITHUB_UPDATE_SKILL_ID
                }
            )

        except Exception as exc:

            first_error = exc

        # ----------------------------------------------------
        # 兼容简单字符串 schema。
        # ----------------------------------------------------

        if first_error is not None:

            try:

                load_tool.invoke(
                    GITHUB_UPDATE_SKILL_ID
                )

            except Exception as second_error:

                raise RuntimeError(
                    "重新加载 GitHub Update "
                    "Skill 失败。\n"
                    "dict 调用错误："
                    f"{type(first_error).__name__}: "
                    f"{first_error}\n"
                    "string 调用错误："
                    f"{type(second_error).__name__}: "
                    f"{second_error}"
                ) from second_error

        # ----------------------------------------------------
        # 加载后重新刷新 Agent Active Tools。
        #
        # SecureAgentRuntime 已 Hook
        # _refresh_tool_bindings，
        # 所以刷新以后仍然会经过安全 ToolNode。
        # ----------------------------------------------------

        self._refresh_agent_tools()

        # ----------------------------------------------------
        # 最终不能只相信 load_skill 返回值。
        #
        # 必须验证：
        #
        # Skill 确实加载；
        # Tool 确实 Active。
        # ----------------------------------------------------

        refreshed_status = (
            runtime.status()
        )

        refreshed_loaded_ids = set(
            refreshed_status.get(
                "loaded_skill_ids",
                [],
            )
            or []
        )

        if (
            GITHUB_UPDATE_SKILL_ID
            not in refreshed_loaded_ids
        ):

            raise RuntimeError(
                "调用 load_skill 后，"
                "SkillRuntime 仍未报告 "
                f"{GITHUB_UPDATE_SKILL_ID} "
                "已加载。"
            )

        self._assert_update_tool_active()

        print(
            "[GithubUpdateRunner] "
            "Skill Runtime 恢复成功。"
        )

        print(
            "[GithubUpdateRunner] "
            f"{GITHUB_UPDATE_TOOL_NAME} "
            "已重新进入 Active Tools。"
        )

    # ========================================================
    # Tool Refresh
    # ========================================================

    def _refresh_agent_tools(
        self,
    ) -> None:
        """刷新动态 Skill Tools。"""

        refresh = getattr(
            self.agent,
            "_refresh_tool_bindings",
            None,
        )

        if not callable(
            refresh
        ):

            raise RuntimeError(
                "Agent 没有 "
                "_refresh_tool_bindings()，"
                "无法刷新动态 Skill Tool。"
            )

        refresh()

    def _assert_update_tool_active(
        self,
    ) -> None:
        """确认 update Tool 已进入 Active Tools。"""

        getter = getattr(
            self.agent,
            "get_active_tool_names",
            None,
        )

        if not callable(
            getter
        ):

            raise RuntimeError(
                "Agent 没有 "
                "get_active_tool_names()。"
            )

        active_tools = set(
            getter()
        )

        if (
            GITHUB_UPDATE_TOOL_NAME
            not in active_tools
        ):

            raise RuntimeError(
                f"{GITHUB_UPDATE_SKILL_ID} "
                "已经加载，"
                f"但 {GITHUB_UPDATE_TOOL_NAME} "
                "没有进入 Active Tools。"
            )

    # ========================================================
    # Translate Agent Result
    # ========================================================

    def _translate_result(
        self,
        *,
        result: Any,
        thread_id: str,
        rejection_requested: bool,
    ) -> WorkflowOutcome:
        """把 Agent Runtime Result 转成 Job WorkflowOutcome。"""

        pending = (
            self._get_pending_approval(
                thread_id=thread_id,
                result=result,
            )
        )

        # ----------------------------------------------------
        # HITL
        # ----------------------------------------------------

        if pending is not None:

            print()

            print(
                "[GithubUpdateRunner] "
                "检测到 Tool HITL 中断。"
            )

            return (
                WorkflowOutcome
                .waiting_tool_approval(
                    summary=(
                        "GitHub Update Agent "
                        "正在等待高风险 Tool 审批。"
                    ),
                    pending_approval=pending,
                )
            )

        # ----------------------------------------------------
        # Reject 恢复完成。
        #
        # Tool 没执行。
        # Job 业务结束。
        # ----------------------------------------------------

        if rejection_requested:

            return (
                WorkflowOutcome
                .canceled(
                    (
                        "用户拒绝了 GitHub Update "
                        "高风险 Tool 调用。"
                    )
                )
            )

        # ----------------------------------------------------
        # 防止 max_steps 被误判成功。
        # ----------------------------------------------------

        stopped_by_max_steps = bool(
            getattr(
                result,
                "stopped_by_max_steps",
                False,
            )
        )

        if stopped_by_max_steps:

            raise RuntimeError(
                "GitHub Update Agent "
                "达到 max_steps 后停止，"
                "任务没有正常完成。"
            )

        answer = str(
            getattr(
                result,
                "answer",
                "",
            )
            or ""
        ).strip()

        if not answer:

            answer = (
                "GitHub Update Agent "
                "执行完成。"
            )

        return (
            WorkflowOutcome
            .completed(
                answer
            )
        )

    # ========================================================
    # Pending Approval
    # ========================================================

    def _get_pending_approval(
        self,
        *,
        thread_id: str,
        result: Any,
    ) -> Any | None:
        """检查 Agent 是否停在 HITL interrupt。

        优先走 SecureAgentRuntime
        提供的 get_pending_approval()。

        final_state 作为 fallback。
        """

        owners = [
            self.agent,

            getattr(
                self.agent,
                "runtime_guard",
                None,
            ),

            getattr(
                self.agent,
                "tool_policy_guard",
                None,
            ),
        ]

        for owner in owners:

            if owner is None:
                continue

            getter = getattr(
                owner,
                "get_pending_approval",
                None,
            )

            if not callable(
                getter
            ):
                continue

            try:

                pending = getter(
                    thread_id=thread_id
                )

            except TypeError:

                pending = getter(
                    thread_id
                )

            if pending is not None:

                return pending

        # ----------------------------------------------------
        # Result fallback。
        # ----------------------------------------------------

        final_state = getattr(
            result,
            "final_state",
            None,
        )

        if isinstance(
            final_state,
            dict,
        ):

            interrupts = (
                final_state.get(
                    "__interrupt__"
                )
                or
                final_state.get(
                    "interrupts"
                )
            )

            if interrupts:

                return interrupts

        if isinstance(
            result,
            dict,
        ):

            interrupts = (
                result.get(
                    "__interrupt__"
                )
                or
                result.get(
                    "interrupts"
                )
            )

            if interrupts:

                return interrupts

        return None

    # ========================================================
    # Thread ID
    # ========================================================

    @staticmethod
    def _build_thread_id(
        run: ScheduledJobRun,
    ) -> str:
        """构造稳定的 Scheduler Agent thread_id。"""

        return (
            "scheduler:"
            f"{run.job_name}:"
            f"{run.run_id}"
        )