from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


from raglab.application.secure_agent_factory import (
    build_secure_agent,
)


from raglab.evaluation.models.e2e_observation import (
    E2ETurnObservation,
)


class _BenchmarkNoWriteGithubUpdateCoordinator:
    """Benchmark-only update executor that never touches business storage."""

    def __init__(self) -> None:
        self.execution_count = 0
        self.ledger: list[dict[str, Any]] = []

    def execute_tool_call(
        self,
        *,
        execute_update,
        manual_actor: str = "agent_manual_request",
    ) -> dict[str, Any]:
        del execute_update

        self.execution_count += 1
        entry = {
            "effect_id": f"benchmark-no-write-{self.execution_count}",
            "tool_name": "update_github_intelligence",
            "actor": manual_actor,
            "status": "SIMULATED_SUCCESS",
            "dry_run": True,
            "business_database_updated": False,
        }
        self.ledger.append(entry)

        return {
            "status": "success",
            "tool": "update_github_intelligence",
            "message": (
                "人工审批已经通过，HITL pending 已清除；"
                "本次 Benchmark dry-run 已执行完成，不需要再次审批。"
                "GitHub 采集和业务数据库写入均被安全跳过。"
            ),
            "benchmark": entry,
        }



class SecureRuntimeAdapter:
    """
    SecureAgentRuntime 真实测试适配器。

    Evaluation 层只依赖 Runtime 行为，
    不依赖 Agent 内部实现。
    """


    TOOL_CAPABILITY_MAP = {

        "search_knowledge_base":
            "knowledge_retrieval",


        "search_github_intelligence":
            "github_semantic_retrieval",


        "query_github_intelligence_sql":
            "structured_query",


        "get_github_intelligence_schema":
            "structured_query",


        "list_skills":
            "skill_management",


        "load_skill":
            "skill_management",


        "update_github_intelligence":
            "github_update_action",
    }



    RETRIEVAL_CAPABILITIES = {

        "knowledge_retrieval",

        "github_semantic_retrieval",

        "structured_query",

    }



    def __init__(
        self,
        config_path: Path,
        *,
        no_write_external_actions: bool = False,
    ):

        self._benchmark_coordinator = None

        if no_write_external_actions:
            from raglab.agent import github_intelligence_skill

            self._benchmark_coordinator = (
                _BenchmarkNoWriteGithubUpdateCoordinator()
            )
            github_intelligence_skill._GITHUB_UPDATE_JOB_COORDINATOR = (
                self._benchmark_coordinator
            )

        self.runtime = (
            build_secure_agent(
                config_path
            )
        )



    def send(
        self,
        user_input: str,
        *,
        thread_id: str,
        user_id: str = "test_user",
    ) -> E2ETurnObservation:


        start = time.perf_counter()


        ledger_count_before = self._ledger_count()

        try:
            result = self.runtime.run(
                user_input,
                thread_id=thread_id,
                user_id=user_id,
            )
        except RuntimeError:
            pending = self.inspect_pending(thread_id)
            if pending is None:
                raise

            # A LangGraph interrupt can leave the final AI message empty.  The
            # pending checkpoint is the successful outcome for this turn, not
            # an Agent failure.
            result = self._pending_result(
                thread_id=thread_id,
                question=user_input,
            )


        latency = (
            time.perf_counter()
            -
            start
        ) * 1000


        observation = self._convert_result(
            result,
            latency,
        )

        ledger_count_after = self._ledger_count()
        decision = str(user_input).strip().lower()
        observation.state.update({
            "approval_decision": (
                "approve" if decision == "/approve" else
                "reject" if decision == "/reject" else ""
            ),
            "external_effect_ledger_entry_created": (
                ledger_count_after > ledger_count_before
            ),
            "external_effect_ledger": self._ledger_snapshot(),
            "no_duplicate_side_effect": (
                len({item["effect_id"] for item in self._ledger_snapshot()})
                == ledger_count_after
            ),
            "business_database_updated": False
                if self._benchmark_coordinator is not None else None,
        })
        return observation

    def _ledger_count(self) -> int:
        if self._benchmark_coordinator is None:
            return 0
        return len(self._benchmark_coordinator.ledger)

    def _ledger_snapshot(self) -> list[dict[str, Any]]:
        if self._benchmark_coordinator is None:
            return []
        return [dict(item) for item in self._benchmark_coordinator.ledger]

    def _pending_result(self, *, thread_id: str, question: str):
        config = self.runtime.base_agent._build_config(thread_id)
        snapshot = self.runtime.base_agent.graph.get_state(config)
        final_state = dict(getattr(snapshot, "values", {}) or {})

        return SimpleNamespace(
            answer="等待人工审批。",
            completed_normally=False,
            tool_trace=list(final_state.get("tool_trace", [])),
            thread_id=thread_id,
            final_state=final_state,
            state=final_state,
            model_trace=list(final_state.get("model_trace", [])),
            turn_llm_call_count=int(final_state.get("turn_llm_calls", 0)),
            turn_tool_call_count=int(final_state.get("turn_tool_calls", 0)),
            turn_summary_call_count=int(final_state.get("turn_summary_calls", 0)),
            question=question,
        )



    def approve(
        self,
        *,
        thread_id: str,
        user_id: str = "test_user",
    ):

        return self.send(
            "/approve",
            thread_id=thread_id,
            user_id=user_id,
        )



    def reject(
        self,
        *,
        thread_id: str,
        user_id: str = "test_user",
    ):

        return self.send(
            "/reject",
            thread_id=thread_id,
            user_id=user_id,
        )



    def inspect_pending(
        self,
        thread_id: str,
    ) -> dict[str, Any] | None:


        return (
            self.runtime
            .get_pending_approval(
                thread_id
            )
        )



    def _extract_context_pipeline(
        self,
        result,
    ) -> dict[str, Any]:

        """
        从 Runtime 返回结果中提取 Context Pipeline。

        不假设固定内部结构。
        """

        final_state = (

            result.final_state

            if isinstance(
                result.final_state,
                dict,
            )

            else {}

        )


        context_pipeline = (
            final_state.get(
                "context_pipeline"
            )
        )


        if isinstance(
            context_pipeline,
            dict,
        ):

            return context_pipeline



        # 某些 Runtime 会把 trace 保存在 state 其他字段。
        state = getattr(
            result,
            "state",
            None,
        )


        if isinstance(
            state,
            dict,
        ):

            context_pipeline = (
                state.get(
                    "context_pipeline"
                )
            )


            if isinstance(
                context_pipeline,
                dict,
            ):

                return context_pipeline



        # 最后尝试 raw trace
        trace = getattr(
            result,
            "model_trace",
            None,
        )


        if isinstance(
            trace,
            dict,
        ):

            context_pipeline = (
                trace.get(
                    "context_pipeline"
                )
            )


            if isinstance(
                context_pipeline,
                dict,
            ):

                return context_pipeline



        return {}



    def _convert_result(
        self,
        result,
        latency_ms: float,
    ) -> E2ETurnObservation:


        tool_calls = (
            result.tool_trace
            or
            []
        )


        capability_groups = []


        for item in tool_calls:

            name = (
                item.get(
                    "name",
                    "",
                )
            )


            if not name:

                continue


            capability_groups.append(

                self.TOOL_CAPABILITY_MAP.get(
                    name,
                    "unknown",
                )

            )



        pending = False


        try:

            pending = (
                self.runtime
                .get_pending_approval(
                    result.thread_id
                )
                is not None
            )


        except Exception:

            pending = False



        final_state = (

            result.final_state

            if isinstance(
                result.final_state,
                dict,
            )

            else {}

        )



        context_trace = (
            self._extract_context_pipeline(
                result
            )
        )


        memory_trace = (
            final_state.get(
                "memory",
                {}
            )
        )



        return E2ETurnObservation(


            answer=result.answer,


            completed_normally=(
                result.completed_normally
            ),


            tool_calls=tool_calls,


            capability_groups_used=list(
                set(
                    capability_groups
                )
            ),


            total_latency_ms=latency_ms,


            pending_human_approval=pending,


            write_side_effect_count=(

                final_state.get(
                    "write_side_effect_count",
                    0,
                )

            ),


            state={


                "thread_id":
                    result.thread_id,


                "llm_calls":
                    result.turn_llm_call_count,


                "tool_calls":
                    result.turn_tool_call_count,


                "summary_calls":
                    result.turn_summary_call_count,


                "loaded_skills":
                    final_state.get(
                        "loaded_skills",
                        [],
                    ),


                "context_pipeline":
                    context_trace,


                "memory_trace":
                    memory_trace,


                "model_trace":
                    result.model_trace,


                "tool_trace":
                    result.tool_trace,


                "final_state":
                    final_state,


                "raw_result":
                    result,

            },

        )
