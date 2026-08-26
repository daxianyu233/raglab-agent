from __future__ import annotations

import time

from typing import Any

from raglab.evaluation.full_agent_e2e_adapter import (
    E2ETurnObservation,
)


class FullAgentAdapter:
    """
    SecureAgentRuntime Evaluation Adapter.

    将真实 Agent Runtime 输出转换为统一 E2E Observation。

    支持:
        SecureAgentRuntime
        PersistentLangGraphResult
    """

    TOOL_CAPABILITY_MAP = {

        # Knowledge RAG
        "search_knowledge_base":
            "knowledge_retrieval",


        # Github Intelligence
        "search_github_intelligence":
            "github_retrieval",


        # SQL
        "query_github_intelligence_sql":
            "structured_query",


        # Skill
        "list_skills":
            "skill_management",

        "load_skill":
            "skill_management",
    }


    RETRIEVAL_CAPABILITIES = {

        "knowledge_retrieval",

        "github_retrieval",

        "structured_query",

    }


    def __init__(
        self,
        runtime,
        thread_id: str = "evaluation-thread",
        user_id: str = "test_user",
    ):

        self.runtime = runtime

        self.thread_id = thread_id

        self.user_id = user_id



    def reset_case(
        self,
        case_id: str,
    ) -> None:

        """
        当前 Runtime 使用 LangGraph Checkpoint。

        后续如果需要隔离 Case，
        只需要切换 thread_id。
        """

        self.thread_id = (
            f"evaluation-{case_id}"
        )



    def apply_setup(
        self,
        setup: list[dict[str, Any]],
    ) -> None:

        """
        Phase 8D-2 暂无额外 setup。

        保留 Protocol 接口。
        """

        return



    def send(
        self,
        user_input: str,
    ) -> E2ETurnObservation:


        start = time.perf_counter()


        result = self.runtime.run(

            user_input,

            thread_id=self.thread_id,

            user_id=self.user_id,

        )


        latency = (

            time.perf_counter()

            -

            start

        ) * 1000



        return self._convert_result(

            result,

            latency,

        )



    def inspect_state(
        self,
    ) -> dict[str, Any]:


        state = getattr(

            self.runtime,

            "base_agent",

            None,

        )


        if state is None:

            return {}


        return {

            "runtime":
                type(
                    self.runtime
                ).__name__,

            "agent":
                type(
                    state
                ).__name__,

        }



    def _convert_result(
        self,
        result,
        latency_ms: float,
    ) -> E2ETurnObservation:


        tool_calls = []

        capability_groups = []


        raw_tool_trace = getattr(

            result,

            "tool_trace",

            [],

        )


        for item in raw_tool_trace:


            if not isinstance(
                item,
                dict,
            ):
                continue



            name = (

                item.get(
                    "tool"
                )

                or

                item.get(
                    "name"
                )

            )


            if not name:

                continue



            tool_calls.append(

                item

            )



            capability = (

                self.TOOL_CAPABILITY_MAP.get(

                    name,

                    "unknown"

                )

            )


            capability_groups.append(

                capability

            )



        pending = False


        if hasattr(
            self.runtime,
            "get_pending_approval",
        ):


            approval = (

                self.runtime
                .get_pending_approval(
                    self.thread_id
                )

            )


            pending = (

                approval
                is not None

            )



        return E2ETurnObservation(

            answer=getattr(

                result,

                "answer",

                "",

            ),


            completed_normally=getattr(

                result,

                "completed_normally",

                False,

            ),


            tool_calls=tool_calls,


            capability_groups_used=list(

                set(
                    capability_groups
                )

            ),


            total_latency_ms=(

                getattr(

                    result,

                    "total_latency_ms",

                    latency_ms,

                )

            ),


            pending_human_approval=pending,


            state=(

                getattr(

                    result,

                    "final_state",

                    {},

                )

            ),

        )
    