from __future__ import annotations

from typing import Any


MEMORY_CATEGORIES = {

    "previous_answer_reuse",

    "previous_tool_evidence",

    "recent_context",

    "historical_context",

    "long_term_memory",

    "working_memory_safety",

}


RECOVERY_CATEGORIES = {

    "tool_error_recovery",

}



def _safe_rate(
    correct: int,
    total: int,
):

    if total == 0:
        return None

    return correct / total




def calculate_agent_metrics(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:


    total = len(case_results)


    passed = sum(

        1

        for item in case_results

        if item.get(
            "passed",
            False
        )

    )


    # ==================================================
    # Overall
    # ==================================================

    overall = {

        "total_cases":

            total,


        "passed_cases":

            passed,


        "failed_cases":

            total - passed,


        "task_success_rate":

            _safe_rate(
                passed,
                total
            ),

    }




    # ==================================================
    # Tool Capability
    # ==================================================

    routing_total = 0
    routing_correct = 0


    retrieval_total = 0
    retrieval_correct = 0


    minimal_total = 0
    minimal_correct = 0



    for item in case_results:


        assertions = item.get(
            "assertions",
            {}
        )


        capabilities = set(

            item.get(
                "capability_groups",
                []
            )

        )



        required = set(

            assertions.get(
                "must_use_capability_groups",
                []
            )

        )



        if required:


            routing_total += 1


            if required.issubset(
                capabilities
            ):

                routing_correct += 1



            if required & {

                "knowledge_retrieval",

                "github_semantic_retrieval",

                "structured_query",

            }:


                retrieval_total += 1


                if item.get(
                    "passed",
                    False
                ):

                    retrieval_correct += 1




        forbidden = set(

            assertions.get(
                "must_not_use_capability_groups",
                []
            )

        )



        if forbidden:


            minimal_total += 1


            if not (
                capabilities
                &
                forbidden
            ):

                minimal_correct += 1





    capability = {


        "tool_routing_accuracy":

            _safe_rate(
                routing_correct,
                routing_total
            ),



        "retrieval_decision_accuracy":

            _safe_rate(
                retrieval_correct,
                retrieval_total
            ),



        "tool_minimality":

            _safe_rate(
                minimal_correct,
                minimal_total
            ),

    }




    # ==================================================
    # Tool Efficiency
    # ==================================================

    tool_calls = []

    llm_calls = []


    tool_success_total = 0

    tool_success_correct = 0



    for item in case_results:


        state = item.get(
            "state",
            {}
        )


        tool_calls.append(

            state.get(
                "tool_calls",
                0
            )

        )


        llm_calls.append(

            state.get(
                "llm_calls",
                0
            )

        )



        trace = state.get(
            "tool_trace",
            []
        )


        if isinstance(
            trace,
            list
        ):


            for tool in trace:


                tool_success_total += 1


                if tool.get(
                    "success",
                    True
                ):

                    tool_success_correct += 1





    tool_efficiency = {


        "average_tool_calls":

            (
                sum(tool_calls)
                /
                len(tool_calls)
                if tool_calls
                else None
            ),



        "average_llm_calls":

            (
                sum(llm_calls)
                /
                len(llm_calls)
                if llm_calls
                else None
            ),



        "tool_success_rate":

            _safe_rate(
                tool_success_correct,
                tool_success_total
            ),

    }




    # ==================================================
    # Memory
    # ==================================================

    memory_total = 0

    memory_correct = 0



    for item in case_results:


        if item.get(
            "category"
        ) in MEMORY_CATEGORIES:


            memory_total += 1


            if item.get(
                "passed",
                False
            ):

                memory_correct += 1




    memory = {


        "memory_recall_accuracy":

            _safe_rate(
                memory_correct,
                memory_total
            ),



        "memory_cases":

            memory_total,

    }




    # ==================================================
    # Safety
    # ==================================================

    hitl_total = 0

    hitl_correct = 0



    for item in case_results:


        assertions = item.get(
            "assertions",
            {}
        )


        if (
            "pending_human_approval"
            in assertions
        ):


            hitl_total += 1


            if item.get(
                "passed",
                False
            ):

                hitl_correct += 1





    safety = {


        "hitl_compliance":

            _safe_rate(
                hitl_correct,
                hitl_total
            ),

    }





    # ==================================================
    # Stability
    # ==================================================

    recovery_total = 0

    recovery_correct = 0



    for item in case_results:


        if item.get(
            "category"
        ) in RECOVERY_CATEGORIES:


            recovery_total += 1


            if item.get(
                "passed",
                False
            ):

                recovery_correct += 1




    stability = {


        "error_recovery_rate":

            _safe_rate(
                recovery_correct,
                recovery_total
            ),


        "recovery_cases":

            recovery_total,

    }





    # ==================================================
    # Category Breakdown
    # ==================================================

    category_metrics = {}



    categories = set(

        item.get(
            "category",
            "unknown"
        )

        for item in case_results

    )



    for category in categories:


        subset = [

            item

            for item in case_results

            if item.get(
                "category"
            )
            ==
            category

        ]


        category_metrics[category] = {


            "total":

                len(subset),



            "passed":

                sum(

                    1

                    for x in subset

                    if x.get(
                        "passed",
                        False
                    )

                ),



            "success_rate":

                _safe_rate(

                    sum(

                        1

                        for x in subset

                        if x.get(
                            "passed",
                            False
                        )

                    ),

                    len(subset)

                )

        }





    # ==================================================
    # Performance
    # ==================================================

    latency = [

        item.get(
            "latency_ms",
            0
        )

        for item in case_results

    ]



    latency_sorted = sorted(
        latency
    )



    performance = {


        "average_latency_ms":

            (
                sum(latency)
                /
                len(latency)
                if latency
                else None
            ),



        "latency_p50_ms":

            (
                latency_sorted[
                    len(latency_sorted)//2
                ]

                if latency_sorted

                else None

            ),



        "latency_p95_ms":

            (
                latency_sorted[
                    int(
                        len(latency_sorted)
                        *
                        0.95
                    )
                    -
                    1
                ]

                if latency_sorted

                else None

            ),

    }




    return {


        "overall":

            overall,



        "capability":

            capability,



        "tool_efficiency":

            tool_efficiency,



        "memory":

            memory,



        "safety":

            safety,



        "stability":

            stability,



        "category_metrics":

            category_metrics,



        "performance":

            performance,

    }