from __future__ import annotations

from typing import Any



def _safe_rate(
    correct: int,
    total: int,
):
    if total == 0:
        return None

    return correct / total



def calculate_context_metrics(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Context Pipeline Evaluation Metrics.

    Statistics:

    1. Planner execution
    2. Context retrieval execution
    3. Context token usage
    4. Context budget fitting
    5. Context utilization
    6. Compression efficiency
    7. Tool message integrity
    """


    planner_total = 0
    planner_success = 0


    retrieval_total = 0
    retrieval_success = 0



    context_tokens = []

    utilization_rates = []


    compression_ratios = []

    compression_saved_tokens = []



    budget_total = 0
    budget_fit = 0



    integrity_total = 0
    integrity_success = 0




    for item in case_results:


        state = item.get(
            "state",
            {}
        )


        context_pipeline = state.get(
            "context_pipeline",
            {}
        )


        if not context_pipeline:

            continue



        # ==================================
        # Planner
        # ==================================

        plan = context_pipeline.get(
            "plan",
            {}
        )


        planner_total += 1


        if isinstance(
            plan,
            dict
        ) and plan:

            planner_success += 1




        # ==================================
        # Retrieval
        # ==================================

        retrieval = context_pipeline.get(
            "retrieval",
            {}
        )


        retrieval_total += 1


        if isinstance(
            retrieval,
            dict
        ):

            retrieval_success += 1




        # ==================================
        # Budget
        # ==================================

        budget = context_pipeline.get(
            "budget",
            {}
        )


        if isinstance(
            budget,
            dict
        ):


            budget_total += 1


            if budget.get(
                "fits",
                False
            ):

                budget_fit += 1



            estimated_tokens = budget.get(
                "estimated_message_tokens"
            )


            context_limit = budget.get(
                "model_context_limit_tokens"
            )


            if isinstance(
                estimated_tokens,
                int
            ):

                context_tokens.append(
                    estimated_tokens
                )



            if (
                isinstance(
                    estimated_tokens,
                    int
                )

                and

                isinstance(
                    context_limit,
                    int
                )

                and context_limit > 0

            ):


                utilization_rates.append(

                    estimated_tokens
                    /
                    context_limit

                )





        # ==================================
        # Compression
        # ==================================

        compression = context_pipeline.get(
            "compression",
            {}
        )


        if isinstance(
            compression,
            dict
        ):


            ratio = compression.get(
                "compression_ratio"
            )


            if isinstance(
                ratio,
                (float, int)
            ):

                compression_ratios.append(
                    float(ratio)
                )



            saved_tokens = compression.get(
                "tokens_saved"
            )


            if isinstance(
                saved_tokens,
                int
            ):

                compression_saved_tokens.append(
                    saved_tokens
                )





        # ==================================
        # Integrity
        # ==================================

        integrity_total += 1


        if context_pipeline.get(
            "tool_pair_integrity_ok",
            False
        ):

            integrity_success += 1





    return {


        "planner_execution_rate":

            _safe_rate(
                planner_success,
                planner_total
            ),



        "history_retrieval_execution_rate":

            _safe_rate(
                retrieval_success,
                retrieval_total
            ),



        "average_context_tokens":

            (
                sum(context_tokens)
                /
                len(context_tokens)

                if context_tokens

                else None
            ),



        "context_budget_fit_rate":

            _safe_rate(
                budget_fit,
                budget_total
            ),



        "average_context_utilization_rate":

            (
                sum(utilization_rates)
                /
                len(utilization_rates)

                if utilization_rates

                else None
            ),



        "compression_activation_rate":

            _safe_rate(

                len(
                    compression_ratios
                ),

                len(
                    case_results
                )

            ),



        "average_compression_ratio":

            (
                sum(compression_ratios)
                /
                len(compression_ratios)

                if compression_ratios

                else None

            ),



        "average_compression_saved_tokens":

            (
                sum(compression_saved_tokens)
                /
                len(compression_saved_tokens)

                if compression_saved_tokens

                else 0

            ),



        "tool_pair_integrity_rate":

            _safe_rate(
                integrity_success,
                integrity_total
            ),



        "evaluated_context_cases":

            planner_total,

    }