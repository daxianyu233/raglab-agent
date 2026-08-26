from __future__ import annotations

from typing import Any



def _safe_average(
    values: list[int | float],
) -> float | None:

    if not values:
        return None

    return sum(values) / len(values)



def _extract_usage_metadata(
    state: dict[str, Any],
) -> list[dict[str, float]]:

    """
    从 Agent model_trace 中提取 LLM token 使用信息。

    兼容：
    - input_tokens
    - prompt_tokens
    - output_tokens
    - completion_tokens
    - total_tokens

    """

    model_trace = state.get(
        "model_trace",
        [],
    )


    if not isinstance(
        model_trace,
        list,
    ):

        return []



    usages = []



    for trace in model_trace:


        if not isinstance(
            trace,
            dict,
        ):

            continue



        usage = trace.get(
            "usage_metadata",
            {},
        )


        if not isinstance(
            usage,
            dict,
        ):

            continue



        input_tokens = (

            usage.get(
                "input_tokens",
                0,
            )

            or

            usage.get(
                "prompt_tokens",
                0,
            )

            or

            0

        )



        output_tokens = (

            usage.get(
                "output_tokens",
                0,
            )

            or

            usage.get(
                "completion_tokens",
                0,
            )

            or

            0

        )



        total_tokens = (

            usage.get(
                "total_tokens",
                0,
            )

            or

            (
                input_tokens
                +
                output_tokens
            )

        )



        usages.append(

            {

                "input_tokens":

                    float(
                        input_tokens
                    ),


                "output_tokens":

                    float(
                        output_tokens
                    ),


                "total_tokens":

                    float(
                        total_tokens
                    ),

            }

        )



    return usages




def calculate_token_metrics(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:

    """
    LLM Token Usage Metrics.

    Statistics:

    1. Average input tokens
    2. Average output tokens
    3. Average total tokens
    4. Maximum token consumption
    5. LLM call count

    """

    input_tokens = []

    output_tokens = []

    total_tokens = []



    for item in case_results:


        state = item.get(
            "state",
            {},
        )


        usages = _extract_usage_metadata(
            state
        )



        for usage in usages:


            input_tokens.append(

                usage[
                    "input_tokens"
                ]

            )


            output_tokens.append(

                usage[
                    "output_tokens"
                ]

            )


            total_tokens.append(

                usage[
                    "total_tokens"
                ]

            )



    return {


        "average_input_tokens":

            _safe_average(
                input_tokens
            ),



        "average_output_tokens":

            _safe_average(
                output_tokens
            ),



        "average_total_tokens":

            _safe_average(
                total_tokens
            ),



        "max_total_tokens":

            (
                max(total_tokens)

                if total_tokens

                else None
            ),



        "llm_call_count":

            len(
                total_tokens
            ),

    }