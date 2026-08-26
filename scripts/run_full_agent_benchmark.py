from __future__ import annotations


import argparse

import json

from pathlib import Path

from typing import Any

from uuid import uuid4



from raglab.evaluation.agent_metrics import (
    calculate_agent_metrics,
)


from raglab.evaluation.context_metrics import (
    calculate_context_metrics,
)


from raglab.evaluation.token_metrics import (
    calculate_token_metrics,
)


from raglab.evaluation.e2e_case_loader import (
    load_e2e_cases,
)


from raglab.evaluation.e2e_assertion import (
    evaluate_case,
)


from raglab.evaluation.models.e2e_case import (
    E2ECaseResult,
)


from raglab.evaluation.adapters.secure_runtime_adapter import (
    SecureRuntimeAdapter,
)



ROOT = Path(__file__).resolve().parents[1]



DATASET_PATH = (

    ROOT
    /
    "raglab"
    /
    "evaluation"
    /
    "datasets"
    /
    "full_agent_e2e_v2.json"

)



CONFIG_PATH = (

    ROOT
    /
    "config"
    /
    "agent.yaml"

)



REPORT_PATH = (

    ROOT
    /
    "reports"
    /
    "full_agent_e2e_report.json"

)





def sanitize_json(
    value: Any,
):

    if value is None:

        return None



    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value



    if isinstance(
        value,
        dict,
    ):

        return {

            str(k):

                sanitize_json(v)

            for k, v in value.items()

        }



    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        return [

            sanitize_json(v)

            for v in value

        ]



    if hasattr(
        value,
        "type",
    ) and hasattr(
        value,
        "content",
    ):

        return {

            "type":

                value.type,


            "content":

                str(
                    value.content
                ),

        }



    return str(value)





def build_metric_inputs(
    results: list[E2ECaseResult],
    cases,
):


    case_map = {

        case.case_id:

            case

        for case in cases

    }



    metric_inputs = []



    for result in results:


        case = case_map.get(
            result.case_id
        )


        if case is None:

            continue



        metric_inputs.append(

            {

                "case_id":

                    result.case_id,


                "category":

                    result.category,


                "passed":

                    result.passed,


                "assertions":

                    case.assertions,


                "capability_groups":

                    result.observation.get(
                        "capabilities",
                        [],
                    ),



                "latency_ms":

                    result.latency_ms,



                "state":

                    result.observation.get(
                        "state",
                        {},
                    ),

            }

        )



    return metric_inputs





def main():



    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--case",

        type=str,

        default=None,

        help="Run one specific benchmark case",

    )



    args = parser.parse_args()




    print("=" * 80)

    print(
        "Full Agent E2E Benchmark"
    )

    print("=" * 80)




    cases = load_e2e_cases(
        DATASET_PATH
    )



    if args.case:


        cases = [

            c

            for c in cases

            if c.case_id == args.case

        ]



        if not cases:

            raise ValueError(

                f"Case not found: {args.case}"

            )




    print(
        f"Loaded cases: {len(cases)}"
    )




    adapter = SecureRuntimeAdapter(
        CONFIG_PATH,
        no_write_external_actions=True,
    )




    results = []




    case_threads = {}
    benchmark_run_id = uuid4().hex[:12]

    for case in cases:



        print()

        print(
            f"[RUN] {case.case_id}"
        )


        print(
            f"Category: {case.category}"
        )



        try:

            thread_id = (
                f"benchmark-{benchmark_run_id}-{case.case_id}"
            )

            for setup in case.setup:
                if setup.get("role") != "pending_real_tool_action":
                    continue

                created_by_case = setup.get("created_by_case")
                if created_by_case and created_by_case in case_threads:
                    thread_id = case_threads[created_by_case]
                    continue

                trigger = (
                    "请动态加载 github-intelligence-update 能力，然后立即调用 "
                    "update_github_intelligence 执行更新；进入人工审批后停止等待。"
                )
                pending_observation = adapter.send(
                    trigger,
                    thread_id=thread_id,
                    user_id="benchmark_user",
                )
                if not pending_observation.pending_human_approval:
                    raise RuntimeError("benchmark setup did not create HITL pending")



            observation = adapter.send(

                case.user_input,


                thread_id=thread_id,


                user_id="benchmark_user",

            )

            case_threads[case.case_id] = thread_id




            passed, errors = evaluate_case(

                observation,

                case.assertions,

            )




            result = E2ECaseResult(


                case_id=

                    case.case_id,



                category=

                    case.category,



                passed=

                    passed,



                observation={


                    "answer":

                        observation.answer,



                    "tools":

                        observation.tool_calls,



                    "capabilities":

                        observation.capability_groups_used,



                    "pending":

                        observation.pending_human_approval,



                    "state":

                        observation.state,



                },



                mismatches=

                    errors,



                latency_ms=

                    observation.total_latency_ms
                    or 0,

            )



            results.append(
                result
            )




            print(

                "PASS"

                if passed

                else

                f"FAIL: {errors}"

            )




        except Exception as e:



            results.append(


                E2ECaseResult(



                    case_id=

                        case.case_id,



                    category=

                        case.category,



                    passed=False,



                    observation={

                        "state": {}

                    },



                    mismatches=[

                        str(e)

                    ],



                    error_type=

                        type(e).__name__,



                    error_message=

                        str(e),

                )

            )



            print(
                "ERROR:",
                e
            )






    metric_inputs = build_metric_inputs(

        results,

        cases,

    )



    print(
        "DEBUG metric inputs:",
        len(metric_inputs)
    )



    if metric_inputs:

        print(

            "DEBUG context_pipeline exists:",

            bool(

                metric_inputs[0]

                .get(
                    "state",
                    {}
                )

                .get(
                    "context_pipeline",
                    {}
                )

            )

        )





    agent_metrics = calculate_agent_metrics(

        metric_inputs

    )



    context_metrics = calculate_context_metrics(

        metric_inputs

    )



    token_metrics = calculate_token_metrics(

        metric_inputs

    )




    metrics = {


        "agent":

            agent_metrics,



        "context":

            context_metrics,



        "token":

            token_metrics,

    }





    report = {


        "metrics":

            metrics,



        "cases":

            [

                {


                    "case_id":

                        r.case_id,



                    "category":

                        r.category,



                    "passed":

                        r.passed,



                    "mismatches":

                        r.mismatches,



                    "latency_ms":

                        r.latency_ms,



                    "observation":

                        sanitize_json(

                            r.observation

                        ),

                }


                for r in results

            ]

    }




    REPORT_PATH.parent.mkdir(

        parents=True,

        exist_ok=True,

    )





    REPORT_PATH.write_text(

        json.dumps(

            report,

            indent=2,

            ensure_ascii=False,

        ),

        encoding="utf-8",

    )





    print()

    print("=" * 80)

    print(
        "Benchmark Finished"
    )

    print("=" * 80)



    print(

        json.dumps(

            metrics,

            indent=2,

            ensure_ascii=False,

        )

    )



    print()

    print(
        "Report:",
        REPORT_PATH
    )





if __name__ == "__main__":

    main()
