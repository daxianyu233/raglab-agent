"""Repeatedly probe one Context Planner evaluation case.

Use this to distinguish a systematic Planner bug from LLM instability.

Run:
    python -m scripts.probe_context_planner_case fresh_004 --repeat 7
"""

from __future__ import annotations

import argparse
from collections import Counter

from raglab.agent.context_planner import (
    ContextPlanner,
)
from raglab.evaluation.planner_evaluator import (
    ContextPlannerEvaluator,
    load_planner_cases,
)
from raglab.settings import (
    CONFIG_DIR,
    PROJECT_ROOT,
)
from scripts.ask_rag import (
    create_deepseek_model,
    load_yaml_config,
    require_mapping,
)


DATASET = (
    PROJECT_ROOT
    / "raglab"
    / "evaluation"
    / "datasets"
    / "context_planner_v1_1_adjudicated.json"
)


def build_planner() -> ContextPlanner:
    config = load_yaml_config(
        CONFIG_DIR
        / "agent.yaml"
    )

    model_config = require_mapping(
        config,
        "model",
    )

    return ContextPlanner(
        chat_model=(
            create_deepseek_model(
                model_config
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "case_id",
        type=str,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError(
            "--repeat 必须大于 0。"
        )

    cases = load_planner_cases(
        DATASET
    )

    selected = [
        case
        for case in cases
        if case.case_id
        == args.case_id
    ]

    if not selected:
        raise ValueError(
            f"找不到 case_id={args.case_id!r}"
        )

    case = selected[0]

    evaluator = (
        ContextPlannerEvaluator(
            planner=(
                build_planner()
            )
        )
    )

    pass_count = 0
    scope_counts = Counter()
    retrieval_required_counts = Counter()
    retrieval_allowed_counts = Counter()

    latencies = []

    print("=" * 88)
    print(
        "Context Planner Targeted Probe"
    )
    print("=" * 88)

    print(
        "Case：",
        case.case_id,
    )
    print(
        "Input：",
        case.user_input,
    )
    print(
        "Expected：",
        case.expected,
    )
    print(
        "Repeat：",
        args.repeat,
    )

    for index in range(
        args.repeat
    ):
        result = (
            evaluator.evaluate_case(
                case
            )
        )

        if result.passed:
            pass_count += 1

        actual = result.actual

        scope_counts[
            str(
                actual.get(
                    "history_scope",
                    "<error>",
                )
            )
        ] += 1

        retrieval_required_counts[
            str(
                actual.get(
                    "external_retrieval_required",
                    "<error>",
                )
            )
        ] += 1

        retrieval_allowed_counts[
            str(
                actual.get(
                    "external_retrieval_allowed",
                    "<error>",
                )
            )
        ] += 1

        latencies.append(
            result.latency_ms
        )

        print()
        print(
            f"Run {index + 1}: "
            f"{'PASS' if result.passed else 'FAIL'}"
        )

        print(
            "  history_scope：",
            actual.get(
                "history_scope"
            ),
        )

        print(
            "  external_retrieval_required：",
            actual.get(
                "external_retrieval_required"
            ),
        )

        print(
            "  external_retrieval_allowed：",
            actual.get(
                "external_retrieval_allowed"
            ),
        )

        if result.mismatches:
            for mismatch in (
                result.mismatches
            ):
                print(
                    "  - ",
                    mismatch,
                )

    print()
    print("=" * 88)
    print("Probe Summary")
    print("=" * 88)

    print(
        "Pass：",
        f"{pass_count}/{args.repeat}",
    )
    print(
        "history_scope distribution：",
        dict(
            scope_counts
        ),
    )
    print(
        "retrieval_required distribution：",
        dict(
            retrieval_required_counts
        ),
    )
    print(
        "retrieval_allowed distribution：",
        dict(
            retrieval_allowed_counts
        ),
    )

    if latencies:
        print(
            "mean latency：",
            f"{sum(latencies) / len(latencies):.2f} ms",
        )


if __name__ == "__main__":
    main()
