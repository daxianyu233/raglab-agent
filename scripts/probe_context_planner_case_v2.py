"""Targeted Context Planner probe with all decision fields.

Usage:
    python -m scripts.probe_context_planner_case_v2 evidence_004 --repeat 7
    python -m scripts.probe_context_planner_case_v2 recent_002 --repeat 7
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from raglab.agent.context_planner import ContextPlanner
from raglab.evaluation.planner_evaluator import (
    ContextPlannerEvaluator,
    load_planner_cases,
)
from raglab.settings import CONFIG_DIR, PROJECT_ROOT
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

FIELDS = (
    "history_required",
    "history_scope",
    "previous_answer_required",
    "raw_tool_evidence_required",
    "external_retrieval_required",
    "external_retrieval_allowed",
    "long_term_memory_required",
)


def build_planner() -> ContextPlanner:
    config = load_yaml_config(
        CONFIG_DIR / "agent.yaml"
    )

    model_config = require_mapping(
        config,
        "model",
    )

    return ContextPlanner(
        chat_model=create_deepseek_model(
            model_config
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

    matched = [
        case
        for case in cases
        if case.case_id == args.case_id
    ]

    if not matched:
        raise ValueError(
            f"找不到 case_id={args.case_id!r}"
        )

    case = matched[0]

    evaluator = ContextPlannerEvaluator(
        planner=build_planner()
    )

    distributions = defaultdict(Counter)
    pass_count = 0
    latencies = []

    print("=" * 96)
    print("Context Planner Targeted Probe V2")
    print("=" * 96)
    print("Case：", case.case_id)
    print("Category：", case.category)
    print("Input：", case.user_input)
    print("Expected：", case.expected)
    print("Repeat：", args.repeat)

    for index in range(args.repeat):
        result = evaluator.evaluate_case(
            case
        )

        if result.passed:
            pass_count += 1

        latencies.append(
            result.latency_ms
        )

        print()
        print(
            f"Run {index + 1}: "
            f"{'PASS' if result.passed else 'FAIL'}"
        )

        if result.error_type:
            print(
                "  ERROR：",
                result.error_type,
                result.error_message,
            )
            continue

        for field in FIELDS:
            value = result.actual.get(
                field
            )

            distributions[field][
                repr(value)
            ] += 1

            expected_marker = ""

            if field in case.expected:
                expected_value = (
                    case.expected[field]
                )

                expected_marker = (
                    "  ✓"
                    if value == expected_value
                    else (
                        "  ✗ expected="
                        + repr(expected_value)
                    )
                )

            print(
                f"  {field}: "
                f"{value!r}"
                f"{expected_marker}"
            )

        if result.mismatches:
            print("  mismatches：")

            for mismatch in (
                result.mismatches
            ):
                print(
                    "    -",
                    mismatch,
                )

    print()
    print("=" * 96)
    print("Probe Summary")
    print("=" * 96)

    print(
        "Pass：",
        f"{pass_count}/{args.repeat}",
    )

    for field in FIELDS:
        print(
            f"{field} distribution：",
            dict(
                distributions[field]
            ),
        )

    if latencies:
        print(
            "mean latency：",
            f"{sum(latencies) / len(latencies):.2f} ms",
        )


if __name__ == "__main__":
    main()