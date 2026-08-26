"""Phase 8A: evaluate the real DeepSeek Context Planner.

Run:
    python -m scripts.evaluate_context_planner

Optional:
    python -m scripts.evaluate_context_planner --repeat 3
    python -m scripts.evaluate_context_planner --show-all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "raglab"
    / "evaluation"
    / "datasets"
    / "context_planner_v1.json"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
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

    model = create_deepseek_model(
        model_config
    )

    return ContextPlanner(
        chat_model=model
    )


def fmt_rate(value):
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def print_metrics(
    metrics: dict,
) -> None:
    print()
    print("=" * 88)
    print("Context Planner Evaluation Summary")
    print("=" * 88)

    print(
        "Cases：",
        metrics["case_count"],
    )
    print(
        "Case Pass Rate：",
        fmt_rate(
            metrics[
                "case_pass_rate"
            ]
        ),
    )
    print(
        "Error Rate：",
        fmt_rate(
            metrics[
                "error_rate"
            ]
        ),
    )

    print()
    print("核心错误指标：")

    for key, title in [
        (
            "wrong_history_injection",
            "Wrong History Injection",
        ),
        (
            "missed_history_dependency",
            "Missed History Dependency",
        ),
        (
            "unnecessary_external_retrieval",
            "Unnecessary Retrieval",
        ),
        (
            "retrieval_block",
            "Retrieval Block",
        ),
    ]:
        current = metrics[key]

        print(
            f"  {title}："
            f"{current['count']}/"
            f"{current['eligible_cases']} "
            f"({fmt_rate(current['rate'])})"
        )

    print()
    print("专项决策准确率：")
    print(
        "  Raw Evidence：",
        fmt_rate(
            metrics[
                "raw_evidence_decision_accuracy"
            ]
        ),
    )
    print(
        "  Previous Answer：",
        fmt_rate(
            metrics[
                "previous_answer_decision_accuracy"
            ]
        ),
    )
    print(
        "  Long-term Memory：",
        fmt_rate(
            metrics[
                "long_term_memory_decision_accuracy"
            ]
        ),
    )

    print()
    print("Per-field Accuracy：")

    for field, value in (
        metrics[
            "per_field"
        ].items()
    ):
        print(
            f"  {field}: "
            f"{value['correct']}/"
            f"{value['total']} "
            f"({fmt_rate(value['accuracy'])})"
        )

    print()
    print("Per-category Pass Rate：")

    for category, value in (
        metrics[
            "per_category"
        ].items()
    ):
        print(
            f"  {category}: "
            f"{value['passed']}/"
            f"{value['total']} "
            f"({fmt_rate(value['pass_rate'])})"
        )

    latency = metrics[
        "latency_ms"
    ]

    print()
    print("Latency：")
    print(
        "  mean：",
        (
            f"{latency['mean']:.2f} ms"
            if latency["mean"]
            is not None
            else "N/A"
        ),
    )
    print(
        "  median：",
        (
            f"{latency['median']:.2f} ms"
            if latency["median"]
            is not None
            else "N/A"
        ),
    )
    print(
        "  p95：",
        (
            f"{latency['p95']:.2f} ms"
            if latency["p95"]
            is not None
            else "N/A"
        ),
    )

    usage = metrics[
        "usage_tokens"
    ]

    print()
    print("Planner Token Usage：")
    print(
        "  cases_with_usage：",
        usage[
            "cases_with_usage"
        ],
    )
    print(
        "  input_tokens_total：",
        usage[
            "input_tokens_total"
        ],
    )
    print(
        "  output_tokens_total：",
        usage[
            "output_tokens_total"
        ],
    )
    print(
        "  total_tokens_total：",
        usage[
            "total_tokens_total"
        ],
    )
    print(
        "  avg_total_tokens：",
        usage[
            "avg_total_tokens_per_usage_case"
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "重复完整数据集的次数；"
            "第一版基线建议先用 1，"
            "后续稳定性测试可用 3。"
        ),
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
    )

    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError(
            "--repeat 必须大于 0。"
        )

    cases = load_planner_cases(
        args.dataset
    )

    planner = build_planner()

    evaluator = (
        ContextPlannerEvaluator(
            planner=planner
        )
    )

    all_results = []
    run_metrics = []

    print("=" * 88)
    print("Phase 8A Context Planner Evaluation")
    print("=" * 88)
    print(
        "Dataset：",
        args.dataset,
    )
    print(
        "Case 数量：",
        len(cases),
    )
    print(
        "Repeat：",
        args.repeat,
    )

    for repeat_index in range(
        args.repeat
    ):
        print()
        print(
            f"--- Run "
            f"{repeat_index + 1}/"
            f"{args.repeat} ---"
        )

        results, metrics = (
            evaluator.evaluate(
                cases
            )
        )

        all_results.extend(
            {
                "repeat_index": (
                    repeat_index + 1
                ),
                **result.to_dict(),
            }
            for result in results
        )

        run_metrics.append(
            metrics
        )

        failures = [
            result
            for result in results
            if not result.passed
        ]

        print(
            "Pass：",
            f"{metrics['case_pass_count']}/"
            f"{metrics['case_count']}",
            fmt_rate(
                metrics[
                    "case_pass_rate"
                ]
            ),
        )

        if failures:
            print()
            print("Failures：")

            for result in failures:
                print(
                    f"  [{result.case_id}] "
                    f"{result.category}"
                )

                if result.error_type:
                    print(
                        "    ERROR："
                        f"{result.error_type}: "
                        f"{result.error_message}"
                    )
                else:
                    for mismatch in (
                        result.mismatches
                    ):
                        print(
                            "    - "
                            + mismatch
                        )

        if args.show_all:
            print()
            print("All Cases：")

            for result in results:
                print(
                    f"  "
                    f"{'PASS' if result.passed else 'FAIL'} "
                    f"{result.case_id}"
                )

        print_metrics(
            metrics
        )

    timestamp = time.strftime(
        "%Y%m%d-%H%M%S"
    )

    DEFAULT_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        DEFAULT_REPORT_DIR
        / (
            "context_planner_"
            + timestamp
            + ".json"
        )
    )

    report = {
        "evaluation": (
            "context_planner"
        ),
        "dataset": str(
            args.dataset
        ),
        "case_count": len(
            cases
        ),
        "repeat": args.repeat,
        "run_metrics": (
            run_metrics
        ),
        "results": (
            all_results
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        "Evaluation report：",
        report_path,
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
