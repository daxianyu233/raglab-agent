"""Pure regression test for Phase 8A evaluation infrastructure.

Run:
    python -m scripts.test_evaluation_harness
"""

from raglab.evaluation.metrics import (
    compute_planner_metrics,
)
from raglab.evaluation.schemas import (
    PlannerCaseResult,
)


def main():
    results = [
        PlannerCaseResult(
            case_id="a",
            category="self_contained",
            passed=True,
            expected={
                "history_scope": "none",
                "external_retrieval_required": False,
            },
            actual={
                "history_scope": "none",
                "external_retrieval_required": False,
                "external_retrieval_allowed": True,
            },
            mismatches=(),
            latency_ms=100.0,
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
            navigation_characters=500,
        ),
        PlannerCaseResult(
            case_id="b",
            category="fresh_retrieval",
            passed=False,
            expected={
                "history_scope": "none",
                "external_retrieval_required": True,
                "external_retrieval_allowed": True,
            },
            actual={
                "history_scope": "previous_turn",
                "external_retrieval_required": False,
                "external_retrieval_allowed": True,
            },
            mismatches=(
                "history_scope",
                "external_retrieval_required",
            ),
            latency_ms=200.0,
            usage_metadata={
                "input_tokens": 110,
                "output_tokens": 25,
                "total_tokens": 135,
            },
            navigation_characters=520,
        ),
    ]

    metrics = compute_planner_metrics(
        results
    )

    assert metrics[
        "case_count"
    ] == 2

    assert metrics[
        "case_pass_rate"
    ] == 0.5

    assert metrics[
        "wrong_history_injection"
    ][
        "count"
    ] == 1

    assert metrics[
        "retrieval_block"
    ][
        "count"
    ] == 1

    assert metrics[
        "usage_tokens"
    ][
        "total_tokens_total"
    ] == 255

    print(
        "Evaluation Harness Phase 8A "
        "纯逻辑回归测试通过"
    )


if __name__ == "__main__":
    main()
