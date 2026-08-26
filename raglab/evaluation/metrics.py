"""Metrics for Context Planner evaluation."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Sequence

from raglab.evaluation.schemas import (
    PlannerCaseResult,
)


def _safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator <= 0:
        return None

    return numerator / denominator


def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float | None:
    clean = sorted(
        float(value)
        for value in values
    )

    if not clean:
        return None

    if len(clean) == 1:
        return clean[0]

    rank = (
        percentile
        / 100.0
        * (len(clean) - 1)
    )

    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return clean[lower]

    fraction = rank - lower

    return (
        clean[lower]
        + (
            clean[upper]
            - clean[lower]
        )
        * fraction
    )


def _usage_token(
    usage: dict[str, Any],
    *names: str,
) -> int:
    for name in names:
        value = usage.get(name)

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            return int(value)

    return 0


def compute_planner_metrics(
    results: Sequence[
        PlannerCaseResult
    ],
) -> dict[str, Any]:
    total = len(results)

    passed = sum(
        result.passed
        for result in results
    )

    errors = [
        result
        for result in results
        if result.error_type
    ]

    field_total: Counter[str] = Counter()
    field_correct: Counter[str] = Counter()

    scope_confusion: dict[
        str,
        Counter[str]
    ] = defaultdict(Counter)

    none_expected = 0
    wrong_history_injection = 0

    history_expected = 0
    missed_history_dependency = 0

    retrieval_not_required = 0
    unnecessary_retrieval = 0

    retrieval_required = 0
    retrieval_blocked = 0

    raw_evidence_total = 0
    raw_evidence_correct = 0

    previous_answer_total = 0
    previous_answer_correct = 0

    ltm_total = 0
    ltm_correct = 0

    latencies = []
    navigation_chars = []

    input_tokens_total = 0
    output_tokens_total = 0
    total_tokens_total = 0
    usage_cases = 0

    category_total: Counter[str] = Counter()
    category_passed: Counter[str] = Counter()

    for result in results:
        category_total[
            result.category
        ] += 1

        if result.passed:
            category_passed[
                result.category
            ] += 1

        if result.error_type:
            continue

        actual = result.actual
        expected = result.expected

        latencies.append(
            float(result.latency_ms)
        )

        navigation_chars.append(
            int(
                result.navigation_characters
            )
        )

        usage = dict(
            result.usage_metadata
            or {}
        )

        input_tokens = _usage_token(
            usage,
            "input_tokens",
            "prompt_tokens",
        )

        output_tokens = _usage_token(
            usage,
            "output_tokens",
            "completion_tokens",
        )

        total_tokens = _usage_token(
            usage,
            "total_tokens",
        )

        if (
            input_tokens
            or output_tokens
            or total_tokens
        ):
            usage_cases += 1
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens

            if total_tokens:
                total_tokens_total += (
                    total_tokens
                )
            else:
                total_tokens_total += (
                    input_tokens
                    + output_tokens
                )

        for field, expected_value in (
            expected.items()
        ):
            field_total[field] += 1

            if actual.get(field) == (
                expected_value
            ):
                field_correct[field] += 1

        if "history_scope" in expected:
            expected_scope = str(
                expected[
                    "history_scope"
                ]
            )
            actual_scope = str(
                actual.get(
                    "history_scope",
                    "<missing>",
                )
            )

            scope_confusion[
                expected_scope
            ][actual_scope] += 1

            if expected_scope == "none":
                none_expected += 1

                if actual_scope != "none":
                    wrong_history_injection += 1
            else:
                history_expected += 1

                if actual_scope == "none":
                    missed_history_dependency += 1

        if (
            expected.get(
                "external_retrieval_required"
            )
            is False
        ):
            retrieval_not_required += 1

            if (
                actual.get(
                    "external_retrieval_required"
                )
                is True
            ):
                unnecessary_retrieval += 1

        if (
            expected.get(
                "external_retrieval_required"
            )
            is True
        ):
            retrieval_required += 1

            if (
                actual.get(
                    "external_retrieval_required"
                )
                is not True
                or actual.get(
                    "external_retrieval_allowed"
                )
                is False
            ):
                retrieval_blocked += 1

        if (
            "raw_tool_evidence_required"
            in expected
        ):
            raw_evidence_total += 1

            if (
                actual.get(
                    "raw_tool_evidence_required"
                )
                == expected[
                    "raw_tool_evidence_required"
                ]
            ):
                raw_evidence_correct += 1

        if (
            "previous_answer_required"
            in expected
        ):
            previous_answer_total += 1

            if (
                actual.get(
                    "previous_answer_required"
                )
                == expected[
                    "previous_answer_required"
                ]
            ):
                previous_answer_correct += 1

        if (
            "long_term_memory_required"
            in expected
        ):
            ltm_total += 1

            if (
                actual.get(
                    "long_term_memory_required"
                )
                == expected[
                    "long_term_memory_required"
                ]
            ):
                ltm_correct += 1

    per_field = {}

    for field in sorted(
        field_total
    ):
        per_field[field] = {
            "correct": (
                field_correct[field]
            ),
            "total": (
                field_total[field]
            ),
            "accuracy": (
                _safe_rate(
                    field_correct[field],
                    field_total[field],
                )
            ),
        }

    per_category = {}

    for category in sorted(
        category_total
    ):
        per_category[category] = {
            "passed": (
                category_passed[
                    category
                ]
            ),
            "total": (
                category_total[
                    category
                ]
            ),
            "pass_rate": (
                _safe_rate(
                    category_passed[
                        category
                    ],
                    category_total[
                        category
                    ],
                )
            ),
        }

    return {
        "case_count": total,
        "case_pass_count": passed,
        "case_pass_rate": (
            _safe_rate(
                passed,
                total,
            )
        ),
        "error_count": len(
            errors
        ),
        "error_rate": (
            _safe_rate(
                len(errors),
                total,
            )
        ),
        "per_field": per_field,
        "per_category": per_category,
        "history_scope_confusion": {
            expected_scope: dict(
                counts
            )
            for expected_scope, counts
            in scope_confusion.items()
        },
        "wrong_history_injection": {
            "count": (
                wrong_history_injection
            ),
            "eligible_cases": (
                none_expected
            ),
            "rate": (
                _safe_rate(
                    wrong_history_injection,
                    none_expected,
                )
            ),
        },
        "missed_history_dependency": {
            "count": (
                missed_history_dependency
            ),
            "eligible_cases": (
                history_expected
            ),
            "rate": (
                _safe_rate(
                    missed_history_dependency,
                    history_expected,
                )
            ),
        },
        "unnecessary_external_retrieval": {
            "count": (
                unnecessary_retrieval
            ),
            "eligible_cases": (
                retrieval_not_required
            ),
            "rate": (
                _safe_rate(
                    unnecessary_retrieval,
                    retrieval_not_required,
                )
            ),
        },
        "retrieval_block": {
            "count": (
                retrieval_blocked
            ),
            "eligible_cases": (
                retrieval_required
            ),
            "rate": (
                _safe_rate(
                    retrieval_blocked,
                    retrieval_required,
                )
            ),
        },
        "raw_evidence_decision_accuracy": (
            _safe_rate(
                raw_evidence_correct,
                raw_evidence_total,
            )
        ),
        "previous_answer_decision_accuracy": (
            _safe_rate(
                previous_answer_correct,
                previous_answer_total,
            )
        ),
        "long_term_memory_decision_accuracy": (
            _safe_rate(
                ltm_correct,
                ltm_total,
            )
        ),
        "latency_ms": {
            "mean": (
                statistics.fmean(
                    latencies
                )
                if latencies
                else None
            ),
            "median": (
                statistics.median(
                    latencies
                )
                if latencies
                else None
            ),
            "p95": (
                _percentile(
                    latencies,
                    95,
                )
            ),
            "min": (
                min(latencies)
                if latencies
                else None
            ),
            "max": (
                max(latencies)
                if latencies
                else None
            ),
        },
        "navigation_characters": {
            "mean": (
                statistics.fmean(
                    navigation_chars
                )
                if navigation_chars
                else None
            ),
            "max": (
                max(
                    navigation_chars
                )
                if navigation_chars
                else None
            ),
        },
        "usage_tokens": {
            "cases_with_usage": (
                usage_cases
            ),
            "input_tokens_total": (
                input_tokens_total
            ),
            "output_tokens_total": (
                output_tokens_total
            ),
            "total_tokens_total": (
                total_tokens_total
            ),
            "avg_total_tokens_per_usage_case": (
                total_tokens_total
                / usage_cases
                if usage_cases
                else None
            ),
        },
    }
