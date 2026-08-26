from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from raglab.evaluation.models.e2e_case import (
    E2ECaseResult,
)


def calculate_e2e_metrics(
    results: Iterable[E2ECaseResult],
) -> dict:


    results = list(results)


    total = len(results)


    if total == 0:

        return {}



    passed = sum(
        1
        for r in results
        if r.passed
    )


    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "passed": 0,
        }
    )


    for r in results:

        item = category_stats[
            r.category
        ]

        item["total"] += 1

        if r.passed:

            item["passed"] += 1



    category_metrics = {}


    for category, stat in category_stats.items():

        category_metrics[category] = {

            "total":
                stat["total"],

            "passed":
                stat["passed"],

            "success_rate":
                stat["passed"]
                /
                stat["total"],

        }



    latency_values = [

        r.latency_ms

        for r in results

        if r.latency_ms is not None

    ]



    return {


        "total_cases":
            total,


        "passed_cases":
            passed,


        "failed_cases":
            total - passed,


        "overall_success_rate":
            passed / total,


        "category_metrics":
            category_metrics,


        "average_latency_ms":
            (
                sum(latency_values)
                /
                len(latency_values)
            )
            if latency_values
            else 0,

    }