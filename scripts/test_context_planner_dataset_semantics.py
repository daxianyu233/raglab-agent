"""Validate adjudicated Context Planner dataset semantics.

Run:
    python -m scripts.test_context_planner_dataset_semantics
"""

from __future__ import annotations

import json

from raglab.settings import PROJECT_ROOT


DATASET = (
    PROJECT_ROOT
    / "raglab"
    / "evaluation"
    / "datasets"
    / "context_planner_v1_1_adjudicated.json"
)


def main() -> None:
    cases = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    assert len(cases) == 38

    ids = {
        case["case_id"]
        for case in cases
    }

    assert len(ids) == 38

    for case in cases:
        expected = case.get(
            "expected",
            {}
        )
        tags = set(
            case.get(
                "tags",
                []
            )
        )

        if (
            "raw_tool_evidence"
            in tags
        ):
            assert (
                expected.get(
                    "raw_tool_evidence_required"
                )
                is True
            ), case["case_id"]

        if (
            "conversation_record_only"
            in tags
        ):
            assert (
                expected.get(
                    "raw_tool_evidence_required"
                )
                is False
            ), case["case_id"]

        if (
            case["category"]
            == "recent_turns"
        ):
            assert (
                "previous_answer_required"
                not in expected
            ), case["case_id"]

        if (
            "fresh_external"
            in tags
        ):
            assert (
                expected.get(
                    "external_retrieval_required"
                )
                is True
            ), case["case_id"]

        if (
            "history_only"
            in tags
        ):
            assert (
                expected.get(
                    "external_retrieval_required"
                )
                is False
            ), case["case_id"]

            assert (
                expected.get(
                    "external_retrieval_allowed"
                )
                is False
            ), case["case_id"]

    print(
        "Context Planner Dataset v1.1 "
        "语义一致性测试通过"
    )


if __name__ == "__main__":
    main()
