"""Context Planner evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from raglab.agent.context_plan import (
    NavigationContext,
    TurnIndexItem,
)
from raglab.agent.context_planner import (
    ContextPlanner,
)
from raglab.evaluation.metrics import (
    compute_planner_metrics,
)
from raglab.evaluation.schemas import (
    PlannerCaseResult,
    PlannerEvaluationCase,
)


def load_planner_cases(
    path: Path,
) -> list[PlannerEvaluationCase]:
    raw = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(raw, list):
        raise ValueError(
            "Planner evaluation dataset 顶层必须是 JSON array。"
        )

    cases = []

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "每个 Planner case 必须是 JSON object。"
            )

        cases.append(
            PlannerEvaluationCase(
                case_id=str(
                    item["case_id"]
                ),
                category=str(
                    item["category"]
                ),
                user_input=str(
                    item["user_input"]
                ),
                previous_turn_available=bool(
                    item.get(
                        "previous_turn_available",
                        True,
                    )
                ),
                historical_archive_available=bool(
                    item.get(
                        "historical_archive_available",
                        True,
                    )
                ),
                long_term_memory_available=bool(
                    item.get(
                        "long_term_memory_available",
                        True,
                    )
                ),
                expected=dict(
                    item.get(
                        "expected",
                        {},
                    )
                ),
                nonempty_fields=tuple(
                    str(value)
                    for value
                    in item.get(
                        "nonempty_fields",
                        [],
                    )
                ),
                empty_fields=tuple(
                    str(value)
                    for value
                    in item.get(
                        "empty_fields",
                        [],
                    )
                ),
                tags=tuple(
                    str(value)
                    for value
                    in item.get(
                        "tags",
                        [],
                    )
                ),
                notes=str(
                    item.get(
                        "notes",
                        "",
                    )
                ),
            )
        )

    ids = [
        case.case_id
        for case in cases
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Planner evaluation dataset 存在重复 case_id。"
        )

    return cases


def _dummy_previous_turn() -> TurnIndexItem:
    return TurnIndexItem(
        turn_id="eval-previous-turn",
        user_goal=(
            "占位目录项；Planner 不应读取其正文。"
        ),
        assistant_outcome=(
            "占位结果；仅用于声明历史可用。"
        ),
        entities=[],
        has_tool_evidence=True,
        tool_names=[
            "evaluation_placeholder_tool"
        ],
    )


def _dummy_historical_turn() -> TurnIndexItem:
    return TurnIndexItem(
        turn_id="eval-historical-turn",
        user_goal=(
            "较早历史占位目录项。"
        ),
        assistant_outcome=(
            "仅用于声明历史归档可用。"
        ),
        entities=[],
        has_tool_evidence=True,
        tool_names=[
            "evaluation_placeholder_tool"
        ],
    )


def build_navigation_context(
    case: PlannerEvaluationCase,
) -> NavigationContext:
    recent_turns = (
        [
            _dummy_previous_turn()
        ]
        if case.previous_turn_available
        else []
    )

    history_candidates = (
        [
            _dummy_historical_turn()
        ]
        if case.historical_archive_available
        else []
    )

    thread_summary = (
        "该线程存在更早历史；Planner 不可读取历史正文。"
        if case.historical_archive_available
        else ""
    )

    capability_catalog = [
        "conversation_history: 可恢复当前线程历史 Human/AI/Tool 证据",
        "knowledge_rag: 可检索 PDF/知识库语义资料",
        "github_rag: 可检索已有 GitHub 项目、热点和日报语义资料",
        "github_sql: 可查询 GitHub Intelligence 结构化数据库",
        "skills: 主 Agent 可按需加载动态 Skill；Planner 不直接加载",
    ]

    if case.long_term_memory_available:
        capability_catalog.append(
            "long_term_memory: 可检索跨线程用户事实、偏好和长期记忆"
        )

    return NavigationContext(
        current_user_input=(
            case.user_input
        ),
        thread_summary=(
            thread_summary
        ),
        recent_turns=(
            recent_turns
        ),
        history_candidates=(
            history_candidates
        ),
        capability_catalog=(
            capability_catalog
        ),
        runtime_notes=[
            "evaluation_mode=true"
        ],
    )


def _check_case(
    case: PlannerEvaluationCase,
    actual: dict[str, Any],
) -> list[str]:
    mismatches = []

    for field, expected_value in (
        case.expected.items()
    ):
        actual_value = actual.get(
            field,
            "<missing>",
        )

        if actual_value != expected_value:
            mismatches.append(
                f"{field}: expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )

    for field in case.nonempty_fields:
        value = actual.get(field)

        if not str(
            value
            or ""
        ).strip():
            mismatches.append(
                f"{field}: expected non-empty, actual={value!r}"
            )

    for field in case.empty_fields:
        value = actual.get(field)

        if (
            value is not None
            and str(value).strip()
        ):
            mismatches.append(
                f"{field}: expected empty, actual={value!r}"
            )

    return mismatches


class ContextPlannerEvaluator:
    def __init__(
        self,
        *,
        planner: ContextPlanner,
    ) -> None:
        self.planner = planner

    def evaluate_case(
        self,
        case: PlannerEvaluationCase,
    ) -> PlannerCaseResult:
        context = (
            build_navigation_context(
                case
            )
        )

        try:
            result = self.planner.plan(
                context
            )

            actual = (
                result.plan.model_dump(
                    mode="json"
                )
            )

            mismatches = (
                _check_case(
                    case,
                    actual,
                )
            )

            return PlannerCaseResult(
                case_id=case.case_id,
                category=case.category,
                passed=not mismatches,
                expected=dict(
                    case.expected
                ),
                actual=actual,
                mismatches=tuple(
                    mismatches
                ),
                latency_ms=(
                    result.latency_ms
                ),
                usage_metadata=dict(
                    result.usage_metadata
                    or {}
                ),
                navigation_characters=(
                    result.navigation_characters
                ),
            )

        except Exception as exc:
            return PlannerCaseResult(
                case_id=case.case_id,
                category=case.category,
                passed=False,
                expected=dict(
                    case.expected
                ),
                actual={},
                mismatches=(
                    "planner_exception",
                ),
                latency_ms=0.0,
                usage_metadata={},
                navigation_characters=0,
                error_type=(
                    type(exc).__name__
                ),
                error_message=str(
                    exc
                ),
            )

    def evaluate(
        self,
        cases: Sequence[
            PlannerEvaluationCase
        ],
    ) -> tuple[
        list[PlannerCaseResult],
        dict[str, Any],
    ]:
        results = [
            self.evaluate_case(
                case
            )
            for case in cases
        ]

        metrics = (
            compute_planner_metrics(
                results
            )
        )

        return (
            results,
            metrics,
        )
