"""Evaluation schemas for RAG-LAB.

Phase 8A focuses on Context Planner evaluation.
The evaluation layer is intentionally independent from production Agent logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CORE_EXACT_FIELDS = (
    "history_required",
    "history_scope",
    "previous_answer_required",
    "raw_tool_evidence_required",
    "external_retrieval_required",
    "external_retrieval_allowed",
    "long_term_memory_required",
)


@dataclass(frozen=True)
class PlannerEvaluationCase:
    case_id: str
    category: str
    user_input: str

    # Planner only sees availability booleans, not history bodies.
    previous_turn_available: bool = True
    historical_archive_available: bool = True
    long_term_memory_available: bool = True

    # Only fields present here are scored by exact match.
    expected: dict[str, Any] = field(
        default_factory=dict
    )

    # Semantic/shape assertions for open fields such as history_query.
    nonempty_fields: tuple[str, ...] = ()
    empty_fields: tuple[str, ...] = ()

    tags: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class PlannerCaseResult:
    case_id: str
    category: str
    passed: bool

    expected: dict[str, Any]
    actual: dict[str, Any]

    mismatches: tuple[str, ...]
    latency_ms: float
    usage_metadata: dict[str, Any]
    navigation_characters: int

    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "mismatches": list(self.mismatches),
            "latency_ms": self.latency_ms,
            "usage_metadata": self.usage_metadata,
            "navigation_characters": (
                self.navigation_characters
            ),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
