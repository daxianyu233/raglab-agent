from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class E2ECase:

    case_id: str

    category: str

    user_input: str

    setup: list[dict[str, Any]] = field(
        default_factory=list
    )

    real_data_dependencies: list[str] = field(
        default_factory=list
    )

    assertions: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class E2ECaseResult:

    case_id: str

    category: str

    passed: bool

    observation: dict[str, Any]

    mismatches: list[str] = field(
        default_factory=list
    )

    latency_ms: float = 0.0

    error_type: str = ""

    error_message: str = ""
