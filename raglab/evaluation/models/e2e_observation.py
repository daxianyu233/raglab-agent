from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class E2ETurnObservation:
    """
    Agent E2E统一观察对象。

    Evaluation 不依赖生产 Agent 内部结构。
    """


    answer: str = ""


    completed_normally: bool = False


    tool_calls: list[dict[str, Any]] = field(
        default_factory=list
    )


    capability_groups_used: list[str] = field(
        default_factory=list
    )


    total_latency_ms: float | None = None



    input_tokens: int | None = None

    output_tokens: int | None = None

    total_tokens: int | None = None



    pending_human_approval: bool = False



    write_side_effect_count: int = 0



    state: dict[str, Any] = field(
        default_factory=dict
    )