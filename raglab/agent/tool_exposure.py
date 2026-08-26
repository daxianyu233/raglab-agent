"""ContextPlan-aware Tool Schema Exposure - Phase 7C.

Active Tools:
    Runtime 当前具备、执行层可以识别的完整工具集合。

Exposed Tools:
    当前 Human Turn 根据 ContextPlan 真正展示给 LLM 的 schema 子集。

安全边界：
- 本模块只负责 schema exposure；
- SecureToolNode 仍负责执行层硬拦截；
- exposure 失败时保持能力可用，由 SecureToolNode 兜底，
  避免由于观测/分类组件故障误伤正常 Agent。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.tools import BaseTool


PolicyResolver = Callable[
    [str],
    Any,
]


@dataclass(frozen=True)
class ToolExposureDecision:
    active_tools: list[BaseTool]
    exposed_tools: list[BaseTool]

    active_tool_names: list[str]
    exposed_tool_names: list[str]
    hidden_tool_names: list[str]

    retrieval_allowed: bool | None
    filtering_applied: bool

    reason: str


def select_tools_for_context(
    *,
    active_tools: Sequence[BaseTool],
    context_pipeline_enabled: bool,
    context_plan: dict[str, Any] | None,
    policy_resolver: PolicyResolver | None,
) -> ToolExposureDecision:
    active = list(
        active_tools
    )

    active_names = [
        str(tool.name)
        for tool in active
    ]

    raw_plan = (
        context_plan
        if isinstance(
            context_plan,
            dict,
        )
        else {}
    )

    if not context_pipeline_enabled:
        return ToolExposureDecision(
            active_tools=active,
            exposed_tools=active,
            active_tool_names=active_names,
            exposed_tool_names=list(
                active_names
            ),
            hidden_tool_names=[],
            retrieval_allowed=None,
            filtering_applied=False,
            reason=(
                "context_pipeline_disabled"
            ),
        )

    retrieval_allowed = (
        raw_plan.get(
            "external_retrieval_allowed"
        )
    )

    if not isinstance(
        retrieval_allowed,
        bool,
    ):
        return ToolExposureDecision(
            active_tools=active,
            exposed_tools=active,
            active_tool_names=active_names,
            exposed_tool_names=list(
                active_names
            ),
            hidden_tool_names=[],
            retrieval_allowed=None,
            filtering_applied=False,
            reason=(
                "retrieval_permission_unspecified"
            ),
        )

    if retrieval_allowed:
        return ToolExposureDecision(
            active_tools=active,
            exposed_tools=active,
            active_tool_names=active_names,
            exposed_tool_names=list(
                active_names
            ),
            hidden_tool_names=[],
            retrieval_allowed=True,
            filtering_applied=False,
            reason=(
                "retrieval_allowed"
            ),
        )

    if not callable(
        policy_resolver
    ):
        # Exposure 层 fail-open；
        # SecureToolNode 仍会在执行层 fail-closed。
        return ToolExposureDecision(
            active_tools=active,
            exposed_tools=active,
            active_tool_names=active_names,
            exposed_tool_names=list(
                active_names
            ),
            hidden_tool_names=[],
            retrieval_allowed=False,
            filtering_applied=False,
            reason=(
                "policy_resolver_unavailable_"
                "runtime_backstop_required"
            ),
        )

    exposed: list[
        BaseTool
    ] = []

    hidden_names: list[
        str
    ] = []

    for tool in active:
        tool_name = str(
            tool.name
        ).strip()

        policy = (
            policy_resolver(
                tool_name
            )
        )

        access_type = str(
            getattr(
                policy,
                "context_access_type",
                "",
            )
            or ""
        ).strip().upper()

        if (
            access_type
            == "RETRIEVAL"
        ):
            hidden_names.append(
                tool_name
            )
            continue

        exposed.append(
            tool
        )

    exposed_names = [
        str(tool.name)
        for tool in exposed
    ]

    return ToolExposureDecision(
        active_tools=active,
        exposed_tools=exposed,
        active_tool_names=active_names,
        exposed_tool_names=(
            exposed_names
        ),
        hidden_tool_names=(
            hidden_names
        ),
        retrieval_allowed=False,
        filtering_applied=bool(
            hidden_names
        ),
        reason=(
            "retrieval_tools_hidden"
            if hidden_names
            else "no_retrieval_tools_active"
        ),
    )