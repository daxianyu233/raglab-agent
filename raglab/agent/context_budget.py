"""RAGLab Token Budget Manager - Phase 6B.

核心原则：
    Selection first, Compression second.

本模块只回答：
1. 当前已经选中的 model_input 估算有多少 tokens；
2. 加上 Tool Schema / 输出预留 / 安全余量后是否超预算；
3. 哪些 Context Source 占用了多少预算。

本模块不负责：
- 删除消息；
- 压缩 Tool Result；
- 重新检索；
- 改写 ContextPlan；
- 调用 LLM。

真正的 Pruning / Compression 放到 Phase 6C。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from raglab.agent.context_assembler import (
    ContextAssemblyResult,
)
from raglab.agent.context_manager import (
    estimate_message_tokens,
)


@dataclass(frozen=True)
class ContextBudgetConfig:
    """一次模型调用的硬预算。

    model_context_limit_tokens:
        模型完整 context window。

    reserved_output_tokens:
        为模型回答预留的输出空间。
        不能把整个 context window 全部占成输入。

    tool_schema_tokens:
        bind_tools 后 Tool Schemas 也会占 provider input token，
        但它们不在 messages 列表中，所以单独计入。

    safety_margin_tokens:
        给 tokenizer 估算误差、provider wrapper、隐藏协议开销
        留出的安全余量。
    """

    model_context_limit_tokens: int
    reserved_output_tokens: int

    tool_schema_tokens: int = 0
    safety_margin_tokens: int = 0

    def __post_init__(
        self,
    ) -> None:
        integer_fields = {
            "model_context_limit_tokens": (
                self.model_context_limit_tokens
            ),
            "reserved_output_tokens": (
                self.reserved_output_tokens
            ),
            "tool_schema_tokens": (
                self.tool_schema_tokens
            ),
            "safety_margin_tokens": (
                self.safety_margin_tokens
            ),
        }

        for name, value in integer_fields.items():
            if not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{name} 必须是 int。"
                )

            if value < 0:
                raise ValueError(
                    f"{name} 不能小于 0。"
                )

        if (
            self.model_context_limit_tokens
            <= 0
        ):
            raise ValueError(
                "model_context_limit_tokens 必须大于 0。"
            )

        fixed_reservation = (
            self.reserved_output_tokens
            + self.tool_schema_tokens
            + self.safety_margin_tokens
        )

        if fixed_reservation >= (
            self.model_context_limit_tokens
        ):
            raise ValueError(
                "输出预留 + Tool Schema + 安全余量 "
                "已经占满或超过模型 Context Window。"
            )

    @property
    def available_message_tokens(
        self,
    ) -> int:
        return (
            self.model_context_limit_tokens
            - self.reserved_output_tokens
            - self.tool_schema_tokens
            - self.safety_margin_tokens
        )


@dataclass(frozen=True)
class ContextBudgetReport:
    """一次 Context Budget 评估结果。"""

    model_context_limit_tokens: int

    reserved_output_tokens: int
    tool_schema_tokens: int
    safety_margin_tokens: int

    available_message_tokens: int
    estimated_message_tokens: int

    remaining_message_tokens: int
    overflow_tokens: int

    fits: bool
    compression_required: bool

    # 总保留成本：
    # message + output reserve + schema + safety
    estimated_total_reserved_tokens: int

    context_window_utilization: float
    message_budget_utilization: float

    source_message_counts: dict[
        str,
        int,
    ]

    source_estimated_tokens: dict[
        str,
        int,
    ]

    largest_source: str | None
    largest_source_tokens: int

    message_count: int

    # Phase 6B 只给诊断，不执行策略。
    recommended_action: str


class ContextBudgetManager:
    """评估 Context 是否装得下。"""

    def evaluate(
        self,
        *,
        assembly: ContextAssemblyResult,
        config: ContextBudgetConfig,
    ) -> ContextBudgetReport:
        if len(
            assembly.messages
        ) != len(
            assembly.message_sources
        ):
            raise ValueError(
                "assembly.messages 与 message_sources "
                "数量不一致。"
            )

        source_tokens: dict[
            str,
            int,
        ] = defaultdict(
            int
        )

        source_counts: Counter[
            str
        ] = Counter()

        estimated_message_tokens = 0

        for (
            message,
            source,
        ) in zip(
            assembly.messages,
            assembly.message_sources,
        ):
            current_tokens = int(
                estimate_message_tokens(
                    message
                )
            )

            estimated_message_tokens += (
                current_tokens
            )

            source_tokens[
                source
            ] += current_tokens

            source_counts[
                source
            ] += 1

        # 与 Context Audit 应该大致完全一致，
        # 因为目前二者共用同一估算函数。
        audit_estimate = int(
            assembly.context_audit.get(
                "estimated_message_tokens",
                estimated_message_tokens,
            )
        )

        if (
            audit_estimate
            != estimated_message_tokens
        ):
            raise RuntimeError(
                "Token Budget 与 Context Audit "
                "估算结果不一致："
                f"budget={estimated_message_tokens}；"
                f"audit={audit_estimate}"
            )

        available_message_tokens = (
            config.available_message_tokens
        )

        remaining_message_tokens = (
            available_message_tokens
            - estimated_message_tokens
        )

        overflow_tokens = max(
            0,
            -remaining_message_tokens,
        )

        fits = (
            overflow_tokens
            == 0
        )

        estimated_total_reserved_tokens = (
            estimated_message_tokens
            + config.reserved_output_tokens
            + config.tool_schema_tokens
            + config.safety_margin_tokens
        )

        context_window_utilization = (
            estimated_total_reserved_tokens
            / config.model_context_limit_tokens
        )

        message_budget_utilization = (
            estimated_message_tokens
            / available_message_tokens
            if available_message_tokens > 0
            else 1.0
        )

        largest_source: (
            str
            | None
        ) = None

        largest_source_tokens = 0

        if source_tokens:
            (
                largest_source,
                largest_source_tokens,
            ) = max(
                source_tokens.items(),
                key=lambda item: item[1],
            )

        recommended_action = (
            "send_without_compression"
            if fits
            else "compression_or_pruning_required"
        )

        return ContextBudgetReport(
            model_context_limit_tokens=(
                config.model_context_limit_tokens
            ),
            reserved_output_tokens=(
                config.reserved_output_tokens
            ),
            tool_schema_tokens=(
                config.tool_schema_tokens
            ),
            safety_margin_tokens=(
                config.safety_margin_tokens
            ),
            available_message_tokens=(
                available_message_tokens
            ),
            estimated_message_tokens=(
                estimated_message_tokens
            ),
            remaining_message_tokens=(
                remaining_message_tokens
            ),
            overflow_tokens=(
                overflow_tokens
            ),
            fits=fits,
            compression_required=(
                not fits
            ),
            estimated_total_reserved_tokens=(
                estimated_total_reserved_tokens
            ),
            context_window_utilization=(
                context_window_utilization
            ),
            message_budget_utilization=(
                message_budget_utilization
            ),
            source_message_counts=dict(
                source_counts
            ),
            source_estimated_tokens=dict(
                source_tokens
            ),
            largest_source=largest_source,
            largest_source_tokens=int(
                largest_source_tokens
            ),
            message_count=len(
                assembly.messages
            ),
            recommended_action=(
                recommended_action
            ),
        )