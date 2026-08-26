"""RAGLab Agent 上下文审计工具。

Phase 1 只负责“观测”，不改变模型输入，不执行消息裁剪。

目标：
1. 估算每次模型调用前的消息 Token 成本；
2. 统计 System / Human / AI / Tool 各类消息占比；
3. 识别超大的 Tool Result；
4. 检查 AIMessage(tool_calls) 与 ToolMessage 的配对完整性；
5. 为后续 Token-aware Context Manager 提供可观测数据。

注意：
- 这里的 Token 数是调用前的保守估算值，不是模型供应商的最终计费 Token；
- 实际 input token 应以后续 response.usage_metadata 为准；
- 后续可根据“估算值 vs 实际 usage”校准估算器。
"""

from __future__ import annotations

import json
import math
import re

from collections import Counter
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


_CJK_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)

_LATIN_OR_NUMBER_PATTERN = re.compile(
    r"[A-Za-z0-9_]+"
)

_NONSPACE_PATTERN = re.compile(
    r"\S"
)


def message_content_to_text(
    content: Any,
) -> str:
    """把 LangChain Message content 转成可估算文本。"""

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):
        return content

    if isinstance(
        content,
        list,
    ):
        parts: list[str] = []

        for item in content:

            if isinstance(
                item,
                str,
            ):
                parts.append(
                    item
                )
                continue

            if isinstance(
                item,
                dict,
            ):
                text = item.get(
                    "text"
                )

                if isinstance(
                    text,
                    str,
                ):
                    parts.append(
                        text
                    )
                    continue

                try:
                    parts.append(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                except TypeError:
                    parts.append(
                        str(
                            item
                        )
                    )

                continue

            parts.append(
                str(
                    item
                )
            )

        return "\n".join(
            parts
        )

    if isinstance(
        content,
        dict,
    ):
        try:
            return json.dumps(
                content,
                ensure_ascii=False,
                default=str,
            )
        except TypeError:
            return str(
                content
            )

    return str(
        content
    )


def estimate_text_tokens(
    text: str,
) -> int:
    """保守估算中英文混合文本的 Token 数。

    这里只用于 Context Budget 的前置估算，不声称与 DeepSeek
    服务端 tokenizer 完全一致。

    估算策略：
    - CJK 字符按约 1 token / 字符；
    - 英文/数字连续串按约 4 字符 / token；
    - 其他非空白符号按较小额外开销计入。
    """

    normalized = str(
        text
    )

    if not normalized:
        return 0

    cjk_count = len(
        _CJK_PATTERN.findall(
            normalized
        )
    )

    without_cjk = (
        _CJK_PATTERN.sub(
            " ",
            normalized,
        )
    )

    latin_matches = (
        _LATIN_OR_NUMBER_PATTERN.findall(
            without_cjk
        )
    )

    latin_tokens = sum(
        max(
            1,
            math.ceil(
                len(item) / 4
            ),
        )
        for item
        in latin_matches
    )

    without_latin = (
        _LATIN_OR_NUMBER_PATTERN.sub(
            " ",
            without_cjk,
        )
    )

    punctuation_count = len(
        _NONSPACE_PATTERN.findall(
            without_latin
        )
    )

    punctuation_tokens = (
        math.ceil(
            punctuation_count / 2
        )
        if punctuation_count
        else 0
    )

    return max(
        1,
        cjk_count
        + latin_tokens
        + punctuation_tokens,
    )


def _extract_ai_tool_calls(
    message: AIMessage,
) -> list[dict[str, Any]]:
    """兼容 LangChain 常见 Tool Call 表达。"""

    raw_calls = getattr(
        message,
        "tool_calls",
        None,
    )

    if isinstance(
        raw_calls,
        list,
    ):
        return [
            item
            for item
            in raw_calls
            if isinstance(
                item,
                dict,
            )
        ]

    additional_kwargs = getattr(
        message,
        "additional_kwargs",
        {},
    )

    if isinstance(
        additional_kwargs,
        dict,
    ):
        raw_calls = (
            additional_kwargs.get(
                "tool_calls"
            )
        )

        if isinstance(
            raw_calls,
            list,
        ):
            return [
                item
                for item
                in raw_calls
                if isinstance(
                    item,
                    dict,
                )
            ]

    return []


def _tool_call_id(
    tool_call: dict[str, Any],
) -> str:
    value = (
        tool_call.get(
            "id"
        )
        or tool_call.get(
            "tool_call_id"
        )
    )

    return (
        str(
            value
        ).strip()
        if value is not None
        else ""
    )


def _tool_call_payload_text(
    tool_call: dict[str, Any],
) -> str:
    """把 Tool Call 名称和参数纳入输入成本估算。"""

    try:
        return json.dumps(
            tool_call,
            ensure_ascii=False,
            default=str,
        )
    except TypeError:
        return str(
            tool_call
        )


def estimate_message_tokens(
    message: BaseMessage,
) -> int:
    """估算单条消息成本。"""

    base_overhead = 4

    content_tokens = (
        estimate_text_tokens(
            message_content_to_text(
                getattr(
                    message,
                    "content",
                    "",
                )
            )
        )
    )

    extra_tokens = 0

    if isinstance(
        message,
        AIMessage,
    ):
        for tool_call in (
            _extract_ai_tool_calls(
                message
            )
        ):
            extra_tokens += (
                estimate_text_tokens(
                    _tool_call_payload_text(
                        tool_call
                    )
                )
            )

    if isinstance(
        message,
        ToolMessage,
    ):
        name = str(
            getattr(
                message,
                "name",
                "",
            )
            or ""
        )

        call_id = str(
            getattr(
                message,
                "tool_call_id",
                "",
            )
            or ""
        )

        extra_tokens += (
            estimate_text_tokens(
                name
            )
            + estimate_text_tokens(
                call_id
            )
        )

    return (
        base_overhead
        + content_tokens
        + extra_tokens
    )


def message_role(
    message: BaseMessage,
) -> str:
    if isinstance(
        message,
        SystemMessage,
    ):
        return "system"

    if isinstance(
        message,
        HumanMessage,
    ):
        return "human"

    if isinstance(
        message,
        ToolMessage,
    ):
        return "tool"

    if isinstance(
        message,
        AIMessage,
    ):
        return "ai"

    return type(
        message
    ).__name__.lower()


def audit_model_input(
    messages: Sequence[
        BaseMessage
    ],
    *,
    oversized_tool_threshold_tokens: int = 2000,
) -> dict[str, Any]:
    """审计一次真正准备发送给模型的消息列表。

    本函数不修改任何消息。
    """

    if oversized_tool_threshold_tokens <= 0:
        raise ValueError(
            "oversized_tool_threshold_tokens "
            "必须大于 0。"
        )

    role_tokens: Counter[str] = (
        Counter()
    )

    role_counts: Counter[str] = (
        Counter()
    )

    message_details: list[
        dict[str, Any]
    ] = []

    ai_tool_call_ids: set[str] = (
        set()
    )

    tool_message_call_ids: set[str] = (
        set()
    )

    oversized_tools: list[
        dict[str, Any]
    ] = []

    total_estimated_tokens = 0
    total_characters = 0

    max_message_tokens = 0
    max_message_index: int | None = (
        None
    )

    for index, message in enumerate(
        messages
    ):
        role = message_role(
            message
        )

        content_text = (
            message_content_to_text(
                getattr(
                    message,
                    "content",
                    "",
                )
            )
        )

        estimated_tokens = (
            estimate_message_tokens(
                message
            )
        )

        character_count = len(
            content_text
        )

        total_estimated_tokens += (
            estimated_tokens
        )

        total_characters += (
            character_count
        )

        role_tokens[
            role
        ] += estimated_tokens

        role_counts[
            role
        ] += 1

        if (
            estimated_tokens
            > max_message_tokens
        ):
            max_message_tokens = (
                estimated_tokens
            )

            max_message_index = (
                index
            )

        detail: dict[
            str,
            Any,
        ] = {
            "index": index,
            "role": role,
            "estimated_tokens": (
                estimated_tokens
            ),
            "characters": (
                character_count
            ),
        }

        if isinstance(
            message,
            AIMessage,
        ):
            tool_calls = (
                _extract_ai_tool_calls(
                    message
                )
            )

            tool_call_ids = [
                _tool_call_id(
                    tool_call
                )
                for tool_call
                in tool_calls
                if _tool_call_id(
                    tool_call
                )
            ]

            ai_tool_call_ids.update(
                tool_call_ids
            )

            detail[
                "tool_call_count"
            ] = len(
                tool_calls
            )

            if tool_call_ids:
                detail[
                    "tool_call_ids"
                ] = tool_call_ids

        if isinstance(
            message,
            ToolMessage,
        ):
            tool_name = str(
                getattr(
                    message,
                    "name",
                    "",
                )
                or ""
            )

            call_id = str(
                getattr(
                    message,
                    "tool_call_id",
                    "",
                )
                or ""
            ).strip()

            detail[
                "tool_name"
            ] = tool_name

            detail[
                "tool_call_id"
            ] = call_id

            if call_id:
                tool_message_call_ids.add(
                    call_id
                )

            if (
                estimated_tokens
                >= oversized_tool_threshold_tokens
            ):
                oversized_tools.append(
                    {
                        "index": index,
                        "tool_name": (
                            tool_name
                        ),
                        "tool_call_id": (
                            call_id
                        ),
                        "estimated_tokens": (
                            estimated_tokens
                        ),
                        "characters": (
                            character_count
                        ),
                    }
                )

        message_details.append(
            detail
        )

    unresolved_tool_call_ids = sorted(
        ai_tool_call_ids
        - tool_message_call_ids
    )

    orphan_tool_message_ids = sorted(
        tool_message_call_ids
        - ai_tool_call_ids
    )

    tool_tokens = int(
        role_tokens.get(
            "tool",
            0,
        )
    )

    tool_token_ratio = (
        tool_tokens
        / total_estimated_tokens
        if total_estimated_tokens > 0
        else 0.0
    )

    return {
        "message_count": len(
            messages
        ),
        "estimated_message_tokens": (
            total_estimated_tokens
        ),
        "total_characters": (
            total_characters
        ),
        "message_counts_by_role": (
            dict(
                role_counts
            )
        ),
        "estimated_tokens_by_role": (
            dict(
                role_tokens
            )
        ),
        "tool_estimated_tokens": (
            tool_tokens
        ),
        "tool_token_ratio": round(
            tool_token_ratio,
            6,
        ),
        "oversized_tool_threshold_tokens": (
            oversized_tool_threshold_tokens
        ),
        "oversized_tool_messages": (
            oversized_tools
        ),
        "oversized_tool_message_count": (
            len(
                oversized_tools
            )
        ),
        "max_message_tokens": (
            max_message_tokens
        ),
        "max_message_index": (
            max_message_index
        ),
        "unresolved_tool_call_ids": (
            unresolved_tool_call_ids
        ),
        "unresolved_tool_call_count": (
            len(
                unresolved_tool_call_ids
            )
        ),
        "orphan_tool_message_ids": (
            orphan_tool_message_ids
        ),
        "orphan_tool_message_count": (
            len(
                orphan_tool_message_ids
            )
        ),
        "tool_pair_integrity_ok": (
            not unresolved_tool_call_ids
            and not orphan_tool_message_ids
        ),
        "message_details": (
            message_details
        ),
    }