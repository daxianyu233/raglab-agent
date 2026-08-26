"""LangChain Message -> Conversation Event Store 适配器。

职责：
1. 扫描当前 LangGraph 可见 messages；
2. 按 HumanMessage 划分 turn；
3. 将 Human / AI / Tool 原始消息幂等归档；
4. 不修改 LangGraph State；
5. 不压缩 Tool Result；
6. 不负责 Context Retrieval。

重要：
- turn_id 直接锚定当前轮 HumanMessage.id；
- AIMessage 的完整 tool_calls 保存在 payload_json；
- 某些 Message 没有 message_id 时，使用稳定派生 event_id，
  避免重复扫描产生重复记录。
"""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)


@dataclass(
    frozen=True,
)
class ConversationArchiveReport:
    """一次 messages 扫描归档的结果。"""

    thread_id: str
    user_id: str

    scanned_message_count: int
    supported_message_count: int

    inserted_event_count: int
    existing_event_count: int
    skipped_message_count: int

    turn_ids: list[str]


def _json_safe_payload(
    message: BaseMessage,
) -> dict[str, Any]:
    """尽可能完整保存 LangChain Message 结构。"""

    model_dump = getattr(
        message,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):
        try:
            payload = model_dump(
                mode="json"
            )

            if isinstance(
                payload,
                dict,
            ):
                result = dict(
                    payload
                )

                result[
                    "_message_class"
                ] = type(
                    message
                ).__name__

                return result

        except Exception:
            pass

        try:
            payload = model_dump()

            if isinstance(
                payload,
                dict,
            ):
                result = dict(
                    payload
                )

                result[
                    "_message_class"
                ] = type(
                    message
                ).__name__

                return result

        except Exception:
            pass

    message_dict = getattr(
        message,
        "__dict__",
        {},
    )

    result = (
        dict(
            message_dict
        )
        if isinstance(
            message_dict,
            dict,
        )
        else {}
    )

    result[
        "_message_class"
    ] = type(
        message
    ).__name__

    result.setdefault(
        "content",
        getattr(
            message,
            "content",
            "",
        ),
    )

    return result


def _content_to_text(
    content: Any,
) -> str:
    """将 Message content 转成可检索文本，不做长度截断。"""

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

                parts.append(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        default=str,
                    )
                )

                continue

            parts.append(
                str(
                    item
                )
            )

        return "\n".join(
            part
            for part in parts
            if part
        )

    if content is None:
        return ""

    return str(
        content
    )


def _message_role(
    message: BaseMessage,
) -> str | None:
    if isinstance(
        message,
        HumanMessage,
    ):
        return "human"

    if isinstance(
        message,
        AIMessage,
    ):
        return "assistant"

    if isinstance(
        message,
        ToolMessage,
    ):
        return "tool"

    if isinstance(
        message,
        SystemMessage,
    ):
        return "system"

    return None


def _normalize_tool_calls(
    message: AIMessage,
) -> list[dict[str, Any]]:
    raw_tool_calls = getattr(
        message,
        "tool_calls",
        None,
    )

    if not raw_tool_calls:
        return []

    results: list[
        dict[str, Any]
    ] = []

    for raw_call in raw_tool_calls:
        if isinstance(
            raw_call,
            dict,
        ):
            results.append(
                dict(
                    raw_call
                )
            )

            continue

        if hasattr(
            raw_call,
            "model_dump",
        ):
            try:
                dumped = (
                    raw_call
                    .model_dump(
                        mode="json"
                    )
                )

                if isinstance(
                    dumped,
                    dict,
                ):
                    results.append(
                        dumped
                    )

                    continue

            except Exception:
                pass

        results.append(
            {
                "raw": str(
                    raw_call
                )
            }
        )

    return results


def _tool_call_identity(
    message: BaseMessage,
) -> tuple[
    str | None,
    str | None,
    list[str],
    list[str],
]:
    """返回单值索引字段 + 完整多调用元数据。"""

    if isinstance(
        message,
        ToolMessage,
    ):
        tool_call_id = str(
            getattr(
                message,
                "tool_call_id",
                "",
            )
            or ""
        ).strip()

        tool_name = str(
            getattr(
                message,
                "name",
                "",
            )
            or ""
        ).strip()

        return (
            tool_call_id
            or None,
            tool_name
            or None,
            (
                [tool_call_id]
                if tool_call_id
                else []
            ),
            (
                [tool_name]
                if tool_name
                else []
            ),
        )

    if not isinstance(
        message,
        AIMessage,
    ):
        return (
            None,
            None,
            [],
            [],
        )

    tool_calls = (
        _normalize_tool_calls(
            message
        )
    )

    tool_call_ids: list[str] = []
    tool_names: list[str] = []

    for call in tool_calls:
        call_id = str(
            call.get(
                "id",
                call.get(
                    "tool_call_id",
                    "",
                ),
            )
            or ""
        ).strip()

        tool_name = str(
            call.get(
                "name",
                "",
            )
            or ""
        ).strip()

        if (
            not tool_name
            and isinstance(
                call.get(
                    "function"
                ),
                dict,
            )
        ):
            tool_name = str(
                call[
                    "function"
                ].get(
                    "name",
                    "",
                )
                or ""
            ).strip()

        if call_id:
            tool_call_ids.append(
                call_id
            )

        if tool_name:
            tool_names.append(
                tool_name
            )

    # 数据库顶层索引列只能放一个值。
    # 单调用时直接写入；
    # 多调用时完整列表保存在 payload / metadata。
    single_call_id = (
        tool_call_ids[0]
        if len(
            tool_call_ids
        ) == 1
        else None
    )

    single_tool_name = (
        tool_names[0]
        if len(
            tool_names
        ) == 1
        else None
    )

    return (
        single_call_id,
        single_tool_name,
        tool_call_ids,
        tool_names,
    )


def _stable_hash(
    value: Any,
) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _build_turn_id(
    *,
    thread_id: str,
    message: HumanMessage,
    human_ordinal: int,
    payload: dict[str, Any],
) -> str:
    message_id = str(
        getattr(
            message,
            "id",
            "",
        )
        or ""
    ).strip()

    if message_id:
        return (
            "turn:"
            + message_id
        )

    digest = _stable_hash(
        {
            "thread_id": thread_id,
            "human_ordinal": (
                human_ordinal
            ),
            "payload": payload,
        }
    )

    return (
        "turn:derived:"
        + digest
    )


def _build_fallback_event_id(
    *,
    thread_id: str,
    turn_id: str,
    role: str,
    ordinal_in_turn: int,
    payload: dict[str, Any],
) -> str:
    digest = _stable_hash(
        {
            "thread_id": (
                thread_id
            ),
            "turn_id": turn_id,
            "role": role,
            "ordinal_in_turn": (
                ordinal_in_turn
            ),
            "payload": payload,
        }
    )

    return (
        "derived:"
        + digest
    )


def archive_messages_to_event_store(
    *,
    store: ConversationEventStore,
    user_id: str,
    thread_id: str,
    messages: Sequence[
        BaseMessage
    ],
    include_system_messages: bool = False,
) -> ConversationArchiveReport:
    """将当前可见 messages 幂等归档到 Event Store。

    turn 划分规则：
        每个 HumanMessage 开启一个新 turn；
        后续 AI / Tool 都属于该 Human turn，
        直到出现下一个 HumanMessage。

    由于 Agent 当前给 HumanMessage 明确生成 message id，
    turn_id 可以稳定锚定 HumanMessage.id。
    """

    normalized_user_id = str(
        user_id
    ).strip()

    normalized_thread_id = str(
        thread_id
    ).strip()

    if not normalized_user_id:
        raise ValueError(
            "user_id 不能为空。"
        )

    if not normalized_thread_id:
        raise ValueError(
            "thread_id 不能为空。"
        )

    scanned_message_count = len(
        messages
    )

    supported_message_count = 0
    inserted_event_count = 0
    existing_event_count = 0
    skipped_message_count = 0

    turn_ids: list[str] = []

    current_turn_id: (
        str
        | None
    ) = None

    human_ordinal = 0
    ordinal_in_turn = 0

    for message in messages:
        role = _message_role(
            message
        )

        if role is None:
            skipped_message_count += 1
            continue

        if (
            role == "system"
            and not include_system_messages
        ):
            skipped_message_count += 1
            continue

        payload = (
            _json_safe_payload(
                message
            )
        )

        if isinstance(
            message,
            HumanMessage,
        ):
            human_ordinal += 1

            current_turn_id = (
                _build_turn_id(
                    thread_id=(
                        normalized_thread_id
                    ),
                    message=message,
                    human_ordinal=(
                        human_ordinal
                    ),
                    payload=payload,
                )
            )

            ordinal_in_turn = 0

            if (
                current_turn_id
                not in turn_ids
            ):
                turn_ids.append(
                    current_turn_id
                )

        if current_turn_id is None:
            # Conversation State 正常情况下应从 HumanMessage
            # 开始。遇到无法归属到任何用户轮次的消息时，
            # 当前阶段选择跳过，而不是伪造语义归属。
            skipped_message_count += 1
            continue

        ordinal_in_turn += 1
        supported_message_count += 1

        message_id = str(
            getattr(
                message,
                "id",
                "",
            )
            or ""
        ).strip()

        (
            tool_call_id,
            tool_name,
            tool_call_ids,
            tool_names,
        ) = _tool_call_identity(
            message
        )

        metadata = {
            "message_class": (
                type(
                    message
                ).__name__
            ),
            "ordinal_in_turn": (
                ordinal_in_turn
            ),
            "tool_call_ids": (
                tool_call_ids
            ),
            "tool_names": (
                tool_names
            ),
        }

        event_id = None

        if not message_id:
            event_id = (
                _build_fallback_event_id(
                    thread_id=(
                        normalized_thread_id
                    ),
                    turn_id=(
                        current_turn_id
                    ),
                    role=role,
                    ordinal_in_turn=(
                        ordinal_in_turn
                    ),
                    payload=payload,
                )
            )

        _, inserted = (
            store.append_event(
                user_id=(
                    normalized_user_id
                ),
                thread_id=(
                    normalized_thread_id
                ),
                turn_id=(
                    current_turn_id
                ),
                event_type="message",
                role=role,
                message_id=(
                    message_id
                    or None
                ),
                tool_call_id=(
                    tool_call_id
                ),
                tool_name=(
                    tool_name
                ),
                content_text=(
                    _content_to_text(
                        getattr(
                            message,
                            "content",
                            "",
                        )
                    )
                ),
                payload=payload,
                metadata=metadata,
                event_id=event_id,
            )
        )

        if inserted:
            inserted_event_count += 1

        else:
            existing_event_count += 1

    return ConversationArchiveReport(
        thread_id=(
            normalized_thread_id
        ),
        user_id=(
            normalized_user_id
        ),
        scanned_message_count=(
            scanned_message_count
        ),
        supported_message_count=(
            supported_message_count
        ),
        inserted_event_count=(
            inserted_event_count
        ),
        existing_event_count=(
            existing_event_count
        ),
        skipped_message_count=(
            skipped_message_count
        ),
        turn_ids=turn_ids,
    )