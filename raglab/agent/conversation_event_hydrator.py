"""Conversation Event -> LangChain Message Hydrator."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from raglab.agent.conversation_event_store import (
    ConversationEvent,
)


def _payload_dict(
    event: ConversationEvent,
) -> dict[str, Any]:
    payload = event.payload
    return dict(payload) if isinstance(payload, dict) else {}


def _content(
    event: ConversationEvent,
    payload: dict[str, Any],
) -> Any:
    return payload.get(
        "content",
        event.content_text,
    )


def _message_id(
    event: ConversationEvent,
    payload: dict[str, Any],
) -> str | None:
    value = payload.get("id") or event.message_id
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def hydrate_conversation_event(
    event: ConversationEvent,
) -> BaseMessage:
    """恢复一条 Conversation Event 为 LangChain Message。"""

    payload = _payload_dict(event)
    content = _content(event, payload)
    message_id = _message_id(event, payload)

    additional_kwargs = payload.get(
        "additional_kwargs",
        {},
    )
    if not isinstance(additional_kwargs, dict):
        additional_kwargs = {}

    response_metadata = payload.get(
        "response_metadata",
        {},
    )
    if not isinstance(response_metadata, dict):
        response_metadata = {}

    if event.role == "human":
        return HumanMessage(
            content=content,
            id=message_id,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
            name=payload.get("name"),
        )

    if event.role == "assistant":
        raw_tool_calls = payload.get(
            "tool_calls",
            [],
        )
        tool_calls = (
            list(raw_tool_calls)
            if isinstance(raw_tool_calls, list)
            else []
        )

        raw_invalid_tool_calls = payload.get(
            "invalid_tool_calls",
            [],
        )
        invalid_tool_calls = (
            list(raw_invalid_tool_calls)
            if isinstance(raw_invalid_tool_calls, list)
            else []
        )

        return AIMessage(
            content=content,
            id=message_id,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
            name=payload.get("name"),
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    if event.role == "tool":
        tool_call_id = str(
            payload.get(
                "tool_call_id",
                event.tool_call_id or "",
            )
            or ""
        ).strip()

        if not tool_call_id:
            raise ValueError(
                "Tool Event 缺少 tool_call_id，无法安全恢复 ToolMessage。"
            )

        kwargs: dict[str, Any] = {
            "content": content,
            "tool_call_id": tool_call_id,
            "id": message_id,
            "name": payload.get("name") or event.tool_name,
            "additional_kwargs": additional_kwargs,
            "response_metadata": response_metadata,
        }

        if "artifact" in payload:
            kwargs["artifact"] = payload["artifact"]

        if "status" in payload:
            kwargs["status"] = payload["status"]

        try:
            return ToolMessage(**kwargs)

        except TypeError:
            # 兼容较旧 langchain_core 版本。
            kwargs.pop("artifact", None)
            kwargs.pop("status", None)
            return ToolMessage(**kwargs)

    raise ValueError(
        "当前 Event role 不支持 Hydrate："
        f"{event.role!r}"
    )


def hydrate_conversation_events(
    events: Sequence[ConversationEvent],
) -> list[BaseMessage]:
    return [
        hydrate_conversation_event(event)
        for event in events
    ]