"""Context Pipeline Phase 7A 正式 Agent 接线回归测试。

运行：
    python -m scripts.test_context_pipeline_integration

不调用真实 DeepSeek。
验证：
1. Planner 在一个 Human Turn 中只执行一次；
2. Event Store 上一轮历史进入第一次 Main LLM；
3. Tool 返回后重新 Assembler/Budget，但不重新 Planner；
4. 第二次 Main LLM 能看到当前 ToolMessage；
5. ContextPlan / Retrieval 保存在 Graph State；
6. Context Pipeline Trace 可观测。
"""

from __future__ import annotations

import tempfile

from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from raglab.agent.context_plan import (
    ContextPlan,
)
from raglab.agent.context_planner import (
    ContextPlannerResult,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.long_term_memory_agent import (
    LongTermMemoryRetrievalAgent,
)


@tool
def echo_context_test(
    value: str,
) -> str:
    """Phase 7A 接线测试 Tool。"""

    return (
        "TOOL_EVIDENCE:"
        + value
    )


class FakePlanner:
    def __init__(
        self,
    ) -> None:
        self.call_count = 0
        self.last_navigation = None

    def plan(
        self,
        navigation_context: Any,
    ) -> ContextPlannerResult:
        self.call_count += 1
        self.last_navigation = (
            navigation_context
        )

        return ContextPlannerResult(
            plan=ContextPlan(
                task_intent=(
                    "continue_with_history_and_tool"
                ),
                response_goal=(
                    "结合上一轮回答和当前 Tool 继续回答"
                ),
                history_required=True,
                history_scope=(
                    "previous_turn"
                ),
                history_query=None,
                previous_answer_required=True,
                raw_tool_evidence_required=False,
                external_retrieval_required=True,
                external_retrieval_allowed=True,
                long_term_memory_required=False,
                long_term_memory_query=None,
                referenced_entities=[],
                temporal_scope=None,
                confidence=1.0,
            ),
            latency_ms=1.0,
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
            },
            navigation_characters=100,
            raw_model_output=(
                '{"history_scope":"previous_turn"}'
            ),
        )


class FakeChatModel:
    """同时充当 bind_tools 后模型和普通 finalize 模型。"""

    def __init__(
        self,
    ) -> None:
        self.calls: list[
            list[
                BaseMessage
            ]
        ] = []

        self.bound_tool_names: list[
            str
        ] = []

    def bind_tools(
        self,
        tools: Any,
    ) -> "FakeChatModel":
        self.bound_tool_names = [
            str(
                current_tool.name
            )
            for current_tool
            in tools
        ]

        return self

    def invoke(
        self,
        messages: Any,
    ) -> AIMessage:
        current_messages = list(
            messages
        )

        self.calls.append(
            current_messages
        )

        # 第一次 Main LLM：调用 Tool。
        if len(
            self.calls
        ) == 1:
            return AIMessage(
                content="",
                id="ai-current-tool-call",
                tool_calls=[
                    {
                        "id": (
                            "call-current-001"
                        ),
                        "name": (
                            "echo_context_test"
                        ),
                        "args": {
                            "value": "phase7a"
                        },
                        "type": "tool_call",
                    }
                ],
            )

        # 第二次 Main LLM：Tool 已经返回。
        return AIMessage(
            content=(
                "FINAL_FROM_CONTEXT_PIPELINE"
            ),
            id="ai-current-final",
        )


def build_previous_turn(
    store: ConversationEventStore,
    *,
    thread_id: str,
) -> None:
    turn_id = "turn:history-001"

    store.append_event(
        user_id="local-user",
        thread_id=thread_id,
        turn_id=turn_id,
        event_type="message",
        role="human",
        message_id="human-history-001",
        content_text=(
            "上一轮用户问题：解释 Context Planner。"
        ),
        payload={
            "type": "human",
            "content": (
                "上一轮用户问题：解释 Context Planner。"
            ),
            "id": "human-history-001",
        },
    )

    store.append_event(
        user_id="local-user",
        thread_id=thread_id,
        turn_id=turn_id,
        event_type="message",
        role="assistant",
        message_id="ai-history-001",
        content_text=(
            "PREVIOUS_ANSWER_FROM_EVENT_STORE"
        ),
        payload={
            "type": "ai",
            "content": (
                "PREVIOUS_ANSWER_FROM_EVENT_STORE"
            ),
            "id": "ai-history-001",
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    )


def content_texts(
    messages: list[
        BaseMessage
    ],
) -> list[str]:
    return [
        str(
            getattr(
                message,
                "content",
                "",
            )
            or ""
        )
        for message in messages
    ]


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        event_store = (
            ConversationEventStore(
                Path(temp_dir)
                / "conversation_events.sqlite3"
            )
        )

        thread_id = (
            "thread-context-pipeline-integration"
        )

        build_previous_turn(
            event_store,
            thread_id=thread_id,
        )

        planner = FakePlanner()
        model = FakeChatModel()

        agent = (
            LongTermMemoryRetrievalAgent(
                chat_model=model,
                tools=[
                    echo_context_test
                ],
                max_steps=4,
                keep_recent_turns=2,
                summarize_trigger_turns=5,
                conversation_event_store=(
                    event_store
                ),
                context_planner=planner,
                context_pipeline_enabled=True,
                context_window_tokens=4096,
                reserved_output_tokens=512,
                context_safety_margin_tokens=128,
            )
        )

        print(
            "=" * 80
        )
        print(
            "1. 运行完整 Agent：Planner -> History -> Main LLM -> Tool -> Main LLM"
        )
        print(
            "=" * 80
        )

        result = agent.run(
            "继续上一轮，并调用测试 Tool。",
            thread_id=thread_id,
            user_id="local-user",
        )

        assert (
            result.answer
            == "FINAL_FROM_CONTEXT_PIPELINE"
        )

        assert (
            planner.call_count
            == 1
        )

        assert len(
            model.calls
        ) == 2

        print(
            "Planner 调用次数：",
            planner.call_count,
        )

        print(
            "Main LLM 调用次数：",
            len(
                model.calls
            ),
        )

        print(
            "[PASS] 一个 Human Turn 只调用一次 Planner"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "2. 第一次 Main LLM 已拿到 Event Store 上一轮"
        )
        print(
            "=" * 80
        )

        first_texts = content_texts(
            model.calls[0]
        )

        assert any(
            (
                "PREVIOUS_ANSWER_FROM_EVENT_STORE"
                in text
            )
            for text in first_texts
        )

        assert any(
            (
                "继续上一轮"
                in text
            )
            for text in first_texts
        )

        print(
            "[PASS] Retrieved Conversation History 已进入正式 model_input"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "3. Tool 返回后重新组装 Context"
        )
        print(
            "=" * 80
        )

        second_call = model.calls[1]

        tool_messages = [
            message
            for message
            in second_call
            if isinstance(
                message,
                ToolMessage,
            )
        ]

        assert len(
            tool_messages
        ) == 1

        assert (
            "TOOL_EVIDENCE:phase7a"
            in str(
                tool_messages[0]
                .content
            )
        )

        assert (
            tool_messages[0]
            .tool_call_id
            == "call-current-001"
        )

        print(
            "[PASS] 第二次 Main LLM 看到了当前 Tool Result，Tool Pair 完整"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "4. ContextPlan / Retrieval 已写入 Checkpoint State"
        )
        print(
            "=" * 80
        )

        state = (
            agent.get_thread_state(
                thread_id
            )
        )

        assert (
            state.get(
                "context_pipeline_enabled"
            )
            is True
        )

        assert (
            state[
                "context_plan"
            ][
                "history_scope"
            ]
            == "previous_turn"
        )

        assert (
            state[
                "context_retrieval"
            ][
                "selected_turn_ids"
            ]
            == [
                "turn:history-001"
            ]
        )

        print(
            "[PASS] Tool Loop / HITL Resume 可从 State 恢复同一份 Context Plan"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "5. Context Pipeline Trace"
        )
        print(
            "=" * 80
        )

        assert len(
            result.model_trace
        ) == 2

        for trace in (
            result.model_trace
        ):
            pipeline_trace = (
                trace.get(
                    "context_pipeline",
                    {},
                )
            )

            assert (
                pipeline_trace.get(
                    "enabled"
                )
                is True
            )

            assert (
                pipeline_trace[
                    "budget"
                ][
                    "fits"
                ]
                is True
            )

            assert (
                pipeline_trace.get(
                    "tool_pair_integrity_ok"
                )
                is True
            )

        print(
            "[PASS] 每次 Main LLM 调用都有 Context/Budget/Compression 可观测 Trace"
        )

        print()
        print(
            "=" * 80
        )
        print(
            "Context Pipeline Phase 7A 正式 Agent 接线回归测试通过"
        )
        print(
            "=" * 80
        )

        event_store.close()


if __name__ == "__main__":
    main()