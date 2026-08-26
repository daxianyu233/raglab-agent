"""基于 LangGraph 实现的 Retrieval Agent。

该模块将之前手写的 Agent for 循环改造成显式状态图。

图结构：

START
  ↓
agent
  ├─ 没有 tool_calls → END
  │
  └─ 有 tool_calls
          ↓
        tools
          ↓
    是否达到最大决策步数？
      ├─ 否 → agent
      └─ 是 → finalize → END

各节点职责：

agent:
    调用绑定工具后的聊天模型，
    由模型决定是否产生工具调用。

tools:
    使用 LangGraph ToolNode 执行工具，
    并把结果作为 ToolMessage 写回状态。

finalize:
    达到步骤上限后，禁止继续调用工具，
    根据已经取得的工具结果生成最终回答。

当前版本暂未加入 Checkpointer。
因此图状态只存在于一次 run() 调用期间。
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Sequence,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.prebuilt import ToolNode

from raglab.agent.retrieval_agent import (
    RETRIEVAL_AGENT_SYSTEM_PROMPT,
)
from raglab.generation.rag_chain import (
    ChatModelProtocol,
    extract_answer_text,
    extract_response_metadata,
    extract_usage_metadata,
)


class RetrievalGraphState(
    MessagesState
):
    """LangGraph Retrieval Agent 的共享状态。

    MessagesState 已经包含：

        messages

    并且会使用消息 Reducer 自动追加新消息，
    而不是覆盖已有消息。

    下面增加本项目需要的统计字段。
    """

    llm_calls: int
    tool_calls: int

    stopped_by_max_steps: bool

    model_trace: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    tool_trace: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]


@dataclass(frozen=True)
class LangGraphRetrievalResult:
    """一次 LangGraph Agent 运行结果。"""

    question: str
    answer: str

    messages: list[BaseMessage]

    llm_call_count: int
    tool_call_count: int

    stopped_by_max_steps: bool
    completed_normally: bool

    model_trace: list[
        dict[str, Any]
    ]

    tool_trace: list[
        dict[str, Any]
    ]

    total_latency_ms: float

    final_state: dict[str, Any]


def normalize_tool_calls(
    message: BaseMessage,
) -> list[dict[str, Any]]:
    """读取 AIMessage 中的工具调用。"""

    if not isinstance(
        message,
        AIMessage,
    ):
        return []

    tool_calls = getattr(
        message,
        "tool_calls",
        None,
    )

    if not tool_calls:
        return []

    if not isinstance(
        tool_calls,
        list,
    ):
        raise TypeError(
            "AIMessage.tool_calls 必须是列表，"
            f"实际类型：{type(tool_calls)!r}"
        )

    normalized: list[
        dict[str, Any]
    ] = []

    for tool_call in tool_calls:
        if not isinstance(
            tool_call,
            dict,
        ):
            raise TypeError(
                "工具调用必须是字典，"
                f"实际类型：{type(tool_call)!r}"
            )

        normalized.append(
            dict(tool_call)
        )

    return normalized


def content_to_text(
    content: Any,
) -> str:
    """将消息内容转换为可记录的文本。"""

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):
        return content

    return str(content)


def build_tool_message_map(
    messages: Sequence[BaseMessage],
) -> dict[str, ToolMessage]:
    """根据 tool_call_id 建立 ToolMessage 映射。"""

    message_map: dict[
        str,
        ToolMessage,
    ] = {}

    for message in messages:
        if not isinstance(
            message,
            ToolMessage,
        ):
            continue

        tool_call_id = str(
            message.tool_call_id
        ).strip()

        if tool_call_id:
            message_map[
                tool_call_id
            ] = message

    return message_map


class LangGraphRetrievalAgent:
    """基于 StateGraph 的知识库检索 Agent。"""

    def __init__(
        self,
        *,
        chat_model: ChatModelProtocol,
        tools: Sequence[BaseTool],
        max_steps: int = 4,
        system_prompt: str = (
            RETRIEVAL_AGENT_SYSTEM_PROMPT
        ),
    ) -> None:
        """初始化 LangGraph Retrieval Agent。

        Parameters
        ----------
        chat_model:
            支持 invoke() 和 bind_tools() 的聊天模型。

        tools:
            提供给模型的工具。

            当前项目仍然只传入：
            search_knowledge_base

        max_steps:
            最多允许多少次带工具能力的模型决策。

            如果第 max_steps 次模型调用仍产生工具调用，
            程序会执行该工具，然后进入 finalize 节点。

            finalize 会额外调用一次不绑定工具的模型，
            生成最终回答。

        system_prompt:
            Agent 系统提示。
        """

        if not hasattr(
            chat_model,
            "invoke",
        ):
            raise TypeError(
                "chat_model 必须实现 invoke()。"
            )

        if not hasattr(
            chat_model,
            "bind_tools",
        ):
            raise TypeError(
                "chat_model 必须实现 bind_tools()。"
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps 必须大于 0。"
            )

        if not tools:
            raise ValueError(
                "至少需要提供一个工具。"
            )

        normalized_tools: list[
            BaseTool
        ] = []

        tool_names: set[str] = set()

        for current_tool in tools:
            if not isinstance(
                current_tool,
                BaseTool,
            ):
                raise TypeError(
                    "tools 中只能包含 BaseTool，"
                    f"实际类型："
                    f"{type(current_tool)!r}"
                )

            tool_name = str(
                current_tool.name
            ).strip()

            if not tool_name:
                raise ValueError(
                    "工具名称不能为空。"
                )

            if tool_name in tool_names:
                raise ValueError(
                    f"工具名称重复：{tool_name}"
                )

            tool_names.add(
                tool_name
            )

            normalized_tools.append(
                current_tool
            )

        normalized_prompt = str(
            system_prompt
        ).strip()

        if not normalized_prompt:
            raise ValueError(
                "system_prompt 不能为空。"
            )

        self.chat_model = chat_model
        self.tools = normalized_tools
        self.max_steps = int(
            max_steps
        )

        self.system_prompt = (
            normalized_prompt
        )

        self.tool_enabled_model = (
            self.chat_model.bind_tools(
                self.tools
            )
        )

        # ToolNode 负责读取最后一个 AIMessage
        # 中的 tool_calls，并执行对应工具。
        self.tool_node = ToolNode(
            self.tools,
            handle_tool_errors=True,
        )

        self.graph = (
            self._build_graph()
        )

    def _model_node(
        self,
        state: RetrievalGraphState,
    ) -> dict[str, Any]:
        """模型决策节点。

        输入：
            当前完整消息状态。

        输出：
            一个新的 AIMessage；
            更新模型调用次数与轨迹。
        """

        messages = list(
            state["messages"]
        )

        model_input: list[
            BaseMessage
        ] = [
            SystemMessage(
                content=self.system_prompt
            ),
            *messages,
        ]

        start_time = (
            time.perf_counter()
        )

        response = (
            self.tool_enabled_model.invoke(
                model_input
            )
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "模型节点必须返回 AIMessage，"
                f"实际类型：{type(response)!r}"
            )

        tool_calls = (
            normalize_tool_calls(
                response
            )
        )

        current_llm_calls = int(
            state.get(
                "llm_calls",
                0,
            )
        ) + 1

        trace = {
            "node": "agent",
            "llm_call_index": (
                current_llm_calls
            ),
            "has_tool_calls": bool(
                tool_calls
            ),
            "tool_call_count": len(
                tool_calls
            ),
            "tool_calls": tool_calls,
            "latency_ms": latency_ms,
            "usage_metadata": (
                extract_usage_metadata(
                    response
                )
            ),
            "response_metadata": (
                extract_response_metadata(
                    response
                )
            ),
        }

        return {
            "messages": [response],
            "llm_calls": (
                current_llm_calls
            ),
            "model_trace": [trace],
        }

    def _tools_node(
        self,
        state: RetrievalGraphState,
    ) -> dict[str, Any]:
        """工具执行节点。

        使用 LangGraph ToolNode 执行最后一个
        AIMessage 中的全部工具调用。

        ToolNode 返回的 ToolMessage 会追加到
        messages 状态中。
        """

        messages = list(
            state["messages"]
        )

        if not messages:
            raise RuntimeError(
                "工具节点没有收到任何消息。"
            )

        last_message = messages[-1]

        tool_calls = normalize_tool_calls(
            last_message
        )

        if not tool_calls:
            raise RuntimeError(
                "进入工具节点时，"
                "最后一个 AIMessage "
                "没有 tool_calls。"
            )

        start_time = (
            time.perf_counter()
        )

        # 传入完整 state，而不仅是 messages。
        # 后续工具如果需要读取图状态，
        # 可以通过 ToolRuntime 获取。
        result = self.tool_node.invoke(
            state
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "ToolNode 返回值必须是字典，"
                f"实际类型：{type(result)!r}"
            )

        returned_messages = result.get(
            "messages",
            [],
        )

        if not isinstance(
            returned_messages,
            list,
        ):
            raise TypeError(
                "ToolNode 返回的 messages "
                "必须是列表。"
            )

        tool_message_map = (
            build_tool_message_map(
                returned_messages
            )
        )

        traces: list[
            dict[str, Any]
        ] = []

        for tool_call in tool_calls:
            tool_call_id = str(
                tool_call.get(
                    "id",
                    "",
                )
            ).strip()

            tool_name = str(
                tool_call.get(
                    "name",
                    "",
                )
            ).strip()

            arguments = tool_call.get(
                "args",
                {},
            )

            tool_message = (
                tool_message_map.get(
                    tool_call_id
                )
            )

            output = (
                content_to_text(
                    tool_message.content
                )
                if tool_message is not None
                else ""
            )

            traces.append(
                {
                    "node": "tools",
                    "tool_call_id": (
                        tool_call_id
                    ),
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "output": output,
                    # 多个工具可能由 ToolNode 并行执行，
                    # 因此这里记录的是整个 ToolNode 耗时。
                    "node_latency_ms": (
                        latency_ms
                    ),
                }
            )

        current_tool_calls = int(
            state.get(
                "tool_calls",
                0,
            )
        ) + len(tool_calls)

        return {
            "messages": (
                returned_messages
            ),
            "tool_calls": (
                current_tool_calls
            ),
            "tool_trace": traces,
        }

    def _finalize_node(
        self,
        state: RetrievalGraphState,
    ) -> dict[str, Any]:
        """达到步骤上限后的最终回答节点。

        此处使用没有绑定工具的原始模型，
        因此模型不能继续产生工具调用。
        """

        messages = list(
            state["messages"]
        )

        final_instruction = HumanMessage(
            content=(
                "已经达到本次 Agent 的工具决策"
                "步骤上限。请停止继续检索，"
                "仅根据当前已经获得的工具结果"
                "生成最终回答。"
                "如果已有资料不足，请明确说明"
                "现有检索资料不足以回答该问题。"
            )
        )

        model_input: list[
            BaseMessage
        ] = [
            SystemMessage(
                content=self.system_prompt
            ),
            *messages,
            final_instruction,
        ]

        start_time = (
            time.perf_counter()
        )

        response = self.chat_model.invoke(
            model_input
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "finalize 节点必须返回 AIMessage，"
                f"实际类型：{type(response)!r}"
            )

        current_llm_calls = int(
            state.get(
                "llm_calls",
                0,
            )
        ) + 1

        trace = {
            "node": "finalize",
            "llm_call_index": (
                current_llm_calls
            ),
            "has_tool_calls": False,
            "tool_call_count": 0,
            "latency_ms": latency_ms,
            "usage_metadata": (
                extract_usage_metadata(
                    response
                )
            ),
            "response_metadata": (
                extract_response_metadata(
                    response
                )
            ),
        }

        return {
            "messages": [response],
            "llm_calls": (
                current_llm_calls
            ),
            "stopped_by_max_steps": True,
            "model_trace": [trace],
        }

    def _route_after_model(
        self,
        state: RetrievalGraphState,
    ) -> str:
        """模型节点后的条件路由。

        有工具调用：
            前往 tools。

        没有工具调用：
            结束图执行。
        """

        messages = state["messages"]

        if not messages:
            raise RuntimeError(
                "模型路由没有收到消息。"
            )

        last_message = messages[-1]

        if normalize_tool_calls(
            last_message
        ):
            return "tools"

        return END

    def _route_after_tools(
        self,
        state: RetrievalGraphState,
    ) -> str:
        """工具节点后的条件路由。

        尚未达到步骤上限：
            回到 agent，让模型读取工具结果。

        已达到步骤上限：
            进入 finalize，禁止继续调用工具。
        """

        llm_calls = int(
            state.get(
                "llm_calls",
                0,
            )
        )

        if llm_calls >= self.max_steps:
            return "finalize"

        return "agent"

    def _build_graph(
        self,
    ) -> Any:
        """构建并编译 StateGraph。"""

        builder = StateGraph(
            RetrievalGraphState
        )

        builder.add_node(
            "agent",
            self._model_node,
        )

        builder.add_node(
            "tools",
            self._tools_node,
        )

        builder.add_node(
            "finalize",
            self._finalize_node,
        )

        builder.add_edge(
            START,
            "agent",
        )

        builder.add_conditional_edges(
            "agent",
            self._route_after_model,
            [
                "tools",
                END,
            ],
        )

        builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            [
                "agent",
                "finalize",
            ],
        )

        builder.add_edge(
            "finalize",
            END,
        )

        return builder.compile()

    def run(
        self,
        question: str,
        *,
        history_messages: Sequence[
            BaseMessage
        ] | None = None,
    ) -> LangGraphRetrievalResult:
        """执行一次完整的 LangGraph Agent。

        Parameters
        ----------
        question:
            当前用户问题。

        history_messages:
            可选的历史消息。

            当前图暂未绑定 Checkpointer，
            因此多轮历史仍由调用方显式传入。
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "question 必须是字符串。"
            )

        normalized_question = (
            question.strip()
        )

        if not normalized_question:
            raise ValueError(
                "question 不能为空。"
            )

        initial_messages: list[
            BaseMessage
        ] = []

        if history_messages:
            initial_messages.extend(
                list(history_messages)
            )

        initial_messages.append(
            HumanMessage(
                content=normalized_question
            )
        )

        initial_state: (
            RetrievalGraphState
        ) = {
            "messages": (
                initial_messages
            ),
            "llm_calls": 0,
            "tool_calls": 0,
            "stopped_by_max_steps": False,
            "model_trace": [],
            "tool_trace": [],
        }

        total_start = (
            time.perf_counter()
        )

        final_state = self.graph.invoke(
            initial_state,
            config={
                # 防止配置错误造成无限图循环。
                "recursion_limit": (
                    self.max_steps * 3
                    + 5
                )
            },
        )

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        messages = list(
            final_state.get(
                "messages",
                [],
            )
        )

        if not messages:
            raise RuntimeError(
                "LangGraph 执行结束后"
                "没有返回任何消息。"
            )

        final_message = messages[-1]

        answer = extract_answer_text(
            final_message
        ).strip()

        stopped_by_max_steps = bool(
            final_state.get(
                "stopped_by_max_steps",
                False,
            )
        )

        return LangGraphRetrievalResult(
            question=normalized_question,
            answer=answer,
            messages=messages,

            llm_call_count=int(
                final_state.get(
                    "llm_calls",
                    0,
                )
            ),

            tool_call_count=int(
                final_state.get(
                    "tool_calls",
                    0,
                )
            ),

            stopped_by_max_steps=(
                stopped_by_max_steps
            ),

            completed_normally=(
                not stopped_by_max_steps
            ),

            model_trace=list(
                final_state.get(
                    "model_trace",
                    [],
                )
            ),

            tool_trace=list(
                final_state.get(
                    "tool_trace",
                    [],
                )
            ),

            total_latency_ms=(
                total_latency_ms
            ),

            final_state=dict(
                final_state
            ),
        )