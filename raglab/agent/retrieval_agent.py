"""基于 Tool Calling 的知识库检索 Agent。

该 Agent 不让模型选择 BM25、Dense 或 Hybrid。

系统已经提前确定：

    search_knowledge_base
    → 内部固定使用 BM25

模型只负责决定：

1. 当前问题是否需要查询知识库；
2. 使用什么完整查询进行检索；
3. 返回多少条候选资料；
4. 第一次检索不足时，是否修改查询再次检索；
5. 何时停止工具调用并生成最终回答。

完整循环：

用户问题
→ 模型决策
→ 是否产生 tool_calls
    ├─ 是：Python 执行检索工具
    │      ↓
    │   ToolMessage 返回模型
    │      ↓
    │   模型继续决策
    │
    └─ 否：当前模型输出作为最终答案

达到最大步数后，程序将停止继续调用工具，
并要求模型根据已有工具结果生成最终答案。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from raglab.generation.rag_chain import (
    ChatModelProtocol,
    extract_answer_text,
    extract_response_metadata,
    extract_usage_metadata,
)


RETRIEVAL_AGENT_SYSTEM_PROMPT = """你是一个基于本地知识库回答问题的检索助手。

你可以使用 search_knowledge_base 工具查询本地知识库。

请严格遵守以下规则：

1. 涉及知识库中的技术概念、实现方法、实验数据、配置、步骤或结论时，必须先调用知识库检索工具。
2. 不得仅凭模型自身记忆回答知识库事实问题。
3. 用户只是打招呼、致谢或要求修改当前回答表达时，可以不调用工具。
4. 工具查询参数必须是完整、独立、适合检索的问题，不能使用“它”“这个”等模糊指代。
5. 第一次检索结果不足时，可以修改查询后再次检索。
6. 不要机械重复完全相同的工具调用。
7. 最多进行必要数量的检索，不要无意义地反复搜索。
8. 最终回答中的事实性结论必须来自工具返回的资料。
9. 关键结论应标注工具结果中的资料编号，例如 [资料1]。
10. 资料不足时，明确说明“现有检索资料不足以回答该问题”，不要使用模型自身知识补全。
11. 资料之间存在冲突时，应指出冲突。
12. 回答应直接、清晰，不要完整复述全部检索结果。"""


@dataclass(frozen=True)
class AgentToolCall:
    """记录一次工具调用。"""

    step_index: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    output: str
    latency_ms: float


@dataclass(frozen=True)
class AgentModelCall:
    """记录一次模型调用。"""

    step_index: int
    has_tool_calls: bool
    tool_call_count: int
    latency_ms: float
    usage_metadata: dict[str, Any]
    response_metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalAgentResult:
    """一次 Retrieval Agent 执行结果。"""

    question: str
    answer: str

    model_call_count: int
    tool_call_count: int
    completed_normally: bool
    stopped_by_max_steps: bool

    model_calls: list[AgentModelCall]
    tool_calls: list[AgentToolCall]
    messages: list[BaseMessage]

    total_latency_ms: float
    final_response: Any


def normalize_tool_call(
    tool_call: Any,
) -> tuple[
    str,
    str,
    dict[str, Any],
]:
    """统一 LangChain 模型返回的工具调用格式。

    Returns
    -------
    tuple
        tool_call_id
        tool_name
        arguments
    """

    if not isinstance(
        tool_call,
        dict,
    ):
        raise TypeError(
            "无法识别工具调用格式："
            f"{type(tool_call)!r}"
        )

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

    if not tool_call_id:
        raise ValueError(
            "工具调用缺少 id。"
        )

    if not tool_name:
        raise ValueError(
            "工具调用缺少 name。"
        )

    if not isinstance(
        arguments,
        dict,
    ):
        raise TypeError(
            "工具调用参数必须是字典："
            f"{arguments!r}"
        )

    return (
        tool_call_id,
        tool_name,
        dict(arguments),
    )


def build_tool_map(
    tools: Sequence[BaseTool],
) -> dict[str, BaseTool]:
    """根据工具名称建立查询表。"""

    tool_map: dict[str, BaseTool] = {}

    for current_tool in tools:
        if not isinstance(
            current_tool,
            BaseTool,
        ):
            raise TypeError(
                "tools 中只能包含 BaseTool，"
                f"实际类型：{type(current_tool)!r}"
            )

        tool_name = str(
            current_tool.name
        ).strip()

        if not tool_name:
            raise ValueError(
                "工具名称不能为空。"
            )

        if tool_name in tool_map:
            raise ValueError(
                f"工具名称重复：{tool_name}"
            )

        tool_map[tool_name] = (
            current_tool
        )

    if not tool_map:
        raise ValueError(
            "至少需要提供一个工具。"
        )

    return tool_map


def read_ai_tool_calls(
    response: Any,
) -> list[dict[str, Any]]:
    """读取 AIMessage 中的工具调用。"""

    tool_calls = getattr(
        response,
        "tool_calls",
        None,
    )

    if tool_calls is None:
        return []

    if not isinstance(
        tool_calls,
        list,
    ):
        raise TypeError(
            "模型响应中的 tool_calls "
            "必须是列表。"
        )

    return list(tool_calls)


def tool_output_to_text(
    output: Any,
) -> str:
    """将工具输出转换为普通字符串。"""

    if output is None:
        return ""

    if isinstance(
        output,
        str,
    ):
        return output

    return str(output)


class RetrievalAgent:
    """带有工具调用循环的知识库检索 Agent。"""

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
        """初始化 Retrieval Agent。

        Parameters
        ----------
        chat_model:
            支持 bind_tools() 和 invoke() 的聊天模型。

        tools:
            提供给模型的 LangChain 工具。

            当前项目只传入：
            search_knowledge_base

        max_steps:
            最大模型决策步数。

            一次模型调用算一步。
            工具调用本身不额外占用步骤。

        system_prompt:
            Agent 系统规则。
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

        normalized_system_prompt = (
            str(system_prompt).strip()
        )

        if not normalized_system_prompt:
            raise ValueError(
                "system_prompt 不能为空。"
            )

        self.chat_model = chat_model
        self.tools = list(tools)
        self.tool_map = build_tool_map(
            self.tools
        )

        self.max_steps = int(
            max_steps
        )

        self.system_prompt = (
            normalized_system_prompt
        )

        self.tool_enabled_model = (
            self.chat_model.bind_tools(
                self.tools
            )
        )

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        step_index: int,
    ) -> tuple[
        AgentToolCall,
        ToolMessage,
    ]:
        """执行一次模型产生的工具调用。"""

        (
            tool_call_id,
            tool_name,
            arguments,
        ) = normalize_tool_call(
            tool_call
        )

        selected_tool = (
            self.tool_map.get(
                tool_name
            )
        )

        if selected_tool is None:
            output_text = (
                "工具调用失败："
                f"不存在工具 {tool_name!r}。"
            )

            latency_ms = 0.0

        else:
            start_time = (
                time.perf_counter()
            )

            try:
                output = (
                    selected_tool.invoke(
                        arguments
                    )
                )

                output_text = (
                    tool_output_to_text(
                        output
                    )
                )

            except Exception as error:
                output_text = (
                    "工具执行失败："
                    f"{type(error).__name__}："
                    f"{error}"
                )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

        tool_record = AgentToolCall(
            step_index=step_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            output=output_text,
            latency_ms=latency_ms,
        )

        tool_message = ToolMessage(
            content=output_text,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

        return (
            tool_record,
            tool_message,
        )

    def run(
        self,
        question: str,
        *,
        history_messages: Sequence[
            BaseMessage
        ] | None = None,
    ) -> RetrievalAgentResult:
        """执行一次完整的 Agent 循环。

        Parameters
        ----------
        question:
            当前用户问题。

        history_messages:
            可选的历史消息。

            当前第一版可以不提供。
            后续多轮 Agent 会继续扩展该部分。
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

        messages: list[BaseMessage] = [
            SystemMessage(
                content=self.system_prompt
            ),
        ]

        if history_messages:
            messages.extend(
                list(history_messages)
            )

        messages.append(
            HumanMessage(
                content=normalized_question
            )
        )

        model_call_records: list[
            AgentModelCall
        ] = []

        tool_call_records: list[
            AgentToolCall
        ] = []

        total_start = (
            time.perf_counter()
        )

        final_response: Any | None = None
        completed_normally = False
        stopped_by_max_steps = False

        for step_index in range(
            1,
            self.max_steps + 1,
        ):
            model_start = (
                time.perf_counter()
            )

            response = (
                self.tool_enabled_model.invoke(
                    messages
                )
            )

            model_latency_ms = (
                time.perf_counter()
                - model_start
            ) * 1000.0

            if not isinstance(
                response,
                AIMessage,
            ):
                raise TypeError(
                    "模型响应必须是 AIMessage，"
                    f"实际类型：{type(response)!r}"
                )

            messages.append(response)

            current_tool_calls = (
                read_ai_tool_calls(
                    response
                )
            )

            model_call_records.append(
                AgentModelCall(
                    step_index=step_index,
                    has_tool_calls=bool(
                        current_tool_calls
                    ),
                    tool_call_count=len(
                        current_tool_calls
                    ),
                    latency_ms=(
                        model_latency_ms
                    ),
                    usage_metadata=(
                        extract_usage_metadata(
                            response
                        )
                    ),
                    response_metadata=(
                        extract_response_metadata(
                            response
                        )
                    ),
                )
            )

            if not current_tool_calls:
                final_response = response
                completed_normally = True
                break

            for current_tool_call in (
                current_tool_calls
            ):
                (
                    tool_record,
                    tool_message,
                ) = self._execute_tool_call(
                    current_tool_call,
                    step_index=step_index,
                )

                tool_call_records.append(
                    tool_record
                )

                messages.append(
                    tool_message
                )

        if final_response is None:
            stopped_by_max_steps = True

            messages.append(
                HumanMessage(
                    content=(
                        "已经达到工具调用步骤上限。"
                        "请停止继续调用工具，"
                        "仅根据已有工具结果生成最终回答。"
                        "如果资料不足，请明确说明。"
                    )
                )
            )

            final_response = (
                self.chat_model.invoke(
                    messages
                )
            )

            messages.append(
                final_response
            )

        answer_text = extract_answer_text(
            final_response
        )

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        return RetrievalAgentResult(
            question=normalized_question,
            answer=answer_text,

            model_call_count=len(
                model_call_records
            ) + (
                1
                if stopped_by_max_steps
                else 0
            ),

            tool_call_count=len(
                tool_call_records
            ),

            completed_normally=(
                completed_normally
            ),

            stopped_by_max_steps=(
                stopped_by_max_steps
            ),

            model_calls=(
                model_call_records
            ),

            tool_calls=(
                tool_call_records
            ),

            messages=messages,

            total_latency_ms=(
                total_latency_ms
            ),

            final_response=(
                final_response
            ),
        )