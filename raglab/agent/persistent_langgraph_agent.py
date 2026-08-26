"""带短期持久状态、滚动摘要、动态 Skill 和节点重试的 LangGraph Retrieval Agent。

本模块负责：

1. 使用 Checkpointer 按 thread_id 保存会话状态；
2. 保留最近若干轮详细消息，并将更早消息压缩为滚动摘要；
3. 让模型决定是否调用检索工具；
4. 限制单轮工具调用次数，避免无限循环；
5. 对模型、工具和摘要节点的暂时性错误执行分类重试；
6. 保留模型调用、工具调用和摘要调用的本轮 Trace；
7. 支持 SkillRuntime 按需加载 Skill；
8. Skill 加载后动态刷新 LLM Tool Schema 和 ToolNode。

动态 Skill 的核心流程：

    Agent
      ↓
    基础 Tools
      ↓
    load_skill
      ↓
    SkillRuntime 状态变化
      ↓
    下一次 Agent 节点重新获取 Active Tools
      ↓
    bind_tools(active_tools)
      ↓
    Skill 专属 Tool 进入模型候选
      ↓
    ToolNode(active_tools)
      ↓
    执行 Skill Tool

错误处理边界：

- 工具参数调用错误：
  由 ToolNode 转换为 ToolMessage，让模型修正；

- 暂时性网络、限流和服务端错误：
  由 RetryPolicy 重试失败节点；

- ValueError、TypeError、NameError 等程序或参数错误：
  不机械重试；

- 最终仍未恢复的异常：
  向 graph.invoke() 外抛出，由控制台或 Web 层记录和提示。
"""

from __future__ import annotations

import time
import uuid

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
    RemoveMessage,
)
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from raglab.agent.context_manager import (
    audit_model_input,
)
from raglab.agent.working_memory import (
    WorkingMemoryAuditConfig,
    audit_working_memory,
    plan_working_memory_compaction,
)
from raglab.agent.langgraph_retrieval_agent import (
    normalize_tool_calls,
)
from raglab.agent.retrieval_agent import (
    RETRIEVAL_AGENT_SYSTEM_PROMPT,
)
from raglab.agent.skill_runtime import (
    SkillRuntime,
)
from raglab.generation.rag_chain import (
    ChatModelProtocol,
    extract_answer_text,
    extract_response_metadata,
    extract_usage_metadata,
)


# ============================================================
# 滚动摘要 Prompt
# ============================================================


ROLLING_SUMMARY_SYSTEM_PROMPT = """你负责维护当前聊天线程的滚动摘要。

摘要只用于帮助后续模型理解这个线程更早的对话，
不是知识库证据，也不是用户跨线程长期记忆。

请遵守以下要求：

1. 优先保留当前线程的目标、用户已明确提供的事实、
   已经确定的方案、完成进度和仍待处理的问题；
2. 保留会影响后续回答的约束、术语定义和关键结论；
3. 不要把临时示例、一次性格式要求或演示参数
   写成长期固定方案；
4. 不要保存完整工具输出、检索分数、Chunk 原文、
   tool_call_id 或临时引用编号；
5. 不要重复同一信息；
6. 不要凭空推断“没有待解决事项”；
   只有用户明确结束相关事项时才可写已完成；
7. 使用简洁中文，内容应可直接替换已有摘要，
   而不是只描述本次新增内容。

建议使用以下结构；
某一部分没有可靠内容时可以省略：

【当前目标】
【已确定方案】
【关键事实与约束】
【已完成进度】
【待继续事项】
"""


# ============================================================
# LangGraph State
# ============================================================


class PersistentRetrievalGraphState(
    MessagesState
):
    """持久化 Retrieval Agent 的图状态。"""

    summary: str
    total_summarized_turns: int

    turn_llm_calls: int
    turn_tool_calls: int
    turn_summary_calls: int

    summary_updated: bool
    summarized_turns_this_run: int
    stopped_by_max_steps: bool

    model_trace: list[
        dict[
            str,
            Any,
        ]
    ]

    tool_trace: list[
        dict[
            str,
            Any,
        ]
    ]

    # --------------------------------------------------------
    # Context Pipeline Phase 7A
    # --------------------------------------------------------
    #
    # 这些字段只保存“本轮” Context Plan / Retrieval，
    # 让 Planner 只运行一次，并能跨 Tool Loop / HITL Resume
    # 在 Checkpoint 中恢复。
    context_pipeline_enabled: bool
    context_current_human_message_id: str
    context_current_turn_id: str

    context_plan: dict[
        str,
        Any,
    ]

    context_retrieval: dict[
        str,
        Any,
    ]

    context_planner_trace: dict[
        str,
        Any,
    ]

    # 最近一次模型节点实际使用的完整 Context Pipeline Trace。
    #
    # 这是一个普通的 last-value channel：每次 agent/finalize
    # 节点执行后覆盖为该次模型输入对应的 trace。必须显式声明
    # 在 State schema 中，否则 LangGraph 会丢弃节点返回值里的
    # schema 外字段，导致 graph.invoke() 的 final_state 无法观测。
    context_pipeline: dict[
        str,
        Any,
    ]

    context_pipeline_fallback_reason: str

    # Working Memory Phase 7E-1:
    # 仅保存审计结果，不改变 Checkpoint。
    working_memory_audit: dict[
        str,
        Any,
    ]

    # 当前 thread id 只用于 Working Memory
    # 与 Conversation Event Store 做归档完整性校验。
    working_memory_thread_id: str


# ============================================================
# Agent Result
# ============================================================


@dataclass(
    frozen=True,
)
class PersistentLangGraphResult:
    """一次持久化 LangGraph Agent 调用的结果。"""

    thread_id: str

    question: str

    answer: str

    messages: list[
        BaseMessage
    ]

    summary: str

    turn_llm_call_count: int

    turn_tool_call_count: int

    turn_summary_call_count: int

    summary_updated: bool

    summarized_turns_this_run: int

    total_summarized_turns: int

    recent_turn_count: int

    stopped_by_max_steps: bool

    completed_normally: bool

    model_trace: list[
        dict[
            str,
            Any,
        ]
    ]

    tool_trace: list[
        dict[
            str,
            Any,
        ]
    ]

    total_message_count: int

    total_latency_ms: float

    final_state: dict[
        str,
        Any,
    ]


# ============================================================
# 通用辅助函数
# ============================================================


def normalize_thread_id(
    thread_id: str,
) -> str:
    """检查并规范化 thread_id。"""

    normalized = str(
        thread_id
    ).strip()

    if not normalized:
        raise ValueError(
            "thread_id 不能为空。"
        )

    return normalized


def count_human_turns(
    messages: Sequence[
        BaseMessage
    ],
) -> int:
    """统计消息列表中的用户轮数。"""

    return sum(
        isinstance(
            message,
            HumanMessage,
        )
        for message
        in messages
    )


def _message_content_to_text(
    content: Any,
) -> str:
    """将 LangChain 消息内容转换为可预览文本。"""

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        list,
    ):
        parts: list[
            str
        ] = []

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

                item_type = str(
                    item.get(
                        "type",
                        "content",
                    )
                )

                parts.append(
                    f"[{item_type}]"
                )

                continue

            parts.append(
                str(
                    item
                )
            )

        return "\n".join(
            part
            for part
            in parts
            if part
        ).strip()

    return str(
        content
    ).strip()


def _truncate_text(
    text: str,
    maximum_characters: int = 800,
) -> str:
    """限制 Trace 和摘要输入中的单段文本长度。"""

    normalized = str(
        text
    ).strip()

    if (
        len(
            normalized
        )
        <= maximum_characters
    ):
        return normalized

    return (
        normalized[
            :maximum_characters
        ]
        + "……"
    )


def _tool_call_name(
    tool_call: dict[
        str,
        Any,
    ],
) -> str:
    """兼容不同工具调用结构并取得工具名。"""

    name = tool_call.get(
        "name"
    )

    if (
        isinstance(
            name,
            str,
        )
        and name.strip()
    ):
        return name.strip()

    function = tool_call.get(
        "function"
    )

    if isinstance(
        function,
        dict,
    ):
        function_name = (
            function.get(
                "name"
            )
        )

        if isinstance(
            function_name,
            str,
        ):
            return (
                function_name
                .strip()
            )

    return "unknown_tool"


def _tool_call_args(
    tool_call: dict[
        str,
        Any,
    ],
) -> Any:
    """兼容不同工具调用结构并取得参数。"""

    if "args" in tool_call:
        return tool_call.get(
            "args"
        )

    function = tool_call.get(
        "function"
    )

    if isinstance(
        function,
        dict,
    ):
        return function.get(
            "arguments"
        )

    return None


def _tool_call_id(
    tool_call: dict[
        str,
        Any,
    ],
) -> str:
    """兼容不同工具调用结构并取得调用 ID。"""

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


def _find_latest_ai_message(
    messages: Sequence[
        BaseMessage
    ],
) -> AIMessage:
    """取得消息列表中最后一个 AIMessage。"""

    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            AIMessage,
        ):
            return message

    raise RuntimeError(
        "当前状态中没有可供工具节点执行的 "
        "AIMessage。"
    )


def _find_tool_message_by_call_id(
    messages: Sequence[
        BaseMessage
    ],
    tool_call_id: str,
) -> ToolMessage | None:
    """按 tool_call_id 找到 ToolMessage。"""

    for message in messages:

        if not isinstance(
            message,
            ToolMessage,
        ):
            continue

        if (
            str(
                getattr(
                    message,
                    "tool_call_id",
                    "",
                )
            )
            == tool_call_id
        ):
            return message

    return None


# ============================================================
# 错误分类与 RetryPolicy
# ============================================================


def is_transient_error(
    error: Exception,
) -> bool:
    """判断异常是否可能通过再次执行而恢复。

    会沿着 __cause__ 和 __context__
    检查被包装的底层异常。

    重试示例：

    - 连接失败；
    - 请求超时；
    - API 429 限流；
    - HTTP 5xx 服务端错误。

    不重试示例：

    - ValueError；
    - TypeError；
    - NameError；
    - SyntaxError；
    - 没有底层暂时异常作为 cause 的
      普通 RuntimeError。
    """

    retryable_type_names = {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
    }

    retryable_status_codes = {
        408,
        409,
        425,
        429,
    }

    non_retryable_programming_errors = (
        ValueError,
        TypeError,
        ArithmeticError,
        ImportError,
        LookupError,
        NameError,
        SyntaxError,
        ReferenceError,
    )

    current_error: (
        BaseException
        | None
    ) = error

    visited_error_ids: set[
        int
    ] = set()

    while (
        current_error
        is not None
    ):

        current_id = id(
            current_error
        )

        if (
            current_id
            in visited_error_ids
        ):
            break

        visited_error_ids.add(
            current_id
        )

        if isinstance(
            current_error,
            non_retryable_programming_errors,
        ):
            # 如果外层错误是程序错误，
            # 但其 cause 是连接异常，
            # 仍继续检查 cause。

            next_error = (
                current_error.__cause__
                or current_error.__context__
            )

            if next_error is None:
                return False

            current_error = (
                next_error
            )

            continue

        if isinstance(
            current_error,
            (
                TimeoutError,
                ConnectionError,
            ),
        ):
            return True

        error_type_name = (
            type(
                current_error
            ).__name__
        )

        if (
            error_type_name
            in retryable_type_names
        ):
            return True

        status_code = getattr(
            current_error,
            "status_code",
            None,
        )

        if isinstance(
            status_code,
            int,
        ):
            if (
                status_code
                in retryable_status_codes
                or status_code
                >= 500
            ):
                return True

        response = getattr(
            current_error,
            "response",
            None,
        )

        response_status_code = (
            getattr(
                response,
                "status_code",
                None,
            )
        )

        if isinstance(
            response_status_code,
            int,
        ):
            if (
                response_status_code
                in retryable_status_codes
                or response_status_code
                >= 500
            ):
                return True

        current_error = (
            current_error.__cause__
            or current_error.__context__
        )

    return False


MODEL_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=4.0,
    jitter=True,
    retry_on=is_transient_error,
)


TOOL_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=2.0,
    jitter=True,
    retry_on=is_transient_error,
)


SUMMARY_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=2.0,
    jitter=True,
    retry_on=is_transient_error,
)


# ============================================================
# Agent 主类
# ============================================================


class PersistentLangGraphRetrievalAgent:
    """
    带短期持久状态、滚动摘要、
    分类重试和动态 Skill Runtime 的 Retrieval Agent。
    """

    def __init__(
        self,
        *,
        chat_model: ChatModelProtocol,
        tools: Sequence[
            BaseTool
        ],
        max_steps: int = 4,
        system_prompt: str = (
            RETRIEVAL_AGENT_SYSTEM_PROMPT
        ),
        keep_recent_turns: int = 4,
        summarize_trigger_turns: int = 7,
        checkpointer: Any | None = None,
        skill_runtime: (
            SkillRuntime
            | None
        ) = None,
    ) -> None:
        """
        初始化 Agent。

        tools：
            Agent 启动时的基础 Tool。

        skill_runtime：
            可选的 SkillRuntime。

            如果存在，则每次 agent/tools 节点执行前
            都重新读取：

                基础 Tools
                +
                已加载 Skill 的 Active Tools

            从而实现 Skill 动态加载。
        """

        if max_steps <= 0:
            raise ValueError(
                "max_steps 必须大于 0。"
            )

        if keep_recent_turns <= 0:
            raise ValueError(
                "keep_recent_turns 必须大于 0。"
            )

        if (
            summarize_trigger_turns
            <= keep_recent_turns
        ):
            raise ValueError(
                "summarize_trigger_turns "
                "必须大于 keep_recent_turns。"
            )

        if not tools:
            raise ValueError(
                "tools 不能为空。"
            )

        normalized_system_prompt = str(
            system_prompt
        ).strip()

        if not normalized_system_prompt:
            raise ValueError(
                "system_prompt 不能为空。"
            )

        self.chat_model = (
            chat_model
        )

        # 注意：
        #
        # self.tools 只保存“基础 Tools”。
        #
        # Skill 专属 Tools 不永久追加到这里，
        # 而是在每次模型 / Tool 节点执行前
        # 从 SkillRuntime 动态取得。
        self.tools = list(
            tools
        )

        self.skill_runtime = (
            skill_runtime
        )

        self.max_steps = int(
            max_steps
        )

        self.system_prompt = (
            normalized_system_prompt
        )

        self.keep_recent_turns = int(
            keep_recent_turns
        )

        self.summarize_trigger_turns = int(
            summarize_trigger_turns
        )

        # ----------------------------------------------------
        # Working Memory Phase 7E-2
        # ----------------------------------------------------
        #
        # keep_recent_turns / summarize_trigger_turns
        # 仍保留为构造参数，仅用于兼容旧 config 与对照 Trace。
        #
        # 真正的 Working Memory Compaction 已改为 Token-aware：
        #
        # >= 12000 tokens
        #     -> 开始尝试整理
        #
        # target ~= 8000 tokens
        #     -> 一旦触发，尽量整理到该目标附近
        #
        # minimum_recent_turns=1
        #     -> 至少保护当前 Turn。
        self.working_memory_audit_config = (
            WorkingMemoryAuditConfig(
                soft_limit_tokens=12000,
                target_tokens=8000,
                oversized_tool_threshold_tokens=4000,
                minimum_recent_turns=1,
            )
        )

        # ----------------------------------------------------
        # 检查模型是否支持 bind_tools()
        # ----------------------------------------------------

        bind_tools = getattr(
            chat_model,
            "bind_tools",
            None,
        )

        if not callable(
            bind_tools
        ):
            raise TypeError(
                "chat_model 必须支持 "
                "bind_tools(tools)。"
            )

        self._bind_tools = (
            bind_tools
        )

        # ----------------------------------------------------
        # 这里仍保留这两个属性，
        # 主要兼容项目中其他可能访问它们的代码。
        #
        # 但它们现在不再是“初始化后永远不变”。
        #
        # 每次进入 agent/tools 节点时，
        # _refresh_tool_bindings()
        # 都会重新生成。
        # ----------------------------------------------------

        self.tool_enabled_model: Any = None

        self.tool_node: (
            ToolNode
            | None
        ) = None

        self.active_tools: list[
            BaseTool
        ] = []

        # 初始化一次基础 Tool Binding。
        self._refresh_tool_bindings()

        self.checkpointer = (
            checkpointer
            if checkpointer is not None
            else InMemorySaver()
        )

        # 子类可以覆盖 _build_graph()，
        # 例如加入 Store 或 Runtime Context。
        self.graph = (
            self._build_graph()
        )

    # ========================================================
    # 动态 Tool Runtime
    # ========================================================

    def _get_active_tools(
        self,
    ) -> list[
        BaseTool
    ]:
        """
        返回当前真正可以给 LLM / ToolNode 使用的 Tool。

        组成：

            基础 Tools
                +
            已加载 Skill Tools

        未加载 Skill 的 Tool
        不会出现在结果中。
        """

        active_tools: list[
            BaseTool
        ] = list(
            self.tools
        )

        if (
            self.skill_runtime
            is not None
        ):
            active_tools.extend(
                self.skill_runtime
                .get_active_skill_tools()
            )

        # ----------------------------------------------------
        # 检查类型和名称重复
        # ----------------------------------------------------

        seen_names: set[
            str
        ] = set()

        for current_tool in (
            active_tools
        ):

            if not isinstance(
                current_tool,
                BaseTool,
            ):
                raise TypeError(
                    "Active Tools 中只能包含 "
                    "BaseTool："
                    f"{type(current_tool)!r}"
                )

            tool_name = str(
                current_tool.name
            ).strip()

            if not tool_name:
                raise ValueError(
                    "Active Tool 名称不能为空。"
                )

            if (
                tool_name
                in seen_names
            ):
                raise ValueError(
                    "Active Tools 中存在重名 Tool："
                    f"{tool_name}"
                )

            seen_names.add(
                tool_name
            )

        return active_tools

    def _refresh_tool_bindings(
        self,
    ) -> list[
        BaseTool
    ]:
        """
        根据当前 SkillRuntime 状态刷新：

            bind_tools(active_tools)

        和：

            ToolNode(active_tools)

        这是动态 Skill 能够生效的关键。

        例如：

        第一次：

            active_tools =
                search_knowledge_base
                search_github_intelligence
                list_skills
                load_skill

        load_skill 执行后：

        第二次：

            active_tools =
                search_knowledge_base
                search_github_intelligence
                list_skills
                load_skill
                update_github_intelligence
        """

        active_tools = (
            self._get_active_tools()
        )

        # 每次都重新 bind。
        #
        # 这样 SkillRuntime 状态变化后，
        # 下一次 agent 节点一定能看到新 Tool。
        self.tool_enabled_model = (
            self._bind_tools(
                active_tools
            )
        )

        # ToolNode 同样根据最新 Tool 集合创建。
        self.tool_node = ToolNode(
            active_tools
        )

        self.active_tools = list(
            active_tools
        )

        return list(
            active_tools
        )

    def get_active_tools(
        self,
    ) -> list[
        BaseTool
    ]:
        """
        对外返回当前 Active Tools。

        主要用于日志、CLI 和测试。
        """

        return list(
            self._get_active_tools()
        )

    def get_active_tool_names(
        self,
    ) -> list[
        str
    ]:
        """
        对外返回当前 Active Tool 名称。
        """

        return [
            str(
                current_tool.name
            )
            for current_tool
            in self._get_active_tools()
        ]

    def get_loaded_skill_ids(
        self,
    ) -> list[
        str
    ]:
        """
        返回当前已加载 Skill id。

        没有 SkillRuntime 时返回空列表。
        """

        if (
            self.skill_runtime
            is None
        ):
            return []

        return list(
            self.skill_runtime
            .loaded_skill_ids()
        )

    # ========================================================
    # 图构建
    # ========================================================

    def _build_graph(
        self,
    ) -> Any:
        """
        构建带分类重试策略的 LangGraph。
        """

        builder = StateGraph(
            PersistentRetrievalGraphState
        )

        builder.add_node(
            "agent",
            self._model_node,
            retry_policy=(
                MODEL_RETRY_POLICY
            ),
        )

        builder.add_node(
            "tools",
            self._tools_node,
            retry_policy=(
                TOOL_RETRY_POLICY
            ),
        )

        builder.add_node(
            "finalize",
            self._finalize_node,
            retry_policy=(
                MODEL_RETRY_POLICY
            ),
        )

        builder.add_node(
            "memory_manager",
            self._memory_manager_node,
            retry_policy=(
                SUMMARY_RETRY_POLICY
            ),
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
                "memory_manager",
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
            "memory_manager",
        )

        builder.add_edge(
            "memory_manager",
            END,
        )

        return builder.compile(
            checkpointer=(
                self.checkpointer
            )
        )

    # ========================================================
    # 模型输入
    # ========================================================

    def _build_model_input(
        self,
        state: (
            PersistentRetrievalGraphState
        ),
    ) -> list[
        BaseMessage
    ]:
        """
        构造：

        1. 基础 System Prompt；
        2. Skill Runtime Prompt；
        3. 滚动摘要；
        4. 最近详细消息。

        Skill Runtime Prompt 每次动态生成。

        因此 Skill 被 load_skill 加载之后，
        下一次 agent 节点就能看到完整
        Skill Instructions。
        """

        model_input: list[
            BaseMessage
        ] = [
            SystemMessage(
                content=(
                    self.system_prompt
                )
            )
        ]

        # ----------------------------------------------------
        # 动态 Skill Runtime Prompt
        # ----------------------------------------------------

        if (
            self.skill_runtime
            is not None
        ):
            runtime_prompt = (
                self.skill_runtime
                .render_runtime_prompt()
                .strip()
            )

            if runtime_prompt:
                model_input.append(
                    SystemMessage(
                        content=(
                            runtime_prompt
                        )
                    )
                )

        # ----------------------------------------------------
        # 滚动摘要
        # ----------------------------------------------------

        summary = str(
            state.get(
                "summary",
                "",
            )
        ).strip()

        if summary:
            model_input.append(
                SystemMessage(
                    content=(
                        "以下是当前聊天线程中"
                        "更早对话的滚动摘要。"
                        "它只用于理解本线程历史，"
                        "不能替代本轮知识库检索证据："
                        "\n\n"
                        f"{summary}"
                    )
                )
            )

        # ----------------------------------------------------
        # 当前保留的详细消息
        # ----------------------------------------------------

        model_input.extend(
            list(
                state.get(
                    "messages",
                    [],
                )
            )
        )

        return model_input

    # ========================================================
    # Agent 模型节点
    # ========================================================

    def _model_node(
        self,
        state: (
            PersistentRetrievalGraphState
        ),
    ) -> dict[
        str,
        Any,
    ]:
        """
        调用支持 Tool Calling 的模型。

        每次进入本节点时都执行：

            _refresh_tool_bindings()

        所以刚刚通过 load_skill
        加载的新 Skill Tool
        会在这里立即生效。
        """

        active_tools = (
            self._refresh_tool_bindings()
        )

        active_tool_names = [
            str(
                current_tool.name
            )
            for current_tool
            in active_tools
        ]

        model_input = (
            self._build_model_input(
                state
            )
        )

        # Phase 1 Context Audit：
        # 只观测实际模型输入，不修改 messages。
        context_audit = (
            audit_model_input(
                model_input
            )
        )

        start_time = (
            time.perf_counter()
        )

        if (
            self.tool_enabled_model
            is None
        ):
            raise RuntimeError(
                "Tool Enabled Model "
                "尚未初始化。"
            )

        response = (
            self.tool_enabled_model
            .invoke(
                model_input
            )
        )

        latency_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000.0
        )

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "agent 节点必须返回 AIMessage，"
                "实际类型："
                f"{type(response)!r}"
            )

        tool_calls = (
            normalize_tool_calls(
                response
            )
        )

        current_llm_calls = (
            int(
                state.get(
                    "turn_llm_calls",
                    0,
                )
            )
            + 1
        )

        model_trace = list(
            state.get(
                "model_trace",
                [],
            )
        )

        model_trace.append(
            {
                "node": (
                    "agent"
                ),

                "llm_call_index": (
                    current_llm_calls
                ),

                "has_tool_calls": bool(
                    tool_calls
                ),

                "tool_call_count": len(
                    tool_calls
                ),

                "tool_calls": (
                    tool_calls
                ),

                # 动态 Skill 调试信息
                "active_tools": (
                    active_tool_names
                ),

                "loaded_skills": (
                    self
                    .get_loaded_skill_ids()
                ),

                # 当前真正发送给模型的上下文审计。
                # 这里只记录，不做裁剪。
                "context_audit": (
                    context_audit
                ),

                "latency_ms": (
                    latency_ms
                ),

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
        )

        return {
            "messages": [
                response
            ],

            "turn_llm_calls": (
                current_llm_calls
            ),

            "model_trace": (
                model_trace
            ),
        }

    # ========================================================
    # Finalize 节点
    # ========================================================

    def _finalize_node(
        self,
        state: (
            PersistentRetrievalGraphState
        ),
    ) -> dict[
        str,
        Any,
    ]:
        """
        达到工具步骤上限后，
        停止继续调用工具并生成最终回答。
        """

        model_input = (
            self._build_model_input(
                state
            )
        )

        model_input.append(
            HumanMessage(
                content=(
                    "已经达到本轮工具调用步骤上限。"
                    "请停止继续调用工具，"
                    "只根据已经获得的工具结果"
                    "生成最终回答。"
                    "资料不足时请明确说明，"
                    "不要虚构。"
                )
            )
        )

        # finalize 也是一次真实 LLM 调用，
        # 因此同样记录 Context Audit。
        context_audit = (
            audit_model_input(
                model_input
            )
        )

        start_time = (
            time.perf_counter()
        )

        response = (
            self.chat_model
            .invoke(
                model_input
            )
        )

        latency_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000.0
        )

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "finalize 节点必须返回 AIMessage，"
                "实际类型："
                f"{type(response)!r}"
            )

        current_llm_calls = (
            int(
                state.get(
                    "turn_llm_calls",
                    0,
                )
            )
            + 1
        )

        model_trace = list(
            state.get(
                "model_trace",
                [],
            )
        )

        model_trace.append(
            {
                "node": (
                    "finalize"
                ),

                "llm_call_index": (
                    current_llm_calls
                ),

                "has_tool_calls": False,

                "tool_call_count": 0,

                "tool_calls": [],

                "active_tools": (
                    self
                    .get_active_tool_names()
                ),

                "loaded_skills": (
                    self
                    .get_loaded_skill_ids()
                ),

                # 当前真正发送给模型的上下文审计。
                # 这里只记录，不做裁剪。
                "context_audit": (
                    context_audit
                ),

                "latency_ms": (
                    latency_ms
                ),

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
        )

        return {
            "messages": [
                response
            ],

            "turn_llm_calls": (
                current_llm_calls
            ),

            "stopped_by_max_steps": (
                True
            ),

            "model_trace": (
                model_trace
            ),
        }

    # ========================================================
    # Tool 节点
    # ========================================================

    def _tools_node(
        self,
        state: (
            PersistentRetrievalGraphState
        ),
    ) -> dict[
        str,
        Any,
    ]:
        """
        执行当前 AIMessage 中的 Tool Call
        并记录 Trace。

        每次进入本节点前也会重新读取
        SkillRuntime 的 Active Tools。

        这里故意不使用 except Exception
        吞掉异常：

        - 参数调用错误由 ToolNode 默认转换为 ToolMessage；
        - 真正执行错误继续抛出，
          让 TOOL_RETRY_POLICY 判断是否重试；
        - 非暂时性程序错误最终暴露给上层。
        """

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        latest_ai_message = (
            _find_latest_ai_message(
                messages
            )
        )

        tool_calls = (
            normalize_tool_calls(
                latest_ai_message
            )
        )

        if not tool_calls:
            raise RuntimeError(
                "tools 节点被调用，"
                "但最新 AIMessage "
                "没有工具调用。"
            )

        # ----------------------------------------------------
        # 关键：
        # 使用当前最新 SkillRuntime 状态
        # ----------------------------------------------------

        active_tools = (
            self._refresh_tool_bindings()
        )

        active_tool_names = {
            str(
                current_tool.name
            )
            for current_tool
            in active_tools
        }

        # ----------------------------------------------------
        # 在真正执行之前额外验证：
        # 模型请求的 Tool 必须仍然是 Active Tool。
        # ----------------------------------------------------

        for tool_call in (
            tool_calls
        ):
            requested_tool_name = (
                _tool_call_name(
                    tool_call
                )
            )

            if (
                requested_tool_name
                not in active_tool_names
            ):
                raise RuntimeError(
                    "模型请求调用当前未激活的 Tool："
                    f"{requested_tool_name}；"
                    "当前 Active Tools："
                    f"{sorted(active_tool_names)}"
                )

        if (
            self.tool_node
            is None
        ):
            raise RuntimeError(
                "ToolNode 尚未初始化。"
            )

        start_time = (
            time.perf_counter()
        )

        tool_output = (
            self.tool_node.invoke(
                {
                    "messages": (
                        messages
                    )
                }
            )
        )

        latency_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000.0
        )

        output_messages = list(
            tool_output.get(
                "messages",
                [],
            )
        )

        tool_messages = [
            message
            for message
            in output_messages
            if isinstance(
                message,
                ToolMessage,
            )
        ]

        previous_tool_calls = int(
            state.get(
                "turn_tool_calls",
                0,
            )
        )

        current_tool_calls = (
            previous_tool_calls
            + len(
                tool_calls
            )
        )

        tool_trace = list(
            state.get(
                "tool_trace",
                [],
            )
        )

        for index, tool_call in enumerate(
            tool_calls,
            start=1,
        ):
            call_id = (
                _tool_call_id(
                    tool_call
                )
            )

            tool_message = (
                _find_tool_message_by_call_id(
                    tool_messages,
                    call_id,
                )
            )

            if (
                tool_message is None
                and index
                <= len(
                    tool_messages
                )
            ):
                tool_message = (
                    tool_messages[
                        index - 1
                    ]
                )

            output_text = (
                _message_content_to_text(
                    tool_message.content
                )
                if tool_message
                is not None
                else ""
            )

            status = (
                str(
                    getattr(
                        tool_message,
                        "status",
                        "success",
                    )
                )
                if tool_message
                is not None
                else "unknown"
            )

            tool_trace.append(
                {
                    "tool_call_index": (
                        previous_tool_calls
                        + index
                    ),

                    "name": (
                        _tool_call_name(
                            tool_call
                        )
                    ),

                    "args": (
                        _tool_call_args(
                            tool_call
                        )
                    ),

                    "tool_call_id": (
                        call_id
                    ),

                    "status": (
                        status
                    ),

                    "output_preview": (
                        _truncate_text(
                            output_text,
                            500,
                        )
                    ),

                    "active_tools_before_execution": (
                        sorted(
                            active_tool_names
                        )
                    ),

                    "loaded_skills": (
                        self
                        .get_loaded_skill_ids()
                    ),

                    # 多个工具并行执行时
                    # 无法准确拆分单 Tool 耗时，
                    # 因此记录本次 ToolNode 总耗时。
                    "tool_node_latency_ms": (
                        latency_ms
                    ),
                }
            )

        return {
            "messages": (
                tool_messages
            ),

            "turn_tool_calls": (
                current_tool_calls
            ),

            "tool_trace": (
                tool_trace
            ),
        }

    # ========================================================
    # 路由
    # ========================================================

    def _route_after_model(
        self,
        state: (
            PersistentRetrievalGraphState
        ),
    ) -> str:
        """
        模型有 Tool Call 时进入 tools，
        否则进入 memory_manager。
        """

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        if not messages:
            raise RuntimeError(
                "agent 节点执行后"
                "没有任何消息。"
            )

        latest_message = (
            messages[-1]
        )

        if not isinstance(
            latest_message,
            AIMessage,
        ):
            raise TypeError(
                "agent 节点结束后的"
                "最后一条消息必须是 "
                "AIMessage。"
            )

        tool_calls = (
            normalize_tool_calls(
                latest_message
            )
        )

        return (
            "tools"
            if tool_calls
            else "memory_manager"
        )

    def _route_after_tools(
        self,
        state: (
            PersistentRetrievalGraphState
        ),
    ) -> str:
        """
        达到单轮 Tool Call 上限时
        进入 finalize。
        """

        tool_call_count = int(
            state.get(
                "turn_tool_calls",
                0,
            )
        )

        if (
            tool_call_count
            >= self.max_steps
        ):
            return "finalize"

        return "agent"

    # ========================================================
    # 滚动摘要
    # ========================================================

    def _format_messages_for_summary(
        self,
        messages: Sequence[
            BaseMessage
        ],
    ) -> str:
        """
        把即将移出的旧消息
        整理为摘要模型输入。
        """

        lines: list[
            str
        ] = []

        for message in messages:

            # ------------------------------------------------
            # Human
            # ------------------------------------------------

            if isinstance(
                message,
                HumanMessage,
            ):
                text = (
                    _truncate_text(
                        _message_content_to_text(
                            message.content
                        ),
                        1600,
                    )
                )

                if text:
                    lines.append(
                        f"用户：{text}"
                    )

                continue

            # ------------------------------------------------
            # AI
            # ------------------------------------------------

            if isinstance(
                message,
                AIMessage,
            ):
                tool_calls = (
                    normalize_tool_calls(
                        message
                    )
                )

                if tool_calls:
                    tool_names = (
                        ", ".join(
                            _tool_call_name(
                                tool_call
                            )
                            for tool_call
                            in tool_calls
                        )
                    )

                    lines.append(
                        "助手调用工具："
                        f"{tool_names}"
                    )

                text = (
                    _truncate_text(
                        _message_content_to_text(
                            message.content
                        ),
                        1800,
                    )
                )

                if text:
                    lines.append(
                        f"助手：{text}"
                    )

                continue

            # ------------------------------------------------
            # Tool
            # ------------------------------------------------

            if isinstance(
                message,
                ToolMessage,
            ):
                tool_name = str(
                    getattr(
                        message,
                        "name",
                        "tool",
                    )
                    or "tool"
                )

                status = str(
                    getattr(
                        message,
                        "status",
                        "success",
                    )
                )

                lines.append(
                    f"工具结果：{tool_name} "
                    f"已返回，status={status}；"
                    "完整工具输出不纳入摘要输入。"
                )

                continue

            # ------------------------------------------------
            # 其他 Message
            # ------------------------------------------------

            text = (
                _truncate_text(
                    _message_content_to_text(
                        getattr(
                            message,
                            "content",
                            "",
                        )
                    ),
                    800,
                )
            )

            if text:
                lines.append(
                    f"其他消息：{text}"
                )

        return "\n\n".join(
            lines
        ).strip()

    def _memory_manager_node(
        self,
        state: (
            PersistentRetrievalGraphState
        ),
    ) -> dict[
        str,
        Any,
    ]:
        """Token-aware Working Memory compaction.

        三层职责：
        1. Event Store 保存完整原始历史；
        2. summary 保存线程近况/进展地图；
        3. checkpoint messages 保存近期原始 Working Memory。

        本节点不再使用“达到第 7 轮 / 固定保留 4 轮”
        作为真实压缩条件。
        """

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        existing_summary = str(
            state.get(
                "summary",
                "",
            )
            or ""
        ).strip()

        thread_id = str(
            state.get(
                "working_memory_thread_id",
                "",
            )
            or ""
        ).strip()

        event_store = getattr(
            self,
            "conversation_event_store",
            None,
        )

        plan = (
            plan_working_memory_compaction(
                messages,
                summary=(
                    existing_summary
                ),
                config=(
                    self
                    .working_memory_audit_config
                ),
                event_store=(
                    event_store
                ),
                thread_id=(
                    thread_id
                ),
                # 仅用于 Trace 对照，不再控制行为。
                legacy_keep_recent_turns=(
                    self.keep_recent_turns
                ),
                legacy_summarize_trigger_turns=(
                    self
                    .summarize_trigger_turns
                ),
            )
        )

        working_memory_audit = {
            "before": (
                plan.audit_before
            ),
            "plan": (
                plan.to_diagnostics()
            ),
            "token_compaction_applied": (
                False
            ),
            "summary_role": (
                "thread_progress_map"
            ),
            "event_store_role": (
                "full_raw_history_source_of_truth"
            ),
        }

        # ----------------------------------------------------
        # 不需要压缩
        # ----------------------------------------------------

        if not plan.should_compact:
            return {
                "summary_updated": (
                    False
                ),
                "summarized_turns_this_run": (
                    0
                ),
                "working_memory_audit": (
                    working_memory_audit
                ),
            }

        # ----------------------------------------------------
        # 需要压，但当前没有安全候选
        #
        # 典型情况：
        # - 当前 Turn 自己很巨大，但尚未归档；
        # - 历史 Turn 尚未完整进入 Event Store；
        # - 存在不可安全清理的协议状态。
        #
        # Fail closed：不删。
        # ----------------------------------------------------

        if (
            not plan.can_compact
            or not plan.messages_to_summarize
        ):
            working_memory_audit[
                "compaction_deferred"
            ] = True

            return {
                "summary_updated": (
                    False
                ),
                "summarized_turns_this_run": (
                    0
                ),
                "working_memory_audit": (
                    working_memory_audit
                ),
            }

        messages_to_summarize = list(
            plan.messages_to_summarize
        )

        retained_messages = list(
            plan.retained_messages
        )

        summarized_turns = (
            plan.selected_turn_count
        )

        # ----------------------------------------------------
        # 先更新 Thread Summary，后删除 Checkpoint 原始 Turn。
        #
        # 如果摘要 LLM 失败，本节点抛错，
        # RemoveMessage 根本不会被返回，因此不会丢 Working State。
        # ----------------------------------------------------

        existing_summary_text = (
            existing_summary
            if existing_summary
            else "暂无。"
        )

        formatted_history = (
            self._format_messages_for_summary(
                messages_to_summarize
            )
        )

        summary_prompt = (
            "已有线程工作摘要：\n"
            f"{existing_summary_text}"
            "\n\n"
            "以下已完成历史 Turn 即将从 "
            "Checkpoint Working Messages 中移出，"
            "但完整原始记录已经保存在 "
            "Conversation Event Store。\n\n"
            "请把这些 Turn 中会影响后续工作的目标、"
            "已确定方案、关键事实、约束、进度和待办"
            "吸收到线程摘要中：\n\n"
            f"{formatted_history}"
            "\n\n"
            "请输出更新后的完整线程工作摘要。"
        )

        start_time = (
            time.perf_counter()
        )

        response = (
            self.chat_model.invoke(
                [
                    SystemMessage(
                        content=(
                            ROLLING_SUMMARY_SYSTEM_PROMPT
                        )
                    ),
                    HumanMessage(
                        content=(
                            summary_prompt
                        )
                    ),
                ]
            )
        )

        latency_ms = (
            (
                time.perf_counter()
                - start_time
            )
            * 1000.0
        )

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "memory_manager 节点必须返回 "
                "AIMessage，实际类型："
                f"{type(response)!r}"
            )

        updated_summary = (
            extract_answer_text(
                response
            )
            .strip()
        )

        if not updated_summary:
            raise ValueError(
                "线程工作摘要模型返回了空摘要。"
            )

        current_llm_calls = (
            int(
                state.get(
                    "turn_llm_calls",
                    0,
                )
            )
            + 1
        )

        current_summary_calls = (
            int(
                state.get(
                    "turn_summary_calls",
                    0,
                )
            )
            + 1
        )

        total_summarized_turns = (
            int(
                state.get(
                    "total_summarized_turns",
                    0,
                )
            )
            + summarized_turns
        )

        model_trace = list(
            state.get(
                "model_trace",
                [],
            )
        )

        model_trace.append(
            {
                "node": (
                    "memory_manager"
                ),
                "mode": (
                    "token_aware_working_memory"
                ),
                "llm_call_index": (
                    current_llm_calls
                ),
                "summary_call_index": (
                    current_summary_calls
                ),
                "summarized_turns": (
                    summarized_turns
                ),
                "selected_turn_keys": list(
                    plan.selected_turn_keys
                ),
                "retained_turns": (
                    plan.retained_turn_count
                ),
                "working_memory_tokens_before": (
                    plan.before_tokens
                ),
                "working_memory_tokens_predicted_after": (
                    plan.predicted_after_tokens
                ),
                "working_memory_target_tokens": (
                    plan.target_tokens
                ),
                "predicted_target_reached": (
                    plan.predicted_target_reached
                ),
                "archive_verified_turn_keys": list(
                    plan.archive_verified_turn_keys
                ),
                "latency_ms": (
                    latency_ms
                ),
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
        )

        # ----------------------------------------------------
        # 只有摘要已经成功生成后，才真正更新 Checkpoint。
        # ----------------------------------------------------

        message_updates: list[
            BaseMessage
        ] = [
            cast(
                BaseMessage,
                RemoveMessage(
                    id=(
                        REMOVE_ALL_MESSAGES
                    )
                ),
            ),
            *retained_messages,
        ]

        working_memory_after = (
            audit_working_memory(
                retained_messages,
                summary=(
                    updated_summary
                ),
                config=(
                    self
                    .working_memory_audit_config
                ),
                legacy_keep_recent_turns=(
                    self.keep_recent_turns
                ),
                legacy_summarize_trigger_turns=(
                    self
                    .summarize_trigger_turns
                ),
            )
        )

        working_memory_audit[
            "token_compaction_applied"
        ] = True

        working_memory_audit[
            "after"
        ] = (
            working_memory_after
        )

        working_memory_audit[
            "compaction_deferred"
        ] = False

        return {
            "messages": (
                message_updates
            ),
            "summary": (
                updated_summary
            ),
            "total_summarized_turns": (
                total_summarized_turns
            ),
            "turn_llm_calls": (
                current_llm_calls
            ),
            "turn_summary_calls": (
                current_summary_calls
            ),
            "summary_updated": (
                True
            ),
            "summarized_turns_this_run": (
                summarized_turns
            ),
            "model_trace": (
                model_trace
            ),
            "working_memory_audit": (
                working_memory_audit
            ),
        }

    # ========================================================
    # Checkpointer 操作
    # ========================================================

    def _build_config(
        self,
        thread_id: str,
    ) -> dict[
        str,
        Any,
    ]:
        """构造 LangGraph 调用配置。"""

        normalized_thread_id = (
            normalize_thread_id(
                thread_id
            )
        )

        # 路径大致为：
        #
        # agent
        # → tools
        # → agent
        # → tools
        # → ...
        # → memory_manager
        #
        # recursion_limit
        # 只是图级最后保险。
        #
        # 正常上限仍由 max_steps 控制。

        recursion_limit = max(
            25,
            self.max_steps
            * 4
            + 10,
        )

        return {
            "configurable": {
                "thread_id": (
                    normalized_thread_id
                ),
            },

            "recursion_limit": (
                recursion_limit
            ),
        }

    def get_thread_state(
        self,
        thread_id: str,
    ) -> dict[
        str,
        Any,
    ]:
        """读取指定 thread 的最新状态。"""

        snapshot = (
            self.graph.get_state(
                self._build_config(
                    thread_id
                )
            )
        )

        values = getattr(
            snapshot,
            "values",
            None,
        )

        if not values:
            return {}

        return dict(
            values
        )

    def get_thread_messages(
        self,
        thread_id: str,
    ) -> list[
        BaseMessage
    ]:
        """读取指定 thread 当前保留的详细消息。"""

        state = (
            self.get_thread_state(
                thread_id
            )
        )

        return list(
            state.get(
                "messages",
                [],
            )
            or []
        )

    def get_thread_summary(
        self,
        thread_id: str,
    ) -> str:
        """读取指定 thread 的滚动摘要。"""

        state = (
            self.get_thread_state(
                thread_id
            )
        )

        return str(
            state.get(
                "summary",
                "",
            )
        ).strip()

    def clear_thread(
        self,
        thread_id: str,
    ) -> None:
        """
        删除指定 thread 的全部 Checkpoint 状态。

        注意：

        SkillRuntime 当前属于 Agent Runtime，
        不是 thread checkpoint。

        因此 clear_thread()
        不会自动卸载已经加载的 Skill。
        """

        normalized_thread_id = (
            normalize_thread_id(
                thread_id
            )
        )

        delete_thread = getattr(
            self.checkpointer,
            "delete_thread",
            None,
        )

        if not callable(
            delete_thread
        ):
            raise NotImplementedError(
                "当前 Checkpointer 不支持 "
                "delete_thread(thread_id)。"
            )

        delete_thread(
            normalized_thread_id
        )

    # ========================================================
    # 对外执行入口
    # ========================================================
    def _build_checkpoint_config(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        """构造读取指定 Checkpoint 的 LangGraph config。"""

        normalized_checkpoint_id = str(
            checkpoint_id
        ).strip()

        if not normalized_checkpoint_id:
            raise ValueError(
                "checkpoint_id 不能为空。"
            )

        config = self._build_config(
            thread_id
        )

        configurable = dict(
            config.get(
                "configurable",
                {},
            )
        )

        configurable[
            "checkpoint_id"
        ] = normalized_checkpoint_id

        config[
            "configurable"
        ] = configurable

        return config


    @staticmethod
    def _checkpoint_snapshot_to_dict(
        snapshot: Any,
        *,
        include_values: bool = False,
    ) -> dict[str, Any]:
        """将 LangGraph StateSnapshot 转换为普通字典。

        重点保留：

        - checkpoint_id；
        - step；
        - source；
        - metadata writes；
        - next nodes；
        - tasks；
        - messages；
        - summary；
        - model/tool trace。

        注意：

        metadata / config 等对象统一按 Mapping
        处理，而不是只接受内置 dict，
        以兼容不同 LangGraph 版本。
        """

        # ----------------------------------------------------
        # Config
        # ----------------------------------------------------

        raw_config = (
            getattr(
                snapshot,
                "config",
                None,
            )
            or {}
        )

        config = (
            dict(raw_config)
            if isinstance(
                raw_config,
                Mapping,
            )
            else {}
        )

        raw_configurable = (
            config.get(
                "configurable",
                {},
            )
            or {}
        )

        configurable = (
            dict(raw_configurable)
            if isinstance(
                raw_configurable,
                Mapping,
            )
            else {}
        )

        checkpoint_id = str(
            configurable.get(
                "checkpoint_id",
                "",
            )
            or ""
        )

        # ----------------------------------------------------
        # Parent Config
        # ----------------------------------------------------

        raw_parent_config = (
            getattr(
                snapshot,
                "parent_config",
                None,
            )
            or {}
        )

        parent_config = (
            dict(raw_parent_config)
            if isinstance(
                raw_parent_config,
                Mapping,
            )
            else {}
        )

        raw_parent_configurable = (
            parent_config.get(
                "configurable",
                {},
            )
            or {}
        )

        parent_configurable = (
            dict(
                raw_parent_configurable
            )
            if isinstance(
                raw_parent_configurable,
                Mapping,
            )
            else {}
        )

        parent_checkpoint_id = str(
            parent_configurable.get(
                "checkpoint_id",
                "",
            )
            or ""
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        raw_metadata = (
            getattr(
                snapshot,
                "metadata",
                None,
            )
            or {}
        )

        metadata = (
            dict(raw_metadata)
            if isinstance(
                raw_metadata,
                Mapping,
            )
            else {}
        )

        source = str(
            metadata.get(
                "source",
                "",
            )
            or ""
        )

        step = metadata.get(
            "step"
        )

        # ----------------------------------------------------
        # Writes
        # ----------------------------------------------------
        #
        # 官方 StateSnapshot.metadata["writes"]
        # 记录产生当前 checkpoint 的节点输出。
        #
        # 例如：
        #
        # writes = {
        #     "agent": {...}
        # }
        #
        # 某些 checkpoint（尤其初始 step）
        # writes 可能本身就是 None。
        # ----------------------------------------------------

        raw_writes = metadata.get(
            "writes"
        )

        write_nodes: list[str] = []

        if isinstance(
            raw_writes,
            Mapping,
        ):
            write_nodes = [
                str(
                    current
                )
                for current
                in raw_writes.keys()
            ]

        # ----------------------------------------------------
        # State Values
        # ----------------------------------------------------

        raw_values = (
            getattr(
                snapshot,
                "values",
                None,
            )
            or {}
        )

        values = (
            dict(raw_values)
            if isinstance(
                raw_values,
                Mapping,
            )
            else {}
        )

        messages = list(
            values.get(
                "messages",
                [],
            )
            or []
        )

        summary = str(
            values.get(
                "summary",
                "",
            )
            or ""
        ).strip()

        # ----------------------------------------------------
        # Next
        # ----------------------------------------------------

        next_nodes = [
            str(
                current
            )
            for current in (
                getattr(
                    snapshot,
                    "next",
                    (),
                )
                or ()
            )
        ]

        # ----------------------------------------------------
        # Tasks
        # ----------------------------------------------------
        #
        # tasks 表示当前 checkpoint 所调度的任务。
        #
        # 通常其 name 与 next node 对应。
        #
        # task 还可能携带：
        # - error
        # - interrupts
        # ----------------------------------------------------

        tasks = list(
            getattr(
                snapshot,
                "tasks",
                (),
            )
            or ()
        )

        task_names: list[str] = []
        task_errors: list[str] = []

        interrupt_count = 0

        for task in tasks:

            task_name = str(
                getattr(
                    task,
                    "name",
                    "",
                )
                or ""
            )

            if task_name:
                task_names.append(
                    task_name
                )

            error = getattr(
                task,
                "error",
                None,
            )

            if error:
                task_errors.append(
                    str(
                        error
                    )
                )

            interrupts = (
                getattr(
                    task,
                    "interrupts",
                    None,
                )
                or ()
            )

            interrupt_count += len(
                interrupts
            )

        # ----------------------------------------------------
        # Trace
        # ----------------------------------------------------

        model_trace = list(
            values.get(
                "model_trace",
                [],
            )
            or []
        )

        tool_trace = list(
            values.get(
                "tool_trace",
                [],
            )
            or []
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result: dict[str, Any] = {

            "checkpoint_id": (
                checkpoint_id
            ),

            "parent_checkpoint_id": (
                parent_checkpoint_id
            ),

            "created_at": str(
                getattr(
                    snapshot,
                    "created_at",
                    "",
                )
                or ""
            ),

            "step": step,

            "source": source,

            # metadata 中真正读取到的 writes
            "write_nodes": (
                write_nodes
            ),

            # 后面 list_thread_checkpoints()
            # 会在必要时补 fallback。
            "completed_nodes": (
                list(
                    write_nodes
                )
            ),

            "writes_available": bool(
                write_nodes
            ),

            "next_nodes": (
                next_nodes
            ),

            "task_names": (
                task_names
            ),

            "task_errors": (
                task_errors
            ),

            "interrupt_count": (
                interrupt_count
            ),

            "message_count": len(
                messages
            ),

            "has_summary": bool(
                summary
            ),

            "summary": summary,

            "turn_llm_calls": int(
                values.get(
                    "turn_llm_calls",
                    0,
                )
                or 0
            ),

            "turn_tool_calls": int(
                values.get(
                    "turn_tool_calls",
                    0,
                )
                or 0
            ),

            "turn_summary_calls": int(
                values.get(
                    "turn_summary_calls",
                    0,
                )
                or 0
            ),

            "model_trace_count": len(
                model_trace
            ),

            "tool_trace_count": len(
                tool_trace
            ),

            "state_keys": sorted(
                str(
                    current
                )
                for current
                in values.keys()
            ),

            # 开发调试用：
            # 保留 metadata 的 key，
            # 不直接输出整个 writes value。
            "metadata_keys": sorted(
                str(
                    current
                )
                for current
                in metadata.keys()
            ),
        }

        if include_values:

            result[
                "values"
            ] = values

            result[
                "metadata"
            ] = metadata

        return result

    def list_thread_checkpoints(
        self,
        thread_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """读取指定 thread 最近的 Checkpoint 历史。

        LangGraph 返回顺序：

            newest
            ↓
            oldest

        除读取真实 metadata.writes 外，
        如果 writes 缺失，则根据：

            上一个更旧 Checkpoint 的 next

        推断“刚刚执行完成的节点”。

        例如：

            step50:
                next = agent

            step51:
                writes 缺失

        可以合理推断：

            step51 刚刚完成的是 agent。

        这种推断只作为展示 fallback，
        不参与 Graph 执行和恢复。
        """

        if limit <= 0:
            raise ValueError(
                "limit 必须大于 0。"
            )

        config = self._build_config(
            thread_id
        )

        # ----------------------------------------------------
        # 多取一个 snapshot
        # ----------------------------------------------------
        #
        # 因为要利用“更旧一条”的 next
        # 推断当前 snapshot 的 completed node。
        # ----------------------------------------------------

        raw_snapshots: list[Any] = []

        for snapshot in (
            self.graph.get_state_history(
                config
            )
        ):

            raw_snapshots.append(
                snapshot
            )

            if (
                len(
                    raw_snapshots
                )
                >= limit + 1
            ):
                break

        checkpoints = [
            self._checkpoint_snapshot_to_dict(
                snapshot
            )
            for snapshot
            in raw_snapshots
        ]

        # ----------------------------------------------------
        # 补充 Completed Node
        # ----------------------------------------------------
        #
        # history 是 newest -> oldest。
        #
        # checkpoints[index + 1]
        # 是当前 checkpoint 的“上一张旧照片”。
        #
        # 旧照片的 next：
        #
        #     agent
        #
        # 表示随后执行 agent，
        # 因此新照片通常就是 agent 执行之后的状态。
        # ----------------------------------------------------

        for index, checkpoint in enumerate(
            checkpoints
        ):

            write_nodes = list(
                checkpoint.get(
                    "write_nodes",
                    [],
                )
                or []
            )

            # 有官方 writes：
            # 直接相信真实 metadata。
            if write_nodes:

                checkpoint[
                    "completed_nodes"
                ] = write_nodes

                checkpoint[
                    "completed_nodes_source"
                ] = "metadata.writes"

            else:

                fallback_nodes: list[
                    str
                ] = []

                if (
                    index + 1
                    < len(
                        checkpoints
                    )
                ):

                    older_checkpoint = (
                        checkpoints[
                            index + 1
                        ]
                    )

                    fallback_nodes = list(
                        older_checkpoint.get(
                            "next_nodes",
                            [],
                        )
                        or []
                    )

                checkpoint[
                    "completed_nodes"
                ] = fallback_nodes

                checkpoint[
                    "completed_nodes_source"
                ] = (
                    "previous_checkpoint.next"
                    if fallback_nodes
                    else "unknown"
                )

            # ------------------------------------------------
            # Human-friendly Event
            # ------------------------------------------------

            source = str(
                checkpoint.get(
                    "source",
                    "",
                )
            )

            completed_nodes = list(
                checkpoint.get(
                    "completed_nodes",
                    [],
                )
                or []
            )

            next_nodes = list(
                checkpoint.get(
                    "next_nodes",
                    [],
                )
                or []
            )

            if source == "input":

                event = (
                    "新输入进入 Graph"
                )

            elif completed_nodes:

                completed_text = (
                    ", ".join(
                        completed_nodes
                    )
                )

                event = (
                    f"{completed_text} 执行完成"
                )

            elif not next_nodes:

                event = (
                    "Graph 已完成"
                )

            else:

                event = (
                    "Graph 状态推进"
                )

            checkpoint[
                "event"
            ] = event

        return checkpoints[
            :limit
        ]

    def get_thread_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        """读取指定 thread 的某一个 Checkpoint。"""

        normalized_checkpoint_id = str(
            checkpoint_id
        ).strip()

        if not normalized_checkpoint_id:
            raise ValueError(
                "checkpoint_id 不能为空。"
            )

        config = (
            self._build_checkpoint_config(
                thread_id,
                normalized_checkpoint_id,
            )
        )

        # ----------------------------------------------------
        # 先确认 Checkpoint 确实存在
        # ----------------------------------------------------

        get_tuple = getattr(
            self.checkpointer,
            "get_tuple",
            None,
        )

        if callable(
            get_tuple
        ):

            checkpoint_tuple = (
                get_tuple(
                    config
                )
            )

            if checkpoint_tuple is None:
                return {}

        snapshot = self.graph.get_state(
            config
        )

        result = (
            self._checkpoint_snapshot_to_dict(
                snapshot,
                include_values=True,
            )
        )

        returned_checkpoint_id = str(
            result.get(
                "checkpoint_id",
                "",
            )
        )

        if (
            returned_checkpoint_id
            != normalized_checkpoint_id
        ):
            return {}

        return result


    def run(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> PersistentLangGraphResult:
        """
        在指定 thread 中执行一轮 Agent 问答。
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

        normalized_thread_id = (
            normalize_thread_id(
                thread_id
            )
        )

        config = (
            self._build_config(
                normalized_thread_id
            )
        )

        input_state: dict[
            str,
            Any,
        ] = {
            "messages": [
                HumanMessage(
                    content=(
                        normalized_question
                    ),
                    id=(
                        "human-"
                        + uuid.uuid4().hex
                    ),
                )
            ],

            # 这些字段按“本轮”统计，
            # 因此每次新请求都覆盖为初值。

            "turn_llm_calls": (
                0
            ),

            "turn_tool_calls": (
                0
            ),

            "turn_summary_calls": (
                0
            ),

            "summary_updated": (
                False
            ),

            "summarized_turns_this_run": (
                0
            ),

            "stopped_by_max_steps": (
                False
            ),

            "model_trace": [],

            "tool_trace": [],

            # Working Memory Phase 7E-2
            "working_memory_audit": {},
            "working_memory_thread_id": (
                normalized_thread_id
            ),
        }

        total_start = (
            time.perf_counter()
        )

        final_state = (
            self.graph.invoke(
                input_state,
                config=config,
            )
        )

        total_latency_ms = (
            (
                time.perf_counter()
                - total_start
            )
            * 1000.0
        )

        messages = list(
            final_state.get(
                "messages",
                [],
            )
        )

        if not messages:
            raise RuntimeError(
                "图执行完成后没有返回消息。"
            )

        final_message = (
            messages[-1]
        )

        if not isinstance(
            final_message,
            AIMessage,
        ):
            raise RuntimeError(
                "图执行完成后的最后一条消息"
                "不是 AIMessage，"
                "实际类型："
                f"{type(final_message)!r}"
            )

        answer = (
            extract_answer_text(
                final_message
            )
            .strip()
        )

        stopped_by_max_steps = bool(
            final_state.get(
                "stopped_by_max_steps",
                False,
            )
        )

        return PersistentLangGraphResult(
            thread_id=(
                normalized_thread_id
            ),

            question=(
                normalized_question
            ),

            answer=(
                answer
            ),

            messages=(
                messages
            ),

            summary=str(
                final_state.get(
                    "summary",
                    "",
                )
            ).strip(),

            turn_llm_call_count=int(
                final_state.get(
                    "turn_llm_calls",
                    0,
                )
            ),

            turn_tool_call_count=int(
                final_state.get(
                    "turn_tool_calls",
                    0,
                )
            ),

            turn_summary_call_count=int(
                final_state.get(
                    "turn_summary_calls",
                    0,
                )
            ),

            summary_updated=bool(
                final_state.get(
                    "summary_updated",
                    False,
                )
            ),

            summarized_turns_this_run=int(
                final_state.get(
                    "summarized_turns_this_run",
                    0,
                )
            ),

            total_summarized_turns=int(
                final_state.get(
                    "total_summarized_turns",
                    0,
                )
            ),

            recent_turn_count=(
                count_human_turns(
                    messages
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

            total_message_count=len(
                messages
            ),

            total_latency_ms=(
                total_latency_ms
            ),

            final_state=dict(
                final_state
            ),
        )
