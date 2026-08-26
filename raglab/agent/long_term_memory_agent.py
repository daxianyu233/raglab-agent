"""带跨线程长期记忆的 LangGraph Retrieval Agent。

短期记忆：
    由 Checkpointer 按 thread_id 保存。

长期记忆：
    由 Store 按 user_id namespace 保存。

同一个 user_id 的不同 thread_id：
    短期消息相互隔离；
    长期记忆可以共享。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from raglab.agent.context_manager import (
    audit_model_input,
)
from raglab.agent.context_plan import (
    ContextPlan,
)
from raglab.agent.context_planner import (
    ContextPlanner,
)
from raglab.agent.context_runtime import (
    AgentContextPipeline,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.langgraph_retrieval_agent import (
    normalize_tool_calls,
)
from raglab.agent.persistent_langgraph_agent import (
    PersistentLangGraphResult,
    PersistentLangGraphRetrievalAgent,
    PersistentRetrievalGraphState,
    count_human_turns,
    normalize_thread_id,
)
from raglab.agent.retrieval_agent import (
    RETRIEVAL_AGENT_SYSTEM_PROMPT,
)
from raglab.agent.skill_runtime import SkillRuntime
from raglab.agent.tool_exposure import (
    ToolExposureDecision,
    select_tools_for_context,
)
from raglab.generation.rag_chain import (
    ChatModelProtocol,
    extract_answer_text,
    extract_response_metadata,
    extract_usage_metadata,
)
from raglab.observability.runtime_events import emit_runtime_event


@dataclass(frozen=True)
class LongTermMemoryContext:
    """一次图调用所需的运行时上下文。

    user_id 不属于 thread 图状态，
    因此通过 Runtime Context 传入。
    """

    user_id: str


def normalize_user_id(
    user_id: str,
) -> str:
    """检查并规范化 user_id。"""

    normalized = str(
        user_id
    ).strip()

    if not normalized:
        raise ValueError(
            "user_id 不能为空。"
        )

    return normalized


def normalize_memory_key(
    key: str,
) -> str:
    """检查长期记忆的唯一键。"""

    normalized = str(
        key
    ).strip()

    if not normalized:
        raise ValueError(
            "长期记忆 key 不能为空。"
        )

    return normalized


def build_memory_namespace(
    user_id: str,
) -> tuple[str, str, str]:
    """构造用户长期记忆命名空间。"""

    return (
        "users",
        normalize_user_id(
            user_id
        ),
        "memories",
    )


class LongTermMemoryRetrievalAgent(
    PersistentLangGraphRetrievalAgent
):
    """同时支持短期与长期记忆的 Agent。"""

    def __init__(
        self,
        *,
        chat_model: ChatModelProtocol,
        tools: Sequence[BaseTool],
        max_steps: int = 4,
        system_prompt: str = (
            RETRIEVAL_AGENT_SYSTEM_PROMPT
        ),
        keep_recent_turns: int = 4,
        summarize_trigger_turns: int = 7,
        checkpointer: Any | None = None,
        store: BaseStore | None = None,
        maximum_loaded_memories: int = 50,
        skill_runtime: SkillRuntime | None = None,
        conversation_event_store: (
            ConversationEventStore
            | None
        ) = None,
        context_planner: (
            ContextPlanner
            | None
        ) = None,
        context_pipeline_enabled: bool = True,
        context_window_tokens: int = 32768,
        reserved_output_tokens: int = 4096,
        context_safety_margin_tokens: int = 1024,
        context_recent_turn_limit: int = 3,
        context_historical_turn_limit: int = 3,
    ) -> None:
        """初始化长期记忆 Agent。

        store 未提供时使用 InMemoryStore。

        InMemoryStore 仅适合当前进程内的
        学习和功能验证。
        """

        if maximum_loaded_memories <= 0:
            raise ValueError(
                "maximum_loaded_memories "
                "必须大于 0。"
            )

        self.long_term_store = (
            store
            if store is not None
            else InMemoryStore()
        )

        self.maximum_loaded_memories = int(
            maximum_loaded_memories
        )

        # ----------------------------------------------------
        # Context Pipeline Phase 7A
        # ----------------------------------------------------
        #
        # Event Store 是完整会话历史 Source of Truth。
        # SecureAgentRuntime 负责写入；
        # 本 Agent 在新 Human Turn 开始前负责读取。
        self.conversation_event_store = (
            conversation_event_store
            or ConversationEventStore()
        )

        self.context_planner = (
            context_planner
            or ContextPlanner(
                chat_model=chat_model
            )
        )

        self.context_pipeline_enabled = bool(
            context_pipeline_enabled
        )

        self.context_pipeline = (
            AgentContextPipeline(
                planner=(
                    self.context_planner
                ),
                event_store=(
                    self.conversation_event_store
                ),
                model_context_limit_tokens=(
                    int(
                        context_window_tokens
                    )
                ),
                reserved_output_tokens=(
                    int(
                        reserved_output_tokens
                    )
                ),
                safety_margin_tokens=(
                    int(
                        context_safety_margin_tokens
                    )
                ),
                recent_turn_limit=(
                    int(
                        context_recent_turn_limit
                    )
                ),
                historical_turn_limit=(
                    int(
                        context_historical_turn_limit
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # Context Tool Exposure Phase 7C
        # ----------------------------------------------------
        #
        # SecureAgentRuntime 创建后会把：
        #
        #     SQLiteToolPolicyStore.get_policy
        #
        # 注入这里。
        #
        # LongTermMemoryAgent 不直接 import runtime_security，
        # 避免 Context / Security 两层形成循环依赖。
        self.context_tool_policy_resolver = None

        # 父类构造函数最终会调用
        # self._build_graph()。
        #
        # 由于当前对象是子类实例，
        # 实际会调用本类覆盖后的
        # _build_graph()。
        super().__init__(
            chat_model=chat_model,
            tools=tools,
            max_steps=max_steps,
            system_prompt=system_prompt,
            keep_recent_turns=(
                keep_recent_turns
            ),
            summarize_trigger_turns=(
                summarize_trigger_turns
            ),
            checkpointer=checkpointer,
            skill_runtime=skill_runtime,
        )

    def remember(
        self,
        *,
        user_id: str,
        key: str,
        content: str,
        category: str = "explicit",
        source: str = "user_command",
    ) -> dict[str, Any]:
        """新增或更新一条长期记忆。

        相同 namespace 和 key 再次 put，
        会更新原来的记忆。
        """

        normalized_user_id = (
            normalize_user_id(
                user_id
            )
        )

        normalized_key = (
            normalize_memory_key(
                key
            )
        )

        normalized_content = str(
            content
        ).strip()

        if not normalized_content:
            raise ValueError(
                "长期记忆 content 不能为空。"
            )

        normalized_category = str(
            category
        ).strip() or "explicit"

        normalized_source = str(
            source
        ).strip() or "user_command"

        namespace = (
            build_memory_namespace(
                normalized_user_id
            )
        )

        value = {
            "content": (
                normalized_content
            ),
            "category": (
                normalized_category
            ),
            "source": (
                normalized_source
            ),
        }

        self.long_term_store.put(
            namespace,
            normalized_key,
            value,
            # 当前版本未启用向量索引。
            index=False,
        )

        return {
            "user_id": (
                normalized_user_id
            ),
            "namespace": namespace,
            "key": normalized_key,
            "value": value,
        }

    def get_memory(
        self,
        *,
        user_id: str,
        key: str,
    ) -> dict[str, Any] | None:
        """按 key 读取一条长期记忆。"""

        namespace = (
            build_memory_namespace(
                user_id
            )
        )

        normalized_key = (
            normalize_memory_key(
                key
            )
        )

        item = self.long_term_store.get(
            namespace,
            normalized_key,
        )

        if item is None:
            return None

        return {
            "key": item.key,
            "value": dict(
                item.value
            ),
            "namespace": tuple(
                item.namespace
            ),
            "created_at": str(
                item.created_at
            ),
            "updated_at": str(
                item.updated_at
            ),
        }

    def list_memories(
        self,
        *,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """列出用户的全部长期记忆。"""

        namespace = (
            build_memory_namespace(
                user_id
            )
        )

        items = self.long_term_store.search(
            namespace,
            limit=(
                self.maximum_loaded_memories
            ),
        )

        results: list[
            dict[str, Any]
        ] = []

        for item in items:
            results.append(
                {
                    "key": item.key,
                    "value": dict(
                        item.value
                    ),
                    "namespace": tuple(
                        item.namespace
                    ),
                    "created_at": str(
                        item.created_at
                    ),
                    "updated_at": str(
                        item.updated_at
                    ),
                }
            )

        results.sort(
            key=lambda current: str(
                current["key"]
            )
        )

        return results

    def forget(
        self,
        *,
        user_id: str,
        key: str,
    ) -> bool:
        """删除一条长期记忆。

        Returns
        -------
        bool
            删除前存在时返回 True；
            不存在时返回 False。
        """

        namespace = (
            build_memory_namespace(
                user_id
            )
        )

        normalized_key = (
            normalize_memory_key(
                key
            )
        )

        existing = (
            self.long_term_store.get(
                namespace,
                normalized_key,
            )
        )

        if existing is None:
            return False

        self.long_term_store.delete(
            namespace,
            normalized_key,
        )

        return True

    def _load_long_term_memories(
        self,
        runtime: Runtime[
            LongTermMemoryContext
        ],
    ) -> list[dict[str, Any]]:
        """从运行时 Store 读取用户长期记忆。"""

        user_id = normalize_user_id(
            runtime.context.user_id
        )

        namespace = (
            build_memory_namespace(
                user_id
            )
        )

        if runtime.store is None:
            return []

        items = runtime.store.search(
            namespace,
            limit=(
                self.maximum_loaded_memories
            ),
        )

        memories: list[
            dict[str, Any]
        ] = []

        for item in items:
            value = dict(
                item.value
            )

            content = str(
                value.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                continue

            memories.append(
                {
                    "key": item.key,
                    "content": content,
                    "category": str(
                        value.get(
                            "category",
                            "unknown",
                        )
                    ),
                }
            )

        memories.sort(
            key=lambda current: str(
                current["key"]
            )
        )

        return memories

    def _format_long_term_memories(
        self,
        memories: Sequence[
            dict[str, Any]
        ],
    ) -> str:
        """将长期记忆转换为模型上下文。"""

        lines: list[str] = []

        for memory in memories:
            key = str(
                memory.get(
                    "key",
                    "unknown",
                )
            )

            category = str(
                memory.get(
                    "category",
                    "unknown",
                )
            )

            content = str(
                memory.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                continue

            lines.append(
                f"- [{category}] "
                f"{key}: {content}"
            )

        return "\n".join(
            lines
        ).strip()

    def _build_system_prompt_with_skills(
        self,
    ) -> str:
        """构造包含已加载 Skill 指令的系统提示。

        Skill 发现与加载由 SkillRuntime 负责。

        当某个 Skill 被 load_skill 成功加载后，
        这里会把该 Skill 的完整 SKILL.md 指令
        注入下一次模型决策上下文。

        因此模型会经历：

            Discover
            -> Load
            -> Read instructions
            -> Execute
        """

        base_prompt = str(
            self.system_prompt
        ).strip()

        if self.skill_runtime is None:
            return base_prompt

        loaded_skill_instructions = str(
            self.skill_runtime.render_loaded_instructions()
        ).strip()

        if not loaded_skill_instructions:
            return base_prompt

        return (
            f"{base_prompt}\n\n"
            "以下内容来自当前 Skill Runtime。"
            "只有已经加载的 Skill 才会在这里"
            "提供完整执行说明。\n\n"
            f"{loaded_skill_instructions}"
        )

    def _build_model_input_with_memory(
        self,
        state: PersistentRetrievalGraphState,
        runtime: Runtime[
            LongTermMemoryContext
        ],
    ) -> list[BaseMessage]:
        """构造带长期记忆的模型输入。

        顺序：

            Agent 系统提示
            长期用户记忆
            当前 thread 滚动摘要
            当前 thread 最近详细消息
        """

        model_input: list[
            BaseMessage
        ] = [
            SystemMessage(
                content=(
                    self._build_system_prompt_with_skills()
                )
            )
        ]

        memories = (
            self._load_long_term_memories(
                runtime
            )
        )

        memory_text = (
            self._format_long_term_memories(
                memories
            )
        )

        if memory_text:
            model_input.append(
                SystemMessage(
                    content=(
                        "以下是当前用户的"
                        "跨会话长期记忆。"
                        "这些信息可用于理解用户"
                        "的稳定偏好和背景，"
                        "但不能作为知识库事实"
                        "或本轮检索证据："
                        "\n\n"
                        f"{memory_text}"
                    )
                )
            )

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
                        "更早会话的滚动摘要。"
                        "它只用于理解本线程历史，"
                        "不能替代本轮检索证据："
                        "\n\n"
                        f"{summary}"
                    )
                )
            )

        model_input.extend(
            list(
                state["messages"]
            )
        )

        return model_input

    def _build_skill_runtime_prompt_for_context(
        self,
    ) -> str:
        """只返回已加载 Skill 的执行说明，供 Assembler 单独计费。"""

        if self.skill_runtime is None:
            return ""

        loaded = str(
            self.skill_runtime
            .render_loaded_instructions()
        ).strip()

        if not loaded:
            return ""

        return (
            "以下内容来自当前 Skill Runtime。"
            "只有已经加载的 Skill 才会提供完整执行说明。\n\n"
            f"{loaded}"
        )

    def _build_context_pipeline_input_with_memory(
        self,
        state: PersistentRetrievalGraphState,
        runtime: Runtime[
            LongTermMemoryContext
        ],
        *,
        active_tools: Sequence[
            BaseTool
        ],
        finalize_instruction: str | None = None,
    ) -> tuple[
        list[
            BaseMessage
        ],
        dict[
            str,
            Any,
        ],
    ]:
        """用新 Context Pipeline 构造一次真实模型输入。"""

        raw_plan = state.get(
            "context_plan"
        )

        if not isinstance(
            raw_plan,
            dict,
        ):
            raise ValueError(
                "Context Pipeline 已启用但 State 中缺少 context_plan。"
            )

        plan = ContextPlan.model_validate(
            raw_plan
        )

        memory_text = ""

        if (
            plan.long_term_memory_required
        ):
            memories = (
                self._load_long_term_memories(
                    runtime
                )
            )

            memory_text = (
                self._format_long_term_memories(
                    memories
                )
            )

        built = (
            self.context_pipeline
            .build_for_model(
                state=dict(
                    state
                ),
                base_system_prompt=(
                    self.system_prompt
                ),
                skill_runtime_prompt=(
                    self._build_skill_runtime_prompt_for_context()
                ),
                long_term_memory_text=(
                    memory_text
                ),
                thread_summary=str(
                    state.get(
                        "summary",
                        "",
                    )
                    or ""
                ),
                active_tools=(
                    active_tools
                ),
                finalize_instruction=(
                    finalize_instruction
                ),
            )
        )

        return (
            built.messages,
            built.diagnostics,
        )

    def _apply_context_tool_exposure(
        self,
        *,
        state: PersistentRetrievalGraphState,
        active_tools: Sequence[
            BaseTool
        ],
    ) -> ToolExposureDecision:
        """根据本轮 ContextPlan 重新绑定 LLM Tool Schemas。

        注意：
        - Secure ToolNode 仍保留完整 Active Tools；
        -这里只改变 LLM 看得到的 Tool Schemas；
        - Budget 也应只计算 exposed_tools。
        """

        pipeline_enabled = bool(
            state.get(
                "context_pipeline_enabled",
                False,
            )
            and
            self.context_pipeline_enabled
        )

        raw_plan = state.get(
            "context_plan",
            {},
        )

        decision = (
            select_tools_for_context(
                active_tools=(
                    active_tools
                ),
                context_pipeline_enabled=(
                    pipeline_enabled
                ),
                context_plan=(
                    raw_plan
                    if isinstance(
                        raw_plan,
                        dict,
                    )
                    else {}
                ),
                policy_resolver=(
                    getattr(
                        self,
                        "context_tool_policy_resolver",
                        None,
                    )
                ),
            )
        )

        # _refresh_tool_bindings() 已经让 ToolNode/SecureToolNode
        # 看到完整 Active Tools。
        #
        # 这里仅重绑模型侧 schema。
        self.tool_enabled_model = (
            self._bind_tools(
                decision.exposed_tools
            )
            if decision.exposed_tools
            else self.chat_model
        )

        return decision

    def _model_node(
        self,
        state: PersistentRetrievalGraphState,
        runtime: Runtime[
            LongTermMemoryContext
        ],
    ) -> dict[str, Any]:
        """调用带工具能力的 Agent 模型。"""

        event_thread_id = str(state.get("working_memory_thread_id", "") or "")

        active_tools = (
            self._refresh_tool_bindings()
        )

        exposure = (
            self._apply_context_tool_exposure(
                state=state,
                active_tools=(
                    active_tools
                ),
            )
        )

        exposed_tools = list(
            exposure.exposed_tools
        )

        pipeline_enabled = bool(
            state.get(
                "context_pipeline_enabled",
                False,
            )
            and self.context_pipeline_enabled
        )

        if pipeline_enabled:
            (
                model_input,
                context_pipeline_trace,
            ) = (
                self._build_context_pipeline_input_with_memory(
                    state,
                    runtime,
                    # Phase 7C:
                    # Tool Schema Budget 只计算真正暴露给 LLM
                    # 的 schemas，而不是完整 Runtime Active Tools。
                    active_tools=(
                        exposed_tools
                    ),
                )
            )
        else:
            model_input = (
                self._build_model_input_with_memory(
                    state,
                    runtime,
                )
            )

            context_pipeline_trace = {
                "enabled": False,
                "fallback_reason": str(
                    state.get(
                        "context_pipeline_fallback_reason",
                        "",
                    )
                    or ""
                ),
            }

        context_audit = (
            audit_model_input(
                model_input
            )
        )

        start_time = (
            time.perf_counter()
        )

        llm_call_index = int(state.get("turn_llm_calls", 0)) + 1
        emit_runtime_event(
            "model_started",
            {
                "node": "agent",
                "thread_id": event_thread_id,
                "llm_call_index": llm_call_index,
                "exposed_tool_count": len(exposed_tools),
                "message": f"Agent 模型开始第 {llm_call_index} 次决策。",
            },
        )
        try:
            response = self.tool_enabled_model.invoke(model_input)
        except Exception as error:
            emit_runtime_event(
                "model_failed",
                {
                    "node": "agent",
                    "thread_id": event_thread_id,
                    "llm_call_index": llm_call_index,
                    "error_type": type(error).__name__,
                    "message": "Agent 模型调用失败。",
                },
            )
            raise

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "模型节点必须返回 "
                "AIMessage，"
                f"实际类型："
                f"{type(response)!r}"
            )

        tool_calls = (
            normalize_tool_calls(
                response
            )
        )

        emit_runtime_event(
            "model_completed",
            {
                "node": "agent",
                "thread_id": event_thread_id,
                "llm_call_index": llm_call_index,
                "latency_ms": round(latency_ms, 2),
                "tool_call_count": len(tool_calls),
                "tool_names": [str(call.get("name", "")) for call in tool_calls],
                "message": (
                    f"Agent 模型第 {llm_call_index} 次决策完成，"
                    f"产生 {len(tool_calls)} 个 Tool Call。"
                ),
            },
        )

        current_llm_calls = int(
            state.get(
                "turn_llm_calls",
                0,
            )
        ) + 1

        model_trace = list(
            state.get(
                "model_trace",
                [],
            )
        )

        model_trace.append(
            {
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
                "active_tool_names": [
                    str(
                        current_tool.name
                    )
                    for current_tool
                    in active_tools
                ],
                "exposed_tool_names": list(
                    exposure
                    .exposed_tool_names
                ),
                "hidden_tool_names": list(
                    exposure
                    .hidden_tool_names
                ),
                "tool_exposure": {
                    "filtering_applied": (
                        exposure
                        .filtering_applied
                    ),
                    "retrieval_allowed": (
                        exposure
                        .retrieval_allowed
                    ),
                    "reason": (
                        exposure.reason
                    ),
                },
                "context_audit": (
                    context_audit
                ),
                "context_pipeline": (
                    context_pipeline_trace
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
            "messages": [response],
            "turn_llm_calls": (
                current_llm_calls
            ),
            "model_trace": (
                model_trace
            ),
            "context_pipeline": (
                context_pipeline_trace
            ),
        }

    def _finalize_node(
        self,
        state: PersistentRetrievalGraphState,
        runtime: Runtime[
            LongTermMemoryContext
        ],
    ) -> dict[str, Any]:
        """达到最大步骤后停止工具调用。"""

        event_thread_id = str(state.get("working_memory_thread_id", "") or "")

        finalize_instruction = (
            "已经达到本轮工具决策步骤上限。"
            "请停止继续调用工具，"
            "仅根据已经获得的工具结果生成最终回答。"
            "资料不足时请明确说明。"
        )

        pipeline_enabled = bool(
            state.get(
                "context_pipeline_enabled",
                False,
            )
            and self.context_pipeline_enabled
        )

        if pipeline_enabled:
            (
                model_input,
                context_pipeline_trace,
            ) = (
                self._build_context_pipeline_input_with_memory(
                    state,
                    runtime,
                    active_tools=[],
                    finalize_instruction=(
                        finalize_instruction
                    ),
                )
            )
        else:
            model_input = (
                self._build_model_input_with_memory(
                    state,
                    runtime,
                )
            )

            model_input.append(
                HumanMessage(
                    content=(
                        finalize_instruction
                    )
                )
            )

            context_pipeline_trace = {
                "enabled": False,
                "fallback_reason": str(
                    state.get(
                        "context_pipeline_fallback_reason",
                        "",
                    )
                    or ""
                ),
            }

        context_audit = (
            audit_model_input(
                model_input
            )
        )

        start_time = (
            time.perf_counter()
        )

        llm_call_index = int(state.get("turn_llm_calls", 0)) + 1
        emit_runtime_event(
            "model_started",
            {
                "node": "finalize",
                "thread_id": event_thread_id,
                "llm_call_index": llm_call_index,
                "message": "Finalizer 模型开始生成最终回答。",
            },
        )
        try:
            response = self.chat_model.invoke(model_input)
        except Exception as error:
            emit_runtime_event(
                "model_failed",
                {
                    "node": "finalize",
                    "thread_id": event_thread_id,
                    "llm_call_index": llm_call_index,
                    "error_type": type(error).__name__,
                    "message": "Finalizer 模型调用失败。",
                },
            )
            raise

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "finalize 节点必须返回 "
                "AIMessage，"
                f"实际类型："
                f"{type(response)!r}"
            )

        emit_runtime_event(
            "model_completed",
            {
                "node": "finalize",
                "thread_id": event_thread_id,
                "llm_call_index": llm_call_index,
                "latency_ms": round(latency_ms, 2),
                "tool_call_count": 0,
                "message": "Finalizer 模型已生成最终回答。",
            },
        )

        current_llm_calls = int(
            state.get(
                "turn_llm_calls",
                0,
            )
        ) + 1

        model_trace = list(
            state.get(
                "model_trace",
                [],
            )
        )

        model_trace.append(
            {
                "node": "finalize",
                "llm_call_index": (
                    current_llm_calls
                ),
                "has_tool_calls": False,
                "tool_call_count": 0,
                "tool_calls": [],
                "context_audit": (
                    context_audit
                ),
                "context_pipeline": (
                    context_pipeline_trace
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
            "messages": [response],
            "turn_llm_calls": (
                current_llm_calls
            ),
            "stopped_by_max_steps": True,
            "model_trace": (
                model_trace
            ),
            "context_pipeline": (
                context_pipeline_trace
            ),
        }

    def _build_graph(
        self,
    ) -> Any:
        """构建带 Runtime Context 和 Store 的图。"""

        builder = StateGraph(
            PersistentRetrievalGraphState,
            context_schema=(
                LongTermMemoryContext
            ),
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

        builder.add_node(
            "memory_manager",
            self._memory_manager_node,
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
            ),
            store=(
                self.long_term_store
            ),
        )

    def run(
        self,
        question: str,
        *,
        thread_id: str,
        user_id: str,
    ) -> PersistentLangGraphResult:
        """在指定用户和线程下执行一轮问答。"""

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

        normalized_user_id = (
            normalize_user_id(
                user_id
            )
        )

        config = self._build_config(
            normalized_thread_id
        )

        current_human_message_id = (
            "human-"
            + uuid.uuid4().hex
        )

        current_human = HumanMessage(
            content=(
                normalized_question
            ),
            id=(
                current_human_message_id
            ),
        )

        context_pipeline_enabled = bool(
            self.context_pipeline_enabled
        )

        context_plan: dict[
            str,
            Any,
        ] = {}

        context_retrieval: dict[
            str,
            Any,
        ] = {}

        context_planner_trace: dict[
            str,
            Any,
        ] = {}

        context_pipeline_fallback_reason = ""

        if context_pipeline_enabled:
            context_started_at = time.perf_counter()
            emit_runtime_event(
                "context_pipeline_started",
                {
                    "thread_id": normalized_thread_id,
                    "message": "Context Pipeline 开始规划并恢复会话上下文。",
                },
            )
            try:
                prepared_context = (
                    self.context_pipeline
                    .prepare_new_turn(
                        question=(
                            normalized_question
                        ),
                        thread_id=(
                            normalized_thread_id
                        ),
                        thread_summary=(
                            self.get_thread_summary(
                                normalized_thread_id
                            )
                        ),
                    )
                )

                context_plan = dict(
                    prepared_context
                    .context_plan
                )

                context_retrieval = dict(
                    prepared_context
                    .context_retrieval
                )

                context_planner_trace = dict(
                    prepared_context
                    .context_planner_trace
                )

                emit_runtime_event(
                    "context_pipeline_completed",
                    {
                        "thread_id": normalized_thread_id,
                        "latency_ms": round(
                            (time.perf_counter() - context_started_at) * 1000.0,
                            2,
                        ),
                        "retrieval_keys": sorted(context_retrieval.keys()),
                        "message": "Context Pipeline 已完成上下文规划与恢复。",
                    },
                )

            except Exception as exc:
                # Context Pipeline 是增强层。
                # 初次正式接入阶段如果 Planner / Event Store
                # 自身异常，保留旧 Context Path 作为可观测 fallback，
                # 不让整套 Agent 因增强层故障直接不可用。
                context_pipeline_enabled = False

                context_pipeline_fallback_reason = (
                    type(
                        exc
                    ).__name__
                    + ": "
                    + str(
                        exc
                    )
                )

                context_planner_trace = {
                    "error": (
                        context_pipeline_fallback_reason
                    ),
                }

                emit_runtime_event(
                    "context_pipeline_failed",
                    {
                        "thread_id": normalized_thread_id,
                        "error_type": type(exc).__name__,
                        "message": "Context Pipeline 失败，已回退到基础上下文路径。",
                    },
                )

        input_state = {
            "messages": [
                current_human
            ],
            "turn_llm_calls": 0,
            "turn_tool_calls": 0,
            "turn_summary_calls": 0,
            "summary_updated": False,
            "summarized_turns_this_run": 0,
            "stopped_by_max_steps": False,
            "model_trace": [],
            "tool_trace": [],

            # ----------------------------------------------
            # Working Memory Phase 7E-2
            # ----------------------------------------------
            "working_memory_audit": {},
            "working_memory_thread_id": (
                normalized_thread_id
            ),

            # ----------------------------------------------
            # Context Pipeline Phase 7A
            # ----------------------------------------------
            "context_pipeline_enabled": (
                context_pipeline_enabled
            ),
            "context_current_human_message_id": (
                current_human_message_id
            ),
            "context_current_turn_id": (
                "turn:"
                + current_human_message_id
            ),
            "context_plan": (
                context_plan
            ),
            "context_retrieval": (
                context_retrieval
            ),
            "context_planner_trace": (
                context_planner_trace
            ),
            # _model_node / _finalize_node 会把本轮实际使用的
            # Context Pipeline Trace 写回这个 State channel。
            "context_pipeline": {},
            "context_pipeline_fallback_reason": (
                context_pipeline_fallback_reason
            ),
        }

        total_start = (
            time.perf_counter()
        )

        final_state = self.graph.invoke(
            input_state,
            config=config,
            context=LongTermMemoryContext(
                user_id=(
                    normalized_user_id
                )
            ),
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
                "图执行完成后"
                "没有返回消息。"
            )

        answer = extract_answer_text(
            messages[-1]
        ).strip()

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
            answer=answer,
            messages=messages,

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
