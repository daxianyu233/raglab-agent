"""自动提取长期记忆的 LangGraph Agent。

长期记忆写入机制：

1. 用户可通过 remember() 明确写入；
2. 滚动摘要触发时，自动整理即将移出的用户消息；
3. 会话切换或退出时，可调用 flush_long_term_memory()
   对尚未整理的最近消息进行保底处理。

注意：

- 只从 HumanMessage 中提取用户事实和偏好；
- 不把助手回答的知识内容保存为用户记忆；
- 已处理的用户消息通过内容哈希去重；
- 当前 Store 仍然是 InMemoryStore，程序退出后会丢失。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.runtime import Runtime

from raglab.agent.long_term_memory_agent import (
    LongTermMemoryContext,
    LongTermMemoryRetrievalAgent,
    build_memory_namespace,
    normalize_user_id,
)
from raglab.agent.persistent_langgraph_agent import (
    PersistentRetrievalGraphState,
)
from raglab.generation.rag_chain import (
    extract_answer_text,
    extract_response_metadata,
    extract_usage_metadata,
)


LONG_TERM_MEMORY_EXTRACTION_PROMPT = """你负责从用户对话中提取跨会话长期记忆。

长期记忆是：
即使用户切换到另一个聊天线程，仍然有助于后续服务用户的稳定信息。

只允许提取以下类别：

1. preference
   用户长期表达偏好，例如回答格式、代码交付方式、沟通方式。

2. environment
   用户长期使用的开发环境、操作系统、主要工具或硬件。

3. profile
   用户明确提供的稳定个人背景，例如所在城市、就业状态、专业方向。

4. project
   用户长期进行的项目、已确定的稳定技术方案或持续目标。

5. constraint
   跨会话仍然成立的明确约束，例如预算、技术限制、不可接受事项。

不得保存：

1. 普通知识问答内容；
2. 助手给出的技术解释；
3. 临时测试代号；
4. 一次性的格式转换要求；
5. 当前轮工具调用结果；
6. BM25 分数、Chunk 内容和临时引用编号；
7. 未经用户明确说明的推测；
8. 仅对当前聊天线程有意义的信息。

已有记忆可能需要更新或删除。

请只输出合法 JSON，不要输出 Markdown 代码块，不要解释。

格式：

{
  "operations": [
    {
      "action": "upsert",
      "key": "snake_case_key",
      "category": "preference",
      "content": "需要保存的完整中文事实",
      "confidence": 0.95,
      "reason": "为什么该信息跨会话仍然有效"
    },
    {
      "action": "delete",
      "key": "需要删除的已有记忆键",
      "category": "preference",
      "content": "",
      "confidence": 0.95,
      "reason": "用户明确否定或修改了旧信息"
    }
  ]
}

没有值得保存的信息时输出：

{
  "operations": []
}
"""


class AutomaticMemoryGraphState(
    PersistentRetrievalGraphState
):
    """增加长期记忆自动整理状态。"""

    long_term_extracted_message_hashes: list[str]

    last_auto_memory_report: dict[
        str,
        Any,
    ]


def normalize_message_text(
    text: str,
) -> str:
    """规范化用户消息，便于计算稳定哈希。"""

    normalized = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    return normalized


def build_message_hash(
    text: str,
) -> str:
    """计算用户消息内容哈希。"""

    normalized = normalize_message_text(
        text
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def extract_json_object(
    text: str,
) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象。"""

    normalized = str(text).strip()

    normalized = re.sub(
        r"^```(?:json)?\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"\s*```$",
        "",
        normalized,
    ).strip()

    try:
        parsed = json.loads(
            normalized
        )

    except json.JSONDecodeError:
        start_index = normalized.find(
            "{"
        )

        end_index = normalized.rfind(
            "}"
        )

        if (
            start_index < 0
            or end_index <= start_index
        ):
            raise ValueError(
                "长期记忆模型没有返回合法 JSON。"
            )

        parsed = json.loads(
            normalized[
                start_index:
                end_index + 1
            ]
        )

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "长期记忆模型返回值必须是 JSON 对象。"
        )

    return parsed


def normalize_memory_operation(
    operation: Any,
) -> dict[str, Any] | None:
    """验证并规范化一条记忆操作。"""

    if not isinstance(
        operation,
        dict,
    ):
        return None

    action = str(
        operation.get(
            "action",
            "",
        )
    ).strip().lower()

    if action not in {
        "upsert",
        "delete",
    }:
        return None

    raw_key = str(
        operation.get(
            "key",
            "",
        )
    ).strip().lower()

    key = re.sub(
        r"[^a-z0-9_]+",
        "_",
        raw_key,
    ).strip("_")

    if not key:
        return None

    category = str(
        operation.get(
            "category",
            "unknown",
        )
    ).strip().lower()

    allowed_categories = {
        "preference",
        "environment",
        "profile",
        "project",
        "constraint",
    }

    if category not in allowed_categories:
        return None

    content = str(
        operation.get(
            "content",
            "",
        )
    ).strip()

    if (
        action == "upsert"
        and not content
    ):
        return None

    try:
        confidence = float(
            operation.get(
                "confidence",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        confidence = 0.0

    confidence = max(
        0.0,
        min(
            confidence,
            1.0,
        ),
    )

    reason = str(
        operation.get(
            "reason",
            "",
        )
    ).strip()

    return {
        "action": action,
        "key": key,
        "category": category,
        "content": content,
        "confidence": confidence,
        "reason": reason,
    }


class AutomaticLongTermMemoryAgent(
    LongTermMemoryRetrievalAgent
):
    """带批量自动长期记忆提取的 Agent。"""

    def __init__(
        self,
        *,
        minimum_memory_confidence: float = 0.80,
        **kwargs: Any,
    ) -> None:
        """初始化 Agent。

        minimum_memory_confidence:
            只有达到该置信度的记忆操作
            才会真正写入 Store。
        """

        if not (
            0.0
            <= minimum_memory_confidence
            <= 1.0
        ):
            raise ValueError(
                "minimum_memory_confidence "
                "必须位于 [0, 1]。"
            )

        self.minimum_memory_confidence = float(
            minimum_memory_confidence
        )

        super().__init__(
            **kwargs
        )

    def _build_graph(
        self,
    ) -> Any:
        """构建支持自动记忆提取的图。"""

        builder = StateGraph(
            AutomaticMemoryGraphState,
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

    def _get_existing_memories(
        self,
        *,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """读取用户已有长期记忆。"""

        return self.list_memories(
            user_id=user_id
        )

    def _collect_unprocessed_user_messages(
        self,
        *,
        messages: Sequence[
            BaseMessage
        ],
        processed_hashes: set[str],
    ) -> tuple[
        list[dict[str, str]],
        list[str],
    ]:
        """提取尚未处理的用户消息。"""

        candidates: list[
            dict[str, str]
        ] = []

        new_hashes: list[str] = []

        for message in messages:
            if not isinstance(
                message,
                HumanMessage,
            ):
                continue

            text = normalize_message_text(
                str(message.content)
            )

            if not text:
                continue

            message_hash = (
                build_message_hash(
                    text
                )
            )

            if (
                message_hash
                in processed_hashes
            ):
                continue

            candidates.append(
                {
                    "hash": message_hash,
                    "content": text,
                }
            )

            new_hashes.append(
                message_hash
            )

        return candidates, new_hashes

    def _call_memory_extractor(
        self,
        *,
        user_id: str,
        candidates: Sequence[
            dict[str, str]
        ],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        """调用模型生成结构化记忆操作。"""

        existing_memories = (
            self._get_existing_memories(
                user_id=user_id
            )
        )

        extraction_input = {
            "user_id": user_id,
            "existing_memories": (
                existing_memories
            ),
            "new_user_messages": list(
                candidates
            ),
        }

        start_time = (
            time.perf_counter()
        )

        response = self.chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        LONG_TERM_MEMORY_EXTRACTION_PROMPT
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        extraction_input,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
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
                "长期记忆提取模型必须返回 "
                "AIMessage。"
            )

        raw_text = extract_answer_text(
            response
        ).strip()

        parsed = extract_json_object(
            raw_text
        )

        raw_operations = parsed.get(
            "operations",
            [],
        )

        if not isinstance(
            raw_operations,
            list,
        ):
            raise ValueError(
                "operations 必须是列表。"
            )

        operations: list[
            dict[str, Any]
        ] = []

        for raw_operation in (
            raw_operations
        ):
            normalized = (
                normalize_memory_operation(
                    raw_operation
                )
            )

            if normalized is None:
                continue

            operations.append(
                normalized
            )

        trace = {
            "node": (
                "long_term_memory_extractor"
            ),
            "candidate_message_count": len(
                candidates
            ),
            "operation_count": len(
                operations
            ),
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

        return operations, trace

    def _apply_memory_operations(
        self,
        *,
        user_id: str,
        operations: Sequence[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        """应用模型生成的记忆操作。"""

        applied: list[
            dict[str, Any]
        ] = []

        namespace = (
            build_memory_namespace(
                user_id
            )
        )

        for operation in operations:
            confidence = float(
                operation.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                confidence
                < self.minimum_memory_confidence
            ):
                continue

            action = str(
                operation["action"]
            )

            key = str(
                operation["key"]
            )

            if action == "upsert":
                value = {
                    "content": (
                        operation["content"]
                    ),
                    "category": (
                        operation["category"]
                    ),
                    "source": (
                        "automatic_extraction"
                    ),
                    "confidence": (
                        confidence
                    ),
                    "reason": (
                        operation["reason"]
                    ),
                }

                self.long_term_store.put(
                    namespace,
                    key,
                    value,
                )

                applied.append(
                    {
                        **operation,
                        "result": "upserted",
                    }
                )

                continue

            if action == "delete":
                existing = (
                    self.long_term_store.get(
                        namespace,
                        key,
                    )
                )

                if existing is None:
                    continue

                self.long_term_store.delete(
                    namespace,
                    key,
                )

                applied.append(
                    {
                        **operation,
                        "result": "deleted",
                    }
                )

        return applied

    def _extract_and_apply(
        self,
        *,
        state: dict[str, Any],
        messages: Sequence[
            BaseMessage
        ],
        user_id: str,
        trigger: str,
    ) -> dict[str, Any]:
        """提取并应用尚未处理的长期记忆。"""

        processed_hashes = set(
            state.get(
                "long_term_extracted_message_hashes",
                [],
            )
            or []
        )

        candidates, new_hashes = (
            self._collect_unprocessed_user_messages(
                messages=messages,
                processed_hashes=(
                    processed_hashes
                ),
            )
        )

        if not candidates:
            return {
                "called": False,
                "report": {
                    "trigger": trigger,
                    "candidate_message_count": 0,
                    "operation_count": 0,
                    "applied_count": 0,
                    "status": "nothing_to_process",
                },
                "processed_hashes": list(
                    processed_hashes
                ),
                "trace": None,
            }

        operations, trace = (
            self._call_memory_extractor(
                user_id=user_id,
                candidates=candidates,
            )
        )

        applied = (
            self._apply_memory_operations(
                user_id=user_id,
                operations=operations,
            )
        )

        updated_hashes = (
            processed_hashes
            | set(new_hashes)
        )

        report = {
            "trigger": trigger,
            "candidate_message_count": len(
                candidates
            ),
            "operation_count": len(
                operations
            ),
            "applied_count": len(
                applied
            ),
            "operations": operations,
            "applied_operations": applied,
            "status": "completed",
        }

        return {
            "called": True,
            "report": report,
            "processed_hashes": sorted(
                updated_hashes
            ),
            "trace": trace,
        }

    def _memory_manager_node(
        self,
        state: AutomaticMemoryGraphState,
        runtime: Runtime[
            LongTermMemoryContext
        ],
    ) -> dict[str, Any]:
        """滚动摘要时同时整理长期记忆。"""

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        human_positions = [
            index
            for index, message in enumerate(
                messages
            )
            if isinstance(
                message,
                HumanMessage,
            )
        ]

        current_turn_count = len(
            human_positions
        )

        if (
            current_turn_count
            < self.summarize_trigger_turns
        ):
            return super()._memory_manager_node(
                state
            )

        keep_start_index = (
            human_positions[
                -self.keep_recent_turns
            ]
        )

        messages_to_summarize = (
            messages[
                :keep_start_index
            ]
        )

        user_id = normalize_user_id(
            runtime.context.user_id
        )

        try:
            extraction_result = (
                self._extract_and_apply(
                    state=state,
                    messages=(
                        messages_to_summarize
                    ),
                    user_id=user_id,
                    trigger=(
                        "rolling_summary"
                    ),
                )
            )

        except Exception as error:
            extraction_result = {
                "called": False,
                "processed_hashes": list(
                    state.get(
                        "long_term_extracted_message_hashes",
                        [],
                    )
                    or []
                ),
                "trace": None,
                "report": {
                    "trigger": (
                        "rolling_summary"
                    ),
                    "status": "failed",
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
            }

        summary_update = (
            super()._memory_manager_node(
                state
            )
        )

        summary_update[
            "long_term_extracted_message_hashes"
        ] = extraction_result[
            "processed_hashes"
        ]

        summary_update[
            "last_auto_memory_report"
        ] = extraction_result[
            "report"
        ]

        if extraction_result["called"]:
            original_trace = list(
                state.get(
                    "model_trace",
                    [],
                )
            )

            summary_trace = list(
                summary_update.get(
                    "model_trace",
                    original_trace,
                )
            )

            new_summary_trace = (
                summary_trace[
                    len(original_trace):
                ]
            )

            extraction_trace = (
                extraction_result[
                    "trace"
                ]
            )

            summary_update[
                "model_trace"
            ] = [
                *original_trace,
                extraction_trace,
                *new_summary_trace,
            ]

            summary_update[
                "turn_llm_calls"
            ] = (
                int(
                    summary_update.get(
                        "turn_llm_calls",
                        state.get(
                            "turn_llm_calls",
                            0,
                        ),
                    )
                )
                + 1
            )

        return summary_update

    def flush_long_term_memory(
        self,
        *,
        thread_id: str,
        user_id: str,
        trigger: str = "session_flush",
    ) -> dict[str, Any]:
        """保底整理当前线程尚未处理的消息。

        可在：

        - /new
        - /exit
        - 用户主动结束会话
        - 定时整理任务

        中调用。
        """

        normalized_user_id = (
            normalize_user_id(
                user_id
            )
        )

        state = self.get_thread_state(
            thread_id
        )

        messages = list(
            state.get(
                "messages",
                [],
            )
            or []
        )

        if not messages:
            return {
                "trigger": trigger,
                "status": "empty_thread",
                "candidate_message_count": 0,
                "applied_count": 0,
            }

        try:
            result = self._extract_and_apply(
                state=state,
                messages=messages,
                user_id=(
                    normalized_user_id
                ),
                trigger=trigger,
            )

        except Exception as error:
            return {
                "trigger": trigger,
                "status": "failed",
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

        if result["called"]:
            self.graph.update_state(
                self._build_config(
                    thread_id
                ),
                {
                    "long_term_extracted_message_hashes": (
                        result[
                            "processed_hashes"
                        ]
                    ),
                    "last_auto_memory_report": (
                        result["report"]
                    ),
                },
                as_node="memory_manager",
            )

        return result["report"]

    def get_last_auto_memory_report(
        self,
        *,
        thread_id: str,
    ) -> dict[str, Any]:
        """读取当前线程最近一次自动整理报告。"""

        state = self.get_thread_state(
            thread_id
        )

        report = state.get(
            "last_auto_memory_report",
            {},
        )

        if not isinstance(
            report,
            dict,
        ):
            return {}

        return dict(report)