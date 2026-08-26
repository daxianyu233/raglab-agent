

"""多轮对话 RAG Chain。

在单步 RAG 的基础上增加：

1. 内存中的多轮对话历史；
2. 对含有省略或指代的追问进行独立问题改写；
3. 使用改写后的问题执行知识库检索；
4. 使用原问题、对话历史和检索资料生成回答；
5. 保存每一轮的问题、检索问题和答案。

当前仍然是固定流程，不是 Agent：

用户问题
→ 判断是否需要改写
→ 必要时调用模型改写
→ Retriever 检索
→ 构造上下文
→ 调用模型回答
→ 保存历史
→ 结束本轮
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from raglab.generation.context_builder import (
    BuiltContext,
    ContextReference,
    build_context,
)
from raglab.generation.rag_chain import (
    ChatModelProtocol,
    RetrievalResult,
    call_retriever,
    extract_answer_text,
    extract_response_metadata,
    extract_usage_metadata,
    normalize_retrieval_results,
)


RetrieverFunction = Callable[..., Any]


CONVERSATIONAL_RAG_SYSTEM_PROMPT = """你是一个基于知识库检索资料进行多轮问答的助手。

请严格遵守以下规则：

1. 对话历史只能用于理解用户当前问题中的指代、省略和上下文关系。
2. 事实性答案只能依据本轮提供的“检索资料”。
3. 不得把对话历史中未经检索资料支持的内容当作事实依据。
4. 关键事实后应标注对应资料编号，例如 [资料1]。
5. 一句话同时依据多条资料时，可以标注 [资料1][资料2]。
6. 检索资料不足时，应明确说明“现有检索资料不足以回答该问题”。
7. 不得编造资料中没有出现的名称、数字、步骤、接口或结论。
8. 不要编造不存在的资料编号。
9. 资料之间存在冲突时，应明确指出冲突。
10. 回答应直接、清晰，不要机械复述全部检索资料。"""


QUERY_REWRITE_SYSTEM_PROMPT = """你负责将多轮对话中的当前追问改写成一个独立、完整、适合知识库检索的问题。

要求：

1. 结合对话历史补全“它”“这个”“上述方法”等指代。
2. 保留用户原始意图，不要增加新的问题。
3. 不要回答问题。
4. 不要解释改写过程。
5. 只输出改写后的一个问题。
6. 当前问题本身已经完整时，原样输出。"""


@dataclass(frozen=True)
class ConversationTurn:
    """保存一轮对话。"""

    turn_index: int
    question: str
    retrieval_question: str
    answer: str


@dataclass(frozen=True)
class ConversationalRAGAnswer:
    """一次多轮 RAG 问答的完整结果。"""

    turn_index: int
    question: str
    retrieval_question: str
    query_rewritten: bool
    answer: str

    retrieved_documents: list[Document]
    selected_documents: list[Document]
    references: list[ContextReference]

    context: str
    context_character_count: int
    context_truncated: bool

    retrieval_top_k: int
    retrieved_count: int
    selected_count: int
    history_turn_count: int

    rewrite_latency_ms: float
    retrieval_latency_ms: float
    context_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    rewrite_response: Any | None
    model_response: Any

    rewrite_usage_metadata: dict[str, Any]
    generation_usage_metadata: dict[str, Any]
    response_metadata: dict[str, Any]


class ConversationalRAGChain:
    """带有多轮历史的固定流程 RAG Chain。"""

    VALID_REWRITE_MODES = {
        "auto",
        "always",
        "never",
    }

    FOLLOWUP_HINTS = (
        "它",
        "这个",
        "那个",
        "这种",
        "这些",
        "上述",
        "前面",
        "刚才",
        "该方法",
        "该模型",
        "该系统",
        "该部分",
        "这样",
        "那么",
        "那它",
        "还有呢",
        "为什么呢",
        "继续",
    )

    def __init__(
        self,
        *,
        retriever: RetrieverFunction,
        chat_model: ChatModelProtocol,
        retrieval_top_k: int = 5,
        max_documents: int = 5,
        max_context_characters: int = 8000,
        max_history_turns: int = 6,
        rewrite_mode: str = "auto",
        include_metadata: bool = True,
        deduplicate: bool = True,
    ) -> None:
        """初始化多轮 RAG Chain。

        Parameters
        ----------
        retriever:
            检索函数，例如 bm25_index.search。

        chat_model:
            实现 invoke(messages) 的聊天模型。

        retrieval_top_k:
            每轮检索返回的候选数量。

        max_documents:
            最多放入最终上下文的 Chunk 数量。

        max_context_characters:
            最终检索上下文的最大字符数。

        max_history_turns:
            构造当前 Prompt 时最多保留多少轮历史。

        rewrite_mode:
            追问改写模式：

            auto:
                仅在问题疑似包含指代或省略时改写。

            always:
                存在历史时，每轮都改写。

            never:
                永不改写，直接使用当前问题检索。

        include_metadata:
            是否在上下文中加入来源信息。

        deduplicate:
            是否根据 chunk_id 去重。
        """

        if not callable(retriever):
            raise TypeError(
                "retriever 必须是可调用对象。"
            )

        if not hasattr(
            chat_model,
            "invoke",
        ):
            raise TypeError(
                "chat_model 必须实现 invoke()。"
            )

        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k 必须大于 0。"
            )

        if max_documents <= 0:
            raise ValueError(
                "max_documents 必须大于 0。"
            )

        if max_context_characters <= 0:
            raise ValueError(
                "max_context_characters 必须大于 0。"
            )

        if max_history_turns < 0:
            raise ValueError(
                "max_history_turns 不能小于 0。"
            )

        normalized_rewrite_mode = (
            str(rewrite_mode)
            .strip()
            .lower()
        )

        if normalized_rewrite_mode not in (
            self.VALID_REWRITE_MODES
        ):
            raise ValueError(
                "rewrite_mode 只能是："
                "auto、always 或 never。"
            )

        self.retriever = retriever
        self.chat_model = chat_model

        self.retrieval_top_k = int(
            retrieval_top_k
        )

        self.max_documents = int(
            max_documents
        )

        self.max_context_characters = int(
            max_context_characters
        )

        self.max_history_turns = int(
            max_history_turns
        )

        self.rewrite_mode = (
            normalized_rewrite_mode
        )

        self.include_metadata = bool(
            include_metadata
        )

        self.deduplicate = bool(
            deduplicate
        )

        self.history: list[
            ConversationTurn
        ] = []

    def clear_history(self) -> None:
        """清空当前会话历史。"""

        self.history.clear()

    def get_history(
        self,
    ) -> list[ConversationTurn]:
        """返回当前会话历史副本。"""

        return list(self.history)

    def _recent_history(
        self,
    ) -> list[ConversationTurn]:
        """获取当前 Prompt 使用的最近历史。"""

        if self.max_history_turns == 0:
            return []

        return self.history[
            -self.max_history_turns:
        ]

    def _format_history_text(
        self,
    ) -> str:
        """将最近对话历史转换成文本。"""

        recent_history = (
            self._recent_history()
        )

        if not recent_history:
            return "暂无历史对话。"

        blocks: list[str] = []

        for turn in recent_history:
            blocks.append(
                f"第 {turn.turn_index} 轮\n"
                f"用户：{turn.question}\n"
                f"助手：{turn.answer}"
            )

        return "\n\n".join(blocks)

    def _should_rewrite(
        self,
        question: str,
    ) -> bool:
        """判断当前问题是否需要独立化改写。"""

        if not self.history:
            return False

        if self.rewrite_mode == "never":
            return False

        if self.rewrite_mode == "always":
            return True

        normalized_question = (
            question.strip()
        )

        contains_followup_hint = any(
            hint in normalized_question
            for hint in self.FOLLOWUP_HINTS
        )

        is_short_followup = (
            len(normalized_question) <= 20
            and (
                normalized_question.startswith(
                    (
                        "那",
                        "那么",
                        "还有",
                        "为什么",
                        "怎么",
                        "是否",
                    )
                )
                or normalized_question.endswith(
                    (
                        "呢？",
                        "吗？",
                        "呢",
                        "吗",
                    )
                )
            )
        )

        return (
            contains_followup_hint
            or is_short_followup
        )

    def _rewrite_question(
        self,
        question: str,
    ) -> tuple[
        str,
        bool,
        Any | None,
        float,
        dict[str, Any],
    ]:
        """必要时将当前追问改写为独立检索问题。"""

        if not self._should_rewrite(
            question
        ):
            return (
                question,
                False,
                None,
                0.0,
                {},
            )

        history_text = (
            self._format_history_text()
        )

        user_prompt = (
            "以下是对话历史：\n\n"
            f"{history_text}\n\n"
            "当前用户追问：\n"
            f"{question}\n\n"
            "请将当前追问改写成一个独立、完整、"
            "适合知识库检索的问题。"
        )

        messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    QUERY_REWRITE_SYSTEM_PROMPT
                )
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]

        start_time = time.perf_counter()

        response = self.chat_model.invoke(
            messages
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        rewritten_question = (
            extract_answer_text(
                response
            )
            .strip()
            .strip('"')
            .strip("“”")
        )

        if not rewritten_question:
            return (
                question,
                False,
                response,
                latency_ms,
                extract_usage_metadata(
                    response
                ),
            )

        return (
            rewritten_question,
            rewritten_question != question,
            response,
            latency_ms,
            extract_usage_metadata(
                response
            ),
        )

    def _retrieve(
        self,
        retrieval_question: str,
    ) -> list[RetrievalResult]:
        """执行本轮检索。"""

        raw_results = call_retriever(
            retriever=self.retriever,
            question=retrieval_question,
            top_k=self.retrieval_top_k,
        )

        if raw_results is None:
            return []

        return normalize_retrieval_results(
            list(raw_results)
        )

    def _build_context(
        self,
        retrieval_results: list[
            RetrievalResult
        ],
    ) -> BuiltContext:
        """构造本轮受控上下文。"""

        documents = [
            result.document
            for result in retrieval_results
        ]

        return build_context(
            documents=documents,
            max_documents=(
                self.max_documents
            ),
            max_characters=(
                self.max_context_characters
            ),
            include_metadata=(
                self.include_metadata
            ),
            deduplicate=(
                self.deduplicate
            ),
        )

    def _build_answer_messages(
        self,
        *,
        question: str,
        retrieval_question: str,
        built_context: BuiltContext,
    ) -> list[BaseMessage]:
        """构造本轮回答使用的消息。"""

        history_text = (
            self._format_history_text()
        )

        context_text = (
            built_context.text.strip()
        )

        if context_text:
            context_section = (
                "本轮检索资料：\n\n"
                f"{context_text}"
            )
        else:
            context_section = (
                "本轮没有检索到可用资料。"
            )

        human_prompt = (
            f"最近对话历史：\n\n"
            f"{history_text}\n\n"
            f"当前用户原始问题：\n"
            f"{question}\n\n"
            f"用于知识库检索的独立问题：\n"
            f"{retrieval_question}\n\n"
            f"{context_section}\n\n"
            "请回答当前用户原始问题。\n"
            "事实性内容只能依据本轮检索资料，"
            "并在关键结论后标注资料编号。"
        )

        return [
            SystemMessage(
                content=(
                    CONVERSATIONAL_RAG_SYSTEM_PROMPT
                )
            ),
            HumanMessage(
                content=human_prompt
            ),
        ]

    def answer(
        self,
        question: str,
    ) -> ConversationalRAGAnswer:
        """执行一轮多轮 RAG 问答。"""

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

        total_start = (
            time.perf_counter()
        )

        (
            retrieval_question,
            query_rewritten,
            rewrite_response,
            rewrite_latency_ms,
            rewrite_usage_metadata,
        ) = self._rewrite_question(
            normalized_question
        )

        retrieval_start = (
            time.perf_counter()
        )

        retrieval_results = (
            self._retrieve(
                retrieval_question
            )
        )

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000.0

        context_start = (
            time.perf_counter()
        )

        built_context = (
            self._build_context(
                retrieval_results
            )
        )

        answer_messages = (
            self._build_answer_messages(
                question=normalized_question,
                retrieval_question=(
                    retrieval_question
                ),
                built_context=built_context,
            )
        )

        context_latency_ms = (
            time.perf_counter()
            - context_start
        ) * 1000.0

        generation_start = (
            time.perf_counter()
        )

        model_response = (
            self.chat_model.invoke(
                answer_messages
            )
        )

        generation_latency_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000.0

        answer_text = extract_answer_text(
            model_response
        )

        turn_index = (
            len(self.history) + 1
        )

        self.history.append(
            ConversationTurn(
                turn_index=turn_index,
                question=normalized_question,
                retrieval_question=(
                    retrieval_question
                ),
                answer=answer_text,
            )
        )

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        return ConversationalRAGAnswer(
            turn_index=turn_index,
            question=normalized_question,
            retrieval_question=(
                retrieval_question
            ),
            query_rewritten=query_rewritten,
            answer=answer_text,

            retrieved_documents=[
                result.document
                for result in retrieval_results
            ],

            selected_documents=list(
                built_context.documents
            ),

            references=list(
                built_context.references
            ),

            context=built_context.text,
            context_character_count=(
                built_context.character_count
            ),
            context_truncated=(
                built_context.truncated
            ),

            retrieval_top_k=(
                self.retrieval_top_k
            ),
            retrieved_count=len(
                retrieval_results
            ),
            selected_count=(
                built_context.selected_count
            ),
            history_turn_count=len(
                self.history
            ),

            rewrite_latency_ms=(
                rewrite_latency_ms
            ),
            retrieval_latency_ms=(
                retrieval_latency_ms
            ),
            context_latency_ms=(
                context_latency_ms
            ),
            generation_latency_ms=(
                generation_latency_ms
            ),
            total_latency_ms=(
                total_latency_ms
            ),

            rewrite_response=(
                rewrite_response
            ),
            model_response=(
                model_response
            ),

            rewrite_usage_metadata=(
                rewrite_usage_metadata
            ),
            generation_usage_metadata=(
                extract_usage_metadata(
                    model_response
                )
            ),
            response_metadata=(
                extract_response_metadata(
                    model_response
                )
            ),
        )