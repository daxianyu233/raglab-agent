"""统一的 RAG 问答链。

完整流程：

用户问题
→ Retriever 检索候选文档
→ 统一检索结果格式
→ 构造受控上下文
→ 构造 RAG Prompt
→ 调用聊天模型
→ 返回答案、来源和耗时信息

本模块不绑定具体模型提供商。

只要聊天模型对象实现：

    model.invoke(messages)

即可接入，例如：

1. OpenAI Chat Model；
2. Ollama 本地模型；
3. Hugging Face 模型；
4. 其他 LangChain Chat Model；
5. 自定义模型适配器。
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
)

from raglab.generation.context_builder import (
    BuiltContext,
    ContextReference,
    build_context,
)
from raglab.generation.prompt_builder import (
    RAGPrompt,
    build_rag_prompt,
)


class ChatModelProtocol(Protocol):
    """聊天模型需要满足的最小接口。"""

    def invoke(
        self,
        input: Any,
        **kwargs: Any,
    ) -> Any:
        """调用聊天模型。"""


RetrieverFunction = Callable[
    ...,
    Sequence[Any],
]


@dataclass(frozen=True)
class RetrievalResult:
    """统一后的单条检索结果。"""

    document: Document
    score: float | None
    original_rank: int


@dataclass(frozen=True)
class RAGAnswer:
    """一次完整 RAG 问答的输出。"""

    question: str
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

    retrieval_latency_ms: float
    context_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    model_response: Any
    response_metadata: dict[str, Any]
    usage_metadata: dict[str, Any]


def safe_float(
    value: Any,
) -> float | None:
    """尝试将值转换为浮点数。"""

    if value is None:
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def normalize_retrieval_item(
    item: Any,
    original_rank: int,
) -> RetrievalResult:
    """将不同检索器返回的数据统一为 RetrievalResult。

    支持以下形式：

    1. Document
    2. tuple[Document, score]
    3. dict
    4. 具有 document 属性的对象
    """

    if isinstance(
        item,
        Document,
    ):
        return RetrievalResult(
            document=item,
            score=None,
            original_rank=original_rank,
        )

    if (
        isinstance(item, tuple)
        and len(item) >= 1
        and isinstance(item[0], Document)
    ):
        score = (
            safe_float(item[1])
            if len(item) >= 2
            else None
        )

        return RetrievalResult(
            document=item[0],
            score=score,
            original_rank=original_rank,
        )

    if isinstance(
        item,
        dict,
    ):
        document = (
            item.get("document")
            or item.get("doc")
        )

        if not isinstance(
            document,
            Document,
        ):
            raise TypeError(
                "字典形式的检索结果缺少有效 Document："
                f"{item}"
            )

        score = safe_float(
            item.get(
                "score",
                item.get(
                    "distance"
                ),
            )
        )

        return RetrievalResult(
            document=document,
            score=score,
            original_rank=original_rank,
        )

    document = getattr(
        item,
        "document",
        None,
    )

    if isinstance(
        document,
        Document,
    ):
        score = safe_float(
            getattr(
                item,
                "score",
                None,
            )
        )

        return RetrievalResult(
            document=document,
            score=score,
            original_rank=original_rank,
        )

    raise TypeError(
        "无法识别检索结果格式："
        f"type={type(item)!r}, "
        f"value={item!r}"
    )


def normalize_retrieval_results(
    results: Sequence[Any],
) -> list[RetrievalResult]:
    """统一一组检索结果。"""

    normalized: list[
        RetrievalResult
    ] = []

    for rank, item in enumerate(
        results,
        start=1,
    ):
        normalized.append(
            normalize_retrieval_item(
                item,
                original_rank=rank,
            )
        )

    return normalized


def call_retriever(
    retriever: RetrieverFunction,
    question: str,
    top_k: int,
) -> Sequence[Any]:
    """调用不同签名的检索函数。

    优先支持：

        search(query=question, top_k=top_k)

    同时兼容：

        search(question, top_k)
        retrieve(question, top_k=top_k)
        similarity_search(question, k=top_k)
    """

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    try:
        signature = inspect.signature(
            retriever
        )

        parameter_names = set(
            signature.parameters
        )

    except (
        TypeError,
        ValueError,
    ):
        parameter_names = set()

    if (
        "query" in parameter_names
        and "top_k" in parameter_names
    ):
        return retriever(
            query=question,
            top_k=top_k,
        )

    if (
        "query" in parameter_names
        and "k" in parameter_names
    ):
        return retriever(
            query=question,
            k=top_k,
        )

    if "top_k" in parameter_names:
        return retriever(
            question,
            top_k=top_k,
        )

    if "k" in parameter_names:
        return retriever(
            question,
            k=top_k,
        )

    try:
        return retriever(
            question,
            top_k,
        )

    except TypeError:
        return retriever(
            question
        )


def normalize_message_content(
    content: Any,
) -> str:
    """将聊天模型输出转换成普通字符串。

    LangChain 模型的 message.content 可能是：

    1. str
    2. list[str]
    3. list[dict]
    """

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        list,
    ):
        text_parts: list[str] = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                text_parts.append(item)

            elif isinstance(
                item,
                dict,
            ):
                text_value = (
                    item.get("text")
                    or item.get("content")
                )

                if text_value is not None:
                    text_parts.append(
                        str(text_value)
                    )

            else:
                text_parts.append(
                    str(item)
                )

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


def extract_answer_text(
    response: Any,
) -> str:
    """从模型响应中提取答案文本。"""

    if isinstance(
        response,
        str,
    ):
        answer = response.strip()

    elif isinstance(
        response,
        BaseMessage,
    ):
        answer = normalize_message_content(
            response.content
        )

    elif hasattr(
        response,
        "content",
    ):
        answer = normalize_message_content(
            getattr(
                response,
                "content",
            )
        )

    else:
        answer = str(
            response
        ).strip()

    if not answer:
        raise RuntimeError(
            "聊天模型返回了空答案。"
        )

    return answer


def extract_response_metadata(
    response: Any,
) -> dict[str, Any]:
    """读取模型响应中的 metadata。"""

    metadata = getattr(
        response,
        "response_metadata",
        None,
    )

    if isinstance(
        metadata,
        dict,
    ):
        return dict(metadata)

    return {}


def extract_usage_metadata(
    response: Any,
) -> dict[str, Any]:
    """读取模型响应中的 Token 使用信息。"""

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    if isinstance(
        usage,
        dict,
    ):
        return dict(usage)

    response_metadata = (
        extract_response_metadata(
            response
        )
    )

    for key in (
        "token_usage",
        "usage",
        "usage_metadata",
    ):
        value = response_metadata.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):
            return dict(value)

    return {}


class RAGChain:
    """连接 Retriever、上下文构造器和聊天模型。"""

    def __init__(
        self,
        *,
        retriever: RetrieverFunction,
        chat_model: ChatModelProtocol,
        retrieval_top_k: int = 5,
        max_documents: int = 5,
        max_context_characters: int = 8000,
        include_metadata: bool = True,
        deduplicate: bool = True,
    ) -> None:
        """初始化 RAG 问答链。

        Parameters
        ----------
        retriever:
            检索函数，例如：

                bm25_index.search
                hybrid_retriever.retrieve

        chat_model:
            实现 invoke(messages) 的聊天模型。

        retrieval_top_k:
            Retriever 最初返回多少个候选。

        max_documents:
            最多将多少个候选放入最终上下文。

        max_context_characters:
            最终上下文最大字符数。

        include_metadata:
            是否在上下文中加入资料编号、标题和页码。

        deduplicate:
            是否在构造上下文前去除重复 Chunk。
        """

        if not callable(
            retriever
        ):
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

        self.include_metadata = bool(
            include_metadata
        )

        self.deduplicate = bool(
            deduplicate
        )

    def retrieve(
        self,
        question: str,
    ) -> list[RetrievalResult]:
        """执行文档检索。"""

        normalized_question = (
            question.strip()
        )

        if not normalized_question:
            raise ValueError(
                "question 不能为空。"
            )

        raw_results = call_retriever(
            retriever=self.retriever,
            question=normalized_question,
            top_k=self.retrieval_top_k,
        )

        if raw_results is None:
            return []

        return normalize_retrieval_results(
            list(raw_results)
        )

    def build_context(
        self,
        retrieval_results: Sequence[
            RetrievalResult
        ],
    ) -> BuiltContext:
        """根据检索结果构造受控上下文。"""

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

    def build_prompt(
        self,
        question: str,
        built_context: BuiltContext,
    ) -> RAGPrompt:
        """构造发送给模型的 Prompt。"""

        return build_rag_prompt(
            question=question,
            built_context=built_context,
        )

    def answer(
        self,
        question: str,
    ) -> RAGAnswer:
        """执行一次完整的 RAG 问答。"""

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

        retrieval_start = (
            time.perf_counter()
        )

        retrieval_results = (
            self.retrieve(
                normalized_question
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
            self.build_context(
                retrieval_results
            )
        )

        rag_prompt = (
            self.build_prompt(
                question=normalized_question,
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
                rag_prompt.messages
            )
        )

        generation_latency_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000.0

        answer_text = (
            extract_answer_text(
                model_response
            )
        )

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        return RAGAnswer(
            question=normalized_question,
            answer=answer_text,

            retrieved_documents=[
                result.document
                for result
                in retrieval_results
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

            model_response=(
                model_response
            ),

            response_metadata=(
                extract_response_metadata(
                    model_response
                )
            ),

            usage_metadata=(
                extract_usage_metadata(
                    model_response
                )
            ),
        )