from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from langchain_core.documents import Document


# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# 项目模块
# ============================================================

from raglab.intelligence.retriever import (
    IntelligenceRetriever,
    document_to_dict,
)


# ============================================================
# 问答配置
# ============================================================

# 默认使用 Hybrid：
#
# Dense Top-20 + BM25 Top-20
# → RRF 融合
# → Top-K 证据
DEFAULT_RETRIEVAL_MODE = "hybrid"

# 最终提供给 DeepSeek 的证据 Chunk 数。
DEFAULT_TOP_K = 6

# 单个证据最多使用的字符数。
MAX_CHARACTERS_PER_EVIDENCE = 1800

# 全部证据的最大字符数。
MAX_TOTAL_EVIDENCE_CHARACTERS = 9000

# 是否在回答后显示命中的证据摘要。
SHOW_EVIDENCE_SUMMARY = True

# 是否保存每次问答记录。
SAVE_QA_RECORDS = True

# 问答记录目录。
QA_REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "intelligence_qa"
)


# ============================================================
# DeepSeek 配置
# ============================================================

DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
).rstrip("/")

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-v4-flash",
).strip()

DEEPSEEK_TIMEOUT_SECONDS = 180

DEEPSEEK_TEMPERATURE = 0.1

DEEPSEEK_MAX_TOKENS = 1800

# 优先发送关闭思考模式的参数。
#
# 如果当前接口不接受 thinking 参数，
# 程序会自动移除该参数并重试一次。
SEND_THINKING_FIELD = True


# ============================================================
# 回答规则
# ============================================================

NO_EVIDENCE_ANSWER = (
    "当前知识库中没有足够证据回答该问题。"
)

SYSTEM_PROMPT = """
你是一个 GitHub 技术情报问答助手。

你必须严格遵守以下规则：

1. 只能依据用户问题下方提供的“检索证据”回答。
2. 不得使用检索证据之外的项目知识、新闻、常识或猜测。
3. 每个重要事实后必须标注对应证据编号，例如：
   [证据1]
   [证据2]
4. 一句话由多条证据共同支持时，可以标注：
   [证据1][证据3]
5. 不得虚构证据编号。
6. 不得把“可能”“推测”写成确定事实。
7. 当证据只能支持部分结论时，应明确说明证据的边界。
8. 当现有证据不足以回答核心问题时，必须直接回答：
   当前知识库中没有足够证据回答该问题。
9. 不要输出参考文献列表，因为系统会在答案后单独展示证据来源。
10. 使用中文回答，项目名、框架名和技术术语可保留英文。
11. 优先给出直接结论，再给出必要解释。
12. 不要描述你的内部推理过程。
""".strip()


# ============================================================
# 基础工具
# ============================================================


def compact_text(
    value: Any,
) -> str:
    """
    压缩连续空白。
    """
    if value is None:
        return ""

    return " ".join(
        str(
            value
        ).split()
    ).strip()


def shorten_text(
    value: Any,
    max_length: int,
) -> str:
    """
    截断文本。
    """
    text = compact_text(
        value
    )

    if len(
        text
    ) <= max_length:
        return text

    return (
        text[
            :max_length
        ].rstrip()
        + "……"
    )


def normalize_mode(
    value: str,
) -> str:
    """
    检查检索模式。
    """
    mode = compact_text(
        value
    ).lower()

    if mode not in {
        "bm25",
        "dense",
        "hybrid",
    }:
        raise ValueError(
            "检索模式只能是 "
            "bm25、dense 或 hybrid。"
        )

    return mode


def safe_number_text(
    value: Any,
    digits: int = 8,
) -> str:
    """
    安全格式化数值。
    """
    if value is None:
        return "-"

    try:
        return f"{float(value):.{digits}f}"
    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )


# ============================================================
# 永久环境变量加载
# ============================================================


def load_user_environment(
    variable_name: str,
) -> str:
    """
    加载环境变量。

    优先顺序：

    1. 当前 Python 进程；
    2. Windows 当前用户永久环境变量。
    """
    value = os.getenv(
        variable_name,
        "",
    ).strip()

    if (
        not value
        and os.name == "nt"
    ):
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Environment",
            ) as key:
                stored_value, _ = (
                    winreg.QueryValueEx(
                        key,
                        variable_name,
                    )
                )

            value = str(
                stored_value
            ).strip()

            if value:
                os.environ[
                    variable_name
                ] = value

        except (
            FileNotFoundError,
            OSError,
        ):
            value = ""

    if not value:
        raise RuntimeError(
            f"未设置 {variable_name}。"
        )

    if not value.isascii():
        raise RuntimeError(
            f"{variable_name} 包含非 ASCII 字符，"
            "可能保存了中文占位符。"
        )

    return value


# ============================================================
# 文档元数据
# ============================================================


def document_title(
    document: Document,
) -> str:
    """
    获取证据标题。
    """
    metadata = document.metadata

    for key in (
        "title",
        "topic_name",
        "repository",
        "document_id",
        "doc_id",
        "chunk_id",
    ):
        value = compact_text(
            metadata.get(
                key
            )
        )

        if value:
            return value

    return "未命名证据"


def document_date(
    document: Document,
) -> str:
    """
    获取证据日期。
    """
    metadata = document.metadata

    for key in (
        "snapshot_date",
        "date",
        "created_date",
    ):
        value = compact_text(
            metadata.get(
                key
            )
        )

        if value:
            return value

    return "-"


def document_repository(
    document: Document,
) -> str:
    """
    获取仓库名称。
    """
    return (
        compact_text(
            document.metadata.get(
                "repository"
            )
        )
        or "-"
    )


def document_type(
    document: Document,
) -> str:
    """
    获取文档类型。
    """
    return (
        compact_text(
            document.metadata.get(
                "doc_type"
            )
        )
        or "-"
    )


# ============================================================
# 证据去重与构造
# ============================================================


def deduplicate_documents(
    documents: list[Document],
) -> list[Document]:
    """
    根据 chunk_id 去重。

    如果缺少 chunk_id，则使用标题和正文组合去重。
    """
    results: list[Document] = []

    seen_keys: set[str] = set()

    for document in documents:
        metadata = document.metadata

        chunk_id = compact_text(
            metadata.get(
                "chunk_id"
            )
        )

        if chunk_id:
            key = (
                "chunk_id:"
                + chunk_id
            )
        else:
            key = (
                "fallback:"
                + document_title(
                    document
                )
                + ":"
                + compact_text(
                    document.page_content
                )[:300]
            )

        if key in seen_keys:
            continue

        seen_keys.add(
            key
        )

        results.append(
            document
        )

    return results


def build_evidence_context(
    documents: list[Document],
) -> tuple[
    str,
    list[Document],
]:
    """
    将检索结果转换为编号证据。

    同时限制总字符数，避免提示词过长。
    """
    evidence_blocks: list[str] = []

    used_documents: list[Document] = []

    used_characters = 0

    for document in documents:
        evidence_number = (
            len(
                used_documents
            )
            + 1
        )

        content = shorten_text(
            document.page_content,
            MAX_CHARACTERS_PER_EVIDENCE,
        )

        block = (
            f"[证据{evidence_number}]\n"
            f"标题：{document_title(document)}\n"
            f"日期：{document_date(document)}\n"
            f"类型：{document_type(document)}\n"
            f"仓库：{document_repository(document)}\n"
            f"Chunk ID："
            f"{document.metadata.get('chunk_id', '-')}\n"
            f"内容：{content}"
        )

        prospective_length = (
            used_characters
            + len(
                block
            )
        )

        if (
            used_documents
            and prospective_length
            > MAX_TOTAL_EVIDENCE_CHARACTERS
        ):
            break

        evidence_blocks.append(
            block
        )

        used_documents.append(
            document
        )

        used_characters = (
            prospective_length
        )

    return (
        "\n\n".join(
            evidence_blocks
        ),
        used_documents,
    )


# ============================================================
# DeepSeek 客户端
# ============================================================


class DeepSeekClient:
    """
    简单的 DeepSeek Chat Completions 客户端。
    """

    def __init__(
        self,
    ) -> None:
        self.api_key = (
            load_user_environment(
                "DEEPSEEK_API_KEY"
            )
        )

        self.base_url = (
            DEEPSEEK_BASE_URL
        )

        self.model = (
            DEEPSEEK_MODEL
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Authorization": (
                    "Bearer "
                    + self.api_key
                ),

                "Content-Type": (
                    "application/json"
                ),

                "Accept": (
                    "application/json"
                ),
            }
        )

    def _send_request(
        self,
        payload: dict[str, Any],
    ) -> requests.Response:
        """
        发送一次 API 请求。
        """
        return self.session.post(
            (
                self.base_url
                + "/chat/completions"
            ),
            json=payload,
            timeout=(
                DEEPSEEK_TIMEOUT_SECONDS
            ),
        )

    def answer(
        self,
        *,
        question: str,
        evidence_context: str,
    ) -> tuple[
        str,
        dict[str, Any],
        dict[str, Any],
    ]:
        """
        根据检索证据回答问题。

        返回：

        answer,
        usage,
        response_metadata
        """
        user_prompt = (
            "用户问题：\n"
            f"{question}\n\n"
            "检索证据：\n"
            f"{evidence_context}\n\n"
            "请严格依据以上证据回答。"
        )

        payload: dict[str, Any] = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                    ),
                },

                {
                    "role": "user",
                    "content": (
                        user_prompt
                    ),
                },
            ],

            "temperature": (
                DEEPSEEK_TEMPERATURE
            ),

            "max_tokens": (
                DEEPSEEK_MAX_TOKENS
            ),

            "stream": False,
        }

        if SEND_THINKING_FIELD:
            payload[
                "thinking"
            ] = {
                "type": "disabled"
            }

        response = self._send_request(
            payload
        )

        # 某些 DeepSeek 接口或模型可能不接受
        # thinking 字段。
        #
        # 遇到相关 400 错误时自动移除并重试。
        if (
            response.status_code
            == 400
            and "thinking"
            in response.text.lower()
            and "thinking"
            in payload
        ):
            payload.pop(
                "thinking",
                None,
            )

            response = self._send_request(
                payload
            )

        if not response.ok:
            try:
                error_data = (
                    response.json()
                )
            except ValueError:
                error_data = {
                    "raw_response": (
                        response.text
                    )
                }

            raise RuntimeError(
                "DeepSeek API 调用失败："
                f"HTTP {response.status_code}\n"
                + json.dumps(
                    error_data,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        data = response.json()

        choices = data.get(
            "choices"
        )

        if not isinstance(
            choices,
            list,
        ) or not choices:
            raise RuntimeError(
                "DeepSeek 返回中没有 choices。"
            )

        first_choice = choices[0]

        message = first_choice.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise RuntimeError(
                "DeepSeek 返回中没有有效 message。"
            )

        answer_text = compact_text(
            message.get(
                "content"
            )
        )

        if not answer_text:
            raise RuntimeError(
                "DeepSeek 返回了空答案。"
            )

        usage = data.get(
            "usage"
        )

        if not isinstance(
            usage,
            dict,
        ):
            usage = {}

        response_metadata = {
            "id": data.get(
                "id"
            ),

            "model": data.get(
                "model"
            ),

            "finish_reason": (
                first_choice.get(
                    "finish_reason"
                )
            ),

            "created": data.get(
                "created"
            ),
        }

        return (
            answer_text,
            usage,
            response_metadata,
        )


# ============================================================
# 终端显示
# ============================================================


def print_separator(
    character: str = "=",
    length: int = 100,
) -> None:
    print(
        character
        * length
    )


def print_retriever_status(
    retriever: IntelligenceRetriever,
) -> None:
    """
    显示索引状态。
    """
    status = retriever.status()

    print_separator()

    print(
        "GitHub 技术情报问答系统"
    )

    print_separator(
        "-"
    )

    print(
        "Embedding：",
        status.get(
            "embedding_model"
        ),
    )

    print(
        "Chroma Chunk 数：",
        status.get(
            "chroma_document_count"
        ),
    )

    print(
        "BM25 加载方式：",
        status.get(
            "bm25_load_method"
        ),
    )

    print(
        "BM25 Chunk 数：",
        status.get(
            "bm25_document_count"
        ),
    )

    print(
        "DeepSeek 模型：",
        DEEPSEEK_MODEL,
    )

    print_separator()


def print_evidence_summary(
    documents: list[Document],
) -> None:
    """
    显示答案使用的证据来源。
    """
    if not documents:
        return

    print()

    print_separator(
        "-"
    )

    print(
        "证据来源"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        print()

        print(
            f"[证据{index}] "
            f"{document_title(document)}"
        )

        print(
            "  日期：",
            document_date(
                document
            ),
        )

        print(
            "  类型：",
            document_type(
                document
            ),
        )

        print(
            "  仓库：",
            document_repository(
                document
            ),
        )

        print(
            "  Chunk：",
            metadata.get(
                "chunk_id",
                "-",
            ),
        )

        if metadata.get(
            "rrf_score"
        ) is not None:
            print(
                "  RRF：",
                safe_number_text(
                    metadata.get(
                        "rrf_score"
                    )
                ),
            )

            print(
                "  Dense 排名：",
                metadata.get(
                    "dense_rank",
                    "-",
                ),
                " | BM25 排名：",
                metadata.get(
                    "bm25_rank",
                    "-",
                ),
            )

    print_separator(
        "-"
    )


# ============================================================
# 记录保存
# ============================================================


def save_qa_record(
    *,
    question: str,
    answer: str,
    retrieval_mode: str,
    top_k: int,
    retrieval_elapsed_ms: float,
    generation_elapsed_ms: float,
    documents: list[Document],
    usage: dict[str, Any],
    response_metadata: dict[str, Any],
) -> Path:
    """
    保存完整问答记录。
    """
    now = datetime.now()

    date_folder = (
        QA_REPORT_ROOT
        / now.strftime(
            "%Y-%m-%d"
        )
    )

    date_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        date_folder
        / (
            "qa_"
            + now.strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".json"
        )
    )

    data = {
        "question": question,

        "answer": answer,

        "retrieval": {
            "mode": retrieval_mode,

            "top_k": top_k,

            "elapsed_ms": (
                retrieval_elapsed_ms
            ),

            "result_count": len(
                documents
            ),

            "results": [
                {
                    "evidence_number": (
                        index
                    ),

                    **document_to_dict(
                        document,
                        include_text=True,
                    ),
                }
                for index, document in enumerate(
                    documents,
                    start=1,
                )
            ],
        },

        "generation": {
            "model": (
                DEEPSEEK_MODEL
            ),

            "elapsed_ms": (
                generation_elapsed_ms
            ),

            "usage": usage,

            "response_metadata": (
                response_metadata
            ),
        },

        "created_at": (
            now.isoformat()
        ),
    }

    output_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


# ============================================================
# 问答主逻辑
# ============================================================


def ask_question(
    *,
    retriever: IntelligenceRetriever,
    deepseek_client: DeepSeekClient,
    question: str,
    retrieval_mode: str,
    top_k: int,
) -> str:
    """
    执行一次完整 RAG 问答。
    """
    normalized_question = compact_text(
        question
    )

    if not normalized_question:
        print(
            "问题不能为空。"
        )

        return ""

    mode = normalize_mode(
        retrieval_mode
    )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    print()

    print_separator()

    print(
        "问题：",
        normalized_question,
    )

    print(
        "检索模式：",
        mode,
    )

    print(
        "Top-K：",
        top_k,
    )

    print_separator(
        "-"
    )

    retrieval_started_at = (
        time.perf_counter()
    )

    retrieved_documents = (
        retriever.search(
            normalized_question,
            mode=mode,
            top_k=top_k,
        )
    )

    retrieved_documents = (
        deduplicate_documents(
            retrieved_documents
        )
    )

    retrieval_elapsed_ms = (
        time.perf_counter()
        - retrieval_started_at
    ) * 1000.0

    if not retrieved_documents:
        print()

        print(
            NO_EVIDENCE_ANSWER
        )

        print()

        print(
            "检索耗时：",
            f"{retrieval_elapsed_ms:.2f} ms",
        )

        return NO_EVIDENCE_ANSWER

    (
        evidence_context,
        used_documents,
    ) = build_evidence_context(
        retrieved_documents
    )

    print(
        "检索完成：",
        f"{len(used_documents)} 条证据，"
        f"{retrieval_elapsed_ms:.2f} ms",
    )

    print(
        "正在调用 DeepSeek 生成答案……"
    )

    generation_started_at = (
        time.perf_counter()
    )

    (
        answer,
        usage,
        response_metadata,
    ) = deepseek_client.answer(
        question=normalized_question,
        evidence_context=(
            evidence_context
        ),
    )

    generation_elapsed_ms = (
        time.perf_counter()
        - generation_started_at
    ) * 1000.0

    print()

    print_separator(
        "-"
    )

    print(
        "回答"
    )

    print()

    print(
        answer
    )

    print()

    print_separator(
        "-"
    )

    print(
        "检索耗时：",
        f"{retrieval_elapsed_ms:.2f} ms",
    )

    print(
        "生成耗时：",
        f"{generation_elapsed_ms:.2f} ms",
    )

    if usage:
        print(
            "Token 使用：",
            usage,
        )

    if SHOW_EVIDENCE_SUMMARY:
        print_evidence_summary(
            used_documents
        )

    if SAVE_QA_RECORDS:
        output_path = save_qa_record(
            question=(
                normalized_question
            ),

            answer=answer,

            retrieval_mode=mode,

            top_k=top_k,

            retrieval_elapsed_ms=(
                retrieval_elapsed_ms
            ),

            generation_elapsed_ms=(
                generation_elapsed_ms
            ),

            documents=used_documents,

            usage=usage,

            response_metadata=(
                response_metadata
            ),
        )

        print(
            "问答记录：",
            output_path,
        )

    print_separator()

    return answer


# ============================================================
# 交互命令
# ============================================================


def print_help() -> None:
    """
    显示交互命令。
    """
    print()

    print(
        "可用命令："
    )

    print(
        "  /mode hybrid   使用 Hybrid RRF"
    )

    print(
        "  /mode bm25     使用 BM25"
    )

    print(
        "  /mode dense    使用 Dense"
    )

    print(
        "  /topk 6        修改证据数量"
    )

    print(
        "  /status        查看索引状态"
    )

    print(
        "  /help          查看命令"
    )

    print(
        "  /exit          退出"
    )

    print()


def interactive_loop(
    *,
    retriever: IntelligenceRetriever,
    deepseek_client: DeepSeekClient,
) -> None:
    """
    连续问答模式。
    """
    current_mode = (
        normalize_mode(
            DEFAULT_RETRIEVAL_MODE
        )
    )

    current_top_k = int(
        DEFAULT_TOP_K
    )

    print()

    print(
        "已进入技术情报问答模式。"
    )

    print(
        f"当前检索模式：{current_mode}"
    )

    print(
        f"当前 Top-K：{current_top_k}"
    )

    print(
        "输入 /help 查看命令。"
    )

    print()

    while True:
        try:
            user_input = input(
                "问题 > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print()

            print(
                "问答系统已退出。"
            )

            break

        if not user_input:
            continue

        if user_input.startswith(
            "/"
        ):
            lowered = (
                user_input.lower()
            )

            if lowered in {
                "/exit",
                "/quit",
                "/q",
            }:
                print(
                    "问答系统已退出。"
                )

                break

            if lowered == "/help":
                print_help()

                continue

            if lowered == "/status":
                print_retriever_status(
                    retriever
                )

                continue

            if lowered.startswith(
                "/mode "
            ):
                requested_mode = (
                    user_input.split(
                        maxsplit=1
                    )[1]
                )

                try:
                    current_mode = (
                        normalize_mode(
                            requested_mode
                        )
                    )

                    print(
                        "检索模式已切换为：",
                        current_mode,
                    )

                except ValueError as exc:
                    print(
                        f"错误：{exc}"
                    )

                continue

            if lowered.startswith(
                "/topk "
            ):
                raw_value = (
                    user_input.split(
                        maxsplit=1
                    )[1]
                )

                try:
                    new_top_k = int(
                        raw_value
                    )

                    if new_top_k <= 0:
                        raise ValueError

                    current_top_k = (
                        new_top_k
                    )

                    print(
                        "Top-K 已修改为：",
                        current_top_k,
                    )

                except ValueError:
                    print(
                        "Top-K 必须是大于 0 的整数。"
                    )

                continue

            print(
                "未知命令。输入 /help 查看帮助。"
            )

            continue

        try:
            ask_question(
                retriever=retriever,

                deepseek_client=(
                    deepseek_client
                ),

                question=user_input,

                retrieval_mode=(
                    current_mode
                ),

                top_k=current_top_k,
            )

        except Exception as exc:
            print()

            print(
                "问答失败："
            )

            print(
                exc
            )

            print()


# ============================================================
# 主函数
# ============================================================


def main() -> None:
    """
    启动问答系统。
    """
    print(
        "正在加载持久化检索库……"
    )

    retriever = IntelligenceRetriever(
        default_mode=(
            normalize_mode(
                DEFAULT_RETRIEVAL_MODE
            )
        ),

        default_top_k=(
            DEFAULT_TOP_K
        ),
    )

    print(
        "持久化检索库加载完成。"
    )

    print(
        "正在初始化 DeepSeek 客户端……"
    )

    deepseek_client = (
        DeepSeekClient()
    )

    print(
        "DeepSeek 客户端初始化完成。"
    )

    print_retriever_status(
        retriever
    )

    # 支持单次命令行问答：
    #
    # python scripts\ask_intelligence.py "今天有哪些热点？"
    if len(
        sys.argv
    ) > 1:
        question = " ".join(
            sys.argv[
                1:
            ]
        ).strip()

        ask_question(
            retriever=retriever,

            deepseek_client=(
                deepseek_client
            ),

            question=question,

            retrieval_mode=(
                DEFAULT_RETRIEVAL_MODE
            ),

            top_k=(
                DEFAULT_TOP_K
            ),
        )

        return

    interactive_loop(
        retriever=retriever,
        deepseek_client=(
            deepseek_client
        ),
    )


if __name__ == "__main__":
    main()