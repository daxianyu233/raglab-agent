from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
# 复用持久化检索器
# ============================================================

from raglab.intelligence.retriever import (
    IntelligenceRetriever,
    document_to_dict,
)


# ============================================================
# 配置区
# ============================================================

# 默认检索模式：
#
# bm25：
#   关键词和专有名词检索较强。
#
# dense：
#   语义相似检索较强。
#
# hybrid：
#   BM25 + Dense + RRF 融合。
DEFAULT_MODE = "hybrid"

# 默认返回数量。
DEFAULT_TOP_K = 5

# 每条结果最多显示多少字符。
TEXT_PREVIEW_LENGTH = 700

# 是否显示完整正文。
SHOW_FULL_TEXT = False

# 是否在启动时打印索引状态。
SHOW_INDEX_STATUS = True

# 是否将每次检索结果保存为 JSON。
SAVE_QUERY_RESULTS = True

# 检索记录保存目录。
QUERY_REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "intelligence_search"
)

# 如果填写问题，脚本启动后只执行一次。
#
# 例如：
#
# FIXED_QUESTION = "今天有哪些值得关注的 Agent 项目？"
#
# 设置为 None 时进入连续交互模式。
FIXED_QUESTION: str | None = None


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


def safe_float_text(
    value: Any,
    digits: int = 8,
) -> str:
    """
    将数值安全转换为显示字符串。
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


def normalize_mode(
    value: str,
) -> str:
    """
    校验检索模式。
    """
    normalized = compact_text(
        value
    ).lower()

    if normalized not in {
        "bm25",
        "dense",
        "hybrid",
    }:
        raise ValueError(
            "检索模式只能是 "
            "bm25、dense 或 hybrid。"
        )

    return normalized


def shorten_text(
    text: str,
    max_length: int,
) -> str:
    """
    截断显示文本。
    """
    normalized = compact_text(
        text
    )

    if len(
        normalized
    ) <= max_length:
        return normalized

    return (
        normalized[
            :max_length
        ].rstrip()
        + "……"
    )


# ============================================================
# 文档元数据
# ============================================================


def document_title(
    document: Document,
) -> str:
    """
    获取用于终端展示的标题。
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

    return "未命名文档"


def document_source_label(
    document: Document,
) -> str:
    """
    获取文档来源标签。
    """
    metadata = document.metadata

    doc_type = compact_text(
        metadata.get(
            "doc_type"
        )
    )

    snapshot_date = compact_text(
        metadata.get(
            "snapshot_date"
        )
    )

    repository = compact_text(
        metadata.get(
            "repository"
        )
    )

    parts = [
        item
        for item in (
            doc_type,
            snapshot_date,
            repository,
        )
        if item
    ]

    return (
        " | ".join(
            parts
        )
        if parts
        else "-"
    )


# ============================================================
# 终端展示
# ============================================================


def print_separator(
    character: str = "=",
    length: int = 100,
) -> None:
    print(
        character
        * length
    )


def print_index_status(
    retriever: IntelligenceRetriever,
) -> None:
    """
    打印当前持久化索引状态。
    """
    status = retriever.status()

    print_separator()

    print(
        "GitHub 技术情报持久化检索系统"
    )

    print_separator(
        "-"
    )

    print(
        "默认模式：",
        status.get(
            "default_mode"
        ),
    )

    print(
        "Embedding：",
        status.get(
            "embedding_model"
        ),
    )

    print(
        "Chroma：",
        status.get(
            "chroma_directory"
        ),
    )

    print(
        "Collection：",
        status.get(
            "chroma_collection_name"
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

    print_separator()


def print_document(
    document: Document,
    rank: int,
) -> None:
    """
    输出一条检索结果。
    """
    metadata = document.metadata

    print_separator(
        "-"
    )

    print(
        f"[证据 {rank}] "
        f"{document_title(document)}"
    )

    print(
        "来源：",
        document_source_label(
            document
        ),
    )

    print(
        "document_id：",
        metadata.get(
            "document_id"
        )
        or metadata.get(
            "doc_id"
        )
        or "-",
    )

    print(
        "chunk_id：",
        metadata.get(
            "chunk_id",
            "-",
        ),
    )

    print(
        "检索方式：",
        metadata.get(
            "retrieval_method",
            "-",
        ),
    )

    if (
        metadata.get(
            "retrieval_method"
        )
        == "hybrid_rrf"
    ):
        print(
            "RRF 分数：",
            safe_float_text(
                metadata.get(
                    "rrf_score"
                )
            ),
        )

        print(
            "Dense 排名：",
            metadata.get(
                "dense_rank",
                "-",
            ),
            " | Dense 原始分数：",
            safe_float_text(
                metadata.get(
                    "dense_raw_score"
                )
            ),
        )

        print(
            "BM25 排名：",
            metadata.get(
                "bm25_rank",
                "-",
            ),
            " | BM25 原始分数：",
            safe_float_text(
                metadata.get(
                    "bm25_raw_score"
                )
            ),
        )

    else:
        print(
            "原始分数：",
            safe_float_text(
                metadata.get(
                    "retrieval_raw_score"
                )
            ),
        )

    print()

    if SHOW_FULL_TEXT:
        print(
            document.page_content.strip()
        )
    else:
        print(
            shorten_text(
                document.page_content,
                TEXT_PREVIEW_LENGTH,
            )
        )


def print_results(
    question: str,
    mode: str,
    documents: list[Document],
    elapsed_ms: float,
) -> None:
    """
    输出本次完整检索结果。
    """
    print()

    print_separator()

    print(
        "问题：",
        question,
    )

    print(
        "模式：",
        mode,
    )

    print(
        "命中数量：",
        len(
            documents
        ),
    )

    print(
        "检索耗时：",
        f"{elapsed_ms:.2f} ms",
    )

    if not documents:
        print()

        print(
            "未检索到相关证据。"
        )

        print_separator()

        return

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        print_document(
            document,
            rank,
        )

    print_separator()


# ============================================================
# 结果保存
# ============================================================


def save_query_result(
    *,
    question: str,
    mode: str,
    top_k: int,
    elapsed_ms: float,
    documents: list[Document],
) -> Path:
    """
    保存一次检索结果。
    """
    now = datetime.now()

    date_folder = (
        QUERY_REPORT_ROOT
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
            "search_"
            + now.strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            + ".json"
        )
    )

    data = {
        "question": question,
        "mode": mode,
        "top_k": top_k,
        "elapsed_ms": elapsed_ms,
        "created_at": (
            now.isoformat()
        ),
        "result_count": len(
            documents
        ),
        "results": [
            {
                "rank": rank,
                **document_to_dict(
                    document,
                    include_text=True,
                ),
            }
            for rank, document in enumerate(
                documents,
                start=1,
            )
        ],
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
# 单次检索
# ============================================================


def run_search(
    retriever: IntelligenceRetriever,
    *,
    question: str,
    mode: str,
    top_k: int,
) -> list[Document]:
    """
    执行一次检索。
    """
    normalized_question = compact_text(
        question
    )

    if not normalized_question:
        print(
            "问题不能为空。"
        )

        return []

    selected_mode = normalize_mode(
        mode
    )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    started_at = (
        time.perf_counter()
    )

    documents = retriever.search(
        normalized_question,
        mode=selected_mode,
        top_k=top_k,
    )

    elapsed_ms = (
        time.perf_counter()
        - started_at
    ) * 1000.0

    print_results(
        normalized_question,
        selected_mode,
        documents,
        elapsed_ms,
    )

    if SAVE_QUERY_RESULTS:
        output_path = save_query_result(
            question=(
                normalized_question
            ),
            mode=selected_mode,
            top_k=top_k,
            elapsed_ms=elapsed_ms,
            documents=documents,
        )

        print(
            "检索记录：",
            output_path,
        )

    return documents


# ============================================================
# 交互命令
# ============================================================


def print_help() -> None:
    """
    打印交互命令。
    """
    print()

    print(
        "可用命令："
    )

    print(
        "  /mode bm25    切换为 BM25"
    )

    print(
        "  /mode dense   切换为 Dense"
    )

    print(
        "  /mode hybrid  切换为 Hybrid"
    )

    print(
        "  /topk 5       修改返回数量"
    )

    print(
        "  /status       查看索引状态"
    )

    print(
        "  /help         查看帮助"
    )

    print(
        "  /exit         退出"
    )

    print()


def handle_command(
    command: str,
    *,
    retriever: IntelligenceRetriever,
    current_mode: str,
    current_top_k: int,
) -> tuple[
    bool,
    str,
    int,
]:
    """
    处理交互命令。

    返回：

    should_continue,
    new_mode,
    new_top_k
    """
    normalized = compact_text(
        command
    )

    lowered = normalized.lower()

    if lowered in {
        "/exit",
        "/quit",
        "/q",
    }:
        return (
            False,
            current_mode,
            current_top_k,
        )

    if lowered == "/help":
        print_help()

        return (
            True,
            current_mode,
            current_top_k,
        )

    if lowered == "/status":
        print_index_status(
            retriever
        )

        return (
            True,
            current_mode,
            current_top_k,
        )

    if lowered.startswith(
        "/mode "
    ):
        requested_mode = (
            normalized.split(
                maxsplit=1
            )[1]
        )

        try:
            new_mode = normalize_mode(
                requested_mode
            )

        except ValueError as exc:
            print(
                f"错误：{exc}"
            )

            return (
                True,
                current_mode,
                current_top_k,
            )

        print(
            "检索模式已切换为：",
            new_mode,
        )

        return (
            True,
            new_mode,
            current_top_k,
        )

    if lowered.startswith(
        "/topk "
    ):
        value = normalized.split(
            maxsplit=1
        )[1]

        try:
            new_top_k = int(
                value
            )

            if new_top_k <= 0:
                raise ValueError

        except ValueError:
            print(
                "top_k 必须是大于 0 的整数。"
            )

            return (
                True,
                current_mode,
                current_top_k,
            )

        print(
            "Top-K 已修改为：",
            new_top_k,
        )

        return (
            True,
            current_mode,
            new_top_k,
        )

    print(
        "未知命令。输入 /help 查看帮助。"
    )

    return (
        True,
        current_mode,
        current_top_k,
    )


# ============================================================
# 交互模式
# ============================================================


def interactive_loop(
    retriever: IntelligenceRetriever,
) -> None:
    """
    启动连续检索。
    """
    current_mode = normalize_mode(
        DEFAULT_MODE
    )

    current_top_k = int(
        DEFAULT_TOP_K
    )

    print()

    print(
        "已进入技术情报检索模式。"
    )

    print(
        f"当前模式：{current_mode}"
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
                "检索问题 > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print()

            print(
                "检索系统已退出。"
            )

            break

        if not user_input:
            continue

        if user_input.startswith(
            "/"
        ):
            (
                should_continue,
                current_mode,
                current_top_k,
            ) = handle_command(
                user_input,
                retriever=retriever,
                current_mode=current_mode,
                current_top_k=(
                    current_top_k
                ),
            )

            if not should_continue:
                print(
                    "检索系统已退出。"
                )

                break

            continue

        try:
            run_search(
                retriever,
                question=user_input,
                mode=current_mode,
                top_k=current_top_k,
            )

        except Exception as exc:
            print()

            print(
                "检索失败：",
                exc,
            )


# ============================================================
# 主函数
# ============================================================


def main() -> None:
    print(
        "正在加载持久化检索库……"
    )

    retriever = IntelligenceRetriever(
        default_mode=(
            normalize_mode(
                DEFAULT_MODE
            )
        ),
        default_top_k=(
            DEFAULT_TOP_K
        ),
    )

    print(
        "持久化检索库加载完成。"
    )

    if SHOW_INDEX_STATUS:
        print_index_status(
            retriever
        )

    if FIXED_QUESTION:
        run_search(
            retriever,
            question=FIXED_QUESTION,
            mode=DEFAULT_MODE,
            top_k=DEFAULT_TOP_K,
        )

        return

    interactive_loop(
        retriever
    )


if __name__ == "__main__":
    main()