"""RAG 上下文构造模块。

主要职责：

检索结果 Document
→ 去除重复 Chunk
→ 按检索顺序选择文档
→ 添加可追踪的资料编号
→ 控制上下文长度
→ 生成可直接交给 LLM 的上下文文本

本模块不负责检索，也不负责调用语言模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document


@dataclass(frozen=True)
class ContextReference:
    """记录一条进入最终上下文的资料。"""

    reference_id: str
    rank: int
    chunk_id: str | None
    doc_id: str | None
    title: str | None
    page_number: int | str | None
    text_length: int


@dataclass(frozen=True)
class BuiltContext:
    """上下文构造结果。"""

    text: str
    documents: list[Document]
    references: list[ContextReference]
    character_count: int
    selected_count: int
    omitted_count: int
    truncated: bool


def normalize_text(
    text: str,
) -> str:
    """清理 Chunk 中多余的空白字符。

    保留段落换行，但将连续空格和连续空行进行压缩。
    """

    normalized = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    normalized = re.sub(
        r"[ \t]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\n[ \t]+",
        "\n",
        normalized,
    )

    normalized = re.sub(
        r"\n{3,}",
        "\n\n",
        normalized,
    )

    return normalized.strip()


def get_chunk_id(
    document: Document,
) -> str | None:
    """读取有效的 chunk_id。"""

    chunk_id = document.metadata.get(
        "chunk_id"
    )

    if chunk_id is None:
        return None

    normalized = str(chunk_id).strip()

    if not normalized:
        return None

    return normalized


def deduplicate_documents(
    documents: Sequence[Document],
) -> list[Document]:
    """按照 chunk_id 去除重复文档。

    没有 chunk_id 的文档使用正文内容作为去重依据。
    保留第一次出现的文档，从而维持检索排序。
    """

    deduplicated: list[Document] = []
    seen_keys: set[str] = set()

    for document in documents:
        if not isinstance(
            document,
            Document,
        ):
            raise TypeError(
                "documents 中只能包含 Document，"
                f"实际类型：{type(document)!r}"
            )

        chunk_id = get_chunk_id(
            document
        )

        if chunk_id is not None:
            unique_key = (
                f"chunk_id:{chunk_id}"
            )
        else:
            normalized_text = normalize_text(
                document.page_content
            )

            unique_key = (
                f"content:{normalized_text}"
            )

        if unique_key in seen_keys:
            continue

        seen_keys.add(unique_key)
        deduplicated.append(document)

    return deduplicated


def format_source_header(
    document: Document,
    reference_id: str,
) -> str:
    """为单个 Chunk 生成资料标题。"""

    metadata = document.metadata

    title = metadata.get(
        "title"
    )

    doc_id = metadata.get(
        "doc_id"
    )

    page_number = metadata.get(
        "page_number"
    )

    parts = [
        f"[{reference_id}]",
    ]

    if title is not None:
        normalized_title = str(
            title
        ).strip()

        if normalized_title:
            parts.append(
                f"标题：{normalized_title}"
            )

    if doc_id is not None:
        normalized_doc_id = str(
            doc_id
        ).strip()

        if normalized_doc_id:
            parts.append(
                f"文档ID：{normalized_doc_id}"
            )

    if page_number is not None:
        parts.append(
            f"页码：{page_number}"
        )

    return " | ".join(parts)


def build_context(
    documents: Sequence[Document],
    *,
    max_documents: int = 5,
    max_characters: int = 8000,
    include_metadata: bool = True,
    deduplicate: bool = True,
) -> BuiltContext:
    """将检索结果构造成 LLM 上下文。

    Parameters
    ----------
    documents:
        检索器返回的、已经按相关性排列的 Document。

    max_documents:
        最多选择多少个 Chunk。

    max_characters:
        最终上下文允许的最大字符数。

        这里暂时使用字符数限制，是为了不依赖某个具体
        LLM 的 Tokenizer。后续接入具体模型后，会增加
        精确 Token 计数。

    include_metadata:
        是否在正文前添加资料编号、标题、文档 ID 和页码。

    deduplicate:
        是否按照 chunk_id 或正文内容进行去重。

    Returns
    -------
    BuiltContext
        包含最终上下文文本、实际选择的 Document、
        引用信息和截断统计。
    """

    if max_documents <= 0:
        raise ValueError(
            "max_documents 必须大于 0。"
        )

    if max_characters <= 0:
        raise ValueError(
            "max_characters 必须大于 0。"
        )

    original_documents = list(
        documents
    )

    if deduplicate:
        candidate_documents = (
            deduplicate_documents(
                original_documents
            )
        )
    else:
        candidate_documents = (
            original_documents
        )

    candidate_documents = (
        candidate_documents[
            :max_documents
        ]
    )

    selected_documents: list[
        Document
    ] = []

    references: list[
        ContextReference
    ] = []

    context_blocks: list[str] = []
    current_length = 0
    truncated = False

    for rank, document in enumerate(
        candidate_documents,
        start=1,
    ):
        text = normalize_text(
            document.page_content
        )

        if not text:
            continue

        reference_id = (
            f"资料{rank}"
        )

        if include_metadata:
            header = format_source_header(
                document,
                reference_id,
            )

            block = (
                f"{header}\n"
                f"{text}"
            )
        else:
            block = text

        separator = (
            "\n\n"
            if context_blocks
            else ""
        )

        required_length = (
            len(separator)
            + len(block)
        )

        remaining_characters = (
            max_characters
            - current_length
            - len(separator)
        )

        if required_length <= (
            max_characters
            - current_length
        ):
            final_block = block

        elif not context_blocks:
            # 第一条资料本身超过最大长度时，
            # 保留其开头，避免返回完全空的上下文。
            if remaining_characters <= 0:
                break

            if remaining_characters > 3:
                final_block = (
                    block[
                        :remaining_characters - 3
                    ]
                    + "..."
                )
            else:
                final_block = block[
                    :remaining_characters
                ]

            truncated = True

        else:
            # 已有完整资料进入上下文时，
            # 不再把后续 Chunk 从中间截断。
            truncated = True
            break

        context_blocks.append(
            final_block
        )

        current_length += (
            len(separator)
            + len(final_block)
        )

        selected_documents.append(
            document
        )

        metadata = document.metadata

        references.append(
            ContextReference(
                reference_id=reference_id,
                rank=rank,
                chunk_id=get_chunk_id(
                    document
                ),
                doc_id=(
                    None
                    if metadata.get(
                        "doc_id"
                    ) is None
                    else str(
                        metadata.get(
                            "doc_id"
                        )
                    )
                ),
                title=(
                    None
                    if metadata.get(
                        "title"
                    ) is None
                    else str(
                        metadata.get(
                            "title"
                        )
                    )
                ),
                page_number=metadata.get(
                    "page_number"
                ),
                text_length=len(
                    final_block
                ),
            )
        )

        if truncated:
            break

    final_text = "\n\n".join(
        context_blocks
    )

    omitted_count = (
        len(original_documents)
        - len(selected_documents)
    )

    return BuiltContext(
        text=final_text,
        documents=selected_documents,
        references=references,
        character_count=len(final_text),
        selected_count=len(
            selected_documents
        ),
        omitted_count=max(
            omitted_count,
            0,
        ),
        truncated=truncated,
    )