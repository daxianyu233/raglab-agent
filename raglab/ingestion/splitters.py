"""文本分块模块。

当前基线流程：

页面级 Document
→ RecursiveCharacterTextSplitter
→ Chunk 级 Document
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """创建递归字符切分器。

    Args:
        chunk_size:
            每个 Chunk 的目标最大字符数。

        chunk_overlap:
            相邻 Chunk 的目标重叠字符数。

    Returns:
        配置完成的 RecursiveCharacterTextSplitter。
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size 必须大于 0，当前值：{chunk_size}"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap 不能小于 0，"
            f"当前值：{chunk_overlap}"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap 必须小于 chunk_size，"
            f"当前值：{chunk_overlap} >= {chunk_size}"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )


def split_page_documents(
    documents: Sequence[Document],
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """将页面级 Document 切分成 Chunk 级 Document。

    原始 metadata 会继承到每个 Chunk。
    本函数还会为 Chunk 增加：

    - chunk_id
    - chunk_index
    - chunk_char_count
    """
    if not documents:
        return []

    chunks = text_splitter.split_documents(
        list(documents)
    )

    # 分别记录每个文档、每一页产生了多少个 Chunk。
    page_chunk_counters: dict[
        tuple[str, object],
        int,
    ] = defaultdict(int)

    for chunk in chunks:
        doc_id = str(
            chunk.metadata.get(
                "doc_id",
                "UNKNOWN-DOC",
            )
        )

        page_number = chunk.metadata.get(
            "page_number",
            "UNKNOWN-PAGE",
        )

        counter_key = (
            doc_id,
            page_number,
        )

        chunk_index = page_chunk_counters[
            counter_key
        ]

        page_chunk_counters[counter_key] += 1

        if isinstance(page_number, int):
            page_label = f"P{page_number:03d}"
        else:
            page_label = "PUNKNOWN"

        chunk.metadata.update(
            {
                "chunk_id": (
                    f"{doc_id}-"
                    f"{page_label}-"
                    f"C{chunk_index:03d}"
                ),
                "chunk_index": chunk_index,
                "chunk_char_count": len(
                    chunk.page_content
                ),
            }
        )

    return chunks