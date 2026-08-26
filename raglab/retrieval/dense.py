"""Dense Retrieval 检索模块。

负责从已经建立好的 Chroma 向量库中，
检索与用户问题最接近的 Chunk。

当前阶段返回：

- Document；
- Chroma 给出的距离 distance。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document


def dense_similarity_search_with_score(
    vector_store: Chroma,
    query: str,
    top_k: int,
    metadata_filter: Mapping[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """执行带距离分数的 Dense Retrieval。

    Args:
        vector_store:
            已经打开的 Chroma 向量库。

        query:
            用户输入的问题。

        top_k:
            最多返回多少个 Chunk。

        metadata_filter:
            可选的 Metadata 过滤条件。

            例如：

                {"corpus": "langgraph"}

            表示只在 langgraph 语料中检索。

    Returns:
        list[tuple[Document, float]]:
            每个结果包含：

            - 检索到的 Chunk Document；
            - Chroma 返回的距离值。

            当前 Chroma 使用默认 L2 距离，
            因此距离越小通常表示越相关。
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "查询文本不能为空。"
        )

    if top_k <= 0:
        raise ValueError(
            f"top_k 必须大于 0，当前值：{top_k}"
        )

    # 没有 Metadata 过滤条件时，
    # 不向 Chroma 传递 filter 参数。
    if metadata_filter is None:
        return vector_store.similarity_search_with_score(
            query=cleaned_query,
            k=top_k,
        )

    return vector_store.similarity_search_with_score(
        query=cleaned_query,
        k=top_k,
        filter=dict(metadata_filter),
    )