"""Chroma 向量数据库管理模块。

负责：

1. 创建或打开持久化 Chroma；
2. 将 Chunk 写入 Chroma；
3. 重建基线向量库。

不负责：

- 加载 PDF；
- 文本分块；
- 创建 Embedding 模型；
- 处理用户问题。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def open_chroma_store(
    embeddings: Embeddings,
    persist_directory: Path,
    collection_name: str,
) -> Chroma:
    """创建或打开一个持久化 Chroma 向量库。

    如果目录和 Collection 已经存在，就打开已有数据；
    如果不存在，就创建新的向量库。

    Args:
        embeddings:
            查询和新增文档时使用的 Embedding 模型。

        persist_directory:
            Chroma 数据在硬盘上的保存目录。

        collection_name:
            Chroma Collection 名称。

    Returns:
        Chroma:
            可执行写入和检索操作的向量库对象。
    """
    if not collection_name.strip():
        raise ValueError(
            "Chroma collection_name 不能为空。"
        )

    # 确保持久化目录存在。
    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(
            persist_directory.resolve()
        ),
    )


def rebuild_chroma_store(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    persist_directory: Path,
    collection_name: str,
) -> Chroma:
    """删除旧索引，并使用当前 Chunk 重新建立向量库。

    当前项目处于实验阶段，切分参数和 Embedding 模型
    可能经常变化，因此基线建库采用完整重建方式。

    Args:
        chunks:
            要写入 Chroma 的 Chunk Document。

        embeddings:
            用于生成 Chunk 向量的模型。

        persist_directory:
            Chroma 数据保存目录。

        collection_name:
            Collection 名称。

    Returns:
        Chroma:
            建立完成的向量库。
    """
    if not chunks:
        raise ValueError(
            "Chunk 列表为空，不能建立向量库。"
        )

    # 当前基线采用完整重建。
    #
    # 删除旧目录可以避免重复执行建库脚本后，
    # 相同 Chunk 被重复写入。
    if persist_directory.exists():
        shutil.rmtree(persist_directory)

    vector_store = open_chroma_store(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # 使用前面生成的稳定 chunk_id 作为 Chroma 记录 ID。
    #
    # 例如：
    # DOC-LC-001-P001-C000
    chunk_ids: list[str] = []

    for chunk in chunks:
        chunk_id = chunk.metadata.get(
            "chunk_id"
        )

        if not isinstance(chunk_id, str):
            raise ValueError(
                "Chunk 缺少有效的 chunk_id："
                f"{chunk.metadata}"
            )

        chunk_ids.append(chunk_id)

    # 确认所有 Chunk ID 唯一。
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "检测到重复的 chunk_id，"
            "不能建立向量库。"
        )

    # add_documents() 内部会执行：
    #
    # chunk.page_content
    # → Embedding
    # → 向量
    # → 写入 Chroma
    #
    # 同时保存：
    # - 文本内容
    # - metadata
    # - ID
    vector_store.add_documents(
        documents=list(chunks),
        ids=chunk_ids,
    )

    return vector_store