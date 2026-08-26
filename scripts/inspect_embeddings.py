"""检查 Embedding 输出并手动执行一次语义检索。

当前流程：

PDF
→ 页面级 Document
→ Chunk级 Document
→ Embedding向量
→ 手动计算余弦相似度
→ 输出最相关的Top-K Chunk

这里暂时不使用 Chroma，
目的是先看清向量检索的基本过程。
"""

from __future__ import annotations

from math import sqrt

import yaml

from raglab.embeddings.factory import (
    create_huggingface_embeddings,
)
from raglab.ingestion.loaders import load_pdf_corpus
from raglab.ingestion.splitters import (
    create_recursive_splitter,
    split_page_documents,
)
from raglab.settings import BASELINE_CONFIG_PATH


def load_baseline_config() -> dict:
    """读取完整的 baseline.yaml 配置。"""
    with BASELINE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "baseline.yaml 的根节点必须是字典。"
        )

    return config


def vector_norm(vector: list[float]) -> float:
    """计算向量的 L2 范数。"""
    return sqrt(
        sum(value * value for value in vector)
    )


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """计算两个向量的余弦相似度。

    余弦相似度越接近 1，通常表示方向越相似。
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            "两个向量的维度不一致："
            f"{len(vector_a)} != {len(vector_b)}"
        )

    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    dot_product = sum(
        value_a * value_b
        for value_a, value_b in zip(
            vector_a,
            vector_b,
            strict=True,
        )
    )

    return dot_product / (norm_a * norm_b)


def normalize_preview(
    text: str,
    max_length: int = 220,
) -> str:
    """整理文本并截断，仅用于终端显示。"""
    normalized = " ".join(
        text.strip().split()
    )

    if len(normalized) <= max_length:
        return normalized

    return normalized[:max_length] + "..."


def main() -> None:
    # ---------------------------------------------------------
    # 1. 读取配置
    # ---------------------------------------------------------

    config = load_baseline_config()

    splitter_config = config["splitter"]
    embedding_config = config["embedding"]
    retrieval_config = config["retrieval"]

    chunk_size = int(
        splitter_config["chunk_size"]
    )

    chunk_overlap = int(
        splitter_config["chunk_overlap"]
    )

    model_name = str(
        embedding_config["model"]
    )

    normalize_embeddings = bool(
        embedding_config[
            "normalize_embeddings"
        ]
    )

    top_k = int(
        retrieval_config["top_k"]
    )

    # ---------------------------------------------------------
    # 2. 加载和切分文本
    # ---------------------------------------------------------

    page_documents = load_pdf_corpus()

    text_splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = split_page_documents(
        documents=page_documents,
        text_splitter=text_splitter,
    )

    if not chunks:
        raise RuntimeError(
            "没有生成任何 Chunk，无法执行 Embedding。"
        )

    print("=" * 70)
    print("RAGLab Embedding 检查")
    print("=" * 70)

    print(f"页面数量：{len(page_documents)}")
    print(f"Chunk 数量：{len(chunks)}")
    print(f"Embedding 模型：{model_name}")
    print(
        "是否归一化："
        f"{normalize_embeddings}"
    )

    # ---------------------------------------------------------
    # 3. 创建 Embedding 模型
    # ---------------------------------------------------------

    print("\n正在加载 Embedding 模型……")

    embeddings = create_huggingface_embeddings(
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        device="cpu",
    )

    # ---------------------------------------------------------
    # 4. 将所有 Chunk 转换成文本列表
    # ---------------------------------------------------------

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # ---------------------------------------------------------
    # 5. 批量生成文档向量
    # ---------------------------------------------------------

    print("正在生成 Chunk 向量……")

    document_vectors = (
        embeddings.embed_documents(
            chunk_texts
        )
    )

    if len(document_vectors) != len(chunks):
        raise RuntimeError(
            "向量数量与 Chunk 数量不一致："
            f"{len(document_vectors)} != "
            f"{len(chunks)}"
        )

    # ---------------------------------------------------------
    # 6. 查看第一个向量
    # ---------------------------------------------------------

    first_vector = document_vectors[0]

    print("\n" + "=" * 70)
    print("第一个 Chunk 的向量信息")
    print("=" * 70)

    print(
        "chunk_id："
        f"{chunks[0].metadata['chunk_id']}"
    )

    print(
        f"向量维度：{len(first_vector)}"
    )

    print(
        "向量前 10 个数值："
    )

    print(first_vector[:10])

    print(
        "向量 L2 范数："
        f"{vector_norm(first_vector):.6f}"
    )

    # ---------------------------------------------------------
    # 7. 将用户问题转换成查询向量
    # ---------------------------------------------------------

    query = (
        "Retriever 和 Vector Store "
        "有什么区别？"
    )

    print("\n" + "=" * 70)
    print("查询向量")
    print("=" * 70)

    print(f"问题：{query}")

    query_vector = embeddings.embed_query(
        query
    )

    print(
        f"查询向量维度：{len(query_vector)}"
    )

    print(
        "查询向量前 10 个数值："
    )

    print(query_vector[:10])

    print(
        "查询向量 L2 范数："
        f"{vector_norm(query_vector):.6f}"
    )

    # ---------------------------------------------------------
    # 8. 手动计算问题与每个 Chunk 的相似度
    # ---------------------------------------------------------

    scored_chunks: list[
        tuple[float, int]
    ] = []

    for index, document_vector in enumerate(
        document_vectors
    ):
        score = cosine_similarity(
            query_vector,
            document_vector,
        )

        scored_chunks.append(
            (score, index)
        )

    # 按相似度从高到低排序。
    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # 只保留 Top-K。
    top_results = scored_chunks[:top_k]

    # ---------------------------------------------------------
    # 9. 输出最相关的 Chunk
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"手动语义检索 Top-{top_k}")
    print("=" * 70)

    for rank, (
        score,
        chunk_index,
    ) in enumerate(
        top_results,
        start=1,
    ):
        chunk = chunks[chunk_index]

        print("\n" + "-" * 70)
        print(f"排名：{rank}")
        print(f"相似度：{score:.6f}")

        print(
            "chunk_id："
            f"{chunk.metadata['chunk_id']}"
        )

        print(
            "doc_id："
            f"{chunk.metadata['doc_id']}"
        )

        print(
            "title："
            f"{chunk.metadata['title']}"
        )

        print(
            "page_number："
            f"{chunk.metadata['page_number']}"
        )

        print("文本：")
        print(
            normalize_preview(
                chunk.page_content
            )
        )


if __name__ == "__main__":
    main()