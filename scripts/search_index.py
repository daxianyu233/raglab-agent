"""打开已有 Chroma，并执行交互式 Dense Retrieval。

这个脚本不会重新：

- 读取 PDF；
- 划分 Chunk；
- 生成全部文档向量；
- 建立向量数据库。

它只负责：

1. 加载同一个 Embedding 模型；
2. 打开已经持久化的 Chroma；
3. 将用户问题转换成查询向量；
4. 返回距离最近的 Top-K Chunk。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from raglab.embeddings.factory import (
    create_huggingface_embeddings,
)
from raglab.retrieval.dense import (
    dense_similarity_search_with_score,
)
from raglab.settings import (
    BASELINE_CONFIG_PATH,
    PROJECT_ROOT,
)
from raglab.vectorstores.chroma_store import (
    open_chroma_store,
)


def load_baseline_config() -> dict[str, Any]:
    """读取 baseline.yaml。"""
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


def resolve_project_path(
    path_value: str,
) -> Path:
    """将项目相对路径转换成绝对路径。"""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def normalize_preview(
    text: str,
    max_length: int = 320,
) -> str:
    """整理文本格式并截断，仅用于终端显示。"""
    normalized = " ".join(
        text.strip().split()
    )

    if len(normalized) <= max_length:
        return normalized

    return normalized[:max_length] + "..."


def print_results(
    results: list[tuple[Any, float]],
) -> None:
    """打印检索结果。"""
    if not results:
        print("\n没有检索到结果。")
        return

    print("\n" + "=" * 70)
    print(f"检索结果 Top-{len(results)}")
    print("=" * 70)

    for rank, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 70)
        print(f"排名：{rank}")

        # 当前默认使用 L2 距离：
        # 数值越小，表示向量越接近。
        print(f"L2 distance：{distance:.6f}")

        print(
            "chunk_id："
            f"{document.metadata.get('chunk_id')}"
        )

        print(
            "doc_id："
            f"{document.metadata.get('doc_id')}"
        )

        print(
            "title："
            f"{document.metadata.get('title')}"
        )

        print(
            "corpus："
            f"{document.metadata.get('corpus')}"
        )

        print(
            "page_number："
            f"{document.metadata.get('page_number')}"
        )

        print("文本：")
        print(
            normalize_preview(
                document.page_content
            )
        )


def main() -> None:
    """打开持久化向量库并循环接收查询。"""
    config = load_baseline_config()

    embedding_config = config["embedding"]
    vector_store_config = config[
        "vector_store"
    ]
    retrieval_config = config["retrieval"]

    model_name = str(
        embedding_config["model"]
    )

    normalize_embeddings = bool(
        embedding_config[
            "normalize_embeddings"
        ]
    )

    collection_name = str(
        vector_store_config[
            "collection_name"
        ]
    )

    persist_directory = resolve_project_path(
        str(
            vector_store_config[
                "persist_directory"
            ]
        )
    )

    top_k = int(
        retrieval_config["top_k"]
    )

    metadata_filter = retrieval_config.get(
        "metadata_filter"
    )

    # open_chroma_store() 在目录不存在时会创建目录。
    #
    # 查询脚本中我们不希望悄悄创建一个空库，
    # 所以要提前检查向量库是否已经构建。
    if (
        not persist_directory.exists()
        or not any(persist_directory.iterdir())
    ):
        raise FileNotFoundError(
            "没有找到已经建立的 Chroma 索引。\n"
            "请先运行：\n"
            "python -m scripts.build_index\n"
            f"检查目录：{persist_directory}"
        )

    print("=" * 70)
    print("RAGLab 持久化 Dense Retrieval")
    print("=" * 70)

    print(f"Embedding 模型：{model_name}")
    print(f"Collection：{collection_name}")
    print(f"Top-K：{top_k}")
    print(f"向量库目录：{persist_directory}")

    # 查询仍需要加载 Embedding 模型，
    # 因为用户问题需要被转换成查询向量。
    #
    # 但不会重新生成全部文档向量。
    print("\n正在加载 Embedding 模型……")

    embeddings = create_huggingface_embeddings(
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        device="cpu",
    )

    # 使用相同目录和 Collection 名称，
    # 打开之前建立并持久化的 Chroma。
    vector_store = open_chroma_store(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print("\n向量库已打开。")
    print("输入 q 或直接回车可退出。")

    # 只加载一次模型和向量库，
    # 然后连续查询多个问题。
    while True:
        query = input(
            "\n请输入问题："
        ).strip()

        if not query or query.lower() == "q":
            print("查询结束。")
            break

        results = (
            dense_similarity_search_with_score(
                vector_store=vector_store,
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        )

        print_results(results)


if __name__ == "__main__":
    main()