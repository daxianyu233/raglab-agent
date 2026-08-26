"""建立并持久化 RAGLab 基线 Chroma 向量库。

执行流程：

baseline.yaml
→ 加载 PDF
→ 页面级 Document
→ Chunk
→ Embedding
→ Chroma
→ storage/chroma/baseline
"""

from __future__ import annotations

from pathlib import Path

import yaml

from raglab.embeddings.factory import (
    create_huggingface_embeddings,
)
from raglab.ingestion.loaders import (
    load_pdf_corpus,
)
from raglab.ingestion.splitters import (
    create_recursive_splitter,
    split_page_documents,
)
from raglab.settings import (
    BASELINE_CONFIG_PATH,
    PROJECT_ROOT,
)
from raglab.vectorstores.chroma_store import (
    rebuild_chroma_store,
)


def load_baseline_config() -> dict:
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
    """将配置文件中的路径转换成绝对路径。

    配置中的：

        storage/chroma/baseline

    会被转换成：

        项目根目录/storage/chroma/baseline
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main() -> None:
    # 1. 读取配置。
    config = load_baseline_config()

    splitter_config = config["splitter"]
    embedding_config = config["embedding"]
    vector_store_config = config[
        "vector_store"
    ]

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

    print("=" * 70)
    print("RAGLab Chroma 基线索引构建")
    print("=" * 70)

    print(
        f"Embedding 模型：{model_name}"
    )
    print(
        f"Collection：{collection_name}"
    )
    print(
        f"持久化目录：{persist_directory}"
    )

    # 2. 加载 PDF。
    print("\n正在加载 PDF……")

    page_documents = load_pdf_corpus()

    # 3. 创建文本切分器。
    text_splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 4. 生成 Chunk。
    print("正在切分文本……")

    chunks = split_page_documents(
        documents=page_documents,
        text_splitter=text_splitter,
    )

    if not chunks:
        raise RuntimeError(
            "没有生成任何 Chunk。"
        )

    print(
        f"页面数量：{len(page_documents)}"
    )
    print(
        f"Chunk 数量：{len(chunks)}"
    )

    # 5. 创建 Embedding 模型。
    print("\n正在加载 Embedding 模型……")

    embeddings = create_huggingface_embeddings(
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        device="cpu",
    )

    # 6. 建立持久化 Chroma。
    print(
        "正在生成向量并写入 Chroma……"
    )

    rebuild_chroma_store(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print("\n" + "=" * 70)
    print("Chroma 索引构建完成")
    print("=" * 70)

    print(
        f"写入 Chunk 数量：{len(chunks)}"
    )
    print(
        f"保存目录：{persist_directory}"
    )
    print(
        "后续查询不需要重新读取 PDF。"
    )


if __name__ == "__main__":
    main()